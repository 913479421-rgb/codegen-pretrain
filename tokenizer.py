# -*- coding: utf-8 -*-
"""
tokenizer.py —— 自研字节级 BPE 分词器（一代升级版，零第三方依赖）
====================================================================
升级特性：
  - GPT-2 式「正则预分割 + 字节级 BPE」：先把文本按词/空白/标点粗分，
    再在每段内做字节级 BPE —— 对代码注释、缩进、token 边界更友好。
  - 多特殊 token 支持：<|endoftext|>、FIM 占位符等固定前缀 id。
  - 兼容训练时增量统计与按训练顺序编码（与训练一致）。

用法：
    tok = BPETokenizer(special_tokens=["<|endoftext|>"])
    tok.train(texts, vocab_size=8000); tok.save("data/vocab.json")
    ids = tok.encode("def add(a, b):\n    return a + b")
    text = tok.decode(ids)
"""
import re
import json
from collections import Counter

# 内置 re 的预分割正则（覆盖代码常见字符，含 ASCII 与常见非 ASCII）。
# 优先级：特殊 token 会被单独切出；其余按「单词/数字/空白/其他标点」粗分。
_GP2_SPLIT_PAT = re.compile(
    r"<\|[^|]*\|>|"       # 特殊 token（如 <|endoftext|>）
    r"\s+|"               # 空白
    r"[A-Za-z_][A-Za-z0-9_]*|"  # 标识符/关键字
    r"\d+|"               # 数字
    r".",                 # 其他单个字符
    re.DOTALL
)


class BPETokenizer:
    def __init__(self, special_tokens=None):
        special_tokens = special_tokens or ["<|endoftext|>"]
        self.special = {t: i for i, t in enumerate(special_tokens)}
        # 初始 id：特殊 token 之后是 256 个字节
        base = len(self.special)
        self.byte_to_id = {i: base + i for i in range(256)}
        self.id_to_byte = {v: k for k, v in self.byte_to_id.items()}
        self.merges = {}
        self.vocab_size = base + 256
        # 预分割正则：把特殊 token 或非空白片段切开
        self._split = _GP2_SPLIT_PAT

    # ---------------- 预分割 ----------------
    def _pre_split(self, text):
        """把文本切成片段列表（特殊 token 单列，其余按预分割正则）。"""
        parts = []
        i = 0
        n = len(text)
        while i < n:
            # 先找特殊 token
            found = False
            for st in self.special:
                if text.startswith(st, i):
                    parts.append(st)
                    i += len(st)
                    found = True
                    break
            if found:
                continue
            m = self._split.match(text, i)
            if m:
                parts.append(m.group(0))
                i = m.end()
            else:
                parts.append(text[i])
                i += 1
        return parts

    # ---------------- 训练 ----------------
    def train(self, text_chunks, vocab_size=8000, verbose=True):
        assert vocab_size > self.vocab_size, "vocab_size 必须大于初始大小"
        # 逐 chunk 预分割并转字节 id，同时保留片段边界（跨片段不合并）
        segments = []
        for c in text_chunks:
            for part in self._pre_split(c):
                seg = self._bytes_to_ids(part.encode("utf-8"))
                if seg:
                    segments.append(seg)
        print(f"[tokenizer] 预分割出 {len(segments)} 个片段")

        pair_counts = Counter()
        for s in segments:
            for a, b in zip(s, s[1:]):
                pair_counts[(a, b)] += 1

        num_merges = vocab_size - self.vocab_size
        for step in range(num_merges):
            if not pair_counts:
                print("[tokenizer] 无可合并 pair，提前结束")
                break
            (a, b), _ = pair_counts.most_common(1)[0]
            new_id = self.vocab_size
            self.vocab_size += 1
            self.merges[(a, b)] = new_id

            for s in segments:
                i, L = 0, len(s)
                while i < L - 1:
                    if s[i] == a and s[i + 1] == b:
                        if i > 0:
                            pair_counts[(s[i - 1], a)] -= 1
                        if i + 2 < L:
                            pair_counts[(b, s[i + 2])] -= 1
                        s[i] = new_id
                        del s[i + 1]
                        L -= 1
                        if i > 0:
                            pair_counts[(s[i - 1], new_id)] += 1
                        if i + 1 < L:
                            pair_counts[(new_id, s[i + 1])] += 1
                        i += 1
                    else:
                        i += 1
                if step % 1000 == 0:
                    for k in list(pair_counts):
                        if pair_counts[k] <= 0:
                            del pair_counts[k]
            if (step + 1) % 1000 == 0 and verbose:
                print(f"  merge {step+1}/{num_merges}: 词表={self.vocab_size}")

        print(f"[tokenizer] BPE 训练完成，词表大小 = {self.vocab_size}")
        return self.vocab_size

    # ---------------- 编码 ----------------
    def encode(self, text):
        ids = []
        for part in self._pre_split(text):
            if part in self.special:
                ids.append(self.special[part])
                continue
            seq = self._bytes_to_ids(part.encode("utf-8"))
            for (a, b), new_id in self.merges.items():
                out = []
                i, L = 0, len(seq)
                while i < L:
                    if i + 1 < L and seq[i] == a and seq[i + 1] == b:
                        out.append(new_id); i += 2
                    else:
                        out.append(seq[i]); i += 1
                seq = out
            ids.extend(seq)
        return ids

    # ---------------- 解码 ----------------
    def decode(self, ids):
        if not isinstance(ids, (list, tuple)):
            ids = ids.tolist()
        raw = bytearray()
        for i in ids:
            if i in self.id_to_byte:
                raw.append(self.id_to_byte[i])
            else:
                # 可能是特殊 token 或合并 id
                expanded = self._expand(i)
                if expanded is not None:
                    raw.extend(expanded)
                else:
                    raw.extend(str(i).encode())  # 兜底
        return raw.decode("utf-8", errors="replace")

    def _expand(self, idx):
        for (a, b), new_id in self.merges.items():
            if new_id == idx:
                left = self._expand(a)
                right = self._expand(b)
                if left is None or right is None:
                    return None
                return left + right
        if idx in self.id_to_byte:
            return bytearray([self.id_to_byte[idx]])
        # 反向找特殊 token
        for t, tid in self.special.items():
            if tid == idx:
                return t.encode("utf-8")
        return None

    def _bytes_to_ids(self, b):
        return [self.byte_to_id[x] for x in b]

    # ---------------- 保存 / 加载 ----------------
    def save(self, path):
        data = {
            "special": self.special,
            "byte_to_id": {str(k): v for k, v in self.byte_to_id.items()},
            "merges": {f"{a},{b}": v for (a, b), v in self.merges.items()},
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[tokenizer] 词表已保存: {path} (size={self.vocab_size})")

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(special_tokens=list(data["special"].keys()))
        tok.special = data["special"]
        tok.byte_to_id = {int(k): v for k, v in data["byte_to_id"].items()}
        tok.id_to_byte = {v: k for k, v in tok.byte_to_id.items()}
        tok.merges = {tuple(int(x) for x in k.split(",")): v
                      for k, v in data["merges"].items()}
        tok.vocab_size = data["vocab_size"]
        return tok
