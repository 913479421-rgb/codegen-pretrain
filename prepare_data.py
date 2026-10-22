#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data.py —— 准备预训练语料：下载代码 → 清洗 → 训练BPE → 生成token序列
============================================================================
两种数据来源（二选一）：
  1) 从公开语料下载 Python 代码（The Stack / github-code 流式，可能需登录）
  2) 指向本地一个含 .py 文件的目录（最稳妥，任何机器都能跑通）

用法：
    # 方式2：本地代码目录（推荐先这样跑通）
    python prepare_data.py --local-dir ./my_py_code --tiny

    # 方式1：从 HuggingFace 流式下载（需网络，可能需 HF token）
    python prepare_data.py --hf --max-mb 100

输出：
    data/code_train.bin   预训练 token 序列（uint16）
    data/code_eval.bin    留出的验证序列
    data/vocab.json       BPE 词表
    data/meta.json        元信息
"""
import os, sys, json, random, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

TRAIN_SPLIT = 0.95


def collect_local(root):
    """递归收集目录下所有 .py 文件文本。"""
    texts = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".py"):
                p = os.path.join(dirpath, fn)
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        texts.append(f.read())
                except Exception:
                    continue
    print(f"[data] 本地目录收集到 {len(texts)} 个 .py 文件")
    return texts


def collect_hf(max_mb, source="github-code"):
    """从 HuggingFace 流式下载代码，累积到约 max_mb。
    source 可选：github-code（codeparrot）或 stack（HuggingFaceCode/stack-v3-train，The Stack v3）。"""
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[错误] 未安装 datasets：{e}")
        sys.exit(1)
    if source == "stack":
        repo = "HuggingFaceCode/stack-v3-train"
        texts, total, i = [], 0, 0
        ds = load_dataset(repo, streaming=True, split="train")
        for ex in ds:
            content = ex.get("content") or ""
            lang = (ex.get("lang") or ex.get("language") or "")
            if (lang in ("Python", "python") or str(ex.get("file_name", "")).endswith(".py")) \
                    and content.strip():
                texts.append(content)
                total += len(content.encode("utf-8"))
                if total >= max_mb * 1024 * 1024:
                    break
            i += 1
            if i > 2_000_000:
                break
    else:
        ds = load_dataset("codeparrot/github-code", streaming=True, split="train")
        texts, total, i = [], 0, 0
        for ex in ds:
            fname = (ex.get("file_name") or "")
            content = ex.get("content") or ""
            if fname.endswith(".py") and content.strip():
                texts.append(content)
                total += len(content.encode("utf-8"))
                if total >= max_mb * 1024 * 1024:
                    break
            i += 1
            if i > 500_000:
                break
    print(f"[data] 从 HF({source}) 下载到 {len(texts)} 个代码文件，约 {total/1e6:.1f} MB")
    return texts


def dedup(texts):
    """简单内容去重（按文件哈希），减少重复语料。"""
    import hashlib
    seen, uniq = set(), []
    for t in texts:
        h = hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()
        if h not in seen:
            seen.add(h)
            uniq.append(t)
    print(f"[data] 去重: {len(texts)} -> {len(uniq)}")
    return uniq


def clean(text):
    """基础清洗：去掉过长行、空代码块、非打印字符。"""
    lines = []
    for ln in text.split("\n"):
        if len(ln) > 2000:      # 超长行（通常是被压缩的）丢弃
            continue
        lines.append(ln.rstrip())
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", default="", help="本地代码目录")
    ap.add_argument("--hf", action="store_true", help="从 HuggingFace 下载")
    ap.add_argument("--source", default="github-code",
                    choices=["github-code", "stack"], help="HF 数据源：github-code 或 The Stack v3")
    ap.add_argument("--max-mb", type=int, default=100, help="HF 下载上限 MB")
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--no-dedup", action="store_true", help="关闭去重")
    ap.add_argument("--tiny", action="store_true",
                    help="小规模冒烟：减小词表、减少数据，快速跑通")
    args = ap.parse_args()

    # ---------- 1. 收集文本 ----------
    if args.local_dir:
        texts = collect_local(args.local_dir)
    elif args.hf:
        texts = collect_hf(args.max_mb, args.source)
    else:
        print("必须指定 --local-dir 或 --hf 之一")
        sys.exit(1)

    if args.tiny:
        texts = texts[:2000]
        args.vocab_size = min(args.vocab_size, 1024)
        print("[data] tiny 模式：只用前 2000 个文件，词表=1024")

    texts = [clean(t) for t in texts if t.strip()]
    print(f"[data] 清洗后 {len(texts)} 个文件")
    if not args.no_dedup:
        texts = dedup(texts)

    # ---------- 2. 训练 BPE ----------
    from tokenizer import BPETokenizer
    tok = BPETokenizer()
    tok.train(texts, vocab_size=args.vocab_size)
    tok.save(os.path.join(DATA, "vocab.json"))

    # ---------- 3. 编码成 token 序列 ----------
    random.seed(42)
    random.shuffle(texts)
    split = int(len(texts) * TRAIN_SPLIT)
    def encode_all(chunk):
        ids = []
        for t in chunk:
            ids.extend(tok.encode(t))
            ids.append(0)   # <|endoftext|> 分隔
        return ids

    train_ids = encode_all(texts[:split])
    eval_ids = encode_all(texts[split:])
    print(f"[data] train tokens={len(train_ids)}, eval tokens={len(eval_ids)}")

    # 存为 uint16 numpy（词表 < 65536 时可用）
    train_ids = np.array(train_ids, dtype=np.uint16)
    eval_ids = np.array(eval_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(DATA, "code_train.bin"))
    eval_ids.tofile(os.path.join(DATA, "code_eval.bin"))

    meta = {"vocab_size": tok.vocab_size, "train_tokens": int(len(train_ids)),
            "eval_tokens": int(len(eval_ids))}
    with open(os.path.join(DATA, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[data] 完成 ✅")
    print(f"  训练集: data/code_train.bin ({len(train_ids)} tokens)")
    print(f"  词表  : data/vocab.json (size={tok.vocab_size})")
    print("下一步: python train.py --config small")


if __name__ == "__main__":
    main()
