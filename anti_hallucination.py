# -*- coding: utf-8 -*-
"""
anti_hallucination.py —— 防 AI 幻觉引擎
=========================================
四道防线，把生成结果的幻觉概率压到最低：

  1. 置信度幻觉评分（confidence_score）
     —— 用模型逐 token 对数概率求几何均值，低置信 = 高幻觉风险，标记并可选重生成。
  2. 语法/结构拦截（parse_gate）
     —— 用解析引擎拒绝无法通过 AST 校验的输出（幻觉最常表现为语法错误/未定义名）。
  3. 检索增强（retrieve_augment）
     —— 从本地真实代码库检索相似片段，给生成提供真实依据，减少编造。
  4. 自一致性去幻觉（self_consistency）
     —— 多次采样取「通过校验 + 多数一致」的结果，天然抑制随机幻觉。

用法（配合 decode.py / generate.py 使用）：
    from anti_hallucination import *
"""
import os
import re
import ast
from code_parse import parse_check, run_check


def confidence_score(scores):
    """scores: 逐 token 对数概率列表。返回 0~1 置信度（几何均值 exp(mean logp)）。"""
    if not scores:
        return 0.0
    import math
    mean_logp = sum(scores) / len(scores)
    return math.exp(mean_logp)


def parse_gate(code, require_exec=False, allow_undefined=False):
    """解析闸门：语法校验 + 可选可执行校验。返回 (ok, report)。"""
    res = parse_check(code)
    report = {
        "syntax_ok": res.ok,
        "undefined": res.undefined,
        "score": res.score,
        "error": res.error,
        "defs": res.n_defs,
    }
    if not res.ok:
        return False, report
    if not allow_undefined and res.undefined:
        # 未定义名视为幻觉高风险（引用不存在的 API/函数）
        report["hallucination"] = f"未定义名: {res.undefined}"
        return False, report
    if require_exec:
        ok, err, out = run_check(code)
        report["exec_ok"] = ok
        report["exec_error"] = err
        if not ok:
            return False, report
    return True, report


class Retriever:
    """轻量检索增强：从本地代码库按关键词/函数名检索真实片段。"""
    def __init__(self, code_dir="data/stream", top_k=3):
        self.top_k = top_k
        self.snippets = []
        if os.path.isdir(code_dir):
            for dirpath, _, files in os.walk(code_dir):
                for fn in files:
                    if fn.endswith(".py"):
                        try:
                            with open(os.path.join(dirpath, fn), "r",
                                      encoding="utf-8", errors="ignore") as f:
                                self.snippets.append(f.read())
                        except Exception:
                            pass
        print(f"[retriever] 加载 {len(self.snippets)} 个真实代码片段")

    def retrieve(self, query, k=None):
        """按 query 中的标识符/关键词返回最相关真实片段。"""
        k = k or self.top_k
        q_words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query))
        scored = []
        for s in self.snippets:
            sw = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s))
            inter = len(q_words & sw)
            if inter > 0:
                scored.append((inter, s))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:k]]


def self_consistency(model, tok, prompt_ids, n_samples=5, max_tokens=80,
                     gate=True, device=None, temperature=0.8):
    """多次采样 → 各自过校验 → 选择通过校验且多数一致的最佳结果。"""
    import torch
    votes = {}
    outputs = []
    for _ in range(n_samples):
        idx = torch.tensor([prompt_ids], dtype=torch.long,
                           device=device or next(model.parameters()).device)
        out, scores = model.generate(idx, max_tokens, temperature=temperature,
                                     top_k=40, top_p=0.9, return_scores=True)
        text = tok.decode(out[0][len(prompt_ids):].tolist())
        ok = True
        if gate:
            ok, _ = parse_gate(text, allow_undefined=False)
        outputs.append((ok, text, confidence_score(scores)))
        if ok:
            key = text.strip()[:80]
            votes[key] = votes.get(key, 0) + 1
    # 优先选通过校验且票数最高的；否则选置信度最高的
    ok_cands = [(votes.get(t.strip()[:80], 0), t, c) for ok, t, c in outputs if ok]
    if ok_cands:
        ok_cands.sort(key=lambda x: (-x[0], -x[2]))
        return ok_cands[0][1]
    outputs.sort(key=lambda x: -x[2])
    return outputs[0][1]
