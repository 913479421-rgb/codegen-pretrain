#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_data.py —— 世界级代码数据集采集器（持续训练的数据上游）
================================================================
从全球顶级开源代码语料流式抓取，落入 data/stream/ 目录，
供 continuous_train.py 持续增量训练消费。

可选数据源（均为业界标杆级代码语料）：
  stack        The Stack v3（BigCode/HuggingFace，5T token，全球最大开源代码语料）
  github-code  codeparrot/github-code（StarCoder 训练语料）
  codecontests DeepMind CodeContests（竞赛题，代码智能评测标杆）
  humaneval    OpenAI HumanEval（代码能力评测基准）

用法：
    python collect_data.py --source stack --max-mb 200 --out-dir data/stream
    python collect_data.py --source humaneval
"""
import os, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "data", "stream")


def collect_stack(max_mb):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceCode/stack-v3-train", streaming=True, split="train")
    texts, total, i = [], 0, 0
    for ex in ds:
        content = ex.get("content") or ""
        if content.strip():
            texts.append(content)
            total += len(content.encode("utf-8"))
            if total >= max_mb * 1024 * 1024:
                break
        i += 1
        if i > 2_000_000:
            break
    return texts


def collect_githubcode(max_mb):
    from datasets import load_dataset
    ds = load_dataset("codeparrot/github-code", streaming=True, split="train")
    texts, total, i = [], 0, 0
    for ex in ds:
        content = ex.get("content") or ""
        if content.strip():
            texts.append(content)
            total += len(content.encode("utf-8"))
            if total >= max_mb * 1024 * 1024:
                break
        i += 1
        if i > 500_000:
            break
    return texts


def collect_codecontests():
    # 只取 problem 描述与解法文本，规模小，适合做补充
    from datasets import load_dataset
    ds = load_dataset("deepmind/code_contests", split="train")
    texts = []
    for ex in ds:
        texts.append(ex.get("description") or "")
        if ex.get("solutions"):
            sols = ex["solutions"]["solution"]
            texts.extend(s for s in sols if isinstance(s, str))
    return texts


def collect_humaneval():
    # OpenAI HumanEval 题目（含解法），做代码评测/补充
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    texts = []
    for ex in ds:
        texts.append(ex.get("prompt") or "")
        texts.append(ex.get("canonical_solution") or "")
    return texts


SOURCES = {
    "stack": collect_stack,
    "github-code": collect_githubcode,
    "codecontests": collect_codecontests,
    "humaneval": collect_humaneval,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(SOURCES.keys()))
    ap.add_argument("--max-mb", type=int, default=100)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--no-overwrite", action="store_true", help="不覆盖已存在的同名文件")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f">>> 采集数据源: {args.source} -> {args.out_dir}")
    texts = SOURCES[args.source](args.max_mb)
    texts = [t for t in texts if t and t.strip()]
    print(f"    共 {len(texts)} 条样本")

    written = 0
    for k, t in enumerate(texts):
        path = os.path.join(args.out_dir, f"{args.source}_{k:06d}.py")
        if args.no_overwrite and os.path.exists(path):
            continue
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(t)
        written += 1
    print(f">>> 已写入 {written} 个文件。之后运行 continuous_train.py 增量训练即可消费。")


if __name__ == "__main__":
    main()
