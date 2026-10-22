#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py —— 模型评测（一代新增）
================================
评测维度：
  1. 验证集困惑度 Perplexity（越低越好）
  2. 若干代码生成样例（基座 vs EMA 对比可选）
  3. 简单 pass@k 估算：给定函数签名，统计能生成正确骨架的次数

用法：
    python eval.py --checkpoint checkpoints/final.pt --ppl-only
    python eval.py --checkpoint checkpoints/final.pt --samples 5 --pass-k 3
"""
import os, sys, math, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TrainConfig
from model import GPT
from tokenizer import BPETokenizer
from train import load_tokens


def build_model(checkpoint, device, use_ema=False):
    ck = torch.load(checkpoint, map_location="cpu")
    cfg = TrainConfig()
    for k, v in ck["config"].items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    model = GPT(cfg).to(device)
    src = ck.get("ema") if (use_ema and ck.get("ema")) else ck["model"]
    model.load_state_dict(src)
    model.eval()
    return model, cfg


@torch.no_grad()
def perplexity(model, device, iters=100, block_size=256, batch_size=16, use_amp=False):
    data = load_tokens("data/code_eval.bin")
    if len(data) < batch_size * block_size:
        data = np.tile(data, (batch_size * block_size) // len(data) + 1)
    total, cnt = 0.0, 0
    pos = 0
    n = batch_size * block_size
    for _ in range(iters):
        if pos + n + 1 > len(data):
            pos = 0
        chunk = data[pos:pos + n + 1]
        pos += n
        x = torch.from_numpy(chunk[:-1].reshape(batch_size, block_size)).to(device)
        y = torch.from_numpy(chunk[1:].reshape(batch_size, block_size)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(use_amp and device.type == "cuda")):
            _, loss = model(x, y)
        total += loss.item() * (batch_size * block_size)
        cnt += batch_size * block_size
    return math.exp(total / cnt)


# 简单的 pass@k 骨架正确性检查（教学级近似，非标准 HumanEval）
_SKELETONS = [
    ("def fibonacci(n):\n", ["return"]),
    ("def is_even(x):\n", ["return"]),
    ("def add(a, b):\n", ["return"]),
    ("class Calculator:\n", ["def __init__"]),
]


def pass_k(model, tok, device, k=3, max_tokens=60):
    correct = 0
    total = 0
    for prompt, must in _SKELETONS:
        ids = tok.encode(prompt)
        if not ids:
            ids = [0]
        hit = False
        for _ in range(k):
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(idx, max_tokens, temperature=0.8, top_k=40, top_p=0.9)
            gen = tok.decode(out[0][len(ids):].tolist())
            if all(m in gen for m in must):
                hit = True
                break
        total += 1
        correct += int(hit)
    return correct / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/final.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ppl-only", action="store_true")
    ap.add_argument("--samples", type=int, default=3, help="生成样例数")
    ap.add_argument("--pass-k", type=int, default=3, help="pass@k 的 k")
    ap.add_argument("--use-ema", action="store_true")
    args = ap.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    model, cfg = build_model(args.checkpoint, device, args.use_ema)
    tok = BPETokenizer.load(os.path.join("data", "vocab.json"))
    tag = "EMA" if args.use_ema else "base"

    print(f"[eval] 模型参数量 {model.param_count()/1e6:.1f}M ({tag})")
    ppl = perplexity(model, device, iters=cfg.eval_iters, block_size=cfg.block_size,
                     batch_size=cfg.batch_size)
    print(f"[eval] 验证集困惑度 Perplexity = {ppl:.2f}")

    if not args.ppl_only:
        pk = pass_k(model, tok, device, k=args.pass_k)
        print(f"[eval] pass@{args.pass_k} (骨架正确率) = {pk:.2f}")
        print("\n[样例生成]")
        for prompt, _ in _SKELETONS[:args.samples]:
            ids = tok.encode(prompt)
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(idx, 80, temperature=0.7, top_k=40, top_p=0.9)
            print("=" * 50)
            print(prompt + tok.decode(out[0][len(ids):].tolist()))


if __name__ == "__main__":
    main()
