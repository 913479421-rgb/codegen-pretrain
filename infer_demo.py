#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_demo.py —— 推理加速命令行（投机解码 + 量化 + 编译）
==========================================================
统一演示模型生成速度引擎：

    # 投机解码（主模型 + 草稿模型）
    python infer_demo.py --checkpoint checkpoints/final.pt \
        --draft checkpoints/final.pt --speculative --gamma 4 \
        --prompt "def fibonacci(n):" --max-tokens 60

    # 普通生成测速（对比）
    python infer_demo.py --checkpoint checkpoints/final.pt --bench

    # 量化模型推理
    python infer_demo.py --quant checkpoints/quant_8bit.pt \
        --prompt "def add(" --max-tokens 20

说明：投机解码的草稿模型（--draft）用与主模型同 tokenizer 的小模型效果最好，
      草稿与主模型分布越接近，加速越明显（2~3×）。
"""
import os, sys, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))


def load_ckpt(path, device):
    import torch
    from config import TrainConfig
    from model import GPT
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in ck and hasattr(ck["model"], "generate"):  # 量化模型对象
        m = ck["model"].to(device); m.eval(); return m
    cfg = TrainConfig()
    for k, v in (ck.get("config") or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    m = GPT(cfg).to(device)
    m.load_state_dict(ck.get("ema") or ck["model"])
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/final.pt")
    ap.add_argument("--draft", default="", help="草稿模型路径（投机解码用）")
    ap.add_argument("--quant", default="", help="量化模型路径")
    ap.add_argument("--speculative", action="store_true")
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--bench", action="store_true", help="测速对比")
    ap.add_argument("--prompt", default="def fibonacci(n):\n")
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    from tokenizer import BPETokenizer
    tok = BPETokenizer.load(os.path.join("data", "vocab.json"))
    prompt_ids = tok.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [0]

    if args.quant:
        model = load_ckpt(args.quant, device)
        from infer_engine import quantized_generate
        t0 = time.time()
        text = quantized_generate(model, tok, prompt_ids, max_new_tokens=args.max_tokens)
        dt = time.time() - t0
        n = len(tok.encode(text))
        print(f"[量化推理] {n/dt:.1f} tok/s")
        print(text)
        return

    model = load_ckpt(args.checkpoint, device)

    if args.speculative:
        draft_path = args.draft or args.checkpoint
        draft = load_ckpt(draft_path, device)
        # 对齐草稿 vocab
        if getattr(draft, "cfg", None) and draft.cfg.vocab_size != model.cfg.vocab_size:
            from config import get_config
            from model import GPT
            c2 = get_config("tiny"); c2.vocab_size = model.cfg.vocab_size
            draft = GPT(c2).to(device)
        from infer_engine import speculative_generate
        t0 = time.time()
        text, acc, steps = speculative_generate(model, draft, tok, prompt_ids,
                                                gamma=args.gamma,
                                                max_new_tokens=args.max_tokens)
        dt = time.time() - t0
        n = len(tok.encode(text))
        print(f"[投机解码] {n/dt:.1f} tok/s | 接受={acc}/{n} | 主前向步数={steps}")
        print("生成：")
        print(text)
        return

    if args.bench:
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        t0 = time.time(); model.generate(idx, args.max_tokens, temperature=0.8)
        dt = time.time() - t0
        print(f"[普通生成] {args.max_tokens/dt:.1f} tok/s")
        return

    out = model.generate(torch.tensor([prompt_ids], dtype=torch.long, device=device),
                         args.max_tokens, temperature=0.8)
    print(tok.decode(out[0][len(prompt_ids):].tolist()))


if __name__ == "__main__":
    main()
