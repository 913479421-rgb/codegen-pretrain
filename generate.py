#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate.py —— 用训练好的模型生成代码（一代升级版）
======================================================
用法：
    python generate.py --checkpoint checkpoints/final.pt \
        --prompt "def fibonacci(n):" --max-tokens 100 --use-ema
    # 交互模式
    python generate.py --checkpoint checkpoints/final.pt --interactive
"""
import os, sys, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TrainConfig
from model import GPT
from tokenizer import BPETokenizer


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/final.pt")
    ap.add_argument("--prompt", default="def fibonacci(n):\n")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--min-p", type=float, default=0.0, help="min-p 采样（低置信token过滤）")
    ap.add_argument("--typical-p", type=float, default=0.0, help="典型采样")
    ap.add_argument("--repeat-penalty", type=float, default=1.0)
    ap.add_argument("--use-ema", action="store_true", help="用 EMA 权重生成")
    ap.add_argument("--beam", type=int, default=0, help="束搜索宽度（>1 启用束搜索）")
    ap.add_argument("--self-consistency", type=int, default=0,
                    help="自一致性采样数（>1 启用，抑制幻觉）")
    ap.add_argument("--gate", action="store_true",
                    help="防幻觉解析闸门：拒绝语法/未定义名错误输出")
    ap.add_argument("--retrieve", default="",
                    help="检索增强：指向本地真实代码库目录，生成前检索依据")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    model, cfg = build_model(args.checkpoint, device, args.use_ema)
    tok = BPETokenizer.load(os.path.join("data", "vocab.json"))

    retriever = None
    if args.retrieve:
        from anti_hallucination import Retriever
        retriever = Retriever(args.retrieve)

    def run(prompt):
        # 检索增强：附加最相关真实片段作为上下文依据
        if retriever is not None:
            hits = retriever.retrieve(prompt)
            if hits:
                prompt = "# 参考真实代码:\n" + hits[0][:800] + "\n# 任务:\n" + prompt
        ids = tok.encode(prompt)
        if not ids:
            ids = [0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        print("[你输入]\n" + prompt)

        if args.beam > 1:
            from decode import beam_search
            gen, _ = beam_search(model, tok, ids, max_new_tokens=args.max_tokens,
                                 beam_width=args.beam, temperature=args.temperature)
        elif args.self_consistency > 1:
            from decode import self_consistency
            gen = self_consistency(model, tok, ids, n_samples=args.self_consistency,
                                   max_tokens=args.max_tokens, gate=args.gate,
                                   temperature=args.temperature, device=device)
        else:
            out, scores = model.generate(idx, args.max_tokens,
                                         temperature=args.temperature,
                                         top_k=args.top_k, top_p=args.top_p,
                                         min_p=args.min_p, typical_p=args.typical_p,
                                         repeat_penalty=args.repeat_penalty,
                                         return_scores=True)
            gen = tok.decode(out[0][len(ids):].tolist())
            if args.gate:
                from anti_hallucination import parse_gate, confidence_score
                ok, rep = parse_gate(gen)
                conf = confidence_score(scores)
                print(f"  [防幻觉] 语法OK={ok} 置信度={conf:.3f} "
                      f"未定义名={rep.get('undefined', [])}")
        print("=" * 60)
        print("[模型生成]\n" + gen)
        print("=" * 60)

    if args.interactive:
        print("交互模式（Ctrl+D 退出）：")
        while True:
            try:
                p = input(">>> ")
            except EOFError:
                break
            if p.strip():
                run(p)
    else:
        run(args.prompt)


if __name__ == "__main__":
    main()
