#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_benchmark.py —— 第三方世界级基准评测（HumanEval / MBPP）
==============================================================
代码生成的国际标准评测：HumanEval（OpenAI）与 MBPP（Google）。
指标：pass@k —— 模型生成的代码通过对应单元测试的比例。

用法：
    # 下载基准数据（需网络；会自动尝试从 HuggingFace 拉取）
    python eval_benchmark.py --download --out data/bench

    # 评测（加载 checkpoint，默认评测前 N 题）
    python eval_benchmark.py --checkpoint checkpoints/final.pt --dataset humaneval \
        --num-tasks 5 --samples 3

    # 标准 pass@1（k=1）
    python eval_benchmark.py --checkpoint checkpoints/final.pt --dataset humaneval --k 1
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))


def download_dataset(out_dir, name="humaneval"):
    """下载基准数据为 JSONL：{prompt, test, entry_point}。"""
    os.makedirs(out_dir, exist_ok=True)
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[错误] 未安装 datasets：{e}")
        sys.exit(1)
    path = os.path.join(out_dir, name + ".jsonl")
    repo = {"humaneval": "openai_humaneval", "mbpp": "mbpp"}[name]
    split = "test" if name == "humaneval" else "test"
    ds = load_dataset(repo, split=split)
    with open(path, "w", encoding="utf-8") as f:
        for ex in ds:
            rec = {"prompt": ex.get("prompt", ""),
                   "entry_point": ex.get("entry_point", ""),
                   "test": ex.get("test", "")}
            if name == "mbpp":
                rec = {"prompt": ex.get("text", ""),
                       "entry_point": ex.get("test_code", ""),
                       "test": ex.get("test_code", "")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[bench] 已下载 {name} -> {path}")
    return path


def evaluate(checkpoint, dataset, num_tasks, k, samples_per_task,
             use_ema, max_tokens):
    import torch
    from config import TrainConfig
    from model import GPT
    from tokenizer import BPETokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint, map_location="cpu")
    cfg = TrainConfig()
    for kk, v in ck["config"].items():
        if hasattr(cfg, kk):
            setattr(cfg, kk, v)
    model = GPT(cfg).to(device)
    src = ck.get("ema") if (use_ema and ck.get("ema")) else ck["model"]
    model.load_state_dict(src)
    model.eval()
    tok = BPETokenizer.load(os.path.join("data", "vocab.json"))

    data_path = os.path.join("data", "bench", dataset + ".jsonl")
    if not os.path.exists(data_path):
        print(f"[错误] 缺基准数据 {data_path}，先运行 --download")
        sys.exit(1)
    tasks = [json.loads(l) for l in open(data_path, encoding="utf-8")][:num_tasks]

    passed = 0
    for t in tasks:
        prompt = t["prompt"]
        test_code = t["test"]
        entry = t.get("entry_point", "")
        ids = tok.encode(prompt)
        task_pass = 0
        for _ in range(samples_per_task):
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(idx, max_tokens, temperature=0.8, top_k=40, top_p=0.9)
            gen = tok.decode(out[0][len(ids):].tolist())
            # 组装可执行测试
            full = prompt + "\n" + gen + "\n" + test_code
            try:
                import subprocess
                r = subprocess.run([sys.executable, "-c", full],
                                   capture_output=True, timeout=10)
                if r.returncode == 0:
                    task_pass += 1
            except Exception:
                pass
        if task_pass > 0:
            passed += 1

    total = len(tasks)
    print(f"[bench] {dataset} pass@{k} = {passed}/{total} = "
          f"{passed/total:.3f}" if total else "无任务")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", default="data/bench")
    ap.add_argument("--dataset", default="humaneval", choices=["humaneval", "mbpp"])
    ap.add_argument("--checkpoint", default="checkpoints/final.pt")
    ap.add_argument("--num-tasks", type=int, default=5)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--samples", type=int, default=0, help="每题的采样数（默认=k）")
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    if args.download:
        download_dataset(args.out, args.dataset)
        return
    samples = args.samples or args.k
    evaluate(args.checkpoint, args.dataset, args.num_tasks, args.k,
             samples, args.use_ema, args.max_tokens)


if __name__ == "__main__":
    main()
