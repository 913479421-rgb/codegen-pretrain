#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
continuous_train.py —— 持续训练引擎
=====================================
自动循环：
  1) 扫描 data/stream/ 目录，找出尚未处理的新代码文件
  2) 用已有 BPE 词表把新代码编码成 token，追加到 data/code_train.bin
  3) 更新 meta 与已处理索引
  4) 有新增数据时，从最新 checkpoint 续训（--resume）跑一轮
  5) 休眠 interval 秒，循环（模型持续在增长语料上学习）

用法：
    # 先初始化语料与词表（一次）
    python prepare_data.py --local-dir data/stream --tiny --no-dedup

    # 持续训练（无限循环，Ctrl+C 停止）
    python continuous_train.py --config small --watch-dir data/stream \
        --steps-per-round 1000 --interval 60

    # 配合定时任务：每 N 分钟跑一轮（见 README cron 示例）
"""
import os, sys, json, time, subprocess, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
INDEX = os.path.join(DATA, "stream_index.json")


def scan_new(watch_dir, index):
    """返回尚未处理的新 .py 文件列表。"""
    new = []
    for dirpath, _, files in os.walk(watch_dir):
        for fn in files:
            if fn.endswith(".py"):
                p = os.path.join(dirpath, fn)
                if p not in index:
                    new.append(p)
    return new


def encode_append(paths, tok, index):
    """把新文件编码追加到 code_train.bin，更新索引与 meta。"""
    ids = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                t = f.read()
            if t.strip():
                ids.extend(tok.encode(t))
                ids.append(0)   # <|endoftext|>
        except Exception:
            continue
        index[p] = True
    if not ids:
        return 0
    import numpy as np
    arr = np.array(ids, dtype=np.uint16)
    # 追加到训练序列
    if os.path.exists(os.path.join(DATA, "code_train.bin")):
        old = np.fromfile(os.path.join(DATA, "code_train.bin"), dtype=np.uint16)
        arr = np.concatenate([old, arr])
    arr.tofile(os.path.join(DATA, "code_train.bin"))
    # 更新 meta
    meta_path = os.path.join(DATA, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta["train_tokens"] = int(len(arr))
    meta["vocab_size"] = tok.vocab_size
    json.dump(meta, open(meta_path, "w"), ensure_ascii=False, indent=2)
    with open(INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f)
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="small")
    ap.add_argument("--watch-dir", default=os.path.join(DATA, "stream"))
    ap.add_argument("--steps-per-round", type=int, default=1000)
    ap.add_argument("--interval", type=int, default=60, help="扫描间隔秒")
    ap.add_argument("--rounds", type=int, default=0, help="跑多少轮（0=无限）")
    args = ap.parse_args()

    # 需要先有词表（由 prepare_data 生成）
    vocab_path = os.path.join(DATA, "vocab.json")
    if not os.path.exists(vocab_path):
        print("[错误] 找不到 data/vocab.json，请先运行：")
        print("  python prepare_data.py --local-dir data/stream --tiny --no-dedup")
        sys.exit(1)
    from tokenizer import BPETokenizer
    tok = BPETokenizer.load(vocab_path)

    index = json.load(open(INDEX)) if os.path.exists(INDEX) else {}
    os.makedirs(args.watch_dir, exist_ok=True)
    print(f"[continuous] 持续训练启动 | config={args.config} | watch={args.watch_dir}")

    def latest_ckpt():
        ckpt_dir = os.path.join(HERE, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            return ""
        ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
                 if f.endswith(".pt") and f != "config.json"]
        return max(ckpts, key=os.path.getmtime) if ckpts else ""

    round_no = 0
    try:
        while True:
            round_no += 1
            new = scan_new(args.watch_dir, index)
            if new:
                n = encode_append(new, tok, index)
                print(f"[continuous] 第{round_no}轮：新增 {len(new)} 文件, {n} tokens")
                cmd = [sys.executable, os.path.join(HERE, "train.py"),
                       "--config", args.config, "--steps", str(args.steps_per_round)]
                resume = latest_ckpt()
                if resume:
                    cmd += ["--resume", resume]
                print(f"[continuous] 执行训练: {' '.join(cmd)}")
                subprocess.run(cmd, cwd=HERE)
            else:
                print(f"[continuous] 第{round_no}轮：无新数据，等待 {args.interval}s")
            if args.rounds and round_no >= args.rounds:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[continuous] 已停止（索引与数据已保存，随时可重启续训）")


if __name__ == "__main__":
    main()
