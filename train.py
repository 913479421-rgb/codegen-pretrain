#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py —— 从零预训练脚本（一代升级版）
==========================================
升级特性：
  - 正确解耦的 AdamW（权重衰减只在权重，不看 bias/norm）
  - 线性 warmup + cosine 衰减
  - 梯度裁剪 + 梯度累积
  - EMA（指数移动平均）权重，生成用 EMA 版本更稳
  - 混合精度（GPU 上自动 bf16/fp16 autocast，显著省显存提速）

用法：
    python train.py --config small
    python train.py --config tiny --steps 300
    python train.py --resume checkpoints/ckpt_5000.pt
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config, save_config
from model import GPT


def load_tokens(path):
    return np.fromfile(path, dtype=np.uint16).astype(np.int64)


class DataLoader:
    def __init__(self, path, batch_size, block_size, seed=42, rank=0, world_size=1):
        self.data = load_tokens(path)
        self.batch_size = batch_size
        self.block_size = block_size
        self.rank = rank
        self.world_size = world_size
        # 多卡：每 rank 读自己那一份数据分片
        if world_size > 1:
            per = len(self.data) // world_size
            self.data = self.data[rank * per:(rank + 1) * per]
        self.pos = 0
        self.rng = np.random.default_rng(seed + rank)

    def next_batch(self):
        n = self.batch_size * self.block_size
        if len(self.data) < n + 1:
            buf = np.tile(self.data, (n + 1) // len(self.data) + 1)
        else:
            buf = self.data
        if self.pos + n + 1 > len(buf):
            self.pos = 0
        chunk = buf[self.pos:self.pos + n + 1]
        self.pos += n
        return (torch.from_numpy(chunk[:-1].reshape(self.batch_size, self.block_size)),
                torch.from_numpy(chunk[1:].reshape(self.batch_size, self.block_size)))


@torch.no_grad()
def estimate_loss(model, train_loader, eval_loader, iters, block_size, device, use_amp):
    model.eval()
    out = {}
    for name, loader in [("train", train_loader), ("val", eval_loader)]:
        losses = []
        for _ in range(iters):
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=use_amp and device.type == "cuda"):
                _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = float(np.mean(losses))
    model.train()
    return out


@torch.no_grad()
def ema_update(ema_model, model, decay):
    ema_p = ema_model.state_dict()
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            ema_p[k].mul_(decay).add_(v, alpha=1 - decay)
        else:
            ema_p[k].copy_(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="small", choices=["tiny", "small", "medium", "large"])
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compile", action="store_true",
                    help="世界级引擎：启用 torch.compile 图编译加速")
    ap.add_argument("--wandb", action="store_true",
                    help="世界级引擎：启用 WandB 训练可视化")
    ap.add_argument("--ds-config", default="",
                    help="DeepSpeed 配置（世界级引擎：多卡 ZeRO），如 config/ds_zero3.json")
    ap.add_argument("--ddp", action="store_true",
                    help="多卡 DDP（用 torchrun 启动），数据按 rank 分片")
    args = ap.parse_args()

    cfg = get_config(args.config)
    if args.steps:
        cfg.max_steps = args.steps
    if args.compile:
        cfg.use_compile = True
    if args.wandb:
        cfg.use_wandb = True

    # ---------- 多卡 DDP 初始化 ----------
    rank, world_size = 0, 1
    is_main = True
    if args.ddp:
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        is_main = (rank == 0)
        print(f"[ddp] rank={rank}/{world_size}")

    root = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(root, "data", "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            cfg.vocab_size = json.load(f)["vocab_size"]
    else:
        print("[警告] 找不到 data/meta.json，请先运行 prepare_data.py")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    save_config(cfg, os.path.join(cfg.ckpt_dir, "config.json"))

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print(f"[train] 设备={device} | AMP={'on' if (cfg.use_amp and device.type=='cuda') else 'off'} dtype={amp_dtype}")

    train_loader = DataLoader("data/code_train.bin", cfg.batch_size, cfg.block_size, cfg.seed, rank, world_size)
    eval_loader = DataLoader(cfg.eval_data_path, cfg.batch_size, cfg.block_size, cfg.seed + 1, rank, world_size)

    model = GPT(cfg).to(device)
    print(f"[train] 模型参数量 = {model.param_count()/1e6:.1f}M")

    # 多卡 DDP 包装
    if args.ddp:
        try:
            from torch.nn.parallel import DistributedDataParallel
            model = DistributedDataParallel(model, device_ids=None if device.type == "cpu" else [rank])
        except Exception as e:
            print(f"[警告] DDP 包装失败：{e}")

    # 世界级引擎：torch.compile 图编译（提升训练吞吐）
    if cfg.use_compile and torch.__version__ >= "2.0":
        try:
            model = torch.compile(model)
            print("[train] 已启用 torch.compile 图编译")
        except Exception as e:
            print(f"[警告] torch.compile 失败，跳过：{e}")

    # 世界级引擎：WandB 可视化（仅主进程）
    wandb = None
    if cfg.use_wandb and is_main:
        try:
            import wandb
            wandb.init(project=cfg.wandb_project, config=cfg.__dict__)
            print("[train] 已连接 WandB")
        except Exception as e:
            print(f"[警告] WandB 不可用，跳过：{e}")
            wandb = None

    # 正确 AdamW：权重衰减只作用于 >=2 维（权重），bias/norm 不衰减
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=(device.type == "cuda"))

    # EMA 模型
    ema_model = None
    if cfg.use_ema:
        ema_model = GPT(cfg).to(device)
        ema_model.load_state_dict(model.state_dict())
        for p in ema_model.parameters():
            p.requires_grad_(False)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        if ema_model and "ema" in ck:
            ema_model.load_state_dict(ck["ema"])
        print(f"[train] 从 {args.resume} 续训，step={start_step}")

    def lr_at(step):
        if step < cfg.warmup_steps:
            return cfg.lr * step / cfg.warmup_steps
        prog = (step - cfg.warmup_steps) / max(1, (cfg.max_steps - cfg.warmup_steps))
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1)))

    print(f"[train] 开始训练，共 {cfg.max_steps} 步 (等效batch={cfg.batch_size*cfg.grad_accum})")
    model.train()

    # ---------- DeepSpeed 引擎路径 ----------
    if args.ds_config:
        from train_deepspeed import run_deepspeed
        run_deepspeed(model, opt, cfg, args, train_loader, eval_loader,
                      device, amp_dtype, ema_model, is_main, wandb, lr_at, start_step)
        return

    t0 = time.time()
    step = start_step
    micro = 0
    opt.zero_grad()
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.type == "cuda"))

    while step < cfg.max_steps:
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda" if device.type == "cuda" else "cpu",
                            dtype=amp_dtype, enabled=(cfg.use_amp and device.type == "cuda")):
            _, loss = model(x, y)
        loss = loss / cfg.grad_accum
        scaler.scale(loss).backward()
        micro += 1

        if micro % cfg.grad_accum == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            if ema_model:
                ema_update(ema_model, model, cfg.ema_decay)

            if step % cfg.log_every == 0:
                lv = loss.item() * cfg.grad_accum
                print(f"step {step}/{cfg.max_steps} | loss {lv:.4f} "
                      f"| lr {opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.0f}s")
                if wandb is not None:
                    wandb.log({"loss": lv, "lr": opt.param_groups[0]["lr"], "step": step})

            if step % cfg.eval_every == 0 and step > 0 and is_main:
                ev = estimate_loss(model, train_loader, eval_loader, cfg.eval_iters,
                                   cfg.block_size, device, cfg.use_amp)
                print(f"  [eval] step {step} | train {ev['train']:.4f} | val {ev['val']:.4f}")
                if ema_model:
                    eve = estimate_loss(ema_model, train_loader, eval_loader,
                                        cfg.eval_iters, cfg.block_size, device, cfg.use_amp)
                    print(f"  [eval-EMA] step {step} | train {eve['train']:.4f} | val {eve['val']:.4f}")
                if wandb is not None:
                    wandb.log({"val_loss": ev["val"], "train_loss_eval": ev["train"], "step": step})

            if step % cfg.save_every == 0 and step > 0 and is_main:
                path = os.path.join(cfg.ckpt_dir, f"ckpt_{step}.pt")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "ema": ema_model.state_dict() if ema_model else None,
                            "step": step, "config": cfg.__dict__}, path)
                print(f"  [save] {path}")

            step += 1
            micro = 0
            if step % 200 == 0:
                t0 = time.time()

    if is_main:
        path = os.path.join(cfg.ckpt_dir, "final.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "ema": ema_model.state_dict() if ema_model else None,
                    "step": step, "config": cfg.__dict__}, path)
        print(f"[train] 训练完成 ✅ 最终权重: {path} (含 EMA)")
        print("下一步: python eval.py --checkpoint checkpoints/final.pt")
    if args.ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
