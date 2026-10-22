#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_deepspeed.py —— DeepSpeed 引擎训练循环（世界级多卡 ZeRO）
================================================================
被 train.py 在 --ds-config 时调用。用 deepspeed.initialize 包装模型与优化器，
由 DeepSpeed 引擎统一管理：梯度累积、混合精度、梯度裁剪、ZeRO 分片、CPU offload。

用法（见 run_production.sh）：
    deepspeed train.py --config large --ds-config config/ds_zero3.json \
        --steps 120000
"""
import os, time, math
import torch
import torch.nn as nn


def run_deepspeed(model, opt, cfg, args, train_loader, eval_loader,
                  device, amp_dtype, ema_model, is_main, wandb, lr_at, start_step):
    """DeepSpeed 引擎训练。model 需已在 cpu（DS 自行搬到各卡）。"""
    try:
        import deepspeed
    except Exception as e:
        raise RuntimeError(f"DeepSpeed 未安装：{e}。请 pip install deepspeed 或用 --ddp 多卡。")

    # 移到 CPU 供 DeepSpeed 初始化（DS 会分配到各 GPU）
    model = model.cpu()
    # 用 DeepSpeed 初始化包装 model + 自定义 AdamW
    ds_engine, opt_ds, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=opt,
        config=args.ds_config,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )
    local_rank = ds_engine.local_rank
    is_local_main = (local_rank in (0, -1))
    if is_main and is_local_main:
        print(f"[deepspeed] 引擎启动 | ZeRO stage={args.ds_config} | world={ds_engine.world_size}")

    # DeepSpeed 引擎在 step 内处理 lr 调度则用 ds lr；此处仍手动设 lr 基准
    for g in opt_ds.param_groups:
        g["lr"] = lr_at(start_step) if hasattr(opt_ds, "param_groups") else cfg.lr

    ds_engine.train()
    t0 = time.time()
    step = start_step
    micro = 0
    opt_ds.zero_grad()

    while step < cfg.max_steps:
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        outputs = ds_engine(x, labels=y)
        loss = outputs["loss"] / cfg.grad_accum
        ds_engine.backward(loss)
        micro += 1

        if micro % cfg.grad_accum == 0:
            ds_engine.step()
            opt_ds.zero_grad()
            micro = 0

            if ema_model and ds_engine.module is not None:
                _ema_update(ema_model, ds_engine.module, cfg.ema_decay)

            if step % cfg.log_every == 0 and is_local_main:
                lv = loss.item() * cfg.grad_accum
                print(f"[DS] step {step}/{cfg.max_steps} | loss {lv:.4f} | {time.time()-t0:.0f}s")
                if wandb is not None:
                    wandb.log({"loss": lv, "step": step})

            if step % cfg.save_every == 0 and step > 0 and is_main and is_local_main:
                path = os.path.join(cfg.ckpt_dir, f"ckpt_{step}.pt")
                torch.save({"model": ds_engine.module.state_dict(),
                            "opt": opt_ds.state_dict(),
                            "ema": ema_model.state_dict() if ema_model else None,
                            "step": step, "config": cfg.__dict__}, path)
                print(f"  [save] {path}")
            step += 1
            if step % 200 == 0:
                t0 = time.time()

    if is_main and is_local_main:
        path = os.path.join(cfg.ckpt_dir, "final.pt")
        torch.save({"model": ds_engine.module.state_dict(),
                    "opt": opt_ds.state_dict(),
                    "ema": ema_model.state_dict() if ema_model else None,
                    "step": step, "config": cfg.__dict__}, path)
        print(f"[deepspeed] 训练完成 ✅ {path}")


@torch.no_grad()
def _ema_update(ema_model, model, decay):
    ema_p = ema_model.state_dict()
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            ema_p[k].mul_(decay).add_(v, alpha=1 - decay)
        else:
            ema_p[k].copy_(v)
