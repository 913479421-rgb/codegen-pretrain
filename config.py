# -*- coding: utf-8 -*-
"""
config.py —— 一代架构超参数（分档：tiny / small / medium / large）
====================================================================
升级亮点：支持 RoPE / GQA / SwiGLU / RMSNorm / 混合精度 / KV缓存 等一代特性。
选档：python train.py --config small
"""
from dataclasses import dataclass, asdict
import json


@dataclass
class TrainConfig:
    # ---------- 数据 ----------
    data_path: str = "data/code_train.bin"
    eval_data_path: str = "data/code_eval.bin"
    vocab_path: str = "data/vocab.json"

    # ---------- 模型规模 ----------
    vocab_size: int = 8000
    n_layer: int = 6
    n_head: int = 6                 # 查询头数
    n_kv_head: int = 2              # KV 头数（GQA；< n_head 即分组查询注意力）
    n_embd: int = 192
    block_size: int = 256           # 上下文长度
    dropout: float = 0.1

    # ---------- 一代架构开关 ----------
    use_rope: bool = True           # RoPE 旋转位置编码
    use_swiglu: bool = True         # SwiGLU 门控激活
    use_rmsnorm: bool = True        # RMSNorm
    use_kv_cache: bool = True       # 推理 KV 缓存
    use_sdpa: bool = True           # 世界级引擎：FlashAttention 内核（PyTorch SDPA）

    # ---------- 训练引擎 ----------
    use_compile: bool = False       # torch.compile 图编译（显存够、装好依赖时开）
    use_wandb: bool = False         # WandB 训练可视化
    wandb_project: str = "codegen-pretrain"

    # ---------- 训练 ----------
    batch_size: int = 16
    grad_accum: int = 4             # 等效 batch = batch_size * grad_accum
    max_steps: int = 20000
    lr: float = 3e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    use_amp: bool = True            # 混合精度（GPU 上自动 bf16/fp16）
    use_ema: bool = True            # 指数移动平均权重
    ema_decay: float = 0.999

    # ---------- 保存/日志 ----------
    ckpt_dir: str = "checkpoints"
    log_every: int = 20
    save_every: int = 500
    eval_every: int = 500
    eval_iters: int = 100
    seed: int = 42


CONFIGS = {
    "tiny":  dict(vocab_size=6000, n_layer=4, n_head=4, n_kv_head=2, n_embd=128,
                  block_size=128, batch_size=16, max_steps=5000, lr=3e-3),
    "small": dict(vocab_size=8000, n_layer=6, n_head=6, n_kv_head=2, n_embd=192,
                  block_size=256, batch_size=16, max_steps=20000, lr=3e-3),
    "medium": dict(vocab_size=12000, n_layer=12, n_head=12, n_kv_head=4, n_embd=768,
                   block_size=512, batch_size=8, max_steps=50000, lr=2e-3),
    "large": dict(vocab_size=16000, n_layer=24, n_head=16, n_kv_head=8, n_embd=1024,
                  block_size=1024, batch_size=4, max_steps=120000, lr=1.5e-3),
}


def get_config(name: str) -> TrainConfig:
    cfg = TrainConfig()
    if name in CONFIGS:
        for k, v in CONFIGS[name].items():
            setattr(cfg, k, v)
    return cfg


def save_config(cfg: TrainConfig, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
