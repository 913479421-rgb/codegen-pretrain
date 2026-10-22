#!/usr/bin/env bash
# ============================================================
# 01_install_env.sh —— 从零预训练工程的环境安装
# 核心仅 torch + numpy（BPE 与模型自研，无其他第三方依赖）
# 可选世界级引擎（失败不影响核心运行，自动降级）：
#   - wandb       训练可视化
#   - deepspeed   多卡 ZeRO（可选）
#   - datasets    世界级语料采集（The Stack 等）
# 用法：bash 01_install_env.sh
# ============================================================
set -e
ENV_DIR="$(cd "$(dirname "$0")" && pwd)/.venv"

python3 --version | grep -qE "3\.(9|10|11|12)" || { echo "需要 Python 3.9~3.12"; exit 1; }

[ -d "$ENV_DIR" ] || python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo ">>> 安装 numpy ..."
pip install numpy

echo ">>> 安装 PyTorch ..."
# GPU 优先：CUDA 12.1 版；若本机无 GPU 会自动退到 CPU（冒烟测试可用）
pip install torch --index-url https://download.pytorch.org/whl/cu121 \
    || { echo ">>> cu121 失败，尝试 CPU 版（仅用于冒烟/小模型）"; pip install torch; }

echo ">>> 安装可选世界级引擎（wandb / datasets / deepspeed）..."
pip install wandb datasets 2>/dev/null || echo "  [跳过] 可选引擎未装，核心仍可用"
# deepspeed 体积大且依赖特定 torch，失败也接受
pip install deepspeed 2>/dev/null || echo "  [跳过] deepspeed 未装（单卡/小规模不受影响）"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "| CUDA 可用:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
try:
    import wandb; print("WandB: 可用")
except Exception: print("WandB: 未装(可选)")
try:
    import deepspeed; print("DeepSpeed: 可用")
except Exception: print("DeepSpeed: 未装(可选)")
PY

echo "==========================================="
echo "环境就绪。下一步："
echo "  1) 准备语料：python prepare_data.py --local-dir <你的代码目录> --tiny --no-dedup"
echo "  2) 训练：    python train.py --config tiny"
echo "  3) 评测/生成：python eval.py --checkpoint checkpoints/final.pt"
echo "  4) 持续训练：python continuous_train.py --config small --watch-dir data/stream"
echo "==========================================="
