#!/usr/bin/env bash
# ============================================================
# 00_check_env.sh —— 环境检查（GPU/CUDA/Python）
# 用法：bash 00_check_env.sh
# ============================================================
set -e
echo "== Python ==" && python3 --version
echo "" && echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "[警告] 无 nvidia-smi。本工程核心逻辑不依赖 GPU，可先用 CPU 冒烟跑通。"
fi
echo "" && echo "== 结论 =="
echo "GPU 就绪 -> 跑大配置 (small/medium)；仅 CPU -> 用 tiny 配置体验全流程。"
