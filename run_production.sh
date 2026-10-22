#!/usr/bin/env bash
# ============================================================
# run_production.sh —— 生产级大规模训练一键脚本
# 串起：几GB The Stack 语料 → large 档 → DeepSpeed 多卡 →
#       几万步 → 持续训练长期挂着。
#
# 用法（在你有 GPU 的机器上）：
#   bash run_production.sh                     # 用默认：The Stack 3GB, large, DS
#   STACK_MB=6000 bash run_production.sh       # 拉到 6GB
#   GPUS=4 bash run_production.sh              # 4 卡
#
# 说明：需要已装 deepspeed（01_install_env.sh 已含，失败可 pip install deepspeed）。
# ============================================================
set -e
cd "$(dirname "$0")"

# ---------- 参数（环境变量可覆盖）----------
STACK_MB="${STACK_MB:-3000}"   # The Stack 下载上限（MB），默认 3GB
CONFIG="${CONFIG:-large}"      # 档位
GPUS="${GPUS:-2}"              # DeepSpeed 使用卡数
STEPS="${STEPS:-50000}"        # 训练步数（几万步）
DS_CFG="${DS_CFG:-config/ds_zero2.json}"  # 卡不多用 zero2；卡多/显存紧用 zero3
INTERVAL="${INTERVAL:-120}"    # 持续训练扫描间隔（秒）

source .venv/bin/activate
echo "================ 生产级训练 ================"
echo "  语料: The Stack 上限 ${STACK_MB}MB | 档位: ${CONFIG} | GPU: ${GPUS}"
echo "  步数: ${STEPS} | DeepSpeed: ${DS_CFG} | 持续训练间隔: ${INTERVAL}s"

# ---------- 0. 环境自检 ----------
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || {
    echo "[警告] 未检测到 GPU。DeepSpeed 需多卡，请确认机器配置。"; }

# ---------- 1. 采集世界级语料（几 GB The Stack 到 data/stream）----------
echo ">>> [1/5] 采集 The Stack 语料（约 ${STACK_MB}MB）..."
mkdir -p data/stream
python collect_data.py --source stack --max-mb "$STACK_MB" --out-dir data/stream \
    || echo "[警告] 在线采集失败（可能无网络/需HF token）。可用本地代码目录替代："
    echo "        python collect_data.py --source github-code --max-mb $STACK_MB"

# ---------- 2. 初始化语料与 BPE 词表 ----------
echo ">>> [2/5] 初始化语料与 BPE 词表（large 档词表自动放大）..."
python prepare_data.py --local-dir data/stream --vocab-size 16000

# ---------- 3. 训练（DeepSpeed 多卡，几万步）----------
echo ">>> [3/5] DeepSpeed 多卡训练（large 档, $STEPS 步）..."
if command -v deepspeed >/dev/null 2>&1; then
    deepspeed --num_gpus "$GPUS" train.py --config "$CONFIG" \
        --ds-config "$DS_CFG" --steps "$STEPS"
else
    echo "[警告] 未装 deepspeed，回退单卡/CPU（用 torchrun 可多卡 DDP）。"
    python train.py --config "$CONFIG" --steps "$STEPS"
fi

# ---------- 4. 评测 ----------
echo ">>> [4/5] 评测（Perplexity + 样例）..."
python eval.py --checkpoint checkpoints/final.pt --use-ema --samples 3 || true

# ---------- 5. 持续训练长期挂着 ----------
echo ">>> [5/5] 启动持续训练（长期挂着，Ctrl+C 停止；随时重启续训）..."
echo "     新语料会持续增量加入并自动续训。"
python continuous_train.py --config "$CONFIG" \
    --watch-dir data/stream --steps-per-round 5000 --interval "$INTERVAL"
