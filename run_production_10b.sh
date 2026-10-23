#!/usr/bin/env bash
# ============================================================
# run_production_10b.sh —— 10GB参数模型生产级训练一键脚本
# 串起：The Stack 几GB语料 → 10b档 → DeepSpeed多卡 →
#       12万步 → 持续训练长期挂。
# 由控制台动作 prod_10b 触发，或命令行直接跑。
#
# 用法（在你有 GPU 的机器上，或控制台点"生产训练10b"）：
#   bash run_production_10b.sh
#   STACK_MB=6000 GPUS=8 bash run_production_10b.sh
# ============================================================
set -e
cd "$(dirname "$0")"

# ---------- 参数（环境变量可覆盖）----------
STACK_MB="${STACK_MB:-3000}"     # The Stack 下载上限(MB)，默认3GB
CONFIG="${CONFIG:-10b}"          # 10GB参数档
GPUS="${GPUS:-8}"                # 多卡数
STEPS="${STEPS:-120000}"         # 训练步数
DS_CFG="${DS_CFG:-config/ds_zero3.json}"  # 10B 用 ZeRO-3 分片
VOCAB=32000                      # 10b 档词表必须 32000
INTERVAL="${INTERVAL:-120}"      # 持续训练扫描间隔(秒)

echo "================ 10GB 参数生产级训练 ================"
echo "  语料: The Stack ≤${STACK_MB}MB | 档位: ${CONFIG} | GPU: ${GPUS}"
echo "  词表: ${VOCAB} | 步数: ${STEPS} | DeepSpeed: ${DS_CFG}"

# ---------- 0. GPU 自检 ----------
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || {
    echo "[警告] 未检测到 GPU。10b 档需多卡 GPU，请确认机器。"; }

# ---------- 1. 采集 The Stack 几GB 语料 ----------
echo ">>> [1/4] 采集 The Stack 语料(≤${STACK_MB}MB)..."
mkdir -p data/stream
python collect_data.py --source stack --max-mb "$STACK_MB" --out-dir data/stream \
    || { echo "[错误] 采集失败(可能无网络/需HF token)。请自行放语料到 data/stream/"; exit 1; }

# ---------- 2. 初始化词表 32000 ----------
echo ">>> [2/4] 初始化 BPE 词表(32000)..."
python prepare_data.py --local-dir data/stream --vocab-size "$VOCAB"

# ---------- 3. DeepSpeed 多卡训练 10b ----------
echo ">>> [3/4] DeepSpeed ${GPUS}卡 训练 10b (${STEPS}步)..."
if command -v deepspeed >/dev/null 2>&1; then
    deepspeed --num_gpus "$GPUS" train.py --config "$CONFIG" \
        --ds-config "$DS_CFG" --steps "$STEPS"
else
    echo "[警告] 未装 deepspeed，回退单卡(10b需大显存，建议装 deepspeed)。"
    python train.py --config "$CONFIG" --steps "$STEPS"
fi

# ---------- 4. 持续训练长期挂 ----------
echo ">>> [4/4] 启动持续训练(长期挂, 新语料自动增量续训)..."
python continuous_train.py --config "$CONFIG" \
    --watch-dir data/stream --steps-per-round 5000 --interval "$INTERVAL"
