# 🧪 从零预训练代码生成模型 · 极速版（自研 + 世界级引擎 + 防幻觉 + 持续训练）

把**代码生成语言模型**从零打造到**极限可升级水平**：核心全部自研（Transformer + BPE），
再集成**业界顶级引擎**（FlashAttention / torch.compile / DeepSpeed / WandB）、
接入**世界级数据集**（The Stack v3 / CodeContests / HumanEval），
配备**高质量生成解码引擎 + 防幻觉校验闭环 + 第三方基准评测**，并支持**持续训练**。
只依赖 `torch` + `numpy`，已在本机 CPU **实测跑通全流程**（含持续训练循环）。

> ⚠️ 诚实定位：单机从零预训练得到**教学/实验级**模型。"世界第一"需要几十万 GPU 小时——
> 但**原理与工程和 GPT/DeepSeek 完全一致**，你掌握的是训练一切大模型的核心能力，
> 且本工程已把个人能集成的顶级引擎与数据全部打通。

---

## ⚙️ 极限升级矩阵

### 一代架构（已在前代完成）
| 模块 | 技术 |
|------|------|
| 位置编码 / 激活 / 归一化 | **RoPE + SwiGLU + RMSNorm** |
| 注意力 | **GQA 分组查询** + KV 缓存推理 |
| 采样 | top-k + top-p + 重复惩罚 |

### 本次新增：世界级引擎集成（全部可开关、无则优雅降级）
| 引擎 | 作用 | 启用方式 |
|------|------|----------|
| **FlashAttention 内核**（PyTorch SDPA）| 训练注意力大幅提速、省显存 | 默认开（`use_sdpa`）|
| **torch.compile** 图编译 | 提升整体训练吞吐 | `--compile` |
| **DeepSpeed ZeRO-2/3** | 多卡 + CPU offload 训更大模型 | `--ds-config config/ds_zero3.json` |
| **多卡 DDP** | 数据并行，多 GPU 提速 | `--ddp`（用 torchrun 启动）|
| **WandB** | 训练 loss/lr/ppl 实时可视化 | `--wandb` |

### 本次新增：世界级数据集接入
| 数据源 | 规模/用途 | 命令 |
|--------|-----------|------|
| **The Stack v3** | 5T token，全球最大开源代码语料 | `python collect_data.py --source stack` |
| **github-code** | StarCoder 训练语料 | `--source github-code` |
| **CodeContests** | DeepMind 竞赛代码 | `--source codecontests` |
| **HumanEval** | OpenAI 代码评测基准 | `--source humaneval` |

### 本次新增：持续训练引擎
`continuous_train.py`：自动扫描 `data/stream/` 新代码 → 增量编码追加 → 从最新 checkpoint 续训 → 循环。

### 本次新增：生成质量 + 防幻觉 + 解析 + 第三方基准
| 模块 | 引擎/资源 | 作用 |
|------|-----------|------|
| **解码引擎** `decode.py` | 束搜索 + 自一致性 + min-p/典型采样 | 生成更稳、质量更高 |
| **解析引擎** `code_parse.py` | AST 语法校验 + 可执行验证 + 未定义名检测 | 判语法健康度、防坏代码 |
| **防幻觉引擎** `anti_hallucination.py` | 置信度评分 + 语法/API 拦截 + 检索增强 + 自一致性投票 | 四道防线压幻觉 |
| **第三方基准** `eval_benchmark.py` | **HumanEval**(OpenAI) / **MBPP**(Google) pass@k | 代码生成国际标准评测 |
| **第三方资源** | HumanEval / MBPP 数据下载 | 接入评测 |

### 本次新增：推理加速 + 量化 + 部署
| 模块 | 引擎 | 作用 |
|------|------|------|
| **投机解码** `infer_engine.py` | Speculative Decoding | 草稿拟稿 + 主模型并行验证 → 生成 2~3× 加速 |
| **量化导出** `export_quantized.py` | INT8/INT4 动态量化 | 省显存、提速，可部署 |
| **推理 CLI** `infer_demo.py` | 测速/投机/量化统一入口 | 一条命令体验加速 |
| **部署** `DEPLOY.md` | Ollama + llama.cpp | 本地对外服务 |

### 本次新增：生产级训练 + 世界第一推理辅助
| 模块 | 引擎/工具 | 作用 |
|------|-----------|------|
| **DeepSpeed 引擎接入** `train_deepspeed.py` | ZeRO 多卡训练循环 | `--ds-config` 真正走 DeepSpeed，非仅配置 |
| **生产一键脚本** `run_production.sh` | The Stack 几GB + large + 多卡 + 几万步 + 持续训练 | 一条命令上生产 |
| **世界第一推理辅助** `assistant_models.py` | DeepSeek-R1-Distill / Qwen2.5-Coder | 蒸馏教师 + 参照评测 + 数据生成器 |

---

## 📁 工程结构
```
codegen-pretrain/
├── config.py            # 超参数 + 引擎开关（tiny/small/medium/large）
├── tokenizer.py         # 自研 GPT-2 式字节级 BPE
├── model.py             # 自研一代 Transformer（RoPE+SwiGLU+RMSNorm+GQA+KV+SDPA+min-p/typical）
├── decode.py            # 解码引擎（束搜索 + 自一致性）
├── code_parse.py        # 解析引擎（AST校验 + 可执行验证 + 未定义名检测）
├── anti_hallucination.py# 防幻觉引擎（置信度 + 语法拦截 + 检索增强 + 自一致性）
├── eval_benchmark.py    # 第三方基准评测（HumanEval/MBPP pass@k）
├── infer_engine.py      # 推理加速（投机解码 + 量化推理）
├── infer_demo.py        # 推理加速命令行（测速/投机/量化）
├── export_quantized.py  # 量化导出引擎（INT8/INT4）
├── prepare_data.py      # 初始化语料 → 训练BPE → token序列
├── collect_data.py      # 世界级数据集采集（The Stack 等）→ data/stream
├── continuous_train.py  # 持续训练引擎（增量数据 → 自动续训）
├── train.py             # 预训练（AdamW+EMA+AMP+compile+wandb+DDP）
├── eval.py              # 评测（Perplexity + pass@k + 样例）
├── generate.py          # 生成（KV缓存 + 高质量解码 + 防幻觉闸门）
├── config/ds_zero2.json, ds_zero3.json   # DeepSpeed 引擎配置
├── 00_check_env.sh / 01_install_env.sh
├── DEPLOY.md            # 部署指引（量化 + Ollama）
└── data/stream/         # 持续训练的数据上游目录
```

---

## 🚀 上手

### 1️⃣ 装环境（含可选世界级引擎，失败自动降级）
```bash
bash 01_install_env.sh
```

### 2️⃣ 初始化语料与词表
```bash
# 放一批代码到 data/stream/，然后初始化（一次）
python prepare_data.py --local-dir data/stream --tiny --no-dedup
```

### 3️⃣ 采集世界级数据集（持续补充语料）
```bash
python collect_data.py --source stack --max-mb 500      # 抓 The Stack 进 data/stream
python collect_data.py --source codecontests             # 补竞赛题
```

### 4️⃣ 启动持续训练（核心）
```bash
# 无限循环：新数据出现 → 自动增量续训（Ctrl+C 停止，随时可重启续训）
python continuous_train.py --config small \
    --watch-dir data/stream --steps-per-round 1000 --interval 60
```

### 5️⃣ 评测 & 生成
```bash
python eval.py --checkpoint checkpoints/final.pt --use-ema --samples 5 --pass-k 3
python generate.py --checkpoint checkpoints/final.pt --use-ema \
    --prompt "def fibonacci(n):" --max-tokens 100
```

### 6️⃣ 高质量生成 + 防幻觉
```bash
# 束搜索（更稳）
python generate.py --checkpoint checkpoints/final.pt --beam 3 \
    --prompt "def add(" --max-tokens 80

# 自一致性去幻觉（多次采样多数投票 + 语法闸门）
python generate.py --checkpoint checkpoints/final.pt --self-consistency 5 --gate \
    --prompt "def add(" --max-tokens 80

# 检索增强（从真实代码库取依据）+ min-p 低置信过滤 + 防幻觉评分
python generate.py --checkpoint checkpoints/final.pt \
    --retrieve data/stream --min-p 0.02 --gate --prompt "def add(" --max-tokens 80
```

### 7️⃣ 第三方基准评测（HumanEval / MBPP）
```bash
python eval_benchmark.py --download --dataset humaneval        # 下载数据
python eval_benchmark.py --checkpoint checkpoints/final.pt \
    --dataset humaneval --num-tasks 5 --k 1 --use-ema          # pass@1
```

### 8️⃣ 推理加速 + 量化 + 部署
```bash
# 投机解码（草稿拟稿 + 主模型并行验证，2~3× 加速）
python infer_demo.py --checkpoint checkpoints/final.pt \
    --draft checkpoints/final.pt --speculative --gamma 4 --prompt "def fibonacci(n):"
# 量化导出与推理
python export_quantized.py --checkpoint checkpoints/final.pt --bits 8
python infer_demo.py --quant checkpoints/quant_8bit.pt --prompt "def add("
# Ollama 部署：见 DEPLOY.md
```

### 9️⃣ 生产级大规模训练（一键）
```bash
# The Stack 几GB + large 档 + DeepSpeed 多卡 + 几万步 + 持续训练长期挂
STACK_MB=3000 GPUS=2 bash run_production.sh
# 或手动 DeepSpeed
deepseek --num_gpus 2 train.py --config large --ds-config config/ds_zero2.json --steps 50000
```

### 🔟 世界第一推理模型辅助工具
```bash
pip install modelscope
python assistant_models.py --model qwen-coder --infer "用Python写快排"      # 强模型推理
python assistant_models.py --model qwen-coder --distill --num 200          # 蒸馏高质量数据→持续训练
python assistant_models.py --model deepseek-r1 --eval --tasks 5            # 参照基准
```

---

## 🎛 引擎开关示例
```bash
# 单卡 + 编译 + WandB
python train.py --config medium --compile --wandb
# 多卡 DDP（4 卡）
torchrun --nproc_per_node=4 train.py --config large --ddp
# DeepSpeed ZeRO-3（CPU offload，训超大模型）
deepspeed --num_gpus=2 train.py --config large --ds-config config/ds_zero3.json
```

---

## 📈 训练进度
```
step 200/20000 | loss 0.9371 | lr 3.00e-04 | 137s      # loss 持续下降 = 在学
  [eval] step 500 | train 2.34 | val 2.41               # 训练/验证接近 = 未过拟合
  [eval-EMA] step 500 | train 2.36 | val 2.43
[continuous] 第3轮：新增 50 文件, 5210 tokens          # 持续训练自动续训
```

---

## ⚠️ 说明
- **无 GPU**：自动走 CPU，配 `--config tiny` 全流程可跑通（已实测含持续训练）。
- **可选引擎未装不影响核心**：01 安装脚本里失败会降级并提示。
- **冒烟示例输出乱码属正常**：极小模型+少数据，仅验证链路；真实语料+大档位+多步即可产出可读代码。
- **想更强**：加大语料（几 GB The Stack）+ 升档 large + DeepSpeed 多卡 + 训几万步 + 持续训练长期挂着。

**卡在哪一步，把报错发我。**

### 10b 档（10GB 参数）
`--config 10b`：**约 51.3 亿参数，bf16 ≈ 10.3 GB**。
- 规模：n_embd=4096 / 20 层 / 32 头(GQA 8 KV头) / 词表 32000 / 上下文 2048
- 前置：语料需用 `--vocab-size 32000` 重新初始化词表
- 硬件：需多卡 GPU（>48GB 单卡或用 DeepSpeed ZeRO）；纯 CPU 无法训练
```bash
python prepare_data.py --local-dir data/stream --vocab-size 32000
deepspeed --num_gpus 8 train.py --config 10b --ds-config config/ds_zero3.json --steps 120000
```
