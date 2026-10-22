# 🚀 部署指引：量化模型 + Ollama 本地服务

训练好的模型可以两种方式落地使用：

## 方式 A：量化模型直接推理（最简单）
```bash
# 导出 INT8
python export_quantized.py --checkpoint checkpoints/final.pt --bits 8
# 加载量化模型生成
python export_quantized.py --bits 8 --load --prompt "def add(a, b):"
```
- INT8：精度几乎无损，速度/显存明显改善（PyTorch 内置，无第三方依赖）
- INT4：更省显存（需 `pip install torchao`，脚本自动探测）

配合推理加速：
```bash
# 投机解码（草稿模型需与主模型 tokenizer 一致；草稿可先用同构小模型）
python infer_demo.py --checkpoint checkpoints/final.pt --speculative --gamma 4 \
    --prompt "def fibonacci(n):"
# 测速对比
python infer_demo.py --checkpoint checkpoints/final.pt --bench
# 量化模型推理
python infer_demo.py --quant checkpoints/quant_8bit.pt --prompt "def add("
```

---

## 方式 B：Ollama 部署（对外提供服务，推荐）
Ollama 是主流本地大模型服务框架，可一键起 HTTP API、命令行交互。

### 步骤 1：导出 GGUF（需 llama.cpp 标准工具链）
本工程自研模型为 PyTorch 格式。转 GGUF 需用 llama.cpp 的转换工具（标准做法）：
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make
# 把 PyTorch checkpoint 先整理成 HF 目录，再转 GGUF（Q4_K_M 常用）
python convert_hf_to_gguf.py <你的模型目录> --outfile codegen.gguf --outtype q4_k_m
```

### 步骤 2：创建 Ollama 模型
```bash
# 写 Modelfile
cat > Modelfile <<'EOF'
FROM ./codegen.gguf
TEMPLATE """{{ .Prompt }}"""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

ollama create codegen -f Modelfile
```

### 步骤 3：使用
```bash
ollama run codegen "用 Python 写一个快速排序"   # 命令行
# HTTP API
curl http://localhost:11434/api/generate -d '{
  "model": "codegen", "prompt": "def quicksort(arr):",
  "stream": false}'
```

---

## ⚠️ 说明
- **GGUF/Ollama 依赖 llama.cpp**，这是把 PyTorch 模型转为通用部署格式的标准工具链。
- 本工程自研模型的 RoPE/GQA/SwiGLU 与主流架构兼容，转 GGUF 时需按 llama.cpp 架构
  映射核对词表与 layer 命名；规模小时直接用量化推理（方式 A）更省事。
- 冒烟模型的输出是随机的（未充分训练），部署真实模型需先用真实语料训好。
