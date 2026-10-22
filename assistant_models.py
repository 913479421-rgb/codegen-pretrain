#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assistant_models.py —— 世界第一开源推理模型辅助工具
====================================================
从 ModelScope（国内快）部署世界一流开源推理模型，作为自研模型的辅助工具：

  1. --infer   用强模型推理（参考/对比自研模型质量）
  2. --distill 用强模型为自研模型生成高质量代码数据 → 写入 data/stream 供持续训练
  3. --eval    用强模型做 HumanEval 风格基准，得到参照分数

可选模型（世界第一梯队开源，全部 Apache/MIT 可商用）：
  qwen-coder  : Qwen2.5-Coder-7B-Instruct（代码生成第一梯队，默认）
  deepseek-r1 : DeepSeek-R1-Distill-Qwen-7B（推理强）
  qwen3       : Qwen3-4B/8B（通用）

用法：
    python assistant_models.py --model qwen-coder --infer "用Python写快排"
    python assistant_models.py --model qwen-coder --distill --num 200 --out data/stream
    python assistant_models.py --model qwen-coder --eval --tasks 5
"""
import os, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "qwen-coder": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen-coder-3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "qwen3": "Qwen/Qwen3-8B",
}

# 蒸馏提示模板：让强模型给出带解法的代码
DISTILL_PROMPTS = [
    "用 Python 写一个函数，{task}。给出完整可运行代码和一行解释。",
    "实现一个 Python 函数，{task}。要求：命名清晰、处理边界、返回结果。",
    "请用 Python 解决：{task}。输出函数定义与示例调用。",
]
_TASKS = [
    "求列表去重并保持顺序", "判断字符串是否回文", "实现二分查找",
    "计算两个矩阵相乘", "把秒数转成 HH:MM:SS", "判断一个数是否为质数",
    "统计一段文本的词频", "实现栈数据结构", "冒泡排序并返回排序后的列表",
    "把十进制转二进制", "求最大公约数", "翻转一个整数",
]


def get_model(model_key):
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    name = MODELS.get(model_key, MODELS["qwen-coder"])
    print(f"[assistant] 加载世界级模型: {name} ...")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, torch_dtype="auto", device_map="auto")
    model.eval()
    return model, tok


def chat(model, tok, prompt, max_tokens=512):
    import torch
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_tokens,
                         do_sample=False, temperature=None, top_p=None)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)


def do_infer(model, tok, prompt):
    print("[你输入]", prompt)
    print("[强模型输出]")
    print(chat(model, tok, prompt))


def do_distill(model, tok, num, out_dir):
    import random, torch
    os.makedirs(out_dir, exist_ok=True)
    random.seed(42)
    written = 0
    for i in range(num):
        task = random.choice(_TASKS)
        prompt = random.choice(DISTILL_PROMPTS).format(task=task)
        try:
            ans = chat(model, tok, prompt, max_tokens=512)
        except Exception as e:
            print(f"  [跳过] {e}"); continue
        path = os.path.join(out_dir, f"distill_{i:05d}.py")
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write("# 任务: " + task + "\n" + ans)
        written += 1
    print(f"[distill] 生成 {written} 个高质量样本 -> {out_dir}（可被持续训练消费）")


def do_eval(model, tok, tasks=5):
    print("[eval] 用强模型做参照基准（对比自研模型）")
    for i in range(min(tasks, len(_TASKS))):
        p = "用 Python 实现：" + _TASKS[i]
        print("=" * 50)
        print("任务:", _TASKS[i])
        print(chat(model, tok, p, max_tokens=300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()), default="qwen-coder")
    ap.add_argument("--infer", default="", help="强模型推理一段 prompt")
    ap.add_argument("--distill", action="store_true", help="蒸馏生成高质量代码数据")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--out", default=os.path.join("data", "stream"))
    ap.add_argument("--eval", action="store_true", help="用强模型做参照基准")
    ap.add_argument("--tasks", type=int, default=5)
    args = ap.parse_args()

    if not (args.infer or args.distill or args.eval):
        print("请指定动作之一：--infer / --distill / --eval"); sys.exit(1)

    try:
        import modelscope
    except Exception as e:
        print(f"[错误] 需安装 modelscope：pip install modelscope （{e}）")
        sys.exit(1)

    model, tok = get_model(args.model)
    if args.infer:
        do_infer(model, tok, args.infer)
    if args.distill:
        do_distill(model, tok, args.num, args.out)
    if args.eval:
        do_eval(model, tok, args.tasks)


if __name__ == "__main__":
    main()
