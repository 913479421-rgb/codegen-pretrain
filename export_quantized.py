#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_quantized.py —— 量化导出引擎（INT8 / INT4）
====================================================
用 PyTorch 内置动态量化（无第三方 torchao 依赖）把训练好的模型导出为：
  - INT8 动态量化：精度几乎无损，速度/显存明显改善
  - INT4 分组量化：进一步压缩（若 torchao 可用则启用，否则提示）

导出后加载量化模型，配合 infer_engine.quantized_generate 快速生成。

用法：
    # 导出 INT8
    python export_quantized.py --checkpoint checkpoints/final.pt --bits 8
    # 导出 INT4（依赖 torchao，可选）
    python export_quantized.py --checkpoint checkpoints/final.pt --bits 4
    # 加载量化模型并生成
    python export_quantized.py --checkpoint checkpoints/final.pt --load --prompt "def add("
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))


def build_model(checkpoint, device):
    import torch
    from config import TrainConfig
    from model import GPT
    ck = torch.load(checkpoint, map_location="cpu")
    cfg = TrainConfig()
    for k, v in ck["config"].items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    model = GPT(cfg)
    src = ck.get("ema") if ck.get("ema") else ck["model"]
    model.load_state_dict(src)
    model.eval()
    return model, cfg, ck


def export_int8(checkpoint, out_path):
    import torch
    from torch.ao.quantization import quantize_dynamic
    model, cfg, ck = build_model(checkpoint, "cpu")
    # 只量化线性层（Linear），保持精度
    qmodel = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    # 保存整个量化模型对象（保留量化模块结构）
    torch.save({"model": qmodel, "config": ck["config"],
                "bits": 8, "quantized": True}, out_path)
    print(f"[量化] INT8 导出完成 -> {out_path}")
    return out_path


def export_int4(checkpoint, out_path):
    try:
        import torchao
    except Exception as e:
        print(f"[跳过] INT4 需 torchao，未安装（{e}）。已改用 INT8 导出。")
        return export_int8(checkpoint, out_path)
    import torch
    from torchao.quantization import quantize_
    from torchao.quantization.quant_api import int4_weight_only
    model, cfg, ck = build_model(checkpoint, "cpu")
    quantize_(model, int4_weight_only())
    torch.save({"model": model.state_dict(), "config": ck["config"],
                "bits": 4, "quantized": True}, out_path)
    print(f"[量化] INT4 导出完成 -> {out_path}")
    return out_path


def load_quantized(path, device):
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ck["model"]            # 已是量化好的模型对象
    model.eval()
    print(f"[量化] 已加载 {ck.get('bits', '?')}bit 量化模型")
    return model, ck.get("config", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/final.pt")
    ap.add_argument("--bits", type=int, choices=[4, 8], default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--load", action="store_true", help="加载量化模型并生成")
    ap.add_argument("--prompt", default="def add(a, b):\n")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    out = args.out or os.path.join("checkpoints", f"quant_{args.bits}bit.pt")

    if args.load:
        model, _ = load_quantized(out, device)
        from tokenizer import BPETokenizer
        tok = BPETokenizer.load(os.path.join("data", "vocab.json"))
        from infer_engine import quantized_generate
        text = quantized_generate(model, tok, tok.encode(args.prompt),
                                  max_new_tokens=args.max_tokens)
        print("[你输入]", args.prompt)
        print("[量化生成]", text)
    else:
        if args.bits == 8:
            export_int8(args.checkpoint, out)
        else:
            export_int4(args.checkpoint, out)


if __name__ == "__main__":
    main()
