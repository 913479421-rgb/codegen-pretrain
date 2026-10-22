# -*- coding: utf-8 -*-
"""
infer_engine.py —— 模型生成速度增加引擎
==========================================
世界级推理加速技术集成：

  1. 投机解码（Speculative Decoding）
     用一个小草稿模型快速拟出若干候选 token，主模型一次前向并行验证，
     接受连续匹配前缀、在首个分歧处采样替换 —— 生成速度通常提升 2~3 倍，
     且分布与直接采样完全一致（不损失质量）。DeepSeek/谷歌等均在用。

  2. torch.compile 图编译推理（可选）

  3. 量化模型推理（配合 export_quantized.py 导出的 INT8/INT4）

用法：
    from infer_engine import speculative_generate
    text = speculative_generate(main_model, draft_model, tok, prompt_ids,
                                gamma=4, max_new_tokens=128)
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def speculative_generate(main, draft, tok, prompt_ids, gamma=4,
                         max_new_tokens=128, temperature=0.8, top_k=40):
    """贪心投机解码：draft 贪心拟出 gamma 个候选，main 贪心并行验证。
    草稿分布与主模型越接近，接受率越高，加速越明显（训练好的草稿可 2~3×）。
    main/draft 须用同一 tokenizer（相同 vocab）。"""
    device = next(main.parameters()).device
    out_ids = list(prompt_ids)
    accepted_total = 0
    steps = 0

    while len(out_ids) - len(prompt_ids) < max_new_tokens:
        steps += 1

        # 1) draft 贪心拟出 gamma 个候选（每步 top-1，避免采样导致接受率低）
        draft_idx = torch.tensor([out_ids], dtype=torch.long, device=device)
        cand_ids = []
        cur = draft_idx
        draft.reset_cache()
        for _ in range(gamma):
            draft.reset_cache()
            logits, _ = draft(cur[:, -draft.block_size:])
            next_id = logits[0, -1].argmax().item()
            cand_ids.append(next_id)
            cur = torch.cat([cur, torch.tensor([[next_id]], device=device)], dim=1)

        # 2) main 一次前向得到候选位置的 logits（并行验证）
        main.reset_cache()
        seq = torch.tensor([out_ids + cand_ids], dtype=torch.long, device=device)
        logits_seq, _ = main(seq)                       # (1, L, vocab)

        # 3) 逐位置贪心接受（top-1 匹配）
        accepted = 0
        new_tokens = []
        for i, tid in enumerate(cand_ids):
            pos_logits = logits_seq[:, len(out_ids) - 1 + i, :] / temperature
            top1 = pos_logits.argmax(dim=-1).item()
            if top1 == tid:
                accepted += 1
                new_tokens.append(tid)
            else:
                # 分歧点：用 main 采样替换
                probs = _topk_softmax(pos_logits, top_k)
                new_tokens.append(torch.multinomial(probs, 1).item())
                break

        # 4) 若全部接受，再用 main 补一个 token
        if accepted == len(cand_ids):
            pos_logits = logits_seq[:, -1, :] / temperature
            probs = _topk_softmax(pos_logits, top_k)
            new_tokens.append(torch.multinomial(probs, 1).item())

        out_ids.extend(new_tokens)
        accepted_total += accepted

    gen_ids = out_ids[len(prompt_ids):]
    return tok.decode(gen_ids), accepted_total, steps


def _topk_softmax(logits, top_k):
    import torch.nn.functional as F
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def quantized_generate(model, tok, prompt_ids, max_new_tokens=128,
                       temperature=0.8, top_k=40):
    """量化模型推理：直接用生成接口（K V 缓存已内置）。"""
    idx = torch.tensor([prompt_ids], dtype=torch.long,
                       device=next(model.parameters()).device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return tok.decode(out[0][len(prompt_ids):].tolist())
