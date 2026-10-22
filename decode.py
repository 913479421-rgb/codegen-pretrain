# -*- coding: utf-8 -*-
"""
decode.py —— 世界级解码引擎（生成质量）
========================================
在朴素采样之上提供高质量解码策略：

  1. beam_search：束搜索，多路径保留最优，确定性生成更稳
  2. self_consistency：自一致性（多次采样 + 多数投票），抑制随机幻觉
  3. min-p / typical：低配采样策略（模型层已支持）

配合 anti_hallucination 的校验闸门使用效果最佳。
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def beam_search(model, tok, prompt_ids, max_new_tokens=80, beam_width=3,
                temperature=0.8, top_k=40, repeat_penalty=1.0):
    """束搜索解码。返回 (best_text, avg_logprob)。"""
    device = next(model.parameters()).device
    vocab = model.cfg.vocab_size
    # beam: (seq_ids, logprob_sum, token_count)
    start = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    beams = [(start, 0.0, 0)]
    for _ in range(max_new_tokens):
        new_beams = []
        for seq, lp_sum, cnt in beams:
            model.reset_cache()  # 束搜索全序列重算，禁用 KV 缓存累积
            logits, _ = model(seq[:, -model.block_size:])
            logits = logits[:, -1, :] / temperature
            if repeat_penalty != 1.0:
                for t in seq[0].tolist():
                    logits[0, t] /= repeat_penalty
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            logp = F.log_softmax(logits, dim=-1)[0]
            # 取 top beam_width 个候选
            top_lp, top_ids = logp.topk(beam_width)
            for l, tid in zip(top_lp, top_ids):
                new_seq = torch.cat([seq, tid.unsqueeze(0).unsqueeze(0)], dim=1)
                new_beams.append((new_seq, lp_sum + l.item(), cnt + 1))
        new_beams.sort(key=lambda x: x[1] / max(1, x[2]), reverse=True)
        beams = new_beams[:beam_width]
        # 若所有候选都结束在 <|endoftext|> 等可停止，简化：继续到 max
    best_seq, lp, cnt = beams[0]
    gen_ids = best_seq[0][len(prompt_ids):].tolist()
    text = tok.decode(gen_ids)
    avg_lp = lp / max(1, cnt)
    return text, avg_lp


def self_consistency(model, tok, prompt_ids, n_samples=5, max_tokens=80,
                     gate=True, temperature=0.8, device=None):
    """自一致性解码（委托给 anti_hallucination）。"""
    from anti_hallucination import self_consistency as _sc
    return _sc(model, tok, prompt_ids, n_samples=n_samples, max_tokens=max_tokens,
               gate=gate, temperature=temperature, device=device)
