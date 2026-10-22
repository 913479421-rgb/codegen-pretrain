# -*- coding: utf-8 -*-
"""
model.py —— 一代架构 decoder-only 语言模型（从零实现，无现成模型库）
======================================================================
升级特性：
  - RMSNorm（比 LayerNorm 快且稳）
  - SwiGLU 门控前馈（LLaMA/主流一代模型标配）
  - RoPE 旋转位置编码（可外推更长上下文）
  - GQA 分组查询注意力（减少 KV 显存/带宽）
  - 推理 KV 缓存（生成速度显著提升）
  - 兼容混合精度（调用方用 autocast）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_rope(dim, block_size, theta=10000.0, device="cpu"):
    """预计算 RoPE 的 cos/sin 表。dim 需为偶数。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(block_size).float()
    freqs = torch.outer(t, inv_freq)                # (T, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)         # (T, dim)
    cos_cached = emb.cos().to(device)
    sin_cached = emb.sin().to(device)
    return cos_cached, sin_cached


def apply_rope(x, cos, sin):
    """x: (B, n_head, T, head_dim)。对每个 head 应用旋转。"""
    head_dim = x.size(-1)
    half = head_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # 旋转：x_out = x*cos + rotate(x)*sin
    cos = cos[: x.size(2), :].unsqueeze(0).unsqueeze(0)  # (1,1,T,head_dim)
    sin = sin[: x.size(2), :].unsqueeze(0).unsqueeze(0)
    x1r = -x2
    x2r = x1
    return (x * cos) + (torch.cat([x1r, x2r], dim=-1) * sin)


class CausalSelfAttention(nn.Module):
    """GQA 分组查询注意力 + KV 缓存。"""
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_kv_cache = cfg.use_kv_cache
        self.use_sdpa = cfg.use_sdpa
        self.c_q = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_k = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim)
        self.c_v = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        # 训练时的因果掩码
        self.register_buffer("mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size))
        self.cache_k = None
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, cos=None, sin=None):
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # RoPE 只对 q,k 应用
        if cos is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # KV 缓存：推理时拼接历史
        if self.use_kv_cache and not self.training:
            if self.cache_k is not None:
                k = torch.cat([self.cache_k, k], dim=2)
                v = torch.cat([self.cache_v, v], dim=2)
            self.cache_k, self.cache_v = k, v
        k_len = k.size(2)

        # GQA：把 KV 头扩展到 Q 头数（分组重复）
        n_groups = self.n_head // self.n_kv_head
        k = k.repeat_interleave(n_groups, dim=1)
        v = v.repeat_interleave(n_groups, dim=1)

        # 世界级引擎：FlashAttention 内核（PyTorch SDPA，训练时启用）
        if self.use_sdpa and self.training:
            # is_causal=True 自动生成下三角因果掩码（内部走 FlashAttention 内核）
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            mask = self.mask[:, :, :T, :k_len]
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.c_proj(y))


class SwiGLU(nn.Module):
    """SwiGLU 门控前馈：gate(x) * up(x)。"""
    def __init__(self, dim, dropout):
        super().__init__()
        hidden = 4 * dim
        self.w1 = nn.Linear(dim, hidden, bias=False)  # gate
        self.w2 = nn.Linear(hidden, dim, bias=False)  # out
        self.w3 = nn.Linear(dim, hidden, bias=False)  # up
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        return self.dropout(self.w2(gate * up))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd) if cfg.use_rmsnorm else nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd) if cfg.use_rmsnorm else nn.LayerNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg.n_embd, cfg.dropout) if cfg.use_swiglu \
            else nn.Sequential(
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd),
                nn.Dropout(cfg.dropout))

    def forward(self, x, cos=None, sin=None):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    """一代 decoder-only 因果语言模型。"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.use_rope = cfg.use_rope
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd) if cfg.use_rmsnorm else nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.use_rope:
            # RoPE 作用在单头维度 head_dim 上
            self.cos, self.sin = precompute_rope(cfg.n_embd // cfg.n_head,
                                                 cfg.block_size)
            self.register_buffer("cos_cached", self.cos, persistent=False)
            self.register_buffer("sin_cached", self.sin, persistent=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size
        # 计算 loss 的训练/验证前向不使用 KV 缓存（避免跨 batch 累积）
        if targets is not None:
            self.reset_cache()
        x = self.drop(self.token_emb(idx))
        cos = self.cos_cached if self.use_rope else None
        sin = self.sin_cached if self.use_rope else None
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    def reset_cache(self):
        for blk in self.blocks:
            blk.attn.reset_cache()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40,
                 top_p=0.9, min_p=0.0, typical_p=0.0, repeat_penalty=1.0,
                 return_scores=False):
        """自回归采样（支持 KV 缓存 + min-p/典型采样）。
        返回 idx；若 return_scores=True，返回 (idx, 每个新 token 的对数概率列表)。
        """
        self.reset_cache()
        scores = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1:] if self.cfg.use_kv_cache and idx.size(1) > 1 else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if repeat_penalty != 1.0:
                for t in idx[0].tolist():
                    logits[0, t] /= repeat_penalty

            # top_k
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            # top_p (nucleus)
            if top_p < 1.0:
                sorted_l, sorted_i = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                remove = cum - F.softmax(sorted_l, dim=-1) > top_p
                sorted_l[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(-1, sorted_i, sorted_l)
            # min-p：过滤低于「max_prob * min_p」的 token
            if min_p > 0.0:
                probs_pre = F.softmax(logits, dim=-1)
                maxp = probs_pre.max(dim=-1, keepdim=True).values
                logits[probs_pre < maxp * min_p] = float("-inf")
            # 典型采样（typical sampling）：保留熵接近期望熵的 token
            if typical_p > 0.0:
                probs_t = F.softmax(logits, dim=-1)
                logprobs_t = torch.log_softmax(logits, dim=-1)
                ent = -torch.sum(probs_t * logprobs_t, dim=-1, keepdim=True)
                ent_cond = ent - (-logprobs_t)
                probs_t[torch.abs(ent_cond) > typical_p] = 0.0
                probs_t = probs_t / probs_t.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                logits = torch.log(probs_t.clamp_min(1e-8))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if return_scores:
                logp = torch.log(probs.gather(-1, next_id))
                scores.append(logp.item())
            idx = torch.cat([idx, next_id], dim=1)
        self.reset_cache()
        return (idx, scores) if return_scores else idx

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
