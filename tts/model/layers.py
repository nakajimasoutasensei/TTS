"""Transformer building blocks for the fast AR (the slow AR uses Qwen3 from
``transformers``)."""

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(x.dtype)


def rope_cache(seq_len: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)  # [L, D/2]
    angles = torch.cat([angles, angles], dim=-1)  # [L, D]
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, L, D]; rotate-half convention
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return (x * cos + rotated * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, n_head: int, n_kv_head: int, head_dim: int):
        super().__init__()
        self.n_head, self.n_kv_head, self.head_dim = n_head, n_kv_head, head_dim
        self.q = nn.Linear(dim, n_head * head_dim, bias=False)
        self.k = nn.Linear(dim, n_kv_head * head_dim, bias=False)
        self.v = nn.Linear(dim, n_kv_head * head_dim, bias=False)
        self.o = nn.Linear(n_head * head_dim, dim, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q = self.q_norm(self.q(x).view(b, l, self.n_head, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k(x).view(b, l, self.n_kv_head, self.head_dim)).transpose(1, 2)
        v = self.v(x).view(b, l, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self.n_head != self.n_kv_head
        )
        return self.o(out.transpose(1, 2).reshape(b, l, -1))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, dim: int, n_head: int, n_kv_head: int, head_dim: int, hidden: int, eps: float):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps)
        self.attn = Attention(dim, n_head, n_kv_head, head_dim)
        self.mlp_norm = RMSNorm(dim, eps)
        self.mlp = MLP(dim, hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))
