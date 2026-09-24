"""Fast AR: predicts codebooks 1..N-1 of one frame (ARCHITECTURE.md §4.3).

Input sequence for a frame (length N):
    [proj(h), E_0(c_0), E_1(c_1), ..., E_{N-2}(c_{N-2})]
Output at position j (1 <= j <= N-1) predicts c_j with head j-1, so it has
seen the slow-AR state h and codes c_0..c_{j-1}.
"""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from tts.model.layers import Block, RMSNorm, rope_cache


@dataclass
class FastARConfig:
    dim: int = 1024
    n_layer: int = 4
    n_head: int = 16
    n_kv_head: int = 16
    head_dim: int = 64
    intermediate_size: int = 4096
    norm_eps: float = 1e-6
    rope_base: float = 10000.0


class FastAR(nn.Module):
    def __init__(self, config: FastARConfig, input_dim: int, num_codebooks: int, codebook_size: int):
        super().__init__()
        if num_codebooks < 2:
            raise ValueError("fast AR needs at least 2 codebooks")
        c = config
        self.config = c
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.in_proj = nn.Linear(input_dim, c.dim, bias=False)
        # One table per input codebook c_0..c_{N-2}, flattened into one embedding.
        self.embeddings = nn.Embedding((num_codebooks - 1) * codebook_size, c.dim)
        self.layers = nn.ModuleList(
            Block(c.dim, c.n_head, c.n_kv_head, c.head_dim, c.intermediate_size, c.norm_eps)
            for _ in range(c.n_layer)
        )
        self.norm = RMSNorm(c.dim, c.norm_eps)
        # One output head per predicted codebook c_1..c_{N-1}.
        self.heads = nn.Parameter(torch.empty(num_codebooks - 1, codebook_size, c.dim))
        nn.init.normal_(self.heads, std=c.dim**-0.5)
        cos, sin = rope_cache(num_codebooks, c.head_dim, c.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        offsets = torch.arange(num_codebooks - 1) * codebook_size
        self.register_buffer("offsets", offsets, persistent=False)

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        l = x.shape[1]
        cos = self.rope_cos[:l].to(x.dtype)
        sin = self.rope_sin[:l].to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)

    def forward(self, hidden: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced pass.

        hidden: [F, input_dim] slow-AR states, one per frame
        codes:  [F, N] codes of each frame
        returns logits [F, N-1, codebook_size] for c_1..c_{N-1}
        """
        x0 = self.in_proj(hidden).unsqueeze(1)
        xs = self.embeddings(codes[:, :-1] + self.offsets)
        out = self._run(torch.cat([x0, xs], dim=1))[:, 1:]  # [F, N-1, dim]
        return torch.einsum("fnd,nkd->fnk", out, self.heads.to(out.dtype))

    @torch.inference_mode()
    def generate(
        self,
        hidden: torch.Tensor,
        code0: torch.Tensor,
        sample: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """hidden [B, input_dim], code0 [B] -> codes [B, N] (c_0 included).

        ``sample`` maps logits [B, K] to ids [B]. The sequence is at most N
        long, so it is recomputed each step instead of using a KV cache.
        """
        x = self.in_proj(hidden).unsqueeze(1)
        codes = [code0]
        for j in range(1, self.num_codebooks):
            emb = self.embeddings(codes[-1] + self.offsets[j - 1]).unsqueeze(1)
            x = torch.cat([x, emb], dim=1)
            h = self._run(x)[:, -1]
            logits = h @ self.heads[j - 1].to(h.dtype).T
            codes.append(sample(logits))
        return torch.stack(codes, dim=1)
