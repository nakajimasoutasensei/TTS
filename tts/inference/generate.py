"""Autoregressive generation for the Dual-AR model (ARCHITECTURE.md §4.5)."""

from dataclasses import dataclass

import torch
from transformers import DynamicCache

from tts.model.dual_ar import DualAR
from tts.text.prompt import Sample


@dataclass
class SamplingConfig:
    temperature: float = 0.8  # 0 = greedy
    top_p: float = 0.8
    top_k: int = 30
    # Repetition-aware sampling for codebook 0: if the sampled semantic token
    # occurred in the last ``ras_window`` frames, resample with the looser
    # ``ras_*`` settings. 0 disables it.
    ras_window: int = 10
    ras_temperature: float = 1.0
    ras_top_p: float = 0.9
    # Fast AR (codebooks 1..N-1).
    fast_temperature: float = 0.8
    fast_top_p: float = 0.8
    fast_top_k: int = 30


def sample_logits(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits [B, V] -> ids [B]."""
    logits = logits.float()
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if 0 < top_k < logits.shape[-1]:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, idx = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        # Drop a token if the mass before it already exceeds top_p (keeps >= 1).
        drop = probs.cumsum(dim=-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.inference_mode()
def generate(
    model: DualAR,
    prompt: Sample,
    max_frames: int = 1500,
    sampling: SamplingConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate audio codes [N, T] for a prompt from ``build_prompt``.

    Stops at <|im_end|> or after ``max_frames`` frames (1500 = 2 min at 12.5 Hz).
    """
    s = sampling or SamplingConfig()
    cfg = model.config
    device = next(model.parameters()).device

    tokens = prompt.tokens.unsqueeze(0).to(device)
    codes = prompt.codes.unsqueeze(0).to(device)
    audio_mask = prompt.audio_mask.unsqueeze(0).to(device)

    # Constrained decoding: codebook 0 may only emit semantic tokens or <|im_end|>.
    vocab = model.slow.config.vocab_size
    bias = torch.full((vocab,), float("-inf"), device=device)
    bias[cfg.semantic_begin_id : cfg.semantic_begin_id + cfg.codebook_size] = 0
    bias[cfg.im_end_id] = 0

    def fast_sample(logits: torch.Tensor) -> torch.Tensor:
        return sample_logits(logits, s.fast_temperature, s.fast_top_p, s.fast_top_k, generator)

    cache = DynamicCache()
    out = model.slow.model(
        inputs_embeds=model.embed(tokens, codes, audio_mask), past_key_values=cache, use_cache=True
    )
    hidden = out.last_hidden_state[:, -1]
    frames: list[torch.Tensor] = []
    recent: list[int] = []

    for _ in range(max_frames):
        logits = model.slow.lm_head(hidden).float() + bias
        token = sample_logits(logits, s.temperature, s.top_p, s.top_k, generator)
        if s.ras_window > 0 and token.item() != cfg.im_end_id and token.item() in recent:
            token = sample_logits(logits, s.ras_temperature, s.ras_top_p, s.top_k, generator)
        if token.item() == cfg.im_end_id:
            break

        code0 = token - cfg.semantic_begin_id
        frame = model.fast.generate(hidden, code0, fast_sample)  # [1, N]
        frames.append(frame[0])
        if s.ras_window > 0:
            recent = (recent + [token.item()])[-s.ras_window :]

        out = model.slow.model(
            inputs_embeds=model.embed(
                token.view(1, 1),
                frame.view(1, cfg.num_codebooks, 1),
                torch.ones(1, 1, dtype=torch.bool, device=device),
            ),
            past_key_values=cache,
            use_cache=True,
        )
        hidden = out.last_hidden_state[:, -1]

    if not frames:
        return torch.zeros(cfg.num_codebooks, 0, dtype=torch.long, device=device)
    return torch.stack(frames, dim=1)
