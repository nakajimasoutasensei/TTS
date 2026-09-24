"""Lightweight reconstruction metrics for codec evaluation.

These are cheap proxies for tracking regressions; perceptual quality is judged
with MOS predictors and listening tests (ARCHITECTURE.md §7).
"""

import torch
import torchaudio.transforms as T


def _align(ref: torch.Tensor, est: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(ref.shape[-1], est.shape[-1])
    return ref[..., :n], est[..., :n]


def si_sdr(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB over the last dim. Higher is better."""
    ref, est = _align(ref.float(), est.float())
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    scale = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    target = scale * ref
    noise = est - target
    return 10 * torch.log10((target.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))


class LogMelDistance:
    """Mean L1 distance between log-mel spectrograms. Lower is better."""

    def __init__(self, sample_rate: int, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 80):
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

    def __call__(self, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        ref, est = _align(ref.float().cpu(), est.float().cpu())
        ref_mel = torch.log(self.mel(ref).clamp(min=1e-5))
        est_mel = torch.log(self.mel(est).clamp(min=1e-5))
        return (ref_mel - est_mel).abs().mean(dim=tuple(range(1, ref_mel.dim())))
