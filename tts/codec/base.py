"""Codec interface used by the rest of the project.

The Dual-AR model only depends on this interface, so the codec can be swapped
(ARCHITECTURE.md §4.1, option B) without touching model code.

Shape conventions:
    waveform: float tensor [B, 1, S] at ``sample_rate``, roughly in [-1, 1]
    codes:    int64 tensor [B, N, T], N = ``num_codebooks``, T = frames,
              values in [0, ``codebook_size``). Codebook 0 is the semantic one.
"""

from abc import ABC, abstractmethod
import math

import torch


class AudioCodec(ABC):
    sample_rate: int
    frame_rate: float
    num_codebooks: int
    codebook_size: int

    @property
    def hop_length(self) -> int:
        """Waveform samples per codec frame."""
        return round(self.sample_rate / self.frame_rate)

    def num_frames(self, num_samples: int) -> int:
        """Number of frames produced when encoding ``num_samples`` samples."""
        return math.ceil(num_samples / self.hop_length)

    @abstractmethod
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """[B, 1, S] waveform -> [B, N, T] codes."""

    @abstractmethod
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """[B, N, T] codes -> [B, 1, T * hop_length] waveform."""

    def encode_batch(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode variable-length mono waveforms ([S] each) in one call.

        Zero-pads on the right and cuts each result to ``num_frames(S)``.
        This is exact only for causal codecs, where padding after a clip
        cannot change its codes; non-causal codecs must override this.
        """
        if not waveforms:
            return []
        length = max(w.shape[-1] for w in waveforms)
        batch = torch.zeros(len(waveforms), 1, length, dtype=waveforms[0].dtype)
        for i, w in enumerate(waveforms):
            batch[i, 0, : w.shape[-1]] = w
        codes = self.encode(batch)
        return [codes[i, :, : self.num_frames(w.shape[-1])] for i, w in enumerate(waveforms)]

    def check_codes(self, codes: torch.Tensor) -> None:
        if codes.dim() != 3 or codes.shape[1] != self.num_codebooks:
            raise ValueError(
                f"expected codes of shape [B, {self.num_codebooks}, T], "
                f"got {tuple(codes.shape)}"
            )
        if codes.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"codes must be integer, got {codes.dtype}")
        if codes.numel() and (codes.min() < 0 or codes.max() >= self.codebook_size):
            raise ValueError(f"codes must be in [0, {self.codebook_size})")

    @staticmethod
    def check_waveform(waveform: torch.Tensor) -> None:
        if waveform.dim() != 3 or waveform.shape[1] != 1:
            raise ValueError(
                f"expected mono waveform of shape [B, 1, S], got {tuple(waveform.shape)}"
            )
        if not waveform.is_floating_point():
            raise ValueError(f"waveform must be floating point, got {waveform.dtype}")
