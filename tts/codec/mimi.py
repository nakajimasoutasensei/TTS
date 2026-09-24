"""Mimi codec (Kyutai), via the Hugging Face ``transformers`` implementation.

Weights: ``kyutai/mimi`` on Hugging Face, CC-BY 4.0 (see docs/LICENSES.md).
Mimi runs at 24 kHz / 12.5 Hz with up to 32 RVQ codebooks of size 2048;
codebook 0 is distilled from WavLM and carries semantic content. We use the
first 8 codebooks (ARCHITECTURE.md §4.1).
"""

import torch
from transformers import MimiConfig, MimiModel

from tts.codec.base import AudioCodec

DEFAULT_REPO = "kyutai/mimi"
DEFAULT_NUM_CODEBOOKS = 8


class MimiCodec(AudioCodec):
    def __init__(self, model: MimiModel, num_codebooks: int = DEFAULT_NUM_CODEBOOKS):
        config: MimiConfig = model.config
        if not 1 <= num_codebooks <= config.num_quantizers:
            raise ValueError(
                f"num_codebooks must be in [1, {config.num_quantizers}], got {num_codebooks}"
            )
        self.model = model.eval()
        self.sample_rate = config.sampling_rate
        self.frame_rate = config.frame_rate
        self.num_codebooks = num_codebooks
        self.codebook_size = config.codebook_size

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: str = DEFAULT_REPO,
        num_codebooks: int = DEFAULT_NUM_CODEBOOKS,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "MimiCodec":
        model = MimiModel.from_pretrained(repo_or_path, torch_dtype=dtype).to(device)
        return cls(model, num_codebooks)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        self.check_waveform(waveform)
        waveform = waveform.to(device=self.device, dtype=self.dtype)
        codes = self.model.encode(waveform, num_quantizers=self.num_codebooks).audio_codes
        return codes.long()

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        self.check_codes(codes)
        audio = self.model.decode(codes.to(self.device)).audio_values
        return audio.float()
