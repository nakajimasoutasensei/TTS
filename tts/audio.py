"""Audio file I/O helpers."""

from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Load a file as mono float32 [1, S] at ``sample_rate``."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [S, C]
    wav = torch.from_numpy(data.T).mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = AF.resample(wav, sr, sample_rate)
    return wav


def save_audio(path: str | Path, wav: torch.Tensor, sample_rate: int) -> None:
    """Save a [1, S] or [S] waveform as 16-bit PCM WAV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().float().cpu().reshape(-1).clamp(-1.0, 1.0)
    sf.write(str(path), wav.numpy(), sample_rate, subtype="PCM_16")
