import pytest
import torch
from transformers import MimiConfig, MimiModel

from tts.audio import load_audio, save_audio
from tts.codec import MimiCodec
from tts.eval.metrics import LogMelDistance, si_sdr


@pytest.fixture(scope="module")
def codec() -> MimiCodec:
    # Randomly initialized full-size Mimi: tests shapes and plumbing offline,
    # without downloading weights. Audio quality is checked by
    # scripts/codec_roundtrip.py with the real checkpoint.
    torch.manual_seed(0)
    return MimiCodec(MimiModel(MimiConfig()), num_codebooks=8)


def test_properties(codec):
    assert codec.sample_rate == 24000
    assert codec.frame_rate == 12.5
    assert codec.hop_length == 1920
    assert codec.codebook_size == 2048
    assert codec.num_frames(24000) == 13
    assert codec.num_frames(1920 * 4) == 4


@pytest.mark.parametrize("num_samples", [1920 * 4, 24000, 24000 + 500])
def test_roundtrip_shapes(codec, num_samples):
    wav = torch.randn(2, 1, num_samples) * 0.1
    codes = codec.encode(wav)
    assert codes.shape == (2, 8, codec.num_frames(num_samples))
    assert codes.dtype == torch.int64
    assert 0 <= codes.min() and codes.max() < codec.codebook_size

    audio = codec.decode(codes)
    assert audio.shape == (2, 1, codes.shape[-1] * codec.hop_length)
    assert audio.dtype == torch.float32


def test_encode_is_deterministic(codec):
    wav = torch.randn(1, 1, 24000) * 0.1
    assert torch.equal(codec.encode(wav), codec.encode(wav))


def test_rejects_bad_inputs(codec):
    with pytest.raises(ValueError):
        codec.encode(torch.randn(1, 24000))  # missing channel dim
    with pytest.raises(ValueError):
        codec.encode(torch.randn(1, 2, 24000))  # stereo
    with pytest.raises(ValueError):
        codec.decode(torch.zeros(1, 4, 10, dtype=torch.long))  # wrong codebook count
    with pytest.raises(ValueError):
        codec.decode(torch.full((1, 8, 10), 2048, dtype=torch.long))  # out of range
    with pytest.raises(ValueError):
        MimiCodec(codec.model, num_codebooks=33)


def test_audio_io_resamples(tmp_path):
    t = torch.arange(16000) / 16000
    save_audio(tmp_path / "a.wav", 0.5 * torch.sin(2 * torch.pi * 440 * t), 16000)
    wav = load_audio(tmp_path / "a.wav", 24000)
    assert wav.shape == (1, 24000)
    assert wav.abs().max() <= 1.0


def test_metrics():
    torch.manual_seed(0)
    ref = torch.randn(2, 1, 24000)
    noisy = ref + 0.1 * torch.randn_like(ref)
    assert (si_sdr(ref, ref) > 60).all()
    assert (si_sdr(ref, noisy) - 20).abs().max() < 1.0  # 10x amplitude ratio ~ 20 dB
    assert (si_sdr(ref, 3.0 * ref) > 60).all()  # scale invariant

    mel = LogMelDistance(24000)
    assert torch.allclose(mel(ref, ref), torch.zeros(2))
    assert (mel(ref, noisy) > 0).all()
