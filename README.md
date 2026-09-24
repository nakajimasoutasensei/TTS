# TTS

Low-requirement expressive English TTS (6 GB minimum / 8 GB recommended VRAM),
using a Dual-AR + RVQ codec design inspired by Fish Audio S2 Pro.
All code and weights are original; Fish Speech is a design reference only.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the architecture plan.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

On the GB10 (DGX Spark, aarch64), use NVIDIA's NGC PyTorch container and run
`pip install -e ".[dev]"` inside it, so the preinstalled CUDA build of torch is kept.

## Codec round-trip (milestone M1)

```bash
python scripts/codec_roundtrip.py path/to/wavs --out outputs/roundtrip
```

Downloads `kyutai/mimi` on first use, encodes each file to 8 codebooks at
12.5 Hz, decodes it back, and reports SI-SDR, log-mel distance, and real-time
factor. Listen to the files in `outputs/roundtrip` to judge quality.

Mimi weights are CC-BY 4.0 by Kyutai; see [docs/LICENSES.md](docs/LICENSES.md).
