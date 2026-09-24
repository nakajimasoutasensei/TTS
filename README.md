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

## Overfit test (milestone M2)

Put ~10 short clips with transcripts in one folder (`clip1.wav` + `clip1.txt`, ...), then:

```bash
python scripts/overfit.py data/overfit --out outputs/overfit --steps 300
```

Downloads `Qwen/Qwen3-0.6B` and `kyutai/mimi`, trains the Dual-AR model until
it memorizes the clips, then generates each transcript back. Compare
`*.gen.wav` (model) with `*.codec.wav` (codec round trip): they should sound
the same.

## Code layout

| Path | Contents |
|---|---|
| `tts/codec/` | codec interface + Mimi wrapper |
| `tts/text/` | tokenizer with control/semantic tokens, prompt layout, batching |
| `tts/model/` | Dual-AR model: Qwen3 slow AR + fast AR |
| `tts/inference/` | sampling and autoregressive generation |
| `tts/eval/` | reconstruction metrics |
| `scripts/` | round-trip and overfit runners |
