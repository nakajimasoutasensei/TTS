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

## Tokenizing data (milestone M3)

```bash
# Local folder: <speaker>/<clip>.wav + <speaker>/<clip>.txt
python scripts/tokenize_dataset.py local data/raw/mydata --out data/shards/mydata

# Hugging Face preset (tts/data/sources.py); check column names with a small run first
python scripts/tokenize_dataset.py preset mls_en --out data/shards/mls_en --limit 20
```

Clips are normalized, filtered, encoded with Mimi, and written as shards.
Re-running the same command resumes where it stopped. See `stats.json` for
kept/dropped counts. Check each dataset's license in `docs/LICENSES.md`
before tokenizing it.

## Training (milestone M4)

```bash
python scripts/train.py configs/pretrain_s.yaml
python scripts/train.py configs/pretrain_s.yaml max_steps=1000 lr=1e-4   # overrides
```

Logs go to `runs/<name>/metrics.jsonl`. Every `eval_every` steps it records the
validation loss and writes generated samples to `runs/<name>/samples/`, next to
the ground truth in `samples/reference/`. Re-running the same command resumes
from the latest checkpoint.

## Code layout

| Path | Contents |
|---|---|
| `tts/codec/` | codec interface + Mimi wrapper |
| `tts/text/` | tokenizer with control/semantic tokens, prompt layout, batching |
| `tts/model/` | Dual-AR model: Qwen3 slow AR + fast AR |
| `tts/inference/` | sampling and autoregressive generation |
| `tts/data/` | transcript normalization, filters, sources, shards, training dataset + batching |
| `tts/train/` | training config and loop (checkpoints, resume, eval samples) |
| `tts/eval/` | reconstruction metrics |
| `scripts/` | codec round-trip, overfit, tokenization, and training runners |
