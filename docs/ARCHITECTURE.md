# Architecture Plan

Status: **draft v0.2** — decisions D1–D4 resolved (see §9).

## 1. Goal

An English, expressive TTS model with voice cloning, targeting quality close
to Fish Audio S2 Pro while running on consumer GPUs:

| Target | VRAM (inference, batch 1) |
|---|---|
| Minimum | **6 GB** |
| Recommended | **8 GB** |

Secondary goals: streaming output, inline emotion/style tags (`[whisper]`,
`[laugh]`, ...), multi-speaker turns, zero-shot cloning from a 10–30 s reference.

## 2. Ground rules (licensing)

Fish Speech is used **only as a design reference**. Its code and weights are
under the Fish Audio Research License (non-commercial, broad "derivative work"
definition that includes models trained on its outputs).

- All code in this repo is written from scratch. No files copied from `fish_speech/`.
- No Fish weights are loaded, fine-tuned, or quantized.
- No distillation from S2 Pro outputs (explicitly prohibited by its license).
- Every third-party component (backbone, codec, dataset) must have its license
  recorded in `docs/LICENSES.md` before use.

## 3. System overview

Same overall shape as S2 Pro (codec + Dual-AR), scaled down.

```
text + [tags] + speaker prompt (reference codes)
                 │
      ┌──────────▼──────────┐
      │ Slow AR (time axis) │  0.6B–1.7B, Qwen3-initialized
      │  predicts codebook 0│
      └──────────┬──────────┘
                 │ hidden state h_t (one per frame)
      ┌──────────▼──────────┐
      │ Fast AR (depth axis)│  ~100M, 4 layers
      │ predicts cb 1..N-1  │
      └──────────┬──────────┘
                 │ N codes per frame
      ┌──────────▼──────────┐
      │ Codec decoder       │  12.5 Hz, 8 codebooks, 24 kHz
      └──────────┬──────────┘
                 ▼
               audio
```

Comparison with the reference:

| Component | Fish S2 Pro (reference) | This project |
|---|---|---|
| Slow AR | ~4B (Qwen3-based) | 0.6B or 1.7B (Qwen3 base) |
| Fast AR | ~400M | ~100M |
| Codec frame rate | ~21.5 Hz | ~12.5 Hz |
| Codebooks | 10 (1×4096 + 9×1024) | 8 |
| Sample rate | 44.1 kHz | 24 kHz |
| Min VRAM | 24 GB | 6 GB |

## 4. Components

### 4.1 Audio codec

Requirements: low frame rate, residual VQ, **codebook 0 carries semantic
(content) information** so the slow AR can focus on "what is said" and the fast
AR on "how it sounds".

| Option | Pros | Cons |
|---|---|---|
| **A. Adopt Mimi (Kyutai)** — 24 kHz, 12.5 Hz, RVQ, cb0 distilled from WavLM | Ready now; fits Dual-AR exactly; streaming | Fixed design; license to verify (reported CC-BY 4.0) |
| B. Train own codec | Full control, can tune for target languages | Large separate project (weeks of GPU time) |

Decision (D4): **A — adopt Mimi** for v1. Revisit B only if the codec becomes the quality ceiling.

Token budget at 12.5 Hz: 1 min of audio = 750 frames. A 4096-token context holds
roughly a 30 s reference + text + ~3 min of generated speech.

### 4.2 Slow AR (time axis)

- Decoder-only transformer, initialized from a **Qwen3** base (Apache-2.0, GQA).
  Its tokenizer is multilingual; v1 only needs English, but this keeps the door
  open for more languages later.
- Vocabulary = text tokens + special tokens + codebook-0 tokens
  (`<|semantic:0|>` … `<|semantic:K-1|>`).
- Input embedding per audio frame = text/semantic embedding + sum of the
  embeddings of all N codebooks of that frame (scaled by `1/sqrt(N+1)`), so the
  model sees full acoustic context of previous frames.
- Output: next token logits; during audio generation logits are masked to
  semantic tokens + `<|im_end|>` (constrained decoding).

| Size | Layers | Hidden | Heads (Q/KV) | head_dim | Params |
|---|---|---|---|---|---|
| S (6 GB) | 28 | 1024 | 16 / 8 | 128 | ~0.6B |
| M (8 GB) | 28 | 2048 | 16 / 8 | 128 | ~1.7B |

Decision (D3): **S (0.6B) is the primary training target** on the available
hardware (see §6.3). M is a stretch goal once the pipeline is proven.

### 4.3 Fast AR (depth axis)

- Small transformer that runs **per frame** over the codebook axis
  (sequence length = N = 8).
- Input step 0: projection of slow-AR hidden state `h_t`; step k: embedding of
  code k-1.
- Predicts codebooks 1..N-1 autoregressively; its KV cache is reset every frame.

Config (~100M): 4 layers, dim 1024, 16 heads, SwiGLU FFN 4096, shared
codebook-size output head (2048).

### 4.4 Prompt format

Chat-style, one turn per speaker segment:

```
<|im_start|>system
<|voice|> {reference transcript} {reference codes}<|im_end|>
<|im_start|>user
<|speaker:0|>Hey, how are you? [laugh] Nice to finally meet you.<|im_end|>
<|im_start|>assistant
<|audio_start|>{generated codes ...}<|im_end|>
```

- Tags are plain text inside the user turn (free-form, not a fixed enum).
- `<|speaker:i|>` selects a speaker from a multi-speaker reference.

### 4.5 Sampling

- Slow AR: temperature / top-p / top-k + repetition-aware sampling (resample at
  higher temperature if a semantic token repeats inside a short window).
- Fast AR: temperature / top-p, no constraint.
- Streaming: decode codec every ~0.5–1 s of frames; Mimi's decoder is causal.

## 5. VRAM budget (inference, batch 1, 4k context)

KV cache for Qwen3 S/M: 2 × 28 layers × 8 KV heads × 128 × 2 bytes ≈ 112 KB/token.

| Item | 6 GB profile | 8 GB profile |
|---|---|---|
| Slow AR weights | M int8 ≈ 1.8 GB **or** S bf16 ≈ 1.2 GB | M bf16 ≈ 3.4 GB |
| Fast AR (bf16) | ~0.2 GB | ~0.2 GB |
| Codec | ~0.2 GB | ~0.2 GB |
| KV cache (4k tokens) | ~0.45 GB | ~0.45 GB |
| Activations + CUDA context | ~1.0–1.5 GB | ~1.0–1.5 GB |
| **Total** | **~3.7–4.2 GB** | **~5.3–5.8 GB** |

Headroom is intentional (desktop compositor, browser, other apps share VRAM).
Numbers must be re-measured with `torch.cuda.max_memory_allocated` in M5.

## 6. Training pipeline

### 6.1 Data

Pipeline: collect → VAD segmentation (5–30 s) → ASR transcription → filtering
(DNSMOS/UTMOS, SNR, CER between two ASR passes) → speaker clustering → tag
annotation (emotion/events via audio-LLM or classifier) → codec tokenization →
sharded storage.

English only (D1). Because D2 keeps commercial use possible, **only datasets
whose license allows commercial use are allowed** (no NC / research-only terms).
Candidates (each must be verified and recorded in `docs/LICENSES.md` before use):

| Dataset | Approx. English hours | Reported license | Status |
|---|---|---|---|
| Emilia-YODAS (English part) | tens of thousands | CC-BY 4.0 | candidate |
| MLS English (LibriVox) | ~44k | CC-BY 4.0 | candidate |
| LibriHeavy (LibriVox) | ~50k | audio public domain; check annotations | candidate |
| People's Speech | ~30k | CC-BY / CC-BY-SA mix | candidate, check SA terms |
| Common Voice (English) | ~2k validated | CC0 | candidate |
| Emilia (original, non-YODAS) | — | CC-BY-NC | **excluded** (NC) |
| GigaSpeech | — | non-commercial audio terms | **excluded** |

LibriVox-based sets overlap heavily (MLS, LibriHeavy); deduplicate by source
book/chapter. Audiobook speech is read-style, so conversational/expressive
sources (Emilia-YODAS, People's Speech) matter for expressiveness.

Realistic scale: 30k–60k hours after filtering (vs ~10M for S2). Data quality
and diversity is the **main risk** to the quality goal, not model size.

### 6.2 Stages

| Stage | What | Loss / signal |
|---|---|---|
| P0 | Codec: adopt (or train, if D4 = B) | — |
| P1 | Pretrain slow + fast AR on (text, codes) | CE on cb0 (slow) + CE on cb1..N-1 (fast) |
| P2 | SFT: tags, multi-speaker, cloning prompts, multi-turn | same CE, curated data |
| P3 | Preference / RL alignment (DPO first, GRPO later) | rewards: ASR WER, speaker similarity, UTMOS, tag adherence |
| P4 | Quantization + inference optimization | int8 / int4 weight-only, `torch.compile`, CUDA graphs |

### 6.3 Training hardware: NVIDIA GB10 (rented)

GB10 is the Grace Blackwell "superchip" used in NVIDIA DGX Spark:

- Blackwell GPU + 20-core **Arm (aarch64)** Grace CPU on one package.
- **128 GB unified memory** shared by CPU and GPU, with ~273 GB/s bandwidth
  (much lower than a datacenter GPU's HBM).
- Compute is roughly in the range of a single mid/high-end desktop GPU for BF16
  training; NVIDIA's headline "1 PFLOP" figure is FP4 with sparsity.

What this means for us:

| Aspect | Impact |
|---|---|
| 128 GB memory | Large batches and full fine-tuning of 0.6B/1.7B fit easily; no need for FSDP/ZeRO. |
| Single-GPU compute | Pretraining must stay small: S (0.6B) model, ~50k hours. |
| Low bandwidth | Fine for training (compute-bound at large batch); slow for batch-1 inference, so **not** representative of target GPUs. |
| aarch64 + new GPU arch | Use NVIDIA's NGC PyTorch container (arm64) instead of plain `pip install torch`; some wheels may need building from source. |
| Not a 6 GB card | VRAM benchmarks (M5) must cap memory (`torch.cuda.set_per_process_memory_fraction`) and should be confirmed on a real 6–8 GB GPU. |

Rough compute estimate for pretraining (P1), using FLOPs ≈ 6 × params × tokens:

- 50k hours × 12.5 Hz = 2.25B frames.
- Slow AR S (0.6B): 6 × 0.6B × 2.25B ≈ 8 × 10¹⁸ FLOPs per epoch.
- Fast AR (0.1B, 8 positions per frame): 6 × 0.1B × 18B ≈ 1.1 × 10¹⁹ FLOPs per epoch.
- Total ≈ 2 × 10¹⁹ FLOPs per epoch. At an assumed 40–60 TFLOPS sustained BF16,
  that is **~4–6 days per epoch**, i.e. **~2–3 weeks for 3 epochs**.
- M (1.7B) would be ~3.4 × 10¹⁹ per epoch (~7–10 days), hence M is a stretch goal.

These are estimates; the real throughput is measured in M2 and the plan is
adjusted then. Rental cost should be checked against these durations.

## 7. Evaluation

| Metric | Tool | Purpose |
|---|---|---|
| WER | Whisper-large-v3 | intelligibility |
| Speaker similarity | WavLM-ECAPA cosine | cloning fidelity |
| UTMOS / DNSMOS | open MOS predictors | naturalness proxy |
| RTF, TTFA, peak VRAM | own benchmark script | the 6/8 GB goal |
| Tag adherence | emotion classifier + small human eval | expressiveness |

Public benchmark: Seed-TTS Eval (English test set) for comparison with published
numbers, plus a held-out English test set with expressive and conversational speech.

## 8. Repository layout (planned)

```
TTS/
├── docs/
│   ├── ARCHITECTURE.md      # this file
│   └── LICENSES.md          # every third-party asset + license
├── configs/                 # model / train / data configs (YAML)
├── tts/
│   ├── codec/               # codec wrapper (Mimi adapter) + optional own codec
│   ├── model/
│   │   ├── slow_ar.py       # Qwen3-initialized time-axis model
│   │   ├── fast_ar.py       # depth-axis model
│   │   └── dual_ar.py       # combined model + generate()
│   ├── text/                # prompt format, tag handling, normalization
│   ├── data/                # dataset, sharding, collate
│   ├── train/               # training loops (pretrain, SFT, DPO/GRPO)
│   └── inference/           # sampling, streaming, quantization
├── scripts/                 # data prep, tokenization, benchmarks
└── tests/
```

## 9. Decisions

| ID | Question | Decision | Consequence |
|---|---|---|---|
| D1 | Target languages | **English only** (v1) | English datasets and eval only; Qwen3 tokenizer keeps expansion possible |
| D2 | Commercial or research | **Both** — keep commercial use possible | Only commercially usable licenses for data, codec, backbone; strictest rule wins |
| D3 | Training compute | **1× NVIDIA GB10 (rented)** | Primary model size S (0.6B); ~50k hours; see §6.3 |
| D4 | Codec | **Adopt Mimi** | No codec training in v1; 12.5 Hz, 8 codebooks, 24 kHz |

## 10. Milestones

| ID | Deliverable | Exit criterion |
|---|---|---|
| M0 | Decisions D1–D4, `LICENSES.md` | decisions done; licenses verified for backbone, codec, first datasets |
| M1 | Codec wrapper + round-trip test | encode→decode on test set, measured quality |
| M2 | Dual-AR model code + tiny overfit run | overfits 10 utterances, generates intelligible audio |
| M3 | Data pipeline + first tokenized shards | ≥1k hours tokenized, filters validated |
| M4 | P1 pretraining (S size first) | WER / SIM on eval set tracked, beats baseline |
| M5 | Inference: quantization, streaming, VRAM benchmark | runs in ≤6 GB (S/int8 M) and ≤8 GB (M) |
| M6 | P2 SFT (tags, multi-speaker, cloning) | tag adherence + SIM targets |
| M7 | P3 alignment (DPO → GRPO) | WER/SIM/UTMOS improve without regressions |
