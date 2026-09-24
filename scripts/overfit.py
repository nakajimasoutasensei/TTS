"""Milestone M2: overfit the Dual-AR model on a handful of real utterances.

Put ~10 short clips (3–15 s) in one directory, each with a transcript next to
it: ``clip1.wav`` + ``clip1.txt``. Then, on the GB10:

    python scripts/overfit.py data/overfit --out outputs/overfit --steps 300

The script encodes the clips with Mimi, initializes the slow AR from Qwen3,
trains until the clips are memorized, generates each transcript back, and
writes ``<name>.gen.wav`` (model output) next to ``<name>.codec.wav`` (codec
round trip, the best achievable). Passing criterion: the generated files are
intelligible and match their transcript.
"""

import argparse
from pathlib import Path
import time

import torch

from tts.audio import load_audio, save_audio
from tts.codec import MimiCodec
from tts.inference import SamplingConfig, generate
from tts.model import DualAR
from tts.text import TTSTokenizer, build_prompt, build_training_sample, collate

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="directory of audio + .txt transcript pairs")
    parser.add_argument("--out", type=Path, default=Path("outputs/overfit"))
    parser.add_argument("--base", default="Qwen/Qwen3-0.6B", help="Qwen3 checkpoint for the slow AR")
    parser.add_argument("--codec", default="kyutai/mimi")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", action="store_true", help="save the trained model to --out/model")
    args = parser.parse_args()

    clips = sorted(
        f for f in args.data.iterdir() if f.suffix.lower() in AUDIO_EXTS and f.with_suffix(".txt").exists()
    )
    if not clips:
        raise SystemExit(f"no audio files with matching .txt transcripts in {args.data}")

    device = torch.device(args.device)
    codec = MimiCodec.from_pretrained(args.codec, args.num_codebooks, device=device)
    tok = TTSTokenizer.from_pretrained(args.base, codec.codebook_size)

    texts, targets = [], []
    for f in clips:
        text = f.with_suffix(".txt").read_text().strip()
        codes = codec.encode(load_audio(f, codec.sample_rate).unsqueeze(0))[0].cpu()
        texts.append(text)
        targets.append(codes)
        save_audio(args.out / f"{f.stem}.codec.wav", codec.decode(codes.unsqueeze(0)), codec.sample_rate)
        print(f"{f.name}: {codes.shape[1]} frames | {text[:60]}")

    model = DualAR.from_qwen3(args.base, tok, num_codebooks=args.num_codebooks).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")
    batch = collate([build_training_sample(tok, t, c) for t, c in zip(texts, targets)], tok.pad_id)
    batch = {k: v.to(device) for k, v in batch.items()}

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    use_bf16 = device.type == "cuda"
    model.train()
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(**batch)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            print(
                f"step {step}: loss={out['loss'].item():.4f} slow={out['slow_loss'].item():.4f} "
                f"fast={out['fast_loss'].item():.4f} ({time.perf_counter() - t0:.0f}s)"
            )

    model.eval()
    greedy = SamplingConfig(temperature=0, fast_temperature=0, ras_window=0)
    exact = 0
    for f, text, target in zip(clips, texts, targets):
        prompt = build_prompt(tok, args.num_codebooks, text)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            codes = generate(model, prompt, max_frames=target.shape[1] + 50, sampling=greedy)
        match = codes.shape == target.shape and torch.equal(codes.cpu(), target)
        exact += match
        if codes.shape[1]:
            save_audio(args.out / f"{f.stem}.gen.wav", codec.decode(codes.unsqueeze(0)), codec.sample_rate)
        print(f"{f.name}: generated {codes.shape[1]} / {target.shape[1]} frames, exact={match}")
    print(f"exact reproductions: {exact}/{len(clips)} — listen to {args.out}/*.gen.wav")

    if args.save:
        model.save_pretrained(args.out / "model")
        tok.save_pretrained(args.out / "model")


if __name__ == "__main__":
    main()
