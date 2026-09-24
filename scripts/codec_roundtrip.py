"""Milestone M1: encode -> decode audio files with Mimi and report metrics.

Usage (needs Hugging Face access for the first download of kyutai/mimi):

    python scripts/codec_roundtrip.py path/to/wavs --out outputs/roundtrip \
        --device cuda --num-codebooks 8

Writes reconstructed files to --out and prints per-file SI-SDR, log-mel
distance, and real-time factor, plus the averages.
"""

import argparse
from pathlib import Path
import time

import torch

from tts.audio import load_audio, save_audio
from tts.codec import MimiCodec
from tts.eval.metrics import LogMelDistance, si_sdr

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="audio files or directories")
    parser.add_argument("--out", type=Path, default=Path("outputs/roundtrip"))
    parser.add_argument("--model", default="kyutai/mimi")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    files = []
    for p in args.inputs:
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS) if p.is_dir() else [p]
    if not files:
        raise SystemExit("no audio files found")

    codec = MimiCodec.from_pretrained(args.model, args.num_codebooks, device=args.device)
    mel_dist = LogMelDistance(codec.sample_rate)
    print(f"Mimi: {codec.sample_rate} Hz, {codec.frame_rate} Hz frames, {codec.num_codebooks} codebooks")

    rows = []
    for f in files:
        wav = load_audio(f, codec.sample_rate).unsqueeze(0)  # [1, 1, S]
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        codes = codec.encode(wav)
        recon = codec.decode(codes)[..., : wav.shape[-1]].cpu()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        duration = wav.shape[-1] / codec.sample_rate
        row = (si_sdr(wav, recon).item(), mel_dist(wav, recon).item(), elapsed / duration)
        rows.append(row)
        save_audio(args.out / f"{f.stem}.wav", recon, codec.sample_rate)
        print(f"{f.name}: {duration:.1f}s frames={codes.shape[-1]} "
              f"si_sdr={row[0]:.2f}dB mel_l1={row[1]:.3f} rtf={row[2]:.3f}")

    means = [sum(r[i] for r in rows) / len(rows) for i in range(3)]
    print(f"mean over {len(rows)} files: si_sdr={means[0]:.2f}dB mel_l1={means[1]:.3f} rtf={means[2]:.3f}")


if __name__ == "__main__":
    main()
