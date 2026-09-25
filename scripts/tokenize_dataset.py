"""Milestone M3: normalize, filter, and tokenize a corpus into shards.

Examples:

    # local folder: speaker_dir/clip.wav + speaker_dir/clip.txt
    python scripts/tokenize_dataset.py local data/raw/mydata --out data/shards/mydata

    # LJSpeech-style: metadata.csv with "file|transcript" lines, audio in wavs/
    python scripts/tokenize_dataset.py metadata data/raw/my_voice/metadata.csv --out data/shards/my_voice

    # Hugging Face preset (see tts/data/sources.py PRESETS); try --limit first
    python scripts/tokenize_dataset.py preset mls_en --out data/shards/mls_en --limit 20

Re-running with the same --out resumes: ids already in the shards are skipped.
Writes stats.json with kept/dropped counts and hours.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Iterator

import torch

from tts.codec import MimiCodec
from tts.data import FilterConfig, ShardWriter, check_utterance, normalize_text
from tts.data.sources import PRESETS, Item, hf_source, local_source, metadata_source


def make_source(args: argparse.Namespace, sample_rate: int) -> Iterator[Item]:
    if args.kind == "local":
        return local_source(args.input, sample_rate, source_name=args.name or Path(args.input).name)
    if args.kind == "metadata":
        return metadata_source(
            args.input, sample_rate, source_name=args.name or Path(args.input).parent.name, audio_dir=args.audio_dir
        )
    import datasets

    preset = dict(PRESETS[args.input])
    load_kwargs = {k: preset.pop(k) for k in ("path", "name", "split", "data_files") if k in preset}
    ds = datasets.load_dataset(**load_kwargs, streaming=True)
    return hf_source(ds, sample_rate, source_name=args.name or args.input, **preset)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=["local", "metadata", "preset"])
    parser.add_argument("input", help="directory (local), metadata file (metadata), or preset name (preset)")
    parser.add_argument("--audio-dir", help="metadata: audio folder (default: <metadata dir>/wavs)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", help="source name used as id prefix (default: input name)")
    parser.add_argument("--codec", default="kyutai/mimi")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-seconds", type=float, default=600.0, help="padded audio per encode call")
    parser.add_argument("--buffer", type=int, default=256, help="clips sorted together to reduce padding")
    parser.add_argument("--min-duration", type=float, default=FilterConfig.min_duration)
    parser.add_argument("--max-duration", type=float, default=FilterConfig.max_duration)
    parser.add_argument("--limit", type=int, help="stop after this many source rows")
    args = parser.parse_args(argv)

    codec = MimiCodec.from_pretrained(args.codec, args.num_codebooks, device=args.device)
    filters = FilterConfig(min_duration=args.min_duration, max_duration=args.max_duration)
    max_batch_samples = int(args.batch_seconds * codec.sample_rate)
    stats: Counter = Counter()
    kept_seconds = 0.0
    t0 = time.perf_counter()

    with ShardWriter(args.out, args.num_codebooks) as writer:
        buffer: list[Item] = []

        def flush_buffer() -> None:
            nonlocal kept_seconds
            buffer.sort(key=lambda item: item[1].shape[-1])
            while buffer:
                # Pop longest first; padded size = longest length x batch size.
                batch = [buffer.pop()]
                longest = batch[0][1].shape[-1]
                while buffer and longest * (len(batch) + 1) <= max_batch_samples:
                    batch.append(buffer.pop())
                codes = codec.encode_batch([wav for _, wav in batch])
                for (utt, _), c in zip(batch, codes):
                    writer.add(utt, c)
                    kept_seconds += utt.duration
                stats["kept"] += len(batch)

        for n, (utt, wav) in enumerate(make_source(args, codec.sample_rate)):
            if args.limit is not None and n >= args.limit:
                break
            stats["seen"] += 1
            if utt.id in writer.done_ids:
                stats["skipped_existing"] += 1
                continue
            utt.text = normalize_text(utt.text)
            reason = check_utterance(utt, filters)
            if reason:
                stats[f"drop_{reason}"] += 1
                continue
            buffer.append((utt, wav))
            if len(buffer) >= args.buffer:
                flush_buffer()
                print(f"seen {stats['seen']} kept {stats['kept']} "
                      f"({kept_seconds / 3600:.2f} h new) {time.perf_counter() - t0:.0f}s", flush=True)
        flush_buffer()

    stats_path = args.out / "stats.json"
    previous = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    runs = previous.get("runs", []) + [{**stats, "new_hours": round(kept_seconds / 3600, 3)}]
    stats_path.write_text(json.dumps({"total_utterances": len(writer.done_ids), "runs": runs}, indent=2))
    print(json.dumps(dict(stats), indent=2))


if __name__ == "__main__":
    main()
