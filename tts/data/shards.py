"""Sharded storage of tokenized utterances.

Each shard is a pair:
  shard-00000.npy    int16 [N, total_frames]  codes of all clips, concatenated
  shard-00000.jsonl  one line per clip: Utterance fields + "offset", "frames"

Codebook entries (< 2048) fit in int16, halving storage vs int32. The .npy
files are memory-mapped when reading, so a corpus much larger than RAM works.
Shards are written atomically (tmp file + rename), and a restarted job skips
ids already present, so long tokenization runs can be resumed.
"""

import json
from pathlib import Path

import numpy as np
import torch

from tts.data.records import Utterance


def _shard_paths(root: Path) -> list[Path]:
    return sorted(root.glob("shard-*.jsonl"))


class ShardWriter:
    def __init__(self, root: str | Path, num_codebooks: int, frames_per_shard: int = 2_000_000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.num_codebooks = num_codebooks
        self.frames_per_shard = frames_per_shard
        existing = _shard_paths(self.root)
        self.next_index = int(existing[-1].stem.split("-")[1]) + 1 if existing else 0
        self.done_ids: set[str] = set()
        for p in existing:
            with p.open() as f:
                self.done_ids.update(json.loads(line)["id"] for line in f)
        self._codes: list[np.ndarray] = []
        self._meta: list[dict] = []
        self._frames = 0

    def add(self, utt: Utterance, codes: torch.Tensor) -> None:
        if codes.dim() != 2 or codes.shape[0] != self.num_codebooks:
            raise ValueError(f"expected codes [{self.num_codebooks}, T], got {tuple(codes.shape)}")
        if utt.id in self.done_ids:
            raise ValueError(f"duplicate utterance id {utt.id!r}")
        arr = codes.cpu().numpy().astype(np.int16)
        self._meta.append({**utt.to_dict(), "offset": self._frames, "frames": arr.shape[1]})
        self._codes.append(arr)
        self._frames += arr.shape[1]
        self.done_ids.add(utt.id)
        if self._frames >= self.frames_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self._meta:
            return
        stem = f"shard-{self.next_index:05d}"
        codes = np.concatenate(self._codes, axis=1)
        tmp_npy = self.root / f"{stem}.tmp.npy"
        np.save(tmp_npy, codes)
        tmp_npy.rename(self.root / f"{stem}.npy")
        # The .jsonl is written last: its presence marks the shard complete.
        tmp_jsonl = self.root / f"{stem}.jsonl.tmp"
        tmp_jsonl.write_text("".join(json.dumps(m) + "\n" for m in self._meta))
        tmp_jsonl.rename(self.root / f"{stem}.jsonl")
        self.next_index += 1
        self._codes, self._meta, self._frames = [], [], 0

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ShardedCorpus:
    """Random access to one or more tokenized corpora written by ``ShardWriter``.

    Ids and speakers are prefixed with the source name at tokenization time,
    so corpora can be mixed without collisions.
    """

    def __init__(self, roots: str | Path | list[str | Path]):
        self.roots = [Path(r) for r in (roots if isinstance(roots, list) else [roots])]
        self.entries: list[dict] = []
        self._shard_of: list[int] = []
        self._npy: list[Path] = []
        for root in self.roots:
            for p in _shard_paths(root):
                self._npy.append(p.with_suffix(".npy"))
                with p.open() as f:
                    for line in f:
                        self.entries.append(json.loads(line))
                        self._shard_of.append(len(self._npy) - 1)
        if not self.entries:
            raise ValueError(f"no shards found in {[str(r) for r in self.roots]}")
        ids = [e["id"] for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate utterance ids across corpora")
        self._arrays: dict[int, np.ndarray] = {}
        self.by_speaker: dict[str, list[int]] = {}
        for i, e in enumerate(self.entries):
            if e.get("speaker"):
                self.by_speaker.setdefault(e["speaker"], []).append(i)

    def __len__(self) -> int:
        return len(self.entries)

    def utterance(self, i: int) -> Utterance:
        return Utterance.from_dict(self.entries[i])

    def frames(self, i: int) -> int:
        return self.entries[i]["frames"]

    def codes(self, i: int) -> torch.Tensor:
        """Codes [N, T] as int64."""
        shard = self._shard_of[i]
        if shard not in self._arrays:  # opened lazily so DataLoader workers each mmap
            self._arrays[shard] = np.load(self._npy[shard], mmap_mode="r")
        e = self.entries[i]
        arr = self._arrays[shard][:, e["offset"] : e["offset"] + e["frames"]]
        return torch.from_numpy(arr.astype(np.int64))
