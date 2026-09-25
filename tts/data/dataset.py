"""Training dataset and length-aware batching over a ``ShardedCorpus``."""

import random
from typing import Iterator

import torch
from torch.utils.data import Dataset, Sampler

from tts.data.shards import ShardedCorpus
from tts.text.prompt import Sample, build_training_sample
from tts.text.tokenizer import TTSTokenizer


class TTSDataset(Dataset):
    """Turns corpus entries into training samples.

    With probability ``ref_prob`` (and when the speaker has another clip), a
    different clip of the same speaker (at most ``max_ref_frames`` long, never
    cut, so it matches its transcript) is added as the voice reference, which
    teaches zero-shot cloning. Otherwise the sample has no reference, which
    teaches unconditioned generation.
    """

    def __init__(
        self,
        corpus: ShardedCorpus,
        tokenizer: TTSTokenizer,
        num_codebooks: int = 8,
        ref_prob: float = 0.5,
        max_ref_frames: int = 250,  # 20 s at 12.5 Hz
        seed: int = 0,
    ):
        self.corpus = corpus
        self.tok = tokenizer
        self.num_codebooks = num_codebooks
        self.ref_prob = ref_prob
        self.max_ref_frames = max_ref_frames
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Changes which references are drawn, so each epoch sees new pairs."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.corpus)

    def _codes(self, i: int) -> torch.Tensor:
        codes = self.corpus.codes(i)
        if codes.shape[0] < self.num_codebooks:
            raise ValueError(f"corpus has {codes.shape[0]} codebooks, need {self.num_codebooks}")
        return codes[: self.num_codebooks]

    def reference_index(self, i: int) -> int | None:
        # Seeded by (seed, epoch, i): deterministic across workers and restarts.
        rng = random.Random(hash((self.seed, self.epoch, i)))
        speaker = self.corpus.entries[i].get("speaker")
        if not speaker or rng.random() >= self.ref_prob:
            return None
        others = [
            j
            for j in self.corpus.by_speaker.get(speaker, [])
            if j != i and self.corpus.frames(j) <= self.max_ref_frames
        ]
        return rng.choice(others) if others else None

    def __getitem__(self, i: int) -> Sample:
        utt = self.corpus.utterance(i)
        ref = self.reference_index(i)
        ref_text = ref_codes = None
        if ref is not None:
            ref_text = self.corpus.utterance(ref).text
            ref_codes = self._codes(ref)
        return build_training_sample(self.tok, utt.text, self._codes(i), ref_text=ref_text, ref_codes=ref_codes)

    def approx_length(self, i: int) -> int:
        """Upper-bound sequence length estimate used for batching."""
        text_tokens = len(self.corpus.entries[i]["text"]) // 3 + 16
        ref = self.max_ref_frames + 64 if self.ref_prob > 0 else 0
        return self.corpus.frames(i) + text_tokens + ref


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Batches of similar length whose padded size stays under ``max_tokens``.

    Indices are shuffled, split into pools of ``pool_size``, sorted by length
    inside each pool, packed greedily, and the batches are shuffled again.
    """

    def __init__(
        self,
        lengths: list[int],
        max_tokens: int,
        shuffle: bool = True,
        pool_size: int = 10_000,
        seed: int = 0,
    ):
        if max(lengths) > max_tokens:
            raise ValueError(f"longest sample ({max(lengths)}) exceeds max_tokens ({max_tokens})")
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.pool_size = pool_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        rng = random.Random(hash((self.seed, self.epoch)))
        order = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(order)
        batches = []
        for start in range(0, len(order), self.pool_size):
            pool = sorted(order[start : start + self.pool_size], key=self.lengths.__getitem__)
            batch: list[int] = []
            longest = 0
            for i in pool:
                longest_if_added = max(longest, self.lengths[i])
                if batch and longest_if_added * (len(batch) + 1) > self.max_tokens:
                    batches.append(batch)
                    batch, longest_if_added = [], self.lengths[i]
                batch.append(i)
                longest = longest_if_added
            if batch:
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())
