"""Prompt layout (ARCHITECTURE.md §4.4).

    <|im_start|>system\n<|voice|>{ref text}<|audio_start|>{ref frames}<|im_end|>\n   (optional)
    <|im_start|>user\n<|speaker:i|>{text}<|im_end|>\n
    <|im_start|>assistant\n<|audio_start|>{frames}<|im_end|>

Each audio frame occupies one position: its token is the semantic token for
codebook 0, and all N codes of the frame are stored in ``codes`` at the same
position (``audio_mask`` marks these positions). Only the assistant frames and
the final <|im_end|> are trained (``labels``).
"""

from dataclasses import dataclass

import torch

from tts.text.tokenizer import TTSTokenizer

IGNORE_INDEX = -100


@dataclass
class Sample:
    tokens: torch.Tensor  # [L] int64
    codes: torch.Tensor  # [N, L] int64, zeros outside audio positions
    audio_mask: torch.Tensor  # [L] bool
    labels: torch.Tensor  # [L] int64, IGNORE_INDEX where not trained

    def __len__(self) -> int:
        return self.tokens.shape[0]


class _Builder:
    def __init__(self, tok: TTSTokenizer, num_codebooks: int):
        self.tok = tok
        self.n = num_codebooks
        self.tokens: list[int] = []
        self.codes: list[torch.Tensor] = []
        self.audio: list[bool] = []
        self.train: list[bool] = []

    def text(self, ids: list[int], train: bool = False) -> None:
        self.tokens += ids
        self.codes += [torch.zeros(self.n, len(ids), dtype=torch.long)]
        self.audio += [False] * len(ids)
        self.train += [train] * len(ids)

    def frames(self, codes: torch.Tensor, train: bool = False) -> None:
        if codes.dim() != 2 or codes.shape[0] != self.n:
            raise ValueError(f"expected codes [{self.n}, T], got {tuple(codes.shape)}")
        codes = codes.long().cpu()
        self.tokens += (codes[0] + self.tok.semantic_begin_id).tolist()
        self.codes += [codes]
        self.audio += [True] * codes.shape[1]
        self.train += [train] * codes.shape[1]

    def build(self) -> Sample:
        tokens = torch.tensor(self.tokens, dtype=torch.long)
        train = torch.tensor(self.train, dtype=torch.bool)
        return Sample(
            tokens=tokens,
            codes=torch.cat(self.codes, dim=1),
            audio_mask=torch.tensor(self.audio, dtype=torch.bool),
            labels=torch.where(train, tokens, IGNORE_INDEX),
        )


def _prefix(
    b: _Builder,
    text: str,
    speaker: int,
    ref_text: str | None,
    ref_codes: torch.Tensor | None,
) -> None:
    tok = b.tok
    if (ref_text is None) != (ref_codes is None):
        raise ValueError("ref_text and ref_codes must be given together")
    if ref_codes is not None:
        b.text([tok.im_start_id] + tok.encode_text("system\n") + [tok.voice_id])
        b.text(tok.encode_text(ref_text) + [tok.audio_start_id])
        b.frames(ref_codes)
        b.text([tok.im_end_id] + tok.encode_text("\n"))
    b.text([tok.im_start_id] + tok.encode_text("user\n") + [tok.speaker_id(speaker)])
    b.text(tok.encode_text(text) + [tok.im_end_id] + tok.encode_text("\n"))
    b.text([tok.im_start_id] + tok.encode_text("assistant\n") + [tok.audio_start_id])


def build_prompt(
    tok: TTSTokenizer,
    num_codebooks: int,
    text: str,
    speaker: int = 0,
    ref_text: str | None = None,
    ref_codes: torch.Tensor | None = None,
) -> Sample:
    """Prompt for generation; ends right after the assistant <|audio_start|>."""
    b = _Builder(tok, num_codebooks)
    _prefix(b, text, speaker, ref_text, ref_codes)
    return b.build()


def build_training_sample(
    tok: TTSTokenizer,
    text: str,
    codes: torch.Tensor,
    speaker: int = 0,
    ref_text: str | None = None,
    ref_codes: torch.Tensor | None = None,
) -> Sample:
    """Prompt + target frames [N, T] + <|im_end|>, with labels on the target."""
    b = _Builder(tok, codes.shape[0])
    _prefix(b, text, speaker, ref_text, ref_codes)
    b.frames(codes, train=True)
    b.text([tok.im_end_id], train=True)
    return b.build()


def collate(samples: list[Sample], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad samples into a batch. Right padding needs no attention mask
    under causal attention: padded positions come after all real ones."""
    length = max(len(s) for s in samples)
    n = samples[0].codes.shape[0]
    batch = {
        "tokens": torch.full((len(samples), length), pad_id, dtype=torch.long),
        "codes": torch.zeros(len(samples), n, length, dtype=torch.long),
        "audio_mask": torch.zeros(len(samples), length, dtype=torch.bool),
        "labels": torch.full((len(samples), length), IGNORE_INDEX, dtype=torch.long),
    }
    for i, s in enumerate(samples):
        batch["tokens"][i, : len(s)] = s.tokens
        batch["codes"][i, :, : len(s)] = s.codes
        batch["audio_mask"][i, : len(s)] = s.audio_mask
        batch["labels"][i, : len(s)] = s.labels
    return batch
