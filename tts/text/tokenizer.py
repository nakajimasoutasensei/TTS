"""Text tokenizer extended with TTS special tokens and semantic audio tokens.

Wraps a Hugging Face tokenizer (Qwen3's in practice) and appends:
  - control tokens: <|voice|>, <|audio_start|>, <|speaker:i|>
    (<|im_start|> / <|im_end|> are reused from Qwen3 if present)
  - one token per codebook-0 entry: <|s:0|> ... <|s:K-1|>, contiguous ids
    starting at ``semantic_begin_id`` (ARCHITECTURE.md §4.2).
"""

from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VOICE = "<|voice|>"
AUDIO_START = "<|audio_start|>"
NUM_SPEAKERS = 8


def speaker_token(i: int) -> str:
    return f"<|speaker:{i}|>"


def semantic_token(i: int) -> str:
    return f"<|s:{i}|>"


CONTROL_TOKENS = [IM_START, IM_END, VOICE, AUDIO_START] + [
    speaker_token(i) for i in range(NUM_SPEAKERS)
]


class TTSTokenizer:
    def __init__(self, hf_tokenizer: PreTrainedTokenizerBase, codebook_size: int):
        self.hf = hf_tokenizer
        self.codebook_size = codebook_size

        vocab = self.hf.get_vocab()
        missing = [t for t in CONTROL_TOKENS if t not in vocab]
        if missing:
            self.hf.add_tokens(missing, special_tokens=True)

        first = semantic_token(0)
        if first not in self.hf.get_vocab():
            self.hf.add_tokens(
                [semantic_token(i) for i in range(codebook_size)], special_tokens=True
            )
        self.semantic_begin_id = self.hf.convert_tokens_to_ids(first)
        last_id = self.hf.convert_tokens_to_ids(semantic_token(codebook_size - 1))
        if last_id != self.semantic_begin_id + codebook_size - 1:
            raise ValueError("semantic tokens must have contiguous ids")

        self.im_start_id = self.hf.convert_tokens_to_ids(IM_START)
        self.im_end_id = self.hf.convert_tokens_to_ids(IM_END)
        self.voice_id = self.hf.convert_tokens_to_ids(VOICE)
        self.audio_start_id = self.hf.convert_tokens_to_ids(AUDIO_START)
        self.pad_id = self.hf.pad_token_id if self.hf.pad_token_id is not None else self.im_end_id

    @classmethod
    def from_pretrained(cls, path: str | Path, codebook_size: int) -> "TTSTokenizer":
        return cls(AutoTokenizer.from_pretrained(str(path)), codebook_size)

    def save_pretrained(self, path: str | Path) -> None:
        self.hf.save_pretrained(str(path))

    def __len__(self) -> int:
        return len(self.hf)

    @property
    def semantic_end_id(self) -> int:
        """Last semantic token id (inclusive)."""
        return self.semantic_begin_id + self.codebook_size - 1

    def speaker_id(self, i: int) -> int:
        if not 0 <= i < NUM_SPEAKERS:
            raise ValueError(f"speaker index must be in [0, {NUM_SPEAKERS})")
        return self.hf.convert_tokens_to_ids(speaker_token(i))

    def encode_text(self, text: str) -> list[int]:
        """Encode plain text. Special-token strings in ``text`` are not
        interpreted, so user input cannot inject control tokens."""
        return self.hf(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
