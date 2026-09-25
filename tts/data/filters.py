"""Per-utterance filters applied before tokenization (ARCHITECTURE.md §6.1)."""

from dataclasses import dataclass

from tts.data.records import Utterance
from tts.data.text import is_clean_english


@dataclass
class FilterConfig:
    min_duration: float = 1.0  # seconds
    max_duration: float = 30.0
    # Characters per second; catches misaligned transcripts (too few/many words
    # for the audio). Typical English read speech is ~12–16.
    min_chars_per_second: float = 4.0
    max_chars_per_second: float = 30.0
    min_chars: int = 2


def check_utterance(utt: Utterance, cfg: FilterConfig) -> str | None:
    """Return the reason to drop ``utt``, or None to keep it."""
    if not is_clean_english(utt.text):
        return "text_charset"
    if len(utt.text) < cfg.min_chars:
        return "text_too_short"
    if utt.duration < cfg.min_duration:
        return "too_short"
    if utt.duration > cfg.max_duration:
        return "too_long"
    cps = len(utt.text) / utt.duration
    if cps < cfg.min_chars_per_second:
        return "speech_too_slow"
    if cps > cfg.max_chars_per_second:
        return "speech_too_fast"
    return None
