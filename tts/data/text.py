"""English transcript normalization.

Deliberately light: the slow AR reads raw text with the Qwen3 tokenizer, so
we keep casing and punctuation (they carry prosody) and only fix encoding
noise. Numbers are left as digits; the model learns to read them from data.
"""

import re
import unicodedata

_REPLACEMENTS = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": " - ", "−": "-",
    "…": "...", " ": " ",
}
_ALLOWED = re.compile(r"^[A-Za-z0-9 .,;:!?'\"()\-&%$/+=@#*\[\]]*$")
_SPACES = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")


def normalize_text(text: str) -> str:
    """Normalize a transcript; returns "" if it has no usable content."""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = _SPACES.sub(" ", text).strip()
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text


def is_clean_english(text: str) -> bool:
    """True if ``text`` only uses characters we expect in English transcripts
    and contains at least one letter."""
    return bool(text) and bool(_ALLOWED.match(text)) and any(c.isalpha() for c in text)
