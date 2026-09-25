from dataclasses import asdict, dataclass


@dataclass
class Utterance:
    """Metadata for one training clip. ``id`` must be unique within a corpus;
    prefix it with the source name when mixing corpora."""

    id: str
    text: str
    speaker: str | None = None
    source: str = ""
    duration: float = 0.0  # seconds

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Utterance":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
