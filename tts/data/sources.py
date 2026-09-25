"""Audio sources for tokenization. Each yields ``(Utterance, waveform [S])``
at the requested sample rate; ``Utterance.duration`` is filled from the audio.
"""

import io
from pathlib import Path
from typing import Any, Iterator

import soundfile as sf
import torch
import torchaudio.functional as AF

from tts.audio import load_audio
from tts.data.records import Utterance

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
Item = tuple[Utterance, torch.Tensor]


def _decode_bytes(data: bytes, sample_rate: int) -> torch.Tensor:
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.T).mean(dim=0)
    return AF.resample(wav, sr, sample_rate) if sr != sample_rate else wav


def local_source(root: str | Path, sample_rate: int, source_name: str = "local") -> Iterator[Item]:
    """Audio files with a same-named .txt transcript, anywhere under ``root``.
    The speaker is the name of the file's parent directory."""
    root = Path(root)
    for f in sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS):
        txt = f.with_suffix(".txt")
        if not txt.exists():
            continue
        wav = load_audio(f, sample_rate)[0]
        rel = f.relative_to(root).with_suffix("").as_posix()
        yield (
            Utterance(
                id=f"{source_name}/{rel}",
                text=txt.read_text().strip(),
                speaker=f"{source_name}/{f.parent.name}",
                source=source_name,
                duration=wav.shape[-1] / sample_rate,
            ),
            wav,
        )


def metadata_source(
    metadata: str | Path,
    sample_rate: int,
    source_name: str = "metadata",
    speaker: str = "speaker0",
    audio_dir: str | Path | None = None,
) -> Iterator[Item]:
    """LJSpeech-style metadata: one ``file|transcript`` line per clip.

    ``file`` is relative to ``audio_dir`` (default: ``<metadata dir>/wavs``);
    a missing extension means ``.wav``. Extra ``|`` columns (e.g. LJSpeech's
    normalized text) are allowed: the last column is used as the transcript.
    All clips get the same speaker.
    """
    metadata = Path(metadata)
    root = Path(audio_dir) if audio_dir else metadata.parent / "wavs"
    for line_no, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 2:
            raise ValueError(f"{metadata}:{line_no}: expected 'file|transcript'")
        name, text = parts[0].strip(), parts[-1].strip()
        path = root / name
        if not path.suffix:
            path = path.with_suffix(".wav")
        wav = load_audio(path, sample_rate)[0]
        yield (
            Utterance(
                id=f"{source_name}/{Path(name).with_suffix('').as_posix()}",
                text=text,
                speaker=f"{source_name}/{speaker}",
                source=source_name,
                duration=wav.shape[-1] / sample_rate,
            ),
            wav,
        )


def _get(row: dict, key: str) -> Any:
    """Nested lookup: "json.text" -> row["json"]["text"]."""
    value: Any = row
    for part in key.split("."):
        value = value[part]
    return value


def _row_audio(audio: Any, sample_rate: int) -> torch.Tensor:
    """Accepts raw bytes, {"bytes", "path"}, or an already decoded
    {"array", "sampling_rate"} (streaming datasets may not expose features)."""
    if isinstance(audio, (bytes, bytearray)):
        return _decode_bytes(bytes(audio), sample_rate)
    if isinstance(audio, dict) and audio.get("array") is not None:
        wav = torch.as_tensor(audio["array"], dtype=torch.float32)
        wav = wav.mean(dim=0) if wav.dim() == 2 else wav
        sr = audio["sampling_rate"]
        return AF.resample(wav, sr, sample_rate) if sr != sample_rate else wav
    if isinstance(audio, dict) and audio.get("bytes") is not None:
        return _decode_bytes(audio["bytes"], sample_rate)
    if isinstance(audio, dict) and audio.get("path"):
        return _decode_bytes(Path(audio["path"]).read_bytes(), sample_rate)
    raise TypeError(f"unsupported audio value: {type(audio).__name__}")


def hf_source(
    dataset: Any,
    sample_rate: int,
    source_name: str,
    audio_column: str = "audio",
    text_column: str = "text",
    speaker_column: str | None = None,
    id_column: str | None = None,
) -> Iterator[Item]:
    """Rows of a Hugging Face ``datasets`` dataset (streaming or not).

    The audio column is read as raw bytes (``Audio(decode=False)``) and decoded
    with soundfile, avoiding a dependency on the library's audio backend.
    Column names support nested keys, e.g. ``json.text`` for WebDataset rows.
    """
    import datasets

    top_audio = audio_column.split(".")[0]
    if isinstance(dataset.features, dict) and isinstance(dataset.features.get(top_audio), datasets.Audio):
        dataset = dataset.cast_column(top_audio, datasets.Audio(decode=False))

    for n, row in enumerate(dataset):
        wav = _row_audio(_get(row, audio_column), sample_rate)
        uid = str(_get(row, id_column)) if id_column else str(n)
        speaker = _get(row, speaker_column) if speaker_column else None
        yield (
            Utterance(
                id=f"{source_name}/{uid}",
                text=str(_get(row, text_column)),
                speaker=f"{source_name}/{speaker}" if speaker is not None else None,
                source=source_name,
                duration=wav.shape[-1] / sample_rate,
            ),
            wav,
        )


# Starting points for the candidate corpora (docs/LICENSES.md). Column names
# are UNVERIFIED (Hugging Face was not reachable when this was written): run
# with --limit 5 first; a KeyError shows the real row layout.
PRESETS: dict[str, dict] = {
    "mls_en": dict(
        path="parler-tts/mls_eng", split="train",
        audio_column="audio", text_column="transcript", speaker_column="speaker_id", id_column="id",
    ),
    "emilia_yodas_en": dict(
        path="amphion/Emilia-Dataset", data_files="Emilia-YODAS/EN/*.tar", split="train",
        audio_column="mp3", text_column="json.text", speaker_column="json.speaker", id_column="json.id",
    ),
    "peoples_speech": dict(
        path="MLCommons/peoples_speech", name="clean", split="train",
        audio_column="audio", text_column="text", id_column="id",
    ),
    "common_voice_en": dict(
        path="mozilla-foundation/common_voice_17_0", name="en", split="train",
        audio_column="audio", text_column="sentence", speaker_column="client_id", id_column="path",
    ),
}
