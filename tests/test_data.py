import io
import json

import datasets
import numpy as np
import pytest
import soundfile as sf
import torch

from tests.conftest import CODEBOOK_SIZE, NUM_CODEBOOKS
from tts.audio import save_audio
from tts.data import (
    FilterConfig,
    ShardedCorpus,
    ShardWriter,
    TokenBudgetBatchSampler,
    TTSDataset,
    Utterance,
    check_utterance,
    normalize_text,
)
from tts.data.sources import hf_source, local_source
from tts.text import collate
from tts.text.prompt import IGNORE_INDEX


def test_normalize_text():
    assert normalize_text("  “Hello,”  she said… ") == '"Hello," she said...'
    assert normalize_text("It’s fine , ok !") == "It's fine, ok!"
    assert normalize_text("a—b") == "a - b"


@pytest.mark.parametrize(
    "text,duration,reason",
    [
        ("Hello there, how are you today?", 2.5, None),
        ("Hello there", 0.5, "too_short"),
        ("Hello there", 40.0, "too_long"),
        ("Hi", 10.0, "speech_too_slow"),
        ("x" * 200, 2.0, "speech_too_fast"),
        ("你好世界你好", 1.5, "text_charset"),
        ("1234 5678", 2.0, "text_charset"),  # no letters
    ],
)
def test_filters(text, duration, reason):
    utt = Utterance(id="a", text=text, duration=duration)
    assert check_utterance(utt, FilterConfig()) == reason


def _codes(frames, seed=0):
    return torch.randint(0, CODEBOOK_SIZE, (NUM_CODEBOOKS, frames), generator=torch.Generator().manual_seed(seed))


def _write_corpus(root, items, frames_per_shard=25):
    with ShardWriter(root, NUM_CODEBOOKS, frames_per_shard=frames_per_shard) as w:
        for utt, codes in items:
            w.add(utt, codes)


def test_shards_roundtrip_and_resume(tmp_path):
    items = [(Utterance(id=f"s/{i}", text=f"clip {i}", speaker=f"spk{i % 2}"), _codes(10 + i, i)) for i in range(6)]
    _write_corpus(tmp_path, items[:4])
    assert len(list(tmp_path.glob("shard-*.jsonl"))) >= 2  # small shard size forces a split

    # Resume: a new writer knows the old ids and continues numbering.
    w = ShardWriter(tmp_path, NUM_CODEBOOKS, frames_per_shard=25)
    assert w.done_ids == {"s/0", "s/1", "s/2", "s/3"}
    with pytest.raises(ValueError):
        w.add(*items[0])
    for utt, codes in items[4:]:
        w.add(utt, codes)
    w.close()

    corpus = ShardedCorpus(tmp_path)
    assert len(corpus) == 6
    for i, (utt, codes) in enumerate(items):
        assert corpus.utterance(i).id == utt.id
        assert torch.equal(corpus.codes(i), codes)
    assert sorted(corpus.by_speaker) == ["spk0", "spk1"]
    assert not list(tmp_path.glob("*.tmp*"))


def test_dataset_references(tok, tmp_path):
    items = [
        (Utterance(id="a", text="hello world", speaker="x"), _codes(8, 1)),
        (Utterance(id="b", text="how are you", speaker="x"), _codes(6, 2)),
        (Utterance(id="c", text="the cat", speaker="y"), _codes(5, 3)),  # speaker with 1 clip
        (Utterance(id="d", text="the dog", speaker="x"), _codes(40, 4)),  # too long as a reference
    ]
    _write_corpus(tmp_path, items, frames_per_shard=1000)
    corpus = ShardedCorpus(tmp_path)

    ds = TTSDataset(corpus, tok, NUM_CODEBOOKS, ref_prob=1.0, max_ref_frames=10)
    assert ds.reference_index(0) == 1  # only b qualifies for a
    assert ds.reference_index(2) is None  # no other clip for y
    assert ds.reference_index(3) in (0, 1)
    s = ds[0]
    # Reference frames are in the prompt (untrained), target frames are trained.
    trained = s.labels != IGNORE_INDEX
    assert s.audio_mask.sum() == 8 + 6
    assert (s.audio_mask & trained).sum() == 8

    no_ref = TTSDataset(corpus, tok, NUM_CODEBOOKS, ref_prob=0.0)
    assert all(no_ref.reference_index(i) is None for i in range(4))
    assert no_ref[0].audio_mask.sum() == 8
    batch = collate([ds[i] for i in range(4)], tok.pad_id)
    assert batch["tokens"].shape[0] == 4
    for i in range(4):
        assert len(ds[i]) <= ds.approx_length(i)


def test_token_budget_sampler():
    lengths = [int(x) for x in np.random.default_rng(0).integers(10, 200, size=500)]
    sampler = TokenBudgetBatchSampler(lengths, max_tokens=1000, pool_size=100)
    batches = list(sampler)
    assert sorted(i for b in batches for i in b) == list(range(500))
    assert all(max(lengths[i] for i in b) * len(b) <= 1000 for b in batches)
    sampler.set_epoch(1)
    assert list(sampler) != batches  # reshuffled
    with pytest.raises(ValueError):
        TokenBudgetBatchSampler([5000], max_tokens=1000)


def test_local_source(tmp_path):
    save_audio(tmp_path / "alice" / "one.wav", torch.zeros(24000), 24000)
    (tmp_path / "alice" / "one.txt").write_text(" Hello there. \n")
    save_audio(tmp_path / "alice" / "no_transcript.wav", torch.zeros(100), 24000)
    items = list(local_source(tmp_path, 16000, source_name="loc"))
    assert len(items) == 1
    utt, wav = items[0]
    assert utt.id == "loc/alice/one" and utt.speaker == "loc/alice"
    assert utt.text == "Hello there." and utt.duration == pytest.approx(1.0)
    assert wav.shape == (16000,)


def test_metadata_source(tmp_path):
    from tts.data.sources import metadata_source

    save_audio(tmp_path / "wavs" / "a1.wav", torch.zeros(24000), 24000)
    save_audio(tmp_path / "wavs" / "a2.wav", torch.zeros(48000), 24000)
    (tmp_path / "metadata.csv").write_text("a1|Hello there.\na2.wav|raw text|Normalized text.\n\n")
    items = list(metadata_source(tmp_path / "metadata.csv", 24000, source_name="me"))
    assert [(u.id, u.text, u.speaker) for u, _ in items] == [
        ("me/a1", "Hello there.", "me/speaker0"),
        ("me/a2", "Normalized text.", "me/speaker0"),
    ]
    assert items[1][0].duration == pytest.approx(2.0)


def test_hf_source_nested_columns():
    def wav_bytes(n, sr):
        buf = io.BytesIO()
        sf.write(buf, np.zeros(n, dtype="float32"), sr, format="WAV")
        return buf.getvalue()

    ds = datasets.Dataset.from_dict(
        {
            "audio": [{"bytes": wav_bytes(16000, 16000), "path": None}, {"bytes": wav_bytes(8000, 8000), "path": None}],
            "json": [{"text": "hello", "spk": "s1", "id": "u1"}, {"text": "bye", "spk": "s2", "id": "u2"}],
        }
    ).cast_column("audio", datasets.Audio())
    items = list(hf_source(ds, 24000, "hf", text_column="json.text", speaker_column="json.spk", id_column="json.id"))
    assert [u.id for u, _ in items] == ["hf/u1", "hf/u2"]
    assert [u.speaker for u, _ in items] == ["hf/s1", "hf/s2"]
    assert all(w.shape == (24000,) and u.duration == pytest.approx(1.0) for u, w in items)


def test_row_audio_formats():
    from tts.data.sources import _row_audio

    buf = io.BytesIO()
    sf.write(buf, np.zeros(8000, dtype="float32"), 8000, format="WAV")
    for value in (buf.getvalue(), {"bytes": buf.getvalue()}, {"array": np.zeros(8000), "sampling_rate": 8000}):
        assert _row_audio(value, 24000).shape == (24000,)
    with pytest.raises(TypeError):
        _row_audio(123, 24000)


def test_tokenize_script_end_to_end(tmp_path, codec_dir):
    from scripts.tokenize_dataset import main

    raw = tmp_path / "raw"
    t = torch.arange(24000 * 2) / 24000
    for spk in ("alice", "bob"):
        for i in range(3):
            save_audio(raw / spk / f"{i}.wav", 0.3 * torch.sin(2 * torch.pi * (200 + 50 * i) * t), 24000)
            (raw / spk / f"{i}.txt").write_text("Hello there, how are you?")
    (raw / "bob" / "2.txt").write_text("你好")  # dropped by the charset filter

    out = tmp_path / "shards"
    argv = ["local", str(raw), "--out", str(out), "--codec", str(codec_dir), "--buffer", "2", "--batch-seconds", "5"]
    main(argv)
    corpus = ShardedCorpus(out)
    assert len(corpus) == 5
    assert all(corpus.codes(i).shape == (8, 25) for i in range(5))
    main(argv)  # resume: nothing new
    stats = json.loads((out / "stats.json").read_text())
    assert stats["total_utterances"] == 5
    assert stats["runs"][0]["drop_text_charset"] == 1
    assert stats["runs"][1]["skipped_existing"] == 5
