import json

import pytest
import torch

from tts.data import ShardWriter, Utterance
from tts.train import Trainer, load_config
from tts.train.config import TrainConfig
from tts.train.trainer import latest_checkpoint, lr_lambda

TEXTS = ["hello world", "how are you today", "the cat and the dog", "fine thanks", "hello the cat", "you are fine"]
TINY_FAST = {"dim": 64, "n_layer": 1, "n_head": 4, "n_kv_head": 4, "head_dim": 16, "intermediate_size": 128}


@pytest.fixture
def shards(tmp_path):
    root = tmp_path / "shards"
    g = torch.Generator().manual_seed(0)
    with ShardWriter(root, 8) as w:
        for i in range(24):
            codes = torch.randint(0, 8, (8, 5 + i % 7), generator=g)  # low entropy: learnable
            w.add(Utterance(id=f"t/{i}", text=TEXTS[i % len(TEXTS)], speaker=f"t/spk{i % 3}"), codes)
    return root


def _cfg(run_dir, shards, tiny_base, codec_dir, **kw) -> TrainConfig:
    values = dict(
        run_dir=str(run_dir), data=[str(shards)], base=str(tiny_base), codec=str(codec_dir),
        fast=TINY_FAST, lr=1e-3, warmup_steps=2, max_steps=6, max_tokens_per_batch=400,
        grad_accum=2, precision="fp32", num_workers=0, log_every=1, eval_every=1000,
        save_every=3, keep_checkpoints=5, num_eval_samples=0, val_fraction=0.2, device="cpu",
    )
    values.update(kw)
    return TrainConfig(**values)


def test_lr_schedule():
    cfg = TrainConfig(run_dir="x", data=[], lr=1.0, warmup_steps=10, max_steps=110, min_lr_ratio=0.1)
    assert lr_lambda(0, cfg) == pytest.approx(0.1)
    assert lr_lambda(9, cfg) == pytest.approx(1.0)
    assert lr_lambda(60, cfg) == pytest.approx(0.55)
    assert lr_lambda(110, cfg) == pytest.approx(0.1)
    assert lr_lambda(500, cfg) == pytest.approx(0.1)


def test_load_config_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("run_dir: r\ndata: [a]\nlr: 1.0e-4\n")
    cfg = load_config(path, ["lr=3e-4", "max_steps=5", "fast={dim: 32}"])
    assert cfg.lr == 3e-4 and cfg.max_steps == 5 and cfg.fast == {"dim": 32}
    with pytest.raises(ValueError):
        load_config(path, ["not_a_key=1"])


def test_train_eval_and_resume_is_exact(tmp_path, shards, tiny_base, codec_dir):
    # Uninterrupted: 6 steps.
    full = Trainer(_cfg(tmp_path / "full", shards, tiny_base, codec_dir))
    full.train()

    # Interrupted after step 3 (a checkpoint), then resumed to step 6.
    part_dir = tmp_path / "part"
    Trainer(_cfg(part_dir, shards, tiny_base, codec_dir, max_steps=3)).train()
    assert latest_checkpoint(part_dir).name == "step-0000003"
    resumed = Trainer(_cfg(part_dir, shards, tiny_base, codec_dir))
    assert resumed.step == 3
    resumed.train()

    for (name, a), (_, b) in zip(full.model.named_parameters(), resumed.model.named_parameters()):
        assert torch.allclose(a, b, atol=1e-6), name

    logs = [json.loads(l) for l in (part_dir / "metrics.jsonl").read_text().splitlines()]
    steps = [r["step"] for r in logs if "loss" in r]
    assert steps == [1, 2, 3, 4, 5, 6]
    assert all(r["loss"] == r["loss"] for r in logs if "loss" in r)  # no NaN

    # Evaluation: validation loss and audio samples through the codec.
    resumed.cfg.num_eval_samples = 2
    resumed.cfg.eval_max_frames = 5
    result = resumed.evaluate()
    assert "val_loss" in result
    samples = part_dir / "samples" / "step-0000006"
    assert len(list(samples.glob("*.txt"))) == 2
    assert len(list((part_dir / "samples" / "reference").glob("*.wav"))) == 2


def test_training_reduces_loss(tmp_path, shards, tiny_base, codec_dir):
    trainer = Trainer(_cfg(tmp_path / "run", shards, tiny_base, codec_dir, max_steps=40, save_every=1000, log_every=1))
    trainer.train()
    losses = [json.loads(l)["loss"] for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines() if '"loss"' in l]
    assert losses[-1] < losses[0] * 0.8, (losses[0], losses[-1])


def test_finetune_from_checkpoint(tmp_path, shards, tiny_base, codec_dir):
    Trainer(_cfg(tmp_path / "pre", shards, tiny_base, codec_dir, max_steps=2, save_every=2)).train()
    ckpt = latest_checkpoint(tmp_path / "pre")

    ft = Trainer(_cfg(tmp_path / "ft", shards, "/nonexistent/base", codec_dir, init_checkpoint=str(ckpt)))
    assert ft.step == 0  # fresh schedule, not a resume
    from tts.model import DualAR

    pre = DualAR.from_pretrained(ckpt)
    for (name, a), (_, b) in zip(pre.named_parameters(), ft.model.named_parameters()):
        assert torch.equal(a, b), name
    ft.train()
    assert latest_checkpoint(tmp_path / "ft").name == "step-0000006"


def test_shipped_configs_load():
    from pathlib import Path

    for path in Path("configs").glob("*.yaml"):
        load_config(path)
