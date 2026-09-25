import time

import numpy as np
import pytest
import yaml

from tests.test_train import TINY_FAST, _cfg, shards  # noqa: F401  (fixture)
from tts.train import Trainer
from tts.ui.backend import (
    ModelSession,
    SynthesisParams,
    TrainingManager,
    list_checkpoints,
    list_sample_steps,
    read_metrics,
    read_samples,
)


@pytest.fixture
def trained_run(tmp_path, shards, tiny_base, codec_dir):  # noqa: F811
    cfg = _cfg(tmp_path / "runs" / "demo", shards, tiny_base, codec_dir, max_steps=4, save_every=2,
               eval_every=4, num_eval_samples=2, eval_max_frames=5)
    Trainer(cfg).train()
    return tmp_path / "runs" / "demo"


def test_model_session_load_and_synthesize(trained_run, codec_dir, tmp_path):
    ckpts = list_checkpoints(trained_run.parent)
    assert [c.rsplit("/", 1)[-1] for c in ckpts] == ["step-0000004", "step-0000002"]

    session = ModelSession()
    with pytest.raises(RuntimeError):
        session.synthesize("hello", SynthesisParams())
    status = session.load(ckpts[0], str(codec_dir), "cpu")
    assert "step-0000004" in status and session.loaded

    params = SynthesisParams(temperature=0.8, max_seconds=1.0, seed=1)
    sr, audio, info = session.synthesize("hello world", params)
    assert sr == 24000 and audio.dtype == np.float32 and audio.ndim == 1
    assert "frames" in info
    # Same seed, same output.
    assert np.array_equal(session.synthesize("hello world", params)[1], audio)

    # Voice reference path: needs a transcript.
    from tts.audio import save_audio
    import torch

    ref = tmp_path / "ref.wav"
    save_audio(ref, 0.1 * torch.randn(24000), 24000)
    with pytest.raises(ValueError):
        session.synthesize("hello", params, ref_audio=str(ref), ref_text="")
    session.synthesize("hello", params, ref_audio=str(ref), ref_text="how are you")
    with pytest.raises(ValueError):
        session.synthesize("   ", params)


def test_metrics_and_samples(trained_run):
    train_df, val_df, latest = read_metrics(trained_run)
    assert set(train_df.series) == {"loss", "slow_loss", "fast_loss"}
    assert len(train_df) == 3 * 4
    assert list(val_df.step.unique()) == [4]
    assert latest["step"] == 4 and "val_loss" in latest

    # Val metrics persist after later training records.
    with (trained_run / "metrics.jsonl").open("a") as f:
        f.write('{"step": 5, "loss": 1.0, "slow_loss": 0.5, "fast_loss": 0.5}\n')
    latest = read_metrics(trained_run)[2]
    assert latest["step"] == 5 and "val_loss" in latest
    assert list_sample_steps(trained_run) == ["step-0000004"]
    samples = read_samples(trained_run, "step-0000004")
    assert len(samples) == 2 and all(s["reference"] for s in samples)


def test_metrics_empty_run_is_numeric(tmp_path):
    train_df, val_df, latest = read_metrics(tmp_path)
    assert train_df.empty and latest == {}
    assert str(train_df["value"].dtype) == "float64" and str(val_df["step"].dtype) == "int64"


def test_training_manager_start_and_stop(tmp_path, shards, tiny_base, codec_dir):  # noqa: F811
    cfg = _cfg(tmp_path / "runs" / "bg", shards, tiny_base, codec_dir, max_steps=100000, save_every=100000,
               eval_every=100000, log_every=1)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()))
    manager = TrainingManager(tmp_path / "runs")

    with pytest.raises(FileNotFoundError):
        manager.start(config_path, ["data=[/does/not/exist]"])
    with pytest.raises(ValueError):
        manager.start(config_path, ["bogus_key=1"])

    run_dir = manager.start(config_path, [])
    assert manager.is_running(run_dir)
    with pytest.raises(RuntimeError):
        manager.start(config_path, [])  # one run per directory
    deadline = time.time() + 120
    while time.time() < deadline and read_metrics(run_dir)[2].get("step", 0) < 3:
        time.sleep(0.5)
    assert manager.stop(run_dir)
    while time.time() < deadline and manager.is_running(run_dir):
        time.sleep(0.5)
    assert not manager.is_running(run_dir)
    log = manager.log_tail(run_dir)
    assert '"event": "stopped"' in log and '"event": "checkpoint"' in log
    assert list_checkpoints(tmp_path / "runs")  # stop wrote a checkpoint
    assert str(tmp_path / "runs" / "bg") in manager.list_runs()


def test_app_builds(tmp_path):
    from tts.ui.app import build_app

    app = build_app(runs_root=str(tmp_path), configs_root=str(tmp_path))
    assert app is not None
