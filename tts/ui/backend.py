"""UI-independent logic for the web UI: model sessions for synthesis and
training process management. Kept free of Gradio so it can be tested and
reused (e.g. by an API server later)."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from tts.audio import load_audio
from tts.codec import MimiCodec
from tts.inference import SamplingConfig, generate
from tts.model import DualAR
from tts.text import TTSTokenizer, build_prompt
from tts.train.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


# ---- model loading and synthesis ---------------------------------------------


def list_checkpoints(runs_root: str | Path) -> list[str]:
    """Checkpoint directories under ``runs_root``, newest step first."""
    root = Path(runs_root)
    ckpts = [p for p in root.glob("*/checkpoints/step-*") if (p / "model.safetensors").exists()]
    return [str(p) for p in sorted(ckpts, key=lambda p: (p.parent.parent.name, p.name), reverse=True)]


@dataclass
class SynthesisParams:
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    max_seconds: float = 30.0
    seed: int | None = None


class ModelSession:
    """One loaded checkpoint + codec, ready to synthesize."""

    def __init__(self) -> None:
        self.model: DualAR | None = None
        self.tok: TTSTokenizer | None = None
        self.codec: MimiCodec | None = None
        self.checkpoint: str | None = None
        self.device = torch.device("cpu")

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self, checkpoint: str, codec: str = "kyutai/mimi", device: str = "auto") -> str:
        """Load a checkpoint written by the trainer. Returns a status summary."""
        path = Path(checkpoint)
        if not (path / "model.safetensors").exists():
            raise FileNotFoundError(f"no model.safetensors in {path}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unload()
        t0 = time.perf_counter()
        self.device = torch.device(device)
        model = DualAR.from_pretrained(path, device=self.device).eval()
        if self.device.type == "cuda":
            model = model.to(torch.bfloat16)
        self.tok = TTSTokenizer.from_pretrained(path, model.config.codebook_size)
        self.codec = MimiCodec.from_pretrained(codec, model.config.num_codebooks, device=self.device)
        self.model = model
        self.checkpoint = str(path)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        memory = ""
        if self.device.type == "cuda":
            memory = f" · GPU memory {torch.cuda.memory_allocated(self.device) / 2**30:.2f} GB"
        run = path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name
        return (
            f"**Loaded** {run} / {path.name}\n\n"
            f"{params:.0f}M params · {model.config.num_codebooks} codebooks · "
            f"{self.device}{memory} · loaded in {time.perf_counter() - t0:.1f}s"
        )

    def unload(self) -> None:
        self.model = self.tok = self.codec = None
        self.checkpoint = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def synthesize(
        self,
        text: str,
        params: SynthesisParams,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> tuple[int, np.ndarray, str]:
        """Returns (sample_rate, float32 waveform, info)."""
        if not self.loaded:
            raise RuntimeError("load a model first")
        text = text.strip()
        if not text:
            raise ValueError("enter some text")
        ref_codes = None
        if ref_audio:
            if not (ref_text and ref_text.strip()):
                raise ValueError("the reference audio needs its transcript")
            wav = load_audio(ref_audio, self.codec.sample_rate)
            ref_codes = self.codec.encode(wav.unsqueeze(0))[0].cpu()
        else:
            ref_text = None

        cfg = self.model.config
        prompt = build_prompt(self.tok, cfg.num_codebooks, text, ref_text=ref_text, ref_codes=ref_codes)
        sampling = SamplingConfig(
            temperature=params.temperature, top_p=params.top_p, top_k=params.top_k,
            fast_temperature=params.temperature, fast_top_p=params.top_p, fast_top_k=params.top_k,
        )
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(params.seed))
        max_frames = max(1, int(params.max_seconds * self.codec.frame_rate))

        t0 = time.perf_counter()
        codes = generate(self.model, prompt, max_frames=max_frames, sampling=sampling, generator=generator)
        if codes.shape[1] == 0:
            raise RuntimeError("the model ended immediately without producing audio")
        audio = self.codec.decode(codes.unsqueeze(0))[0, 0].cpu().numpy()
        elapsed = time.perf_counter() - t0
        seconds = audio.shape[-1] / self.codec.sample_rate
        stopped = "hit max length" if codes.shape[1] >= max_frames else "ended naturally"
        info = f"{seconds:.1f}s audio · {codes.shape[1]} frames · {elapsed:.1f}s to generate (RTF {elapsed / seconds:.2f}) · {stopped}"
        return self.codec.sample_rate, audio, info


# ---- training process management -----------------------------------------------


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.exists():  # guard against a recycled pid
        return "train.py" in cmdline.read_bytes().decode(errors="ignore")
    return True


class TrainingManager:
    """Starts and stops ``scripts/train.py`` as a detached process, so training
    keeps running if the UI is closed; state lives in the run directory."""

    PID_FILE = "train.pid"
    LOG_FILE = "train.log"

    def __init__(self, runs_root: str | Path = "runs"):
        self.runs_root = Path(runs_root)

    def list_runs(self) -> list[str]:
        if not self.runs_root.exists():
            return []
        runs = [p for p in self.runs_root.iterdir() if p.is_dir()]
        return [str(p) for p in sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)]

    def pid(self, run_dir: str | Path) -> int | None:
        f = Path(run_dir) / self.PID_FILE
        if not f.exists():
            return None
        pid = int(f.read_text().strip() or 0)
        return pid if pid and _pid_alive(pid) else None

    def is_running(self, run_dir: str | Path) -> bool:
        return self.pid(run_dir) is not None

    def start(self, config_path: str | Path, overrides: list[str]) -> str:
        """Validate the config, then launch training. Returns the run dir."""
        cfg = load_config(config_path, overrides)  # raises on bad keys/values
        run_dir = Path(cfg.run_dir)
        if self.is_running(run_dir):
            raise RuntimeError(f"training is already running in {run_dir}")
        missing = [d for d in cfg.data if not Path(d).exists()]
        if missing:
            raise FileNotFoundError(f"data directories not found: {', '.join(missing)}")
        run_dir.mkdir(parents=True, exist_ok=True)
        log = (run_dir / self.LOG_FILE).open("a")
        log.write(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} {config_path} {' '.join(overrides)}\n")
        log.flush()
        proc = subprocess.Popen(
            [sys.executable, "-u", str(TRAIN_SCRIPT), str(config_path), *overrides],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            start_new_session=True,  # survives the UI process
        )
        log.close()  # the child keeps its own handle
        (run_dir / self.PID_FILE).write_text(str(proc.pid))
        return str(run_dir)

    def stop(self, run_dir: str | Path) -> bool:
        """Ask training to save a checkpoint and exit (SIGTERM)."""
        pid = self.pid(run_dir)
        if pid is None:
            return False
        os.kill(pid, signal.SIGTERM)
        return True

    def log_tail(self, run_dir: str | Path, lines: int = 40) -> str:
        f = Path(run_dir) / self.LOG_FILE
        if not f.exists():
            return ""
        return "\n".join(f.read_text(errors="ignore").splitlines()[-lines:])


def read_metrics(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(train curves, val curves, latest train record) from metrics.jsonl.
    Curves are long-format frames with columns step / value / series."""
    f = Path(run_dir) / "metrics.jsonl"
    train_rows, val_rows, latest, latest_val = [], [], {}, {}
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # partially written last line
                continue
            if "loss" in r:
                latest = r
                for key in ("loss", "slow_loss", "fast_loss"):
                    train_rows.append({"step": r["step"], "value": r[key], "series": key})
            elif r.get("event") == "eval" and "val_loss" in r:
                latest_val = {k: v for k, v in r.items() if k.startswith("val_")}
                for key in ("val_loss", "val_slow_loss", "val_fast_loss"):
                    val_rows.append({"step": r["step"], "value": r[key], "series": key})
    latest = {**latest, **latest_val}
    return _curves(train_rows), _curves(val_rows), latest


def _curves(rows: list[dict]) -> pd.DataFrame:
    # Explicit dtypes: an empty frame must still be numeric, or the plot
    # locks its y axis to categorical labels for later updates.
    df = pd.DataFrame(rows, columns=["step", "value", "series"])
    return df.astype({"step": "int64", "value": "float64", "series": "str"})


def list_sample_steps(run_dir: str | Path) -> list[str]:
    root = Path(run_dir) / "samples"
    return sorted((p.name for p in root.glob("step-*") if p.is_dir()), reverse=True)


def read_samples(run_dir: str | Path, step: str) -> list[dict]:
    """[{text, generated, reference}] for one eval step (paths or None)."""
    root = Path(run_dir) / "samples"
    out = []
    for txt in sorted((root / step).glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        gen = txt.with_suffix(".wav")
        ref = root / "reference" / f"{txt.stem}.wav"
        out.append(
            {
                "text": txt.read_text().strip(),
                "generated": str(gen) if gen.exists() else None,
                "reference": str(ref) if ref.exists() else None,
            }
        )
    return out
