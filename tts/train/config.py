"""Training configuration: a YAML file plus ``key=value`` overrides."""

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    run_dir: str
    data: list[str]  # shard directories (mixed)
    base: str = "Qwen/Qwen3-0.6B"  # slow-AR init (ignored when resuming)
    # Fine-tuning: start from a trained Dual-AR checkpoint instead of `base`
    # (fresh optimizer and schedule; ignored when resuming this run).
    init_checkpoint: str | None = None
    codec: str = "kyutai/mimi"  # only used for eval audio
    num_codebooks: int = 8
    codebook_size: int = 2048  # must match the codec used to build the shards
    fast: dict = field(default_factory=dict)  # FastARConfig overrides

    # Optimization. Effective batch = max_tokens_per_batch x grad_accum tokens.
    lr: float = 2e-4
    min_lr_ratio: float = 0.1  # cosine decays to lr * min_lr_ratio
    warmup_steps: int = 2000
    max_steps: int = 200_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_tokens_per_batch: int = 16384
    grad_accum: int = 4
    precision: str = "bf16"  # "bf16" (autocast, fp32 master weights) or "fp32"
    gradient_checkpointing: bool = True

    # Data.
    ref_prob: float = 0.5
    max_ref_frames: int = 250
    val_fraction: float = 0.005
    max_val_utterances: int = 500
    num_workers: int = 4
    seed: int = 0

    # Logging, evaluation, checkpoints.
    log_every: int = 10
    eval_every: int = 2000
    save_every: int = 2000
    keep_checkpoints: int = 3
    num_eval_samples: int = 4
    eval_max_frames: int = 500
    device: str = "cuda"

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> TrainConfig:
    values: dict = yaml.safe_load(Path(path).read_text()) if path else {}
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value, got {item!r}")
        value = yaml.safe_load(raw)
        if isinstance(value, str):  # YAML 1.1 reads "3e-4" as a string
            try:
                value = float(value)
            except ValueError:
                pass
        values[key] = value
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return TrainConfig(**values)
