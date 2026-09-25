"""Pretraining loop for the Dual-AR model (ARCHITECTURE.md §6.2, stage P1).

Single-device by design (the GB10 has 128 GB of unified memory, so no
FSDP/ZeRO is needed). Features: bf16 autocast with fp32 master weights,
gradient accumulation, warmup + cosine LR, gradient checkpointing, exact
mid-epoch resume, validation loss, and audio samples at each evaluation.

Run directory layout:
    config.yaml          resolved config of the first run
    metrics.jsonl        one JSON object per log/eval event
    checkpoints/step-XXXXXXX/   model.safetensors, config.json, tokenizer, trainer_state.pt
    samples/step-XXXXXXX/       generated eval audio (samples/reference: ground truth)
"""

from functools import partial
import json
import math
from pathlib import Path
import random
import shutil
import signal
import threading
import time
from typing import Iterator
import zlib

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from tts.audio import save_audio
from tts.data import ShardedCorpus, TokenBudgetBatchSampler, TTSDataset
from tts.inference import SamplingConfig, generate
from tts.model import DualAR, FastARConfig
from tts.text import TTSTokenizer, build_prompt, collate
from tts.train.config import TrainConfig

STATE_FILE = "trainer_state.pt"


def lr_lambda(step: int, cfg: TrainConfig) -> float:
    """Multiplier on cfg.lr: linear warmup, then cosine to min_lr_ratio."""
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def split_indices(corpus: ShardedCorpus, fraction: float, max_val: int) -> tuple[list[int], list[int]]:
    """Deterministic train/val split by id hash, stable when corpora grow."""
    train, val = [], []
    threshold = int(fraction * 2**32)
    for i, e in enumerate(corpus.entries):
        is_val = zlib.crc32(e["id"].encode()) < threshold and len(val) < max_val
        (val if is_val else train).append(i)
    return train, val


def latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted((run_dir / "checkpoints").glob("step-*"))
    complete = [c for c in ckpts if (c / STATE_FILE).exists()]
    return complete[-1] if complete else None


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.device)
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp32": None}[cfg.precision]
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

        resume = latest_checkpoint(self.run_dir)
        if resume is not None:
            print(f"resuming from {resume}")
            self.model = DualAR.from_pretrained(resume)
            self.tok = TTSTokenizer.from_pretrained(resume, self.model.config.codebook_size)
        else:
            self.tok = TTSTokenizer.from_pretrained(cfg.base, cfg.codebook_size)
            self.model = DualAR.from_qwen3(
                cfg.base, self.tok, num_codebooks=cfg.num_codebooks, fast=FastARConfig(**cfg.fast)
            )
            config_path = self.run_dir / "config.yaml"
            if not config_path.exists():
                config_path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        self.model.to(self.device)
        if cfg.gradient_checkpointing:
            self.model.slow.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Data.
        corpus = ShardedCorpus(cfg.data)
        self.dataset = TTSDataset(
            corpus, self.tok, cfg.num_codebooks, ref_prob=cfg.ref_prob, max_ref_frames=cfg.max_ref_frames, seed=cfg.seed
        )
        self.train_idx, self.val_idx = split_indices(corpus, cfg.val_fraction, cfg.max_val_utterances)
        if not self.train_idx:
            raise ValueError("training split is empty")
        self.sampler = TokenBudgetBatchSampler(
            [self.dataset.approx_length(i) for i in self.train_idx], cfg.max_tokens_per_batch, seed=cfg.seed
        )
        print(f"data: {len(self.train_idx)} train / {len(self.val_idx)} val utterances")

        # Optimizer: no weight decay on norms, biases, and embedding tables.
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            (no_decay if p.dim() < 2 or "embed" in name else decay).append(p)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            fused=self.device.type == "cuda",
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, partial(lr_lambda, cfg=cfg))

        self.step = 0
        self.epoch = 0
        self.batch_in_epoch = 0  # micro-batches consumed in the current epoch
        if resume is not None:
            self._load_state(resume)
        self._codec = None

    # ---- data -------------------------------------------------------------

    def _loader(self, indices: list[int], batch_sampler=None, batch_size: int = 1) -> DataLoader:
        batching = {"batch_sampler": batch_sampler} if batch_sampler is not None else {"batch_size": batch_size}
        return DataLoader(
            Subset(self.dataset, indices),
            **batching,
            collate_fn=partial(collate, pad_id=self.tok.pad_id),
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def _batches(self) -> Iterator[dict[str, torch.Tensor]]:
        """Endless stream of training micro-batches, tracking epoch position."""
        while True:
            self.dataset.set_epoch(self.epoch)
            self.sampler.set_epoch(self.epoch, start_batch=self.batch_in_epoch)
            for batch in self._loader(self.train_idx, batch_sampler=self.sampler):
                self.batch_in_epoch += 1
                yield {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            self.epoch += 1
            self.batch_in_epoch = 0

    def _autocast(self):
        return torch.autocast(
            self.device.type, dtype=self.autocast_dtype or torch.float32, enabled=self.autocast_dtype is not None
        )

    # ---- loop -------------------------------------------------------------

    def request_stop(self, *_) -> None:
        """Finish the current step, save a checkpoint, and return from train()."""
        self._stop_requested = True

    def train(self) -> None:
        cfg = self.cfg
        self._stop_requested = False
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self.request_stop)
            signal.signal(signal.SIGINT, self.request_stop)
        batches = self._batches()
        self.model.train()
        t0 = time.perf_counter()
        tokens = frames = 0
        while self.step < cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            sums = {"loss": 0.0, "slow_loss": 0.0, "fast_loss": 0.0}
            for _ in range(cfg.grad_accum):
                batch = next(batches)
                with self._autocast():
                    out = self.model(**batch)
                (out["loss"] / cfg.grad_accum).backward()
                for k in sums:
                    sums[k] += out[k].item() / cfg.grad_accum
                tokens += int(batch["tokens"].numel())
                frames += int(((batch["labels"] != -100) & batch["audio_mask"]).sum())
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip).item()
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            if self.step % cfg.log_every == 0 or self.step == 1:
                elapsed = time.perf_counter() - t0
                self._log(
                    {
                        "step": self.step,
                        **{k: round(v, 5) for k, v in sums.items()},
                        "grad_norm": round(grad_norm, 4),
                        "lr": self.scheduler.get_last_lr()[0],
                        "epoch": self.epoch,
                        "tokens_per_s": round(tokens / elapsed),
                        "audio_hours_per_h": round(frames / 12.5 / elapsed, 2),
                    }
                )
                t0, tokens, frames = time.perf_counter(), 0, 0
            if self.step % cfg.eval_every == 0 or self.step == cfg.max_steps:
                self.evaluate()
                self.model.train()
            if self.step % cfg.save_every == 0 or self.step == cfg.max_steps:
                self.save()
            elif self._stop_requested:
                self.save()
            if self._stop_requested:
                self._log({"step": self.step, "event": "stopped"})
                return
        self._log({"step": self.step, "event": "finished"})

    # ---- evaluation -------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        result: dict = {"step": self.step, "event": "eval"}
        if self.val_idx:
            sums = {"loss": 0.0, "slow_loss": 0.0, "fast_loss": 0.0}
            n = 0
            for batch in self._loader(self.val_idx, batch_size=4):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with self._autocast():
                    out = self.model(**batch)
                for k in sums:
                    sums[k] += out[k].item()
                n += 1
            result.update({f"val_{k}": round(v / n, 5) for k, v in sums.items()})
        if self.cfg.num_eval_samples > 0 and self.val_idx:
            self._write_samples()
        self._log(result)
        return result

    def _write_samples(self) -> None:
        from tts.codec import MimiCodec

        if self._codec is None:
            self._codec = MimiCodec.from_pretrained(self.cfg.codec, self.cfg.num_codebooks, device=self.device)
        corpus = self.dataset.corpus
        out_dir = self.run_dir / "samples" / f"step-{self.step:07d}"
        ref_dir = self.run_dir / "samples" / "reference"
        gen = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        for k, i in enumerate(self.val_idx[: self.cfg.num_eval_samples]):
            text = corpus.utterance(i).text
            ref = self.dataset.reference_index(i)
            prompt = build_prompt(
                self.tok,
                self.cfg.num_codebooks,
                text,
                ref_text=corpus.utterance(ref).text if ref is not None else None,
                ref_codes=self.dataset._codes(ref) if ref is not None else None,
            )
            with self._autocast():
                codes = generate(self.model, prompt, max_frames=self.cfg.eval_max_frames, sampling=SamplingConfig(), generator=gen)
            if codes.shape[1]:
                save_audio(out_dir / f"{k}.wav", self._codec.decode(codes.unsqueeze(0)), self._codec.sample_rate)
            (out_dir / f"{k}.txt").parent.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{k}.txt").write_text(text + "\n")
            if not (ref_dir / f"{k}.wav").exists():
                gt = self.dataset._codes(i).unsqueeze(0).to(self.device)
                save_audio(ref_dir / f"{k}.wav", self._codec.decode(gt), self._codec.sample_rate)

    # ---- checkpoints --------------------------------------------------------

    def save(self) -> Path:
        ckpt_dir = self.run_dir / "checkpoints"
        final = ckpt_dir / f"step-{self.step:07d}"
        tmp = ckpt_dir / f".tmp-step-{self.step:07d}"
        shutil.rmtree(tmp, ignore_errors=True)
        self.model.save_pretrained(tmp)
        self.tok.save_pretrained(tmp)
        state = {
            "step": self.step,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python": random.getstate(),
            },
            "config": self.cfg.to_dict(),
        }
        torch.save(state, tmp / STATE_FILE)
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)
        for old in sorted(ckpt_dir.glob("step-*"))[: -self.cfg.keep_checkpoints]:
            shutil.rmtree(old)
        self._log({"step": self.step, "event": "checkpoint", "path": str(final)})
        return final

    def _load_state(self, ckpt: Path) -> None:
        state = torch.load(ckpt / STATE_FILE, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = state["step"]
        # The saved LR came from the old schedule; recompute it from the
        # current config so e.g. raising max_steps on resume takes effect.
        for group in self.optimizer.param_groups:
            group["lr"] = self.cfg.lr * lr_lambda(self.step, self.cfg)
        self.scheduler.base_lrs = [self.cfg.lr] * len(self.optimizer.param_groups)
        self.epoch = state["epoch"]
        self.batch_in_epoch = state["batch_in_epoch"]
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        random.setstate(state["rng"]["python"])

    def _log(self, record: dict) -> None:
        print(json.dumps(record), flush=True)
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
