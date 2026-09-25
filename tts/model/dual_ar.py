"""Dual-AR model: Qwen3 slow AR over time + fast AR over codebooks
(ARCHITECTURE.md §3–4)."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

from safetensors.torch import load_model, save_model
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from tts.model.fast_ar import FastAR, FastARConfig
from tts.text.prompt import IGNORE_INDEX
from tts.text.tokenizer import TTSTokenizer


@dataclass
class DualARConfig:
    slow: dict  # Qwen3Config.to_dict()
    num_codebooks: int = 8
    codebook_size: int = 2048
    semantic_begin_id: int = 0
    im_end_id: int = 0
    fast: FastARConfig = field(default_factory=FastARConfig)
    fast_loss_weight: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "DualARConfig":
        return cls(**{**d, "fast": FastARConfig(**d["fast"])})


class DualAR(nn.Module):
    def __init__(self, config: DualARConfig, slow: Qwen3ForCausalLM | None = None):
        super().__init__()
        self.config = config
        self.slow = slow if slow is not None else Qwen3ForCausalLM(Qwen3Config.from_dict(config.slow))
        dim = self.slow.config.hidden_size
        n, k = config.num_codebooks, config.codebook_size
        # Embeddings of all N codes of a frame, summed into the slow-AR input.
        self.codebook_embeddings = nn.Embedding(n * k, dim)
        nn.init.normal_(self.codebook_embeddings.weight, std=0.02)
        self.register_buffer("codebook_offsets", torch.arange(n) * k, persistent=False)
        self.fast = FastAR(config.fast, dim, n, k)

    # ---- construction / persistence -------------------------------------

    @classmethod
    def from_qwen3(
        cls,
        base: str | Path,
        tokenizer: TTSTokenizer,
        num_codebooks: int = 8,
        fast: FastARConfig | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "DualAR":
        """Initialize the slow AR from a pretrained Qwen3 checkpoint and extend
        its vocabulary with the tokenizer's control and semantic tokens."""
        slow = Qwen3ForCausalLM.from_pretrained(str(base), dtype=dtype)
        slow.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        config = DualARConfig(
            slow=slow.config.to_dict(),
            num_codebooks=num_codebooks,
            codebook_size=tokenizer.codebook_size,
            semantic_begin_id=tokenizer.semantic_begin_id,
            im_end_id=tokenizer.im_end_id,
            fast=fast or FastARConfig(),
        )
        return cls(config, slow).to(dtype)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.config.slow = self.slow.config.to_dict()
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2, default=str))
        save_model(self, str(path / "model.safetensors"))

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str | torch.device = "cpu") -> "DualAR":
        path = Path(path)
        config = DualARConfig.from_dict(json.loads((path / "config.json").read_text()))
        model = cls(config)
        load_model(model, str(path / "model.safetensors"), device=str(device))
        return model.to(device)

    # ---- forward ----------------------------------------------------------

    @property
    def semantic_end_id(self) -> int:
        return self.config.semantic_begin_id + self.config.codebook_size - 1

    def audio_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Slow-AR logits restricted to what may follow in an audio segment:
        index 0 = <|im_end|>, index 1 + c = semantic token for code c.

        Training and generation both use this (a softmax over K + 1 classes
        instead of the full ~154k vocabulary), so they match exactly, and the
        training loss needs far less memory.
        """
        weight = self.slow.lm_head.weight
        rows = torch.cat(
            [
                weight[self.config.im_end_id : self.config.im_end_id + 1],
                weight[self.config.semantic_begin_id : self.semantic_end_id + 1],
            ]
        )
        return hidden @ rows.T.to(hidden.dtype)

    def audio_index_to_token(self, index: torch.Tensor) -> torch.Tensor:
        """Inverse of the ``audio_logits`` class layout."""
        semantic = index - 1 + self.config.semantic_begin_id
        return torch.where(index == 0, torch.full_like(index, self.config.im_end_id), semantic)

    def token_to_audio_index(self, token: torch.Tensor) -> torch.Tensor:
        is_end = token == self.config.im_end_id
        is_semantic = (token >= self.config.semantic_begin_id) & (token <= self.semantic_end_id)
        if not bool((is_end | is_semantic).all()):
            raise ValueError("trained labels must be semantic tokens or <|im_end|>")
        return torch.where(is_end, torch.zeros_like(token), token - self.config.semantic_begin_id + 1)

    def embed(self, tokens: torch.Tensor, codes: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        """tokens [B, L], codes [B, N, L], audio_mask [B, L] -> [B, L, D]."""
        x = self.slow.get_input_embeddings()(tokens)
        frame = self.codebook_embeddings(codes.transpose(1, 2) + self.codebook_offsets).sum(dim=2)
        scale = 1.0 / math.sqrt(self.config.num_codebooks + 1)
        return torch.where(audio_mask.unsqueeze(-1), (x + frame) * scale, x)

    def forward(
        self,
        tokens: torch.Tensor,
        codes: torch.Tensor,
        audio_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Training loss for a right-padded batch (see ``tts.text.collate``)."""
        hidden = self.slow.model(inputs_embeds=self.embed(tokens, codes, audio_mask)).last_hidden_state

        # Slow AR: next-token loss over the audio classes, where a label exists.
        h_prev = hidden[:, :-1]
        target = labels[:, 1:]
        trained = target != IGNORE_INDEX
        logits = self.audio_logits(h_prev[trained])
        slow_loss = F.cross_entropy(logits.float(), self.token_to_audio_index(target[trained]))

        # Fast AR: for each trained frame, the state that predicted its
        # semantic token plus the frame's codes.
        frames = trained & audio_mask[:, 1:]
        frame_codes = codes[:, :, 1:].transpose(1, 2)[frames]  # [F, N]
        fast_logits = self.fast(h_prev[frames], frame_codes)  # [F, N-1, K]
        fast_loss = F.cross_entropy(
            fast_logits.flatten(0, 1).float(), frame_codes[:, 1:].flatten()
        )

        loss = slow_loss + self.config.fast_loss_weight * fast_loss
        return {"loss": loss, "slow_loss": slow_loss.detach(), "fast_loss": fast_loss.detach()}
