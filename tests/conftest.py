import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config

from tts.model import DualAR, DualARConfig, FastARConfig
from tts.text import TTSTokenizer

WORDS = "system user assistant hello world how are you today fine thanks the a and cat dog".split()
NUM_CODEBOOKS = 4
CODEBOOK_SIZE = 64


def make_tokenizer() -> TTSTokenizer:
    """Tiny word-level tokenizer standing in for Qwen3's (no download)."""
    vocab = {w: i for i, w in enumerate(["[UNK]"] + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
    return TTSTokenizer(hf, codebook_size=CODEBOOK_SIZE)


def make_model(tok: TTSTokenizer, dim: int = 64) -> DualAR:
    slow = Qwen3Config(
        vocab_size=len(tok),
        hidden_size=dim,
        intermediate_size=dim * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=dim // 4,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    config = DualARConfig(
        slow=slow.to_dict(),
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        semantic_begin_id=tok.semantic_begin_id,
        im_end_id=tok.im_end_id,
        fast=FastARConfig(dim=dim, n_layer=2, n_head=4, n_kv_head=4, head_dim=dim // 4, intermediate_size=dim * 2),
    )
    return DualAR(config)


@pytest.fixture
def tok() -> TTSTokenizer:
    return make_tokenizer()


@pytest.fixture(scope="session")
def codec_dir(tmp_path_factory):
    """Randomly initialized Mimi saved locally, for scripts that load by path."""
    import torch
    from transformers import MimiConfig, MimiModel

    path = tmp_path_factory.mktemp("mimi")
    torch.manual_seed(0)
    MimiModel(MimiConfig()).save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory):
    """A tiny local "Qwen3" checkpoint + word-level tokenizer, standing in for
    Qwen/Qwen3-0.6B in trainer tests."""
    from transformers import Qwen3ForCausalLM

    path = tmp_path_factory.mktemp("tiny_qwen")
    vocab = {w: i for i, w in enumerate(["[UNK]"] + WORDS)}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]")
    hf.save_pretrained(path)
    config = Qwen3Config(
        vocab_size=len(hf), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
    )
    Qwen3ForCausalLM(config).save_pretrained(path)
    return path
