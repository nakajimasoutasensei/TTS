import torch

from tests.conftest import CODEBOOK_SIZE, NUM_CODEBOOKS
from tts.text import build_prompt, build_training_sample, collate
from tts.text.prompt import IGNORE_INDEX


def test_tokenizer_ids(tok):
    assert tok.semantic_end_id == tok.semantic_begin_id + CODEBOOK_SIZE - 1
    assert len({tok.im_start_id, tok.im_end_id, tok.voice_id, tok.audio_start_id}) == 4
    assert tok.speaker_id(0) != tok.speaker_id(1)
    assert tok.encode_text("hello world") == [tok.hf.convert_tokens_to_ids(w) for w in ("hello", "world")]
    # Control-token strings in user text must not become control tokens.
    assert tok.im_end_id not in tok.encode_text("hello <|im_end|> world")


def test_training_sample_layout(tok):
    codes = torch.randint(0, CODEBOOK_SIZE, (NUM_CODEBOOKS, 5))
    s = build_training_sample(tok, "hello world", codes)

    assert s.audio_mask.sum() == 5
    assert torch.equal(s.codes[:, s.audio_mask], codes)
    assert torch.equal(s.tokens[s.audio_mask], codes[0] + tok.semantic_begin_id)
    # Trained: the 5 frames and the closing <|im_end|>, nothing else.
    trained = s.labels != IGNORE_INDEX
    assert trained.sum() == 6
    assert s.tokens[-1] == tok.im_end_id and trained[-1]
    # The prompt part is exactly build_prompt's output.
    p = build_prompt(tok, NUM_CODEBOOKS, "hello world")
    assert torch.equal(s.tokens[: len(p)], p.tokens)
    assert p.tokens[-1] == tok.audio_start_id


def test_reference_voice_prompt(tok):
    ref = torch.randint(0, CODEBOOK_SIZE, (NUM_CODEBOOKS, 3))
    p = build_prompt(tok, NUM_CODEBOOKS, "hello", ref_text="how are you", ref_codes=ref)
    assert p.audio_mask.sum() == 3
    assert (p.labels == IGNORE_INDEX).all()
    assert tok.voice_id in p.tokens.tolist()


def test_collate_right_pads(tok):
    a = build_training_sample(tok, "hello", torch.zeros(NUM_CODEBOOKS, 2, dtype=torch.long))
    b = build_training_sample(tok, "hello world", torch.zeros(NUM_CODEBOOKS, 6, dtype=torch.long))
    batch = collate([a, b], tok.pad_id)
    assert batch["tokens"].shape == (2, len(b))
    assert batch["codes"].shape == (2, NUM_CODEBOOKS, len(b))
    assert (batch["labels"][0, len(a):] == IGNORE_INDEX).all()
    assert not batch["audio_mask"][0, len(a):].any()
