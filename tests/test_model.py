import torch

from tests.conftest import CODEBOOK_SIZE, NUM_CODEBOOKS, make_model
from tts.inference import SamplingConfig, generate
from tts.inference.generate import sample_logits
from tts.model import DualAR
from tts.text import build_prompt, build_training_sample, collate

GREEDY = SamplingConfig(temperature=0, fast_temperature=0, ras_window=0)


def _batch(tok, n=3, frames=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    texts = ["hello world", "how are you today", "the cat and the dog"][:n]
    targets = [torch.randint(0, CODEBOOK_SIZE, (NUM_CODEBOOKS, frames), generator=g) for _ in texts]
    samples = [build_training_sample(tok, t, c) for t, c in zip(texts, targets)]
    return texts, targets, collate(samples, tok.pad_id)


def test_forward_loss(tok):
    torch.manual_seed(0)
    model = make_model(tok)
    _, _, batch = _batch(tok)
    out = model(**batch)
    assert out["loss"].isfinite()
    # Untrained model: losses near log(vocab) and log(codebook_size).
    assert abs(out["fast_loss"].item() - torch.log(torch.tensor(CODEBOOK_SIZE)).item()) < 1.0
    out["loss"].backward()
    assert model.fast.heads.grad is not None
    assert model.codebook_embeddings.weight.grad is not None


def test_right_padding_does_not_leak(tok):
    """A short sample padded next to a longer one gets the same hidden states
    as when run alone (right padding + causal attention)."""
    torch.manual_seed(0)
    model = make_model(tok).eval()
    _, targets, _ = _batch(tok, n=2)
    short = build_training_sample(tok, "hello world", targets[0][:, :4])
    long = build_training_sample(tok, "how are you today", targets[1])
    assert len(long) > len(short)

    def hidden(batch):
        emb = model.embed(batch["tokens"], batch["codes"], batch["audio_mask"])
        return model.slow.model(inputs_embeds=emb).last_hidden_state

    alone = hidden(collate([short], tok.pad_id))[0]
    padded = hidden(collate([short, long], tok.pad_id))[0, : len(short)]
    assert torch.allclose(alone, padded, atol=1e-5)


def test_generate_shapes_and_limits(tok):
    torch.manual_seed(0)
    model = make_model(tok).eval()
    prompt = build_prompt(tok, NUM_CODEBOOKS, "hello world")
    codes = generate(model, prompt, max_frames=7, sampling=SamplingConfig(), generator=torch.Generator().manual_seed(0))
    assert codes.shape[0] == NUM_CODEBOOKS and codes.shape[1] <= 7
    assert codes.min() >= 0 and codes.max() < CODEBOOK_SIZE


def test_sample_logits():
    logits = torch.tensor([[0.0, 5.0, 1.0, -1.0]])
    assert sample_logits(logits, 0, 1.0, 0).item() == 1
    g = torch.Generator().manual_seed(0)
    picks = {sample_logits(logits, 1.0, 1.0, 1, g).item() for _ in range(20)}
    assert picks == {1}  # top_k=1
    picks = {sample_logits(logits, 1.0, 0.5, 0, g).item() for _ in range(20)}
    assert picks == {1}  # top token alone has > 50% mass


def test_save_and_load(tok, tmp_path):
    torch.manual_seed(0)
    model = make_model(tok).eval()
    _, _, batch = _batch(tok)
    model.save_pretrained(tmp_path)
    loaded = DualAR.from_pretrained(tmp_path).eval()
    assert torch.allclose(model(**batch)["loss"], loaded(**batch)["loss"])


def test_overfit_and_reproduce(tok):
    """M2 core check: after memorizing 3 utterances, greedy generation from
    each prompt reproduces its exact codes and stops with <|im_end|>. This
    catches any mismatch between the training and generation code paths."""
    torch.manual_seed(0)
    model = make_model(tok)
    texts, targets, batch = _batch(tok)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(500):
        loss = model(**batch)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.02, loss.item()

    model.eval()
    for text, target in zip(texts, targets):
        prompt = build_prompt(tok, NUM_CODEBOOKS, text)
        codes = generate(model, prompt, max_frames=50, sampling=GREEDY)
        assert torch.equal(codes, target), text
