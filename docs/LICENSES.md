# Third-Party Licenses

Every external model, codec, dataset, or code dependency used for training or
shipping must be listed here **before** it is used. Because the project keeps
commercial use possible (ARCHITECTURE.md, D2), anything with non-commercial
(NC), research-only, or "no training" terms is not allowed.

Datasets could not be verified yet: Hugging Face is not reachable from the
development container, so dataset cards must be checked from a machine with
access (e.g. the GB10).

Status values: `candidate` (reported license, not yet checked) → `verified`
(license text read from the official source, link recorded) → `excluded`.

| Asset | Type | Use | Reported license | Status | Source / notes |
|---|---|---|---|---|---|
| Qwen3-0.6B / Qwen3-1.7B (base) | LLM weights | slow AR init | Apache-2.0 | verified | github.com/QwenLM/Qwen3 README "License Agreement": all open-weight models Apache 2.0; re-check the specific HF model card when downloading |
| Mimi (Kyutai), `kyutai/mimi` | codec weights | audio tokenizer | CC-BY 4.0 | verified | github.com/kyutai-labs/moshi README "License": weights CC-BY 4.0. Attribution required (credit Kyutai / cite Moshi paper) |
| Hugging Face `transformers` | code dependency | Mimi + Qwen3 implementations | Apache-2.0 | verified | used as a library, not copied |
| Emilia-YODAS (English) | dataset | P1/P2 | CC-BY 4.0 | candidate | |
| MLS English | dataset | P1 | CC-BY 4.0 | candidate | LibriVox-based |
| LibriHeavy | dataset | P1 | audio public domain; annotations to check | candidate | LibriVox-based, overlaps MLS |
| People's Speech | dataset | P1 | CC-BY / CC-BY-SA | candidate | check share-alike implications |
| Common Voice (English) | dataset | P1/eval | CC0 | candidate | |
| Emilia (non-YODAS) | dataset | — | CC-BY-NC | excluded | non-commercial |
| GigaSpeech | dataset | — | non-commercial audio terms | excluded | non-commercial |
| Fish Speech code / weights / outputs | reference | design reference only | Fish Audio Research License | excluded | never copied, loaded, or distilled |
