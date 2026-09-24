# Third-Party Licenses

Every external model, codec, dataset, or code dependency used for training or
shipping must be listed here **before** it is used. Because the project keeps
commercial use possible (ARCHITECTURE.md, D2), anything with non-commercial
(NC), research-only, or "no training" terms is not allowed.

Status values: `candidate` (reported license, not yet checked) → `verified`
(license text read from the official source, link recorded) → `excluded`.

| Asset | Type | Use | Reported license | Status | Source / notes |
|---|---|---|---|---|---|
| Qwen3-0.6B / Qwen3-1.7B (base) | LLM weights | slow AR init | Apache-2.0 | candidate | Hugging Face `Qwen/` model cards |
| Mimi (Kyutai) | codec weights | audio tokenizer | CC-BY 4.0 (weights), MIT/Apache (code) | candidate | attribution required if CC-BY |
| Emilia-YODAS (English) | dataset | P1/P2 | CC-BY 4.0 | candidate | |
| MLS English | dataset | P1 | CC-BY 4.0 | candidate | LibriVox-based |
| LibriHeavy | dataset | P1 | audio public domain; annotations to check | candidate | LibriVox-based, overlaps MLS |
| People's Speech | dataset | P1 | CC-BY / CC-BY-SA | candidate | check share-alike implications |
| Common Voice (English) | dataset | P1/eval | CC0 | candidate | |
| Emilia (non-YODAS) | dataset | — | CC-BY-NC | excluded | non-commercial |
| GigaSpeech | dataset | — | non-commercial audio terms | excluded | non-commercial |
| Fish Speech code / weights / outputs | reference | design reference only | Fish Audio Research License | excluded | never copied, loaded, or distilled |
