# Held-out benchmark export validation

Complete public splits exported successfully on 2026-11-14. This records schema and
row-coverage validation, not a scan of the training corpus or a model score. No benchmark
text is committed. The manifests record exact source revisions and hashes of the local
compressed exports. Compression timestamps can change hashes on another export.

| Export | Split | Rows | Text segments |
|---|---|---:|---:|
| hellaswag | validation | 10,042 | 90,378 |
| arc_easy | test | 2,376 | 11,877 |
| arc_challenge | test | 1,172 | 5,859 |
| mmlu | test | 14,042 | 70,210 |
| gsm8k | test | 1,319 | 2,638 |
| humaneval | test | 164 | 656 |
| jcommonsenseqa | validation | 1,119 | 6,714 |

Environment: Python 3.11.18, datasets 5.2.0, huggingface-hub 1.32.0.
All seven configurations also passed three-row streaming smoke exports.
An end-to-end check used GSM8K sample export -> byte packing -> PR #41 decontamination:
a copied held-out question was removed and an unrelated control was retained.

Reproduce each complete export with:

```bash
python -m exealm.data.benchmarks --benchmark NAME --revision SHA --out data/benchmarks/NAME
```

Use names/revisions from manifests.json. Public validation splits are used for HellaSwag
and JCommonsenseQA; other variants/private tests are outside this coverage. HellaSwag
license metadata was unconfirmed and is recorded as null, not inferred. Source links,
licenses and extraction policy are documented in docs/data.md.
