# CPU IVF-PQ datastore comparison

Issue #15. One million independently generated normalized Gaussian keys, dimension 32,
256 independent random queries, k=4, seed 0. No external corpus or model checkpoint.
The exact and approximate paths use identical keys, queries and document exclusions.
Documents contain 100 consecutive chunk IDs. Recall is the fraction of exact top-four
IDs recovered; timing is median wall time for the full 256-query call after one warmup,
three measured repetitions. Each benchmark run sets four PyTorch and FAISS threads.

Environment: macOS 27.0.1 arm64, conda-forge Python 3.11.18, PyTorch 2.14.1,
NumPy 2.4.6, FAISS 1.14.3. These builds share one OpenMP runtime. Measurements are
from one host and one seed; latency varies with system load. The JSON files contain
unrounded values and build settings. Construction time is excluded from search latency.

Both indexes use 256 IVF lists, 8 bits per PQ subvector, and 50,000 sampled training keys.
The search re-scores 16 candidates with float32 keys before returning four.

| PQ subquantizers | Probes | Recall@4 | No exclusion, ms | With exclusion, ms |
|---|---|---|---|---|
| exact (m=8 run) | all | 1.000 | 240.07 | 358.27 |
| 8 | 8 | 0.403 | 5.87 | 45.40 |
| 8 | 32 | 0.516 | 20.91 | 134.76 |
| 8 | 128 | 0.528 | 82.10 | 495.51 |
| exact (m=16 run) | all | 1.000 | 235.86 | 360.87 |
| 16 | 8 | 0.550 | 9.28 | 56.70 |
| 16 | 32 | 0.850 | 36.75 | 189.72 |
| 16 | 128 | 0.942 | 143.45 | 712.22 |


m=8: index build 2.82 s; serialized index 16,067,764 bytes.
m=16: index build 5.30 s; serialized index 24,067,764 bytes.

The m=16, 32-probe setting recovers about 85% of the exact neighbors and is faster
in both cases here. At 128 probes recall rises to about 94%, but grouped per-document
exclusion is slower than exact search. The shorter codes lose substantially more recall.
These results do not establish a winning configuration for language-model retrieval.

The selector excludes the source document inside FAISS before ranking. No fixed
over-fetch multiplier is used for exclusion. Sparse eligible results trigger a doubling
of probes, up to all lists; tests cover only two eligible chunks and no eligible chunks.
Approximate neighbor recall and strict document exclusion are separate properties.

Serialized index size does not include retained float32 keys, tokens, document IDs,
or the CPU document grouping. The full datastore is therefore larger than the index.

Reproduce:

```bash
python -m experiments.faiss_datastore --pq_m 8 --out results/faiss_datastore/m8.json
python -m experiments.faiss_datastore --pq_m 16 --out results/faiss_datastore/m16.json
```

The exact backend remains the default. No training recipe or model architecture changes.
A downstream held-out-text comparison is still needed before choosing IVF-PQ for a run.
