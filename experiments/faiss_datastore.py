"""Reproducible CPU recall@4 / latency comparison on random normalized keys.

python -m experiments.faiss_datastore --keys 1000000 --queries 256 --threads 4
"""
import argparse
import json
import platform
import statistics
import time

import numpy as np
import torch
import faiss

from exealm.memory.datastore import Datastore
from exealm.memory.faiss import FaissDatastore


def timed(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return result, statistics.median(samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--keys', type=int, default=1_000_000)
    ap.add_argument('--queries', type=int, default=256)
    ap.add_argument('--dim', type=int, default=32)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--pq_m', type=int, default=8)
    ap.add_argument('--rerank', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    faiss.omp_set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    keys = torch.nn.functional.normalize(torch.randn(a.keys, a.dim), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(a.queries, a.dim), dim=-1)
    docs = torch.arange(a.keys) // 100
    exclude = torch.arange(a.queries) % ((a.keys + 99) // 100)
    tokens = torch.zeros(a.keys, 1, dtype=torch.long)
    exact = Datastore(tokens, docs)
    exact.keys = keys
    approx = FaissDatastore(tokens, docs, nlist=256, pq_m=a.pq_m, nbits=8, seed=a.seed, rerank=a.rerank)
    approx.keys = keys
    start = time.perf_counter()
    approx.rebuild_index()
    build = time.perf_counter() - start
    rows = []
    for excluded in (False, True):
        ex = exclude if excluded else None
        (truth, _), elapsed = timed(lambda: exact.search(q, 4, ex), a.repeats)
        rows.append(dict(backend='exact', exclude=excluded, recall_at_4=1., seconds=elapsed))
        for probe in (8, 32, 128):
            approx.nprobe = probe
            (found, valid), elapsed = timed(lambda: approx.search(q, 4, ex), a.repeats)
            recall = (found[:, :, None] == truth[:, None, :]).any(-1).float().mean().item()
            assert valid.all()
            if ex is not None:
                assert not (docs[found] == ex[:, None]).any()
            rows.append(dict(backend='ivf-pq', nprobe=probe, exclude=excluded,
                             recall_at_4=recall, seconds=elapsed))
    result = dict(config=vars(a), platform=platform.platform(), torch=torch.__version__,
                  numpy=np.__version__, faiss=faiss.__version__, build_seconds=build,
                  index_bytes=len(faiss.serialize_index(approx.index)), rows=rows)
    text = json.dumps(result, indent=2)
    print(text)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
