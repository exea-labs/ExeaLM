"""Matched two-seed CPU correctness pilot for circuit scale initialization.

python -m experiments.cold_scale_smoke --out runs/cold-scale-smoke
The reference is the previous current-batch RMS fallback. This is a tiny synthetic
sanity comparison, not evidence of real-text quality or a new architecture result.
"""
import argparse
import contextlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from exealm import ExeaLM, get_config
from exealm.model.tonn import TONNLayer
from exealm.train.pretrain import main as pretrain


def previous_scale(self, y, order):
    """Frozen reference from main 04d2289: unmeasured slots use current-batch RMS."""
    with torch.no_grad():
        rms = y.detach().float().pow(2).mean(dim=(0, 1, 3)).sqrt()
        previous = self.scale.index_select(0, order)
        used = torch.where(previous > 0, previous, rms)
        if self.training:
            self.scale.index_copy_(0, order, torch.where(previous > 0, .99 * previous + .01 * rms, rms))
    return y / (used.view(1, 1, -1, 1).to(y.dtype) + 1e-3)


def prefix_candidate(self, y, order):
    """Rejected candidate: row-wise prefix RMS also removes leakage but worsened both pilot seeds."""
    with torch.no_grad():
        squared = y.detach().float().pow(2)
        rms = squared.mean(dim=(0, 1, 3)).sqrt()
        previous = self.scale.index_select(0, order)
        count = torch.arange(1, y.shape[1] + 1, device=y.device).view(1, -1, 1)
        prefix = (squared.mean(dim=3).cumsum(dim=1) / count).sqrt()
        used = torch.where(previous.view(1, 1, -1) > 0, previous.view(1, 1, -1), prefix)
        if self.training:
            self.scale.index_copy_(0, order, torch.where(previous > 0, .99 * previous + .01 * rms, rms))
    return y / (used.unsqueeze(-1).to(y.dtype) + 1e-3)


def leakage():
    torch.manual_seed(0)
    model = ExeaLM(get_config('tiny', menu=('I',), pinned={}, n_layers=1, ffn=0, lags=(0,))).eval()
    x = torch.randint(0, 256, (2, 24))
    changed = x.clone()
    changed[0, -1] = (changed[0, -1] + 17) % 256
    other = x.clone()
    other[1] = (other[1] + 39) % 256
    with torch.no_grad():
        base = model(x)[0]
        future = model(changed)[0]
        unrelated = model(other)[0]
    return dict(future_prefix_max_diff=float((base[0, :12] - future[0, :12]).abs().max()),
                unrelated_row_max_diff=float((base[0] - unrelated[0]).abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs/cold-scale-smoke')
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    fixed = TONNLayer._unit_scale
    records = []
    configs = {}
    invariants = {}
    try:
        for name, function in [('previous', previous_scale), ('causal', fixed), ('prefix_candidate', prefix_candidate)]:
            TONNLayer._unit_scale = function
            invariants[name] = leakage()
            for seed in [0, 1]:
                out = root / f'{name}-seed{seed}'
                with (root / f'{name}-seed{seed}.txt').open('w') as log, contextlib.redirect_stdout(log):
                    pretrain(['--preset', 'tiny', '--data', 'synthetic', '--steps', '100',
                              '--batch', '4', '--seq', '64', '--eval_every', '100', '--eval_batches', '2',
                              '--search_every', '20', '--datastore_chunks', '100', '--refresh_every', '40',
                              '--device', 'cpu', '--seed', str(seed), '--out', str(out)])
                events = [json.loads(line) for line in (out / 'log.jsonl').read_text().splitlines()]
                final = next(e for e in reversed(events) if e['event'] == 'eval')
                if str(seed) in configs:
                    assert configs[str(seed)] == events[0]['cfg']
                configs[str(seed)] = events[0]['cfg']
                records.append(dict(normalization=name, seed=seed, params=events[0]['params'],
                                    val_bits_per_token=final['val_bits_per_token'], train_loss=final['train_loss']))
    finally:
        TONNLayer._unit_scale = fixed
    result = dict(torch=torch.__version__, numpy=np.__version__, platform=platform.platform(), baseline='04d2289 current-batch RMS fallback',
                  config=dict(preset='tiny', steps=100, batch=4, seq=64, eval_batches=2,
                              search_every=20, datastore_chunks=100, refresh_every=40, threads=1),
                  training_input_tokens_per_run=25_600, model_configs=configs, leakage=invariants, runs=records)
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
