"""Build a tiny synthetic checkpoint and run HellaSwag through the harness.

python -m experiments.harness_smoke --out runs/harness-smoke --limit 20
Requires exealm[eval]. Uses the project's public tokenizer; downloads no model weights.
This is an integration check, not a model-quality experiment.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

from exealm import ExeaLM, get_config
from exealm.data.tokenizer import load
from exealm.eval.harness import main as evaluate
from exealm.train.pretrain import make_optimizer, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs/harness-smoke')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--tokenizer_revision', default='main')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('limit must be positive')
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    revision = HfApi().model_info('exea-labs/ExeaLM', revision=args.tokenizer_revision).sha
    file = hf_hub_download('exea-labs/ExeaLM', 'tokenizer/tokenizer.json', revision=revision)
    tokenizer = load(file)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    cfg = get_config('tiny', vocab_size=tokenizer.get_vocab_size())
    model = ExeaLM(cfg)
    opt = make_optimizer(model, .001, .1)
    for _ in range(3):
        idx = torch.randint(0, cfg.vocab_size, (1, 257))
        opt.zero_grad()
        loss = model(idx[:, :-1], targets=idx[:, 1:])[1]
        loss.backward()
        opt.step()
    save(str(root / 'model.pt'), model, opt, 3, cfg, 'tonn')
    (root / 'source.json').write_text(json.dumps(dict(tokenizer_repo='exea-labs/ExeaLM',
        tokenizer_path='tokenizer/tokenizer.json', revision=revision, seed=0, cfg=asdict(cfg),
        training_steps=3, training_data='independent uniform token IDs; no benchmark training data'), indent=2))
    del model, opt
    evaluate(['--ckpt', str(root / 'model.pt'), '--tokenizer', file, '--tasks', 'hellaswag',
              '--limit', str(args.limit), '--max_length', '256', '--threads', '4',
              '--out', str(root / 'hellaswag.json')])


if __name__ == '__main__':
    main()
