"""Evaluate a checkpoint through EleutherAI's harness (solo, no external memory).

python -m exealm.eval.harness --ckpt runs/pretrain/model.pt --tokenizer tokenizer --tasks hellaswag --limit 20
Install exealm[eval]. Use the exact tokenizer the checkpoint was trained with.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.utils import get_rolling_token_windows, handle_non_serializable, make_disjoint_window

from ..data.tokenizer import load
from ..train.pretrain import load_model, pick_device


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@register_model('exealm')
class ExeaLMHarness(TemplateLM):
    """Serial requests avoid padding or cross-example board communication.

    Likelihoods use the harness's context/continuation tokenization convention.
    A long continuation is scored in disjoint target blocks; left context is
    truncated to fit each block. No continuation tokens are silently discarded.
    Generation is greedy and re-reads a bounded context at each step.
    """

    def __init__(self, ckpt, tokenizer='tokenizer', device='cpu', max_length=1024, max_gen_toks=128):
        super().__init__()
        if max_length < 1 or max_gen_toks < 0:
            raise ValueError('positive max_length and nonnegative max_gen_toks required')
        self.max_length, self.max_gen_toks = int(max_length), int(max_gen_toks)
        self._device = pick_device(device)
        self.model, state = load_model(str(ckpt), self.device)
        del state  # discard optimizer/training state during evaluation
        self.model.eval()
        self.tokenizer = load(str(tokenizer))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        if self.tokenizer.get_vocab_size() != self.model.cfg.vocab_size:
            raise ValueError('tokenizer vocabulary does not match checkpoint')
        self._eot = self.tokenizer.token_to_id('<|endoftext|>')
        if self._eot is None:
            raise ValueError('tokenizer must define <|endoftext|>')
        tok_file = Path(tokenizer)
        tok_file = tok_file / 'tokenizer.json' if tok_file.is_dir() else tok_file
        self.tokenizer_hash = file_hash(tok_file)
        self.checkpoint_hash = file_hash(ckpt)

    @property
    def eot_token_id(self):
        return self._eot

    @property
    def tokenizer_name(self):
        return 'exealm-' + self.tokenizer_hash

    def tok_encode(self, string, add_special_tokens=False, **kwargs):
        return self.tokenizer.encode(string, add_special_tokens=bool(add_special_tokens)).ids

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def get_model_info(self):
        return dict(checkpoint_sha256=self.checkpoint_hash, tokenizer_sha256=self.tokenizer_hash,
                    max_length=self.max_length, evaluation_memory='solo; no external datastore or board')

    def loglikelihood(self, requests, disable_tqdm=False):
        encoded = []
        for request in requests:
            context, continuation = request.args
            if not continuation:
                context_ids, continuation_ids = [self.eot_token_id], []
            elif not context:
                continuation_ids = self.tok_encode(continuation)
                context_ids = [self.eot_token_id]
                if continuation_ids and continuation_ids[0] == self.eot_token_id:
                    continuation_ids = continuation_ids[1:]
            else:
                context_ids, continuation_ids = self._encode_pair(context, continuation)
            encoded.append(((context, continuation), context_ids, continuation_ids))
        return self._loglikelihood_tokens(encoded)

    @torch.inference_mode()
    def _score(self, context, continuation):
        if not continuation:
            return 0.0, True
        context = context or [self.eot_token_id]
        total, greedy = 0.0, True
        for start in range(0, len(continuation), self.max_length):
            targets = continuation[start:start + self.max_length]
            joined = (context + continuation[:start] + targets)[-(self.max_length + 1):]
            inputs = torch.tensor([joined[:-1]], device=self.device, dtype=torch.long)
            logits = self.model(inputs)[0][0, -len(targets):].float()
            target = torch.tensor(targets, device=self.device, dtype=torch.long)
            total += float(logits.log_softmax(-1).gather(1, target[:, None]).sum())
            greedy = greedy and bool((logits.argmax(-1) == target).all())
        return total, greedy

    def _loglikelihood_tokens(self, requests, **kwargs):
        results = []
        for cache_key, context, continuation in requests:
            result = self._score(context, continuation)
            results.append(result)
            if cache_key is not None:
                self.cache_hook.add_partial('loglikelihood', cache_key, result)
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        results = []
        for request in requests:
            (text,) = request.args
            windows = get_rolling_token_windows(self.tok_encode(text), self.eot_token_id,
                                                self.max_length, context_len=1)
            total = sum((self._score(*make_disjoint_window(window))[0] for window in windows), 0.0)
            results.append(total)
            self.cache_hook.add_partial('loglikelihood_rolling', (text,), total)
        return results

    @torch.inference_mode()
    def generate_until(self, requests, disable_tqdm=False):
        results = []
        for request in requests:
            prompt, options = request.args
            kwargs = dict(options)
            until = kwargs.pop('until', []) or []
            until = [until] if isinstance(until, str) else until
            if not isinstance(until, list) or not all(isinstance(s, str) for s in until):
                raise ValueError('until must be a string or list of strings')
            limit = kwargs.pop('max_gen_toks', self.max_gen_toks)
            if not isinstance(limit, int) or limit < 0:
                raise ValueError('max_gen_toks must be a nonnegative integer')
            if kwargs.pop('do_sample', False) or kwargs.pop('temperature', 0) not in (0, None):
                raise ValueError('this adapter supports greedy generation only')
            if kwargs:
                raise ValueError(f'unsupported generation options: {sorted(kwargs)}')
            prefix = self.tok_encode(prompt) or [self.eot_token_id]
            generated, decoded = [], ''
            for _ in range(0 if '' in until else limit):
                tokens = (prefix + generated)[-self.max_length:]
                idx = torch.tensor([tokens], device=self.device, dtype=torch.long)
                nxt = int(self.model(idx)[0][0, -1].argmax())
                if nxt == self.eot_token_id:
                    break
                generated.append(nxt)
                decoded = self.tok_decode(generated)
                stops = [decoded.find(s) for s in until if s in decoded]
                if stops:
                    decoded = decoded[:min(stops)]
                    break
            results.append(decoded)
            self.cache_hook.add_partial('generate_until', (prompt, options), decoded)
        return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--tokenizer', default='tokenizer')
    parser.add_argument('--tasks', default='hellaswag', help='comma-separated harness task names')
    parser.add_argument('--include_path', default=None, help='directory of custom harness task YAML files')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--max_gen_toks', type=int, default=128)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--out', default=None, help='new JSON results file')
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error('threads and limit must be positive')
    if args.out and Path(args.out).exists():
        parser.error('--out must be a new file')
    torch.set_num_threads(args.threads)
    model = ExeaLMHarness(args.ckpt, args.tokenizer, args.device, args.max_length, args.max_gen_toks)
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    manager = TaskManager(include_path=args.include_path) if args.include_path else None
    result = simple_evaluate(model=model, tasks=args.tasks.split(','), limit=args.limit,
                             num_fewshot=0, log_samples=False, bootstrap_iters=0, task_manager=manager, device=str(model.device))
    encoded = json.dumps(result, default=handle_non_serializable, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'x') as f:
            f.write(encoded + '\n')
    print(encoded)
    return result


if __name__ == '__main__':
    main()
