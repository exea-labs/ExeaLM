"""Evaluate a checkpoint through EleutherAI's harness: closed-book (no external memory), or open-book with the library.

python -m exealm.eval.harness --ckpt runs/pretrain/model.pt --tokenizer tokenizer --tasks hellaswag --limit 20
Install exealm[eval]. Use the exact tokenizer the checkpoint was trained with.

Open-book (--library "written=data/library/v1/bm25;human=/path/bm25_pool_p100000"): for every request the question
(the request's context, never its answer choices) is searched in each tier (experiments/library/index.py, BM25 over
token ids), and the top passage of each tier, cut to --rag_tokens tokens, goes before the context with a blank line after
each: the layout RAG post-training uses. Every answer choice of a question gets the same passages. Any model with the
same tokenizer can be scored this way, so baselines get the same library. --stream_window W reads the input as windows of
W tokens with each window's state carried into the next (how the models were trained to read long input, and how the
study's long-document evaluation reads every model); 0 reads it in one pass.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.utils import get_rolling_token_windows, handle_non_serializable, make_disjoint_window

from ..data.tokenizer import load
from ..memory import MemoryContext
from ..train.pretrain import load_model, pick_device

INDEX_PY = Path(__file__).resolve().parents[2] / 'experiments' / 'library' / 'index.py'


def library_index_module():
    """experiments/library/index.py, loaded by path (experiments/library is not a package)."""
    spec = importlib.util.spec_from_file_location('library_index', INDEX_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

    def __init__(self, ckpt, tokenizer='tokenizer', device='cpu', max_length=1024, max_gen_toks=128, library=None,
                 rag_tokens=384, rag_max_df=0.1, stream_window=0, rag_exclude=None):
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
        self.stream_window = int(stream_window)
        self.library, self.rag_tokens, self.rag_max_df = library, int(rag_tokens), float(rag_max_df)
        self.tiers, self._rag_cache = {}, {}
        self.exclude = {}               # tier -> passage numbers never given (they share text with a benchmark)
        for part in (rag_exclude or '').split(';'):
            if part:
                name, path = part.split('=', 1)
                self.exclude[name] = {int(k) for k in json.load(open(path))['drop']}
        if library:
            ix = library_index_module()
            for part in library.split(';'):
                name, paths = part.split('=', 1)
                self.tiers[name] = ix.Index(paths.split('+'))
                if self.device.type == 'cuda':
                    self.tiers[name].to(self.device)
            self._sep = self.tok_encode('\n\n')
            self._skip = (self._eot, *self._sep)

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
        memory = (f'library {self.library}: top passage of each tier, {self.rag_tokens} tokens each; excluded for '
                  f'benchmark overlap: {dict((k, len(v)) for k, v in self.exclude.items())}'
                  if self.library else 'solo; no external datastore or board')
        return dict(checkpoint_sha256=self.checkpoint_hash, tokenizer_sha256=self.tokenizer_hash,
                    max_length=self.max_length, evaluation_memory=memory, stream_window=self.stream_window)

    def with_library(self, context_ids):
        """The context with each tier's top passage for it before it (searched once per context)."""
        if not self.tiers or not context_ids:
            return context_ids
        key = tuple(context_ids)
        if key not in self._rag_cache:
            head = [self._eot]
            for name, idx in self.tiers.items():
                find = idx.search_gpu if self.device.type == 'cuda' else idx.search
                d, _ = find(context_ids, k=5, max_df=self.rag_max_df, skip=self._skip)
                d = [int(x) for x in d if int(x) not in self.exclude.get(name, ())]
                if d:
                    head += idx.passage(d[0])[:self.rag_tokens] + self._sep
            self._rag_cache[key] = head
        return self._rag_cache[key] + list(context_ids)

    def forward_logits(self, inputs):
        """Logits for a (1, T) input: one pass, or windows of stream_window tokens with the state carried."""
        if not self.stream_window or inputs.shape[1] <= self.stream_window:
            return self.model(inputs)[0]
        state, out = {}, []
        for s in range(0, inputs.shape[1], self.stream_window):
            out.append(self.model(inputs[:, s:s + self.stream_window], MemoryContext(stream=state))[0])
        return torch.cat(out, dim=1)

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
            if self.tiers and context:
                context_ids = self.with_library(context_ids)
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
            logits = self.forward_logits(inputs)[0, -len(targets):].float()
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
            if self.tiers:
                prefix = self.with_library(prefix)
            generated, decoded = [], ''
            for _ in range(0 if '' in until else limit):
                tokens = (prefix + generated)[-self.max_length:]
                idx = torch.tensor([tokens], device=self.device, dtype=torch.long)
                nxt = int(self.forward_logits(idx)[0, -1].argmax())
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


def evaluate_checkpoint(ckpt, tokenizer, tasks, limit=None, include_path=None, max_length=1024, max_gen_toks=128,
                        device='cpu', library=None, rag_tokens=384, stream_window=0, rag_exclude=None,
                        log_samples=False):
    """Run harness tasks on one checkpoint and return the harness's result dict (used by main and by eval/auto.py)."""
    model = ExeaLMHarness(ckpt, tokenizer, device, max_length, max_gen_toks, library=library, rag_tokens=rag_tokens,
                          stream_window=stream_window, rag_exclude=rag_exclude)
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    manager = TaskManager(include_path=include_path) if include_path else None
    return simple_evaluate(model=model, tasks=list(tasks), limit=limit, num_fewshot=0, log_samples=log_samples,
                           bootstrap_iters=0, task_manager=manager, device=str(model.device))


def write_samples(result, path):
    """One line per question: task, doc_id, the question's text and whether each metric scored it right. For paired
    tests between runs, and for scoring apart the questions that share text with training data."""
    with open(path, 'x', encoding="utf-8", newline="\n") as f:
        for task, rows in result.pop('samples', {}).items():
            for r in rows:
                doc = r.get('doc', {})
                q = doc.get('question') or doc.get('query') or doc.get('ctx') or ''
                line = dict(task=task, doc_id=r.get('doc_id'), question=q if isinstance(q, str) else str(q))
                for k in ('acc', 'acc_norm'):
                    if k in r:
                        line[k] = float(r[k])
                f.write(json.dumps(line, ensure_ascii=False) + '\n')


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
    parser.add_argument('--library', default=None, help='open-book: "name=index[+index];name=index", e.g. '
                                                        '"written=data/library/v1/bm25;human=/path/bm25_pool_p100000"')
    parser.add_argument('--rag_tokens', type=int, default=384, help='tokens kept of each retrieved passage')
    parser.add_argument('--rag_exclude', default=None, help='"tier=bench_hits.json;..." (experiments/library/'
                                                            'bench_check.py): passages never given, the next best instead')
    parser.add_argument('--stream_window', type=int, default=0, help='read the input as windows of this many tokens '
                                                                     'with the state carried (0: one pass)')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--out', default=None, help='new JSON results file')
    parser.add_argument('--samples', default=None, help='new JSONL file: per-question results (write_samples)')
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error('threads and limit must be positive')
    for p in (args.out, args.samples):
        if p and Path(p).exists():
            parser.error(f'{p} exists: --out and --samples must be new files')
    torch.set_num_threads(args.threads)
    result = evaluate_checkpoint(args.ckpt, args.tokenizer, args.tasks.split(','), args.limit, args.include_path,
                                 args.max_length, args.max_gen_toks, args.device, args.library, args.rag_tokens,
                                 args.stream_window, args.rag_exclude, log_samples=bool(args.samples))
    if args.samples:
        Path(args.samples).parent.mkdir(parents=True, exist_ok=True)
        write_samples(result, args.samples)
    encoded = json.dumps(result, default=handle_non_serializable, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'x', encoding="utf-8", newline="\n") as f:
            f.write(encoded + '\n')
    print(encoded)
    return result


if __name__ == '__main__':
    main()
