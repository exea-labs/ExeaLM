"""Export held-out benchmark text for packing and exact-overlap decontamination.

python -m exealm.data.benchmarks --benchmark gsm8k --out data/benchmarks/gsm8k
Requires exealm[data] for streaming downloads. Exported text is exclusion material,
not training data. No benchmark code (including HumanEval tests) is executed.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile


# Pin the resolved dataset revision in each manifest; validation splits are used
# where evaluation labels/test text are not publicly distributed.
BENCHMARKS = {
    'hellaswag': ('Rowan/hellaswag', 'default', 'validation', None),
    'arc_easy': ('allenai/ai2_arc', 'ARC-Easy', 'test', 'CC-BY-SA-4.0'),
    'arc_challenge': ('allenai/ai2_arc', 'ARC-Challenge', 'test', 'CC-BY-SA-4.0'),
    'mmlu': ('cais/mmlu', 'all', 'test', 'MIT'),
    'gsm8k': ('openai/gsm8k', 'main', 'test', 'MIT'),
    'humaneval': ('openai/openai_humaneval', 'openai_humaneval', 'test', 'MIT'),
    'jcommonsenseqa': ('sbintuitions/JCommonsenseQA', 'default', 'validation', 'CC-BY-SA-4.0'),
}


def text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('required benchmark text must be a nonempty string')
    return value  # retain code indentation, reasoning, Unicode and original whitespace


def choices(value):
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError('expected a list of answer choices')
    return [text(v) for v in value]


def segments(name, row):
    """Return named text segments; unrelated choices never share an n-gram window."""
    if name == 'hellaswag':
        context = text(row['ctx'])
        endings = choices(row['endings'])
        result = [('context', context)]
        for i, ending in enumerate(endings):
            result += [(f'ending_{i}', ending), (f'continuation_{i}', context + ' ' + ending)]
        return result
    if name in ('arc_easy', 'arc_challenge', 'mmlu', 'jcommonsenseqa'):
        question = text(row['question'])
        if name.startswith('arc_'):
            options = row['choices']
            if isinstance(options, dict):
                options = options['text']
            elif isinstance(options, list):
                options = [o['text'] for o in options]
            options = choices(options)
        elif name == 'jcommonsenseqa':
            options = choices([row[f'choice{i}'] for i in range(5)])
        else:
            options = choices(row['choices'])
        return [('question', question)] + [(f'choice_{i}', option) for i, option in enumerate(options)]
    if name == 'gsm8k':
        return [('question', text(row['question'])), ('answer', text(row['answer']))]
    if name == 'humaneval':
        prompt, solution, tests = (text(row[k]) for k in ('prompt', 'canonical_solution', 'test'))
        return [('prompt', prompt), ('solution', solution), ('tests', tests), ('completed_code', prompt + solution)]
    raise ValueError(f'unknown benchmark: {name}')


def export_rows(name, rows, out, revision, limit=None):
    """Atomically publish a fresh export directory and manifest, or leave no output.

    A limit always marks the export as a sample, even if the iterator happens to
    contain fewer rows. Unexpected schemas fail rather than silently dropping text.
    """
    if name not in BENCHMARKS or not revision or (limit is not None and limit < 1):
        raise ValueError('valid benchmark, resolved revision and positive limit required')
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    repo, config, split, license_name = BENCHMARKS[name]
    with tempfile.TemporaryDirectory(prefix='.benchmark-', dir=out.parent) as temp:
        work = Path(temp) / 'export'
        work.mkdir()
        count = parts = 0
        with gzip.open(work / 'text.jsonl.gz', 'wt', encoding='utf-8') as f:
            source = rows if limit is None else itertools.islice(rows, limit)
            for i, row in enumerate(source):
                try:
                    extracted = segments(name, row)
                except (KeyError, TypeError, ValueError) as e:
                    raise ValueError(f'{name} row {i}: {e}') from e
                ident = str(row.get('task_id', row.get('id', row.get('q_id', row.get('ind', i)))))
                for field, content in extracted:
                    f.write(json.dumps(dict(text=content, source=name, benchmark_id=ident, field=field),
                                       ensure_ascii=False) + '\n')
                    parts += 1
                count += 1
        if not count:
            raise ValueError('empty benchmark export')
        digest = hashlib.sha256((work / 'text.jsonl.gz').read_bytes()).hexdigest()
        manifest = dict(benchmark=name, repo=repo, config=config, split=split, revision=revision,
                        source_url=f'https://huggingface.co/datasets/{repo}/tree/{revision}',
                        license=license_name, rows=count, text_segments=parts, limit=limit,
                        sample=limit is not None, sha256=digest,
                        policy='raw fields; HellaSwag continuations and completed HumanEval code also included')
        (work / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        # Refuse an output appearing during export, too.
        if out.exists():
            raise FileExistsError(out)
        os.rename(work, out)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=BENCHMARKS, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--revision', default='main', help='dataset ref; resolved to a commit SHA before download')
    parser.add_argument('--limit', type=int, default=None, help='smoke-test rows only; marks export as sample')
    args = parser.parse_args(argv)
    if Path(args.out).exists():
        parser.error('--out must be a new directory')
    from datasets import load_dataset
    from huggingface_hub import HfApi
    repo, config, split, _ = BENCHMARKS[args.benchmark]
    revision = HfApi().dataset_info(repo, revision=args.revision).sha
    rows = load_dataset(repo, config, split=split, revision=revision, streaming=True)
    manifest = export_rows(args.benchmark, rows, args.out, revision, args.limit)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    main()
