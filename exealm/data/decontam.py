"""Exact token n-gram decontamination of packed shards (same tokenizer required).

Benchmark inputs are held-out text already packed by exealm.data.pack. No n-gram
crosses a document boundary or includes its trailing end-of-text token. A match
means exact token overlap, not semantic equivalence. The index lives in RAM;
training tokens are memory-mapped and processed one document at a time.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


class PackedDocuments:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / 'meta.json').read_text())
        dtype = np.dtype(self.meta['dtype'])
        if dtype not in (np.dtype('uint16'), np.dtype('uint32')):
            raise ValueError('packed tokens must be uint16 or uint32')
        size = (self.path / 'tokens.bin').stat().st_size
        if size != self.meta['n_tokens'] * dtype.itemsize:
            raise ValueError('token file size does not match metadata')
        self.tokens = (np.memmap(self.path / 'tokens.bin', dtype=dtype, mode='r')
                       if size else np.empty(0, dtype=dtype))
        self.offsets = np.load(self.path / 'offsets.npy', allow_pickle=False)
        if (self.offsets.ndim != 1 or self.offsets.dtype.kind not in 'iu'
                or len(self.offsets) != self.meta['n_docs']
                or (len(self.offsets) and (self.offsets[0] != 0
                    or self.offsets[-1] >= len(self.tokens)
                    or np.any(self.offsets[1:] <= self.offsets[:-1])))
                or (not len(self.offsets) and len(self.tokens))):
            raise ValueError('invalid document offsets')
        ref = self.meta['tokenizer']
        self.tokenizer = None
        if ref == 'bytes':
            self.identity, self.eot = 'bytes', 256
            if self.meta['vocab_size'] != 257:
                raise ValueError('byte tokenizer vocabulary must be 257')
        else:
            from .tokenizer import load
            file = Path(ref)
            file = file / 'tokenizer.json' if file.is_dir() else file
            self.identity = hashlib.sha256(file.read_bytes()).hexdigest()
            self.tokenizer = load(str(file))
            self.eot = self.tokenizer.token_to_id('<|endoftext|>')
            if self.eot is None or self.tokenizer.get_vocab_size() != self.meta['vocab_size']:
                raise ValueError('tokenizer does not match packed metadata')

    def __iter__(self):
        for i, start in enumerate(self.offsets):
            end = int(self.offsets[i + 1]) if i + 1 < len(self.offsets) else len(self.tokens)
            doc = self.tokens[int(start):end]
            if doc[-1] != self.eot or np.any(doc >= self.meta['vocab_size']):
                raise ValueError('invalid token IDs or missing document terminator')
            yield doc

    def byte_count(self, doc):
        if self.tokenizer is None:
            return len(doc) - 1
        return len(self.tokenizer.decode(doc[:-1].tolist(), skip_special_tokens=False).encode('utf-8'))


def ngrams(tokens, n):
    for i in range(len(tokens) - n + 1):
        yield tuple(int(t) for t in tokens[i:i + n])


def decontaminate(sources, benchmarks, out, n=13, min_matches=1):
    """Drop a document with >= min_matches matching windows against any one benchmark.

    Counts refer to window occurrences (including repetitions), per benchmark.
    Benchmark shards must contain only the held-out text intended for exclusion.
    """
    if n < 1 or min_matches < 1 or not sources or not benchmarks:
        raise ValueError('positive n/min_matches and nonempty sources/benchmarks required')
    sources = [PackedDocuments(p) for p in sources]
    benches = {name: PackedDocuments(p) for name, p in benchmarks.items()}
    identity = sources[0].identity
    if any(s.identity != identity for s in [*sources, *benches.values()]):
        raise ValueError('all shards must use the identical tokenizer')
    index = {}
    bench_report = {}
    for name, shard in benches.items():
        count = 0
        for doc in shard:
            for gram in ngrams(doc[:-1], n):
                index.setdefault(gram, set()).add(name)
                count += 1
        bench_report[name] = dict(path=str(shard.path.resolve()), windows=count)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite a previous run or an input
    report = dict(n=n, min_matches=min_matches, unit='token', benchmarks=bench_report, sources={})
    for i, shard in enumerate(sources):
        dest = out / f'{i:03d}-{shard.path.name}'
        dest.mkdir()
        counts = dict(read=0, kept=0, dropped=0, matched_documents={name: 0 for name in benches})
        offsets, total, byte_count = [], 0, 0
        with (dest / 'tokens.bin').open('wb') as f:
            for doc in shard:
                hits = {name: 0 for name in benches}
                for gram in ngrams(doc[:-1], n):
                    for name in index.get(gram, ()):
                        hits[name] += 1
                counts['read'] += 1
                for name, count in hits.items():
                    counts['matched_documents'][name] += count >= min_matches
                if any(count >= min_matches for count in hits.values()):
                    counts['dropped'] += 1
                    continue
                offsets.append(total)
                f.write(doc.tobytes())
                total += len(doc)
                byte_count += shard.byte_count(doc)
                counts['kept'] += 1
        np.save(dest / 'offsets.npy', np.asarray(offsets, dtype=np.int64))
        meta = dict(shard.meta, n_tokens=total, n_docs=len(offsets),
                    bytes_per_token=byte_count / max(1, total),
                    decontamination=dict(n=n, min_matches=min_matches, report='../report.json'))
        (dest / 'meta.json').write_text(json.dumps(meta, indent=2))
        counts['output'] = dest.name
        report['sources'][str(shard.path.resolve())] = counts
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', help='packed training directories (one per source)')
    parser.add_argument('--benchmark', action='append', required=True, help='NAME=packed-held-out-directory')
    parser.add_argument('--out', required=True, help='new output directory')
    parser.add_argument('--ngram', type=int, default=13)
    parser.add_argument('--min_matches', type=int, default=1)
    args = parser.parse_args(argv)
    benchmarks = {}
    for spec in args.benchmark:
        name, sep, path = spec.partition('=')
        if not sep or not name or not path or name in benchmarks:
            parser.error('--benchmark requires unique NAME=directory entries')
        benchmarks[name] = path
    result = decontaminate(args.sources, benchmarks, args.out, args.ngram, args.min_matches)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
