import json

import numpy as np
import pytest

from exealm.data.decontam import PackedDocuments, decontaminate, main, ngrams, window_hashes
from exealm.data.pack import pack


def packed(tmp, name, texts):
    src = tmp / (name + '.jsonl')
    src.write_text(''.join(json.dumps({'text': t}) + '\n' for t in texts), encoding="utf-8", newline="\n")
    dest = tmp / name
    pack([str(src)], 'bytes', str(dest))
    return dest


def test_filter_reports_sources_and_preserves_documents(tmp_path):
    benchmark = packed(tmp_path, 'benchmark', ['the benchmark answer', '日本語の質問です'])
    a = packed(tmp_path, 'a', ['the benchmark answer', 'clean', '日本語の質問です'])
    b = packed(tmp_path, 'b', ['another clean document'])
    report = main([str(a), str(b), '--benchmark', f'qa={benchmark}', '--out', str(tmp_path / 'out')])
    assert report['sources'][str(a)]['dropped'] == 2
    assert report['sources'][str(a)]['matched_documents'] == {'qa': 2}
    assert report['sources'][str(b)]['kept'] == 1
    shard = PackedDocuments(tmp_path / 'out' / '000-a')
    assert [bytes(d[:-1].tolist()) for d in shard] == [b'clean']
    assert shard.meta['n_tokens'] == 6 and shard.meta['bytes_per_token'] == 5 / 6
    assert json.loads((tmp_path / 'out' / 'report.json').read_text(encoding="utf-8")) == report


def test_no_windows_cross_document_boundaries_or_terminators(tmp_path):
    bench = packed(tmp_path, 'bench', ['abc', 'def', 'XYZ'])
    src = packed(tmp_path, 'src', ['abcdef', 'cde', 'XYZ', ''])
    result = decontaminate([src], {'test': bench}, tmp_path / 'out', n=4)
    assert result['sources'][str(src)]['dropped'] == 0
    assert result['benchmarks']['test']['windows'] == 0


def test_threshold_is_per_benchmark_and_counts_repeated_windows(tmp_path):
    a = packed(tmp_path, 'a', ['abc'])
    b = packed(tmp_path, 'b', ['def'])
    src = packed(tmp_path, 'src', ['abcdef', 'abcabc'])
    result = decontaminate([src], {'a': a, 'b': b}, tmp_path / 'out', n=3, min_matches=2)
    assert result['sources'][str(src)]['dropped'] == 1
    assert result['sources'][str(src)]['matched_documents'] == {'a': 1, 'b': 0}


def test_all_dropped_output_can_be_read_by_decontamination(tmp_path):
    src = packed(tmp_path, 'src', ['same'])
    decontaminate([src], {'test': src}, tmp_path / 'out', n=2)
    empty = PackedDocuments(tmp_path / 'out' / '000-src')
    assert empty.meta['n_docs'] == 0 and list(empty) == []
    with pytest.raises(FileExistsError):
        decontaminate([src], {'test': src}, tmp_path / 'out', n=2)


def test_rejects_invalid_offsets_and_settings(tmp_path):
    src = packed(tmp_path, 'src', ['one', 'two'])
    with pytest.raises(ValueError):
        decontaminate([src], {'test': src}, tmp_path / 'out', n=0)
    np.save(src / 'offsets.npy', np.array([0, 0]))
    with pytest.raises(ValueError, match='offsets'):
        PackedDocuments(src)


def test_uint32_tokens_round_trip_and_terminator_validation(tmp_path):
    src = packed(tmp_path, 'src', ['clean text'])
    bench = packed(tmp_path, 'bench', ['unrelated'])
    shard = PackedDocuments(src)
    wide = np.asarray(shard.tokens, dtype=np.uint32)
    meta = dict(shard.meta, dtype='uint32')
    del shard  # Windows cannot rewrite a file that is still memory mapped
    wide.tofile(src / 'tokens.bin')
    (src / 'meta.json').write_text(json.dumps(meta), encoding="utf-8", newline="\n")
    decontaminate([src], {'test': bench}, tmp_path / 'out')
    out = PackedDocuments(tmp_path / 'out' / '000-src')
    assert out.tokens.dtype == np.uint32
    assert [d.tolist() for d in out] == [list(b'clean text') + [256]]
    data = np.fromfile(src / 'tokens.bin', dtype=np.uint32)
    data[-1] = 999
    data.tofile(src / 'tokens.bin')
    with pytest.raises(ValueError, match='terminator'):
        list(PackedDocuments(src))


def test_trained_tokenizer_round_trip_and_identity_check(tmp_path):
    pytest.importorskip('tokenizers')
    from exealm.data.tokenizer import train
    raw = tmp_path / 'raw.jsonl'
    raw.write_text(json.dumps({'text': 'a training example with clean text'}) + '\n', encoding="utf-8", newline="\n")
    tokenizer = tmp_path / 'tokenizer'
    train([str(raw)], 280, str(tokenizer))
    src = tmp_path / 'src'
    pack([str(raw)], str(tokenizer), str(src))
    bench = packed(tmp_path, 'bench', ['unrelated'])
    with pytest.raises(ValueError, match='identical tokenizer'):
        decontaminate([src], {'bytes': bench}, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()
    decontaminate([src], {'same': src}, tmp_path / 'out', n=100)
    result = PackedDocuments(tmp_path / 'out' / '000-src')
    assert [d.tolist() for d in result] == [d.tolist() for d in PackedDocuments(src)]
    assert result.meta['bytes_per_token'] == PackedDocuments(src).meta['bytes_per_token']


def test_hash_precheck_never_misses_an_exact_window():
    """Equal windows have equal hashes, and on random tokens the hashes separate different windows."""
    rng = np.random.default_rng(0)
    for dtype, vocab, n in [(np.uint16, 5, 3), (np.uint16, 257, 13), (np.uint32, 70_000, 13)]:
        tokens = rng.integers(0, vocab, size=400).astype(dtype)
        hashes = window_hashes(tokens, n)
        grams = list(ngrams(tokens, n))
        assert hashes.dtype == np.uint64 and len(hashes) == len(grams)
        by_gram = {}
        for g, h in zip(grams, hashes.tolist()):
            assert by_gram.setdefault(g, h) == h
        assert len(set(by_gram.values())) == len(by_gram)
    assert len(window_hashes(np.arange(5, dtype=np.uint16), 6)) == 0
    assert window_hashes(np.zeros(3, dtype=np.uint16), 3)[0] != window_hashes(np.zeros(4, dtype=np.uint16), 4)[0]


def test_filter_matches_the_plain_exact_scan_on_random_documents(tmp_path):
    """The whole filter against a slow reference that compares every window exactly."""
    rng = np.random.default_rng(1)
    alphabet = 'abcd'
    text = lambda k: ''.join(rng.choice(list(alphabet), size=k))
    bench_texts = [text(12) for _ in range(6)]
    docs = [text(int(rng.integers(1, 40))) for _ in range(300)] + [text(9) + bench_texts[0] + text(5)]
    bench, src = packed(tmp_path, 'bench', bench_texts), packed(tmp_path, 'src', docs)
    result = decontaminate([src], {'b': bench}, tmp_path / 'out', n=6)
    grams = {b[i:i + 6] for b in bench_texts for i in range(len(b) - 5)}
    expected = [d for d in docs if not any(d[i:i + 6] in grams for i in range(len(d) - 5))]
    kept = [bytes(d[:-1].tolist()).decode() for d in PackedDocuments(tmp_path / 'out' / '000-src')]
    assert kept == expected and 0 < len(expected) < len(docs)
    assert result['sources'][str(src)]['dropped'] == len(docs) - len(expected)
