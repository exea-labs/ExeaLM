import gzip
import hashlib
import json
from pathlib import Path

import pytest

from exealm.data.benchmarks import BENCHMARKS, export_rows, segments
from exealm.data.pack import pack

FIXTURES = json.loads((Path(__file__).parent / 'fixtures' / 'benchmarks.json').read_text())


@pytest.mark.parametrize('name', BENCHMARKS)
def test_export_preserves_text_and_packs_with_provenance(tmp_path, name):
    row = FIXTURES[name]
    out = tmp_path / name
    manifest = export_rows(name, [row, row], out, revision='fixture-revision')
    assert manifest['rows'] == 2 and not manifest['sample']
    assert manifest['split'] in ('test', 'validation')
    assert manifest['sha256'] == hashlib.sha256((out / 'text.jsonl.gz').read_bytes()).hexdigest()
    with gzip.open(out / 'text.jsonl.gz', 'rt') as f:
        records = [json.loads(line) for line in f]
    expected = segments(name, row)
    assert [(r['field'], r['text']) for r in records] == expected * 2
    assert all(r['source'] == name and r['benchmark_id'] for r in records)
    meta = pack([str(out / 'text.jsonl.gz')], 'bytes', str(tmp_path / 'packed'))
    assert meta['n_docs'] == manifest['text_segments']
    assert json.loads((out / 'manifest.json').read_text()) == manifest


def test_solution_and_reasoning_whitespace_and_all_choices_preserved():
    row = FIXTURES['humaneval']
    out = dict(segments('humaneval', row))
    assert out['solution'] == '    return a + b\n'
    assert out['completed_code'] == row['prompt'] + row['canonical_solution']
    assert out['tests'] == row['test']
    assert dict(segments('gsm8k', FIXTURES['gsm8k']))['answer'].endswith('\n#### 6')
    assert len(segments('jcommonsenseqa', FIXTURES['jcommonsenseqa'])) == 6
    assert len(segments('hellaswag', FIXTURES['hellaswag'])) == 9


def test_limited_export_does_not_consume_extra_rows(tmp_path):
    def rows():
        yield FIXTURES['gsm8k']
        raise AssertionError('must not read beyond limit')
    result = export_rows('gsm8k', rows(), tmp_path / 'out', 'sha', limit=1)
    assert result['sample'] and result['rows'] == 1 and result['limit'] == 1


@pytest.mark.parametrize('rows', [[], [{}], [{'question': 'q', 'answer': None}]])
def test_invalid_or_empty_rows_never_publish_partial_export(tmp_path, rows):
    with pytest.raises(ValueError):
        export_rows('gsm8k', rows, tmp_path / 'out', 'sha')
    assert not (tmp_path / 'out').exists()


def test_refuses_existing_output_and_invalid_limit(tmp_path):
    with pytest.raises(FileExistsError):
        export_rows('gsm8k', [FIXTURES['gsm8k']], tmp_path, 'sha')
    with pytest.raises(ValueError):
        export_rows('gsm8k', [FIXTURES['gsm8k']], tmp_path / 'out', 'sha', limit=0)
