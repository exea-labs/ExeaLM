import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from exealm.data.download import download_hf
from exealm.data.formatters import K2_FORMATTERS, get_k2_formatter
from exealm.data.sources import K2

FIXTURES = json.loads((Path(__file__).parent / 'fixtures' / 'k2_formatters.json').read_text())


def test_every_planned_subset_has_fixtures_and_a_formatter():
    planned = {(repo, subset) for repo, (_, subsets) in K2.items() for subset in subsets}
    assert set(K2_FORMATTERS) == planned == {(f['repo'], f['subset']) for f in FIXTURES}


@pytest.mark.parametrize('fixture', FIXTURES, ids=lambda f: f["repo"] + '/' + f["subset"])
def test_subset_preserves_preformatted_text_and_rejects_metadata(fixture):
    formatter = get_k2_formatter(fixture['repo'], fixture['subset'])
    a, b, broken = fixture['rows']
    assert formatter(a) == a['text']
    assert formatter(b) == 'def invented_double(x):\n    return 2 * x\n'
    assert formatter(broken) is None
    assert a['text'].startswith('user\n') and '\n\nassistant\n' in formatter(a)


@pytest.mark.parametrize('row', [None, [], 'text', {}, {'text': ''}, {'text': ' \t\n'}, {'text': 42},
                               {'messages': [{'role': 'user', 'content': 'not the verified schema'}]}])
def test_broken_rows_do_not_fall_back_to_guessed_text(row):
    assert get_k2_formatter('IFM/TxT360-v2', 'txt360-qa')(row) is None


def test_download_uses_verified_formatter_before_applying_kept_row_limit(tmp_path, monkeypatch):
    def rows(*args, **kwargs):
        assert args == ('IFM/TxT360-v2', 'txt360-qa')
        assert kwargs == dict(split='train', streaming=True)
        yield {'metadata': 'A long but non-document metadata value must not be exported.'}
        yield {'text': '  user\nhello\n\nassistant\nworld\n'}
        raise AssertionError('must stop after one valid document')
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=rows))
    download_hf('IFM/TxT360-v2', 'txt360-qa', 1, str(tmp_path))
    with gzip.open(next(tmp_path.glob('*.jsonl.gz')), 'rt') as f:
        assert [json.loads(line) for line in f] == [dict(text='  user\nhello\n\nassistant\nworld\n',
                                                       source='IFM/TxT360-v2/txt360-qa')]


def test_unknown_subset_and_invalid_limits_fail_before_download(tmp_path):
    with pytest.raises(ValueError, match='No verified'):
        download_hf('IFM/TxT360-v2', 'new-subset', 1, str(tmp_path))
    with pytest.raises(ValueError, match='positive'):
        download_hf('IFM/TxT360-v2', 'txt360-qa', 0, str(tmp_path))
    assert not list(tmp_path.iterdir())
