"""experiments/leave_one_out_data.py holds out the tail of each source and, with --fetch, gets a missing tokenizer."""
import importlib.util
import pathlib
import sys

import pytest

path = pathlib.Path(__file__).parents[1] / "experiments" / "leave_one_out_data.py"   # CI runs bare pytest: load by path
spec = importlib.util.spec_from_file_location("leave_one_out_data", path)
lood = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lood)


def test_split_holds_out_the_last_documents_and_keeps_order():
    docs = ["aaaa", "bbbb", "cccc", "dddd", "eeee"]
    train, val = lood.split(docs, 10e-6)               # 10 bytes: the last three 4-byte documents
    assert train == ["aaaa", "bbbb"] and val == ["cccc", "dddd", "eeee"]
    assert lood.split(docs, 6e-6) == (["aaaa", "bbbb", "cccc"], ["dddd", "eeee"])


def build(monkeypatch, tmp_path, tokenizer, fetch=True):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    for k in lood.SOURCES:
        lood.write([f"{k} {i}" for i in range(4)], str(raw / f"{k}.jsonl.gz"))
    calls = []
    monkeypatch.setattr(lood, "fetch", lambda *a: pytest.fail("every raw file is there, nothing to download"))
    monkeypatch.setattr(lood, "pack", lambda files, tok, out: calls.append((tok, pathlib.Path(out).name)))
    monkeypatch.setattr("exealm.hub._download", lambda name: calls.append(("download", name)) or "hub/tokenizer.json")
    argv = ["leave_one_out_data.py", "--raw", str(raw), "--tokenizer", str(tokenizer), "--out", str(tmp_path / "out"),
            "--val_mb", "1e-6"]
    monkeypatch.setattr(sys, "argv", argv + (["--fetch"] if fetch else []))
    lood.main()
    return calls


def test_fetch_downloads_the_tokenizer_when_it_is_missing(monkeypatch, tmp_path):
    calls = build(monkeypatch, tmp_path, tmp_path / "no_tokenizer")
    assert calls[0] == ("download", "tokenizer/tokenizer.json")
    assert calls[1:] == [("hub/tokenizer.json", n) for n in
                         ["train_en", "train_ja", "train_code", "val_en", "val_ja_wiki", "val_ja_web", "val_code"]]


def test_a_local_tokenizer_is_used_as_is(monkeypatch, tmp_path):
    tok = tmp_path / "bpe32k"
    tok.mkdir()
    for fetch in (True, False):
        calls = build(monkeypatch, tmp_path, tok, fetch)
        assert len(calls) == 7 and all(t == str(tok) for t, _ in calls)


def test_without_fetch_a_missing_tokenizer_is_not_downloaded(monkeypatch, tmp_path):
    calls = build(monkeypatch, tmp_path, tmp_path / "no_tokenizer", fetch=False)
    assert all(t == str(tmp_path / "no_tokenizer") for t, _ in calls)
