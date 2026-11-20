import pytest

from exealm.data.files import expand


def test_patterns_expand_in_sorted_order_and_plain_paths_pass_through(tmp_path):
    for name in ("b.jsonl.gz", "a.jsonl.gz", "c.txt"):
        (tmp_path / name).write_bytes(b"")
    assert expand([str(tmp_path / "*.jsonl.gz")]) == [str(tmp_path / "a.jsonl.gz"), str(tmp_path / "b.jsonl.gz")]
    assert expand([str(tmp_path / "c.txt"), str(tmp_path / "later.bin")]) == [str(tmp_path / "c.txt"), str(tmp_path / "later.bin")]


def test_a_pattern_that_matches_nothing_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="no files match"):
        expand([str(tmp_path / "*.jsonl.gz")])
