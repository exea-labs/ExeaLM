"""Sources stored in many packed directories, mixed by weight."""
import json

import numpy as np
import pytest
import torch

from exealm.data.loader import MultiShards, ShardSet, SyntheticShards, TokenShards, open_data
from exealm.train import pretrain


def shard(path, first, n_tokens, doc_len=50, vocab=256):
    """A packed directory whose token at position i is first + i (mod vocab would hide the source: keep it small)."""
    path.mkdir(parents=True)
    np.arange(first, first + n_tokens, dtype=np.uint16).tofile(path / "tokens.bin")
    offsets = np.arange(0, n_tokens, doc_len, dtype=np.int64)
    np.save(path / "offsets.npy", offsets)
    (path / "meta.json").write_text(json.dumps(dict(dtype="uint16", vocab_size=vocab, n_tokens=n_tokens,
                                                    n_docs=len(offsets), bytes_per_token=1.0)))
    return str(path)


@pytest.fixture
def web(tmp_path):
    """Three shards of one source: tokens 0.., 10000.., 20000.., the second twice as long, plus one too short."""
    return [shard(tmp_path / "web" / "000", 0, 1000, vocab=40000),
            shard(tmp_path / "web" / "001", 10000, 2000, vocab=40000),
            shard(tmp_path / "web" / "002", 20000, 1000, vocab=40000),
            shard(tmp_path / "web" / "003", 30000, 10, vocab=40000)]


def test_shard_set_rows_are_windows_of_one_shard_with_unique_document_ids(web):
    src = ShardSet(web)
    assert src.meta["n_tokens"] == 4010 and src.meta["n_docs"] == 20 + 40 + 20 + 1
    x, y, d = src.batch(64, 16, np.random.default_rng(0))
    assert x.shape == y.shape == d.shape == (64, 16)
    assert (x[:, 1:] - x[:, :-1] == 1).all() and (y == x + 1).all()         # consecutive tokens, target = next token
    which = x[:, 0] // 10000
    assert set(which.tolist()) == {0, 1, 2}                                 # the 10-token shard is too short to draw
    base = torch.tensor([0, 20, 60])[which]
    assert (d == base[:, None] + (x - which[:, None] * 10000) // 50).all()  # shard's document id + ids before it


def test_shards_are_drawn_by_token_count_and_few_stay_open(web):
    src, small = ShardSet(web), ShardSet(web, max_open=1)
    a = src.batch(4000, 8, np.random.default_rng(1))
    b = small.batch(4000, 8, np.random.default_rng(1))
    assert all(torch.equal(u, v) for u, v in zip(a, b)) and len(small._open) == 1
    share = torch.bincount(a[0][:, 0] // 10000, minlength=3) / 4000
    assert torch.allclose(share, torch.tensor([0.25, 0.5, 0.25]), atol=0.03)
    with pytest.raises(ValueError, match="no shard"):
        src.batch(4, 4000, np.random.default_rng(0))


def test_a_swarm_reads_one_shard_and_split_swarms_read_consecutive_windows(web):
    x, y, d = ShardSet(web).batch(32, 16, np.random.default_rng(2), swarm_size=4, policy="split")
    rows = x.view(8, 4, 16)
    assert ((rows[:, :, 0] // 10000) == (rows[:, :1, 0] // 10000)).all()
    assert (rows[:, 1:, 0] - rows[:, :-1, 0] == 16).all()
    assert (y.view(8, 4, 16)[:, :3, -1] == -100).all() and (y.view(8, 4, 16)[:, 3, -1] != -100).all()


def test_mix_follows_the_weights_and_keeps_document_ids_apart(web, tmp_path):
    code = shard(tmp_path / "code", 50000, 3000, vocab=40000)
    mix = MultiShards([web, code], weights=[3, 1])
    x, y, d = mix.batch(4000, 8, np.random.default_rng(3))
    from_code = x[:, 0] >= 50000
    assert abs(float(from_code.float().mean()) - 0.25) < 0.03
    assert int(d[~from_code].max()) < 81 <= int(d[from_code].min()) and mix.meta["n_docs"] == 81 + 60
    by_size = MultiShards([web, code])
    assert by_size.p.tolist() == pytest.approx([4010 / 7010, 3000 / 7010])
    with pytest.raises(ValueError, match="weight"):
        MultiShards([web, code], weights=[1])
    other = shard(tmp_path / "other", 0, 500, vocab=300)
    with pytest.raises(ValueError, match="tokenizer"):
        MultiShards([code, other])


def test_open_data_reads_every_form(web, tmp_path):
    code = shard(tmp_path / "code", 50000, 3000, vocab=40000)
    assert isinstance(open_data("synthetic"), SyntheticShards)
    assert isinstance(open_data(code), TokenShards)
    many = open_data(str(tmp_path / "web" / "*"))
    assert isinstance(many, ShardSet) and len(many.paths) == 4
    spec = tmp_path / "mix.json"
    spec.write_text(json.dumps(dict(sources=dict(web=dict(paths=str(tmp_path / "web" / "*"), weight=0.75),
                                                 code=dict(paths=[code], weight=0.25)))))
    mix = open_data(str(spec))
    assert mix.names == ["web", "code"] and mix.p.tolist() == [0.75, 0.25]
    assert isinstance(mix.sources[0], ShardSet) and isinstance(mix.sources[1], TokenShards)
    spec.write_text(json.dumps(dict(sources=dict(web=dict(paths=web[:2]), code=dict(paths=code, weight=1)))))
    with pytest.raises(ValueError, match="every source a weight"):
        open_data(str(spec))
    with pytest.raises(ValueError, match="no packed directory"):
        open_data(str(tmp_path / "nothing" / "*"))


def test_pretraining_runs_on_a_mix_with_a_separate_library(tmp_path):
    rng = np.random.default_rng(0)

    def text(path, n):
        path.mkdir(parents=True)
        rng.integers(0, 256, size=n).astype(np.uint16).tofile(path / "tokens.bin")
        offsets = np.arange(0, n, 100, dtype=np.int64)
        np.save(path / "offsets.npy", offsets)
        (path / "meta.json").write_text(json.dumps(dict(dtype="uint16", vocab_size=256, n_tokens=n,
                                                        n_docs=len(offsets), bytes_per_token=1.0)))
        return str(path)

    for i in range(3):
        text(tmp_path / "en" / f"{i:03d}", 3000)
    ja, library = text(tmp_path / "ja", 4000), text(tmp_path / "library", 2000)
    spec = tmp_path / "mix.json"
    spec.write_text(json.dumps(dict(sources=dict(en=dict(paths=str(tmp_path / "en" / "*"), weight=0.6),
                                                 ja=dict(paths=ja, weight=0.4)))))
    common = ["--preset", "tiny", "--data", str(spec), "--steps", "6", "--eval_every", "3", "--search", "0",
              "--seq", "32", "--batch", "4", "--device", "cpu"]
    with pytest.raises(ValueError, match="--datastore_data"):
        pretrain.main(common + ["--datastore_chunks", "50", "--out", str(tmp_path / "bad")])
    pretrain.main(common + ["--datastore_chunks", "50", "--datastore_data", library, "--out", str(tmp_path / "run")])
    log = [json.loads(line) for line in open(tmp_path / "run" / "log.jsonl")]
    assert log[-2]["event"] == "eval" and log[-2]["step"] == 6 and np.isfinite(log[-2]["val_bits_per_token"])
