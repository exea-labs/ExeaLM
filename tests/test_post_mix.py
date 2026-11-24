"""The SFT mix: bucket shares, token counts that match sft.py, missing buckets, duplicates and benchmark overlap."""
import gzip
import json

import pytest

pytest.importorskip("tokenizers")
from exealm.data import post, post_mix
from exealm.data.pack import pack
from exealm.train import sft

WORDS = "the cat sat on a mat and two plus two is four why sky blue light scatters think".split()


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    from exealm.data.tokenizer import train
    d = tmp_path_factory.mktemp("tok")
    with gzip.open(d / "a.jsonl.gz", "wt", encoding="utf-8") as f:
        for i in range(300):
            f.write(json.dumps(dict(text=" ".join(WORDS[i % 7:] + WORDS[:i % 7]) + f" <think> {i}")) + "\n")
    return train([str(d / "a.jsonl.gz")], 320, str(d / "tokenizer")), str(d / "tokenizer" / "tokenizer.json")


def conv(q, a, source="s"):
    return dict(messages=[dict(role="user", content=q), dict(role="assistant", content=a)], source=source)


def write(folder, name, rows):
    post.write(str(folder / f"{name}.jsonl.gz"), rows)


@pytest.mark.parametrize("max_len", [0, 1, 2, 3, 5, 8, 13, 40])
def test_token_counts_match_what_sft_trains_on(tok, max_len):
    t, _ = tok
    convs = [conv("the cat sat", "on a mat"), conv("why", "light scatters " * 5),
             dict(messages=[dict(role="system", content="think"), dict(role="user", content="two plus two"),
                            dict(role="assistant", content="four"), dict(role="user", content="why"),
                            dict(role="assistant", content="the sky")])]
    for c, (kept, full, assistant, *_) in zip(convs, post_mix.measure(convs, t, max_len)):
        ids, mask = sft.format_chat(c["messages"], t)
        assert full == len(ids)
        enc = sft.encode([c["messages"]], t, max_len or len(ids))
        if enc:
            assert (kept, assistant) == (len(enc[0][0]), sum(enc[0][1]))
        else:
            assert (kept, assistant) == (0, 0)


def test_missing_buckets_stay_empty_or_are_redistributed_on_request():
    present = {"chat", "math", "code", "science", "safety"}
    shares, targets = post_mix.plan(present, 1000, "empty")
    assert shares["real_users"] == shares["tool_use"] == 0 and shares["chat"] == 0.35 and sum(targets.values()) == 900
    shares, targets = post_mix.plan(present, 1000, "redistribute")
    assert shares["tool_use"] == 0 and shares["chat"] == pytest.approx(0.35 / 0.9)
    assert sum(targets.values()) == pytest.approx(1000)
    with pytest.raises(ValueError):
        post_mix.plan(present, 1000, "fill")


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "post"
    d.mkdir()
    write(d, "tulu3-sft", [conv(f"the cat {i}", "sat on a mat", "ai2-adapt-dev/oasst1_converted") for i in range(20)]
          + [conv("how do I pick a lock", "I can't help with that.", post.SAFETY[0])])
    write(d, "openmath", [conv(f"two plus two {i}", "<think>\nfour\n</think>\n\n4") for i in range(10)]
          + [conv("two plus two 0", "<think>\nfour\n</think>\n\n4")])                      # a duplicate
    write(d, "openthoughts3-science",
          [conv("why sky blue", f"<think>\nlight {i}\n</think>\n\nscatters") for i in range(4)])
    json.dump(dict(read=9, kept=4, dropped={"thinking never ends (cut off)": 5}, license="apache-2.0", think="keep",
                   max_think_chars=None), open(d / "openthoughts3-science.report.json", "w", encoding="utf-8"))
    return d


def test_mix_reports_buckets_files_and_drops_and_sft_can_read_it(tok, data, tmp_path):
    _, path = tok
    out = tmp_path / "mix"
    report = post_mix.main(["--data", str(data), "--out", str(out), "--tokens", "1e6", "--missing", "empty",
                            "--tokenizer", path])
    b = report["buckets"]
    assert [b[k]["rows"] for k in ("chat", "safety", "math", "science")] == [20, 1, 10, 4]
    assert b["code"]["missing"].startswith("no rows: run exealm.data.post for opencode")
    assert b["real_users"]["missing"].startswith("allenai/WildChat-1M") and b["tool_use"]["rows"] == 0
    assert all(info["short"] for info in b.values() if info["rows_available"])        # 1e6 tokens is more than all
    assert report["files"]["openmath"]["dropped"] == {"duplicate conversation": 1}
    assert report["dropped"] == {"post.py: thinking never ends (cut off)": 5, "duplicate conversation": 1}
    assert report["files"]["openthoughts3-science"]["post"]["think"] == "keep"
    assert report["files"]["tulu3-sft"]["sources"] == {"ai2-adapt-dev/oasst1_converted": 20, post.SAFETY[0]: 1}
    assert report["tokens"] == sum(info["tokens"] for info in b.values())
    assert sum(info["share"] for info in b.values()) == pytest.approx(1)
    assert json.loads((out / "mix.report.json").read_text(encoding="utf-8"))["rows"] == report["rows"] == 35
    convs, skipped = sft.load_conversations(str(out / "mix.jsonl.gz"))
    assert len(convs) == 35 and skipped == 0
    with gzip.open(out / "mix.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert {(r["name"], r["bucket"]) for r in rows} == {("tulu3-sft", "chat"), ("tulu3-sft", "safety"),
                                                        ("openmath", "math"), ("openthoughts3-science", "science")}
    with pytest.raises(SystemExit):                                                  # --missing has no default
        post_mix.main(["--data", str(data), "--out", str(out), "--tokenizer", path])


def test_buckets_stop_at_their_target_and_redistribution_raises_it(tok, data):
    t, _ = tok
    _, small = post_mix.build(str(data), t, 200, "empty")
    _, more = post_mix.build(str(data), t, 200, "redistribute")
    for report in (small, more):
        for info in report["buckets"].values():
            last = max(report["files"][n]["tokens"] for n in info["files"]) if info["rows"] else 0
            assert info["tokens"] < info["tokens_target"] + last + 1                # stops after reaching its target
    assert small["buckets"]["chat"]["tokens_target"] == 70 and small["buckets"]["chat"]["rows"] < 20
    assert small["files"]["tulu3-sft"]["unused"] > 0
    assert more["buckets"]["chat"]["tokens_target"] > 70 and more["buckets"]["code"]["share_planned"] == 0


def test_answers_per_prompt_long_rows_and_late_answers(tok, data):
    t, _ = tok
    _, report = post_mix.build(str(data), t, 1e6, "empty", max_per_prompt=1)
    assert report["files"]["openthoughts3-science"]["rows"] == 1
    why = "more than --max_per_prompt answers to one prompt"
    assert report["files"]["openthoughts3-science"]["dropped"] == {why: 3}
    _, report = post_mix.build(str(data), t, 1e6, "empty", max_len=12, long="drop")
    assert report["files"]["openthoughts3-science"]["dropped"] == {"longer than --max_len": 4}
    _, report = post_mix.build(str(data), t, 1e6, "empty", max_len=12)
    assert report["files"]["openthoughts3-science"]["cut"] == 4
    _, report = post_mix.build(str(data), t, 1e6, "empty", max_len=3)
    assert report["rows"] == 0 and report["dropped"]["answer starts past --max_len"] == 36      # all 36 rows


def test_conversations_that_share_benchmark_text_are_dropped(tok, data, tmp_path):
    t, path = tok
    bench = tmp_path / "bench.jsonl.gz"
    with gzip.open(bench, "wt", encoding="utf-8") as f:
        f.write(json.dumps(dict(text="why sky blue light scatters the cat sat on a mat")) + "\n")
    pack([str(bench)], path, str(tmp_path / "packed"))
    write(data, "smoltalk", [conv("tell me", "why sky blue light scatters the cat sat on a mat"),
                             conv("tell me", "why sky blue")])
    import hashlib
    identity = hashlib.sha256(open(path, "rb").read()).hexdigest()
    benchmarks = post_mix.Benchmarks({"toy": str(tmp_path / "packed")}, 8, identity)
    _, report = post_mix.build(str(data), t, 1e6, "empty", benchmarks=benchmarks)
    assert report["files"]["smoltalk"]["dropped"] == {"shares text with benchmark toy": 1}
    assert report["files"]["smoltalk"]["rows"] == 1 and report["benchmarks"]["names"] == ["toy"]
    with pytest.raises(ValueError, match="another tokenizer"):
        post_mix.Benchmarks({"toy": str(tmp_path / "packed")}, 8, "0" * 64)
