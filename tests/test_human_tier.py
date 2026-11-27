"""experiments/library/human_tier.py: passage cutting, the quality filter, and plan-free runs of index -> passages ->
pack on tiny local files with a tiny tokenizer (no network)."""
import gzip
import json
import os
import random

import numpy as np
import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("pyarrow")

from experiments.library import human_tier as ht  # noqa: E402


def test_special_tokens_match_the_tokenizer():
    from exealm.data.tokenizer import SPECIAL
    assert ht.SPECIAL == SPECIAL


def words(rng, n):
    return " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]) + str(rng.randint(0, 99))
                    for _ in range(n)) + "."


def test_passages_cut_at_paragraphs_in_order_and_cover_the_document():
    rng = random.Random(0)
    paras = [words(rng, rng.randint(5, 400)) for _ in range(60)]
    body = "\n\n".join(paras)
    parts = ht.chunk(body, "A title", 4.0, max_per_doc=1000)
    assert len(parts) > 3
    sizes = [ht.nbytes(p) / 4.0 for p in parts]
    assert all(s <= ht.MERGE_MAX for s in sizes)
    assert all(s >= ht.UNIT for s in sizes[:-1])                  # only the last one may be short
    assert "".join("".join(parts).split()) == "".join(body.split())
    starts = [body.index(p) for p in parts]
    assert starts == sorted(starts)
    for p in parts:                                               # whole paragraphs (none here is longer than a unit)
        assert p.split("\n\n")[0] in paras and p.split("\n\n")[-1] in paras
    assert len(ht.chunk(body, "A title", 4.0, max_per_doc=2)) == 2


def test_a_long_paragraph_is_cut_at_lines_then_sentences_then_spaces():
    rng = random.Random(1)
    body = " ".join(words(rng, 30) for _ in range(200))           # one paragraph, no newline
    parts = ht.chunk(body, "T", 4.0, max_per_doc=1000)
    assert len(parts) > 3 and all(p.endswith(".") for p in parts[:-1])
    blob = "x" * 20000
    assert all(ht.nbytes(p) / 4.0 <= ht.MERGE_MAX for p in ht.chunk(blob, "T", 4.0, max_per_doc=1000))
    code = "\n".join(f"    x{i} = f(x{i - 1})" for i in range(2000))
    parts = ht.chunk(code, "repo/file.py", 3.0, max_per_doc=1000)
    assert all(p.startswith("    x") for p in parts)              # cut between lines; indentation kept


def test_quality_filter():
    ok = "This is a normal paragraph of English text about some topic. " * 10
    assert ht.quality(ok, "prose") is None
    assert ht.quality(ok + "<|endoftext|>", "prose") == "special-token string"
    assert ht.quality("1 2 3 4 5 6 7 8 9 " * 40, "prose") == "mostly non-text"
    assert ht.quality(ok + "QUJD" * 100, "prose") == "blob"
    assert ht.quality("\n".join(["the same navigation line here"] * 10), "prose") == "repeated lines"
    assert ht.quality("x" * 10, "prose") == "near-empty"
    assert ht.quality("a = 1\n" * 50 + "b" * 1200, "code") == "minified"


def test_licenses():
    assert ht.arxiv_license("http://creativecommons.org/licenses/by/4.0/") == "CC-BY-4.0"
    assert ht.arxiv_license("http://creativecommons.org/licenses/by-sa/4.0/") == "CC-BY-SA-4.0"
    assert ht.arxiv_license("http://creativecommons.org/publicdomain/zero/1.0/") == "CC0-1.0"
    for bad in ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/", "http://creativecommons.org/licenses/by-nc-sa/4.0/",
                "http://creativecommons.org/licenses/by-nc-nd/4.0/", None):
        assert ht.arxiv_license(bad) is None
    rec = dict(text="<reponame>a/b<filename>x.py\nimport numpy as np\n", metadata=dict(max_stars_repo_licenses="['MIT']"))
    title, body, info = ht.read_code(rec, ht.SOURCES["research_python"], {})
    assert (title, body, info["license"]) == ("a/b/x.py", "import numpy as np\n", "MIT")
    with pytest.raises(ht.Skip):
        ht.read_code(dict(rec, metadata=dict(max_stars_repo_licenses="['GPL-3.0']")), ht.SOURCES["starcoder"], {})
    with pytest.raises(ht.Skip):
        ht.read_code(dict(text="import os\n", metadata={}), ht.SOURCES["research_python"], {})


# ---------------------------------------------------------------------------------------------------------------------
# end to end on local files

def se_doc(rng, i, text=None):
    q = words(rng, 40)
    return dict(id=f"x-{i}", text=text or f"Question {i} about things?\n\n{q}\n\n" + "\n\n".join(
        words(rng, rng.randint(20, 120)) for _ in range(rng.randint(3, 12))),
        metadata=dict(forum="math_stackexchange_com", question_id=i, question_content_license="CC BY-SA 3.0",
                      answer_content_license="CC BY-SA 4.0"))


def write_pack(tmp, name, texts, tok_dir):
    from exealm.data.pack import pack
    src = os.path.join(tmp, f"{name.replace('/', '_')}.jsonl.gz")
    with gzip.open(src, "wt", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps(dict(text=t)) + "\n")
    pack([src], tok_dir, os.path.join(tmp, name))


@pytest.fixture
def world(tmp_path):
    from exealm.data.tokenizer import train
    rng = random.Random(2)
    docs = [se_doc(rng, i) for i in range(400)]
    val_text = docs[4]["text"]                                  # a val document that a pool document copies
    docs.append(se_doc(rng, 400, text=docs[7]["text"]))         # an exact duplicate
    sample = tmp_path / "sample.jsonl.gz"
    with gzip.open(sample, "wt", encoding="utf-8") as f:
        for d in docs[:100]:
            f.write(json.dumps(dict(text=d["text"])) + "\n")
    slice_dir = tmp_path / "slice"
    tok_dir = slice_dir / "tokenizer"
    train([str(sample)], 600, str(tok_dir))
    write_pack(tmp_path, "slice/packed/stackexchange/val", [val_text], str(tok_dir))
    write_pack(tmp_path, "slice/packed/stackexchange/train", [words(rng, 500) for _ in range(5)], str(tok_dir))
    write_pack(tmp_path, "libv1/qa/heldout", ["Question: what is it?\nAnswer: nothing at all, really."], str(tok_dir))
    write_pack(tmp_path, "libv1/tasks/heldout", ["Rewrite it.\n\nSomething else entirely."], str(tok_dir))
    root = tmp_path / "root"
    raw = root / "raw" / "allenai/dolmino-mix-1124" / "data/stackexchange"
    raw.mkdir(parents=True)
    files = []
    for k in range(2):
        p = raw / f"se-{k}.json.gz"
        with gzip.open(p, "wt", encoding="utf-8") as f:
            for d in docs[k::2]:
                f.write(json.dumps(d) + "\n")
        files.append(dict(path=f"data/stackexchange/se-{k}.json.gz", size=os.path.getsize(p), sha256=None))
    ht.write_json(str(root / "manifest" / "order" / "stackexchange.json"),
                  dict(source="stackexchange", repo="allenai/dolmino-mix-1124", revision="local", files=files))
    return dict(root=str(root), slice=str(slice_dir), libv1=str(tmp_path / "libv1"), docs=docs)


def run(world, step, *extra):
    ht.main([step, "--root", world["root"], "--slice", world["slice"], "--libv1", world["libv1"], "--workers", "1",
             "--sources", "stackexchange", *extra])


def test_end_to_end_parts_continue_and_the_pack_is_clean(world):
    from exealm.data.loader import TokenShards
    from tokenizers import Tokenizer
    ht.SOURCES["stackexchange"]["share"] = 1.0                  # the whole part from this source
    try:
        run(world, "index")
        run(world, "passages", "--part", "0", "--part_tokens", "20000")
        run(world, "pack", "--part", "0", "--allow_short")
        run(world, "passages", "--part", "1", "--part_tokens", "1e9", "--allow_short")
        run(world, "pack", "--part", "1", "--allow_short")
    finally:
        ht.SOURCES["stackexchange"]["share"] = 0.10
    tok = Tokenizer.from_file(os.path.join(world["slice"], "tokenizer", "tokenizer.json"))
    seen_text = []
    for part in (0, 1):
        out = os.path.join(world["root"], "human_pool", f"part_{part:03d}")
        sh = TokenShards(out)
        idx = [json.loads(x) for x in open(os.path.join(out, "index.jsonl"), encoding="utf-8")]
        with gzip.open(os.path.join(out, "passages.jsonl.gz"), "rt", encoding="utf-8") as f:
            txt = [json.loads(x) for x in f]
        assert len(idx) == len(txt) == sh.meta["n_docs"] == len(sh.offsets) > 0
        ends = list(sh.offsets[1:]) + [len(sh.tokens)]
        for i, (r, t) in enumerate(zip(idx, txt)):
            assert r["doc"] == i and r["id"] == t["id"]
            ids = sh.tokens[sh.offsets[i]:ends[i]]
            assert ids[-1] == 0 and 0 not in ids[:-1] and len(ids) == r["n_tokens"]
            assert tok.decode(ids[:-1].tolist()) == t["text"]
            assert t["text"].startswith(r["title"] + "\n")
            assert r["license"] == "CC-BY-SA-3.0+CC-BY-SA-4.0" and r["url"].startswith("https://math.stackexchange.com/")
            seen_text.append(t["text"])
        stats = json.load(open(os.path.join(out, "stats.json")))["stackexchange"]["stats"]
        if part == 0:
            assert stats["dropped: val"] >= 1 and sh.meta["n_tokens"] >= 20000
            assert stats.get("dropped: duplicate", 0) == 0
            assert os.path.realpath(os.path.join(world["root"], "human_pool_v0")) == os.path.realpath(out)
        else:
            assert stats["dropped: duplicate"] >= 1
    assert len(seen_text) == len(set(seen_text))                     # parts never repeat a passage
    val_doc = world["docs"][4]["text"]
    assert not any(val_doc.split("\n\n")[1] in t for t in seen_text)  # the val document's text is not in the pool
    d0 = json.load(open(os.path.join(world["root"], "work", "part_000", "stackexchange", "done.json")))
    d1 = json.load(open(os.path.join(world["root"], "work", "part_001", "stackexchange", "done.json")))
    assert d1["start"] == d0["end"] and d1["end"] == dict(file=2, row=0)
