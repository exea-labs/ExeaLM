"""experiments/library/score.py: bits per byte on target tokens, with the library attached or removed or the right entry
pasted, and how often the model's own search finds the source entry. A mock library (experiments/library/build.py
with the mock backend) and tiny random models: no download, no network."""
import importlib.util
import json
import math
import pathlib
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("tokenizers")

from exealm import ExeaLM, get_config  # noqa: E402
from exealm.config import llama_baseline  # noqa: E402
from exealm.data.loader import TokenShards  # noqa: E402
from exealm.model import Llama  # noqa: E402

ROOT = pathlib.Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(f"library_{name}", ROOT / "experiments" / "library" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lb, sc = load("build"), load("score")
TEXT = ["Kyoto", "Photosynthesis", "Mitochondrion", "Albert Einstein", "The Beatles", "Tokyo Tower"]
CODE = ["os.path.join (Python)", "json.load (Python)", "read csv file (Python)", "fs.readFileSync (JavaScript)"]


def build_library(d, text, code, heldout, tokenizer):
    (d / "t.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    lb.main(["write", "--topics", str(d / "t.txt"), "--backend", "mock", "--out", str(d / "e.jsonl")])
    entries = [str(d / "e.jsonl")]
    if code:
        (d / "c.txt").write_text("\n".join(code) + "\n", encoding="utf-8")
        lb.main(["write", "--kind", "code", "--topics", str(d / "c.txt"), "--backend", "mock",
                 "--out", str(d / "c.jsonl")])
        entries.append(str(d / "c.jsonl"))
    lb.main(["morph", *entries, "--backend", "mock", "--heldout", str(heldout), "--out", str(d / "tasks.jsonl")])
    lb.main(["pack", *entries, "--tasks", str(d / "tasks.jsonl"), "--tokenizer", tokenizer, "--heldout", str(heldout),
             "--out", str(d / "lib")])
    return str(d / "lib")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """A mock library packed with a 400-token tokenizer trained here, and a tiny random ExeaLM and Llama for it."""
    from exealm.data.tokenizer import train
    d = tmp_path_factory.mktemp("score")
    with open(d / "corpus.jsonl", "w", encoding="utf-8") as f:
        for t in TEXT + CODE:
            f.write(json.dumps({"text": lb.mock_reply(t) + lb.mock_code_reply(t, "python")}) + "\n")
    train([str(d / "corpus.jsonl")], 400, str(d / "tok"))
    lib = build_library(d, TEXT, CODE, 0.5, str(d / "tok"))
    torch.manual_seed(0)
    cfg = get_config("tiny", vocab_size=400)
    exealm, llama = ExeaLM(cfg), Llama(llama_baseline(cfg))
    paths = dict(exealm=str(d / "exealm.pt"), llama=str(d / "llama.pt"))
    torch.save(dict(model=exealm.state_dict(), cfg=asdict(cfg), kind="tonn", specs=exealm.specs), paths["exealm"])
    torch.save(dict(model=llama.state_dict(), cfg=asdict(llama.cfg), kind="llama", specs=None), paths["llama"])
    return dict(dir=d, lib=lib, tok=str(d / "tok"), **paths)


def test_every_model_gets_its_conditions_on_every_set(setup):
    rows = sc.main([setup["exealm"], setup["llama"], "--library", setup["lib"], "--device", "cpu",
                    "--out", str(setup["dir"] / "score.jsonl")])
    got = {(r["model"], r["set"], r["condition"]) for r in rows}
    assert got == {("ExeaLM", s, c) for s in ("qa", "tasks") for c in ("library", "none", "pasted", "bm25")} | \
                  {("Llama", s, c) for s in ("qa", "tasks") for c in ("none", "pasted", "bm25")}   # no RG: no library
    for r in rows:
        n = sum(1 for _ in open(pathlib.Path(setup["lib"]) / r["set"] / "heldout" / "items.jsonl"))
        assert r["items"] == n and 0 < r["bits_per_byte"] < 20 and math.isfinite(r["bits_per_token"])
        if r["condition"] == "library":
            assert r["searchable"] == n and 0 <= r["search_first"] <= r["search_any"] <= 1
        elif r["condition"] == "bm25":
            assert 0 <= r["bm25_top1"] <= r["bm25_top5"] <= 1 and "search_first" not in r
        else:
            assert "search_first" not in r
    tasks = next(r for r in rows if r["set"] == "tasks" and r["condition"] == "none")
    assert set(tasks["by_kind"]) == {"text", "code"}
    assert [json.loads(line) for line in open(setup["dir"] / "score.jsonl")] == rows


def test_bits_count_the_target_tokens_alone(setup):
    """The batched, padded score equals one row at a time computed by hand, in every condition."""
    lib = TokenShards(setup["lib"])
    items, eot, sep = sc.load_items(setup["lib"], ["qa", "tasks"], "heldout")
    model = sc.build_model(torch.load(setup["exealm"], weights_only=False)).eval()
    C = model.cfg.chunk
    for cond in ("none", "pasted", "library"):
        ds = None
        if cond == "library":
            ds = sc.library_store(lib, model.cfg.rag_chunk_len)
            ds.doc_ids += len(items)
            ds.refresh(model.encoder, model.emb)
        bits = sc.score(model, items, cond, eot, sep, lib, ds, len(items), batch=5)[0]
        for i in (0, len(items) // 2, len(items) - 1):
            entry = sc.entry_ids(lib, items[i]["source_doc"]) if cond == "pasted" else None
            ids, ts = sc.layout(items[i], cond, C, eot, sep, entry)
            assert ids[ts:] == items[i]["target"] and ids[ts - len(items[i]["prefix"]):ts] == items[i]["prefix"]
            if cond == "pasted":
                assert ids[1:1 + len(entry)] == entry
            else:
                assert ts % C == 0 and set(ids[:ts - len(items[i]["prefix"])]) == {eot}
            x = torch.tensor([ids])
            mem = sc.MemoryContext(datastore=ds, doc_ids=torch.full_like(x, i), swarm_size=1)
            with torch.no_grad():
                logits = model(x, mem)[0][0]
            want = F.cross_entropy(logits[ts - 1:-1], x[0, ts:], reduction="sum").item() / math.log(2)
            assert bits[i] == pytest.approx(want, rel=1e-4)


def test_the_search_finds_a_lone_entry_and_scored_rows_never_hide_it(tmp_path, setup):
    """With one entry in the library every search returns its chunks, so the hit rate is 1. Without shifting the
    library's document numbers past the scored rows', row 0 would hide library document 0 from itself."""
    lib_dir = build_library(tmp_path, ["Kyoto"], [], 1.0, setup["tok"])
    rows = sc.main([setup["exealm"], "--library", lib_dir, "--device", "cpu", "--sets", "qa",
                    "--conditions", "library"])
    assert len(rows) == 1 and rows[0]["items"] == 3 and rows[0]["search_first"] == rows[0]["search_any"] == 1.0

    lib = TokenShards(lib_dir)
    items, eot, sep = sc.load_items(lib_dir, ["qa"], "heldout")
    model = sc.build_model(torch.load(setup["exealm"], weights_only=False)).eval()
    ds = sc.library_store(lib, model.cfg.rag_chunk_len)                   # library document 0 is document 0 here
    ds.refresh(model.encoder, model.emb)
    first = sc.score(model, items, "library", eot, sep, lib, ds, offset=0)[1]
    assert first.tolist() == [False, True, True]


def test_the_datastore_holds_every_chunk_of_every_entry(setup):
    lib = TokenShards(setup["lib"])
    ds = sc.library_store(lib, 8)
    for d in range(len(lib.offsets)):
        a = int(lib.offsets[d])
        b = int(lib.offsets[d + 1]) if d + 1 < len(lib.offsets) else len(lib.tokens)
        rows = ds.tokens[ds.doc_ids == d]
        covered = {tuple(r.tolist()) for r in rows}
        for s in list(range(a, b - 7, 8)) + [b - 8]:                       # cut every 8, and the end
            assert tuple(int(t) for t in lib.tokens[s:s + 8]) in covered
    assert len(sc.library_store(lib, 8, n_chunks=50)) == 50                 # or sampled, as pretraining samples


def test_a_model_with_another_vocabulary_is_refused(tmp_path, setup):
    cfg = get_config("tiny", vocab_size=100)
    small = ExeaLM(cfg)
    torch.save(dict(model=small.state_dict(), cfg=asdict(cfg), kind="tonn", specs=small.specs), tmp_path / "s.pt")
    with pytest.raises(SystemExit, match="tokenizer the model was trained with"):
        sc.main([str(tmp_path / "s.pt"), "--library", setup["lib"], "--device", "cpu"])


def test_bm25_finds_the_entry_whose_rare_tokens_the_question_shares(setup):
    """Every held-out item whose prefix shares a token with only its own source entry gets that entry first."""
    lib = TokenShards(setup["lib"])
    items, eot, sep = sc.load_items(setup["lib"], ["qa", "tasks"], "heldout")
    picked, top1, top5 = sc.bm25_pick(lib, items, eot, sep)
    assert len(picked) == len(items) and (top1 <= top5).all()
    owners = {}
    for d in range(len(lib.offsets)):
        for t in set(sc.entry_ids(lib, d)):
            owners.setdefault(t, set()).add(d)
    unique = [i for i, it in enumerate(items)
              if any(owners.get(t) == {it["source_doc"]} for t in set(it["prefix"]) if t not in {eot, *sep})]
    assert all(top5[i] for i in unique)


def test_the_inverted_index_ranks_as_bm25_pick_does(setup, tmp_path):
    """experiments/library/index.py: with every token kept (max_df 1) its scores are bm25_pick's, in one part or two."""
    import numpy as np
    ix = load("index")
    lib = TokenShards(setup["lib"])
    items, eot, sep = sc.load_items(setup["lib"], ["qa", "tasks"], "heldout")
    ix.main(["build", setup["lib"], "--out", str(tmp_path / "one")])
    W = sc.bm25_index(lib)
    idx = ix.Index(str(tmp_path / "one"))
    for it in items:
        q = sorted({t for t in it["prefix"] if t not in {eot, *sep}})
        dense = torch.zeros(W.shape[1])
        dense[q] = 1.0
        want = torch.sparse.mm(W, dense[:, None])[:, 0].numpy()
        d, s = idx.search(it["prefix"], k=3, skip=(eot, *sep))
        np.testing.assert_allclose(s, np.sort(want)[::-1][:3], rtol=1e-5)
        assert want[d[0]] == want.max()
    # Two parts (the same library twice): passage numbers run on, and each part's best is found.
    ix.main(["build", setup["lib"], setup["lib"], "--out", str(tmp_path / "two")])
    two = ix.Index(str(tmp_path / "two"))
    assert two.n == 2 * len(lib.offsets)
    d, s = two.search(items[0]["prefix"], k=2, skip=(eot, *sep))
    assert {int(x) % len(lib.offsets) for x in d} == {int(idx.search(items[0]["prefix"], k=1, skip=(eot, *sep))[0][0])}


def test_rag_pack_puts_the_searched_entry_before_every_item(setup):
    rag_pack = load("rag_pack")
    out = setup["dir"] / "rag"
    rag_pack.main(["--library", setup["lib"], "--split", "heldout", "--device", "cpu", "--out", str(out)])
    pack, lib = TokenShards(str(out)), TokenShards(setup["lib"])
    items, eot, sep = sc.load_items(setup["lib"], ["qa", "tasks"], "heldout")
    picked, _, _ = sc.bm25_pick(lib, items, eot, sep)
    assert pack.meta["n_docs"] == len(items) and pack.meta["eot_id"] == eot
    for i in (0, len(items) - 1):
        a = int(pack.offsets[i])
        want = sc.entry_ids(lib, int(picked[i])) + list(sep) + items[i]["prefix"] + items[i]["target"] + [eot]
        assert [int(t) for t in pack.tokens[a:a + len(want)]] == want


def test_deepseek_v4_chat_mode_prompt():
    """DeepSeek-V4's tokenizer has no chat template; build.chat_text writes its chat mode as its encoding README does."""
    class Tok:
        chat_template = None

        def get_vocab(self):
            return {"<｜User｜>": 1, "<｜Assistant｜>": 2}
    msgs = [dict(role="system", content="Be brief."), dict(role="user", content="What is 2+2?")]
    assert lb.chat_text(Tok(), msgs) == "<｜begin▁of▁sentence｜>Be brief.<｜User｜>What is 2+2?<｜Assistant｜></think>"
    two = msgs + [dict(role="assistant", content="4."), dict(role="user", content="And 3+3?")]
    assert lb.chat_text(Tok(), two).endswith("</think>4.<｜end▁of▁sentence｜><｜User｜>And 3+3?<｜Assistant｜></think>")

    class Plain:
        chat_template = None

        def get_vocab(self):
            return {"a": 1}
    with pytest.raises(SystemExit):
        lb.chat_text(Plain(), msgs)


def test_pool_topics_take_each_documents_first_passage_by_quota(tmp_path):
    """experiments/library/pool_topics.py: one topic per document (its first passage), quotas per source, ML arXiv first."""
    import gzip
    pt = load("pool_topics")
    part = tmp_path / "part_000"
    part.mkdir()
    rows = [dict(source="arxiv", chunk=0, title="Adam: a method", categories="cs.LG"),
            dict(source="arxiv", chunk=1, title="Adam: a method", categories="cs.LG"),
            dict(source="arxiv", chunk=0, title="Galois theory notes", categories="math.GR"),
            dict(source="wikipedia_en", chunk=0, title="Kyoto"),
            dict(source="wikipedia_en", chunk=0, title="kyoto"),                     # the same title: skipped
            dict(source="starcoder", chunk=0, title="main.py")]                      # not in the quota
    with open(part / "index.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with gzip.open(part / "passages.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(dict(text=f"{r['title']}\nbody {i}")) + "\n" for i, r in enumerate(rows))
    out = tmp_path / "topics.jsonl"
    pt.main([str(part), "--out", str(out), "--quota", "arxiv:1,wikipedia_en:5"])
    got = [json.loads(l) for l in open(out)]
    assert sorted(r["topic"] for r in got) == ["Adam: a method", "Kyoto"]          # ML first; one Kyoto
    assert next(r for r in got if r["topic"] == "Adam: a method")["source"] == "Adam: a method\nbody 0"


def test_frontier_prompt_asks_for_an_expert_entry_that_never_cites_its_source():
    m = lb.messages_for("Adam (optimizer)", "en", 250, 450, source="Adam is a method ...", style="frontier")
    assert m[0]["content"] == lb.FRONTIER_SYSTEM and "Never mention the document" in m[1]["content"]
    assert "Adam is a method" in m[1]["content"]
    g = lb.messages_for("Adam (optimizer)", "en", 250, 450, source="Adam is a method ...")
    assert g[0]["content"] == lb.GROUNDED_SYSTEM


def test_tiers_search_each_index_apart(setup, tmp_path, capsys):
    ix = load("index")
    ix.main(["build", setup["lib"], "--out", str(tmp_path / "a")])
    ix.main(["build", setup["lib"], "--out", str(tmp_path / "b"), "--base", "1000"])
    ix.main(["tiers", "--tiers", f"written={tmp_path / 'a'};human={tmp_path / 'b'}", "--library", setup["lib"],
             "--max_df", "1.0"])
    out = capsys.readouterr().out
    assert "written (" in out and "human (" in out and "in either" in out


def test_compact_index_ranks_as_the_full_one(setup, tmp_path):
    """index.py compact: common tokens dropped, 8-bit weights, 3-byte passage numbers; with max_df at the cut the
    top passage is the full index's for nearly every query, and the scores agree to within the 8-bit step."""
    import numpy as np
    ix = load("index")
    ix.main(["build", setup["lib"], "--out", str(tmp_path / "full")])
    ix.main(["compact", str(tmp_path / "full"), "--out", str(tmp_path / "small"), "--max_df", "0.5"])
    full, small = ix.Index(str(tmp_path / "full")), ix.Index(str(tmp_path / "small"))
    items, eot, sep = sc.load_items(setup["lib"], ["qa", "tasks"], "heldout")
    same = 0
    for it in items:
        d1, s1 = full.search(it["prefix"], k=1, max_df=0.5, skip=(eot, *sep))
        d2, s2 = small.search(it["prefix"], k=1, max_df=0.5, skip=(eot, *sep))
        same += int(len(d1) and len(d2) and d1[0] == d2[0])
        if len(d1) and len(d2) and d1[0] == d2[0]:
            np.testing.assert_allclose(s2, s1, rtol=0.05, atol=small.parts[0].scale * 8)
    assert same >= 0.9 * len(items)
    assert small.parts[0].meta["n_postings"] <= full.parts[0].meta["n_postings"]
    assert small.passage(0) == full.passage(0)


def test_gpu_search_paths_rank_as_the_cpu_search(setup, tmp_path):
    """Index.to / search_gpu / search_gpu_batch, run here on CPU tensors: the same top passage as search(), for the
    full and the compact format, one query at a time or in a batch."""
    ix = load("index")
    ix.main(["build", setup["lib"], "--out", str(tmp_path / "full")])
    ix.main(["compact", str(tmp_path / "full"), "--out", str(tmp_path / "small"), "--max_df", "0.5"])
    items, eot, sep = sc.load_items(setup["lib"], ["qa", "tasks"], "heldout")
    skip = (eot, *sep)
    for name in ("full", "small"):
        cpu, dev = ix.Index(str(tmp_path / name)), ix.Index(str(tmp_path / name)).to("cpu")
        batch = dev.search_gpu_batch([it["prefix"] for it in items], k=3, max_df=0.5, skip=skip)
        for it, (bd, bs) in zip(items, batch):
            d, s = cpu.search(it["prefix"], k=3, max_df=0.5, skip=skip)
            g, gs = dev.search_gpu(it["prefix"], k=3, max_df=0.5, skip=skip)
            if len(d):
                assert abs(gs[0] - s[0]) < 1e-3 * max(1, s[0]) and abs(bs[0] - s[0]) < 1e-3 * max(1, s[0])
