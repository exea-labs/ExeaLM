"""experiments/data_mix end to end on tiny local fixtures: a few hundred invented documents per bucket, laid out like
the Hugging Face and LLM-jp hosts, so nothing is downloaded. One module-scoped run of every step; the tests check
its outputs."""
import gzip
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("tokenizers")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                                       # CI runs bare pytest: make `experiments` importable

from exealm.data.benchmarks import export_rows  # noqa: E402
from exealm.data.loader import TokenShards, open_data  # noqa: E402
from experiments.data_mix import config  # noqa: E402
from experiments.data_mix.extract import doc_hash  # noqa: E402

WORDS = ("model data token memory signal circuit layer window search entry paper method result proof theorem graph "
         "number value system network energy field river mountain city history language science study measure "
         "train test sample batch error gradient matrix vector kernel buffer stream query answer question").split()
KANA = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
KANJI = "日本語学研究大東京文化時間世界自然科学技術情報社会経済歴史言語数学物理化生命"
BENCH_Q = ("Natalia sold clips to 48 of her friends in April and then she sold half as many clips in May "
           "how many clips did Natalia sell altogether in April and May")


def en_doc(rng, n_words):
    words = [rng.choice(WORDS) for _ in range(n_words)]
    sents, i = [], 0
    while i < len(words):
        k = rng.randint(6, 14)
        sents.append(" ".join(words[i:i + k]).capitalize() + ".")
        i += k
    return " ".join(sents)


def ja_doc(rng, n_chars):
    out = []
    for i in range(n_chars):
        out.append(rng.choice(KANJI if rng.random() < 0.4 else KANA))
        if i % 23 == 22:
            out.append("。" if rng.random() < 0.6 else "、")
    return "".join(out) + "。"


def code_doc(rng, n_funcs):
    lines = []
    for i in range(n_funcs):
        a, b = rng.choice(WORDS), rng.choice(WORDS)
        lines += [f"def {a}_{b}_{i}(x, y={rng.randint(0, 9)}):", f"    \"\"\"Return the {a} of the {b}.\"\"\"",
                  f"    if x > {rng.randint(0, 99)}:", f"        return [v * {rng.randint(2, 7)} for v in range(x)]",
                  f"    return {{'{a}': x, '{b}': y}}", ""]
    return "\n".join(lines)


def write_jsonl_gz(path, texts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"text": t, "meta": {"id": f"doc-{i}", "url": f"https://example.org/{i}"}},
                               ensure_ascii=False) + "\n")


def write_parquet(path, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.table(cols), path, row_group_size=64)


def make_sources(root, rng):
    """Local copies laid out like the hosts. Returns the planted texts the tests look for."""
    llmjp, hf = os.path.join(root, "llmjp"), os.path.join(root, "hf")
    shared = en_doc(rng, 120)                                   # in en_wiki and in txt360_web: a cross-bucket duplicate
    within = en_doc(rng, 90)                                    # twice in en_wiki, once with other spacing and case
    contaminated = en_doc(rng, 40) + " " + BENCH_Q + " " + en_doc(rng, 40)
    special = en_doc(rng, 60) + " <|endoftext|> " + en_doc(rng, 20)
    for k in range(2):
        docs = [en_doc(rng, rng.randint(60, 400)) for _ in range(150)]
        docs += [en_doc(rng, 1200) for _ in range(20)]          # long documents
        if k == 0:
            docs += [shared, within, "  " + within.upper().replace(" ", "   ") + "\n", contaminated, special]
        write_jsonl_gz(os.path.join(llmjp, "en/en_wiki", f"{k:04d}.jsonl.gz"), docs)
    for k in range(2):
        write_jsonl_gz(os.path.join(llmjp, "ja/ja_wiki", f"{k:04d}.jsonl.gz"),
                       [ja_doc(rng, rng.randint(150, 900)) for _ in range(150)] + [ja_doc(rng, 3000) for _ in range(15)])
    write_jsonl_gz(os.path.join(llmjp, "code/code_stack", "0000.jsonl.gz"),
                   [code_doc(rng, rng.randint(2, 12)) for _ in range(200)] + [code_doc(rng, 40) for _ in range(12)])
    n = 300
    texts = [en_doc(rng, rng.randint(60, 400)) for _ in range(n - 12)] + [en_doc(rng, 1200) for _ in range(12)]
    texts[5] = shared
    lang = ["en"] * n
    lang[7], lang[8] = None, "fr"
    subset = ["commoncrawl"] * n
    subset[9] = "wikipedia"
    write_parquet(os.path.join(hf, "IFM/TxT360-v2/web-high-medium/Medium-High.chunk0-00000.parquet"), {
        "text": texts, "subset": subset, "quality_category": ["Medium-High"] * n,
        "fineweb_edu_classifier_score": [rng.random() * 3 for _ in range(n)], "meta.lang": lang,
        "meta.token_count": [len(t) // 4 for t in texts]})
    for sub in ("code-no-thinking", "codeio"):
        for k in range(2):
            t = [code_doc(rng, rng.randint(2, 10)) for _ in range(120)] + [code_doc(rng, 40) for _ in range(8)]
            write_parquet(os.path.join(hf, "IFM/Code-Reasoning", sub, f"{sub}.chunk0-{k:05d}.parquet"),
                          {"text": t, "token_count": [len(x) // 3 for x in t]})
    t = [en_doc(rng, 80) + "\n\n" + code_doc(rng, rng.randint(2, 12)) for _ in range(150)] + \
        [code_doc(rng, 40) for _ in range(10)]
    write_parquet(os.path.join(hf, "IFM/Code-Reasoning/code-thinking-v1/think.chunk0-00000.parquet"),
                  {"text": t, "token_count": [len(x) // 3 for x in t]})
    return dict(shared=shared, within=within, contaminated=contaminated, special=special)


def small_config(work):
    def bucket(name, repo, target, parts, lic, lang, bpt, **kw):
        return config.bucket(name, repo, target, parts, lic, lang, bpt, "test credit", f"{name} test text", **kw)
    cfg = config.load()
    cfg["seed"] = 7
    cfg["buckets"] = [
        bucket("en_wiki", "pretrain", 15000, [config.llmjp("en/en_wiki")], "CC-BY-SA-3.0", "en", 4.5, share_alike=True,
               anneal_weight=2.0),
        bucket("txt360_web", "pretrain", 15000,
               [config.hf("IFM/TxT360-v2", "web-high-medium/*.parquet", filter=config.TXT360_WEB_FILTER,
                          meta_columns=["subset", "meta.lang"])], "CC-BY-4.0", "en", 4.5, common_crawl=True,
               anneal_weight=0.5),
        bucket("code_reasoning", "pretrain", 20000,
               config.subsets("IFM/Code-Reasoning", ["code-no-thinking", "codeio"]),
               "Apache-2.0", "en+code", 3.0),
        bucket("ja_wiki", "pretrain", 15000, [config.llmjp("ja/ja_wiki")], "CC-BY-SA-3.0", "ja", 4.0, share_alike=True),
        bucket("code_stack", "pretrain_stack", 10000, [config.llmjp("code/code_stack")],
               "The Stack v1 terms; each file keeps its repository's license", "code", 3.0, in_tokenizer_sample=False),
        bucket("post_code", "post", 10000, config.subsets("IFM/Code-Reasoning", ["code-thinking-v1"]), "Apache-2.0",
               "en+code", 3.0),
    ]
    cfg["split"] = dict(val_tokens=1500, val_long_docs=6, val_long_min=4, long_tokens=8 * 32 + 1)
    cfg["fetch"] = dict(margin=0.3, min_files=1)
    cfg["tokenizer"] = dict(vocab_size=600, measure_mb=0.05,
                            sample_mb=dict(en_wiki=0.05, txt360_web=0.05, code_reasoning=0.05, ja_wiki=0.08))
    cfg["decontam"] = dict(ngram=13, min_matches=1, benchmarks=["gsm8k", "humaneval"])
    cfg["raw"] = dict(mb_per_file=0.05)
    path = os.path.join(work, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return cfg, path


def make_benchmarks(work):
    gsm = [dict(question=BENCH_Q, answer="She sold 48 / 2 = 24 clips in May. 48 + 24 = 72. #### 72"),
           dict(question="A robe takes 2 bolts of blue fiber and half that much white fiber.", answer="#### 3")]
    he = [dict(task_id="HumanEval/0", prompt="def has_close_elements(numbers, threshold):\n",
               canonical_solution="    return any(abs(a - b) < threshold for a in numbers for b in numbers)\n",
               test="def check(candidate):\n    assert candidate([1.0, 2.0], 0.5) is False\n")]
    export_rows("gsm8k", iter(gsm), os.path.join(work, "benchmarks", "gsm8k"), "fixture-rev")
    export_rows("humaneval", iter(he), os.path.join(work, "benchmarks", "humaneval"), "fixture-rev")


def run_cli(*args):
    r = subprocess.run([sys.executable, "-m", "experiments.data_mix.pipeline", *args], cwd=ROOT, capture_output=True,
                       text=True, env=dict(os.environ, HF_HUB_OFFLINE="1"))      # tests never touch the network
    assert r.returncode == 0, r.stdout + r.stderr
    return r


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("mix")
    work = str(root / "work")
    os.makedirs(work)
    planted = make_sources(str(root / "src"), random.Random(0))
    cfg, cfg_path = small_config(work)
    make_benchmarks(work)
    os.makedirs(os.path.join(work, "manifest"))
    with open(os.path.join(work, "manifest", "the_stack_terms.md"), "w", encoding="utf-8") as f:
        f.write("## Terms of Use for The Stack\n\n1. Fixture text: the real run copies the official terms here.\n")
    common = ["--work", work, "--config", cfg_path, "--workers", "2", "--local_hf", str(root / "src" / "hf"),
              "--local_llmjp", str(root / "src" / "llmjp")]
    for step in ("fetch", "dedup", "tokenizer", "benchmarks", "tokenize"):
        run_cli(step, *common, "--chunk_mb", "0.02")
    run_cli("assemble", *common, "--raw")
    run_cli("cards", *common)
    run_cli("verify", *common, "--decontam")
    return dict(work=work, cfg=cfg, cfg_path=cfg_path, planted=planted, common=common,
                repos={r: os.path.join(work, "repos", cfg["repos"][r]["repo_id"].split("/")[-1]) for r in cfg["repos"]})


def raw_texts(repo_dir, bucket, split):
    files = sorted(os.listdir(os.path.join(repo_dir, "raw", bucket, split)))
    return [t for f in files for t in pq.read_table(os.path.join(repo_dir, "raw", bucket, split, f)).column("text")
            .to_pylist()]


def all_raw(built):
    out = {}
    for b in config.buckets(built["cfg"]):
        for split in ("train", "val"):
            out[(b["name"], split)] = raw_texts(built["repos"][b["repo"]], b["name"], split)
    return out


def test_packs_open_with_token_shards_and_match_raw_text(built):
    from exealm.data.tokenizer import load
    tok = load(os.path.join(built["repos"]["pretrain"], "tokenizer"))
    for b in config.buckets(built["cfg"]):
        for split in ("train", "val"):
            d = os.path.join(built["repos"][b["repo"]], "packed", b["name"], split)
            s = TokenShards(d)
            assert s.meta["dtype"] == "uint16" and s.meta["vocab_size"] == 600 == tok.get_vocab_size()
            assert len(s.tokens) == s.meta["n_tokens"] and len(s.offsets) == s.meta["n_docs"] > 0
            texts = raw_texts(built["repos"][b["repo"]], b["name"], split)
            assert len(texts) == len(s.offsets)
            ends = list(s.offsets[1:]) + [len(s.tokens)]
            for i in range(0, len(texts), max(1, len(texts) // 10)):
                assert tok.decode(list(s.tokens[s.offsets[i]:ends[i] - 1])) == texts[i]
                assert s.tokens[ends[i] - 1] == tok.token_to_id("<|endoftext|>")
            x, y, docs = s.batch(4, 16, np.random.default_rng(0))
            assert x.shape == (4, 16) and (y[:, :-1] == x[:, 1:]).all()


def test_training_split_meets_target_and_val_has_long_documents(built):
    for b in config.buckets(built["cfg"]):
        tr = TokenShards(os.path.join(built["repos"][b["repo"]], "packed", b["name"], "train"))
        va = TokenShards(os.path.join(built["repos"][b["repo"]], "packed", b["name"], "val"))
        assert b["target"] <= tr.meta["n_tokens"] < b["target"] * 1.2, b["name"]
        assert va.meta["long_docs"] >= built["cfg"]["split"]["val_long_min"], b["name"]
        lens = np.diff(np.append(va.offsets, len(va.tokens)))
        assert (lens - 1 >= built["cfg"]["split"]["long_tokens"]).sum() == va.meta["long_docs"]


def test_mix_json_opens_with_open_data_and_weights_are_targets(built, tmp_path):
    for repo, names in (("pretrain", ["en_wiki", "txt360_web", "code_reasoning", "ja_wiki", "code_stack"]),
                        ("post", ["post_code"])):
        for mix_name in ("mix.json", "mix_val.json") + (("mix_anneal.json",) if repo == "pretrain" else ()):
            spec = json.load(open(os.path.join(built["repos"][repo], mix_name)))
            assert list(spec["sources"]) == names
            targets = {b["name"]: b["target"] / 1e9 * (b["anneal_weight"] if "anneal" in mix_name else 1)
                       for b in built["cfg"]["buckets"]}
            assert all(abs(s["weight"] - targets[n]) < 1e-9 for n, s in spec["sources"].items())
            for s in spec["sources"].values():                 # data/<repo name>/... -> the built folders
                assert s["paths"].startswith("data/exealm-")
                s["paths"] = os.path.join(built["work"], "repos", s["paths"].split("/", 1)[1])
            local = tmp_path / f"{repo}_{mix_name}"
            local.write_text(json.dumps(spec))
            mix = open_data(str(local))
            assert mix.names == names
            x, y, d = mix.batch(8, 16, np.random.default_rng(1))
            assert x.shape == (8, 16)
    public = json.load(open(os.path.join(built["repos"]["pretrain"], "mix_public.json")))
    assert "code_stack" not in public["sources"]


def test_tokenizer_round_trips_english_japanese_and_code(built):
    from exealm.data.tokenizer import SPECIAL, load
    tok = load(os.path.join(built["repos"]["pretrain"], "tokenizer", "tokenizer.json"))
    assert tok.get_vocab_size() <= 65535
    assert [tok.token_to_id(s) for s in SPECIAL] == list(range(len(SPECIAL)))      # exealm.data.tokenizer's order
    for s in ("The quick brown fox jumps over the lazy dog.", "こんにちは、世界。今日は良い天気ですね。",
              "def f(x):\n    return {'a': [i ** 2 for i in range(x)]}\n\t# tab", "数学 and English 混在 text 123"):
        assert tok.decode(tok.encode(s).ids) == s
    rep = json.load(open(os.path.join(built["work"], "reports", "tokenizer.json")))
    assert all(v["ok"] for v in rep["roundtrip"].values())
    assert all(p["roundtrip_failures"] == 0 for p in rep["parts"].values())
    assert "code_stack" not in rep["sample_bytes"]                 # the gated bucket is not in the public tokenizer


def test_splits_do_not_overlap_and_duplicates_are_removed(built):
    texts = all_raw(built)
    where = {}
    for (bucket, split), ts in texts.items():
        for t in ts:
            where.setdefault(doc_hash(t), []).append((bucket, split))
    assert all(len(v) == 1 for v in where.values())             # every text once: no train/val or bucket overlap
    p = built["planted"]
    assert len(where.get(doc_hash(p["within"]), [])) <= 1       # copies with other spacing and case are one text
    shared = where.get(doc_hash(p["shared"]), [])
    assert all(b == "en_wiki" for b, _ in shared)               # the first bucket keeps the cross-bucket copy
    rep = json.load(open(os.path.join(built["work"], "reports", "dedup.json")))
    assert rep["parts"]["en_wiki.0.en_wiki"]["duplicates"] >= 1
    assert rep["parts"]["txt360_web.0.web-high-medium"]["duplicates_of"] == {"en_wiki.0.en_wiki": 1}


def test_filters_and_decontamination(built):
    texts = [t for ts in all_raw(built).values() for t in ts]
    assert not any("<|endoftext|>" in t for t in texts)
    assert not any("Natalia sold clips" in t for t in texts)
    info = [json.load(open(os.path.join(d, f))) for d in
            [os.path.join(built["work"], "extract", "txt360_web.0.web-high-medium")] for f in os.listdir(d)
            if f.endswith(".json")]
    dropped = info[0]["dropped"]
    assert dropped == {"language": 2, "subset wikipedia": 1}
    st = json.load(open(os.path.join(built["work"], "reports", "assemble", "en_wiki.json")))
    got = sum(st["splits"][s]["contaminated_dropped"].get("gsm8k", 0) for s in ("train", "val"))
    selected = json.load(open(os.path.join(built["work"], "reports", "tokenize.json")))["contaminated"]
    assert selected >= got
    ver = json.load(open(os.path.join(built["work"], "reports", "verify.json")))
    assert not ver["problems"]
    assert all(v["decontam"]["dropped"] == 0 for v in ver["packs"].values())   # exealm.data.decontam agrees


def test_rerunning_a_step_changes_nothing(built):
    before = {}
    for d, _, fs in os.walk(os.path.join(built["work"], "tok")):
        for f in fs:
            before[os.path.join(d, f)] = os.path.getmtime(os.path.join(d, f))
    run_cli("fetch", *built["common"])
    run_cli("tokenize", *built["common"], "--chunk_mb", "0.02")
    after = {os.path.join(d, f): os.path.getmtime(os.path.join(d, f))
             for d, _, fs in os.walk(os.path.join(built["work"], "tok")) for f in fs}
    assert before == after                                      # nothing was re-tokenized


def test_upload_dry_run_lists_the_repo_files(built):
    r = subprocess.run([sys.executable, "-m", "experiments.data_mix.upload", "--work", built["work"], "--config",
                        built["cfg_path"], "--dry-run"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    for repo in built["cfg"]["repos"].values():
        assert repo["repo_id"] in out
    pre = built["repos"]["pretrain"]
    expected = sorted(os.path.relpath(os.path.join(d, f), pre).replace(os.sep, "/")
                      for d, _, fs in os.walk(pre) for f in fs if ".cache" not in d)
    for rel in expected:
        assert rel in out
    assert "tokenizer/tokenizer.json" in expected and "README.md" in expected and "mix.json" in expected
    assert "packed/code_stack/train/tokens.bin" not in expected      # The Stack is in its own repo
    stack = built["repos"]["pretrain_stack"]
    assert os.path.exists(os.path.join(stack, "packed", "code_stack", "train", "tokens.bin"))
    assert "gated" in out and "would check" in out and "create_repo" not in out


def test_cards_credit_sources_and_licenses(built):
    card = open(os.path.join(built["repos"]["pretrain"], "README.md"), encoding="utf-8").read()
    for s in ("CC-BY-SA-3.0", "CC-BY-4.0", "Apache-2.0", "Common Crawl", "Share-alike", "en/en_wiki", "ja/ja_wiki",
              "IFM/TxT360-v2", "Decontamination", "gsm8k", "Source files used", "hf download"):
        assert s in card, s
    stack = open(os.path.join(built["repos"]["pretrain_stack"], "README.md"), encoding="utf-8").read()
    assert "extra_gated_prompt" in stack and "Opt-out requests are honoured" in stack
    assert "Terms of Use for The Stack" in stack
    yaml = pytest.importorskip("yaml")
    for repo in ("pretrain", "pretrain_stack", "post", "library"):
        text = open(os.path.join(built["repos"][repo], "README.md"), encoding="utf-8").read()
        meta = yaml.safe_load(text.split("---\n")[1])
        assert meta["pretty_name"].startswith("ExeaLM")
    assert "Terms of Use for The Stack" in yaml.safe_load(stack.split("---\n")[1])["extra_gated_prompt"]
    post = open(os.path.join(built["repos"]["post"], "README.md"), encoding="utf-8").read()
    assert "## Library slot" in post and os.path.exists(os.path.join(built["repos"]["post"], "library", "README.md"))
    lib = open(os.path.join(built["repos"]["library"], "README.md"), encoding="utf-8").read()
    assert "Empty for now" in lib


def test_fetch_sh_downloads_and_writes_local_mixes(built, tmp_path):
    """fetch.sh with a stand-in `hf` that copies from the built repo folders instead of the Hub."""
    stub = tmp_path / "bin" / "hf"
    stub.parent.mkdir()
    stub.write_text("#!/bin/bash\n# stand-in for `hf download ORG/NAME ... --local-dir DIR`\n"
                    "name=${2#*/}; while [ $# -gt 0 ]; do [ \"$1\" = --local-dir ] && dir=$2; shift; done\n"
                    f"mkdir -p \"$dir\" && cp -R \"{built['work']}/repos/$name/.\" \"$dir/\"\n")
    stub.chmod(0o755)
    env = dict(os.environ, PATH=f"{stub.parent}{os.pathsep}{os.environ['PATH']}", HF_TOKEN="test")
    dest = tmp_path / "data"
    r = subprocess.run(["bash", "experiments/data_mix/fetch.sh", "--dest", str(dest), "--post"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    for name, n in (("mix_pretrain.json", 5), ("mix_pretrain_val.json", 5), ("mix_pretrain_anneal.json", 5),
                    ("mix_post.json", 1)):
        spec = json.load(open(dest / name))
        assert len(spec["sources"]) == n
        assert all(os.path.exists(os.path.join(s["paths"], "tokens.bin")) for s in spec["sources"].values())
        assert open_data(str(dest / name)).batch(2, 8, np.random.default_rng(0))[0].shape == (2, 8)
    r = subprocess.run(["bash", "experiments/data_mix/fetch.sh", "--dest", str(tmp_path / "d2"), "--no-stack"],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "code_stack" not in json.load(open(tmp_path / "d2" / "mix_pretrain.json"))["sources"]
    env.pop("HF_TOKEN")
    r = subprocess.run(["bash", "experiments/data_mix/fetch.sh", "--dest", str(tmp_path / "d3")], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 2 and "gated" in r.stdout                # The Stack needs a token


# ---- recipe v2 and the first slice, read from download_slice.py's files --------------------------------------------

SCI = ["numpy", "scipy", "torch", "pandas"]


def py_doc(rng, sci):
    lib = rng.choice(SCI) if sci else rng.choice(["os", "json", "re", "itertools"])
    body = code_doc(rng, rng.randint(3, 8))
    return f"import {lib}\nfrom collections import defaultdict\n\n" + body


def write_olmo(path, docs):
    """OLMo-mix JSON lines: text plus metadata (extension, repository path)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i, (text, ext) in enumerate(docs):
            f.write(json.dumps({"id": str(i), "text": text, "metadata": {"extension": ext,
                                "max_stars_repo_path": f"src/m{i}.{ext}"}}) + "\n")


def make_slice(root, rng):
    """Files laid out as download_slice.py leaves them (<root>/<repo name>/<path>) and its manifest_slice.json."""
    star = []
    for k in range(4):
        docs = [(py_doc(rng, True), "py") for _ in range(60)] + [(py_doc(rng, False), "py") for _ in range(40)] + \
               [(code_doc(rng, rng.randint(2, 8)).replace("def ", "function "), "js") for _ in range(60)]
        rng.shuffle(docs)
        p = f"data/starcoder/v1/documents/mixed-{k:04d}.json.gz"
        write_olmo(os.path.join(root, "olmo-mix-1124", p), docs)
        star.append(p)
    se = []
    for k in range(2):
        p = f"data/stackexchange/stackexchange-{k:04d}.json.gz"
        write_olmo(os.path.join(root, "dolmino-mix-1124", p), [(en_doc(rng, rng.randint(80, 300)), "md") for _ in range(80)])
        se.append(p)
    cr = []
    for k in range(3):
        p = f"code-no-thinking/cr.chunk0-{k:05d}.parquet"
        t = [en_doc(rng, 60) + "\n\n" + code_doc(rng, rng.randint(2, 10)) for _ in range(80)]
        write_parquet(os.path.join(root, "Code-Reasoning", p), {"text": t, "token_count": [len(x) // 3 for x in t]})
        cr.append(p)
    mm = ["megamath-code/00000.parquet", "megamath-web/CC-MAIN-2014-15/000_00000.parquet"]
    write_parquet(os.path.join(root, "MegaMath", mm[0]), {"file_info": ["x"] * 5, "repo_info": ["y"] * 5})
    t = [en_doc(rng, rng.randint(60, 400)) + " 3 + 4 = 7." for _ in range(200)]
    write_parquet(os.path.join(root, "MegaMath", mm[1]), {"text": t, "url": ["u"] * 200, "math_score": [0.9] * 200})

    def entry(repo, files):
        return dict(repo=repo, files=[dict(path=f, bytes=os.path.getsize(os.path.join(root, repo.split("/")[1], f)))
                                      for f in files])
    man = dict(tokens=30e9, skipped=["the_stack_v1", "kaken", "books"],
               sources=dict(research_python=entry("allenai/olmo-mix-1124", star),
                            starcoder=entry("allenai/olmo-mix-1124", star[:2]),
                            stackexchange=entry("allenai/dolmino-mix-1124", se),
                            code_reasoning=entry("IFM/Code-Reasoning", cr),
                            megamath=entry("IFM/MegaMath", mm)))
    with open(os.path.join(root, "manifest_slice.json"), "w") as f:
        json.dump(man, f)
    return os.path.join(root, "manifest_slice.json")


def slice_config(work):
    cfg = config.load()                                          # the real recipe (v2)
    cfg["split"] = dict(val_fraction=0.02, val_tokens=0, val_long_docs=4, val_long_min=2, long_tokens=8 * 16 + 1,
                        val_files=1)
    cfg["tokenizer"] = dict(vocab_size=400, measure_mb=0.03,
                            sample_mb=dict(research_python=0.04, starcoder=0.04, stackexchange=0.03,
                                           code_reasoning=0.03, megamath=0.03))
    cfg["decontam"] = dict(ngram=13, min_matches=1, benchmarks=["gsm8k", "humaneval"])
    cfg["raw"] = dict(mb_per_file=0.05)
    path = os.path.join(work, "config.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def test_recipe_v2_config():
    cfg = config.apply_scale(config.load(), 1.0, 30e9)
    config.check(cfg)
    pre = [b for b in cfg["buckets"] if b["kind"] == "pretrain"]
    assert abs(sum(b["full_target"] for b in pre) - 300e9) < 1e6 and abs(cfg["build"]["fraction"] - 0.1) < 1e-9
    by = {b["name"]: b for b in cfg["buckets"]}
    assert by["research_python"]["target"] == 4.3e9 and by["stackexchange"]["epochs"] == 2
    assert config.unique_target(by["stackexchange"]) == by["stackexchange"]["target"] // 2
    assert by["the_stack_v1"]["repo"] == "pretrain_stack" and by["kaken"]["optional"]
    assert "numpy" in by["research_python"]["parts"][0]["filter"]["python_imports_any"]
    assert by["starcoder"]["parts"][0]["filter"]["python_imports_none"] == \
        by["research_python"]["parts"][0]["filter"]["python_imports_any"]
    assert all(b["enabled"] or b.get("disabled_reason") for b in cfg["buckets"])
    assert all(b["enabled"] for b in pre)                         # every pretraining source has an access spec
    half = config.apply_scale(config.load(), 0.5)
    assert sum(b["target"] for b in half["buckets"] if b["kind"] == "pretrain" and b["enabled"]) == 150e9
    for name in config.DECONTAM:
        from exealm.data.benchmarks import BENCHMARKS
        from experiments.data_mix.bench_extra import EXTRA
        assert name in BENCHMARKS or name in EXTRA


def test_bench_extra_segments():
    from experiments.data_mix.bench_extra import export
    rows = [dict(text="Write a function to add.", code="def add(a, b):\n    return a + b", test_list=["assert add(1, 2) == 3"],
                 test_setup_code="", task_id=1)]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        m = export("mbpp", os.path.join(d, "mbpp"), rows=rows, revision="fixture")
        assert m["rows"] == 1 and m["text_segments"] == 3
        lines = [json.loads(x) for x in gzip.open(os.path.join(d, "mbpp", "text.jsonl.gz"), "rt")]
        assert [x["field"] for x in lines] == ["text", "code", "test_0"]


def raw_set(work, cfg, bucket, split):
    d = os.path.join(work, "repos", "exealm-pretrain", "slice-test", "raw", bucket, split)
    return {doc_hash(t) for f in sorted(os.listdir(d)) for t in pq.read_table(os.path.join(d, f)).column("text").to_pylist()}


def test_slice_from_manifest_and_a_larger_build_extends_it(tmp_path):
    from experiments.data_mix.extract import imported, is_python
    rng = random.Random(3)
    manifest = make_slice(str(tmp_path / "data_raw"), rng)
    work = str(tmp_path / "work")
    os.makedirs(work)
    cfg_path = slice_config(work)
    make_benchmarks(work)
    common = ["--work", work, "--config", cfg_path, "--workers", "2"]
    run_cli("fetch", *common, "--slice_manifest", manifest, "--release", "slice-test", "--tokens", "60000")
    for step in ("dedup", "tokenizer", "benchmarks", "tokenize"):
        run_cli(step, *common, "--chunk_mb", "0.02")
    run_cli("assemble", *common, "--raw")
    run_cli("cards", *common)
    run_cli("verify", *common)
    built = json.load(open(os.path.join(work, "build.json")))
    assert built["release"] == "slice-test" and built["tokens"] == 60000
    rel = os.path.join(work, "repos", "exealm-pretrain", "slice-test")
    mix = json.load(open(os.path.join(rel, "mix.json")))
    assert set(mix["sources"]) == {"research_python", "starcoder", "stackexchange", "code_reasoning", "megamath"}
    assert mix["sources"]["research_python"]["paths"] == "data/exealm-pretrain/slice-test/packed/research_python/train"
    ann = json.load(open(os.path.join(rel, "mix_anneal.json")))
    assert abs(ann["sources"]["code_reasoning"]["weight"] / mix["sources"]["code_reasoning"]["weight"] - 3.0) < 1e-6
    assert os.path.exists(os.path.join(rel, "tokenizer", "tokenizer.json"))
    src = json.load(open(os.path.join(rel, "sources.json")))
    assert src["buckets"]["starcoder"]["files"][0]["path"].startswith("data/starcoder/")
    # the filters: research_python is Python importing a listed library, starcoder never is
    for name, want in (("research_python", True), ("starcoder", False)):
        d = os.path.join(rel, "raw", name, "train")
        for f in os.listdir(d):
            t = pq.read_table(os.path.join(d, f))
            for text, meta in zip(t.column("text").to_pylist(), t.column("meta").to_pylist()):
                sci = is_python(dict(metadata=json.loads(meta))) and bool(imported(text) & set(SCI))
                assert sci == want, name
    mm = [json.load(open(os.path.join(work, "extract", "megamath.0.slice", f)))
          for f in os.listdir(os.path.join(work, "extract", "megamath.0.slice")) if f.endswith(".json")]
    assert any(i.get("skipped") and "no column text" in i["skipped"] for i in mm)
    st = json.load(open(os.path.join(work, "reports", "assemble", "stackexchange.json")))
    target = config.apply_scale(config.load(cfg_path), 1.0, 60000)
    se = next(b for b in target["buckets"] if b["name"] == "stackexchange")
    assert st["splits"]["train"]["target"] == config.unique_target(se)      # 2 epochs: half the target, read twice
    card = open(os.path.join(rel, "README.md"), encoding="utf-8").read()
    assert "first slice" in card and "not in this build" in card and not card.lstrip().startswith("---")

    small = {b: (raw_set(work, None, b, "train"), raw_set(work, None, b, "val")) for b in mix["sources"]}
    tok_before = open(os.path.join(work, "tokenizer", "tokenizer.json"), "rb").read()
    run_cli("fetch", *common, "--tokens", "120000")                # a larger build, same work directory
    for step in ("dedup", "tokenizer", "benchmarks", "tokenize"):
        run_cli(step, *common, "--chunk_mb", "0.02")
    run_cli("assemble", *common, "--raw")
    assert open(os.path.join(work, "tokenizer", "tokenizer.json"), "rb").read() == tok_before
    for b, (train, val) in small.items():
        big_train, big_val = raw_set(work, None, b, "train"), raw_set(work, None, b, "val")
        assert train <= big_train, b                               # the slice is the start of the larger build
        assert val <= big_val and not (big_train & big_val), b
        assert len(big_train) >= len(train)


def test_extract_reads_jsonl_zst_and_skips_files_without_text(tmp_path):
    zstd = pytest.importorskip("zstandard")
    from experiments.data_mix.extract import extract
    src = tmp_path / "shard_00000000_processed.jsonl.zst"
    lines = "".join(json.dumps({"text": f"Document {i} about rivers and mountains.", "url": f"https://e.org/{i}"}) + "\n"
                    for i in range(50))
    src.write_bytes(zstd.ZstdCompressor().compress(lines.encode()))
    info = extract(str(src), str(tmp_path / "out.parquet"), dict(path="x"), dict(meta_keys=["url"]), seed=1)
    assert info["rows_read"] == info["rows_kept"] == 50
    t = pq.read_table(tmp_path / "out.parquet")
    assert t.column("text")[3].as_py() == "Document 3 about rivers and mountains."
    assert json.loads(t.column("meta")[3].as_py())["url"] == "https://e.org/3"
    write_parquet(str(tmp_path / "code.parquet"), {"file_info": ["a"], "repo_info": ["b"]})
    info = extract(str(tmp_path / "code.parquet"), str(tmp_path / "out2.parquet"), dict(path="y"), {}, seed=1)
    assert info["rows_kept"] == 0 and "no column text" in info["skipped"]
