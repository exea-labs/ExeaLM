"""experiments/library/build.py: topics -> write (mock backend) -> pack, end to end, with a tiny tokenizer trained here
and with the project's 32k tokenizer when it is on disk. No model is downloaded and nothing touches the network."""
import gzip
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("tokenizers")

from exealm.data.loader import TokenShards, open_data  # noqa: E402
from exealm.data.pack import pack  # noqa: E402
from exealm.memory.datastore import Datastore  # noqa: E402

ROOT = pathlib.Path(__file__).parents[1]
BUILD = ROOT / "experiments" / "library" / "build.py"
spec = importlib.util.spec_from_file_location("library_build", BUILD)     # CI runs bare pytest: load by path
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)

ARTICLES = [
    "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist. He developed the theory "
    "of relativity and received the 1921 Nobel Prize in Physics.",
    "Photosynthesis\nPhotosynthesis is the process plants use to turn light into chemical energy.",
    "東京都（とうきょうと）は、日本の首都である。人口は約1400万人である。",
    "The Beatles were an English rock band formed in Liverpool in 1960.",
    "Mercury may refer to:\nMercury (planet)\nMercury (element)",
    "albert einstein (1879-1955) was a physicist.",                       # the same topic again, in lower case
    "In mathematics, a group is a set with an operation that combines any two elements.",
]
WIKI_TOPICS = ["Albert Einstein", "Photosynthesis", "東京都", "The Beatles", "group"]


def write_jsonl_gz(path, texts):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    """A 400-token byte-level BPE trained in a second by exealm/data/tokenizer.py, special tokens included."""
    from exealm.data.tokenizer import train
    d = tmp_path_factory.mktemp("tok")
    write_jsonl_gz(d / "corpus.jsonl.gz", ARTICLES * 20 + [lb.mock_reply(f"Topic {i}") for i in range(20)])
    train([str(d / "corpus.jsonl.gz")], 400, str(d / "tiny"))
    return str(d / "tiny")


@pytest.fixture(params=["tiny", "bytes", "bpe32k"])
def tokenizer(request, tiny_tokenizer):
    if request.param == "tiny":
        return tiny_tokenizer
    if request.param == "bytes":
        return "bytes"
    path = ROOT / "results" / "tokenizer" / "bpe32k"
    if not (path / "tokenizer.json").exists():
        pytest.skip("the 32k tokenizer is not on disk (results/tokenizer/bpe32k)")
    return str(path)


@pytest.fixture
def wiki_pack(tmp_path, tokenizer):
    write_jsonl_gz(tmp_path / "wiki.jsonl.gz", ARTICLES)
    pack([str(tmp_path / "wiki.jsonl.gz")], tokenizer, str(tmp_path / "wiki"))
    return str(tmp_path / "wiki")


def test_topics_come_from_packs_records_and_text_files_without_duplicates(tmp_path, tokenizer, wiki_pack):
    out = str(tmp_path / "topics.txt")
    assert lb.main(["topics", wiki_pack, "--tokenizer", tokenizer, "--out", out]) == WIKI_TOPICS
    assert lb.read_lines(out) == WIKI_TOPICS

    with open(tmp_path / "titled.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"title": "Kyoto", "text": "Kyoto was the capital of Japan for a thousand years."}) + "\n")
        f.write(json.dumps({"text": "Osaka is a city in Japan."}) + "\n")
    (tmp_path / "mine.txt").write_text("# my own list\nphotosynthesis\n\nMitochondrion\n", encoding="utf-8")
    got = lb.main(["topics", wiki_pack, str(tmp_path / "titled.jsonl"), str(tmp_path / "mine.txt"),
                   "--tokenizer", tokenizer, "--out", out])
    assert got == WIKI_TOPICS + ["Kyoto", "Osaka", "Mitochondrion"]   # "photosynthesis" is a duplicate

    assert lb.main(["topics", wiki_pack, "--tokenizer", tokenizer, "--limit", "2", "--out", out]) == WIKI_TOPICS[:2]
    picked = lb.main(["topics", wiki_pack, "--tokenizer", tokenizer, "--shuffle", "--limit", "3", "--out", out])
    assert len(picked) == 3 and set(picked) <= set(WIKI_TOPICS)
    assert lb.main(["topics", wiki_pack, "--tokenizer", tokenizer, "--shuffle", "--limit", "3", "--out", out]) == picked


def test_end_to_end_the_pack_is_a_library_the_datastore_reads(tmp_path, tokenizer, wiki_pack):
    (tmp_path / "more.txt").write_text("\n".join(f"Mock subject {i}" for i in range(30)) + "\n", encoding="utf-8")
    topics = str(tmp_path / "topics.txt")
    lb.main(["topics", wiki_pack, str(tmp_path / "more.txt"), "--tokenizer", tokenizer, "--out", topics])
    n = len(WIKI_TOPICS) + 30
    entries = str(tmp_path / "entries.jsonl")
    assert lb.main(["write", "--topics", topics, "--backend", "mock", "--out", entries]) == dict(written=n, rejected=0)
    recs = read_jsonl(entries)
    assert [r["topic"] for r in recs] == lb.read_lines(topics)
    for r in recs:
        assert set(r) == {"topic", "entry", "qa", "model", "prompt_version", "lang", "kind"} and r["kind"] == "text"
        assert r["model"] == "mock" and r["prompt_version"] == lb.PROMPT_VERSION and len(r["qa"]) == 3
        assert r["lang"] == ("ja" if r["topic"] == "東京都" else "en")

    lib = tmp_path / "library"
    # pack runs as a script from another folder: it must find the exealm package on its own
    subprocess.run([sys.executable, str(BUILD), "pack", entries, "--tokenizer", tokenizer, "--out", str(lib),
                    "--heldout", "0.5"], cwd=tmp_path, check=True, capture_output=True)

    shards = TokenShards(str(lib))
    meta = json.loads((lib / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_docs"] == len(shards.offsets) == n
    assert meta["n_tokens"] == len(shards.tokens) and meta["dtype"] == "uint16"
    assert meta["vocab_size"] == lb.load_tokenizer(tokenizer).get_vocab_size()
    text_bytes = sum(len(f"{r['topic']}\n\n{r['entry']}".encode("utf-8")) for r in recs)
    assert meta["bytes_per_token"] == pytest.approx(text_bytes / meta["n_tokens"])
    assert meta["source"]["writers"] == [dict(model="mock", prompt_version=lb.PROMPT_VERSION, entries=n)]
    assert isinstance(open_data(str(lib)), TokenShards)                  # what --datastore_data and --data open

    ds = Datastore.from_shards(shards, 200, 64, seed=1)
    ends = np.append(shards.offsets[1:], len(shards.tokens))
    for row, d in zip(ds.tokens.numpy(), ds.doc_ids.numpy()):
        a = int(shards.offsets[d])
        window = np.asarray(shards.tokens[a:int(ends[d])]).astype(np.int64)
        assert any((window[s:s + 64] == row).all() for s in range(len(window) - 63))   # inside its own document

    index = read_jsonl(lib / "index.jsonl")
    assert [i["doc"] for i in index] == list(range(n))
    assert sorted(i["topic"] for i in index) == sorted(r["topic"] for r in recs)
    splits = [i["split"] for i in index]
    assert splits == sorted(splits, key=lambda s: s == "heldout")         # train topics first, then held-out ones
    n_train = splits.count("train")

    tok = lb.load_tokenizer(tokenizer)
    train, held = read_jsonl(lib / "qa" / "train" / "items.jsonl"), read_jsonl(lib / "qa" / "heldout" / "items.jsonl")
    assert train and held and len(train) + len(held) == 3 * n
    assert not {p["topic"] for p in train} & {p["topic"] for p in held}  # no topic on both sides
    for split, pairs in (("train", train), ("heldout", held)):
        for p in pairs:
            d = p["source_doc"]
            assert index[d]["topic"] == p["topic"] and index[d]["split"] == split
            text = tok.decode([int(t) for t in shards.tokens[int(shards.offsets[d]):int(ends[d])]])
            assert text.startswith(p["topic"]) and p["a"] in text      # the library holds every answer
        pack_ = TokenShards(str(lib / "qa" / split))                      # one document per pair, answer span known
        spans = np.load(lib / "qa" / split / "targets.npy")
        assert len(pack_.offsets) == len(spans) == len(pairs)
        for p, (s, e), off in zip(pairs, spans, pack_.offsets):
            assert tok.decode([int(t) for t in pack_.tokens[s:e]]) == " " + p["a"]
            assert tok.decode([int(t) for t in pack_.tokens[off:s]]) == f"Question: {p['q']}\nAnswer:"
            assert (s - off, e - off) == (p["target_start"], p["target_end"])
            assert p["target_bytes"] == len((" " + p["a"]).encode("utf-8"))
    assert all(p["source_doc"] >= n_train for p in held)

    note = (lib / "SOURCE.md").read_text(encoding="utf-8")
    assert "mock" in note and f"prompt {lb.PROMPT_VERSION}" in note and f"{n:,} documents" in note


def test_write_resumes_and_skips_topics_already_written(tmp_path, monkeypatch):
    topics = tmp_path / "topics.txt"
    topics.write_text("Alpha\nBeta\nGamma\nbeta\nDelta\nEpsilon\n", encoding="utf-8")   # "beta" repeats "Beta"
    out = str(tmp_path / "entries.jsonl")
    asked = []
    real = lb.MockBackend.generate

    def generate(self, chats):                                         # the mock, recording what it was asked
        asked.extend(re.search(r"^Topic: (.*)$", c[-1]["content"], re.M).group(1) for c in chats)
        return real(self, chats)

    monkeypatch.setattr(lb.MockBackend, "generate", generate)
    args = ["write", "--topics", str(topics), "--backend", "mock", "--out", out]
    assert lb.main(args + ["--limit", "2"]) == dict(written=2, rejected=0)
    with open(out, "a", encoding="utf-8") as f:
        f.write('{"topic": "Gamma", "entry": "cut off by a cra')            # a run killed while writing
    asked.clear()
    assert lb.main(args) == dict(written=3, rejected=0)
    assert asked == ["Gamma", "Delta", "Epsilon"]
    assert [r["topic"] for r in read_jsonl(out)] == ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]

    monkeypatch.setattr(lb, "make_backend", lambda a: pytest.fail("every topic is done: no model should load"))
    assert lb.main(args) == dict(written=0, rejected=0)


def test_unusable_replies_go_to_the_rejects_file(tmp_path, monkeypatch):
    (tmp_path / "topics.txt").write_text("Alpha\nBeta\n", encoding="utf-8")
    monkeypatch.setattr(lb.MockBackend, "generate", lambda self, chats: ["no format at all"] * len(chats))
    out = str(tmp_path / "entries.jsonl")
    assert lb.main(["write", "--topics", str(tmp_path / "topics.txt"), "--backend", "mock", "--out", out]) == \
        dict(written=0, rejected=2)
    assert not os.path.exists(out)
    bad = read_jsonl(tmp_path / "entries.rejects.jsonl")
    assert [b["topic"] for b in bad] == ["Alpha", "Beta"] and bad[0]["reason"] == "no QUESTIONS: marker"


def test_parse_keeps_only_answers_stated_in_the_entry():
    entry = ("Kyoto is a city in Japan. Kyoto was the capital of Japan from 794 to 1869. Kyoto has 1.46 million "
             "people. ") * 6
    reply = ("<think>\nlet me think\n</think>\n**ENTRY:**\n## Kyoto\n" + entry +
             "\n**QUESTIONS:**\n1. Q: When did Kyoto become the capital?\n   A: 794.\n"
             "Q: How many people live in Kyoto?\nA: 1.46 million\n"
             "Q: Who founded Kyoto?\nA: Emperor Kanmu\n"                       # not in the entry: dropped
             "Q: Until when was Kyoto the capital?\nA: in 1869\n"              # "1869" is: kept
             "Q: What ended in 1869?\nA: until 1869\n")                        # "until" is not: dropped
    got, qa, why = lb.parse_output(reply, min_qa=3, min_entry_words=50)
    assert why is None and got.startswith("Kyoto\nKyoto is a city") and "**" not in got and "think" not in got
    assert qa == [{"q": "When did Kyoto become the capital?", "a": "794"},
                  {"q": "How many people live in Kyoto?", "a": "1.46 million"},
                  {"q": "Until when was Kyoto the capital?", "a": "in 1869"}]
    assert lb.parse_output(reply, min_qa=4, min_entry_words=50)[2] == "3 of 5 answers are stated in the entry (need 4)"
    assert len(lb.parse_output(reply, min_qa=1, max_qa=2, min_entry_words=50)[1]) == 2
    assert lb.parse_output(reply, min_entry_words=500)[2].startswith("entry of")
    assert lb.parse_output("UNKNOWN")[2] == "unknown topic"
    assert lb.parse_output("ENTRY:\n" + entry)[2] == "no QUESTIONS: marker"
    ja = "ENTRY:\n" + "京都市は日本の都市である。京都は794年から1869年まで日本の首都だった。" * 8 + \
         "\nQUESTIONS:\nQ: 京都はいつ首都になったか。\nA: 794年\nQ: 京都はいつまで首都だったか。\nA: 1869年\n" \
         "Q: 京都は何か。\nA: 日本の都市\nQ: 京都を作ったのは誰か。\nA: 桓武天皇\n"
    assert [p["a"] for p in lb.parse_output(ja)[1]] == ["794年", "1869年", "日本の都市"]


def test_the_split_depends_on_the_topic_alone():
    topics = [f"Topic {i}" for i in range(4000)]
    first = [lb.is_heldout(t, 0.1, "salt") for t in topics[:1000]]
    assert [lb.is_heldout(t, 0.1, "salt") for t in topics][:1000] == first          # more topics move nothing
    assert lb.is_heldout("Kyoto", 0.1, "salt") == lb.is_heldout(" kyoto ", 0.1, "salt")
    assert abs(sum(lb.is_heldout(t, 0.1, "salt") for t in topics) / len(topics) - 0.1) < 0.02
    assert first != [lb.is_heldout(t, 0.1, "other") for t in topics[:1000]]


def test_the_library_is_packed_exactly_like_exealm_data_pack(tmp_path, tokenizer):
    texts = ARTICLES + [lb.mock_reply(f"Topic {i}") for i in range(5)]
    write_jsonl_gz(tmp_path / "docs.jsonl.gz", texts)
    ref = pack([str(tmp_path / "docs.jsonl.gz")], tokenizer, str(tmp_path / "ref"))
    meta, lengths = lb.write_pack(texts, lb.load_tokenizer(tokenizer), tokenizer, str(tmp_path / "ours"), ["x"])
    for name in ("tokens.bin", "offsets.npy"):
        assert (tmp_path / "ref" / name).read_bytes() == (tmp_path / "ours" / name).read_bytes()
    assert {k: meta[k] for k in ref if k != "files"} == {k: ref[k] for k in ref if k != "files"}
    assert sum(lengths) == meta["n_tokens"]


def test_the_bytes_tokenizer_decodes_without_end_of_text():
    assert lb.Bytes().decode(list("東京".encode("utf-8")) + [256]) == "東京"


def test_transformers_backend_runs_on_a_tiny_local_model(tmp_path, tiny_tokenizer, monkeypatch):
    """A randomly initialised two-layer Llama saved to disk: checks the chat template, left padding and decoding of
    the transformers path. Its replies are noise, so every topic lands in the rejects file."""
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from tokenizers import Tokenizer
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(os.path.join(tiny_tokenizer, "tokenizer.json")),
                                  eos_token="<|endoftext|>", pad_token="<|pad|>")
    tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                         "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096)
    model_dir = str(tmp_path / "tiny_llama")
    LlamaForCausalLM(cfg).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    (tmp_path / "topics.txt").write_text("Alpha\nA much longer topic name\n東京都\n", encoding="utf-8")
    out = str(tmp_path / "entries.jsonl")
    assert lb.main(["write", "--topics", str(tmp_path / "topics.txt"), "--backend", "transformers", "--model",
                    model_dir, "--max_new_tokens", "4", "--temperature", "0", "--out", out]) == \
        dict(written=0, rejected=3)
    assert [b["model"] for b in read_jsonl(tmp_path / "entries.rejects.jsonl")] == [model_dir] * 3


# Code topics, code entries, morph tasks and their packs

CODE_FILES = [
    'import os\nimport json\nimport numpy as np\nfrom collections import defaultdict\n\n\n'
    'def read_json_file(path):\n    with open(os.path.join("data", path)) as f:\n        return json.load(f)\n\n\n'
    'def mean_of(values):\n    return np.mean(values)\n',
    'import os, json\n\n\ndef read_json_file(path):\n    return json.load(open(os.path.join("cfg", path)))\n',
    'import numpy as np\nfrom collections import defaultdict\n\n\ndef count_words(text):\n'
    '    counts = defaultdict(int)\n    for w in text.split():\n        counts[w] += 1\n'
    '    return np.mean(list(counts.values()))\n',
    "const fs = require('fs');\nfunction readConfigFile(p) {\n  return JSON.parse(fs.readFileSync(p, 'utf8'));\n}\n",
    "import fs from 'fs';\nexport function readConfigFile(p) {\n  return fs.readFileSync(p);\n}\n",
    'package main\n\nimport "fmt"\n\nfunc printTotal(x int) {\n\tfmt.Println(x)\n}\n',
]
CODE_TOPICS = ["collections (Python)", "collections.defaultdict (Python)", "read config file (JavaScript)",
               "fs (JavaScript)", "fs.readFileSync (JavaScript)", "read json file (Python)",
               "json (Python)", "json.load (Python)", "numpy (Python)", "numpy.mean (Python)",
               "os (Python)", "os.path.join (Python)"]                  # module, API, task in turn, most used first


def test_code_topics_are_mined_from_imports_calls_and_definitions(tmp_path, tokenizer):
    write_jsonl_gz(tmp_path / "code.jsonl.gz", CODE_FILES)
    pack([str(tmp_path / "code.jsonl.gz")], tokenizer, str(tmp_path / "code"))
    (tmp_path / "mine.txt").write_text("binary search (Python)\nos (Python)\n", encoding="utf-8")
    out = str(tmp_path / "code_topics.txt")
    args = ["topics", "--kind", "code", "--tokenizer", tokenizer, "--out", out]
    assert lb.main(args + [str(tmp_path / "code")]) == CODE_TOPICS          # the packed files, decoded
    assert lb.main(args + [str(tmp_path / "code.jsonl.gz")]) == CODE_TOPICS  # the same files as records
    got = lb.main(args + [str(tmp_path / "mine.txt"), str(tmp_path / "code")])
    assert got == ["binary search (Python)", "os (Python)"] + [t for t in CODE_TOPICS if t != "os (Python)"]
    once = lb.main(args + ["--min_docs", "1", str(tmp_path / "code")])
    assert {"fmt (Go)", "fmt.Println (Go)", "print total (Go)", "count words (Python)"} <= set(once)
    assert lb.main(args + ["--limit", "3", str(tmp_path / "code")]) == CODE_TOPICS[:3]


def test_code_facts_tell_languages_apart():
    assert lb.code_facts(CODE_FILES[0])[0] == "python" and lb.code_facts(CODE_FILES[3])[0] == "javascript"
    assert lb.code_facts(CODE_FILES[5])[:3] == ("go", {"fmt"}, {"fmt.Println"})
    java = "package a.b;\nimport java.util.List;\nimport java.io.*;\npublic class X { }\n"
    assert lb.code_facts(java)[:3] == ("java", {"java.util", "java.io"}, {"java.util.List"})
    assert lb.code_facts("use std::collections::HashMap;\nuse crate::x;\nfn count_words() {}\n")[:3] == \
        ("rust", {"std::collections"}, {"std::collections::HashMap"})
    assert lb.code_facts("#include <stdio.h>\n#include \"mine.h\"\nint main() {}\n")[:2] == ("cpp", {"stdio.h"})
    assert lb.code_facts("import Foundation\nfunc greet() {}\n")[0] is None        # Swift: not Python
    assert lb.code_facts("package com.x\nimport kotlin.math.sqrt\nfun main() {}\n")[0] is None
    relative = "from . import local\nfrom ._x import y\ndef run():\n    pass\n"   # the project's own code
    assert lb.code_facts(relative)[:3] == ("python", set(), set())
    assert lb.code_lang_of("fs (JavaScript)") == "javascript" and lb.code_lang_of("binary search") == "python"


def write_code_entries(tmp_path, topics, monkeypatch=None, replies=None):
    (tmp_path / "code_topics.txt").write_text("\n".join(topics) + "\n", encoding="utf-8")
    if replies:
        monkeypatch.setattr(lb.MockBackend, "generate", lambda self, chats: [
            replies(re.search(r"^Topic: (.*)$", c[-1]["content"], re.M).group(1)) for c in chats])
    out = str(tmp_path / "code_entries.jsonl")
    got = lb.main(["write", "--kind", "code", "--topics", str(tmp_path / "code_topics.txt"), "--backend", "mock",
                   "--out", out])
    return got, out


def test_code_entries_keep_python_that_parses(tmp_path, monkeypatch):
    got, out = write_code_entries(tmp_path, ["os.path.join (Python)", "fs.readFileSync (JavaScript)", "binary search"])
    assert got == dict(written=3, rejected=0)
    recs = read_jsonl(out)
    assert [r["lang"] for r in recs] == ["python", "javascript", "python"]
    for r in recs:
        assert r["kind"] == "code" and r["qa"] == [] and r["prompt_version"] == lb.CODE_PROMPT_VERSION
        assert r["entry"].startswith(r["topic"]) and f"```{r['lang']}\n{r['code']}\n```" in r["entry"]
        assert r["entry"].split("```")[-1].strip().startswith(r["topic"])     # the notes name the topic too
        if r["lang"] == "python":
            lb.ast.parse(r["code"])

    def replies(topic):
        tag, code = {"broken (Python)": ("python", "def broken(:\n    pass"),
                     "lenient (JavaScript)": ("javascript", "const x = {a: 1};"),
                     "no notes": ("python", "x = 1")}[topic]
        notes = "" if topic == "no notes" else f"NOTES:\n{topic} has no pitfalls worth a note here.\n"
        return (f"EXPLANATION:\n{topic} is a helper used only in this test of the code library.\n"
                f"CODE:\n```{tag}\n{code}\n```\n{notes}")
    (tmp_path / "x").mkdir()
    got, out = write_code_entries(tmp_path / "x", ["broken (Python)", "lenient (JavaScript)", "no notes"], monkeypatch,
                                  replies)
    assert got == dict(written=1, rejected=2)
    assert [r["topic"] for r in read_jsonl(out)] == ["lenient (JavaScript)"]   # not Python: the tag decides
    bad = {b["topic"]: b["reason"] for b in read_jsonl(tmp_path / "x" / "code_entries.rejects.jsonl")}
    assert bad["broken (Python)"].startswith("python code does not parse") and bad["no notes"] == "no NOTES: part"


@pytest.fixture
def entries(tmp_path):
    """Mock text and code entries: (text entries file, code entries file)."""
    (tmp_path / "topics.txt").write_text("\n".join(f"Mock subject {i}" for i in range(12)) + "\n", encoding="utf-8")
    text = str(tmp_path / "entries.jsonl")
    lb.main(["write", "--topics", str(tmp_path / "topics.txt"), "--backend", "mock", "--out", text])
    _, code = write_code_entries(tmp_path, [f"mock.helper{i} (Python)" for i in range(10)])
    return text, code


def test_morph_writes_tasks_that_change_the_entry(tmp_path, entries, monkeypatch):
    out = str(tmp_path / "tasks.jsonl")
    args = ["morph", *entries, "--backend", "mock", "--heldout", "0.5", "--out", out]
    got = lb.main(args)
    tasks = read_jsonl(out)
    assert got == dict(written=len(tasks), rejected=0)
    ents = {r["topic"]: r for r in lb.read_entries(list(entries))}
    assert {t["topic"] for t in tasks} == set(ents)
    for t in tasks:
        e = ents[t["topic"]]
        assert t["kind"] == e["kind"] and t["prompt_version"] == lb.MORPH_PROMPT_VERSION
        assert lb.check_task(t, e["entry"]) is None
        prefix, target = lb.task_texts(t)
        assert lb.longest_run(lb.run_tokens(e["entry"]), lb.run_tokens(target)) <= 0.5 * len(lb.run_tokens(target))
        if t["kind"] == "code":
            lb.ast.parse(prefix + target)                                  # the whole task document is a program
            assert prefix.startswith("# ")
        if t["also_topic"]:
            assert t["kind"] == "code" and t["type"] == "combine" and t["also_topic"] != t["topic"]
            assert lb.is_heldout(t["also_topic"], 0.5, "exealm-library") == lb.is_heldout(t["topic"], 0.5,
                                                                                         "exealm-library")
    assert {t["type"] for t in tasks if t["kind"] == "text"} == {"compute", "combine", "audience"}
    assert {t["type"] for t in tasks if t["kind"] == "code"} == {"rename", "extend", "edge", "combine"}

    monkeypatch.setattr(lb, "make_backend", lambda a: pytest.fail("every entry has tasks: no model should load"))
    assert lb.main(args) == dict(written=0, rejected=0)


def test_morph_rejects_copies_leaks_and_broken_python(tmp_path, entries, monkeypatch):
    code = lb.read_entries([entries[1]])[0]
    (tmp_path / "one.jsonl").write_text(json.dumps(code) + "\n", encoding="utf-8")
    body = "\n".join(code["code"].splitlines()[3:])                       # the entry's own function body
    sig = code["code"].splitlines()[2]

    def task(kind, instruction, context, target):
        return f"TASK: {kind}\nINSTRUCTION: {instruction}\nCONTEXT:\n{context}\nTARGET:\n{target}\nEND\n"
    reply = (task("rename", "Copy it.", sig, body)
             + task("fix", "Fix this.", code["code"], "def fixed(values):\n    return sum(values) * 2")
             + task("extend", "Add a flag.", "def f(values, flag=False):", "    return (sum(values) if flag else 0")
             + "TASK: edge\nINSTRUCTION: Handle empty lists.\nTARGET:\n    return 0\nEND\n"
             + task("port", "Write it as a class.", "", "class Summer:\n    def __call__(self, values):\n"
                    "        return sum(values) * 3"))
    monkeypatch.setattr(lb.MockBackend, "generate", lambda self, chats: [reply] * len(chats))
    out = str(tmp_path / "tasks.jsonl")
    assert lb.main(["morph", str(tmp_path / "one.jsonl"), "--backend", "mock", "--out", out]) == \
        dict(written=1, rejected=4)
    assert [t["type"] for t in read_jsonl(out)] == ["port"]
    reasons = [b["reason"] for b in read_jsonl(tmp_path / "tasks.rejects.jsonl")]
    assert reasons[0].startswith("target copies") and reasons[1].startswith("instruction or context copies")
    assert reasons[2].startswith("python does not parse") and reasons[3].startswith("a task without")


def test_parse_tasks_reads_sections_and_keeps_indentation():
    reply = ("<think>planning</think>\n**TASK:** extend (add a parameter)\n**INSTRUCTION:** Add a step\nargument.\n"
             "**CONTEXT:**\n```python\ndef f(values, step=1):\n```\n"
             "**TARGET:**\n```python\n    return values[::step]\n```\nEND\n"
             "TASK: audience\nINSTRUCTION: Explain it to a child.\nCONTEXT:\n(nothing)\nTARGET:\nIt is a plant.\n")
    got = lb.parse_tasks(reply)
    assert got == [dict(type="extend", instruction="Add a step argument.", context="def f(values, step=1):",
                        target="    return values[::step]"),
                   dict(type="audience", instruction="Explain it to a child.", context="", target="It is a plant.")]
    assert lb.task_texts(dict(got[0], kind="code", lang="python")) == \
        ("# Add a step argument.\ndef f(values, step=1):\n", "    return values[::step]")
    assert lb.task_texts(dict(got[1], kind="text")) == ("Explain it to a child.\n\n", "It is a plant.")
    assert lb.task_texts(dict(kind="code", lang="javascript", instruction="Do it.", context="", target="f();"))[0] == \
        "// Do it.\n"
    assert lb.task_texts(dict(kind="qa", q="Why?", a="Because")) == ("Question: Why?\nAnswer:", " Because")


def test_pack_puts_tasks_with_their_target_spans_on_their_topic_side(tmp_path, entries, tokenizer):
    tasks = str(tmp_path / "tasks.jsonl")
    lb.main(["morph", *entries, "--backend", "mock", "--heldout", "0.5", "--out", tasks])
    recs = read_jsonl(tasks)
    lone = next(t for t in recs if t["also_topic"])
    other = next(r["topic"] for r in lb.read_entries([entries[1]])
                 if lb.is_heldout(r["topic"], 0.5, "exealm-library") != lb.is_heldout(lone["topic"], 0.5,
                                                                                         "exealm-library"))
    with open(tasks, "a", encoding="utf-8") as f:                          # a combine task across the split
        f.write(json.dumps(dict(lone, also_topic=other)) + "\n")
    lib = tmp_path / "lib"
    meta = lb.main(["pack", *entries, "--tasks", tasks, "--tokenizer", tokenizer, "--heldout", "0.5",
                    "--out", str(lib)])
    assert meta["source"]["entries"] == {"text": 12, "code": 10}
    assert meta["source"]["tasks_dropped"] == {"its second topic is on the other side of the split": 1}

    tok = lb.load_tokenizer(tokenizer)
    library = TokenShards(str(lib))
    ends = np.append(library.offsets[1:], len(library.tokens))
    index = read_jsonl(lib / "index.jsonl")
    seen = {}
    for split in ("train", "heldout"):
        items = read_jsonl(lib / "tasks" / split / "items.jsonl")
        shards, spans = TokenShards(str(lib / "tasks" / split)), np.load(lib / "tasks" / split / "targets.npy")
        assert len(items) == len(shards.offsets) == len(spans)
        for it, (s, e), off in zip(items, spans, shards.offsets):
            assert lb.is_heldout(it["topic"], 0.5, "exealm-library") == (split == "heldout")
            seen[it["topic"]] = split
            assert index[it["source_doc"]]["topic"] == it["topic"]
            assert it["also_doc"] is None or index[it["also_doc"]]["topic"] == it["also_topic"]
            prefix, target = lb.task_texts(it)
            doc = tok.decode([int(t) for t in shards.tokens[off:e]])
            assert doc == prefix + target and tok.decode([int(t) for t in shards.tokens[s:e]]) == target
            assert (s - off, e - off, it["target_bytes"]) == (it["target_start"], it["target_end"],
                                                              len(target.encode("utf-8")))
            entry = tok.decode([int(t) for t in library.tokens[int(library.offsets[it["source_doc"]]):
                                                               int(ends[it["source_doc"]]) - 1]])
            assert entry.split("\n\n", 1)[1] not in doc                    # the entry stays in the library
    assert set(seen.values()) == {"train", "heldout"}
