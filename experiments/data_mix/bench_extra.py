"""Held-out text of the benchmarks in the recipe's decontamination list that exealm.data.benchmarks does not export:
MBPP test, APPS test, CodeContests test, LiveCodeBench, MATH test, SciQ test and PubMedQA (labeled).

Same output as exealm.data.benchmarks: a new folder with text.jsonl.gz (one segment per line: a question, a solution,
a test, each its own document so unrelated fields never share a window) and manifest.json (repo, revision, split,
counts, SHA-256). Needs `datasets` for the Parquet-backed sets; APPS and LiveCodeBench are read from their jsonl files.
The schemas were checked on 2026-11-24 against each dataset's viewer.
"""
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

MAX_SOLUTIONS = 3            # per problem: APPS and CodeContests keep dozens of solutions per problem

EXTRA = {
    "mbpp": dict(repo="google-research-datasets/mbpp", config="full", split="test", license="CC-BY-4.0"),
    "apps": dict(repo="codeparrot/apps", files=["test.jsonl"], split="test", license="MIT"),
    "code_contests": dict(repo="deepmind/code_contests", config="default", split="test", license="CC-BY-4.0"),
    "livecodebench": dict(repo="livecodebench/code_generation_lite",
                          files=["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl",
                                 "test6.jsonl"], split="test", license="CC-BY-4.0 (problem statements: see the card)"),
    "math": dict(repo="EleutherAI/hendrycks_math", configs=["algebra", "counting_and_probability", "geometry",
                                                            "intermediate_algebra", "number_theory", "prealgebra",
                                                            "precalculus"], split="test", license="MIT"),
    "sciq": dict(repo="allenai/sciq", config="default", split="test", license="CC-BY-NC-3.0"),
    "pubmedqa": dict(repo="qiaojin/PubMedQA", config="pqa_labeled", split="train", license="MIT"),
}


def segments(name, row):
    """Named text segments of one row."""
    def s(v):
        return v if isinstance(v, str) and v.strip() else None
    out = []
    if name == "mbpp":
        out = [("text", s(row["text"])), ("code", s(row["code"])), ("setup", s(row.get("test_setup_code")))]
        out += [(f"test_{i}", s(t)) for i, t in enumerate(row.get("test_list") or [])]
    elif name == "apps":
        sols = json.loads(row["solutions"]) if s(row.get("solutions")) else []
        out = [("question", s(row["question"])), ("starter_code", s(row.get("starter_code")))]
        out += [(f"solution_{i}", s(x)) for i, x in enumerate(sols[:MAX_SOLUTIONS])]
    elif name == "code_contests":
        sols = (row.get("solutions") or {}).get("solution") or []
        out = [("description", s(row["description"]))] + [(f"solution_{i}", s(x)) for i, x in enumerate(sols[:MAX_SOLUTIONS])]
    elif name == "livecodebench":
        out = [("question", s(row["question_content"])), ("starter_code", s(row.get("starter_code")))]
    elif name == "math":
        out = [("problem", s(row["problem"])), ("solution", s(row["solution"]))]
    elif name == "sciq":
        out = [("question", s(row["question"])), ("support", s(row.get("support"))),
               ("correct_answer", s(row.get("correct_answer")))]
        out += [(f"distractor{i}", s(row.get(f"distractor{i}"))) for i in (1, 2, 3)]
    elif name == "pubmedqa":
        ctx = " ".join((row.get("context") or {}).get("contexts") or [])
        out = [("question", s(row["question"])), ("context", s(ctx)), ("long_answer", s(row.get("long_answer")))]
    else:
        raise ValueError(f"unknown benchmark {name}")
    return [(k, v) for k, v in out if v]


def rows_of(name, spec, revision):
    if spec.get("files"):
        from huggingface_hub import hf_hub_download
        for f in spec["files"]:
            path = hf_hub_download(spec["repo"], f, repo_type="dataset", revision=revision)
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)
        return
    from datasets import load_dataset
    for config in spec.get("configs") or [spec["config"]]:
        yield from load_dataset(spec["repo"], config, split=spec["split"], revision=revision, streaming=True)


def export(name, out, limit=None, rows=None, revision=None):
    """Write <out>/text.jsonl.gz and manifest.json. `rows` (an iterable) replaces the download, for tests."""
    spec = EXTRA[name]
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if rows is None:
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(spec["repo"]).sha
        rows = rows_of(name, spec, revision)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bench-", dir=out.parent) as tmp:
        work = Path(tmp) / "export"
        work.mkdir()
        n = parts = 0
        with gzip.open(work / "text.jsonl.gz", "wt", encoding="utf-8") as f:
            for i, row in enumerate(rows):
                if limit is not None and i >= limit:
                    break
                for field, text in segments(name, row):
                    f.write(json.dumps(dict(text=text, source=name, benchmark_id=str(i), field=field),
                                       ensure_ascii=False) + "\n")
                    parts += 1
                n += 1
        if not n:
            raise ValueError(f"empty benchmark export: {name}")
        digest = hashlib.sha256((work / "text.jsonl.gz").read_bytes()).hexdigest()
        manifest = dict(benchmark=name, repo=spec["repo"], config=spec.get("config") or ",".join(spec.get("configs", [])
                                                                                               or spec.get("files", [])),
                        split=spec["split"], revision=revision, license=spec["license"], rows=n, text_segments=parts,
                        limit=limit, sample=limit is not None, sha256=digest,
                        source_url=f"https://huggingface.co/datasets/{spec['repo']}",
                        policy=f"raw fields, at most {MAX_SOLUTIONS} solutions per problem")
        (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.rename(work, out)
    return manifest
