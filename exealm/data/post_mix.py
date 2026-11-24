"""The SFT mix: sample the files exealm/data/post.py wrote at DATA.md's Stage 1 shares, toward a token budget, and count
the tokens the way exealm/train/sft.py trains on them.

    python -m exealm.data.post_mix --data data/post --tokens 1.5e9 --missing empty --out data/post/mix

Buckets (DATA.md, Stage 1) and the post.py files that fill them:

    bucket       share  files in --data (<name>.jsonl.gz)
    chat           35%  tulu3-sft (without its safety subsets), smoltalk, dolci-sft
    math           20%  openmath, openthoughts3-math
    code           20%  opencode, openthoughts3-code
    science        10%  openthoughts3-science
    safety          5%  tulu3-safety, and the safety subsets of tulu3-sft (post.SAFETY)
    real_users      5%  none yet: allenai/WildChat-1M needs the personal-data and toxicity filter first
    tool_use        5%  none yet: no verified open set

A bucket with no rows is never filled from another bucket's files. --missing, which has no default, says what happens
to its share: "empty" leaves it empty, so the mix comes out smaller than --tokens; "redistribute" gives it to the
buckets that have rows, in proportion to their shares. A bucket with too few rows is used whole and reported as short;
its shortfall goes to no other bucket in either mode.

Tokens: each conversation is written in the chat template of exealm/chat.py (sft.format_chat: <|im_start|>, the role
token, the content and <|im_end|> for every message), with the tokenizer the model trains with (default: the project's
published 32k tokenizer), and cut to --max_len tokens as sft.py cuts it (default 1024, sft.py's default; 0 for no cut).
Shares and the budget count the tokens sft.py trains on, after the cut; the report also has the uncut length and the
assistant tokens (the ones with a loss). A conversation whose answer starts past --max_len gives sft.py nothing to train
on, so it is dropped here as sft.py drops it. --long drop also drops the conversations longer than --max_len, instead of
cutting them (a reasoning trace cut in the middle never reaches its answer).

Sampling: the rows of a bucket, from all its files, are shuffled together (--seed) and taken in that order until the
bucket's tokens reach its target, so every file gives rows in proportion to the rows it holds: format more rows of a
file to give it more weight. Each row is checked as it is taken, and dropped with its reason counted:
  - exact duplicates: one copy of a conversation (case and spacing ignored) across all files. With --max_per_prompt N,
    at most N answers to the same first user message: OpenThoughts3 has 16 answers per question, OpenMathReasoning
    several, and OpenThoughts3's code questions come partly from OpenCodeReasoning
  - benchmarks: --benchmark NAME=DIR (repeatable) runs the test of exealm/data/decontam.py. DIR is held-out text from
    exealm/data/benchmarks.py, packed by exealm/data/pack.py with the same tokenizer file. A conversation that shares
    --min_matches windows of --ngram tokens with one benchmark is dropped; a window stays inside one message, as it
    stays inside one document in decontam.py
Buckets are filled in the order of the table, so a row that repeats one of an earlier bucket counts against the later.

Output, in --out: mix.jsonl.gz, one conversation per line in post.py's format plus its `bucket` and file `name`, in
random order (sft.py holds out the last --val_frac as validation), and mix.report.json: rows and tokens per bucket and
per file, target and achieved shares, why rows were dropped (in post.py and here), and the options. Every row of every
file is read and tokenized once, so format about as many rows as the mix needs (post.py --max_rows), not whole datasets.
The chosen rows are held in memory, as sft.py holds them.
"""
import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from itertools import chain

import numpy as np

from .decontam import PackedDocuments, window_hashes
from .post import SAFETY

BUCKETS = {
    # bucket: share (DATA.md, Stage 1), the post.py files that fill it
    "chat": (0.35, ("tulu3-sft", "smoltalk", "dolci-sft")),
    "math": (0.20, ("openmath", "openthoughts3-math")),
    "code": (0.20, ("opencode", "openthoughts3-code")),
    "science": (0.10, ("openthoughts3-science",)),
    "safety": (0.05, ("tulu3-safety",)),
    "real_users": (0.05, ()),
    "tool_use": (0.05, ()),
}
FILE_BUCKET = {name: b for b, (_, names) in BUCKETS.items() for name in names}
NOT_FORMATTED = {
    "real_users": "allenai/WildChat-1M is not formatted yet: it needs the personal-data and toxicity filter first",
    "tool_use": "no verified open set yet (DATA.md)",
}
NOT_IN_MIX = {
    "IFM/SFT-Reasoning, IFM/Math-Reasoning, IFM/Code-Reasoning": "listed in DATA.md's Stage 1 table, but each row is "
        "one text with the turns run together, so they stay pretraining text (exealm/data/formatters.py)",
    "dolly-ja": "formatted by post.py, but not in DATA.md's Stage 1 table",
}


def bucket_of(name, rec):
    """The bucket of a row of file `name`: the safety subsets of tulu3-sft go to safety, other rows to their file's."""
    if name == "tulu3-sft" and rec.get("source") in SAFETY:
        return "safety"
    return FILE_BUCKET[name]


def plan(present, tokens, missing):
    """Token target per bucket. present: the buckets that have rows. missing: "empty" or "redistribute"."""
    if missing not in ("empty", "redistribute"):
        raise ValueError(f"missing must be 'empty' or 'redistribute', not {missing!r}")
    shares = {b: share if b in present else 0.0 for b, (share, _) in BUCKETS.items()}
    if missing == "redistribute" and sum(shares.values()):
        total = sum(shares.values())
        shares = {b: s / total for b, s in shares.items()}
    return shares, {b: s * tokens for b, s in shares.items()}


def count(roles, sizes, max_len):
    """(tokens, assistant tokens) of a conversation as sft.encode keeps it: the messages' roles and content lengths in
    tokens, cut to max_len (0: no cut). (0, 0) if the kept part holds no assistant token, which sft.encode drops."""
    pos = assistant = 0
    limit = max_len or 3 * len(sizes) + sum(sizes)
    for role, n in zip(roles, sizes):
        start, pos = pos + 2, pos + 3 + n               # the content and <|im_end|> follow <|im_start|> and the role
        if role == "assistant":
            assistant += max(0, min(pos, limit) - start)
    return (min(pos, limit), assistant) if assistant else (0, 0)


def norm(text):
    return " ".join(text.lower().split())


def digest(obj):
    return hashlib.blake2b(json.dumps(obj, ensure_ascii=False).encode("utf-8"), digest_size=8).digest()


def tokenizer_file(path):
    """The tokenizer.json behind --tokenizer: 'hub' (the published 32k tokenizer), a file or its folder."""
    if path == "hub":
        from ..hub import TOKENIZER, _download
        return _download(TOKENIZER)
    return os.path.join(path, "tokenizer.json") if os.path.isdir(path) else path


class Benchmarks:
    """The windows of --ngram tokens in packed benchmark text (exealm/data/decontam.py), found by their 64-bit hash and
    then compared token by token, so a match is an exact overlap."""

    def __init__(self, specs, n, identity):
        self.n, self.names, self.shards = n, list(specs), []
        hashes, owner, at = [np.empty(0, dtype=np.uint64)], [np.empty(0, dtype=np.int64)], [np.empty(0, dtype=np.int64)]
        for b, (name, path) in enumerate(specs.items()):
            shard = PackedDocuments(path)
            if shard.identity != identity:
                raise ValueError(f"benchmark {name} was packed with another tokenizer than the mix's")
            self.shards.append(shard)
            for start, doc in zip(shard.offsets, shard):
                h = window_hashes(doc[:-1], n)                  # no window takes the end-of-text token
                hashes.append(h)
                owner.append(np.full(len(h), b, dtype=np.int64))
                at.append(int(start) + np.arange(len(h), dtype=np.int64))
        hashes = np.concatenate(hashes)
        order = np.argsort(hashes, kind="stable")
        self.hashes, self.owner, self.at = hashes[order], np.concatenate(owner)[order], np.concatenate(at)[order]

    def hits(self, ids):
        """{benchmark: windows of these tokens that occur in it}, each window counted once per benchmark."""
        ids = np.asarray(ids, dtype=np.int64)
        h = window_hashes(ids, self.n)
        lo, hi = np.searchsorted(self.hashes, h, "left"), np.searchsorted(self.hashes, h, "right")
        out = Counter()
        for i in np.flatnonzero(hi > lo):
            found = {int(self.owner[k]) for k in range(lo[i], hi[i])
                     if np.array_equal(self.shards[self.owner[k]].tokens[self.at[k]:self.at[k] + self.n],
                                       ids[i:i + self.n])}
            out.update(self.names[b] for b in found)
        return out


def read_rows(path):
    """(line number, record or None) for every line of a post.py file; None for a line sft.py could not read."""
    from ..train.sft import _read_messages
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                rec = json.loads(line)
                msgs = _read_messages(rec)
                ok = any(m["role"] == "assistant" and m["content"].strip() for m in msgs)
            except (KeyError, TypeError, ValueError):
                ok = False
            yield i, (dict(rec, messages=msgs) if ok else None)


def measure(recs, tok, max_len, benchmarks=None, min_matches=1):
    """Per conversation: (tokens after the cut, uncut tokens, assistant tokens after the cut, conversation digest, first
    user message digest, the first benchmark it overlaps or None). The messages are tokenized in one batch."""
    texts = [m["content"] for rec in recs for m in rec["messages"]]
    encs = iter(tok.encode_batch(texts) if texts else [])
    out = []
    for rec in recs:
        msgs = rec["messages"]
        ids = [next(encs).ids for _ in msgs]
        roles, sizes = [m["role"] for m in msgs], [len(x) for x in ids]
        kept, assistant = count(roles, sizes, max_len)
        bench = None
        if benchmarks is not None:
            hits = Counter()
            for x in ids:
                hits.update(benchmarks.hits(x))
            bench = next((b for b in benchmarks.names if hits[b] >= min_matches), None)
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        out.append((kept, count(roles, sizes, 0)[0], assistant, digest([[m["role"], norm(m["content"])] for m in msgs]),
                    digest(norm(user)), bench))
    return out


def build(data, tok, tokens, missing, max_len=1024, long="cut", max_per_prompt=0, benchmarks=None, min_matches=1,
          seed=0, batch=512):
    """(the chosen records in random order, report). data: the folder of post.py files. benchmarks: a Benchmarks or
    None. The report is described in the module docstring."""
    if long not in ("cut", "drop"):
        raise ValueError(f"long must be 'cut' or 'drop', not {long!r}")
    if tokens <= 0 or max_len < 0 or max_per_prompt < 0 or min_matches < 1:
        raise ValueError("tokens and min_matches must be positive, max_len and max_per_prompt not negative")
    names = [name for _, files in BUCKETS.values() for name in files]
    files = {name: dict(bucket=FILE_BUCKET[name], file=os.path.join(data, f"{name}.jsonl.gz"), rows_read=0,
                        rows_available=0, rows=0, tokens=0, tokens_full=0, assistant_tokens=0, cut=0, unused=0,
                        dropped=Counter(), sources=Counter()) for name in names}
    # Pass 1: every row's bucket, tokens and checks, without keeping its text.
    cands = []                       # (bucket, file, line, then measure()'s tokens, digests and benchmark per row)
    for f, name in enumerate(names):
        info = files[name]
        if not os.path.exists(info["file"]):
            info["file"] = None
            continue
        rep = os.path.join(data, f"{name}.report.json")
        if os.path.exists(rep):
            with open(rep, encoding="utf-8") as fh:
                post = json.load(fh)
            info["post"] = {k: post[k] for k in ("read", "kept", "dropped", "license", "think", "max_think_chars")
                            if k in post}
        rows = []
        for line, rec in chain(read_rows(info["file"]), [(None, None)]):     # (None, None) measures the last batch
            if line is not None:
                info["rows_read"] += 1
                if rec is None:
                    info["dropped"]["unreadable row"] += 1
                    continue
                rows.append((line, rec))
            if rows and (len(rows) >= batch or line is None):
                for (n, r), m in zip(rows, measure([r for _, r in rows], tok, max_len, benchmarks, min_matches)):
                    cands.append((bucket_of(name, r), f, n, *m))
                rows = []

    present = {c[0] for c in cands}
    shares, targets = plan(present, tokens, missing)
    buckets = {b: dict(share_target=share, share_planned=shares[b], tokens_target=round(targets[b]),
                       files=[n for n in names if FILE_BUCKET[n] == b or (b == "safety" and n == "tulu3-sft")],
                       rows_available=0, rows=0, tokens=0, tokens_full=0, assistant_tokens=0, cut=0)
               for b, (share, _) in BUCKETS.items()}
    for c in cands:
        buckets[c[0]]["rows_available"] += 1
        files[names[c[1]]]["rows_available"] += 1

    # Selection: shuffle each bucket's rows and take them in order until its target is reached.
    rng = np.random.default_rng(seed)
    seen, answers, chosen = set(), Counter(), {}
    for b in BUCKETS:
        rows = [c for c in cands if c[0] == b]
        info = buckets[b]
        for k in rng.permutation(len(rows)):
            bucket, f, line, kept, full, assistant, conv, prompt, bench = rows[k]
            file = files[names[f]]
            if info["tokens"] >= targets[b]:
                file["unused"] += 1
                continue
            why = None
            if not kept:
                why = "answer starts past --max_len"
            elif long == "drop" and max_len and full > max_len:
                why = "longer than --max_len"
            elif bench is not None:
                why = f"shares text with benchmark {bench}"
            elif conv in seen:
                why = "duplicate conversation"
            elif max_per_prompt and answers[prompt] >= max_per_prompt:
                why = "more than --max_per_prompt answers to one prompt"
            if why:
                file["dropped"][why] += 1
                continue
            seen.add(conv)
            answers[prompt] += 1
            chosen[f, line] = b
            for d in (info, file):
                d["rows"] += 1
                d["tokens"] += kept
                d["tokens_full"] += full
                d["assistant_tokens"] += assistant
                d["cut"] += bool(max_len) and full > max_len
        info["short"] = bool(targets[b]) and info["tokens"] < targets[b]
        if not info["rows_available"]:
            info["missing"] = NOT_FORMATTED.get(b) or "no rows: run exealm.data.post for " + ", ".join(info["files"])

    # Pass 2: read the chosen rows again and shuffle them together.
    out = []
    for f, name in enumerate(names):
        if files[name]["file"] is None:
            continue
        for line, rec in read_rows(files[name]["file"]):
            b = chosen.get((f, line))
            if b is not None:
                files[name]["sources"][rec.get("source")] += 1
                out.append(dict(rec, bucket=b, name=name))
    out = [out[k] for k in rng.permutation(len(out))]

    total = sum(info["tokens"] for info in buckets.values())
    for info in buckets.values():
        info["share"] = info["tokens"] / total if total else 0.0
        info["share_of_budget"] = info["tokens"] / tokens
    for info in files.values():
        info["dropped"], info["sources"] = dict(info["dropped"]), dict(info["sources"].most_common())
    dropped = Counter()
    for info in files.values():
        dropped.update(info["dropped"])
        dropped.update({"post.py: " + k: v for k, v in info.get("post", {}).get("dropped", {}).items()})
    report = dict(rows=len(out), tokens=total, tokens_full=sum(i["tokens_full"] for i in buckets.values()),
                  assistant_tokens=sum(i["assistant_tokens"] for i in buckets.values()), tokens_target=round(tokens),
                  missing=missing, max_len=max_len, long=long, max_per_prompt=max_per_prompt, seed=seed,
                  benchmarks=None if benchmarks is None else dict(names=benchmarks.names, ngram=benchmarks.n,
                                                                  min_matches=min_matches),
                  dropped=dict(dropped.most_common()), buckets=buckets, files=files, not_in_mix=NOT_IN_MIX)
    return out, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/post", help="the folder of post.py files")
    ap.add_argument("--out", default="data/post/mix")
    ap.add_argument("--tokens", type=float, default=1.5e9, help="the token budget (DATA.md: about 1 to 2 billion)")
    ap.add_argument("--missing", required=True, choices=["empty", "redistribute"],
                    help="what happens to the share of a bucket with no rows")
    ap.add_argument("--tokenizer", default="hub", help="'hub' (the published 32k tokenizer), or a file or folder")
    ap.add_argument("--max_len", type=int, default=1024, help="sft.py's --max_len: longer rows are cut (0: no cut)")
    ap.add_argument("--long", choices=["cut", "drop"], default="cut", help="cut rows past --max_len, or drop them")
    ap.add_argument("--max_per_prompt", type=int, default=0, help="at most this many answers to a prompt (0: no limit)")
    ap.add_argument("--benchmark", action="append", default=[], help="NAME=packed benchmark folder (repeatable)")
    ap.add_argument("--ngram", type=int, default=13)
    ap.add_argument("--min_matches", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from .tokenizer import load
    path = tokenizer_file(a.tokenizer)
    tok = load(path)
    with open(path, "rb") as f:
        identity = hashlib.sha256(f.read()).hexdigest()
    specs = {}
    for spec in a.benchmark:
        name, sep, folder = spec.partition("=")
        if not sep or not name or not folder or name in specs:
            ap.error("--benchmark takes unique NAME=folder entries")
        specs[name] = folder
    benchmarks = Benchmarks(specs, a.ngram, identity) if specs else None
    out, report = build(a.data, tok, a.tokens, a.missing, a.max_len, a.long, a.max_per_prompt, benchmarks,
                        a.min_matches, a.seed)
    report.update(tokenizer=path, tokenizer_sha256=identity)
    os.makedirs(a.out, exist_ok=True)
    report["file"] = os.path.join(a.out, "mix.jsonl.gz")
    with gzip.open(report["file"], "wt", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "mix.report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)
    print(f"{'bucket':11s} {'target':>7s} {'share':>7s} {'rows':>9s} {'tokens':>14s}")
    for b, info in report["buckets"].items():
        note = " missing: " + info["missing"] if info.get("missing") else " short" if info["short"] else ""
        print(f"{b:11s} {info['share_target']:7.1%} {info['share']:7.1%} {info['rows']:9,d} {info['tokens']:14,d}"
              + note)
    print(f"{'total':11s} {'':7s} {'':7s} {report['rows']:9,d} {report['tokens']:14,d} of {report['tokens_target']:,d}")
    return report


if __name__ == "__main__":
    main()
