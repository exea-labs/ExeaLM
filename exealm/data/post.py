"""Post-training data: turn open datasets into the files the three post-training loops read.

    stage            loop                     file, one JSON object per line
    1. SFT           exealm/train/sft.py      {"messages": [{"role": ..., "content": ...}], "source": ...}
    2. preferences   exealm/train/dpo.py      {"messages": [...], "chosen": "...", "rejected": "...", "source": ...}
    3. RL tasks      exealm/train/grpo.py     {"messages": [...], "answer": "...", "reward": "math", "source": ...}

    python -m exealm.data.post --list                                  # the sources, with licenses and revisions
    python -m exealm.data.post tulu3-sft --max_rows 20000 --out data/post
    python -m exealm.data.post ultrafeedback --max_rows 20000 --out data/post

Each run streams the dataset (no full download), writes data/post/<name>.jsonl.gz and a report
data/post/<name>.report.json with how many rows were read, kept and dropped, and why.

Every source below was checked against its live first rows on 2026-11-16 (the `revision` is the commit those rows came
from) and its license is the one in the dataset card's metadata on that day. Check the card again before a release:
several of these are mixtures whose parts carry their own terms.

Not here, on purpose:
  IFM/SFT-Reasoning      its rows are one `text` field with the prompt and the answer run together, so the turns
                         cannot be separated; it stays pretraining text (exealm/data/formatters.py)
  allenai/WildChat-1M    real user conversations: needs the personal-data and toxicity filter first (to-do)
  tool-use data          no verified open set yet (DATA.md); rows with function calls are dropped here
"""
import argparse
import gzip
import json
import os

ROLES = {"system": "system", "user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant",
         "tool": "tool"}

SOURCES = {
    # name: repo, config, split, kind, license (card metadata), revision checked, note
    "tulu3-sft": dict(repo="allenai/tulu-3-sft-mixture", config="default", split="train", kind="sft",
                      license="odc-by", revision="b14afda60f", note="a mixture; includes safety and refusal subsets"),
    "smoltalk": dict(repo="HuggingFaceTB/smoltalk", config="all", split="train", kind="sft",
                     license="not stated in the card metadata: read the card", revision="5feaf2fd3f", note=""),
    "dolci-sft": dict(repo="allenai/Dolci-Instruct-SFT", config="default", split="train", kind="sft",
                      license="odc-by", revision="bd3c8f3a9b", note="rows with function calls are dropped"),
    "dolly-ja": dict(repo="llm-jp/databricks-dolly-15k-ja", config="default", split="train", kind="sft",
                     license="cc-by-sa-3.0", revision="f8ae87df31", note="Japanese; instruction, context, response"),
    "tulu3-pref": dict(repo="allenai/llama-3.1-tulu-3-8b-preference-mixture", config="default", split="train",
                       kind="pref", license="odc-by", revision="78a6f00785", note=""),
    "dolci-pref": dict(repo="allenai/Dolci-Instruct-DPO", config="default", split="train", kind="pref",
                       license="odc-by", revision="aed155cf32", note=""),
    "ultrafeedback": dict(repo="HuggingFaceH4/ultrafeedback_binarized", config="default", split="train_prefs",
                          kind="pref", license="mit", revision="3949bf5f8c", note=""),
    "rlvr": dict(repo="allenai/RLVR-GSM-MATH-IF-Mixed-Constraints", config="default", split="train", kind="task",
                 license="odc-by", revision="7dbd180f54",
                 note="GSM8K and MATH training questions with few-shot prompts, and instruction-following constraints; "
                      "only the rows with a numeric or exact answer are kept (reward \"math\")"),
}


class Drop(Exception):
    """A row that cannot be used; str(e) is the reason counted in the report."""


def clean_messages(msgs):
    """[{"role", "content"}] with our role names. Raises Drop for anything the chat format cannot hold."""
    if not isinstance(msgs, (list, tuple)) or not msgs:
        raise Drop("no messages")
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            raise Drop("message is not an object")
        if m.get("function_calls") or m.get("function_call") or m.get("tool_calls"):
            raise Drop("function call")
        role = ROLES.get(m.get("role", m.get("from")))
        content = m.get("content", m.get("value"))
        if role is None:
            raise Drop("unknown role")
        if not isinstance(content, str) or not content.strip():
            raise Drop("empty content")
        out.append(dict(role=role, content=content))
    return out


def to_messages(row):
    """One SFT conversation from a row in any of three forms: `messages`, ShareGPT `conversations`, or
    instruction / (input | context) / (output | response)."""
    if "messages" in row or "conversations" in row:
        msgs = clean_messages(row.get("messages") or row.get("conversations"))
    elif "instruction" in row:
        q, extra = row["instruction"], row.get("input") or row.get("context") or ""
        a = row.get("output", row.get("response"))
        if not isinstance(q, str) or not q.strip() or not isinstance(a, str) or not a.strip():
            raise Drop("empty instruction or answer")
        msgs = [dict(role="user", content=q if not extra.strip() else f"{q}\n\n{extra}"),
                dict(role="assistant", content=a)]
    else:
        raise Drop("unknown form")
    if msgs[-1]["role"] != "assistant":
        raise Drop("does not end with an answer")
    return dict(messages=msgs)


def to_preference(row):
    """One preference from `chosen` and `rejected` given as full conversations that share everything but the last
    answer (Tulu 3, Dolci, UltraFeedback), or as two answer strings beside a `prompt`."""
    c, r = row.get("chosen"), row.get("rejected")
    if isinstance(c, str) and isinstance(r, str):
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise Drop("no prompt")
        msgs = [dict(role="user", content=row["prompt"])]
    else:
        c, r = clean_messages(c), clean_messages(r)
        if c[-1]["role"] != "assistant" or r[-1]["role"] != "assistant":
            raise Drop("does not end with an answer")
        if c[:-1] != r[:-1] or len(c) < 2:
            raise Drop("the two conversations differ before the last answer")
        msgs, c, r = c[:-1], c[-1]["content"], r[-1]["content"]
    if not c.strip() or not r.strip() or c == r:
        raise Drop("the two answers are empty or the same")
    return dict(messages=msgs, chosen=c, rejected=r)


def to_task(row):
    """One RL task with a checkable answer from an RLVR row (messages, ground_truth, dataset, constraint_type)."""
    msgs = clean_messages(row.get("messages"))
    if msgs[-1]["role"] != "user":
        raise Drop("does not end with a question")
    if row.get("constraint_type") or row.get("constraint"):
        raise Drop("instruction-following constraint (no checker yet)")
    answer = row.get("ground_truth")
    if not isinstance(answer, str) or not answer.strip():
        raise Drop("no answer")
    return dict(messages=msgs, answer=answer.strip(), reward="math")


CONVERT = dict(sft=to_messages, pref=to_preference, task=to_task)


def convert_all(rows, kind, source, max_rows=None):
    """(converted rows, report). Stops after max_rows kept rows."""
    out, report = [], dict(read=0, kept=0, dropped={})
    for row in rows:
        if max_rows is not None and report["kept"] >= max_rows:
            break
        report["read"] += 1
        try:
            rec = CONVERT[kind](row)
        except Drop as e:
            report["dropped"][str(e)] = report["dropped"].get(str(e), 0) + 1
            continue
        rec["source"] = row.get("source") or row.get("dataset") or row.get("source_dataset") or source
        out.append(rec)
        report["kept"] += 1
    return out, report


def write(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", choices=sorted(SOURCES), help="the source to convert")
    ap.add_argument("--list", action="store_true", help="print the sources and stop")
    ap.add_argument("--max_rows", type=int, default=10000, help="stop after this many kept rows")
    ap.add_argument("--out", default="data/post")
    a = ap.parse_args(argv)
    if a.list or not a.name:
        for name, s in SOURCES.items():
            print(f"{name:14s} {s['kind']:5s} {s['repo']} [{s['config']}/{s['split']}] license: {s['license']}"
                  f" (checked at {s['revision']}){' -- ' + s['note'] if s['note'] else ''}")
        return None
    from datasets import load_dataset                      # pip install -e ".[data]"
    s = SOURCES[a.name]
    rows = load_dataset(s["repo"], s["config"], split=s["split"], streaming=True)
    out, report = convert_all(rows, s["kind"], s["repo"], a.max_rows)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.name}.jsonl.gz")
    write(path, out)
    report.update(name=a.name, repo=s["repo"], config=s["config"], split=s["split"], kind=s["kind"], license=s["license"],
                  file=path)
    json.dump(report, open(os.path.join(a.out, f"{a.name}.report.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    main()
