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

Every source below was checked against its live first rows (the `revision` is the commit those rows came from) and its
license is the one in the dataset card's metadata on that day: 2026-11-16 for the first eight, 2026-11-24 for the
reasoning sets and tulu3-safety. Check the card again before a release: several of these are mixtures whose parts carry
their own terms.

Reasoning sets (openmath, opencode, openthoughts3-*): the answer opens with a thinking section. Every set writes it the
same way here: <think>, a newline, the thinking, a newline, </think>, a blank line, then the answer (plain text: the
tokenizer has no thinking token). A row whose thinking never ends (the generation was cut off) is dropped. DATA.md
doubts that a 0.1B model gains from long traces, so two options choose what to keep, and the report records them:
    --think keep               the default: thinking and answer
    --think strip              the answer only
    --max_think_chars N        drop rows whose thinking is longer than N characters (both modes). Traces run about
                               3 characters per token with the 32k tokenizer (2.8 for math, 3.2 for code and science),
                               and sft.py cuts a conversation at --max_len (1024) tokens, so a kept trace much past
                               3,000 characters is trained on only in part. The median trace is 6,000 to 15,000
                               tokens (results/post_data/lengths.json)
    python -m exealm.data.post openmath --think strip --max_rows 20000 --out data/post
    python -m exealm.data.post openthoughts3-science --max_think_chars 4000 --max_rows 20000 --out data/post

OpenThoughts3 is sorted by domain (code, then math, then science), so each domain streams only its own data files. The
same goes for tulu3-safety: the safety subsets of the Tulu 3 mixture sit in its last two files. Rows from another domain
or subset are dropped and counted. exealm/data/post_mix.py samples these files into the SFT mix.

Not here, on purpose:
  IFM/SFT-Reasoning      its rows are one `text` field with the prompt and the answer run together, so the turns
                         cannot be separated; it stays pretraining text (exealm/data/formatters.py)
  allenai/WildChat-1M    real user conversations: needs the personal-data and toxicity filter first (to-do)
  tool-use data          no verified open set yet (DATA.md); rows with function calls are dropped here
  OpenMathReasoning `tir` and `genselect`: answers that run Python or pick among other answers (tool-like; not used)
  OpenCodeReasoning split_1: its rows hold no question, only an index into TACO or APPS
"""
import argparse
import gzip
import json
import os

ROLES = {"system": "system", "user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant",
         "tool": "tool"}

# The safety and refusal subsets of the Tulu 3 SFT mixture (DATA.md's safety bucket), by their `source` value.
SAFETY = ("ai2-adapt-dev/coconot_converted", "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k",
          "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k")


def shards(first, last, total):
    """Data file names data/train-<i>-of-<total>.parquet for i in first..last."""
    return [f"data/train-{i:05d}-of-{total:05d}.parquet" for i in range(first, last + 1)]


SOURCES = {
    # name: repo, config, split, kind, license (card metadata), revision checked, note. Some also have
    #   files    only these data files are streamed (the config's other files are not read)
    #   keep     (field, values): rows whose field holds another value are dropped
    #   think    the answer opens with a thinking section (--think, --max_think_chars)
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
    "tulu3-safety": dict(repo="allenai/tulu-3-sft-mixture", config="default", split="train", kind="sft",
                         license="odc-by", revision="b14afda60f", files=shards(4, 5, 6), keep=("source", SAFETY),
                         note="the CoCoNot, WildJailbreak and WildGuardMix subsets (about 111k rows, files 4 and 5)"),
    "openmath": dict(repo="nvidia/OpenMathReasoning", config="default", split="cot", kind="sft", license="cc-by-4.0",
                     revision="d3d0866475", think=True,
                     note="math; AoPS problems with long chain-of-thought answers by DeepSeek-R1 and QwQ-32B"),
    "opencode": dict(repo="nvidia/OpenCodeReasoning", config="split_0", split="split_0", kind="sft",
                     license="cc-by-4.0", revision="20a1ca19c0", think=True,
                     note="code; Python answers by DeepSeek-R1; each row has its own license field, and the 38,191 "
                          "Codeforces rows whose license is \"-\" are dropped"),
    "openthoughts3-math": dict(repo="open-thoughts/OpenThoughts3-1.2M", config="default", split="train", kind="sft",
                               license="apache-2.0", revision="61bcf9d4eb", files=shards(25, 109, 120),
                               keep=("domain", ("math",)), think=True,
                               note="answers by QwQ-32B; questions from ai2-adapt-dev/openmath-2-math; about two "
                                    "thirds of the answers were cut off and are dropped"),
    "openthoughts3-code": dict(repo="open-thoughts/OpenThoughts3-1.2M", config="default", split="train", kind="sft",
                               license="apache-2.0", revision="61bcf9d4eb", files=shards(0, 24, 120),
                               keep=("domain", ("code",)), think=True,
                               note="answers by QwQ-32B; questions from StackExchange Code Golf (CC BY-SA) and "
                                    "OpenCodeReasoning; about half of the answers were cut off and are dropped"),
    "openthoughts3-science": dict(repo="open-thoughts/OpenThoughts3-1.2M", config="default", split="train",
                                  kind="sft", license="apache-2.0", revision="61bcf9d4eb", files=shards(110, 119, 120),
                                  keep=("domain", ("science",)), think=True,
                                  note="answers by QwQ-32B; questions from StackExchange Physics (CC BY-SA) and "
                                       "organic-chemistry-questions"),
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


def question_answer(q, a):
    if not isinstance(q, str) or not q.strip() or not isinstance(a, str) or not a.strip():
        raise Drop("empty question or answer")
    return [dict(role="user", content=q), dict(role="assistant", content=a)]


def to_messages(row):
    """One SFT conversation from a row in any of five forms: `messages`, ShareGPT `conversations`,
    instruction / (input | context) / (output | response), problem / generated_solution (OpenMathReasoning), or
    input / output (OpenCodeReasoning, whose rows also carry their own `license`)."""
    if "messages" in row or "conversations" in row:
        msgs = clean_messages(row.get("messages") or row.get("conversations"))
    elif "generated_solution" in row:
        msgs = question_answer(row.get("problem"), row["generated_solution"])
    elif "input" in row and "output" in row and "instruction" not in row:
        if row.get("license") == "-":
            raise Drop("the row states no license")
        if row["input"] == "-":
            raise Drop("the question is not in the row")
        msgs = question_answer(row["input"], row["output"])
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


def split_thinking(text):
    """(thinking, answer) of an answer that opens with <think> and closes it once with </think>. Raises Drop."""
    text = text.strip()
    if not text.startswith("<think>"):
        raise Drop("no thinking section")
    if "</think>" not in text:
        raise Drop("thinking never ends (cut off)")
    if text.count("<think>") > 1 or text.count("</think>") > 1:
        raise Drop("more than one thinking section")
    thinking, answer = text[len("<think>"):].split("</think>")
    if not thinking.strip():
        raise Drop("empty thinking")
    if not answer.strip():
        raise Drop("no answer after the thinking")
    return thinking.strip(), answer.strip()


def with_thinking(msgs, think="keep", max_think_chars=None):
    """The conversation with its last answer's thinking section rewritten in our one form ("keep") or removed
    ("strip"). Raises Drop if the thinking is broken or longer than max_think_chars."""
    if think not in ("keep", "strip"):
        raise ValueError(f"think must be 'keep' or 'strip', not {think!r}")
    thinking, answer = split_thinking(msgs[-1]["content"])
    if max_think_chars is not None and len(thinking) > max_think_chars:
        raise Drop("thinking longer than --max_think_chars")
    content = f"<think>\n{thinking}\n</think>\n\n{answer}" if think == "keep" else answer
    return msgs[:-1] + [dict(role="assistant", content=content)]


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


def convert_all(rows, kind, source, max_rows=None, keep=None, think=None, max_think_chars=None):
    """(converted rows, report). Stops after max_rows kept rows. keep: (field, values) a row must match. think: None
    for sources without a thinking section, else "keep" or "strip" (with_thinking)."""
    out, report = [], dict(read=0, kept=0, dropped={})
    for row in rows:
        if max_rows is not None and report["kept"] >= max_rows:
            break
        report["read"] += 1
        try:
            if keep is not None and row.get(keep[0]) not in keep[1]:
                raise Drop(f"{keep[0]} not selected")
            rec = CONVERT[kind](row)
            if think is not None:
                rec["messages"] = with_thinking(rec["messages"], think, max_think_chars)
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
    ap.add_argument("--think", choices=["keep", "strip"], default=None,
                    help="reasoning sets: keep the thinking section (the default) or keep only the answer")
    ap.add_argument("--max_think_chars", type=int, default=None,
                    help="reasoning sets: drop rows whose thinking is longer than this many characters")
    ap.add_argument("--out", default="data/post")
    a = ap.parse_args(argv)
    if a.list or not a.name:
        for name, s in SOURCES.items():
            print(f"{name:21s} {s['kind']:5s} {s['repo']} [{s['config']}/{s['split']}] license: {s['license']}"
                  f" (checked at {s['revision']}){' -- ' + s['note'] if s['note'] else ''}")
        return None
    s = SOURCES[a.name]
    if not s.get("think") and (a.think is not None or a.max_think_chars is not None):
        ap.error(f"--think and --max_think_chars apply to the reasoning sets only, and {a.name} has no thinking")
    if a.max_think_chars is not None and a.max_think_chars < 1:
        ap.error("--max_think_chars must be positive")
    think = (a.think or "keep") if s.get("think") else None
    from datasets import load_dataset                      # pip install -e ".[data]"
    if s.get("files"):
        rows = load_dataset(s["repo"], data_files=s["files"], split=s["split"], streaming=True)
    else:
        rows = load_dataset(s["repo"], s["config"], split=s["split"], streaming=True)
    out, report = convert_all(rows, s["kind"], s["repo"], a.max_rows, s.get("keep"), think, a.max_think_chars)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.name}.jsonl.gz")
    write(path, out)
    report.update(name=a.name, repo=s["repo"], config=s["config"], split=s["split"], kind=s["kind"], license=s["license"],
                  file=path)
    if think is not None:
        report.update(think=think, max_think_chars=a.max_think_chars)
    json.dump(report, open(os.path.join(a.out, f"{a.name}.report.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    main()
