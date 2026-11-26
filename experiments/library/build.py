"""Build a distilled library: an encyclopedia of text and code written by a large open-weight LLM and packed for
ExeaLM's retrieval circuit (RG) to search, with "morph" tasks that teach the model to adapt what it retrieves.

Why: ExeaLM should rely on library knowledge and morph it: retrieve an entry with RG and adapt it to the task at hand,
not copy it and not hold it in its weights. With rare-word keys the model's own search finds the right Wikipedia
article about 30% of the time, and random web chunks barely help (results/README.md, section 6). So the library is
written for a small model: short, dense entries whose parts each name their topic, so a 64-token chunk stands alone.

Four steps, one subcommand each, run from the repo root:

    topics  a topic list, one per line. --kind text: titles or first lines of documents. --kind code: modules from
            import lines, the functions and APIs called through them, and the names of functions defined in many
            files. Sources: packed directories, .jsonl, .jsonl.gz or .parquet files, or a plain text file of topics.
    write   one entry per topic from an LLM (vLLM, transformers, or a mock for tests), appended to a JSONL file. Text
            entries come with 3 to 5 question/answer pairs. Code entries are an explanation, canonical code (Python
            must parse) and notes. Run the same line again to continue where it stopped.
    morph   3 to 5 tasks per entry that need the entry changed, not copied. Code: rename and retype, extend, handle
            an edge case, combine with a second entry, port to another style, fix a buggy variant. Text: combine two
            facts, restate for another reader, compute from stated numbers.
    pack    the entries tokenized into a packed library (one entry, one document) for --datastore_data, and the
            question/answer pairs and tasks packed as documents with their target spans, split into train and held-out
            by topic.

    python experiments/library/build.py topics data/tokpack/bpe32k/store_wiki --shuffle --limit 20000 \\
        --out data/library/topics.txt
    python experiments/library/build.py topics data/tokpack/bpe32k/train_code --kind code --limit 5000 \\
        --out data/library/code_topics.txt
    python experiments/library/build.py write --topics data/library/topics.txt --model MODEL \\
        --out data/library/entries.jsonl
    python experiments/library/build.py write --kind code --topics data/library/code_topics.txt --model MODEL \\
        --out data/library/code_entries.jsonl
    python experiments/library/build.py morph data/library/entries.jsonl data/library/code_entries.jsonl \\
        --model MODEL --out data/library/tasks.jsonl
    python experiments/library/build.py pack data/library/entries.jsonl data/library/code_entries.jsonl \\
        --tasks data/library/tasks.jsonl --out data/library/v1

--backend mock writes deterministic fake entries and tasks with no model, for tests. experiments/library/score.py scores
a checkpoint on the held-out pairs and tasks. experiments/library/README.md has the recipe for the GPU box.
"""
import argparse
import ast
import datetime
import difflib
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Change a version whenever its prompt changes, so a pack can tell entries and tasks apart
PROMPT_VERSION = "v1"                   # text entries
GROUNDED_PROMPT_VERSION = "v2-grounded" # text entries written from an excerpt of the topic's own document
CODE_PROMPT_VERSION = "code-v1"         # code entries
MORPH_PROMPT_VERSION = "morph-v1"       # morph tasks
DEFAULT_TOKENIZER = "results/tokenizer/bpe32k"
CJK = "".join(f"{chr(a)}-{chr(b)}" for a, b in ((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF)))
JA = re.compile(f"[{CJK}]")                                       # kana and kanji
RECORD_FILES = (".jsonl", ".jsonl.gz", ".json.gz", ".parquet")


def topic_key(topic):
    """The form duplicates are found by: Unicode-normalized, case-folded, single spaces."""
    return " ".join(unicodedata.normalize("NFKC", topic).casefold().split())


def topic_hash(topic):
    return int.from_bytes(hashlib.sha256(topic_key(topic).encode("utf-8")).digest()[:8], "big")


# Topics from text

LEAD_CUTS = (" (", "(", "（", " is ", " was ", " are ", " were ", " refers to ", ",", "とは", "は、", "、")
SENTENCE_END = (".", "。", "!", "?", "！", "？", ":", "：")
PRONOUNS = {"this", "that", "these", "those", "it", "he", "she", "they", "there", "here", "we", "i", "you", "which"}


def topic_from_text(text, max_words=8, max_chars=60):
    """A topic from the start of a document: its first line if that looks like a title, else the subject of its
    first sentence ("Albert Einstein (14 March 1879 - 18 April 1955) was ..." gives "Albert Einstein", "東京都（とう
    きょうと）は、..." gives "東京都"). None when neither works, and for disambiguation pages."""
    lines = [ln.strip() for ln in text.replace(chr(0xFFFD), "").splitlines() if ln.strip()]
    if not lines or re.search(r"may (also )?refer to", " ".join(lines[:2])):
        return None

    def fits(t):
        return bool(t) and len(t.split()) <= max_words and len(t) <= max_chars

    first = lines[0].strip("#=* ")
    # A title has a line after it (else the peek may have cut a sentence short) and no sentence punctuation
    if len(lines) > 1 and fits(first) and not first.endswith(SENTENCE_END) and not re.search("[。、]", first):
        return first
    lead = re.sub(r"^In [^,]{1,40}, (an? )?", "", first)            # "In mathematics, a group is ..." -> "group"
    cuts = [i for i in (lead.find(m) for m in LEAD_CUTS) if i > 0]
    if not cuts:
        return None
    t = lead[:min(cuts)].strip(" \"'「」『』*")
    return t if fits(t) and t.casefold() not in PRONOUNS else None


# Topics from code

LANG_NAMES = {"python": "Python", "javascript": "JavaScript", "java": "Java", "go": "Go", "cpp": "C/C++",
              "rust": "Rust"}
TAG_ALIASES = {"py": "python", "python3": "python", "js": "javascript", "jsx": "javascript", "ts": "typescript",
               "c++": "cpp", "golang": "go", "rs": "rust", "sh": "bash", "shell": "bash"}
HASH_COMMENT = {"python", "ruby", "r", "bash", "perl", "yaml", "toml"}
TASK_SKIP = {"test", "get", "set", "on", "handle", "init", "setup", "teardown", "main"}


def comment_mark(tag):
    return "#" if tag in HASH_COMMENT else "--" if tag in ("sql", "lua", "haskell") else "//"


def words_of(name):
    """An identifier as lower-case words: "readCsvFile" and "read_csv_file" both give ["read", "csv", "file"]."""
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").lower().split()


def task_name(name):
    """A function name as a recurring task ("read_csv_file" -> "read csv file"), or None for private, one-word and
    accessor or test names, which say little about a task."""
    w = words_of(name)
    return None if name.startswith("_") or len(w) < 2 or w[0] in TASK_SKIP else " ".join(w)


def dotted_calls(text):
    """Every called name with a dot in it: "os.path.join(" gives "os.path.join"."""
    return re.findall(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*\(", text)


def code_facts(text):
    """(language tag, modules, APIs, task names) that one source file imports, calls or defines. The language is
    None when no import or definition is recognised. Checked in an order where one language's syntax cannot pass for
    the next one's: Go, Java, Rust, C/C++, JavaScript, Python."""
    mods, apis, tasks = set(), set(), set()

    def defs(pattern):
        tasks.update(filter(None, (task_name(n) for n in re.findall(pattern, text, re.M))))

    if re.search(r"^package \w+", text, re.M) and re.search(r'^import\s*[("]', text, re.M):
        paths = re.findall(r'^\s*(?:import\s+)?(?:\w+\s+)?"([\w./-]+)"\s*$', text, re.M)
        last = {p.rsplit("/", 1)[-1]: p for p in paths}
        mods.update(paths)
        apis.update(f"{last[a]}.{f}" for a, f in re.findall(r"\b(\w+)\.([A-Z]\w*)\s*\(", text) if a in last)
        defs(r"^func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(")
        return "go", mods, apis, tasks
    java = re.findall(r"^import\s+(?:static\s+)?(\w+\.[\w.]+?)(\.\*)?\s*;", text, re.M)
    if java and re.search(r"\b(?:class|interface|enum|record)\s+\w+", text) and "{" in text:
        for imp, star in java:
            mods.add(imp if star else imp.rsplit(".", 1)[0])
            if not star:
                apis.add(imp)
        return "java", mods, apis, tasks
    rust = re.findall(r"^\s*(?:pub\s+)?use\s+((?:\w+::)*\w+)", text, re.M)
    if rust and re.search(r"\bfn\s+\w+", text):
        for path in rust:
            parts = path.split("::")
            if parts[0] in ("crate", "self", "super"):
                continue
            mods.add("::".join(parts[:2]))
            if len(parts) > 2:
                apis.add(path)
        defs(r"^\s*(?:pub\s+)?fn\s+(\w+)")
        return "rust", mods, apis, tasks
    includes = re.findall(r"^\s*#\s*include\s*<([\w./+-]+)>", text, re.M)
    if includes:
        mods.update(includes)
        return "cpp", mods, apis, tasks
    js = re.findall(r"""(?:^\s*import\b[^'"\n]*?\bfrom\s*|\brequire\(\s*)['"]([^'"]+)['"]""", text, re.M)
    if js:
        mods.update(m for m in js if not m.startswith("."))
        alias = dict(re.findall(r"""^\s*import\s+(\w+)\s+from\s*['"]([^'"]+)['"]""", text, re.M)
                     + re.findall(r"""\b(?:const|let|var)\s+(\w+)\s*=\s*require\(\s*['"]([^'"]+)['"]""", text))
        for chain in dotted_calls(text):
            head, rest = chain.split(".", 1)
            if head in alias and not alias[head].startswith("."):
                apis.add(f"{alias[head]}.{rest}")
        defs(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(")
        return "javascript", mods, apis, tasks
    if re.search(r"^package\s+[\w.]+;?\s*$", text, re.M):                # Kotlin, Scala: not read here
        return None, mods, apis, tasks
    name = r"[\w.]+(?:[ \t]+as[ \t]+\w+)?"                                   # "numpy" or "numpy as np"
    plain = re.findall(rf"^[ \t]*import[ \t]+({name}(?:[ \t]*,[ \t]*{name})*)[ \t]*(?:#.*)?$", text, re.M)
    froms = re.findall(r"^[ \t]*from[ \t]+([\w.]+)[ \t]+import[ \t]+\(?([\w \t,*]+)", text, re.M)
    # Python needs a "from ... import" or a "def ...:" line: Swift and Haskell also have bare "import X" lines
    if froms or re.search(r"^[ \t]*(?:async[ \t]+)?def \w+\s*\(.*\)\s*(?:->.*)?:", text, re.M):
        alias = {}
        for group in plain:
            for part in group.split(","):
                bits = part.split()
                mods.add(bits[0])
                if len(bits) == 3:                                     # import numpy as np
                    alias[bits[2]] = bits[0]
                else:                                                  # import os.path binds "os"
                    alias[bits[0].split(".")[0]] = bits[0].split(".")[0]
        for mod, names in froms:
            if mod.startswith("."):                                    # relative: the project's own code
                continue
            mods.add(mod)
            for n in names.split(","):
                if n.split() and n.split()[0] != "*":
                    apis.add(f"{mod}.{n.split()[0]}")
        for chain in dotted_calls(text):
            head, rest = chain.split(".", 1)
            if head in alias:
                apis.add(f"{alias[head]}.{rest}")
        defs(r"^[ \t]*def (\w+)\s*\(")
        return "python", mods, apis, tasks
    return None, mods, apis, tasks


def private(name):
    return any(p.startswith("_") for p in re.split(r"[./]|::", name))


def code_topics(texts, min_docs=2):
    """Code topics ranked by the number of files that use them, one kind at a time in turn (a module, an API, a task,
    the next module, ...), so every kind is represented. A topic reads "os.path.join (Python)"."""
    counts = {"module": Counter(), "api": Counter(), "task": Counter()}
    for text in texts:
        lang, mods, apis, tasks = code_facts(text)
        if lang is None:
            continue
        for kind, found in (("module", mods), ("api", apis), ("task", tasks)):
            counts[kind].update({f"{x} ({LANG_NAMES[lang]})" for x in found if x and not private(x)})
    ranked = [[t for t, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])) if n >= min_docs]
              for c in counts.values()]
    return [t for group in itertools.zip_longest(*ranked) for t in group if t]


def code_lang_of(topic, default="python"):
    """The language tag of a code topic: from its "(Python)" ending if it has one, else the default."""
    m = re.search(r"\(([^()]+)\)\s*$", topic)
    names = {v.casefold(): k for k, v in LANG_NAMES.items()}
    if m and m.group(1).strip().casefold() in names:
        return names[m.group(1).strip().casefold()]
    return default


# Reading sources

class Bytes:
    """--tokenizer bytes, as in exealm/data/pack.py: UTF-8 bytes are tokens 0..255, end-of-text is 256."""

    def get_vocab_size(self):
        return 257

    def token_to_id(self, name):
        return 256 if name == "<|endoftext|>" else None

    def encode_batch(self, texts):
        return [SimpleNamespace(ids=list(t.encode("utf-8"))) for t in texts]

    def decode(self, ids):
        return bytes(int(i) for i in ids if i < 256).decode("utf-8", errors="ignore")


def load_tokenizer(name):
    """The packing tokenizer: a trained tokenizer directory or tokenizer.json (exealm/data/tokenizer.py), or "bytes"."""
    if name == "bytes":
        return Bytes()
    if not os.path.exists(name):
        raise SystemExit(f"no tokenizer at {name}. The 32k tokenizer goes in {DEFAULT_TOKENIZER} "
                         "(experiments/gpu_session/setup.sh puts it there), or pass --tokenizer")
    from exealm.data.tokenizer import load
    return load(name)


def pack_starts(path, tok, peek_tokens, min_doc_tokens, max_docs=0, seed=0):
    """The first peek_tokens tokens of every document of a packed directory, decoded. max_docs > 0: a seeded random
    sample of that many documents instead (scanning every file of a 7.5B-token code pack for its imports takes hours
    on one core)."""
    from exealm.data.loader import TokenShards
    if not os.path.exists(os.path.join(path, "meta.json")):
        raise SystemExit(f"{path} is a directory but not a packed one (no meta.json)")
    shards = TokenShards(path)
    ends = np.append(shards.offsets[1:], len(shards.tokens))
    order = range(len(shards.offsets))
    if max_docs and max_docs < len(shards.offsets):
        order = sorted(np.random.default_rng(seed).choice(len(shards.offsets), max_docs, replace=False).tolist())
    for i in order:
        a, b = int(shards.offsets[i]), int(ends[i])
        if b - a >= max(1, min_doc_tokens):
            yield tok.decode([int(t) for t in shards.tokens[a:min(b, a + peek_tokens)]])


def records(path):
    """(title, text) of every record of a .jsonl, .jsonl.gz or .parquet file; title is None when there is none."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        cols = [c for c in ("title", "text") if c in pf.schema_arrow.names]
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg, columns=cols).to_pylist():
                yield row.get("title"), row.get("text") or ""
        return
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                yield rec.get("title"), rec.get("text") or ""


def read_lines(path):
    """The non-empty lines of a text file, skipping lines that start with #."""
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def read_jsonl(paths):
    """The JSON objects of one or more JSONL files. Lines that do not parse (a run killed while writing) are skipped."""
    out = []
    for p in [paths] if isinstance(paths, str) else paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    out.append(r)
    return out


def is_data(src):
    return os.path.isdir(src) or src.endswith(RECORD_FILES)


def documents(src, tok, peek_tokens, min_doc_tokens, max_docs=0, seed=0):
    """(title, text) of every document of a packed directory (or max_docs of them, sampled) or a records file."""
    if os.path.isdir(src):
        for text in pack_starts(src, tok, peek_tokens, min_doc_tokens, max_docs, seed):
            yield None, text
    else:
        yield from records(src)


def write_lines(path, lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(ln + "\n" for ln in lines)


def cmd_topics(a):
    tok = load_tokenizer(a.tokenizer) if any(os.path.isdir(s) for s in a.sources) else None
    peek = a.peek_tokens or (4096 if a.kind == "code" else 64)
    found = []
    if a.kind == "code":                       # the user's own lists first, then the mined topics, most used first
        found += [t for s in a.sources if not is_data(s) for t in read_lines(s)]
        found += code_topics((text for s in a.sources if is_data(s)
                              for _, text in documents(s, tok, peek, a.min_doc_tokens, a.max_docs, a.seed)), a.min_docs)
    else:
        if a.source_tokens:                       # keep the start of each document, to write its entry from
            peek = max(peek, a.source_tokens)
        for s in a.sources:
            if not is_data(s):
                found += read_lines(s)                                # the user's own topics, taken as they are
                continue
            for title, text in documents(s, tok, peek, a.min_doc_tokens, a.max_docs, a.seed):
                t = title.strip() if title and title.strip() else topic_from_text(text, a.max_words)
                found.append((t, text) if a.source_tokens else t)
    seen, topics, source = set(), [], {}
    for item in found:
        t, text = item if isinstance(item, tuple) else (item, None)
        t = " ".join((t or "").split())
        if t and topic_key(t) not in seen:
            seen.add(topic_key(t))
            topics.append(t)
            source[t] = text
    if a.shuffle:
        random.Random(a.seed).shuffle(topics)
    if a.limit:
        topics = topics[:a.limit]
    if a.source_tokens:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            for t in topics:
                f.write(json.dumps(dict(topic=t, source=source.get(t) or ""), ensure_ascii=False) + "\n")
    else:
        write_lines(a.out, topics)
    print(f"{len(topics)} topics -> {a.out}")
    return topics


# Prompts and parsers

def marker(word):
    """A section marker such as "ENTRY:" or "**TARGET:**" at the start of a line. It takes the rest of its own line
    and one line break, never the indentation of the next line (code)."""
    return re.compile(rf"^[ \t#*_]*{word}[ \t*_]*:[ \t*_]*\n?", re.I | re.M)


ENTRY_MARK, QUESTIONS_MARK = marker("ENTRY"), marker("QUESTIONS(?: AND ANSWERS)?")
EXPLANATION_MARK, CODE_MARK, NOTES_MARK = marker("EXPLANATION"), marker("CODE"), marker("NOTES")
INSTRUCTION_MARK, CONTEXT_MARK, TARGET_MARK = marker("INSTRUCTION"), marker("CONTEXT"), marker("TARGET")
TASK_MARK = re.compile(r"^[ \t#*_]*TASK[ \t*_]*:[ \t*_]*([\w-]*)[^\n]*\n?", re.I | re.M)
END_LINE = re.compile(r"^[ \t*_]*END[ \t*_]*$", re.M)
QA_LINE = re.compile(r"^[\s#*_\-\d.)]*([QA])\d*[\s*_]*[:：][\s*_]*(.*)$", re.I)
FENCE = re.compile(r"```[ \t]*([\w+#.-]*)[^\n]*\n(.*?)\n?[ \t]*```", re.S)
THINK = re.compile(r"<think>.*?</think>", re.S)
NOTHING = {"", "nothing", "(nothing)", "<nothing>", "none", "(none)", "n/a", "-", "empty", "(empty)"}

SYSTEM = ("You write entries for a reference library that a small language model searches while it reads. Write plain, "
          "dense, factual text. State only facts you are sure of: leave out a number or date rather than guess it.")
PROMPT = "\n".join([
    "Topic: {topic}",
    "",
    "Write an encyclopedia entry about this topic in {language}, {length}.",
    "- Begin with a one-sentence definition that names the topic in full.",
    "- Give the key facts, with numbers, dates, places and names.",
    "- Say how the topic relates to other topics, and name them.",
    "- If the topic is mathematical, scientific or technical, include one short worked example.",
    "- Write short declarative sentences in plain paragraphs: no headings, lists, tables or markdown, no opinions, "
    "no filler.",
    "- Name the topic again at the start of every paragraph, so each paragraph makes sense on its own.",
    "- If the topic has several meanings, write about the most common one and say which one in the first sentence.",
    "",
    "Then write 3 to 5 questions about the topic, also in {language}. Each question names the topic, so it makes "
    "sense on its own. Each answer is short (a name, number, date or term) and copied word for word from the entry.",
    "",
    "Use exactly this format, with the markers in English:",
    "ENTRY:",
    "<the entry>",
    "QUESTIONS:",
    "Q: <question>",
    "A: <answer>",
    "",
    "If you know nothing reliable about the topic, write only: UNKNOWN",
])

GROUNDED_SYSTEM = ("You write entries for a reference library that a small language model searches while it reads. "
                   "You write only from the source text you are given: every fact in the entry is stated in it.")
GROUNDED_PROMPT = PROMPT.replace(
    "Topic: {topic}\n",
    "Topic: {topic}\n\nSource, the start of the document this topic comes from:\n<<<\n{source}\n>>>\n").replace(
    "Write an encyclopedia entry about this topic in {language}, {length}.",
    "Write an encyclopedia entry about this topic in {language}, {length}, using only facts stated in the source. "
    "Leave out anything the source does not say, even if you believe it: no outside numbers, dates, names or places.").replace(
    "- Give the key facts, with numbers, dates, places and names.",
    "- Give the key facts the source states, with its numbers, dates, places and names.").replace(
    "If you know nothing reliable about the topic, write only: UNKNOWN",
    "If the source says too little to write the entry, write only: UNKNOWN")
assert GROUNDED_PROMPT.count("{source}") == 1

CODE_SYSTEM = ("You write entries for a code reference library that a small language model searches while it writes "
               "code. Be exact and brief, and show only code you are sure is correct.")
CODE_PROMPT = "\n".join([
    "Topic: {topic}",
    "",
    "Write a reference entry about this {language} topic in three parts.",
    "- EXPLANATION: two to four short sentences: what it is, what it is for, and when to use it.",
    "- CODE: the canonical way to use it, as one complete function or short pattern in a ```{tag} block. Make it "
    "runnable as written where possible, using only the standard library or the package the topic names. Begin the "
    "block with a comment that names the topic, and put a one-line comment before each function.",
    "- NOTES: short sentences on the parameters, the return value and common pitfalls.",
    "Begin the explanation and the notes with the topic name, so each part makes sense on its own. No headings or "
    "markdown outside the code block.",
    "",
    "Use exactly this format, with the markers in English:",
    "EXPLANATION:",
    "<the explanation>",
    "CODE:",
    "```{tag}",
    "<the code>",
    "```",
    "NOTES:",
    "<the notes>",
    "",
    "If the topic is not a {language} library, module, function or task you know well, write only: UNKNOWN",
])

MORPH_SYSTEM = ("You write training tasks for a small language model that looks things up in a reference library and "
                "adapts what it finds. Every task must need the entry's content changed, never copied.")
MORPH_FORMAT = [
    "",
    "Use exactly this format for each task, with the markers in English:",
    "TASK: <kind>",
    "INSTRUCTION: <the instruction>",
    "CONTEXT:",
    "<the context, or nothing>",
    "TARGET:",
    "<the target>",
    "END",
]
CODE_MORPH_PROMPT = "\n".join([
    'Library entry about "{topic}":',
    "<<<",
    "{entry}",
    ">>>",
    "{second}",
    "Write {n} {language} programming tasks that each need this entry's code adapted, not copied. Use a different "
    "kind for each task, from this list:",
    "- rename: the same idea with different names and types, for a different purpose",
    "- extend: add a parameter or a feature",
    "- edge: handle an edge case the entry's code does not handle",
    "{combine}"
    "- port: the same behaviour in another style (a class instead of a function, a loop instead of a comprehension, "
    "iterative instead of recursive)",
    "- fix: CONTEXT holds a buggy variant with different names; TARGET is the corrected code",
    "For each task:",
    "- INSTRUCTION: one or two sentences, what the user asks for. Do not paste the entry.",
    "- CONTEXT: the code the user has written so far (imports and a function signature, for example), or nothing.",
    "- TARGET: the code that completes the task, continuing CONTEXT directly: CONTEXT followed by TARGET is "
    "complete, runnable {language}. Never copy more than one line in a row from the entry.",
    *MORPH_FORMAT,
])
TEXT_MORPH_PROMPT = "\n".join([
    'Library entry about "{topic}":',
    "<<<",
    "{entry}",
    ">>>",
    "",
    "Write {n} tasks in {language} that each need facts from this entry changed or combined, not copied. Use these "
    "kinds, each at least once if the entry allows:",
    "- combine: a question whose answer needs two facts from the entry put together",
    "- audience: part of the entry restated for a different reader (a child, an expert, a traveller)",
    "- compute: work something out from numbers the entry states (a difference, a total, a rate), with the working",
    "For each task:",
    "- INSTRUCTION: one sentence saying what to do.",
    "- CONTEXT: what the user wrote: a question, or nothing.",
    "- TARGET: the answer. Use only facts the entry states, and never copy a whole sentence from it.",
    *MORPH_FORMAT,
])


def lang_of(topic, lang="auto"):
    """The language to write a topic's entry in: Japanese ("ja") for a topic with kana or kanji, else English ("en"),
    unless --lang names one."""
    return lang if lang != "auto" else ("ja" if JA.search(topic) else "en")


def language_name(lang):
    return {"ja": "Japanese", "en": "English"}.get(lang) or LANG_NAMES.get(lang) or lang.capitalize()


def messages_for(topic, lang, min_words, max_words, source=None):
    """The chat for one text topic. Japanese length is in characters, about two per English word. source: an excerpt
    of the topic's own document; the entry is then written from it alone (written from the title alone, a 30B model
    invented records, dates and lab names for niche topics)."""
    length = (f"{2 * min_words} to {2 * max_words} characters long" if lang == "ja"
              else f"{min_words} to {max_words} words long")
    if source:
        return [{"role": "system", "content": GROUNDED_SYSTEM},
                {"role": "user", "content": GROUNDED_PROMPT.format(topic=topic, source=source,
                                                                   language=language_name(lang), length=length)}]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(topic=topic, language=language_name(lang), length=length)}]


def code_messages_for(topic, tag):
    return [{"role": "system", "content": CODE_SYSTEM},
            {"role": "user", "content": CODE_PROMPT.format(topic=topic, language=language_name(tag), tag=tag)}]


def morph_messages_for(entry, n, second=None):
    """The chat asking for n tasks about one entry. second: another entry of the same language and the same side of
    the train/held-out split, offered for a "combine" task (code only)."""
    if entry.get("kind") == "code":
        tag = entry.get("lang") or "python"
        extra = (f'\nSecond library entry, about "{second["topic"]}":\n<<<\n{second["entry"]}\n>>>\n'
                 if second else "")
        combine = "- combine: use this entry together with the second entry\n" if second else ""
        user = CODE_MORPH_PROMPT.format(topic=entry["topic"], entry=entry["entry"], second=extra, n=n,
                                        language=language_name(tag), combine=combine)
    else:
        user = TEXT_MORPH_PROMPT.format(topic=entry["topic"], entry=entry["entry"], n=n,
                                        language=language_name(entry.get("lang") or "en"))
    return [{"role": "system", "content": MORPH_SYSTEM}, {"role": "user", "content": user}]


def n_words(text):
    """Size in English words: Latin words and numbers, plus half a word per kana or kanji."""
    return len(re.findall(r"[^\W_]+", JA.sub(" ", text))) + len(JA.findall(text)) / 2


def norm(s):
    return " ".join(re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", s).casefold()).split())


def grounded(answer, entry):
    """Whether the answer is stated in the entry: as whole words, or (for an answer of a few words) every word longer
    than two letters, and every number, appears in it. Japanese answers must appear as written."""
    a, e = norm(answer), norm(entry)
    if not a:
        return False
    if JA.search(a):
        return a.replace(" ", "") in e.replace(" ", "")
    if f" {a} " in f" {e} ":
        return True
    words = [w for w in a.split() if len(w) > 2 or w.isdigit()]
    have = set(e.split())
    return bool(words) and all(w in have for w in words)


def clean_entry(text):
    """Markdown marks dropped, blank runs shortened: the library holds plain text."""
    lines = [re.sub(r"^#+\s*", "", ln.replace("**", "").replace("__", "")).strip() for ln in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def strip_think(text):
    return THINK.sub("", text).split("</think>")[-1].strip()      # a thinking block, closed or left open


def unknown(text):
    return not text or text.upper().startswith("UNKNOWN")


def parse_output(text, min_qa=3, max_qa=5, min_entry_words=100):
    """A text entry: (entry, qa, None) for a usable reply, else (None, None, reason). Only pairs whose answer is
    stated in the entry are kept (at most max_qa), so a held-out question can be answered from the library alone."""
    text = strip_think(text)
    if unknown(text):
        return None, None, "unknown topic"
    q = QUESTIONS_MARK.search(text)
    if q is None:
        return None, None, "no QUESTIONS: marker"
    e = ENTRY_MARK.search(text, 0, q.start())
    entry = clean_entry(text[e.end() if e else 0:q.start()])
    if n_words(entry) < min_entry_words:
        return None, None, f"entry of {n_words(entry):.0f} words (need {min_entry_words})"
    pairs, question = [], None
    for line in text[q.end():].splitlines():
        m = QA_LINE.match(line)
        if not m:
            continue
        body = m.group(2).strip().strip("*_ ")
        if m.group(1).upper() == "Q":
            question = body
        elif question and body:
            pairs.append({"q": question, "a": body.rstrip(".。 ")})
            question = None
    qa = [p for p in pairs if grounded(p["a"], entry)][:max_qa]
    if len(qa) < min_qa:
        return None, None, f"{len(qa)} of {len(pairs)} answers are stated in the entry (need {min_qa})"
    return entry, qa, None


def python_error(code):
    """None if the code parses as Python, else what is wrong."""
    try:
        ast.parse(code)
        return None
    except (SyntaxError, ValueError) as err:
        return f"{getattr(err, 'msg', err)} (line {getattr(err, 'lineno', '?')})"


def parse_code_output(text, tag="python", min_words=10):
    """A code entry: (entry, code, tag, None) for a usable reply, else (None, None, None, reason). The entry is the
    explanation, the code block and the notes. The code block's own language tag decides the language, and Python
    code must parse."""
    text = strip_think(text)
    if unknown(text):
        return None, None, None, "unknown topic"
    e = EXPLANATION_MARK.search(text)
    c = CODE_MARK.search(text, e.end() if e else 0)
    if c is None:
        return None, None, None, "no CODE: marker"
    f = FENCE.search(text, c.end())
    if f is None:
        return None, None, None, "no code block"
    n = NOTES_MARK.search(text, f.end())
    explanation = clean_entry(text[e.end() if e else 0:c.start()])
    notes = clean_entry(text[n.end():]) if n else ""
    tag = (f.group(1) or tag).lower()
    tag = TAG_ALIASES.get(tag, tag)
    code = f.group(2).strip("\n").rstrip()
    if n_words(explanation) < min_words:
        return None, None, None, f"explanation of {n_words(explanation):.0f} words (need {min_words})"
    if not notes:
        return None, None, None, "no NOTES: part"
    if not code.strip():
        return None, None, None, "empty code block"
    if tag == "python" and python_error(code):
        return None, None, None, f"python code does not parse: {python_error(code)}"
    return f"{explanation}\n\n```{tag}\n{code}\n```\n\n{notes}", code, tag, None


def unfence(s):
    """A context or target section as plain text: the inside of its code block if it has one, blank lines at both ends
    dropped, the indentation of the first line kept."""
    f = FENCE.search(s)
    if f:
        s = f.group(2)
    lines = s.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    s = "\n".join(lines).rstrip()
    return "" if s.strip().lower() in NOTHING else s


def parse_tasks(text):
    """The tasks of a morph reply: dicts with type, instruction, context and target, or with error for a block that
    is missing a part."""
    text = strip_think(text)
    marks = list(TASK_MARK.finditer(text))
    out = []
    for i, m in enumerate(marks):
        block = text[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        ins, ctx, tgt = INSTRUCTION_MARK.search(block), CONTEXT_MARK.search(block), TARGET_MARK.search(block)
        if not (ins and ctx and tgt and ins.start() < ctx.start() < tgt.start()):
            out.append(dict(error="a task without INSTRUCTION, CONTEXT and TARGET in that order", block=block))
            continue
        target = END_LINE.split(block[tgt.end():])[0]
        out.append(dict(type=m.group(1).lower(), instruction=" ".join(block[ins.end():ctx.start()].split()),
                        context=unfence(block[ctx.end():tgt.start()]), target=unfence(target)))
    return out


def task_texts(task):
    """(prefix, target): the text a task document holds before its target, and the target. Training and scoring put
    the loss on the target only.
        question/answer pairs   "Question: ...\\nAnswer:" then " <answer>"
        text tasks              the instruction, a blank line, the context and a blank line, then the target
        code tasks              the instruction as comment lines, the context, then the target, so the whole
                                document is a program (for Python it must parse)"""
    kind = task.get("kind", "text")
    if kind == "qa":
        return f"Question: {task['q']}\nAnswer:", f" {task['a']}"
    if kind == "code":
        c = comment_mark(task.get("lang") or "python")
        head = "".join(f"{c} {ln}\n" for ln in task["instruction"].splitlines() or [""])
        ctx = task["context"].rstrip()
        return head + (ctx + "\n" if ctx.strip() else ""), task["target"]
    ctx = task["context"].strip()
    return f"{task['instruction']}\n\n" + (f"{ctx}\n\n" if ctx else ""), task["target"].strip()


RUN_TOKEN = re.compile(f"[{CJK}]|[^\\W{CJK}]+|[^\\w\\s]")


def run_tokens(text):
    """Words, numbers and punctuation marks, each kana or kanji on its own: the units copies are counted in."""
    return RUN_TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


def longest_run(a, b):
    """The longest stretch of tokens a and b share, in a row."""
    return difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size


def check_task(task, entry, max_copy=0.5, max_leak=24):
    """None for a usable task, else why not. The target must change the entry, not copy it: the longest stretch it
    shares with the entry is at most max_copy of its length. The instruction and context must not carry the entry into
    the task either (at most max_leak tokens in a row), since the entry belongs in the library. Python must parse."""
    if task.get("error"):
        return task["error"]
    if not task["instruction"] or not task["target"].strip():
        return "empty instruction or target"
    prefix, target = task_texts(task)
    e, t = run_tokens(entry), run_tokens(target)
    run = longest_run(e, t)
    if run > max_copy * len(t):
        return f"target copies {run} of its {len(t)} tokens from the entry in a row"
    leak = longest_run(e, run_tokens(prefix))
    if leak > max_leak:
        return f"instruction or context copies {leak} tokens of the entry in a row"
    if task.get("kind") == "code" and (task.get("lang") or "python") == "python" and python_error(prefix + target):
        return f"python does not parse: {python_error(prefix + target)}"
    return None


# Generators. Use an open-weight model whose licence allows training on its outputs (Qwen3 models are Apache-2.0).
# Closed APIs' terms of use generally forbid using their outputs to train other models, so they are not an option.

def chat_text(tok, messages):
    """The prompt as the model's own chat template writes it. enable_thinking=False turns off Qwen3's thinking mode (an
    entry needs no long reasoning, and thinking tokens cost time); templates without that switch ignore it."""
    if not getattr(tok, "chat_template", None):
        raise SystemExit("the model's tokenizer has no chat template: use an instruct (chat) model")
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


class MockBackend:
    """Deterministic fake replies, for tests: no model, no download. Every answer is stated in its entry, every
    Python block parses, and every task passes check_task."""
    batch = 8

    def generate(self, chats):
        out = []
        for c in chats:
            system, user = c[0]["content"], c[-1]["content"]
            if system == MORPH_SYSTEM:
                out.append(mock_morph_reply(user))
            elif system == CODE_SYSTEM:
                out.append(mock_code_reply(re.search(r"^Topic: (.*)$", user, re.M).group(1),
                                           re.search(r"^```(\S+)$", user, re.M).group(1)))
            else:
                out.append(mock_reply(re.search(r"^Topic: (.*)$", user, re.M).group(1)))
        return out


def mock_numbers(topic):
    h = int(hashlib.sha256(topic.encode("utf-8")).hexdigest(), 16)
    return h, 1800 + h % 200, 10 + h % 990


def mock_reply(topic):
    h, year, n = mock_numbers(topic)
    rel = f"Mock Topic {h % 97}"
    entry = (f"{topic} is a made-up subject used to test the library pipeline. {topic} was first described in {year}. "
             f"The standard measure of {topic} is {n} units, and the record measure is {3 * n} units. {topic} is "
             f"closely related to {rel}, which was described {h % 50 + 1} years later.\n\n"
             f"{topic} is studied in three steps: measuring, comparing and recording. Each step of the study of "
             f"{topic} takes about {h % 9 + 2} days. Most work on {topic} has been done in the city of Mockford, "
             f"where the first society for {topic} opened in {year + 5}.\n\n"
             f"{topic} has a simple worked example. If one sample of {topic} measures {n} units, two samples measure "
             f"{2 * n} units together. If a third sample is added, the three samples measure {3 * n} units, which "
             f"equals the record measure of {topic}.")
    qa = [(f"When was {topic} first described?", str(year)),
          (f"What is the standard measure of {topic}?", f"{n} units"),
          (f"What is {topic} closely related to?", rel)]
    return "ENTRY:\n" + entry + "\nQUESTIONS:\n" + "".join(f"Q: {q}\nA: {ans}\n" for q, ans in qa)


def mock_name(topic):
    return "mock_" + "_".join(re.findall(r"[a-z0-9]+", topic_key(topic)))[:40]


def mock_code_reply(topic, tag):
    k = 2 + mock_numbers(topic)[0] % 9
    name = mock_name(topic)
    code = (f"# {topic}: a made-up helper used to test the library pipeline.\n"
            f"# {name}: the sum of the values, times a factor.\n"
            f"def {name}(values, factor={k}):\n"
            f"    total = 0\n"
            f"    for value in values:\n"
            f"        total += value\n"
            f"    return total * factor\n")
    return (f"EXPLANATION:\n{topic} is a made-up helper used to test the library pipeline. {topic} adds up a list of "
            f"numbers and multiplies the sum by a factor, which is {k} by default.\n"
            f"CODE:\n```{tag}\n{code}```\n"
            f"NOTES:\n{topic} takes values, a list of numbers, and factor, a number. {topic} returns 0 for an empty "
            f"list. {topic} does not check its input, so a string in the list raises a TypeError.\n")


def mock_morph_reply(user):
    topic = re.search(r'^Library entry about "(.*)":$', user, re.M).group(1)
    h, year, n = mock_numbers(topic)
    k, name = 2 + h % 9, mock_name(topic)
    if "programming tasks" in user:
        tasks = [("rename", f"Write the same helper for prices, named {name}_prices, with prices and rate.",
                  f"def {name}_prices(prices, rate={k}):",
                  "    amount = 0\n    for price in prices:\n        amount += price\n    return amount * rate"),
                 ("extend", "Add a skip_negative flag that leaves out values below zero.",
                  f"def {name}_flagged(values, factor={k}, skip_negative=False):",
                  "    kept = [v for v in values if not (skip_negative and v < 0)]\n    return sum(kept) * factor"),
                 ("edge", "Return None for an empty list instead of 0.",
                  f"def {name}_or_none(values, factor={k}):",
                  "    if not values:\n        return None\n    return sum(values) * factor")]
        second = re.search(r'^Second library entry, about "(.*)":$', user, re.M)
        if second:
            tasks.append(("combine", f"Add the results of {name} and {mock_name(second.group(1))} for two lists.",
                          f"def {name}_both(first, second):",
                          f"    return sum(first) * {k} + sum(second) * {2 + mock_numbers(second.group(1))[0] % 9}"))
    else:
        tasks = [("compute", "Answer the question and show the working.",
                  f"Question: How much larger is the record measure of {topic} than its standard measure?",
                  f"The record is {3 * n} units and the standard is {n} units, so the record is {2 * n} units "
                  "larger."),
                 ("combine", "Answer the question.",
                  f"Question: How many years after {topic} was first described did its first society open?",
                  f"It was described in {year} and the society opened in {year + 5}, five years later."),
                 ("audience", f"Explain {topic} to a ten-year-old in two sentences.", "",
                  f"{topic} is a pretend subject people made up to check that a computer program works. You can "
                  f"measure it, and a normal amount is {n} units.")]
    return "".join(f"TASK: {t}\nINSTRUCTION: {i}\nCONTEXT:\n{c}\nTARGET:\n{g}\nEND\n" for t, i, c, g in tasks)


class TransformersBackend:
    """Hugging Face transformers: the model in bfloat16, placed on the available GPUs (device_map="auto"), prompts
    generated in batches with left padding."""
    batch = 16

    def __init__(self, model, max_new_tokens, temperature, top_p, seed):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"                                  # new tokens follow every prompt directly
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="auto").eval()
        self.kw = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0, pad_token_id=self.tok.pad_token_id)
        if temperature > 0:
            self.kw.update(temperature=temperature, top_p=top_p)

    def generate(self, chats):
        import torch
        enc = self.tok([chat_text(self.tok, c) for c in chats], return_tensors="pt", padding=True,
                       add_special_tokens=False,                              # the template already has its specials
                       return_token_type_ids=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, **self.kw)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


class VLLMBackend:
    """vLLM: continuous batching, much faster than transformers for thousands of prompts. Outputs come back in the
    order of the prompts."""
    batch = 512

    def __init__(self, model, max_new_tokens, temperature, top_p, seed, tensor_parallel, max_model_len):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, dtype="bfloat16", tensor_parallel_size=tensor_parallel, seed=seed,
                       max_model_len=max_model_len)
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=temperature, top_p=top_p if temperature > 0 else 1.0,
                                     max_tokens=max_new_tokens)

    def generate(self, chats):
        outs = self.llm.generate([chat_text(self.tok, c) for c in chats], self.params)
        return [o.outputs[0].text for o in outs]


def make_backend(a):
    """(backend, the model name to record). --backend auto picks vLLM when it imports, else transformers."""
    kind = a.backend
    if kind == "auto":
        kind = "vllm" if importlib.util.find_spec("vllm") else "transformers"
    if kind == "mock":
        return MockBackend(), "mock"
    if not a.model:
        raise SystemExit("--model is required (a Hugging Face model id or a local folder)")
    if kind == "vllm":
        if importlib.util.find_spec("vllm") is None:
            raise SystemExit("vllm does not import here: install it, or use --backend transformers")
        return VLLMBackend(a.model, a.max_new_tokens, a.temperature, a.top_p, a.seed, a.tensor_parallel,
                           a.max_model_len), a.model
    return TransformersBackend(a.model, a.max_new_tokens, a.temperature, a.top_p, a.seed), a.model


# Writing entries and tasks

def read_sources(path):
    """{topic: source excerpt} of a topics .jsonl written by `topics --source_tokens N`, else {}."""
    if not path.endswith(".jsonl"):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["topic"], r.get("source") or "")
    return out


def read_topics(path):
    """The topics of a topics file (one per line, or a .jsonl of {"topic", "source"}), first of each duplicate kept."""
    if path.endswith(".jsonl"):
        return list(read_sources(path))
    seen, out = set(), []
    for t in read_lines(path):
        if topic_key(t) not in seen:
            seen.add(topic_key(t))
            out.append(t)
    return out


def finished_topics(path):
    """Topic keys already in the output file, so a stopped run picks up where it left off. A partial last line (a run
    killed while writing) is cut off first, so the next record starts on a clean line."""
    done, end = set(), 0
    if not os.path.exists(path):
        return done
    with open(path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            end += len(line)
            try:
                done.add(topic_key(json.loads(line)["topic"]))
            except (ValueError, KeyError, TypeError):
                pass
    if end < os.path.getsize(path):
        with open(path, "rb+") as f:
            f.truncate(end)
        print(f"{path}: cut off a partial last line")
    return done


def rejects_path(out):
    return (out[:-len(".jsonl")] if out.endswith(".jsonl") else out) + ".rejects.jsonl"


def append_jsonl(path, recs):
    if recs:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)


def run_batches(a, todo, make_chat, judge, what):
    """Generate for every item of todo in batches and append what judge keeps to a.out and the rest to the rejects
    file, batch by batch, so a stopped run loses at most one batch. judge(item, reply) -> (kept records, rejects)."""
    backend, model = make_backend(a)
    batch = a.batch or backend.batch
    t0, n_ok, n_bad = time.time(), 0, 0
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        replies = backend.generate([make_chat(x) for x in part])
        good, bad = [], []
        for x, reply in zip(part, replies):
            g, b = judge(x, reply, model)
            good += g
            bad += b
        append_jsonl(a.out, good)
        append_jsonl(rejects_path(a.out), bad)
        n_ok, n_bad = n_ok + len(good), n_bad + len(bad)
        dt = max(time.time() - t0, 1e-6)
        print(f"{i + len(part)}/{len(todo)} {what}: {n_ok} written, {n_bad} rejected, {(i + len(part)) / dt:.2f}/s",
              flush=True)
    return dict(written=n_ok, rejected=n_bad)


def cmd_write(a):
    topics = read_topics(a.topics)
    done = finished_topics(a.out)
    todo = [t for t in topics if topic_key(t) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(topics)} topics: {sum(topic_key(t) in done for t in topics)} already in {a.out}, "
          f"{len(todo)} to write now")
    if not todo:
        return dict(written=0, rejected=0)

    if a.kind == "code":
        def make_chat(t):
            return code_messages_for(t, code_lang_of(t, a.code_lang))

        def judge(t, reply, model):
            entry, code, tag, why = parse_code_output(reply, code_lang_of(t, a.code_lang), a.min_explanation_words)
            if why:
                return [], [dict(topic=t, reason=why, reply=reply, model=model, prompt_version=CODE_PROMPT_VERSION)]
            return [dict(topic=t, entry=entry, qa=[], model=model, prompt_version=CODE_PROMPT_VERSION, lang=tag,
                         kind="code", code=code)], []
    else:
        sources = read_sources(a.topics)
        version = GROUNDED_PROMPT_VERSION if sources else PROMPT_VERSION

        def make_chat(t):
            return messages_for(t, lang_of(t, a.lang), a.min_words, a.max_words, sources.get(t))

        def judge(t, reply, model):
            entry, qa, why = parse_output(reply, a.min_qa, a.max_qa, a.min_entry_words)
            if why:
                return [], [dict(topic=t, reason=why, reply=reply, model=model, prompt_version=version)]
            return [dict(topic=t, entry=entry, qa=qa, model=model, prompt_version=version,
                         lang=lang_of(t, a.lang), kind="text")], []
    return run_batches(a, todo, make_chat, judge, "topics")


def read_entries(paths):
    """The write step's records, the first one of each topic, in file order. kind defaults to "text"."""
    seen, out = set(), []
    for r in read_jsonl(paths):
        if not r.get("topic") or not r.get("entry") or topic_key(r["topic"]) in seen:
            continue
        seen.add(topic_key(r["topic"]))
        r.setdefault("kind", "text")
        out.append(r)
    return out


def is_heldout(topic, share, salt):
    """Whether a topic is held out: its question/answer pairs and tasks are for evaluation. A hash of the topic
    decides, so every pair and task of a topic is on one side, and adding topics never moves an old one."""
    h = hashlib.sha256(f"{salt}\n{topic_key(topic)}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64 < share


def partners(entries, share, salt):
    """For each code entry, another code entry in the same language and on the same side of the split, to combine
    with (chosen by a hash of the topic, so runs agree), or None."""
    groups = {}
    for r in entries:
        if r["kind"] == "code":
            groups.setdefault((r.get("lang"), is_heldout(r["topic"], share, salt)), []).append(r)
    out = {}
    for group in groups.values():
        for r in group:
            others = [o for o in group if o is not r]
            out[topic_key(r["topic"])] = others[topic_hash(r["topic"]) % len(others)] if others else None
    return out


def cmd_morph(a):
    entries = read_entries(a.entries)
    done = finished_topics(a.out)
    todo = [r for r in entries if topic_key(r["topic"]) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(entries)} entries: {sum(topic_key(r['topic']) in done for r in entries)} already have tasks in "
          f"{a.out}, {len(todo)} to do now")
    if not todo:
        return dict(written=0, rejected=0)
    second = partners(entries, a.heldout, a.split_salt)

    def make_chat(r):
        return morph_messages_for(r, a.tasks_per_entry, second.get(topic_key(r["topic"])))

    def judge(r, reply, model):
        tasks, bad = parse_tasks(reply), []
        if not tasks:
            return [], [dict(topic=r["topic"], reason="no TASK: blocks", reply=reply, model=model,
                             prompt_version=MORPH_PROMPT_VERSION)]
        good, other = [], second.get(topic_key(r["topic"]))
        for t in tasks[:a.max_tasks]:
            t = dict(topic=r["topic"], kind=r["kind"], **t, lang=r.get("lang"), model=model,
                     prompt_version=MORPH_PROMPT_VERSION)
            t["also_topic"] = other["topic"] if r["kind"] == "code" and t.get("type") == "combine" and other else None
            why = check_task(t, r["entry"], a.max_copy, a.max_leak)
            (bad if why else good).append(dict(t, reason=why) if why else t)
        return good, bad
    return run_batches(a, todo, make_chat, judge, "entries")


# Packing

def encode_all(tok, texts, batch=1000):
    out = []
    for i in range(0, len(texts), batch):
        out += [list(enc.ids) for enc in tok.encode_batch(texts[i:i + batch])]
    return out


def write_docs(docs, n_bytes, tok, tokenizer, out, files):
    """Pack token lists the way exealm/data/pack.py does (<|endoftext|> after each), so TokenShards, open_data and
    --datastore_data read the result. Returns meta (also in meta.json), the documents' offsets and their lengths."""
    eot = tok.token_to_id("<|endoftext|>")
    if eot is None:
        raise SystemExit("the tokenizer has no <|endoftext|> token")
    V = tok.get_vocab_size()
    dtype = np.uint16 if V <= 65535 else np.uint32
    os.makedirs(out, exist_ok=True)
    offsets, lengths, n_tok = [], [], 0
    with open(os.path.join(out, "tokens.bin"), "wb") as fo:
        for ids in docs:
            arr = np.array(ids + [eot], dtype=dtype)
            fo.write(arr.tobytes())
            offsets.append(n_tok)
            lengths.append(len(arr))
            n_tok += len(arr)
    np.save(os.path.join(out, "offsets.npy"), np.array(offsets, dtype=np.int64))
    meta = dict(dtype=np.dtype(dtype).name, vocab_size=V, n_tokens=n_tok, n_docs=len(offsets),
                bytes_per_token=n_bytes / max(1, n_tok), tokenizer=tokenizer, files=files)
    write_meta(out, meta)
    return meta, offsets, lengths


def write_pack(texts, tok, tokenizer, out, files):
    """One document per text. Returns meta and each document's length in tokens."""
    meta, _, lengths = write_docs(encode_all(tok, texts), sum(len(t.encode("utf-8")) for t in texts), tok, tokenizer,
                                  out, files)
    return meta, lengths


def write_task_pack(items, tok, tokenizer, out, files):
    """One document per item: its prefix, its target, then <|endoftext|> (task_texts). The prefix and the target are
    tokenized apart, so each target's token span is exact: targets.npy holds [start, end) in tokens.bin for every
    document, and items.jsonl the item with target_start and target_end counted from its document's start and
    target_bytes, so a loss or a score can be put on the target alone. meta.json also records the ids of a blank line
    (sep_ids) and of <|endoftext|> (eot_id), which experiments/library/score.py uses."""
    texts = [task_texts(it) for it in items]
    pre, tgt = encode_all(tok, [p for p, _ in texts]), encode_all(tok, [t for _, t in texts])
    keep = [i for i in range(len(items)) if tgt[i]]                  # a target with no tokens has nothing to score
    n_bytes = sum(len((texts[i][0] + texts[i][1]).encode("utf-8")) for i in keep)
    meta, offsets, _ = write_docs([pre[i] + tgt[i] for i in keep], n_bytes, tok, tokenizer, out, files)
    spans = np.array([[o + len(pre[i]), o + len(pre[i]) + len(tgt[i])] for o, i in zip(offsets, keep)],
                     dtype=np.int64).reshape(-1, 2)
    np.save(os.path.join(out, "targets.npy"), spans)
    meta.update(n_target_tokens=int(sum(len(tgt[i]) for i in keep)), eot_id=tok.token_to_id("<|endoftext|>"),
                sep_ids=encode_all(tok, ["\n\n"])[0])
    write_meta(out, meta)
    with open(os.path.join(out, "items.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for d, i in enumerate(keep):
            f.write(json.dumps(dict(doc=d, **items[i], target_start=len(pre[i]), target_end=len(pre[i]) + len(tgt[i]),
                                    target_bytes=len(texts[i][1].encode("utf-8"))), ensure_ascii=False) + "\n")
    return meta, len(items) - len(keep)


def write_meta(out, meta):
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def doc_text(r, title):
    return f"{r['topic']}\n\n{r['entry']}" if title else r["entry"]


TASK_FIELDS = ("topic", "kind", "type", "lang", "instruction", "context", "target", "also_topic", "model",
               "prompt_version")


def cmd_pack(a):
    if not 0 <= a.heldout <= 1:
        raise SystemExit("--heldout is a share between 0 and 1")
    recs = read_entries(a.entries)
    if not recs:
        raise SystemExit("no entries to pack")
    tok = load_tokenizer(a.tokenizer)
    files = [os.path.basename(p) for p in a.entries + (a.tasks or [])]
    for r in recs:
        r["qa"] = [p for p in (r.get("qa") or []) if p.get("q") and p.get("a")]
        r["held"] = is_heldout(r["topic"], a.heldout, a.split_salt)
    # Train topics first, then held-out ones, so a held-out item's document number never equals its own entry's even
    # in a tool that numbers the library's documents from 0 like the text it scores.
    order = [r for r in recs if not r["held"]] + [r for r in recs if r["held"]]
    meta, lengths = write_pack([doc_text(r, a.title) for r in order], tok, a.tokenizer, a.out, files)
    with open(os.path.join(a.out, "index.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for d, (r, n) in enumerate(zip(order, lengths)):
            f.write(json.dumps(dict(doc=d, topic=r["topic"], kind=r["kind"], lang=r.get("lang"),
                                    split="heldout" if r["held"] else "train", n_tokens=n, model=r.get("model"),
                                    prompt_version=r.get("prompt_version")), ensure_ascii=False) + "\n")
    doc_of = {topic_key(r["topic"]): d for d, r in enumerate(order)}
    held_of = {topic_key(r["topic"]): r["held"] for r in order}

    items = {("qa", s): [] for s in ("train", "heldout")} | {("tasks", s): [] for s in ("train", "heldout")}
    for d, r in enumerate(order):
        for p in r["qa"]:
            items["qa", "heldout" if r["held"] else "train"].append(
                dict(topic=r["topic"], kind="qa", source_doc=d, q=p["q"], a=p["a"], lang=r.get("lang"),
                     model=r.get("model"), prompt_version=r.get("prompt_version")))
    dropped = Counter()
    for t in read_jsonl(a.tasks or []):
        key, also = topic_key(t.get("topic", "")), t.get("also_topic")
        if key not in doc_of:
            dropped["its topic has no entry"] += 1
        elif also and topic_key(also) not in doc_of:
            dropped["its second topic has no entry"] += 1
        elif also and held_of[topic_key(also)] != held_of[key]:
            dropped["its second topic is on the other side of the split"] += 1
        else:
            item = {k: t.get(k) for k in TASK_FIELDS}
            item.update(source_doc=doc_of[key], also_doc=doc_of[topic_key(also)] if also else None)
            items["tasks", "heldout" if held_of[key] else "train"].append(item)
    counts = {}
    for (what, split), rows in items.items():
        if rows:                                                       # an empty pack cannot be memory-mapped
            _, empty = write_task_pack(rows, tok, a.tokenizer, os.path.join(a.out, what, split), files)
            dropped["its target has no tokens"] += empty
        counts.setdefault(what, {})[split] = dict(
            topics=len({topic_key(x["topic"]) for x in rows}), items=len(rows),
            **({"by_kind": dict(Counter(x["kind"] for x in rows))} if what == "tasks" else {}))

    writers = Counter((r.get("model") or "unknown", r.get("prompt_version") or "unknown") for r in order)
    changed = max(os.path.getmtime(p) for p in a.entries + (a.tasks or []))
    meta["source"] = dict(
        kind="distilled library", built_by="experiments/library/build.py",
        packed=datetime.date.today().isoformat(),
        entries_changed=datetime.date.fromtimestamp(changed).isoformat(),
        writers=[dict(model=m, prompt_version=v, entries=n) for (m, v), n in writers.items()],
        entries=dict(Counter(r["kind"] for r in order)), title_line=bool(a.title), heldout_share=a.heldout,
        split_salt=a.split_salt,
        library_docs=dict(train=sum(not r["held"] for r in order), heldout=sum(r["held"] for r in order)),
        qa=counts["qa"], tasks=counts["tasks"], tasks_dropped={k: n for k, n in dropped.items() if n},
        chunk_len=a.chunk_len,
        short_entries=sum(n < a.chunk_len for n in lengths),
        suggested_datastore_chunks=math.ceil(3 * meta["n_tokens"] / a.chunk_len))
    write_meta(a.out, meta)
    with open(os.path.join(a.out, "SOURCE.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(source_note(meta))
    print(json.dumps({k: meta[k] for k in ("n_docs", "n_tokens", "vocab_size")}
                     | {k: meta["source"][k] for k in ("entries", "qa", "tasks", "tasks_dropped")}))
    return meta


def source_note(meta):
    """A short README for the packed directory: where the text came from and what is in it."""
    s = meta["source"]

    def side(c, split):
        x = c.get(split) or dict(items=0, topics=0)
        name = "held-out" if split == "heldout" else split
        return f"{x['items']:,} {name} ({x['topics']:,} topic{'' if x['topics'] == 1 else 's'})"

    kinds = ", ".join(f"{n:,} {k}" for k, n in s["entries"].items())
    lines = [
        "# Distilled library",
        "",
        "Encyclopedia entries (text and code) written by an LLM for ExeaLM's retrieval circuit (RG) to search, with",
        "question/answer pairs and morph tasks. Packed by experiments/library/build.py; experiments/library/README.md",
        "says how to train and evaluate with it.",
        "",
        f"- Packed on {s['packed']} from {', '.join(meta['files'])} (last changed {s['entries_changed']}).",
        *[f"- Written by {w['model']} with prompt {w['prompt_version']}: {w['entries']:,} entries."
          for w in s["writers"]],
        f"- Library: {meta['n_docs']:,} documents, one per entry ({kinds}; {s['library_docs']['train']:,} train "
        f"topics first, then {s['library_docs']['heldout']:,} held-out ones), {meta['n_tokens']:,} tokens, "
        f"{meta['bytes_per_token']:.2f} bytes per token.",
        f"- Tokenizer: {meta['tokenizer']} (vocabulary {meta['vocab_size']:,}). "
        + ("Each document starts with its topic on a line of its own." if s["title_line"] else
           "Documents are the entries alone, with no topic line."),
        f"- {s['short_entries']:,} entries are shorter than one {s['chunk_len']}-token chunk, so retrieval never "
        "sees them.",
        f"- Question/answer pairs: {side(s['qa'], 'train')} and {side(s['qa'], 'heldout')}.",
        f"- Morph tasks: {side(s['tasks'], 'train')} and {side(s['tasks'], 'heldout')}."
        + (" Dropped: " + "; ".join(f"{n:,} because {why}" for why, n in s["tasks_dropped"].items()) + "."
           if s["tasks_dropped"] else ""),
        f"- A hash of the topic picks its side (held-out share {s['heldout_share']}, salt {s['split_salt']!r}), so "
        "every pair and task of a topic is on one side.",
        f"- Suggested --datastore_chunks: {s['suggested_datastore_chunks']:,}, about three draws per token "
        "(Datastore.from_shards samples with replacement, so tokens near the ends of entries are drawn less often).",
        "",
        "Files: tokens.bin, offsets.npy and meta.json are the packed library; index.jsonl has one line per document",
        "(doc, topic, kind, lang, split, n_tokens, model, prompt_version). qa/train, qa/heldout, tasks/train and",
        "tasks/heldout are packed directories of one document per item, with targets.npy (each target's token span in",
        "tokens.bin) and items.jsonl (the item, its source_doc in the library, and its target span and bytes).",
        "",
        "The text is model output: check the model's licence before publishing the library or a model trained with it.",
        "",
    ]
    return "\n".join(lines)


def add_generation_args(p):
    p.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers", "mock"],
                   help="auto: vllm if it imports, else transformers. mock: fake replies for tests")
    p.add_argument("--batch", type=int, default=0,
                   help="prompts per generate call (0: 512 for vllm, 16 for transformers)")
    p.add_argument("--max_new_tokens", type=int, default=1536)
    p.add_argument("--temperature", type=float, default=0.7, help="0: greedy")
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="do at most this many new topics in this run (0: all)")
    p.add_argument("--tensor_parallel", type=int, default=1, help="vllm: GPUs to split the model across")
    p.add_argument("--max_model_len", type=int, default=4096, help="vllm: longest prompt plus reply, in tokens")


def add_split_args(p):
    p.add_argument("--heldout", type=float, default=0.1, help="share of topics held out for evaluation")
    p.add_argument("--split_salt", default="exealm-library", help="change it to draw a different split")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("topics", help="write a topic list, one per line")
    t.add_argument("sources", nargs="+",
                   help="packed directories (decoded with --tokenizer), .jsonl, .jsonl.gz or .parquet files with a "
                        "text field and maybe a title field, or plain text files with one topic per line")
    t.add_argument("--kind", default="text", choices=["text", "code"],
                   help="code: modules, APIs and recurring tasks mined from source files, most used first")
    t.add_argument("--out", required=True)
    t.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                   help="to decode packed directories ('bytes' for byte packs)")
    t.add_argument("--limit", type=int, default=0, help="keep at most this many topics (0: all)")
    t.add_argument("--shuffle", action="store_true", help="shuffle before --limit, so the sample spans the source")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--peek_tokens", type=int, default=0,
                   help="tokens decoded from the start of each packed document (0: 64 for text, 4096 for code)")
    t.add_argument("--min_doc_tokens", type=int, default=0, help="skip packed documents shorter than this (stubs)")
    t.add_argument("--source_tokens", type=int, default=0,
                   help="text: also keep the first N tokens of each topic's document, written as a .jsonl that `write` "
                        "grounds each entry in (use a .jsonl --out)")
    t.add_argument("--max_docs", type=int, default=0,
                   help="read a seeded random sample of this many documents per packed source (0: all of them)")
    t.add_argument("--max_words", type=int, default=8, help="text: longest topic taken from a first line")
    t.add_argument("--min_docs", type=int, default=2, help="code: a topic must appear in at least this many files")

    w = sub.add_parser("write", help="ask an LLM for one entry per topic")
    w.add_argument("--topics", required=True, help="a topics file, one per line")
    w.add_argument("--out", required=True, help="JSONL file the entries are appended to (topics in it are skipped)")
    w.add_argument("--kind", default="text", choices=["text", "code"])
    w.add_argument("--model", default=None,
                   help="Hugging Face id or local folder of an open-weight instruct model whose licence allows "
                        "training on its outputs. Text: e.g. Qwen/Qwen3-32B or Qwen/Qwen3-30B-A3B-Instruct-2507. "
                        "Code: e.g. Qwen/Qwen3-Coder-30B-A3B-Instruct or Qwen/Qwen3-30B-A3B-Instruct-2507. All "
                        "Apache-2.0. The first run downloads the weights")
    add_generation_args(w)
    w.add_argument("--min_words", type=int, default=250, help="text entry length asked for (Japanese: twice, in "
                                                              "characters)")
    w.add_argument("--max_words", type=int, default=450)
    w.add_argument("--min_entry_words", type=int, default=100, help="shorter text entries are rejected")
    w.add_argument("--min_qa", type=int, default=3, help="fewer answers stated in the text entry: rejected")
    w.add_argument("--max_qa", type=int, default=5)
    w.add_argument("--lang", default="auto", choices=["auto", "en", "ja"],
                   help="language of text entries. auto: Japanese for topics with kana or kanji, else English")
    w.add_argument("--code_lang", default="python",
                   help="code: language tag for topics without a '(Language)' ending")
    w.add_argument("--min_explanation_words", type=int, default=10, help="code: shorter explanations are rejected")

    m = sub.add_parser("morph", help="ask an LLM for tasks that need each entry adapted, not copied")
    m.add_argument("entries", nargs="+", help="JSONL files from the write step, text and code")
    m.add_argument("--out", required=True, help="JSONL file the tasks are appended to (topics in it are skipped)")
    m.add_argument("--model", default=None, help="as for write; one model can do both kinds")
    add_generation_args(m)
    add_split_args(m)
    m.add_argument("--tasks_per_entry", type=int, default=4, help="tasks asked for per entry (3 to 5)")
    m.add_argument("--max_tasks", type=int, default=5, help="keep at most this many tasks of one reply")
    m.add_argument("--max_copy", type=float, default=0.5,
                   help="reject a target whose longest stretch shared with the entry is more than this share of it")
    m.add_argument("--max_leak", type=int, default=24,
                   help="reject a task whose instruction or context shares more tokens than this in a row with the "
                        "entry")

    p = sub.add_parser("pack", help="pack the library, and the question/answer pairs and tasks with target spans")
    p.add_argument("entries", nargs="+", help="JSONL files from the write step (first record of each topic is used)")
    p.add_argument("--tasks", nargs="*", default=[], help="JSONL files from the morph step")
    p.add_argument("--out", required=True, help="the packed library directory")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="the tokenizer of the models that will read it")
    add_split_args(p)
    p.add_argument("--title", type=int, default=1, help="1: start each document with its topic on a line of its own")
    p.add_argument("--chunk_len", type=int, default=64, help="the model's retrieval chunk (cfg.rag_chunk_len)")

    a = ap.parse_args(argv)
    return {"topics": cmd_topics, "write": cmd_write, "morph": cmd_morph, "pack": cmd_pack}[a.cmd](a)


if __name__ == "__main__":
    main()
