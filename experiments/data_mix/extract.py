"""Read one source file, apply its part's filters, and write the documents it keeps as Parquet.

Output columns, one row per kept document, in source order:
    text        the document: the source's text field byte for byte, or its fields or turns joined (config.py)
    source_row  its row (Parquet) or line (jsonl.gz) number in the source file, from 0
    hash        16 bytes: SHA-1 of the text with whitespace collapsed and lowercased (exealm.data.quality.doc_hash's
                normalization), so documents that differ only in spacing or case count as exact duplicates
    key         a pseudo-random 64-bit number from the hash and the config's seed. Validation and training
                documents are taken in key order, which makes them a uniform sample, and copies share a key
    n_bytes     UTF-8 bytes of the text
    meta        JSON: the source's own metadata (a JSON source's "meta" or "metadata"; chosen columns of a Parquet
                source; a conversation's turns), or null

Every file also gets a <name>.json beside it with where the file came from (repo, commit, path, size, SHA-256) and
what the filters dropped. That JSON is written last, so its presence means the file is done.
"""
import gzip
import hashlib
import json
import os
import re
import time

import pyarrow as pa
import pyarrow.parquet as pq

from exealm.data.tokenizer import SPECIAL

SCHEMA = pa.schema([("text", pa.large_string()), ("source_row", pa.int64()), ("hash", pa.binary(16)),
                    ("key", pa.uint64()), ("n_bytes", pa.int64()), ("meta", pa.large_string())])
BATCH = 4096               # documents per row group, at most
BATCH_BYTES = 64_000_000   # and at most this much text


def doc_hash(text):
    return hashlib.sha1(" ".join(text.split()).lower().encode("utf-8")).digest()[:16]


def doc_key(h, seed):
    return int.from_bytes(hashlib.blake2b(h, digest_size=8, key=str(seed).encode()).digest(), "little")


def column(table, name):
    """A column by its dotted path, whether the file stores it flat ("meta.lang") or nested (meta struct, lang)."""
    if name in table.column_names:
        return table.column(name).to_pylist()
    head, _, rest = name.partition(".")
    if rest and head in table.column_names:
        vals = table.column(head).to_pylist()
        for key in rest.split("."):
            vals = [v.get(key) if isinstance(v, dict) else None for v in vals]
        return vals
    return None


IMPORT_LINE = re.compile(r"^[ \t]*(?:from[ \t]+([\w.]+)[ \t]+import\b|import[ \t]+([\w., \t]+))", re.M)


def is_python(row):
    """A source-code document is Python by its metadata (OLMo's StarCoder: extension or repository path)."""
    md = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    ext = str(md.get("extension") or row.get("ext") or "").lstrip(".").lower()
    path = str(md.get("max_stars_repo_path") or row.get("max_stars_repo_path") or "")
    return ext == "py" or path.endswith(".py")


def imported(text):
    """Top-level module names a Python text imports (from x.y import z, import a, b.c as d)."""
    out = set()
    for m in IMPORT_LINE.finditer(text):
        if m.group(1):
            out.add(m.group(1).split(".")[0])
        else:
            for name in m.group(2).split(","):
                name = name.strip().split(" ")[0].split("\t")[0]
                if name:
                    out.add(name.split(".")[0])
    return out


class Filter:
    """A part's filter: drop reasons are counted by name. Every part drops empty text and text containing one of the
    tokenizer's special-token strings (the tokenizer would read those as control tokens)."""

    def __init__(self, spec):
        spec = spec or {}
        self.lang_in = set(spec["lang_in"]) if spec.get("lang_in") else None
        self.quality_in = set(spec["quality_in"]) if spec.get("quality_in") else None
        self.drop_subsets = set(spec.get("drop_subsets") or [])
        self.min_edu = spec.get("min_edu_score")
        self.min_chars = spec.get("min_chars") or 0
        self.py_any = set(spec.get("python_imports_any") or [])        # research_python: keep these
        self.py_none = set(spec.get("python_imports_none") or [])      # starcoder: drop what research_python keeps
        self.columns = [c for c, on in (("meta.lang", self.lang_in), ("quality_category", self.quality_in),
                                        ("subset", self.drop_subsets),
                                        ("fineweb_edu_classifier_score", self.min_edu is not None)) if on]

    def why(self, text, row):
        if not isinstance(text, str) or not text.strip():
            return "empty text"
        if len(text) < self.min_chars:
            return "shorter than min_chars"
        if any(s in text for s in SPECIAL):
            return "contains a special-token string"
        if self.lang_in is not None and row.get("meta.lang") not in self.lang_in:
            return "language"
        if self.quality_in is not None and row.get("quality_category") not in self.quality_in:
            return "quality category"
        if row.get("subset") in self.drop_subsets:
            return f"subset {row.get('subset')}"
        if self.min_edu is not None:
            s = row.get("fineweb_edu_classifier_score")
            if s is None or s < self.min_edu:
                return "edu score"
        if self.py_any and not (is_python(row) and imported(text) & self.py_any):
            return "not Python importing a scientific or ML library"
        if self.py_none and is_python(row) and imported(text) & self.py_none:
            return "Python importing a scientific or ML library (in research_python)"
        return None


class NoText(Exception):
    """A file without the text column (MegaMath's megamath-code keeps only file_info and repo_info)."""


class Drop(Exception):
    """A row that cannot become a document; str(e) is the reason counted in the file's report."""


def render(fmt, row):
    """The document text of a row (a dict of its columns or JSON keys), and extra metadata, by the part's format."""
    kind = fmt.get("kind", "text")
    if kind == "text":
        return row.get(fmt.get("field", "text")), None
    if kind == "fields":
        vals = []
        for f in fmt["fields"]:
            v = row.get(f)
            if v in (fmt.get("skip_values") or {}).get(f, []):
                raise Drop(f"{f} is {v!r}")
            if not isinstance(v, str) or not v.strip():
                raise Drop(f"no {f}")
            vals.append(v)
        return fmt.get("sep", "\n\n").join(vals), None
    if kind == "messages":
        from exealm.data.post import Drop as PostDrop, clean_messages
        try:
            msgs = clean_messages(row.get(fmt.get("field", "messages")))
        except PostDrop as e:
            raise Drop(str(e)) from e
        return fmt.get("sep", "\n\n").join(m["content"] for m in msgs), dict(messages=msgs)
    raise ValueError(f"unknown format {kind}")


def parquet_rows(path, filt, fmt, meta_columns):
    """(source_row, row dict) for every row, one row group at a time, reading only the columns needed."""
    pf = pq.ParquetFile(path)
    have = {pf.schema.column(i).path for i in range(len(pf.schema))} | set(pf.schema_arrow.names)
    missing = [c for c in filt.columns if c not in have]
    if missing:
        raise ValueError(f"{path}: the filter needs columns {missing}, which the file does not have")
    kind = fmt.get("kind", "text")
    fields = [fmt.get("field", "text")] if kind in ("text", "messages") else list(fmt["fields"])
    for f in fields:
        if f not in have:
            raise NoText(f"no column {f} (columns: {sorted(pf.schema_arrow.names)})")
    meta_cols = [c for c in meta_columns if c in have]
    want = list(dict.fromkeys(fields + filt.columns + meta_cols))
    top = list(dict.fromkeys(c if c in pf.schema_arrow.names else c.split(".")[0] for c in want))
    n = 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=top)
        cols = {c: column(t, c) for c in want}
        for i in range(t.num_rows):
            row = {c: v[i] for c, v in cols.items()}
            row["__meta__"] = {c: row[c] for c in meta_cols} if meta_cols else None
            yield n, row
            n += 1


def open_lines(path):
    """Text lines of .json.gz / .jsonl.gz / .jsonl.zst / .json.zst / .jsonl files."""
    import io
    if path.endswith(".zst") or path.endswith(".zstd"):
        import zstandard
        raw = open(path, "rb")
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def jsonl_rows(path, meta_field="meta", meta_keys=()):
    with open_lines(path) as f:
        for n, line in enumerate(f):
            rec = json.loads(line)
            meta = rec.get(meta_field)
            if meta_keys:
                meta = dict(meta or {}, **{k: rec.get(k) for k in meta_keys})
            rec["__meta__"] = meta
            yield n, rec


def extract(src, dst, provenance, part, seed=0, part_id=None, position=0.0, rank=0):
    """Write the kept documents of `src` to `dst` (Parquet) and `dst`'s .json, by the part's format, filter and metadata
    settings. Returns the .json's contents."""
    t0 = time.time()
    filt = Filter(part.get("filter"))
    fmt = part.get("fmt") or {}
    if src.endswith(".parquet"):
        rows = parquet_rows(src, filt, fmt, list(part.get("meta_columns") or []))
    else:
        rows = jsonl_rows(src, part.get("meta_field", "meta"), part.get("meta_keys") or ())
    tmp = dst + ".tmp"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    read = kept = kept_bytes = read_bytes = buffered = 0
    drops = {}
    buf = {k: [] for k in SCHEMA.names}

    note = None
    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd", compression_level=3) as w:
        def flush():
            if buf["text"]:
                w.write_table(pa.table({k: pa.array(v, type=SCHEMA.field(k).type) for k, v in buf.items()},
                                       schema=SCHEMA))
                for v in buf.values():
                    v.clear()

        try:
            first = next(rows)
        except StopIteration:
            first = None
        except NoText as e:                    # recorded and skipped: the file adds no documents
            first, note = None, str(e)
        import itertools
        for n, row in itertools.chain([first] if first else [], rows if first else []):
            read += 1
            meta = row.get("__meta__")
            try:
                text, extra = render(fmt, row)
            except Drop as e:
                drops[str(e)] = drops.get(str(e), 0) + 1
                continue
            if extra:
                meta = dict(meta or {}, **extra)
            if isinstance(text, str):
                read_bytes += len(text.encode("utf-8"))
            why = filt.why(text, row)
            if why:
                drops[why] = drops.get(why, 0) + 1
                continue
            h = doc_hash(text)
            nb = len(text.encode("utf-8"))
            buf["text"].append(text)
            buf["source_row"].append(n)
            buf["hash"].append(h)
            buf["key"].append(doc_key(h, seed))
            buf["n_bytes"].append(nb)
            buf["meta"].append(None if meta is None else json.dumps(meta, ensure_ascii=False, default=str))
            kept += 1
            kept_bytes += nb
            buffered += nb
            if len(buf["text"]) >= BATCH or buffered >= BATCH_BYTES:
                flush()
                buffered = 0
        flush()
    os.replace(tmp, dst)
    info = dict(part=part_id, provenance=provenance, rows_read=read, rows_kept=kept, text_bytes_read=read_bytes,
                text_bytes_kept=kept_bytes, dropped=drops, filter=part.get("filter") or {}, format=fmt,
                position=position, rank=rank, seconds=round(time.time() - t0, 1), extract=os.path.basename(dst),
                skipped=note)
    with open(dst[: -len(".parquet")] + ".json.tmp", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1, ensure_ascii=False)
    os.replace(dst[: -len(".parquet")] + ".json.tmp", dst[: -len(".parquet")] + ".json")
    return info


def read_rows(path, rows, columns):
    """Rows of a Parquet file (sorted row numbers), reading only the row groups that hold them. Returns a table."""
    import numpy as np
    pf = pq.ParquetFile(path)
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return pa.table({c: pa.array([], type=pf.schema_arrow.field(c).type) for c in columns})
    sizes = np.array([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    groups = np.searchsorted(starts, rows, side="right") - 1
    out = []
    for g in np.unique(groups):
        t = pf.read_row_group(int(g), columns=columns)
        out.append(t.take(pa.array(rows[groups == g] - starts[g])))
    return pa.concat_tables(out)


def row_groups(path):
    """Row count of each row group."""
    pf = pq.ParquetFile(path)
    return [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
