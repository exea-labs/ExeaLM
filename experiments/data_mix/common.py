"""Paths, logging and small helpers shared by the pipeline steps."""
import json
import logging
import os
import threading
import sys
import time

log = logging.getLogger("data_mix")


def setup_logging(work, step):
    """Log to stderr and to <work>/logs/<step>.log (appended, so a resumed run keeps the earlier lines)."""
    os.makedirs(os.path.join(work, "logs"), exist_ok=True)
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stderr), logging.FileHandler(os.path.join(work, "logs", f"{step}.log"))):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.propagate = False


class Paths:
    """Where everything lives under the work directory."""

    def __init__(self, work):
        self.work = os.path.abspath(work)

    def p(self, *parts):
        return os.path.join(self.work, *parts)

    @property
    def manifest(self):
        return self.p("manifest")

    def raw(self, source, path):
        return self.p("raw", source, path)

    def extract_dir(self, pid):
        return self.p("extract", pid)

    def index(self, pid):
        return self.p("index", f"{pid}.npz")

    def tok_dir(self, pid):
        return self.p("tok", pid)

    def report(self, name):
        return self.p("reports", f"{name}.json")

    def tokenizer_dir(self, cfg=None):
        """The tokenizer of this work directory: trained once, kept for every later (larger) build, and copied into
        the pretraining repo folder by assemble."""
        return self.p("tokenizer")

    def repo_dir(self, cfg, repo):
        """<work>/repos/<repo name>/<release>: what upload sends, under <release>/ in the repo when it is set."""
        return self.p("repos", cfg["repos"][repo]["repo_id"].split("/")[-1], cfg.get("release") or "")

    @property
    def benchmarks(self):
        return self.p("benchmarks")


def slug(path):
    """A file name for a source file: its path with / as __ and no extension."""
    base = path.replace("/", "__")
    for ext in (".jsonl.gz", ".parquet", ".jsonl", ".gz"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def write_json(path, obj):
    """Atomically: a reader sees the old file or the new one, never half of one."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"        # one per writer: parallel writers never share it
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def human(n, unit=""):
    for suffix in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.3g}{suffix}{unit}"
        n /= 1000
    return f"{n:.3g}P{unit}"


def gb(n_bytes):
    return f"{n_bytes / 1e9:.2f} GB" if n_bytes >= 1e9 else f"{n_bytes / 1e6:.1f} MB"


class Progress:
    """Logs a line at most every `every` seconds: items done of total, documents, bytes or tokens, rate and ETA."""

    def __init__(self, what, total=None, every=30.0, unit="docs"):
        self.what, self.total, self.every, self.unit = what, total, every, unit
        self.t0 = self.last = time.time()
        self.items = self.n = self.size = 0

    def update(self, items=0, n=0, size=0, force=False):
        self.items += items
        self.n += n
        self.size += size
        now = time.time()
        if force or now - self.last >= self.every:
            self.last = now
            dt = max(now - self.t0, 1e-9)
            eta = ""
            if self.total and self.items:
                eta = f", ETA {(self.total - self.items) * dt / self.items / 60:.1f} min"
            tot = f"/{self.total}" if self.total else ""
            log.info(f"{self.what}: {self.items}{tot} done, {human(self.n)} {self.unit} ({human(self.n / dt)}/s), "
                     f"{human(self.size)} ({human(self.size / dt)}/s){eta}")

    def done(self):
        self.update(force=True)
        return time.time() - self.t0
