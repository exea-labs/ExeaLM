"""Listing and downloading source files: LLM-jp Corpus v4 from NII's GitLab, and Hugging Face datasets.

Every listing is pinned to a commit resolved once and saved under <work>/manifest, so a resumed or repeated run reads
the same files. Every download is checked against the size and SHA-256 its host publishes (the Git LFS pointer on
GitLab, the LFS metadata on Hugging Face).

LLM-jp files are Git LFS objects. The GitLab LFS batch API gives each object's download URL; the /-/raw/ URL is the
fallback. Both are retried with backoff, because the server answers some requests with 500 or times out (seen on
2026-11-24). A download resumes from its .part file when the server honours HTTP Range.

For tests, `local_hf` and `local_llmjp` point at folders laid out like the hosts (<local_hf>/<repo>/<path>,
<local_hf>/urls/<url path>, <local_llmjp>/<path>/*.jsonl.gz), and nothing is downloaded.
"""
import base64
import fnmatch
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .common import log, read_json, write_json

USER_AGENT = "exealm-data-mix/1 (python urllib)"
LOCK = threading.Lock()                     # parts are fetched in threads: one writer of the manifest at a time


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


class Retry(Exception):
    pass


def with_retries(fn, what, attempts=8, base=5.0, cap=300.0):
    """Call fn(); on a network error, a 5xx or a 429, wait (exponential backoff, or what the server asks) and retry."""
    for i in range(attempts):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code not in (408, 429, 500, 502, 503, 504) or i == attempts - 1:
                raise
            wait = min(cap, base * 2 ** i)
            if e.code == 429 and e.headers.get("Retry-After", "").isdigit():
                wait = float(e.headers["Retry-After"]) + 1
            log.warning(f"{what}: HTTP {e.code}, retry {i + 1}/{attempts - 1} in {wait:.0f} s")
        except (urllib.error.URLError, TimeoutError, ConnectionError, Retry, OSError) as e:
            if i == attempts - 1:
                raise
            wait = min(cap, base * 2 ** i)
            log.warning(f"{what}: {type(e).__name__}: {e}, retry {i + 1}/{attempts - 1} in {wait:.0f} s")
        time.sleep(wait * (0.75 + 0.5 * random.random()))


def http_json(url, data=None, headers=None, timeout=120):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), dict(r.headers)


def download_url(url, dst, size=None, sha256=None, headers=None, timeout=300, chunk=1 << 22):
    """Stream url to dst (through dst.part, resuming it when the server honours Range); check size and SHA-256."""
    part = dst + ".part"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if size is not None and have > size:
        os.remove(part)
        have = 0
    h = {"User-Agent": USER_AGENT, **(headers or {})}
    if have:
        h["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        mode = "ab" if (have and r.status == 206) else "wb"
        if mode == "wb":
            have = 0
        with open(part, mode) as f:
            while True:
                b = r.read(chunk)
                if not b:
                    break
                f.write(b)
    got = os.path.getsize(part)
    if size is not None and got != size:
        raise Retry(f"{os.path.basename(dst)}: got {got} bytes of {size}")
    if sha256:
        digest = sha256_file(part)
        if digest != sha256:
            os.remove(part)
            raise Retry(f"{os.path.basename(dst)}: SHA-256 {digest} is not {sha256}")
    os.replace(part, dst)
    return dst


class LlmjpSource:
    """LLM-jp Corpus v4 on GitLab. `ref` (a branch) is resolved to a commit the first time and kept in the manifest."""

    def __init__(self, cfg, manifest_dir, raw_dir, local=None):
        c = cfg["llmjp"]
        self.host, self.project, self.ref_name = c["host"], c["project"], c["ref"]
        self.api = f"{self.host}/api/v4/projects/{urllib.parse.quote(self.project, safe='')}/repository"
        self.manifest_dir, self.raw_dir, self.local = manifest_dir, raw_dir, local
        self._ref = None

    @property
    def ref(self):
        with LOCK:
            return self._resolve()

    def _resolve(self):
        if self._ref is None:
            path = os.path.join(self.manifest_dir, "llmjp_ref.json")
            saved = read_json(path)
            if saved:
                self._ref = saved["commit"]
            elif self.local:
                self._ref = "local"
            else:
                d, _ = with_retries(lambda: http_json(f"{self.api}/commits/{urllib.parse.quote(self.ref_name, safe='')}"),
                                    "resolve LLM-jp ref")
                self._ref = d["id"]
                write_json(path, dict(project=self.project, ref=self.ref_name, commit=d["id"],
                                      committed_date=d.get("committed_date"), resolved_at=time.strftime("%Y-%m-%d")))
                log.info(f"LLM-jp Corpus v4: {self.ref_name} is commit {d['id']}")
        return self._ref

    def list(self, path):
        """Sorted .jsonl.gz paths under `path` (cached in the manifest)."""
        cache = os.path.join(self.manifest_dir, "listings", "llmjp__" + path.replace("/", "__") + ".json")
        saved = read_json(cache)
        if saved:
            return saved["files"]
        if self.local:
            root = os.path.join(self.local, path)
            files = sorted(os.path.relpath(os.path.join(d, f), self.local).replace(os.sep, "/")
                           for d, _, fs in os.walk(root) for f in fs if f.endswith(".jsonl.gz"))
        else:
            files, page = [], 1
            while True:
                url = (f"{self.api}/tree?path={urllib.parse.quote(path)}&recursive=true&per_page=100&page={page}"
                       f"&ref={self.ref}")
                items, headers = with_retries(lambda: http_json(url), f"list {path} page {page}")
                files += [it["path"] for it in items if it["type"] == "blob" and it["path"].endswith(".jsonl.gz")]
                nxt = headers.get("X-Next-Page") or headers.get("x-next-page")
                if not nxt:
                    break
                page = int(nxt)
            files.sort()
        if not files:
            raise SystemExit(f"no .jsonl.gz files under LLM-jp {path}")
        write_json(cache, dict(source="llmjp", path=path, commit=self.ref, files=files))
        return files

    def info(self, path):
        """Size and SHA-256 of one file, from its Git LFS pointer (cached)."""
        cache = os.path.join(self.manifest_dir, "pointers", "llmjp__" + path.replace("/", "__") + ".json")
        saved = read_json(cache)
        if saved:
            return saved
        if self.local:
            full = os.path.join(self.local, path)
            rec = dict(path=path, size=os.path.getsize(full), sha256=sha256_file(full), lfs=False)
        else:
            url = f"{self.api}/files/{urllib.parse.quote(path, safe='')}?ref={self.ref}"
            d, _ = with_retries(lambda: http_json(url), f"pointer {path}")
            content = base64.b64decode(d["content"])
            rec = dict(path=path, blob_id=d["blob_id"], lfs=content.startswith(b"version https://git-lfs"))
            if rec["lfs"]:
                fields = dict(line.split(" ", 1) for line in content.decode().splitlines() if " " in line)
                rec["sha256"] = fields["oid"].split(":", 1)[1]
                rec["size"] = int(fields["size"])
            else:
                rec["sha256"], rec["size"] = d["content_sha256"], d["size"]
        write_json(cache, rec)
        return rec

    def url(self, path):
        return f"{self.host}/{self.project}/-/blob/{self.ref}/{path}"

    def fetch(self, path, attempts=10):
        """(local path, whether we downloaded it)."""
        if self.local:
            return os.path.join(self.local, path), False
        info = self.info(path)
        dst = os.path.join(self.raw_dir, path)
        if os.path.exists(dst) and os.path.getsize(dst) == info["size"]:
            return dst, True

        def lfs_href():
            body = json.dumps(dict(operation="download", transfers=["basic"],
                                   objects=[dict(oid=info["sha256"], size=info["size"])])).encode()
            d, _ = http_json(f"{self.host}/{self.project}.git/info/lfs/objects/batch", data=body,
                             headers={"Accept": "application/vnd.git-lfs+json",
                                      "Content-Type": "application/vnd.git-lfs+json"})
            obj = d["objects"][0]
            if "error" in obj:
                raise Retry(f"LFS batch: {obj['error']}")
            act = obj["actions"]["download"]
            return act["href"], act.get("header", {})

        def get():
            try:
                href, headers = lfs_href() if info.get("lfs") else (None, None)
            except (urllib.error.URLError, KeyError, Retry) as e:
                log.warning(f"LFS batch API failed for {path} ({e}); using the raw URL")
                href, headers = None, None
            if href is None:
                href, headers = f"{self.host}/{self.project}/-/raw/{self.ref}/{path}", {}
            return download_url(href, dst, info["size"], info["sha256"], headers)

        t = time.time()
        with_retries(get, f"download {path}", attempts=attempts)
        dt = time.time() - t
        log.info(f"downloaded {path}: {info['size'] / 1e6:.0f} MB in {dt:.0f} s ({info['size'] / 1e6 / max(dt, 1e-9):.1f} MB/s)")
        return dst, True

    def provenance(self, path):
        info = self.info(path)
        return dict(source="llmjp", project=self.project, revision=self.ref, path=path, size=info["size"],
                    sha256=info["sha256"], url=self.url(path))


class HfSource:
    """A Hugging Face dataset repo. The revision is resolved to a commit once per repo and kept in the manifest.

    reuse: folders that may already hold some files (the box keeps IFM shards in /work/data_raw/<repo name>/). A file
    found there with the size the Hub lists is read in place and never deleted. Gated repos (bigcode/the-stack) need
    a logged-in account that accepted their terms; huggingface_hub picks up HF_TOKEN or the saved login."""

    def __init__(self, manifest_dir, raw_dir, local=None, reuse=()):
        self.manifest_dir, self.raw_dir, self.local = manifest_dir, raw_dir, local
        self.reuse = [r for r in reuse if r]

    def revision(self, repo):
        with LOCK:
            return self._revision(repo)

    def _revision(self, repo):
        path = os.path.join(self.manifest_dir, "hf_refs.json")
        refs = read_json(path, {})
        if repo not in refs:
            if self.local:
                sha = "local"
            else:
                from huggingface_hub import HfApi
                sha = with_retries(lambda: HfApi().dataset_info(repo).sha, f"resolve {repo}")
                log.info(f"{repo}: main is commit {sha}")
            refs = read_json(path, {})
            refs[repo] = sha
            write_json(path, refs)
        return refs[repo]

    def list(self, repo, patterns):
        """Sorted files of the repo matching any of the glob patterns (* also matches /), with size and SHA-256.
        Cached in the manifest, so every later build sees the same list."""
        key = hashlib.sha1(json.dumps([repo, patterns]).encode()).hexdigest()[:10]
        cache = os.path.join(self.manifest_dir, "listings", f"hf__{repo.replace('/', '__')}__{key}.json")
        saved = read_json(cache)
        if saved:
            return saved["files"]
        rev = self.revision(repo)
        prefixes = sorted({p.split("*")[0].rsplit("/", 1)[0] if "/" in p.split("*")[0] else "" for p in patterns})
        files = {}
        for prefix in prefixes:
            if self.local:
                root = os.path.join(self.local, repo, prefix)
                for d, _, fs in os.walk(root):
                    for f in fs:
                        full = os.path.join(d, f)
                        rel = os.path.relpath(full, os.path.join(self.local, repo)).replace(os.sep, "/")
                        files[rel] = dict(path=rel, size=os.path.getsize(full), sha256=None)
            else:
                from huggingface_hub import HfApi
                kw = dict(path_in_repo=prefix) if prefix else {}
                tree = with_retries(lambda: list(HfApi().list_repo_tree(repo, repo_type="dataset", revision=rev,
                                                                        recursive=True, **kw)), f"list {repo}/{prefix}")
                for f in tree:
                    if getattr(f, "size", None) is not None:
                        files[f.path] = dict(path=f.path, size=f.size,
                                             sha256=f.lfs.sha256 if getattr(f, "lfs", None) else None)
        out = sorted((f for f in files.values() if any(fnmatch.fnmatchcase(f["path"], p) for p in patterns)),
                     key=lambda f: f["path"])
        if not out:
            raise SystemExit(f"no files in {repo} match {patterns}")
        if self.local:
            for f in out:
                f["sha256"] = sha256_file(os.path.join(self.local, repo, f["path"]))
        write_json(cache, dict(source="hf", repo=repo, patterns=patterns, revision=rev, files=out))
        return out

    def reused(self, repo, path, size=None):
        """A copy of repo/path already on this machine, under one of the reuse folders, or None."""
        name = repo.split("/")[-1]
        for root in self.reuse:
            for cand in (os.path.join(root, repo, path), os.path.join(root, name, path),
                         os.path.join(root, repo.replace("/", "__"), path), os.path.join(root, name, os.path.basename(path))):
                if os.path.exists(cand) and (size is None or os.path.getsize(cand) == size):
                    return cand
        return None

    def fetch(self, repo, path, size=None, sha256=None):
        """(local path, whether we downloaded it): a reused copy, or a download into raw_dir."""
        if self.local:
            return os.path.join(self.local, repo, path), False
        have = self.reused(repo, path, size)
        if have:
            log.info(f"reusing {have}")
            return have, False
        from huggingface_hub import hf_hub_download
        rev = self.revision(repo)
        dst = os.path.join(self.raw_dir, repo, path)
        if os.path.exists(dst) and (size is None or os.path.getsize(dst) == size):
            return dst, True
        t = time.time()
        try:
            got = with_retries(lambda: hf_hub_download(repo, path, repo_type="dataset", revision=rev,
                                                       local_dir=os.path.join(self.raw_dir, repo)),
                               f"download {repo}/{path}", attempts=10)
        except Exception as e:
            if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
                raise SystemExit(f"{repo} is gated: accept its terms on huggingface.co with the account this machine is "
                                 f"logged in as (hf auth login, or export HF_TOKEN), then rerun") from e
            raise
        dt = time.time() - t
        n = os.path.getsize(got)
        log.info(f"downloaded {repo}/{path}: {n / 1e6:.0f} MB in {dt:.0f} s ({n / 1e6 / max(dt, 1e-9):.1f} MB/s)")
        return got, True

    def provenance(self, repo, f):
        rev = self.revision(repo)
        return dict(source="hf", repo=repo, revision=rev, path=f["path"], size=f["size"], sha256=f["sha256"],
                    url=f"https://huggingface.co/datasets/{repo}/blob/{rev}/{f['path']}")


class UrlSource:
    """Files listed by URL in a text file of a Hugging Face repo: allenai/dolma keeps its file list in urls/*.txt and
    its files on olmo-data.org. The list is pinned with the repo's commit; each file's size is read from the server
    and its SHA-256 computed after download and recorded."""

    def __init__(self, hf, raw_dir, local=None):
        self.hf, self.raw_dir, self.local = hf, raw_dir, local

    def list(self, repo, list_file, contains):
        key = hashlib.sha1(json.dumps([repo, list_file, contains]).encode()).hexdigest()[:10]
        cache = os.path.join(self.hf.manifest_dir, "listings", f"urls__{repo.replace('/', '__')}__{key}.json")
        saved = read_json(cache)
        if saved:
            return saved["files"]
        if self.local:
            path = os.path.join(self.local, repo, list_file)
        else:
            from huggingface_hub import hf_hub_download
            path = with_retries(lambda: hf_hub_download(repo, list_file, repo_type="dataset",
                                                        revision=self.hf.revision(repo)), f"list {repo}/{list_file}")
        urls = sorted(u.strip() for u in open(path, encoding="utf-8") if contains in u and u.strip())
        if not urls:
            raise SystemExit(f"no URL in {repo}/{list_file} contains {contains}")
        files = [dict(path=urllib.parse.urlparse(u).path.lstrip("/"), url=u, size=None, sha256=None) for u in urls]
        write_json(cache, dict(source="urls", repo=repo, list_file=list_file, contains=contains,
                               revision=self.hf.revision(repo), files=files))
        return files

    def size(self, f):
        if f.get("size") is None and not self.local:
            req = urllib.request.Request(f["url"], method="HEAD", headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                f["size"] = int(r.headers.get("Content-Length") or 0)
        return f.get("size") or 0

    def fetch(self, f):
        if self.local:
            return os.path.join(self.local, "urls", f["path"]), False
        dst = os.path.join(self.raw_dir, f["path"])
        t = time.time()
        with_retries(lambda: download_url(f["url"], dst), f"download {f['url']}", attempts=10)
        n = os.path.getsize(dst)
        log.info(f"downloaded {f['url']}: {n / 1e6:.0f} MB in {time.time() - t:.0f} s")
        return dst, True

    def provenance(self, repo, list_file, f, local_path):
        return dict(source="urls", repo=repo, list_file=list_file, revision=self.hf.revision(repo), path=f["path"],
                    url=f["url"], size=os.path.getsize(local_path), sha256=sha256_file(local_path))
