import gzip
import hashlib
import io
import json
import os
import random
import threading
import time
from collections import namedtuple
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from exealm.data import stage as st
from exealm.data.stage import CheckError, Item, cap, fetch, hf_items, llmjp_items, plan, spread


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        srv = self.server
        data = srv.files.get(self.path)
        rng = self.headers.get("Range")
        srv.requests.append((self.path, rng))
        if data is None:
            self.send_error(404)
            return
        if self.path in srv.fail:
            srv.fail.discard(self.path)
            self.send_error(503)
            return
        start = int(rng.split("=")[1].split("-")[0]) if rng and srv.ranges else 0
        if start and start >= len(data):
            self.send_error(416)
            return
        body = data[start:]
        self.send_response(206 if start else 200)
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        cut = srv.cut.pop(self.path, None)
        body = body if cut is None else body[:cut]
        if not srv.slow:
            self.wfile.write(body)
        for i in range(0, len(body) if srv.slow else 0, 4096):
            self.wfile.write(body[i:i + 4096])
            time.sleep(srv.slow)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.files, srv.requests, srv.cut, srv.fail, srv.ranges, srv.slow = {}, [], {}, set(), True, 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def add(srv, name, data):
    """Serve data at /name and return the matching item."""
    srv.files["/" + name] = data
    return Item("test", "sub/" + name, f"http://127.0.0.1:{srv.server_port}/{name}", len(data),
                hashlib.sha256(data).hexdigest())


def blob(n, seed=0):
    return random.Random(seed).randbytes(n)


def read(path):
    with open(path, "rb") as f:
        return f.read()


def test_download_is_checked_and_listed_once(server, tmp_path):
    data = blob(300_000)
    it = add(server, "a.jsonl.gz", data)
    res = st.stage([it], str(tmp_path), workers=1, log=lambda s: None)
    assert res == {"got": 1, "have": 0, "failed": []}
    assert read(tmp_path / "sub" / "a.jsonl.gz") == data
    assert not os.path.exists(tmp_path / "sub" / "a.jsonl.gz.part")
    res = st.stage([it], str(tmp_path), workers=1, log=lambda s: None)
    assert res == {"got": 0, "have": 1, "failed": []}
    assert len(server.requests) == 1                     # the second run downloaded nothing
    lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"source": "test", "path": "sub/a.jsonl.gz", "url": it.url,
                                                     "size": len(data), "sha256": it.sha256}]


def test_a_cut_download_continues_from_the_bytes_it_has(server, tmp_path):
    data = blob(300_000, 1)
    it = add(server, "b.parquet", data)
    server.cut["/b.parquet"] = 100_000
    with pytest.raises(Exception):
        fetch(it, str(tmp_path), retries=0, block=4096)
    assert os.path.getsize(tmp_path / "sub" / "b.parquet.part") == 100_000
    assert fetch(it, str(tmp_path), retries=0, block=4096) == "got"
    assert read(tmp_path / "sub" / "b.parquet") == data
    assert server.requests == [("/b.parquet", None), ("/b.parquet", "bytes=100000-")]


def test_retries_inside_one_call_resume_too(server, tmp_path):
    data = blob(200_000, 2)
    it = add(server, "c.jsonl", data)
    server.cut["/c.jsonl"] = 50_000
    assert fetch(it, str(tmp_path), retries=1, block=4096, wait=0) == "got"
    assert read(tmp_path / "sub" / "c.jsonl") == data
    assert server.requests[-1] == ("/c.jsonl", "bytes=50000-")


def test_server_errors_are_retried(server, tmp_path):
    data = blob(10_000, 3)
    it = add(server, "d.jsonl", data)
    server.fail.add("/d.jsonl")
    assert fetch(it, str(tmp_path), retries=1, wait=0) == "got"
    assert read(tmp_path / "sub" / "d.jsonl") == data


def test_a_shorter_file_on_the_server_fails_the_check(server, tmp_path):
    data = blob(30_000, 11)
    it = add(server, "s.jsonl", data)
    server.files["/s.jsonl"] = data[:10_000]
    server.ranges = True
    with pytest.raises(Exception):
        fetch(it, str(tmp_path), retries=0)
    assert os.path.getsize(tmp_path / "sub" / "s.jsonl.part") == 10_000      # kept: it looks like a cut connection
    with pytest.raises(CheckError):
        fetch(it, str(tmp_path), retries=0)                                # the next request runs past the end
    assert not os.path.exists(tmp_path / "sub" / "s.jsonl.part")


def test_missing_file_is_not_retried(server, tmp_path):
    it = Item("test", "sub/none.jsonl", f"http://127.0.0.1:{server.server_port}/none.jsonl", 10, "0" * 64)
    with pytest.raises(OSError):
        fetch(it, str(tmp_path), retries=3, wait=0)
    assert len(server.requests) == 1


def test_a_server_that_ignores_ranges_gets_a_fresh_start(server, tmp_path):
    data = blob(120_000, 4)
    it = add(server, "e.jsonl.gz", data)
    server.ranges = False
    os.makedirs(tmp_path / "sub")
    (tmp_path / "sub" / "e.jsonl.gz.part").write_bytes(data[:40_000])
    assert fetch(it, str(tmp_path), retries=0) == "got"
    assert read(tmp_path / "sub" / "e.jsonl.gz") == data       # not the first 40,000 bytes twice
    assert server.requests == [("/e.jsonl.gz", "bytes=40000-")]


def test_wrong_bytes_fail_the_check_and_leave_nothing_behind(server, tmp_path):
    good, bad = add(server, "f.jsonl", blob(5_000, 5)), add(server, "g.jsonl", blob(5_000, 6))
    bad = replace(bad, sha256="0" * 64)
    with pytest.raises(CheckError):
        fetch(bad, str(tmp_path), retries=0)
    assert not os.path.exists(tmp_path / "sub" / "g.jsonl") and not os.path.exists(tmp_path / "sub" / "g.jsonl.part")
    res = st.stage([good, bad], str(tmp_path), workers=2, log=lambda s: None)
    assert res["got"] == 1 and [p for p, _ in res["failed"]] == ["sub/g.jsonl"]
    lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["path"] for line in lines] == ["sub/f.jsonl"]


def test_a_stale_partial_file_is_caught_and_the_next_run_repairs_it(server, tmp_path):
    data = blob(50_000, 7)
    it = add(server, "h.jsonl", data)
    os.makedirs(tmp_path / "sub")
    (tmp_path / "sub" / "h.jsonl.part").write_bytes(b"x" * 20_000)       # left by another version of the file
    with pytest.raises(CheckError):
        fetch(it, str(tmp_path), retries=0)
    assert not os.path.exists(tmp_path / "sub" / "h.jsonl.part")
    assert fetch(it, str(tmp_path), retries=0) == "got"
    assert read(tmp_path / "sub" / "h.jsonl") == data


def test_rehash_fetches_a_damaged_file_again(server, tmp_path):
    data = blob(30_000, 8)
    it = add(server, "i.jsonl", data)
    os.makedirs(tmp_path / "sub")
    damaged = bytearray(data)
    damaged[100] ^= 1
    (tmp_path / "sub" / "i.jsonl").write_bytes(bytes(damaged))
    assert fetch(it, str(tmp_path)) == "have"                          # same size, so only --rehash can tell
    assert fetch(it, str(tmp_path), rehash=True) == "got"
    assert read(tmp_path / "sub" / "i.jsonl") == data


def test_many_files_at_once(server, tmp_path):
    items = [add(server, f"m{i}.jsonl", blob(20_000 + i, 10 + i)) for i in range(12)]
    res = st.stage(items, str(tmp_path), workers=4, log=lambda s: None)
    assert res == {"got": 12, "have": 0, "failed": []}
    for it in items:
        assert hashlib.sha256(read(tmp_path / it.path)).hexdigest() == it.sha256
    assert len((tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 12


def test_stage_refuses_when_the_disk_is_too_small(server, tmp_path, monkeypatch):
    it = add(server, "j.jsonl", blob(1_000, 9))
    monkeypatch.setattr(st.shutil, "disk_usage", lambda p: namedtuple("U", "total used free")(10, 10, 999))
    with pytest.raises(SystemExit, match="still to download"):
        st.stage([it], str(tmp_path), log=lambda s: None)
    assert server.requests == []


def test_same_path_twice_is_an_error(server, tmp_path):
    it = add(server, "k.jsonl", blob(10, 0))
    with pytest.raises(ValueError):
        st.stage([it, it], str(tmp_path))


def test_llmjp_plan_reads_sizes_and_sha256_in_batches():
    files = [f"ja/ja_wiki/{i:04d}.jsonl.gz" for i in range(7)] + ["ja/ja_wiki/README.md"]
    calls = []

    def post(url, body):
        paths = json.loads(body["query"].split("paths: ")[1].split(")")[0])
        calls.append(paths)
        nodes = [{"path": p, "rawSize": str(1000 + int(p[-13:-9])), "lfsOid": p[-13:-9] * 16} for p in paths]
        return {"data": {"project": {"repository": {"blobs": {"nodes": nodes}}}}}

    items = llmjp_items("ja/ja_wiki", max_files=5, list_files=lambda s: files, post=post, batch=2)
    assert [len(c) for c in calls] == [2, 2, 1]
    assert [it.size for it in items] == [1000, 1001, 1002, 1004, 1005]        # 5 of 7, spread through the list
    assert items[0] == Item("ja/ja_wiki", "llm-jp-v4/ja/ja_wiki/0000.jsonl.gz",
                            st.LLMJP_RAW + "ja/ja_wiki/0000.jsonl.gz", 1000, "0000" * 16)


def test_llmjp_plan_refuses_a_file_without_a_sha256():
    def post(url, body):
        return {"data": {"project": {"repository": {"blobs": {"nodes": [{"path": "ja/ja_wiki/0000.jsonl.gz",
                                                                          "rawSize": "134", "lfsOid": None}]}}}}}

    with pytest.raises(ValueError, match="no sha256"):
        llmjp_items("ja/ja_wiki", list_files=lambda s: ["ja/ja_wiki/0000.jsonl.gz"], post=post)


def test_hf_plan_keeps_lfs_data_files_pinned_to_one_commit():
    def f(path, size=None, sha=None):
        return SimpleNamespace(path=path, size=size, lfs=SimpleNamespace(size=size, sha256=sha) if sha else None)

    tree = [f("README.md", 3000), f("qa/b.parquet", 20, "b" * 64), f("qa/a.parquet", 10, "a" * 64),
            f("qa/notes.txt", 5, "c" * 64), SimpleNamespace(path="qa")]
    seen = {}

    class Api:
        def dataset_info(self, repo):
            return SimpleNamespace(sha="abc123")

        def list_repo_tree(self, repo, path_in_repo, repo_type, recursive, revision):
            seen.update(repo=repo, folder=path_in_repo, revision=revision)
            return tree

    items = hf_items("IFM/TxT360-v2/qa", api=Api())
    assert seen == {"repo": "IFM/TxT360-v2", "folder": "qa", "revision": "abc123"}
    assert items == [Item("IFM/TxT360-v2/qa", "hf/IFM/TxT360-v2/qa/a.parquet",
                          "https://huggingface.co/datasets/IFM/TxT360-v2/resolve/abc123/qa/a.parquet", 10, "a" * 64),
                     Item("IFM/TxT360-v2/qa", "hf/IFM/TxT360-v2/qa/b.parquet",
                          "https://huggingface.co/datasets/IFM/TxT360-v2/resolve/abc123/qa/b.parquet", 20, "b" * 64)]
    assert len(hf_items("IFM/TxT360-v2/qa", max_files=1, api=Api())) == 1


def test_plan_sends_each_source_to_its_host():
    got = plan(["ja/ja_wiki", "IFM/TxT360-v2"], max_files=3, llmjp=lambda s, n: [("llmjp", s, n)],
               hf=lambda s, n: [("hf", s, n)])
    assert got == [("llmjp", "ja/ja_wiki", 3), ("hf", "IFM/TxT360-v2", 3)]


def test_command_line_plan_only_prints_totals_and_downloads_nothing(monkeypatch, capsys):
    items = [Item("ja/ja_wiki", f"llm-jp-v4/ja/ja_wiki/{i}.jsonl.gz", "http://x", 2_000_000_000, "0" * 64)
             for i in range(3)]
    monkeypatch.setattr(st, "plan", lambda sources, *caps: items)
    monkeypatch.setattr(st, "stage", lambda *a, **k: pytest.fail("plan_only must not download"))
    st.main(["--source", "ja/ja_wiki", "--plan_only"])
    out = capsys.readouterr().out.splitlines()
    assert out[-1].split() == ["total", "3", "files", "6.00", "GB"]


def test_command_line_refuses_a_mistyped_llmjp_path(monkeypatch):
    monkeypatch.setattr(st, "plan", lambda sources, *caps: pytest.fail("no planning for a bad path"))
    with pytest.raises(SystemExit):
        st.main(["--source", "ja/ja_wikii", "--plan_only"])


def sized(sizes, source="ja/ja_wiki"):
    return [Item(source, f"p{i:02d}", "http://x", n, "0" * 64) for i, n in enumerate(sizes)]


def test_spread_keeps_files_from_the_whole_list():
    assert spread(list(range(10)), 3) == [0, 3, 6]
    assert spread(list(range(10)), 10) == list(range(10)) == spread(list(range(10)))
    assert spread(list(range(4)), 0) == []


def test_cap_by_bytes_stays_under_the_budget_and_keeps_one_file():
    items = sized([10] * 10)
    assert [it.path for it in cap(items, max_bytes=35)] == ["p00", "p02", "p05"]     # first, middle, a quarter
    uneven = sized([5, 50, 5, 50, 5, 50])
    assert sum(it.size for it in cap(uneven, max_bytes=60)) <= 60
    assert len(cap(items, max_bytes=1)) == 1                 # smaller than any file: still one file
    assert cap(items, max_files=2, max_bytes=1000) == [items[0], items[5]]
    assert cap(items, max_bytes=10_000) == items
    one_big = sized([1] * 9 + [91])                          # an average-size guess would keep 5 files here
    assert [it.size for it in cap(one_big, max_bytes=50)] == [1] * 9
    mixed = sized([1] * 30 + [8] * 9)                          # like en_dolma/pes2o: 30 small files, then 9 large
    assert sum(it.size for it in cap(mixed, max_bytes=8)) == 8


def test_coarse_to_fine_reaches_every_index_once():
    assert st._coarse_to_fine(10) == [0, 5, 2, 7, 1, 3, 6, 8, 4, 9]
    for n in (1, 2, 7, 39):
        assert sorted(st._coarse_to_fine(n)) == list(range(n))


def test_token_cap_uses_the_published_count(monkeypatch):
    calls = []

    def llmjp(s, n):
        calls.append(n)
        return sized([100] * 10, s)

    monkeypatch.setitem(st.LLMJP, "ja/ja_wiki", (1, "CC BY-SA 3.0"))        # 1B tokens in 1,000 bytes
    items = plan(["ja/ja_wiki"], max_tokens=0.3, llmjp=llmjp)
    assert calls == [None]                                    # sizes of every file, then the cap
    assert [it.path for it in items] == ["p00", "p02", "p05"]
    items = plan(["ja/ja_wiki"], max_tokens=0.3, max_gb=2e-7, llmjp=llmjp)
    assert len(items) == 2                                    # the smaller of the two caps wins
    notes = []
    items = plan(["ja/ja_wiki"], max_tokens=0.3, llmjp=lambda s, n: sized([3_000_000_000], s), log=notes.append)
    assert len(items) == 1 and notes == ["note: ja/ja_wiki keeps one file of 3.00 GB, over its cap of 0.90 GB"]


def test_token_cap_without_a_published_count_is_refused():
    with pytest.raises(SystemExit, match="max_gb"):
        plan(["IFM/TxT360-v2"], max_tokens=1, hf=lambda s, n: pytest.fail("refuse before listing"))
    with pytest.raises(SystemExit, match="max_gb"):
        plan(["ja/ja_cc/level0"], max_tokens=1, llmjp=lambda s, n: pytest.fail("refuse before listing"))


def write_parquet(path, columns):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.table(columns), path, row_group_size=2)


def test_to_jsonl_formats_staged_parquet_and_skips_the_rest(tmp_path):
    staged, out = tmp_path / "stage", tmp_path / "raw"
    k2 = "hf/IFM/TxT360-v2/txt360-qa/part-0.parquet"
    other = "hf/org/qa-set/data/train.parquet"
    write_parquet(str(staged / k2), {"text": ["first doc", None, "", "  ", "second doc\nline 2"],
                                     "meta": ["a", "b", "c", "d", "e"]})
    write_parquet(str(staged / other), {"question": ["What is 2+2?"], "answer": ["4"]})
    with open(staged / "manifest.jsonl", "w", encoding="utf-8") as f:
        for path in (k2, other, "llm-jp-v4/ja/ja_wiki/0000.jsonl.gz", "hf/IFM/TxT360-v2/README.md"):
            f.write(json.dumps({"source": "x", "path": path, "url": "u", "size": 1, "sha256": "0"}) + "\n")
    logs = []
    written = st.to_jsonl(str(staged), str(out), batch=2, log=logs.append)
    k2_out = out / "hf__IFM__TxT360-v2__txt360-qa__part-0.jsonl.gz"
    other_out = out / "hf__org__qa-set__data__train.jsonl.gz"
    assert written == {str(k2_out): 2, str(other_out): 1}
    with gzip.open(k2_out, "rt", encoding="utf-8") as f:
        assert [json.loads(line) for line in f] == [{"text": "first doc", "source": "IFM/TxT360-v2/txt360-qa"},
                                                    {"text": "second doc\nline 2", "source": "IFM/TxT360-v2/txt360-qa"}]
    with gzip.open(other_out, "rt", encoding="utf-8") as f:
        assert [json.loads(line) for line in f] == [{"text": "What is 2+2?\n\n4", "source": "org/qa-set/data"}]
    assert sorted(os.listdir(out)) == sorted([k2_out.name, other_out.name])
    assert st.to_jsonl(str(staged), str(out), log=logs.append) == {}          # a rerun converts nothing
    os.remove(k2_out)
    assert st.to_jsonl(str(staged), str(out), log=logs.append) == {str(k2_out): 2}  # a deleted output comes back


def test_to_jsonl_refuses_a_k2_folder_without_a_verified_formatter(tmp_path):
    path = "hf/IFM/TxT360-v2/new-subset/part-0.parquet"
    write_parquet(str(tmp_path / path), {"text": ["doc"]})
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"path": path}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No verified K2 formatter"):
        st.to_jsonl(str(tmp_path), str(tmp_path / "raw"), log=lambda s: None)


def test_busy_api_answers_are_retried_after_the_wait_they_ask_for(monkeypatch):
    import io
    import urllib.error

    from exealm.data import download
    answers, slept = [429, 503, None], []

    def urlopen(req, timeout):
        code = answers.pop(0)
        if code:
            raise urllib.error.HTTPError("u", code, "busy", {"Retry-After": "7"} if code == 429 else {}, None)
        return io.BytesIO(b'[{"type": "blob", "path": "ja/ja_wiki/0000.jsonl.gz"}]')

    monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(download.time, "sleep", slept.append)
    assert download.get_json("u", wait=1.0) == [{"type": "blob", "path": "ja/ja_wiki/0000.jsonl.gz"}]
    assert slept == [7.0, 2.0]                     # Retry-After when given, else wait * 2 ** try
    answers[:] = [404]
    with pytest.raises(urllib.error.HTTPError):
        download.get_json("u")


def test_a_new_version_of_a_file_is_fetched_even_at_the_same_size(server, tmp_path):
    old, new = blob(20_000, 20), blob(20_000, 21)
    it_old = add(server, "v.jsonl", old)
    st.stage([it_old], str(tmp_path), log=lambda s: None)
    it_new = add(server, "v.jsonl", new)                       # same path and size, new sha256 on the host
    assert st.stage([it_new], str(tmp_path), log=lambda s: None)["got"] == 1
    assert read(tmp_path / "sub" / "v.jsonl") == new
    lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["sha256"] for line in lines] == [it_old.sha256, it_new.sha256]


def test_a_file_the_manifest_does_not_list_is_checked_before_it_counts(server, tmp_path):
    data = blob(20_000, 30)
    it = add(server, "u.jsonl", data)
    good = add(server, "g.jsonl", blob(20_000, 31))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "u.jsonl").write_bytes(blob(20_000, 32))     # right size, wrong bytes, no manifest line
    (tmp_path / "sub" / "g.jsonl").write_bytes(blob(20_000, 31))     # right bytes, no manifest line
    res = st.stage([it, good], str(tmp_path), workers=1, log=lambda s: None)
    assert res == {"got": 1, "have": 1, "failed": []}
    assert read(tmp_path / "sub" / "u.jsonl") == data
    assert [p for p, _ in server.requests] == ["/u.jsonl"]
    lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(line)["path"] for line in lines) == ["sub/g.jsonl", "sub/u.jsonl"]


def test_a_manifest_line_cut_short_is_skipped_and_its_file_checked_again(server, tmp_path):
    a, b = add(server, "a.jsonl", blob(20_000, 40)), add(server, "b.jsonl", blob(20_000, 41))
    st.stage([a, b], str(tmp_path), workers=1, log=lambda s: None)
    man = tmp_path / "manifest.jsonl"
    text = man.read_text(encoding="utf-8")
    man.write_text(text[:-20], encoding="utf-8")                     # killed while writing b's line
    (tmp_path / "sub" / "b.jsonl").write_bytes(blob(20_000, 42))     # and b's bytes are wrong
    server.requests.clear()
    res = st.stage([a, b], str(tmp_path), workers=1, log=lambda s: None)
    assert res == {"got": 1, "have": 1, "failed": []}
    assert [p for p, _ in server.requests] == ["/b.jsonl"]
    assert read(tmp_path / "sub" / "b.jsonl") == blob(20_000, 41)
    assert [e["path"] for e in st._manifest(str(man))] == ["sub/a.jsonl", "sub/b.jsonl"]


def test_a_body_longer_than_published_stops_once_it_passes_the_size(tmp_path):
    reads = []

    class Body(io.BytesIO):
        status, headers = 200, {}

        def read1(self, n=-1):
            reads.append(n)
            return super().read1(n)

    it = Item("test", "sub/long.jsonl", "u", 1000, hashlib.sha256(b"x" * 1000).hexdigest())
    with pytest.raises(CheckError):
        fetch(it, str(tmp_path), opener=lambda url, start: Body(b"x" * 5_000_000), block=1 << 16)
    assert len(reads) <= 2                                       # not the whole 5 MB
    assert not os.path.exists(tmp_path / "sub" / "long.jsonl.part")


def test_an_empty_file_is_staged_without_a_request(tmp_path):
    it = Item("test", "sub/empty.jsonl", "u", 0, hashlib.sha256(b"").hexdigest())
    assert fetch(it, str(tmp_path), opener=lambda url, start: pytest.fail("no request needed")) == "got"
    assert read(tmp_path / "sub" / "empty.jsonl") == b""
    assert fetch(it, str(tmp_path), opener=lambda url, start: pytest.fail("no request needed")) == "have"


def test_ctrl_c_stops_the_running_files_and_drops_the_queued_ones(server, tmp_path):
    import _thread
    items = [add(server, f"slow{i}.jsonl", blob(400_000, 30 + i)) for i in range(4)]
    server.slow = 0.01                                         # about 0.4 MB/s, so a file takes about a second
    threading.Timer(0.4, _thread.interrupt_main).start()       # what Ctrl-C does
    t = time.time()
    with pytest.raises(KeyboardInterrupt):
        st.stage(items, str(tmp_path), workers=2, log=lambda s: None)
    assert time.time() - t < 3
    assert sorted(p for p, _ in server.requests) == ["/slow0.jsonl", "/slow1.jsonl"]     # the queued two never started
    parts = [tmp_path / "sub" / f"slow{i}.jsonl.part" for i in (0, 1)]
    assert all(0 < os.path.getsize(p) < 400_000 for p in parts)                        # kept for the next run
    server.slow = 0
    assert st.stage(items, str(tmp_path), workers=2, log=lambda s: None)["got"] == 4
    assert all(r.startswith("bytes=") for p, r in server.requests[2:] if p in ("/slow0.jsonl", "/slow1.jsonl"))


def test_a_new_version_that_fails_to_download_leaves_the_old_file_in_place(server, tmp_path):
    old, new = blob(20_000, 50), blob(20_000, 51)
    st.stage([add(server, "w.jsonl", old)], str(tmp_path), log=lambda s: None)
    it_new = add(server, "w.jsonl", new)
    server.cut["/w.jsonl"] = 5_000
    res = st.stage([it_new], str(tmp_path), retries=0, log=lambda s: None)
    assert [p for p, _ in res["failed"]] == ["sub/w.jsonl"]
    assert read(tmp_path / "sub" / "w.jsonl") == old              # still the version its manifest line names
    assert os.path.getsize(tmp_path / "sub" / "w.jsonl.part") == 5_000
    assert st.stage([it_new], str(tmp_path), retries=0, log=lambda s: None)["got"] == 1
    assert read(tmp_path / "sub" / "w.jsonl") == new
    assert server.requests[-1] == ("/w.jsonl", "bytes=5000-")


def test_a_damaged_file_is_deleted_even_when_its_new_download_fails(server, tmp_path):
    data = blob(20_000, 52)
    it = add(server, "x.jsonl", data)
    st.stage([it], str(tmp_path), log=lambda s: None)
    damaged = bytearray(data)
    damaged[7] ^= 1
    (tmp_path / "sub" / "x.jsonl").write_bytes(bytes(damaged))
    server.fail.add("/x.jsonl")
    res = st.stage([it], str(tmp_path), retries=0, rehash=True, log=lambda s: None)
    assert [p for p, _ in res["failed"]] == ["sub/x.jsonl"]
    assert not os.path.exists(tmp_path / "sub" / "x.jsonl")      # its manifest line would vouch for it next run


def test_disk_check_counts_a_new_version_while_the_old_one_is_kept(server, tmp_path, monkeypatch):
    st.stage([add(server, "y.jsonl", blob(1_000, 53))], str(tmp_path), log=lambda s: None)
    it_new = add(server, "y.jsonl", blob(1_000, 54))
    server.requests.clear()
    monkeypatch.setattr(st.shutil, "disk_usage", lambda p: namedtuple("U", "total used free")(10, 10, 999))
    with pytest.raises(SystemExit, match="still to download"):
        st.stage([it_new], str(tmp_path), log=lambda s: None)
    assert server.requests == []


def test_to_jsonl_skips_a_missing_file_and_redoes_an_output_made_from_an_older_version(tmp_path):
    staged, out = tmp_path / "stage", tmp_path / "raw"
    a, b = "hf/org/set/a.parquet", "hf/org/set/b.parquet"
    write_parquet(str(staged / a), {"text": ["old a"]})
    with open(staged / "manifest.jsonl", "w", encoding="utf-8") as f:
        for path in (a, b):                                   # b was deleted after a failed re-check
            f.write(json.dumps({"source": "x", "path": path, "url": "u", "size": 1, "sha256": "0"}) + "\n")
    logs = []
    a_out = out / "hf__org__set__a.jsonl.gz"
    assert st.to_jsonl(str(staged), str(out), log=logs.append) == {str(a_out): 1}
    assert any("b.parquet" in line and "missing" in line for line in logs)
    assert st.to_jsonl(str(staged), str(out), log=logs.append) == {}
    write_parquet(str(staged / a), {"text": ["new a", "new a 2"]})     # a new version of a was staged later
    with open(staged / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"source": "x", "path": a, "url": "u", "size": 1, "sha256": "1"}) + "\n")
    assert st.to_jsonl(str(staged), str(out), log=logs.append) == {str(a_out): 2}
    assert st.to_jsonl(str(staged), str(out), log=logs.append) == {}


def test_an_unlisted_file_that_fails_its_hash_is_deleted_even_when_its_download_fails(server, tmp_path):
    it = add(server, "z.jsonl", blob(20_000, 55))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "z.jsonl").write_bytes(blob(20_000, 56))     # right size, wrong bytes, no manifest line
    server.fail.add("/z.jsonl")
    res = st.stage([it], str(tmp_path), retries=0, log=lambda s: None)
    assert [p for p, _ in res["failed"]] == ["sub/z.jsonl"]
    assert not os.path.exists(tmp_path / "sub" / "z.jsonl")


def test_a_new_version_finished_from_an_old_partial_file_is_converted_again(server, tmp_path):
    staged, out = tmp_path / "stage", tmp_path / "raw"
    write_parquet(str(tmp_path / "v1.parquet"), {"text": ["old"]})
    write_parquet(str(tmp_path / "v2.parquet"), {"text": ["new"]})
    v1 = replace(add(server, "a.parquet", read(tmp_path / "v1.parquet")), path="hf/org/set/a.parquet")
    st.stage([v1], str(staged), log=lambda s: None)
    v2 = replace(add(server, "a.parquet", read(tmp_path / "v2.parquet")), path="hf/org/set/a.parquet")
    part = staged / "hf" / "org" / "set" / "a.parquet.part"
    part.write_bytes(read(tmp_path / "v2.parquet"))          # v2 was fully written, then the run stopped
    t = os.path.getmtime(part)
    os.utime(part, (t - 100, t - 100))
    st.to_jsonl(str(staged), str(out), log=lambda s: None)    # converts v1, after v2's last write
    server.requests.clear()
    assert st.stage([v2], str(staged), log=lambda s: None)["got"] == 1
    assert server.requests == []                              # v2 only needed its rename
    written = st.to_jsonl(str(staged), str(out), log=lambda s: None)
    with gzip.open(out / "hf__org__set__a.jsonl.gz", "rt", encoding="utf-8") as f:
        assert [json.loads(line)["text"] for line in f] == ["new"]
    assert list(written.values()) == [1]


def test_a_new_version_with_the_same_timestamp_as_the_old_output_is_converted_again(server, tmp_path):
    staged, out = tmp_path / "stage", tmp_path / "raw"
    write_parquet(str(tmp_path / "v1.parquet"), {"text": ["old"]})
    write_parquet(str(tmp_path / "v2.parquet"), {"text": ["new"]})
    v1 = replace(add(server, "a.parquet", read(tmp_path / "v1.parquet")), path="hf/org/set/a.parquet")
    st.stage([v1], str(staged), log=lambda s: None)
    st.to_jsonl(str(staged), str(out), log=lambda s: None)
    v2 = replace(add(server, "a.parquet", read(tmp_path / "v2.parquet")), path="hf/org/set/a.parquet")
    assert st.stage([v2], str(staged), log=lambda s: None)["got"] == 1
    a, a_out = staged / "hf" / "org" / "set" / "a.parquet", out / "hf__org__set__a.jsonl.gz"
    t = os.stat(a).st_mtime_ns
    os.utime(a_out, ns=(t, t))                     # one clock tick covered both writes (Linux, FAT drives)
    assert st.to_jsonl(str(staged), str(out), log=lambda s: None) == {str(a_out): 1}
    with gzip.open(a_out, "rt", encoding="utf-8") as f:
        assert [json.loads(line)["text"] for line in f] == ["new"]
    assert st.to_jsonl(str(staged), str(out), log=lambda s: None) == {}
