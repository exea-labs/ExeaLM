"""The public part of the library's human tier: every passage whose license allows redistribution with attribution, as
text, with its source, license and URL. The rest stays in the private library that training and the hosted model use.

Left out (docs/library.md, licenses):
- arXiv papers under arXiv's default licence or with none recorded: the licence grants arXiv distribution and no
  reuse to others (info.arxiv.org/help/license/reuse.html). CC-licensed and public-domain papers go in.
- Stack Exchange: its CC BY-SA attribution asks for each author's name linked to their profile, and the data holds
  only user ids. It goes in once names are added.
Everything else goes in under its recorded license: Wikipedia (CC BY-SA, article URL), KAKEN (CC BY 4.0, via LLM-jp's
corpus), peS2o (ODC-BY, as AI2 publishes it), code (the file's license where recorded, else The Stack's permissive set
under the collection's ODC-BY) and math web pages (ODC-BY).

    python experiments/library/public_subset.py /path/human_pool/part_* --out /path/public_human
"""
import argparse
import gzip
import json
import os
import re
from collections import Counter

ARXIV_OPEN = re.compile(r"^(CC-BY|CC0|public domain)", re.I)


def public(r):
    """(True, "") if the passage may be published, else (False, why)."""
    s, lic = r.get("source"), str(r.get("license") or "")
    if s == "stackexchange":
        return False, "stackexchange: author names needed for attribution"
    if s == "arxiv" and not ARXIV_OPEN.match(lic):
        return False, "arxiv: default licence or none recorded"
    if re.search(r"\bNC\b|NonCommercial|\bND\b|NoDerivs", lic):
        return False, "non-commercial or no-derivatives"
    return True, ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    kept, dropped, by_license = Counter(), Counter(), Counter()
    for part in a.parts:
        name = os.path.basename(part.rstrip("/"))
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
        n_in = n_out = 0
        with open(os.path.join(part, "index.jsonl"), encoding="utf-8") as fi, \
                gzip.open(os.path.join(part, "passages.jsonl.gz"), "rt", encoding="utf-8") as fp, \
                gzip.open(os.path.join(a.out, name, "passages.jsonl.gz"), "wt", encoding="utf-8") as fo:
            for line_i, line_p in zip(fi, fp):
                r, p = json.loads(line_i), json.loads(line_p)
                if r.get("id") != p.get("id"):
                    raise SystemExit(f"{part}: index and passages out of step at {r.get('id')} / {p.get('id')}")
                n_in += 1
                ok, why = public(r)
                if not ok:
                    dropped[why] += r.get("n_tokens", 0)
                    continue
                n_out += 1
                kept[r["source"]] += r.get("n_tokens", 0)
                by_license[str(r.get("license"))[:40]] += r.get("n_tokens", 0)
                fo.write(json.dumps(dict(id=r["id"], source=r["source"], title=r.get("title"), text=p["text"],
                                         license=r.get("license"), url=r.get("url"), lang=r.get("lang"),
                                         n_tokens=r.get("n_tokens")), ensure_ascii=False) + "\n")
        print(f"{name}: {n_out:,} of {n_in:,} passages", flush=True)
    stats = dict(kept_tokens=dict(kept), dropped_tokens=dict(dropped),
                 top_licenses=dict(by_license.most_common(30)))
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(f"public: {sum(kept.values()) / 1e9:.2f}B tokens; left out: "
          + ", ".join(f"{k} {v / 1e9:.2f}B" for k, v in dropped.items()), flush=True)


if __name__ == "__main__":
    main()
