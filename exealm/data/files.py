"""Expand file patterns the way a Unix shell would. PowerShell and cmd pass `data/raw/*.jsonl.gz` through untouched."""
import glob
import os


def expand(patterns):
    """Each pattern becomes its sorted matches. A path that exists is kept as it is; a pattern with no match is an error."""
    files = []
    for p in patterns:
        if os.path.exists(p) or not glob.has_magic(p):
            files.append(p)
            continue
        found = sorted(glob.glob(p))
        if not found:
            raise SystemExit(f"no files match {p}")
        files += found
    return files
