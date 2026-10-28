"""Download the public WM-811K dataset (``LSWMD.pkl``, ~2.1 GB) and verify its SHA-256.

Source: an ungated Hugging Face mirror of the file originally released by the MIR Lab
(National Taiwan University) and redistributed on Kaggle. See DATA_LICENSE.md - the dataset is
not part of this repository and is never committed.

Usage::

    python scripts/fetch_dataset.py [--dest data/raw]

Only the standard library is used so this works before any dependency is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/lslattery/wafer-defect-detection/resolve/main/LSWMD.pkl"
SHA256 = "1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65"
SIZE = 2_095_505_977


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def download(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "LSWMD.pkl"
    if target.exists() and target.stat().st_size == SIZE and sha256_file(target) == SHA256:
        print(f"{target} already present and verified")
        return target
    part = target.with_suffix(".pkl.part")
    offset = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(req, timeout=60) as resp, part.open("ab" if offset else "wb") as out:
        if offset and resp.status != 206:  # server ignored the range request: start over
            out.seek(0)
            out.truncate()
            offset = 0
        done = offset
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            print(f"\r{done / 1e9:5.2f} / {SIZE / 1e9:.2f} GB", end="", flush=True)
    print()
    digest = sha256_file(part)
    if digest != SHA256:
        part.unlink()
        sys.exit(f"SHA-256 mismatch ({digest}); download removed, please retry")
    part.replace(target)
    print(f"verified {target}")
    return target


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    download(ap.parse_args().dest)
