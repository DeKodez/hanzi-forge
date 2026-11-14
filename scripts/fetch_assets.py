#!/usr/bin/env python3
"""Download the open data and fonts used by the character forge."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FONTS = ROOT / "assets" / "fonts"

ASSETS = [
    (
        "https://raw.githubusercontent.com/skishore/makemeahanzi/master/dictionary.txt",
        DATA / "makemeahanzi_dictionary.txt",
    ),
    (
        "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz",
        DATA / "cedict.txt.gz",
    ),
    (
        "https://raw.githubusercontent.com/ruddfawcett/hanziDB.csv/master/data/hanziDB.csv",
        DATA / "hanzidb.csv",
    ),
    (
        "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt",
        DATA / "ids.txt",
    ),
    (
        "https://github.com/lxgw/LxgwWenKai/releases/download/v1.530/LXGWWenKai-Regular.ttf",
        FONTS / "LXGWWenKai-Regular.ttf",
    ),
    (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/mashanzheng/MaShanZheng-Regular.ttf",
        FONTS / "MaShanZheng-Regular.ttf",
    ),
]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists  {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"  fetch   {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "hanzi-forge/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        with open(tmp, "wb") as fh:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                received += len(chunk)
                if total:
                    pct = 100 * received / total
                    print(f"\r          {pct:5.1f}%  {received / 1e6:6.1f}/{total / 1e6:.1f} MB", end="")
        print()
    if total and tmp.stat().st_size != total:
        raise RuntimeError(f"incomplete download: {dest} ({tmp.stat().st_size} != {total})")
    tmp.replace(dest)
    print(f"  saved   {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    print("Fetching assets into data/ and assets/fonts/\n")
    DATA.mkdir(parents=True, exist_ok=True)
    FONTS.mkdir(parents=True, exist_ok=True)
    for url, dest in ASSETS:
        download(url, dest)
    print("\nAll assets ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
