#!/usr/bin/env python3
"""Build the character metadata and embedding index."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge.embed import build_index  # noqa: E402
from forge.paths import CEDICT, DICTIONARY, HANZIDB, ensure_dirs  # noqa: E402
from forge.textdata import load_keyword_hints, load_records  # noqa: E402


def main() -> int:
    ensure_dirs()
    started = time.time()

    print("Loading Make Me a Hanzi and CC-CEDICT...")
    records = load_records(DICTIONARY, CEDICT, HANZIDB)
    print(f"Loaded {len(records):,} characters with English glosses.")
    print("Building English keyword -> character hints from CC-CEDICT...")
    hints = load_keyword_hints(CEDICT)

    build_index(records, keyword_hints=hints)
    print(f"Finished in {time.time() - started:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
