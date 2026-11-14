"""Shared helpers for writing results to disk."""

from __future__ import annotations

import re
from pathlib import Path

from .engine import ForgeResult


def slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug or "character")[:limit].strip("-")


def save_images(result: ForgeResult, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = slugify(result.phrase)
    paths: list[Path] = []
    for index, candidate in enumerate(result.candidates, start=1):
        if candidate.image is None:
            continue
        path = outdir / f"{stem}-{result.mode}-{index}.png"
        candidate.image.save(path)
        paths.append(path)
    return paths
