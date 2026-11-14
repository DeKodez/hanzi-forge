#!/usr/bin/env python3
"""Layout and contextual-variant regression checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge.compose import (  # noqa: E402
    LEFT_WIDTH_RATIOS,
    TOP_HEIGHT_CAPS,
    TOP_WIDTH_RATIOS,
    WIDE_TOP,
    _ink_shape,
    compose_mask,
)
from forge.radicals import BASE_FORM, LEFT_VARIANTS, TOP_VARIANTS, shape_for  # noqa: E402

SIZE = 1024
AREA = int(SIZE * 0.74)
OFFSET = (SIZE - AREA) // 2

FAILS: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILS.append(message)


def ink_bbox(mask):
    array = np.asarray(mask) > 32
    ys, xs = np.nonzero(array)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def main() -> int:
    # 1. Positional variants: base forms and left/top/bottom forms.
    variants = [
        ("水", "⿰", 0, "氵"),
        ("心", "⿰", 0, "忄"),
        ("人", "⿰", 0, "亻"),
        ("手", "⿰", 0, "扌"),
        ("言", "⿰", 0, "讠"),
        ("糸", "⿰", 0, "纟"),
        ("金", "⿰", 0, "钅"),
        ("食", "⿰", 0, "饣"),
        ("火", "⿱", 1, "灬"),
        ("艸", "⿱", 0, "艹"),
        ("爪", "⿱", 0, "爫"),
        ("网", "⿱", 0, "罒"),
        ("灬", "⿰", 0, "火"),
        ("忄", "⿰", 1, "心"),
        ("氵", "⿱", 0, "水"),
        ("艹", "⿰", 1, "艸"),
    ]
    for base, structure, position, expected in variants:
        actual = shape_for(base, structure, position)
        check(actual == expected, f"shape_for({base}, {structure}, {position}) = {actual}, expected {expected}")

    # 2. Every left-width radical renders in a left-right composition.
    for radical, ratio in LEFT_WIDTH_RATIOS.items():
        mask = compose_mask("⿰", (radical, "相"), SIZE, "pen")
        bbox = ink_bbox(mask)
        check(bbox is not None, f"left radical {radical} rendered blank")
        check(ratio <= 0.40, f"left radical {radical} width ratio {ratio} is too wide")

    # 3. Every top-cap radical respects its cap and renders.
    for radical, cap in TOP_HEIGHT_CAPS.items():
        if not _ink_shape(radical, "pen").size or _ink_shape(radical, "pen").size == (1, 1):
            continue
        mask = compose_mask("⿱", (radical, "相"), SIZE, "pen")
        array = np.asarray(mask) > 32
        rows = array.sum(axis=1)
        # first interior blank run below the top part
        gap_start = None
        start = None
        for index, count in enumerate(rows):
            if count == 0 and start is None:
                start = index
            if count > 0 and start is not None:
                if start > OFFSET + 10:
                    gap_start = start
                    break
                start = None
        if gap_start is None:
            gap_start = OFFSET + AREA
        top_height = gap_start - (OFFSET + int(AREA * 0.01))
        check(
            top_height <= int(AREA * cap) + 8,
            f"top radical {radical} height {top_height} exceeds cap {int(AREA * cap)}",
        )

    # 4. Enclosures: inner component must stay inside the glyph area.
    for outer, structure in [("门", "⿵"), ("囗", "⿴"), ("广", "⿸"), ("疒", "⿸"), ("辶", "⿺")]:
        mask = compose_mask(structure, (outer, "相"), SIZE, "pen")
        bbox = ink_bbox(mask)
        check(bbox is not None, f"enclosure {outer} rendered blank")
        if bbox:
            check(bbox[0] >= OFFSET - 2 and bbox[2] <= OFFSET + AREA + 2, f"enclosure {outer} exceeds horizontal area")
            check(bbox[1] >= OFFSET - 2 and bbox[3] <= OFFSET + AREA + 2, f"enclosure {outer} exceeds vertical area")

    if FAILS:
        print(f"{len(FAILS)} layout check(s) failed:")
        for failure in FAILS:
            print(" -", failure)
        return 1
    print(
        f"All layout checks passed: {len(variants)} variants, "
        f"{len(LEFT_WIDTH_RATIOS)} left radicals, {len(TOP_HEIGHT_CAPS)} top caps, 5 enclosures."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
