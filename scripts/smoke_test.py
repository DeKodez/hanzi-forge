#!/usr/bin/env python3
"""End-to-end sanity checks for the forge pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge.engine import ForgeEngine, sanitize_layout_text  # noqa: E402

PHRASES = [
    "the ache of missing someone on a rainy night",
    "a fire that never goes out",
    "the feeling of coming home",
    "digital loneliness",
    "quiet strength",
]


def main() -> int:
    engine = ForgeEngine()
    failures = 0

    checks = [
        (sanitize_layout_text("The king sits above the wind. A ruler commands.") == "A ruler commands.",
         "sanitizer removes placement sentences"),
        (engine._is_atomic("龙"), "龙 is atomic"),
        (not engine._is_atomic("铃"), "铃 is a compound"),
        (engine._is_valid_semantic("鹿"), "鹿 allowed as complex semantic"),
        (bool(engine._plan_two_part("龙", "云")), "龙+云 has a conventional plan"),
        (engine._is_real_signature("⿰讠令"), "讠+令 matches the existing 詅 structure"),
        (not engine._is_real_signature("⿱王⿰王王"), "王王王 is a new composition"),
    ]
    for ok, label in checks:
        if not ok:
            print(f"FAIL  {label}")
            failures += 1

    ideographic = engine._ideographic_candidate(
        {
            "type": "ideographic",
            "components": ["王", "王", "王"],
            "structure": "⿱王⿰王王",
            "semantic": ["王", "王", "王"],
            "phonetic": "",
            "reading": "wáng",
            "gloss": "king of kings; sovereignty multiplied",
            "rationale": "Three kings: rule stacked upon rule.",
        },
        "king of kings",
    )
    if ideographic is None or not ideographic.reading:
        print("FAIL  ideographic three-king candidate")
        failures += 1

    leviathan = engine._ideographic_candidate(
        {
            "type": "ideographic",
            "components": ["魚", "龍", "水"],
            "structure": "⿲魚龍水",
            "semantic": ["魚", "龍", "水"],
            "phonetic": "",
            "reading": "lì",
            "gloss": "leviathan; the dragon-fish of the primordial sea.",
            "rationale": "Fish, dragon, and water combine.",
        },
        "leviathan",
    )
    if leviathan is None or leviathan.structure.startswith("⿲"):
        print("FAIL  one-row ideographic layout was not compacted")
        failures += 1

    if not engine._redundant_radicals(("心", "念", "忆")):
        print("FAIL  duplicate heart radicals were not rejected")
        failures += 1
    if not engine._redundant_radicals(("心", "心", "亡")):
        print("FAIL  repeated heart component was not rejected")
        failures += 1
    if engine._redundant_radicals(("王", "王", "王")):
        print("FAIL  deliberate triple repetition was rejected")
        failures += 1

    repeated = engine._repetition_component({
        "literal_meaning": "king of kings",
        "chinese_terms": ["万王之王"],
        "component_candidates": ["王", "口"],
    })
    if repeated != "王":
        print("FAIL  repeated 'king of kings' concept did not select 王")
        failures += 1

    four = engine._ideographic_candidate(
        {
            "type": "ideographic",
            "components": ["水", "火", "土", "木"],
            "structure": "⿰⿱水火⿱土木",
            "semantic": ["水", "火", "土", "木"],
            "phonetic": "",
            "reading": "sì",
            "gloss": "four elements",
            "rationale": "A 2x2 test.",
        },
        "the four elements",
    )
    if four is None or len(four.parts) != 4:
        print("FAIL  four-part ideographic design was rejected")
        failures += 1

    lion_city = engine._ideographic_candidate(
        {
            "type": "ideographic",
            "components": ["狮", "城"],
            "structure": "⿰狮城",
            "semantic": ["狮", "城"],
            "phonetic": "",
            "reading": "shī",
            "gloss": "lion city; Singapore",
            "rationale": "The lion and the city name the Lion City.",
        },
        "singapore - lion city",
    )
    if lion_city is None or lion_city.parts != ("犭", "城"):
        print("FAIL  lion+city was not reduced to radical+component")
        failures += 1

    for phrase in PHRASES:
        result = engine.forge(phrase, count=4, use_llm=False)
        if not result.candidates:
            print(f"FAIL  no candidates for {phrase!r}")
            failures += 1
            continue

        summary = []
        for candidate in result.candidates:
            image = candidate.image
            assert image is not None
            board = np.asarray(image)
            if board.shape != (1024, 1024, 3):
                print(f"FAIL  bad image shape {board.shape}")
                failures += 1
            if board.std() < 8:
                print("FAIL  suspiciously flat image")
                failures += 1
            if (
                candidate.origin == "deterministic"
                and len(candidate.parts) > 1
                and not candidate.reading
            ):
                print("FAIL  deterministic candidate has no reading")
                failures += 1
            if candidate.structure and candidate.structure + "".join(candidate.shapes) in engine.real:
                print(f"FAIL  candidate already exists: {candidate.structure}{''.join(candidate.shapes)}")
                failures += 1
            parts = " + ".join(candidate.shapes)
            summary.append(f"{candidate.structure or 'real'} {parts}")

        top = result.matches[0]
        print(
            f"OK  {phrase!r}\n"
            f"    glyphs: {'; '.join(summary)}\n"
            f"    closest: {top.char} ({top.record.get('pinyin', '')}) "
            f"{top.record.get('definition', '').split(';')[0]} [{top.score:.2f}]"
        )

    real = engine.forge("the feeling of coming home", count=3, mode="real", use_llm=False)
    if len(real.candidates) != 3:
        print("FAIL  real-character mode did not return three candidates")
        failures += 1
    else:
        print("OK  real-character mode")

    if failures:
        print(f"\n{failures} check(s) failed.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
