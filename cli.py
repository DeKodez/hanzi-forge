#!/usr/bin/env python3
"""Command line interface for Character Forge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.engine import ForgeEngine
from forge.output import save_images
from forge.paths import OUTPUTS
from forge.radicals import describe_structure


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Turn an English phrase into a newly composed Chinese-character image."
    )
    parser.add_argument("phrase", help="English phrase to interpret")
    parser.add_argument("-n", "--count", type=int, default=4, help="number of candidates (1-8)")
    parser.add_argument(
        "--mode",
        choices=("new", "real"),
        default="new",
        help="forge a new glyph or render the closest real characters",
    )
    parser.add_argument(
        "--style",
        choices=("brush", "pen"),
        default="brush",
        help="brush calligraphy or pen writing",
    )
    parser.add_argument("-o", "--out", type=Path, default=OUTPUTS, help="output directory")
    parser.add_argument(
        "--seal",
        action="store_true",
        help="stamp the closest real character in a red seal (decorative)",
    )
    parser.add_argument(
        "--llm",
        dest="use_llm",
        action="store_true",
        default=None,
        help="use the LLM creative director (reads .env or DEEPSEEK_API_KEY)",
    )
    parser.add_argument(
        "--no-llm",
        dest="use_llm",
        action="store_false",
        help="disable the LLM creative director",
    )
    parser.add_argument("--json", action="store_true", help="print a machine-readable result")
    args = parser.parse_args()

    count = max(1, min(8, args.count))
    engine = ForgeEngine(style=args.style)
    result = engine.forge(
        args.phrase,
        count=count,
        mode=args.mode,
        use_llm=args.use_llm,
        seal=args.seal,
    )
    paths = save_images(result, args.out)

    if args.json:
        payload = {
            "phrase": result.phrase,
            "mode": result.mode,
            "warning": result.warning,
            "candidates": [
                {
                    "image": str(path),
                    "structure": candidate.structure,
                    "parts": list(candidate.parts),
                    "shapes": list(candidate.shapes),
                    "explanation": candidate.explanation,
                    "origin": candidate.origin,
                    "reading": candidate.reading,
                    "gloss": candidate.gloss,
                    "rationale": candidate.rationale,
                    "semantic": candidate.semantic,
                    "phonetic": candidate.phonetic,
                }
                for path, candidate in zip(paths, result.candidates)
            ],
            "matches": [
                {
                    "character": match.char,
                    "pinyin": match.record.get("pinyin", ""),
                    "definition": match.record.get("definition", ""),
                    "score": round(match.score, 4),
                }
                for match in result.matches
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"\nPhrase: {result.phrase}")
    print(f"Mode:   {result.mode}")
    if result.warning:
        print(f"Note:   {result.warning}")
    print()

    for index, (path, candidate) in enumerate(zip(paths, result.candidates), start=1):
        if candidate.structure:
            op = (
                candidate.structure[0]
                if candidate.structure[0] in {"⿰", "⿱", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺"}
                else None
            )
            if op and len(candidate.parts) == 2:
                label = f"{op} {describe_structure(op)}"
            elif op:
                label = f"{candidate.structure} (nested)"
            else:
                label = candidate.structure
            label += " · " + " + ".join(candidate.shapes)
        else:
            label = "closest real character"
        badge = " [LLM]" if candidate.origin == "llm" else ""
        print(f"{index}. {path}  [{label}]{badge}")
        print(f"   {candidate.explanation}")
        if candidate.rationale:
            print(f"   Rationale: {candidate.rationale}")

    print("\nClosest real characters:")
    for match in result.matches:
        print(
            f"  {match.char}  {match.record.get('pinyin', '') or '—':<12} "
            f"{match.record.get('definition', '').split(';')[0]:<40} {match.score:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
