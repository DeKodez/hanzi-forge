#!/usr/bin/env python3
"""Run the canonical phrases through the full forge and critique pipeline."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge import llm
from forge.compose import compose_ids
from forge.critic import diagnose
from forge.engine import ForgeEngine
from forge.paths import OUTPUTS
from forge.render import finish

PHRASES = ROOT / "scripts" / "canonical_phrases.txt"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def main() -> int:
    engine = ForgeEngine()
    phrases = [
        line.strip()
        for line in PHRASES.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    records = []
    thumbs = []
    for phrase in phrases:
        result = engine.forge(phrase, 3, use_llm=True)
        if not result.candidates:
            print(f"EMPTY {phrase}")
            continue
        candidate = result.candidates[0]
        geometry = engine._candidate_critique(candidate)
        semantic = engine._candidate_semantic_score(candidate, result.analysis)
        mask = compose_ids(
            candidate.structure or "",
            candidate.shapes,
            768,
            engine.style,
            layout=candidate.layout_params,
        )
        image = finish(
            mask,
            phrase,
            reading=candidate.reading,
            gloss=candidate.gloss,
            show_seal=False,
        )
        image_path = OUTPUTS / "canonical" / f"{slug(phrase)}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        vision = llm.vision_critique(
            image,
            phrase,
            candidate.structure or "",
            candidate.shapes,
            geometry,
        )
        record = {
            "phrase": phrase,
            "structure": candidate.structure,
            "shapes": candidate.shapes,
            "reading": candidate.reading,
            "gloss": candidate.gloss,
            "semantic": round(semantic, 3),
            "geometry": geometry,
            "vision": vision,
            "image": str(image_path),
        }
        records.append(record)
        thumbs.append((phrase, image))
        print(
            f"{phrase[:34]:36s} {candidate.structure} "
            f"sem={semantic:.2f} geom={geometry.get('score')} "
            f"vision={vision.get('score') if vision else 'n/a'}"
        )

    OUTPUTS.mkdir(exist_ok=True)
    audit_path = OUTPUTS / "canonical-audit.json"
    audit_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from PIL import Image

    cell = 320
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (241, 235, 223))
    for index, (_, image) in enumerate(thumbs):
        sheet.paste(
            image.convert("RGB").resize((cell - 12, cell - 12)),
            ((index % cols) * cell + 6, (index // cols) * cell + 6),
        )
    sheet_path = OUTPUTS / "canonical-contact-sheet.png"
    sheet.save(sheet_path)
    print(f"saved {audit_path}")
    print(f"saved {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
