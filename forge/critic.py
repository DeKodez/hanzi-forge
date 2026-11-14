"""Layout diagnostics for a rendered glyph candidate.

The critic is deliberately model-independent.  It inspects the same mask the
renderer produces and reports component-scale, stretch, coverage, and overlap
problems that can be fed back to a language model for revision.
"""

from __future__ import annotations

from PIL import Image

from .compose import layout_trace


def _ascii_art(mask: Image.Image, width: int = 44, height: int = 44) -> str:
    bbox = mask.getbbox()
    if bbox is None:
        return ""
    crop = mask.crop(bbox)
    scale = min(width / max(1, crop.width), height / max(1, crop.height))
    tw = max(1, int(crop.width * scale))
    th = max(1, int(crop.height * scale))
    crop = crop.resize((tw, th), Image.Resampling.LANCZOS)
    chars = "@%#*+=-:. "
    rows: list[str] = []
    for y in range(height):
        row: list[str] = []
        for x in range(width):
            if x < (width - tw) // 2 or x >= (width + tw) // 2 or y < (height - th) // 2 or y >= (height + th) // 2:
                row.append(" ")
                continue
            value = crop.getpixel((x - (width - tw) // 2, y - (height - th) // 2))
            row.append(chars[min(len(chars) - 1, value * len(chars) // 256)])
        rows.append("".join(row))
    return "\n".join(rows)


def diagnose(
    structure: str,
    shapes: tuple[str, ...],
    style: str = "brush",
    size: int = 384,
    stroke_counts: dict[str, int] | None = None,
) -> dict:
    """Return a compact, machine-readable critique of a rendered candidate."""
    mask, trace = layout_trace(structure, shapes, size, style)
    if not trace:
        return {
            "structure": structure,
            "coverage": 0.0,
            "score": 99.0,
            "components": [],
            "issues": ["no rendered components"],
        }

    stroke_values = [
        max(1, int((stroke_counts or {}).get(shape, 1)))
        for shape, _, _, _, _ in trace
    ]
    total_strokes = sum(stroke_values) or 1
    total_area = sum(max(1, width * height) for _, width, height, _, _ in trace) or 1
    coverage = sum(mask.histogram()[1:]) / max(1, mask.width * mask.height)

    components: list[dict] = []
    issues: list[str] = []
    stretch_penalty = 0.0
    balance_penalty = 0.0
    for index, (shape, width, height, natural_w, natural_h) in enumerate(trace):
        natural_aspect = natural_w / max(1, natural_h)
        slot_aspect = width / max(1, height)
        stretch = max(slot_aspect / natural_aspect, natural_aspect / slot_aspect)
        area_share = max(1, width * height) / total_area
        stroke_share = stroke_values[index] / total_strokes
        flags: list[str] = []
        if stretch > 2.2:
            flags.append(f"stretched {stretch:.1f}x")
            stretch_penalty += stretch - 2.0
        if area_share < stroke_share * 0.55:
            flags.append(
                f"small: area {area_share:.2f} vs strokes {stroke_share:.2f}"
            )
            balance_penalty += (stroke_share * 0.55 - area_share) * 2.0
        elif area_share > stroke_share * 2.0 and len(trace) > 1:
            flags.append(
                f"large: area {area_share:.2f} vs strokes {stroke_share:.2f}"
            )
            balance_penalty += (area_share - stroke_share * 2.0) * 0.5
        components.append({
            "shape": shape,
            "slot": [width, height],
            "natural": [natural_w, natural_h],
            "area_share": round(area_share, 3),
            "stroke_share": round(stroke_share, 3),
            "stretch": round(stretch, 2),
            "flags": flags,
        })
        issues.extend(f"{shape}: {flag}" for flag in flags)

    if coverage < 0.10:
        issues.append(f"sparse rendering; ink coverage {coverage:.2f}")
    elif coverage > 0.35:
        issues.append(f"dense rendering; ink coverage {coverage:.2f}")
    if structure and "⿻" in structure and len(shapes) > 2:
        issues.append("overlay is nested inside a multi-part composition")

    score = (
        1.8 * balance_penalty
        + 0.8 * stretch_penalty
        + 4.0 * abs(coverage - 0.18)
        + (1.5 if structure and "⿻" in structure and len(shapes) > 2 else 0.0)
    )
    return {
        "structure": structure,
        "coverage": round(coverage, 3),
        "score": round(score, 3),
        "components": components,
        "issues": issues[:12],
        "ascii": _ascii_art(mask),
        "ascii_note": "ASCII is approximate; component metrics are authoritative.",
    }
