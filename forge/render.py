"""Turn a glyph mask into an ink-on-rice-paper image."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import textwrap
from pathlib import Path as FilePath

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .paths import PEN_FONT

PAPER = (243, 235, 217)
INK = (24, 21, 18)
SEAL_RED = (172, 48, 40)


def seed_for(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _paper(size: int, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    gradient = 1.0 + 0.035 * (y / size) - 0.02 * (x / size)

    paper = np.empty((size, size, 3), dtype=np.float32)
    for channel, value in enumerate(PAPER):
        paper[..., channel] = value * gradient

    grain = rng.normal(0.0, 3.4, (size, size, 1)).astype(np.float32)
    paper += grain

    # A few broad, soft stains give the sheet some life.
    for _ in range(5):
        cx, cy = rng.uniform(0, size, 2)
        radius = rng.uniform(size * 0.08, size * 0.28)
        dist = ((x - cx) ** 2 + (y - cy) ** 2) / (radius**2)
        stain = np.exp(-dist)[..., None] * rng.uniform(1.5, 4.0)
        paper -= stain

    return np.clip(paper, 0, 255).astype(np.uint8)


def _vignette(image: Image.Image) -> Image.Image:
    size = image.size[0]
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = (size - 1) / 2
    radius = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
    falloff = np.clip((radius - 0.55) / 0.65, 0, 1) ** 2
    factor = (1.0 - 0.12 * falloff)[..., None]
    array = np.asarray(image).astype(np.float32) * factor
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def _text_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        [
            FilePath("/System/Library/Fonts/Supplemental/Georgia Bold.ttf"),
            FilePath("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"),
            FilePath("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
        ]
        if bold
        else [
            FilePath("/System/Library/Fonts/Supplemental/Georgia.ttf"),
            FilePath("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
            FilePath("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
        ]
    )
    candidates.append(PEN_FONT)
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _fit_line(draw: ImageDraw.ImageDraw, text: str, font, max_width: float) -> str:
    text = " ".join(text.split())
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text:
        text = text[:-1].rstrip()
        candidate = text + "…"
        if draw.textlength(candidate, font=font) <= max_width:
            return candidate
    return ""


def _add_footer(
    image: Image.Image,
    size: int,
    reading: str,
    gloss: str,
) -> None:
    draw = ImageDraw.Draw(image)
    margin = int(size * 0.08)
    rule_y = int(size * 0.885)
    draw.line(
        (margin, rule_y, int(size * 0.78), rule_y),
        fill=(120, 104, 82, 80),
        width=max(1, size // 700),
    )

    if reading:
        font = _text_font(int(size * 0.046), bold=True)
        draw.text(
            (margin, int(size * 0.897)),
            reading,
            font=font,
            fill=(30, 26, 21, 240),
        )

    if gloss:
        short = gloss.split(";")[0].strip() or gloss.strip()
        font = _text_font(int(size * 0.021))
        line = _fit_line(draw, short, font, size * 0.70)
        draw.text(
            (margin, int(size * 0.949)),
            line,
            font=font,
            fill=(76, 66, 54, 235),
        )


def _add_seal(image: Image.Image, char: str, size: int, rng: random.Random) -> None:
    seal_size = int(size * 0.078)
    margin = int(size * 0.072)
    seal = Image.new("RGBA", (seal_size, seal_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(seal)
    radius = max(3, int(seal_size * 0.10))
    draw.rounded_rectangle(
        (0, 0, seal_size - 1, seal_size - 1),
        radius=radius,
        fill=SEAL_RED + (232,),
    )
    inset = max(2, int(seal_size * 0.075))
    draw.rounded_rectangle(
        (inset, inset, seal_size - 1 - inset, seal_size - 1 - inset),
        radius=max(2, radius // 2),
        outline=(255, 245, 232, 150),
        width=max(1, seal_size // 42),
    )

    try:
        font = ImageFont.truetype(str(PEN_FONT), int(seal_size * 0.58))
        draw.text(
            (seal_size / 2, seal_size / 2 + seal_size * 0.015),
            char,
            font=font,
            fill=(249, 242, 228, 240),
            anchor="mm",
        )
    except Exception:
        pass

    seal = seal.rotate(rng.uniform(-3.5, 3.5), resample=Image.Resampling.BICUBIC, expand=True)
    image.alpha_composite(seal, (size - margin - seal.width, size - margin - seal.height))


def finish(
    mask: Image.Image,
    phrase: str,
    seal_char: str | None = None,
    seed: int | None = None,
    reading: str = "",
    gloss: str = "",
    show_seal: bool = False,
) -> Image.Image:
    """Render an L-mode glyph mask as a finished RGB artwork."""
    size = mask.size[0]
    seed = seed_for(phrase) if seed is None else seed
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    paper = Image.fromarray(_paper(size, rng), "RGB").convert("RGBA")

    bleed_alpha = mask.filter(ImageFilter.GaussianBlur(3.0)).point(
        lambda value: int(value * 0.10)
    )
    bleed = Image.new("RGBA", (size, size), INK + (0,))
    bleed.putalpha(bleed_alpha)
    paper = Image.alpha_composite(paper, bleed)

    ink_alpha = mask.filter(ImageFilter.GaussianBlur(0.8))
    ink = Image.new("RGBA", (size, size), INK + (0,))
    ink.putalpha(ink_alpha)
    paper = Image.alpha_composite(paper, ink)

    if reading or gloss:
        _add_footer(paper, size, reading, gloss)
    if show_seal and seal_char:
        _add_seal(paper, seal_char, size, py_rng)

    return _vignette(paper.convert("RGB"))
