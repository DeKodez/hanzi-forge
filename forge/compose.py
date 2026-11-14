"""Compose two glyph components into a new character-shaped mask."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

from .ids import parse_ids
from .paths import BRUSH_FONT, PEN_FONT
from .radicals import LEFT_VARIANTS, TOP_POSITION, canonical, shape_for

WIDE_TOP = set(TOP_POSITION) | {"雨", "艹", "宀"}
MAX_LEAF_STRETCH = 2.8
NARROW_LEFT = set(LEFT_VARIANTS.values()) | {"阝", "刂"}
TOP_HEIGHT_CAPS = {
    "宀": 0.17,
    "艹": 0.22,
    "冖": 0.20,
    "亠": 0.20,
    "覀": 0.26,
    "穴": 0.28,
    "竹": 0.30,
    "罒": 0.26,
    "雨": 0.40, "覀": 0.26, "冫": 0.16, "彐": 0.18,
}
TOP_WIDTH_RATIOS = {
    "宀": 0.88,
}
LEFT_WIDTH_RATIOS = {
    "氵": 0.22, "忄": 0.22, "亻": 0.22, "扌": 0.22, "讠": 0.22,
    "纟": 0.22, "犭": 0.22, "饣": 0.22, "钅": 0.22, "礻": 0.22,
    "衤": 0.22, "阝": 0.22, "刂": 0.22,
    "土": 0.28, "王": 0.28, "足": 0.30, "马": 0.30, "鱼": 0.30,
    "鸟": 0.30, "虫": 0.28, "车": 0.30, "耳": 0.30, "目": 0.30,
    "石": 0.32, "金": 0.30, "食": 0.30, "言": 0.28, "糸": 0.28,
    "木": 0.34, "火": 0.32, "口": 0.30, "女": 0.32, "子": 0.30,
    "山": 0.30, "日": 0.30, "月": 0.30, "田": 0.30, "贝": 0.30,
    "禾": 0.32, "米": 0.32, "骨": 0.34, "革": 0.34, "页": 0.32,
    "齿": 0.32, "龟": 0.34, "鬼": 0.34, "鹿": 0.34, "麻": 0.34,
    "黑": 0.34, "鼓": 0.38, "鼠": 0.36, "鼻": 0.38,
}

LAYOUTS: dict[str, tuple[tuple[float, float, float, float], ...]] = {
    "⿰": ((0.01, 0.06, 0.31, 0.94), (0.30, 0.06, 0.99, 0.94)),
    "⿱": ((0.04, 0.01, 0.96, 0.44), (0.14, 0.43, 0.86, 0.99)),
    "⿴": ((0.00, 0.00, 1.00, 1.00), (0.23, 0.23, 0.77, 0.79)),
    "⿵": ((0.00, 0.00, 1.00, 1.00), (0.27, 0.29, 0.79, 0.86)),
    "⿶": ((0.00, 0.00, 1.00, 1.00), (0.24, 0.14, 0.80, 0.70)),
    "⿷": ((0.00, 0.00, 1.00, 1.00), (0.34, 0.20, 0.84, 0.82)),
    "⿸": ((0.00, 0.00, 1.00, 1.00), (0.37, 0.42, 0.86, 0.90)),
    "⿹": ((0.00, 0.00, 1.00, 1.00), (0.15, 0.34, 0.69, 0.86)),
    "⿺": ((0.00, 0.00, 1.00, 1.00), (0.28, 0.12, 0.84, 0.64)),
}


@lru_cache(maxsize=128)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=4096)
def _can_render(path: str, char: str) -> bool:
    try:
        font = _font(path, 96)
        missing = font.getmask("\U0010ffff")
        mask = font.getmask(char)
    except Exception:
        return False
    if mask.size[0] == 0 or mask.size[1] == 0:
        return False
    return bytes(mask) != bytes(missing)


def font_chain(style: str = "brush") -> tuple[Path, ...]:
    if style == "pen":
        return (PEN_FONT, BRUSH_FONT)
    return (BRUSH_FONT, PEN_FONT)


def font_for(char: str, style: str = "brush") -> Path:
    chain = font_chain(style)
    for path in chain:
        if _can_render(str(path), char):
            return path
    return chain[0]


def can_render(char: str, style: str = "brush") -> bool:
    return any(_can_render(str(path), char) for path in font_chain(style))


@lru_cache(maxsize=1024)
def _ink_shape(shape: str, style: str) -> Image.Image:
    side = 1024
    cell = Image.new("L", (side, side), 0)
    draw = ImageDraw.Draw(cell)
    path = font_for(shape, style)
    font = _font(str(path), int(side * 0.88))
    draw.text((side / 2, side / 2), shape, font=font, fill=255, anchor="mm")
    bbox = cell.getbbox()
    if bbox is None:
        return Image.new("L", (1, 1), 0)
    return cell.crop(bbox)


def _paste_resized(
    canvas: Image.Image,
    ink: Image.Image,
    width: int,
    height: int,
    x: int,
    y: int,
    margin: float = 0.002,
) -> None:
    width = max(1, width)
    height = max(1, height)
    mx = max(1, int(width * margin))
    my = max(1, int(height * margin))
    target_w = max(1, width - 2 * mx)
    target_h = max(1, height - 2 * my)
    scale_x = target_w / max(1, ink.width)
    scale_y = target_h / max(1, ink.height)
    if max(scale_x / scale_y, scale_y / scale_x) > MAX_LEAF_STRETCH:
        scale = min(scale_x, scale_y)
        fit_w = max(1, int(ink.width * scale))
        fit_h = max(1, int(ink.height * scale))
        fitted = ink.resize((fit_w, fit_h), Image.Resampling.LANCZOS)
        slot = Image.new("L", (width, height), 0)
        slot.paste(fitted, ((width - fit_w) // 2, (height - fit_h) // 2))
        region = canvas.crop((x, y, x + width, y + height))
        canvas.paste(ImageChops.lighter(region, slot), (x, y))
        return
    resized = ink.resize((target_w, target_h), Image.Resampling.LANCZOS)
    region = canvas.crop((x, y, x + width, y + height))
    canvas.paste(ImageChops.lighter(region, resized), (x, y))


def render_shape(shape: str, width: int, height: int, style: str = "brush") -> Image.Image:
    """Render one component's ink stretched to fill a width x height slot."""
    ink = _ink_shape(shape, style)
    out = Image.new("L", (max(1, width), max(1, height)), 0)
    _paste_resized(out, ink, width, height, 0, 0)
    return out


def render_shape_fit(
    shape: str, width: int, height: int, style: str = "brush"
) -> Image.Image:
    """Render a full character preserving its aspect ratio inside a box."""
    ink = _ink_shape(shape, style)
    scale = min(width / ink.width, height / ink.height)
    target_w = max(1, int(ink.width * scale))
    target_h = max(1, int(ink.height * scale))
    ink = ink.resize((target_w, target_h), Image.Resampling.LANCZOS)
    out = Image.new("L", (max(1, width), max(1, height)), 0)
    out.paste(ink, ((width - target_w) // 2, (height - target_h) // 2))
    return out


def _top_cap(top: str, bottom: str | tuple, style: str) -> float:
    cap = TOP_HEIGHT_CAPS.get(top, 0.40)
    if top == "宀" and isinstance(bottom, str):
        ink = _ink_shape(bottom, style)
        aspect = ink.width / max(1, ink.height)
        if aspect >= 1.05:
            cap = 0.26
        elif aspect <= 0.75:
            cap = 0.22
        else:
            cap = 0.24
    elif top == "宀" and isinstance(bottom, tuple):
        cap = 0.24
    return cap


def compose_mask(
    structure: str,
    parts: tuple[str, ...],
    size: int,
    style: str = "brush",
    left_scale: str | None = None,
) -> Image.Image:
    """Render *parts* with *structure* into a square L mask of *size*."""
    canvas = Image.new("L", (size, size), 0)
    area = int(size * 0.74)
    offset = (size - area) // 2

    if len(parts) < 2:
        if parts:
            canvas.paste(
                render_shape_fit(parts[0], area, area, style), (offset, offset)
            )
        return canvas

    if structure == "⿱":
        top, bottom = parts[0], parts[1]
        top_ink = _ink_shape(top, style)
        bottom_shape = shape_for(bottom, "⿱", 1)
        short_bottom = bottom_shape == "灬"

        top_ratio = TOP_WIDTH_RATIOS.get(
            top, 0.90 if top in WIDE_TOP else 0.55
        )
        top_w = int(area * top_ratio)
        natural_h = top_ink.height * top_w / max(1, top_ink.width)
        if short_bottom and top not in WIDE_TOP:
            cap = 0.62
        else:
            cap = _top_cap(top, bottom, style)
        top_h = int(min(max(natural_h, area * 0.10), area * cap))
        top_x = offset + (area - top_w) // 2

        gap = int(area * 0.004)
        bottom_w = int(area * 0.72)
        bottom_x = offset + (area - bottom_w) // 2

        if short_bottom:
            bottom_ink = _ink_shape(bottom, style)
            natural_bottom = bottom_ink.height * bottom_w / max(1, bottom_ink.width)
            bottom_h = int(min(max(natural_bottom, area * 0.06), area * 0.20))
            total_h = top_h + gap + bottom_h
            top_y = offset + max(int(area * 0.01), (area - total_h) // 2)
            bottom_y = top_y + top_h + gap
            _paste_resized(canvas, top_ink, top_w, top_h, top_x, top_y)
            _paste_resized(
                canvas, bottom_ink, bottom_w, bottom_h, bottom_x, bottom_y
            )
            return canvas

        top_y = offset + int(area * 0.01)
        _paste_resized(canvas, top_ink, top_w, top_h, top_x, top_y)
        bottom_y = top_y + top_h + gap
        bottom_h = offset + area - bottom_y
        if bottom_h < int(area * 0.30):
            top_h = int(area * 0.42)
            bottom_y = top_y + top_h + gap
            bottom_h = offset + area - bottom_y
        _paste_resized(
            canvas, _ink_shape(bottom, style), bottom_w, bottom_h, bottom_x, bottom_y
        )
        return canvas

    if structure == "⿰":
        left, right = parts[0], parts[1]
        left_ink = _ink_shape(left, style)

        left_h = int(area * 0.88)
        if left in LEFT_WIDTH_RATIOS:
            left_ratio = LEFT_WIDTH_RATIOS[left]
        elif left_scale in {"narrow", "medium", "full"}:
            left_ratio = {"narrow": 0.22, "medium": 0.30, "full": 0.38}[left_scale]
        else:
            left_ratio = 0.24 if left in NARROW_LEFT else 0.38
        left_w = int(area * left_ratio)
        left_x = offset + int(area * 0.01)
        left_y = offset + (area - left_h) // 2
        _paste_resized(canvas, left_ink, left_w, left_h, left_x, left_y)

        gap = int(area * 0.006)
        right_x = left_x + left_w + gap
        right_w = offset + area - right_x
        right_h = int(area * 0.88)
        right_y = offset + (area - right_h) // 2
        _paste_resized(
            canvas, _ink_shape(right, style), right_w, right_h, right_x, right_y
        )
        return canvas

    boxes = LAYOUTS.get(structure, LAYOUTS["⿰"])
    for part, (x0, y0, x1, y1) in zip(parts, boxes):
        box = (
            offset + int(x0 * area),
            offset + int(y0 * area),
            offset + int(x1 * area),
            offset + int(y1 * area),
        )
        width = max(1, box[2] - box[0])
        height = max(1, box[3] - box[1])
        _paste_resized(
            canvas, _ink_shape(part, style), width, height, box[0], box[1]
        )
    return canvas


def _leaves_of(node) -> list[str]:
    if isinstance(node, str):
        return [node]
    leaves: list[str] = []
    for child in node[1:]:
        leaves.extend(_leaves_of(child))
    return leaves


def _complexity(node) -> int:
    return max(1, len(_leaves_of(node)))


def _render_leaf(leaf: str, width: int, height: int, style: str) -> Image.Image:
    out = Image.new("L", (max(1, width), max(1, height)), 0)
    _paste_resized(out, _ink_shape(leaf, style), width, height, 0, 0)
    return out


def _render_node(node, width: int, height: int, style: str) -> Image.Image:
    if isinstance(node, str):
        return _render_leaf(node, width, height, style)

    op = node[0]
    children = node[1:]
    if len(children) == 1:
        return _render_node(children[0], width, height, style)

    width = max(1, width)
    height = max(1, height)
    gap = max(1, int(min(width, height) * 0.004))
    canvas = Image.new("L", (width, height), 0)

    if op in {"⿰", "⿲", "⿱", "⿳"}:
        horizontal = op in {"⿰", "⿲"}
        if len(children) == 2:
            first, second = children
            if horizontal:
                if (
                    isinstance(first, str)
                    and isinstance(second, str)
                    and canonical(first) == canonical(second)
                ):
                    first_w = int((width - gap) * 0.5)
                elif isinstance(first, str):
                    ratio = LEFT_WIDTH_RATIOS.get(
                        first, 0.24 if first in NARROW_LEFT else 0.38
                    )
                    first_w = int(width * ratio)
                else:
                    ratio = _complexity(first) / (
                        _complexity(first) + _complexity(second)
                    )
                    ratio = min(0.48, max(0.35, ratio))
                    first_w = int((width - gap) * ratio)
                compact_left = (
                    isinstance(first, str)
                    and isinstance(second, tuple)
                    and bool(second)
                    and second[0] in {"⿱", "⿳"}
                    and (
                        first in NARROW_LEFT
                        or LEFT_WIDTH_RATIOS.get(first, 1.0) <= 0.22
                    )
                )
                if compact_left:
                    first_w = int(first_w * 0.90)
                second_w = width - first_w - gap
                if second_w < int(width * 0.20):
                    second_w = int(width * 0.20)
                    first_w = width - second_w - gap
                child_h = int(height * 0.88)
                left_h = int(height * 0.80) if compact_left else child_h
                left_y = (height - left_h) // 2
                right_y = (height - child_h) // 2
                left = _render_node(first, first_w, left_h, style)
                right = _render_node(second, second_w, child_h, style)
                region = canvas.crop((0, left_y, first_w, left_y + left_h))
                canvas.paste(ImageChops.lighter(region, left), (0, left_y))
                region = canvas.crop(
                    (first_w + gap, right_y, width, right_y + child_h)
                )
                canvas.paste(
                    ImageChops.lighter(region, right), (first_w + gap, right_y)
                )
            else:
                short_bottom = (
                    isinstance(second, str)
                    and shape_for(second, "⿱", 1) == "灬"
                )
                if (
                    isinstance(first, str)
                    and isinstance(second, str)
                    and canonical(first) == canonical(second)
                ):
                    first_h = int((height - gap) * 0.5)
                elif isinstance(first, str):
                    ink = _ink_shape(first, style)
                    if (
                        isinstance(second, tuple)
                        and len(_leaves_of(second)) == 2
                        and first not in WIDE_TOP
                    ):
                        first_w = int(width * 0.48)
                    else:
                        first_w = int(
                            width * TOP_WIDTH_RATIOS.get(
                                first, 0.90 if first in WIDE_TOP else 0.55
                            )
                        )
                    natural_h = ink.height * first_w / max(1, ink.width)
                    if short_bottom and first not in WIDE_TOP:
                        cap = 0.62
                    else:
                        cap = _top_cap(first, second, style)
                    first_h = int(
                        min(max(natural_h, height * 0.10), height * cap)
                    )
                else:
                    ratio = _complexity(first) / (
                        _complexity(first) + _complexity(second)
                    )
                    ratio = min(0.62, max(0.38, ratio))
                    first_h = int((height - gap) * ratio)

                if short_bottom:
                    bottom_width = int(width * 0.72)
                    bottom_ink = _ink_shape(second, style)
                    natural_bottom = (
                        bottom_ink.height * bottom_width / max(1, bottom_ink.width)
                    )
                    bottom_h = int(
                        min(max(natural_bottom, height * 0.06), height * 0.20)
                    )
                    total_h = first_h + gap + bottom_h
                    top_y = max(0, (height - total_h) // 2)
                    top = _render_node(first, width, first_h, style)
                    region = canvas.crop((0, top_y, width, top_y + first_h))
                    canvas.paste(ImageChops.lighter(region, top), (0, top_y))
                    bottom = _render_leaf(second, bottom_width, bottom_h, style)
                    bottom_x = (width - bottom_width) // 2
                    bottom_y = top_y + first_h + gap
                    region = canvas.crop(
                        (
                            bottom_x,
                            bottom_y,
                            bottom_x + bottom_width,
                            bottom_y + bottom_h,
                        )
                    )
                    canvas.paste(
                        ImageChops.lighter(region, bottom), (bottom_x, bottom_y)
                    )
                    return canvas

                second_h = height - first_h - gap
                if second_h < int(height * 0.20):
                    second_h = int(height * 0.20)
                    first_h = height - second_h - gap
                top = _render_node(first, width, first_h, style)
                if isinstance(second, str):
                    bottom_width = int(width * 0.72)
                    bottom = _render_node(second, bottom_width, second_h, style)
                else:
                    bottom_width = int(width * 0.82)
                    bottom = _render_node(second, bottom_width, second_h, style)
                canvas.paste(
                    ImageChops.lighter(canvas.crop((0, 0, width, first_h)), top),
                    (0, 0),
                )
                bottom_x = (width - bottom_width) // 2
                canvas.paste(
                    ImageChops.lighter(
                        canvas.crop(
                            (bottom_x, first_h + gap, bottom_x + bottom_width, height)
                        ),
                        bottom,
                    ),
                    (bottom_x, first_h + gap),
                )
        else:
            weights = [max(1, _complexity(child)) for child in children]
            total = sum(weights)
            available = (width if horizontal else height) - gap * (len(children) - 1)
            sizes = [max(1, int(available * weight / total)) for weight in weights]
            sizes[-1] = available - sum(sizes[:-1])
            offset = 0
            for child, child_size in zip(children, sizes):
                if horizontal:
                    mask = _render_node(child, child_size, height, style)
                    box = (offset, 0, offset + child_size, height)
                else:
                    mask = _render_node(child, width, child_size, style)
                    box = (0, offset, width, offset + child_size)
                region = canvas.crop(box)
                canvas.paste(ImageChops.lighter(region, mask), box[:2])
                offset += child_size + gap
        return canvas

    if op in {"⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺"}:
        outer, inner = children[0], children[-1]
        outer_mask = _render_node(outer, width, height, style)
        canvas.paste(outer_mask, (0, 0))
        boxes = {
            "⿴": (0.23, 0.23, 0.77, 0.79),
            "⿵": (0.27, 0.29, 0.79, 0.86),
            "⿶": (0.24, 0.14, 0.80, 0.70),
            "⿷": (0.34, 0.20, 0.84, 0.82),
            "⿸": (0.30, 0.36, 0.84, 0.88),
            "⿹": (0.15, 0.34, 0.69, 0.86),
            "⿺": (0.28, 0.12, 0.84, 0.64),
        }
        x0, y0, x1, y1 = boxes[op]
        inner_w = max(1, int((x1 - x0) * width))
        inner_h = max(1, int((y1 - y0) * height))
        inner_mask = _render_node(inner, inner_w, inner_h, style)
        ix, iy = int(x0 * width), int(y0 * height)
        region = canvas.crop((ix, iy, ix + inner_w, iy + inner_h))
        canvas.paste(ImageChops.lighter(region, inner_mask), (ix, iy))
        return canvas

    # Unknown operator: fall back to a horizontal split.
    return _render_node(("⿰", *children), width, height, style)


def compose_ids(
    ids: str,
    parts: tuple[str, ...],
    size: int,
    style: str = "brush",
    left_scale: str | None = None,
) -> Image.Image:
    """Render an arbitrary (possibly nested) IDS tree into a square mask."""
    node = parse_ids(ids)
    if node is None or isinstance(node, str):
        return compose_mask("⿰", tuple(parts[:2]), size, style)

    leaves = _leaves_of(node)
    if len(leaves) == 2 and node[0] in LAYOUTS:
        return compose_mask(node[0], tuple(leaves), size, style, left_scale=left_scale)

    area = int(size * 0.74)
    offset = (size - area) // 2
    mask = _render_node(node, area, area, style)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(mask, (offset, offset))
    return canvas


def layout_trace(
    ids: str,
    parts: tuple[str, ...],
    size: int = 384,
    style: str = "brush",
) -> tuple[Image.Image, list[tuple[str, int, int, int, int]]]:
    """Render once while recording each leaf's slot and natural ink size."""
    global _ink_shape, _paste_resized
    original_ink = _ink_shape
    original_paste = _paste_resized
    shapes_by_id: dict[int, str] = {}
    trace: list[tuple[str, int, int, int, int]] = []

    def traced_ink(shape: str, style: str = "brush") -> Image.Image:
        image = original_ink(shape, style)
        shapes_by_id[id(image)] = shape
        return image

    def traced_paste(
        canvas: Image.Image,
        image: Image.Image,
        width: int,
        height: int,
        x: int,
        y: int,
        margin: float = 0.002,
    ) -> None:
        trace.append(
            (
                shapes_by_id.get(id(image), "?"),
                width,
                height,
                image.width,
                image.height,
            )
        )
        return original_paste(canvas, image, width, height, x, y, margin)

    _ink_shape = traced_ink
    _paste_resized = traced_paste
    try:
        mask = compose_ids(ids, parts, size, style)
    finally:
        _ink_shape = original_ink
        _paste_resized = original_paste
    return mask, trace


def character_mask(char: str, size: int, style: str = "brush") -> Image.Image:
    area = int(size * 0.74)
    offset = (size - area) // 2
    mask = render_shape_fit(char, area, area, style)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(mask, (offset, offset))
    return canvas
