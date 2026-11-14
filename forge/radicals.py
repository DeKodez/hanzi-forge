"""Radical shapes, variants, and placement rules used by the composer.

Chinese characters use different shapes for the same component depending on
where it sits: 水 becomes 氵 on the left, 心 becomes 忄 on the left, 火 becomes
灬 underneath, and so on.  These maps give forged glyphs the right silhouette.
"""

from __future__ import annotations

from .ids import ALL_OPS, STRUCTURE_NAMES

LEFT_VARIANTS = {
    "水": "氵",
    "心": "忄",
    "人": "亻",
    "手": "扌",
    "言": "讠",
    "糸": "纟",
    "犬": "犭",
    "食": "饣",
    "金": "钅",
    "示": "礻",
    "衣": "衤",
    "玉": "王",
    "阜": "阝",
    "刀": "刂",
    "冰": "冫",
}

TOP_VARIANTS = {
    "艸": "艹",
    "爪": "爫",
    "网": "罒",
    "冰": "冫",
}

BOTTOM_VARIANTS = {
    "火": "灬",
}

RIGHT_VARIANTS = {
    "邑": "阝",
}

# Forms that mean the same component, used only to avoid duplicate candidates.
VARIANT_GROUP = {
    "氵": "水",
    "忄": "心",
    "亻": "人",
    "扌": "手",
    "讠": "言",
    "訁": "言",
    "纟": "糸",
    "糹": "糸",
    "犭": "犬",
    "饣": "食",
    "飠": "食",
    "钅": "金",
    "釒": "金",
    "礻": "示",
    "衤": "衣",
    "刂": "刀",
    "灬": "火",
    "爫": "爪",
    "罒": "网",
    "艹": "艸",
    "阝": "阜",
    "冫": "冰",
    "馬": "马",
    "鳥": "鸟",
    "魚": "鱼",
    "車": "车",
    "門": "门",
    "龍": "龙",
    "風": "风",
    "雲": "云",
    "電": "电",
    "貝": "贝",
    "頁": "页",
    "齒": "齿",
    "龜": "龟",
}

TOP_POSITION = set("宀艹雨穴竹罒疒广厂尸户冖冫亠覀彑")

ENCLOSURES = {
    "门": "⿵",
    "囗": "⿴",
    "广": "⿸",
    "疒": "⿸",
    "尸": "⿸",
    "户": "⿸",
    "勹": "⿵",
    "匚": "⿷",
    "冂": "⿵",
    "几": "⿵",
    "辶": "⿺",
    "廴": "⿺",
}

STOP_COMPONENTS = set("一丨丶丿乙亅二亠冂冖凵匚匸十卜卩厶又廴彐彡攴无")

BASE_FORM = {
    "氵": "水",
    "忄": "心",
    "亻": "人",
    "扌": "手",
    "讠": "言",
    "纟": "糸",
    "犭": "犬",
    "饣": "食",
    "钅": "金",
    "礻": "示",
    "衤": "衣",
    "刂": "刀",
    "灬": "火",
    "爫": "爪",
    "罒": "网",
    "艹": "艸",
    "冫": "冰",
}


def canonical(component: str) -> str:
    return VARIANT_GROUP.get(component, component)


def structure_for(component: str) -> str:
    """Pick a natural binary structure for a component used as the anchor."""
    if component in ENCLOSURES:
        return ENCLOSURES[component]
    if component in TOP_POSITION:
        return "⿱"
    return "⿰"


def slot(structure: str, position: int) -> str:
    if structure in {"⿰", "⿷"}:
        return "left" if position == 0 else "right"
    if structure in {"⿱", "⿵", "⿶"}:
        return "top" if position == 0 else "bottom"
    if structure in {"⿴", "⿸", "⿹", "⿺"}:
        return "outer" if position == 0 else "inner"
    return "left"


def shape_for(component: str, structure: str, position: int) -> str:
    where = slot(structure, position)
    if where == "left":
        component = BASE_FORM.get(component, component)
        return LEFT_VARIANTS.get(component) or component

    # A left-side variant used on the right or bottom looks wrong; restore the
    # full form first, then apply the variant for that position if one exists.
    component = BASE_FORM.get(component, component)
    if where == "top":
        return TOP_VARIANTS.get(component) or component
    if where == "bottom":
        return BOTTOM_VARIANTS.get(component) or component
    if where == "right":
        return RIGHT_VARIANTS.get(component) or component
    return component


def describe_structure(structure: str) -> str:
    if structure in STRUCTURE_NAMES:
        return STRUCTURE_NAMES[structure]
    if structure and structure[0] in ALL_OPS:
        return "nested"
    return structure


def describe_layout(structure: str) -> str:
    return {
        "⿰": "side by side",
        "⿱": "stacked",
        "⿴": "inside a full frame",
        "⿵": "under a roof",
        "⿶": "in an open bowl",
        "⿷": "beside a frame",
        "⿸": "tucked under a shelter",
        "⿹": "wrapped from above",
        "⿺": "carried on a sweep",
        "⿻": "overlaid",
        "⿼": "inside a right-hand frame",
        "⿽": "inside a lower-right frame",
        "⿾": "mirrored",
        "⿿": "rotated",
    }.get(structure, "combined")
