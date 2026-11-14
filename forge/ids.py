"""Minimal Ideographic Description Sequence (IDS) parsing.

Make Me a Hanzi stores each character's structure as an IDS string such as
``⿰氵每`` (left-right) or ``⿱雨林`` (top-bottom).  This module parses those
strings just far enough to pull components out and to recognise when a
forged composition already exists as a real character.
"""

from __future__ import annotations

BINARY_OPS = set("⿰⿱⿴⿵⿶⿷⿸⿹⿺⿻⿼⿽")
UNARY_OPS = set("⿾⿿")
TERNARY_OPS = set("⿲⿳")
ALL_OPS = BINARY_OPS | UNARY_OPS | TERNARY_OPS

UNKNOWN = "？"

STRUCTURE_NAMES = {
    "⿰": "left-right",
    "⿱": "top-bottom",
    "⿴": "full enclosure",
    "⿵": "enclosure from above",
    "⿶": "enclosure from below",
    "⿷": "enclosure from the left",
    "⿸": "upper-left enclosure",
    "⿹": "upper-right enclosure",
    "⿺": "lower-left enclosure",
    "⿻": "overlay",
    "⿼": "enclosure from the right",
    "⿽": "enclosure from the lower right",
    "⿾": "horizontal reflection",
    "⿿": "rotation",
}


def _parse_node(text: str, index: int = 0):
    if index >= len(text):
        raise ValueError("truncated IDS")
    char = text[index]
    if char in BINARY_OPS:
        left, index = _parse_node(text, index + 1)
        right, index = _parse_node(text, index)
        return (char, left, right), index
    if char in UNARY_OPS:
        child, index = _parse_node(text, index + 1)
        return (char, child), index
    if char in TERNARY_OPS:
        first, index = _parse_node(text, index + 1)
        second, index = _parse_node(text, index)
        third, index = _parse_node(text, index)
        return (char, first, second, third), index
    if char in ALL_OPS:
        raise ValueError(f"unsupported IDS operator {char!r}")
    return char, index + 1


def parse_ids(decomposition: str | None):
    """Return a nested tuple for an IDS string, or ``None`` if unknown."""
    if not decomposition or decomposition == UNKNOWN:
        return None
    try:
        node, end = _parse_node(decomposition)
    except ValueError:
        return None
    if end != len(decomposition):
        return None
    return node


def to_ids(node) -> str:
    if isinstance(node, str):
        return node
    return node[0] + "".join(to_ids(child) for child in node[1:])


def top_level_components(node) -> tuple[str, ...]:
    """The direct children of a parsed node (leaves only are useful to us)."""
    if node is None or isinstance(node, str):
        return ()
    return tuple(child for child in node[1:] if isinstance(child, str))


def first_leaf_parts(node) -> tuple[str, ...]:
    """Direct children, descending through nested groups until two are found."""
    if node is None or isinstance(node, str):
        return ()
    children = node[1:]
    leaves: list[str] = []
    for child in children:
        if isinstance(child, str):
            leaves.append(child)
        else:
            leaves.extend(first_leaf_parts(child))
    return tuple(leaves)


def structure_of(node) -> str | None:
    if node is None or isinstance(node, str):
        return None
    return node[0]


def load_ids_database(path) -> set[str]:
    """All known character decompositions from the CJKVI/CHISE IDS database."""
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return set()

    known: set[str] = set()
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            for raw in fields[2:]:
                sequence = raw.split("[", 1)[0].replace(" ", "").strip()
                if sequence and sequence[0] in ALL_OPS:
                    known.add(sequence)
    return known


def all_leaves(node) -> list[str]:
    """Every leaf of an IDS tree, in left-to-right order."""
    if node is None:
        return []
    if isinstance(node, str):
        return [node]
    leaves: list[str] = []
    for child in node[1:]:
        leaves.extend(all_leaves(child))
    return leaves


def decomposition_set(records: dict[str, dict]) -> set[str]:
    """Every real character's raw IDS string, for duplicate detection."""
    return {
        record["decomposition"]
        for record in records.values()
        if record.get("decomposition") and record["decomposition"] != UNKNOWN
    }
