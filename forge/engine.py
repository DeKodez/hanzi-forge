"""Phrase -> forged character pipeline.

The heavy lifting is semantic retrieval: a pretrained sentence embedding model
finds the real characters whose dictionary glosses are closest to the English
phrase.  Their radicals and etymological components then vote for the parts of
a new glyph, which is composed and rendered locally.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from PIL import Image

from .compose import (
    LAYOUTS,
    LEFT_WIDTH_RATIOS,
    can_render,
    character_mask,
    compose_ids,
    compose_mask,
    layout_trace,
)
from .embed import Match, SemanticIndex, frequency_prior
from .ids import (
    ALL_OPS,
    all_leaves,
    decomposition_set,
    load_ids_database,
    parse_ids,
    to_ids,
    top_level_components,
)
from .radicals import (
    BASE_FORM,
    ENCLOSURES,
    LEFT_VARIANTS,
    STOP_COMPONENTS,
    TOP_POSITION,
    canonical,
    describe_layout,
    shape_for,
    structure_for,
)
from . import llm
from .paths import IDS_DB
from .render import finish, seed_for
from .textdata import is_cjk, short_meaning

_LAYOUT_WORDS = re.compile(
    r"(\b(?:above|below|on top of|underneath|beneath|beside|side by side|stacked|"
    r"to the (?:left|right)|on the (?:left|right)|left side|right side|inside|"
    r"enclos(?:es|ed|ing)|surrounds|wrapped around|sits? (?:above|over|under|beside))\b)",
    re.IGNORECASE,
)


_TONE_MARKS = {
    "a": ["ā", "á", "ǎ", "à"],
    "o": ["ō", "ó", "ǒ", "ò"],
    "e": ["ē", "é", "ě", "è"],
    "i": ["ī", "í", "ǐ", "ì"],
    "u": ["ū", "ú", "ǔ", "ù"],
    "ü": ["ǖ", "ǘ", "ǚ", "ǜ"],
}
_ACCENT_BASE = {
    mark: base for base, marks in _TONE_MARKS.items() for mark in marks
}


def pinyin_base(pinyin: str) -> str:
    text = (pinyin or "").strip().lower()
    for separator in (",", "/"):
        text = text.split(separator)[0]
    letters: list[str] = []
    for char in text:
        if char in _ACCENT_BASE:
            letters.append(_ACCENT_BASE[char])
        elif char.isalpha() or char == "ü":
            letters.append(char)
    return "".join(letters)


def apply_tone(base: str, tone: int) -> str:
    if not base:
        return ""
    vowel_index = None
    for vowel in ("a", "o", "e"):
        if vowel in base:
            vowel_index = base.index(vowel)
            break
    if vowel_index is None:
        if "iu" in base:
            vowel_index = base.index("u")
        elif "ui" in base:
            vowel_index = base.index("i")
        else:
            positions = [
                index for index, char in enumerate(base) if char in _TONE_MARKS
            ]
            if not positions:
                return base
            vowel_index = positions[-1]
    vowel = base[vowel_index]
    if vowel not in _TONE_MARKS:
        return base
    mark = _TONE_MARKS[vowel][max(0, min(3, tone - 1))]
    return base[:vowel_index] + mark + base[vowel_index + 1:]


def sanitize_layout_text(text: str) -> str:
    """Drop sentences that describe a placement the renderer may have overridden."""
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [sentence for sentence in sentences if not _LAYOUT_WORDS.search(sentence)]
    return " ".join(kept).strip()


FUNCTION_CHARS = set(
    "的了是在我你他她它这那和与就都也但或又还只才已呢吗吧啊呀哦嗯个们着过地得所而且之其不没"
)
FALLBACK_COMPONENTS = ("心", "水", "火", "木", "日", "山", "人", "口")
COMPLEX_SEMANTIC_RADICALS = {
    "鹿", "黑", "鼓", "鼠", "鼻", "麻", "龙", "龍", "鱼", "魚", "鸟", "鳥",
    "马", "馬", "龟", "龜", "齐", "齊", "齿", "齒", "黄", "黃", "黾", "黽",
    "鼎", "龠",
}
COMMON_RADICALS = set("宀艹冖广厂尸户门囗雨穴竹罒疒辶廴匚冂勹")
STRUCTURAL = set("宀艹冖广厂尸户门囗穴辶廴匚冂勹")
PREFERRED_PARTNERS = ("心", "水", "火", "木", "人", "口", "日", "手", "足", "力", "女", "子", "山", "金", "言")
SIMPLE_COMPONENTS = set(
    "口日曰月木水火山土女子人大天小力刀王玉田石禾米虫鱼鸟马车门金食衣言糸犬心手足耳目雨云龙风电光中上下内外出入来去永生"
)
LEFT_RADICALS = set(LEFT_VARIANTS) | set(LEFT_VARIANTS.values()) | set(
    "口日月目石禾米虫鱼马鸟车木火手足耳骨革页王贝田力立矢示女子山土"
)
TOP_RADICALS = set(TOP_POSITION) | {"雨", "艹", "宀"}


@dataclass
class Candidate:
    structure: str | None
    parts: tuple[str, ...]
    shapes: tuple[str, ...]
    explanation: str
    source: str | None = None
    image: Image.Image | None = None
    origin: str = "deterministic"
    reading: str = ""
    gloss: str = ""
    rationale: str = ""
    semantic: str = ""
    phonetic: str = ""
    left_scale: str = ""

    @property
    def layout(self) -> str:
        if not self.structure:
            return ""
        return describe_layout(self.structure)


@dataclass
class ForgeResult:
    phrase: str
    mode: str
    candidates: list[Candidate]
    matches: list[Match]
    components: list[tuple[str, float, list[str]]] = field(default_factory=list)
    warning: str = ""
    llm_used: bool = False
    llm_rejected: int = 0
    analysis: dict | None = None


class ForgeEngine:
    def __init__(self, index: SemanticIndex | None = None, style: str = "brush"):
        self.index = index or SemanticIndex.load()
        self.records = self.index.by_char
        self.real = decomposition_set(self.records)
        self.real |= load_ids_database(IDS_DB)
        self.real_normalized = {self._normalize_ids(sig) for sig in self.real}
        self.valid_pinyin = {
            pinyin_base(str(record.get("pinyin") or ""))
            for record in self.records.values()
            if record.get("pinyin")
        }
        self.valid_pinyin.discard("")
        self.position_counts: dict[str, Counter] = defaultdict(Counter)
        self._build_position_counts()
        self._last_llm_rejected: list[str] = []
        self.style = style

    @staticmethod
    def _normalize_ids(signature: str) -> str:
        return "".join(BASE_FORM.get(char, char) for char in signature)

    def _is_real_signature(self, signature: str) -> bool:
        return (
            signature in self.real
            or self._normalize_ids(signature) in self.real_normalized
        )

    @staticmethod
    def _slot_position(op: str | None, index: int, total: int) -> str:
        if op in {"⿰", "⿲"}:
            return "left" if index == 0 else "right" if index == total - 1 else "middle"
        if op in {"⿱", "⿳"}:
            return "top" if index == 0 else "bottom" if index == total - 1 else "middle"
        if op in {"⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺"}:
            return "outer" if index == 0 else "inner"
        return "other"

    def _build_position_counts(self) -> None:
        def walk(node, op=None, index=0, total=1) -> None:
            if isinstance(node, str):
                if node != "？":
                    position = self._slot_position(op, index, total)
                    self.position_counts[canonical(node)][position] += 1
                return
            children = node[1:]
            for child_index, child in enumerate(children):
                walk(child, node[0], child_index, len(children))

        for record in self.records.values():
            node = parse_ids(record.get("decomposition"))
            if node is not None and not isinstance(node, str):
                walk(node)

    def _dominant_position(self, component: str) -> tuple[str | None, float, int]:
        counts = self.position_counts.get(canonical(component))
        if not counts:
            return None, 0.0, 0
        total = sum(counts.values())
        position, count = counts.most_common(1)[0]
        return position, count / total, total

    def _leaf_position(self, node, target: str) -> str | None:
        def walk(current, op=None, index=0, total=1):
            if isinstance(current, str):
                return self._slot_position(op, index, total) if current == target else None
            children = current[1:]
            for child_index, child in enumerate(children):
                found = walk(child, current[0], child_index, len(children))
                if found:
                    return found
            return None

        return walk(node)

    # ------------------------------------------------------------------ meaning
    def _meaning(self, component: str) -> str:
        record = self.records.get(component) or self.records.get(canonical(component))
        return short_meaning(record) or "component"

    def _component_fits_slot(
        self, component: str, structure: str, position: int
    ) -> bool:
        if structure not in {"⿰", "⿱", "⿷"}:
            return True
        if position != 1:
            return True
        record = self.records.get(component) or self.records.get(canonical(component))
        if record is None:
            return True
        strokes = record.get("stroke_count") or 0
        radical = record.get("radical")
        if radical in ENCLOSURES:
            return False
        limit = 10 if structure == "⿱" else 11
        return strokes <= limit

    def _component_ok(self, component: str) -> bool:
        if not is_cjk(component) or component in STOP_COMPONENTS:
            return False
        if component in FUNCTION_CHARS:
            return False
        return can_render(component, self.style)

    def _is_simple(self, component: str) -> bool:
        if component in SIMPLE_COMPONENTS or component in COMMON_RADICALS:
            return True
        if component in LEFT_RADICALS or component in TOP_RADICALS:
            return True
        record = self.records.get(component) or self.records.get(canonical(component))
        return bool(record and (record.get("stroke_count") or 99) <= 6)

    def _radical_of(self, component: str) -> str | None:
        if component in SIMPLE_COMPONENTS or component in COMMON_RADICALS:
            return None
        record = self.records.get(component) or self.records.get(canonical(component))
        if record is None:
            return None

        options: list[str] = []
        etymology = record.get("etymology") or {}
        semantic = etymology.get("semantic")
        if isinstance(semantic, str) and len(semantic) == 1:
            options.append(semantic)
        radical = record.get("radical")
        if isinstance(radical, str) and len(radical) == 1:
            options.append(radical)
        node = parse_ids(record.get("decomposition"))
        options.extend(top_level_components(node))

        for candidate in options:
            if candidate != component and self._component_ok(candidate) and self._is_simple(candidate):
                return candidate
        return None

    def _can_be_first(self, component: str) -> bool:
        return (
            component in LEFT_RADICALS
            or component in TOP_RADICALS
            or component in ENCLOSURES
            or component in SIMPLE_COMPONENTS
        )

    def _can_be_second(self, component: str) -> bool:
        if component in ENCLOSURES:
            return False
        return component not in TOP_RADICALS or component in SIMPLE_COMPONENTS

    def _pair_options(
        self, first: str, second: str
    ) -> list[tuple[str, tuple[str, str]]]:
        pairs: list[tuple[str, str]] = [(first, second)]
        reduced_first = self._radical_of(first)
        if reduced_first:
            pairs.append((reduced_first, second))
        reduced_second = self._radical_of(second)
        if reduced_second:
            pairs.append((reduced_second, first))

        options: list[tuple[str, tuple[str, str]]] = []
        seen: set[tuple[str, tuple[str, str]]] = set()

        def consider(structure: str, parts: tuple[str, str]) -> None:
            if parts[0] == parts[1]:
                return
            key = (structure, parts)
            if key not in seen:
                seen.add(key)
                options.append(key)

        def add_ordered(a: str, b: str) -> None:
            if not self._can_be_first(a) or not self._can_be_second(b):
                return
            if not self._is_simple(b) and a not in (
                LEFT_RADICALS | TOP_RADICALS | set(ENCLOSURES)
            ):
                return
            if a in ENCLOSURES:
                consider(ENCLOSURES[a], (a, b))
            elif a in TOP_RADICALS:
                consider("⿱", (a, b))
            elif a in LEFT_RADICALS:
                consider("⿰", (a, b))
            else:
                consider(structure_for(a), (a, b))

        for a, b in pairs:
            add_ordered(a, b)
            add_ordered(b, a)
            if self._can_be_first(a) and shape_for(b, "⿱", 1) != b:
                consider("⿱", (a, b))
            if self._can_be_first(b) and shape_for(a, "⿱", 1) != a:
                consider("⿱", (b, a))

        def score(option: tuple[str, tuple[str, str]]) -> int:
            structure, (a, b) = option
            value = 0
            if a in LEFT_RADICALS:
                value += 3
            if a in TOP_RADICALS or a in ENCLOSURES:
                value += 2
            if structure == "⿱" and shape_for(b, structure, 1) == "灬":
                value += 3
            if self._is_simple(a) or self._is_simple(b):
                value += 1
            if not self._is_simple(a) and not self._is_simple(b):
                value -= 5
            return value

        options.sort(key=score, reverse=True)
        return options

    def _semantic_radical(self, component: str) -> str | None:
        record = self.records.get(component) or self.records.get(canonical(component))
        if record is None:
            return None
        etymology = record.get("etymology") or {}
        value = etymology.get("semantic")
        if not (isinstance(value, str) and len(value) == 1):
            value = record.get("radical")
        if isinstance(value, str) and len(value) == 1:
            return canonical(value)
        return None

    def _redundant_radicals(
        self, components
    ) -> tuple[str, str, str] | None:
        canonical_forms = [canonical(component) for component in components]
        if 1 < len(set(canonical_forms)) < len(canonical_forms):
            counts = Counter(canonical_forms)
            repeated = next(form for form, count in counts.items() if count > 1)
            pair = [
                component for component in components
                if canonical(component) == repeated
            ][:2]
            key = self._semantic_radical(pair[0]) or repeated
            return (pair[0], pair[1], key)

        items: list[tuple[str, str]] = []
        for component in components:
            key = self._semantic_radical(component)
            if key:
                items.append((component, key))
        for index, (first, key) in enumerate(items):
            for second, other_key in items[index + 1 :]:
                if key == other_key and first != second:
                    return (first, second, key)
        return None

    def _is_atomic(self, component: str) -> bool:
        if (
            component in SIMPLE_COMPONENTS
            or component in COMMON_RADICALS
            or component in LEFT_RADICALS
            or component in TOP_RADICALS
            or component in ENCLOSURES
        ):
            return True
        record = self.records.get(component) or self.records.get(canonical(component))
        if record is None:
            return True
        strokes = record.get("stroke_count")
        node = parse_ids(record.get("decomposition"))
        compound = node is not None and not isinstance(node, str)
        if strokes is None:
            return not compound
        if compound:
            return strokes <= 9
        return strokes <= 18

    def _is_valid_semantic(self, component: str) -> bool:
        return component in COMPLEX_SEMANTIC_RADICALS or self._is_atomic(component)

    def _component_prior(self, component: str) -> float:
        record = self.records.get(component) or self.records.get(canonical(component))
        if component in COMMON_RADICALS:
            return 1.06
        if record is None:
            return 0.62
        rank = record.get("rank")
        strokes = record.get("stroke_count") or 0
        if not rank:
            prior = 0.62
        else:
            prior = frequency_prior(rank)
            if rank > 6000:
                prior *= 0.9
        if strokes > 15:
            prior *= 0.72
        elif strokes > 12:
            prior *= 0.88
        return prior

    def _record_components(self, record: dict) -> list[tuple[str, float]]:
        components: list[tuple[str, float]] = []
        etymology = record.get("etymology") or {}
        semantic = etymology.get("semantic")
        if isinstance(semantic, str) and len(semantic) == 1:
            components.append((semantic, 2.7))

        radical = record.get("radical")
        if isinstance(radical, str) and len(radical) == 1:
            components.append((radical, 1.45))

        node = parse_ids(record.get("decomposition"))
        for component in top_level_components(node):
            components.append((component, 0.85))
        return components

    def _score_components(
        self, matches: list[Match]
    ) -> list[tuple[str, float, list[str]]]:
        scores: dict[str, float] = defaultdict(float)
        support: dict[str, set[str]] = defaultdict(set)
        evidence: dict[str, list[str]] = defaultdict(list)

        for rank, match in enumerate(matches[:48]):
            record = match.record
            weight = max(match.score, 0.0) / (1.0 + 0.16 * rank)

            for component, multiplier in self._record_components(record):
                if component == record["char"] or not self._component_ok(component):
                    continue
                scores[component] += weight * multiplier * self._component_prior(component)
                support[component].add(record["char"])
                if record["char"] not in evidence[component]:
                    evidence[component].append(record["char"])

        ranked = sorted(
            scores,
            key=lambda component: scores[component]
            * (1.0 + 0.18 * len(support[component])),
            reverse=True,
        )
        return [
            (component, scores[component], evidence[component][:4])
            for component in ranked
            if len(support[component]) >= 2
        ]

    def _merge_hints(
        self, ranked: list[tuple[str, float, list[str]]], hints: list[str]
    ) -> list[tuple[str, float, list[str]]]:
        if not hints:
            return ranked

        scores = {component: score for component, score, _ in ranked}
        evidence = {component: list(items) for component, _, items in ranked}

        for char in hints:
            record = self.records.get(char)
            if record is None or not self._component_ok(char):
                continue
            if (record.get("rank") or 99999) > 5000:
                continue
            if (record.get("stroke_count") or 0) > 12:
                continue

            scores[char] = scores.get(char, 0.0) + 0.9
            evidence.setdefault(char, []).append("dictionary hint")

            for component, multiplier in self._record_components(record):
                if component == char or not self._component_ok(component):
                    continue
                scores[component] = (
                    scores.get(component, 0.0)
                    + 0.45 * multiplier * self._component_prior(component)
                )
                evidence.setdefault(component, []).append(char)

        merged = [
            (component, score, evidence.get(component, [])[:4])
            for component, score in scores.items()
        ]
        merged.sort(
            key=lambda item: item[1] * (1.0 + 0.18 * min(len(item[2]), 4)),
            reverse=True,
        )
        return merged

    # ------------------------------------------------------------------ composing
    def _candidate(
        self,
        structure: str,
        parts: tuple[str, str],
        source: str | None,
        phrase: str,
        allow_complex: str | None = None,
    ) -> Candidate | None:
        if len({canonical(part) for part in parts}) < 2:
            return None
        shapes = tuple(
            shape_for(part, structure, index) for index, part in enumerate(parts)
        )
        if len(set(shapes)) < 2 or any(not self._component_ok(shape) for shape in shapes):
            return None
        for position, part in enumerate(parts):
            if allow_complex and canonical(part) == canonical(allow_complex):
                # A phonetic component carries the reading; in a tall slot it may
                # legitimately be more complex than a semantic radical.
                tall = (structure in {"⿰", "⿷"} and position == 1) or (
                    structure in {"⿴", "⿵", "⿶", "⿸", "⿹", "⿺"} and position == 1
                )
                if tall:
                    continue
            if not self._component_fits_slot(part, structure, position):
                return None
        if self._is_real_signature(structure + "".join(shapes)):
            return None

        first, second = parts
        labels = []
        for part, shape in zip(parts, shapes):
            labels.append(part if part == shape else f"{part} → {shape}")
        explanation = (
            f"{labels[0]} ({self._meaning(first)}) + {labels[1]} ({self._meaning(second)}), "
            f"{describe_layout(structure)} — a forged character for “{phrase}”."
        )
        if source:
            explanation += f" Modeled on {source} ({self._meaning(source)})."
        return Candidate(
            structure=structure,
            parts=parts,
            shapes=shapes,
            explanation=explanation,
            source=source,
        )

    def _partner_order(
        self,
        anchor: str,
        pool: list[str],
        hints: set[str],
        word_for_hint: dict[str, str],
    ) -> list[str]:
        if anchor not in hints:
            return pool

        anchor_word = word_for_hint.get(anchor)

        # Prefer a component carrying a *different* idea from the phrase:
        # fire + never -> 永火, digital + lonely -> 字孤, rain + night -> 雨夜.
        cross = [
            component
            for component in pool
            if component in hints
            and word_for_hint.get(component) != anchor_word
            and canonical(component) != canonical(anchor)
        ]

        by_group = {canonical(component): component for component in pool}
        generic: list[str] = []
        for group in PREFERRED_PARTNERS:
            component = by_group.get(group)
            if (
                component
                and canonical(component) != canonical(anchor)
                and component not in cross
            ):
                generic.append(component)

        ordered = cross + generic
        ordered.extend(
            component
            for component in pool
            if component not in ordered and canonical(component) not in STRUCTURAL
        )
        ordered.extend(
            component
            for component in pool
            if component not in ordered and canonical(component) in STRUCTURAL
        )
        return ordered

    def _build_candidates(
        self,
        ranked: list[tuple[str, float, list[str]]],
        matches: list[Match],
        count: int,
        phrase: str,
        hints: list[str] | None = None,
        hint_groups: dict[str, list[str]] | None = None,
    ) -> tuple[list[Candidate], list[tuple[str, float, list[str]]]]:
        pool: list[str] = []
        groups: set[str] = set()
        for component, _, _ in ranked:
            group = canonical(component)
            if group in groups or not self._component_ok(component):
                continue
            groups.add(group)
            pool.append(component)

        if len(pool) < 2:
            for component in FALLBACK_COMPONENTS:
                if canonical(component) not in groups and self._component_ok(component):
                    pool.append(component)
                    groups.add(canonical(component))

        if len(pool) < 2:
            return [], ranked

        hint_set = set(hints or [])
        word_for_hint: dict[str, str] = {}
        for word, chars in (hint_groups or {}).items():
            for component in chars:
                word_for_hint.setdefault(component, word)
        hint_order: list[str] = []
        group_values = [chars for chars in (hint_groups or {}).values() if chars]
        for pass_index in range(2):
            for chars in group_values:
                if pass_index >= len(chars):
                    continue
                component = chars[pass_index]
                if component in pool and component not in hint_order:
                    hint_order.append(component)
        anchors: list[str] = []
        for component in hint_order + pool:
            if component not in anchors:
                anchors.append(component)
        anchors = anchors[:10]
        candidates: list[Candidate] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        used_pairs: set[frozenset[str]] = set()

        def add(structure: str, parts: tuple[str, str], source: str | None = None) -> bool:
            candidate = self._candidate(structure, parts, source, phrase)
            if candidate is None:
                return False
            key = (candidate.structure or "", candidate.shapes)
            if key in seen:
                return False
            seen.add(key)
            candidates.append(candidate)
            return True

        # Pass one: one new pair per anchor, so every candidate has a different
        # semantic centre rather than the same strongest radical repeated.
        for anchor in anchors:
            for partner in self._partner_order(anchor, pool, hint_set, word_for_hint):
                pair = frozenset((canonical(anchor), canonical(partner)))
                if len(pair) < 2 or pair in used_pairs:
                    continue
                made = False
                for structure, parts in self._pair_options(anchor, partner):
                    if add(structure, parts):
                        made = True
                        break
                if made:
                    used_pairs.add(pair)
                    break
            if len(candidates) >= count + 2:
                break

        # Pass two: alternate silhouettes for the strongest anchors.
        for anchor in anchors:
            for partner in self._partner_order(anchor, pool, hint_set, word_for_hint):
                if canonical(anchor) == canonical(partner):
                    continue
                made = False
                for structure, parts in self._pair_options(anchor, partner):
                    if add(structure, parts):
                        made = True
                        break
                if made:
                    break
            if len(candidates) >= count + 2:
                break

        # Semantic + phonetic, the way most real characters are built.
        best = matches[0].record if matches else None
        if best:
            etymology = best.get("etymology") or {}
            semantic = etymology.get("semantic")
            phonetic = etymology.get("phonetic")
            if (
                isinstance(semantic, str)
                and isinstance(phonetic, str)
                and len(semantic) == 1
                and len(phonetic) == 1
                and self._component_ok(semantic)
                and self._component_ok(phonetic)
                and canonical(semantic) != canonical(phonetic)
                and (self.records.get(phonetic, {}).get("rank") or 99999) <= 5000
                and (self.records.get(phonetic, {}).get("stroke_count") or 0) <= 14
            ):
                before = len(candidates)
                add(structure_for(semantic), (semantic, phonetic), best["char"])
                if len(candidates) > before:
                    candidates.insert(1, candidates.pop())

        # Put the candidates that genuinely combine two ideas from the phrase first.
        def rank(candidate: Candidate) -> tuple[bool, int]:
            words = {
                word_for_hint[part]
                for part in candidate.parts
                if part in word_for_hint
            }
            hinted = sum(1 for part in candidate.parts if part in hint_set)
            return (len(words) >= 2, hinted)

        candidates.sort(key=rank, reverse=True)
        return candidates[:count], ranked

    def _pinyin_of(self, component: str) -> str:
        record = self.records.get(component) or self.records.get(canonical(component))
        if not record:
            return ""
        pinyin = record.get("pinyin") or ""
        for separator in (",", "/"):
            pinyin = pinyin.split(separator)[0]
        return pinyin.strip()

    @staticmethod
    def _one_syllable(reading: str) -> bool:
        if not reading or " " in reading.strip():
            return False
        tones = "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ"
        return sum(1 for char in reading if char in tones) <= 1

    def _reading_ok(self, reading: str) -> bool:
        return bool(reading) and self._one_syllable(reading) and (
            pinyin_base(reading) in self.valid_pinyin
        )

    def _prefer_reading(
        self, candidate: Candidate, analysis: dict | None
    ) -> Candidate:
        if candidate is None:
            return candidate
        if candidate.reading and self._reading_ok(candidate.reading):
            return candidate
        preferred = ""
        if isinstance(analysis, dict):
            preferred = str(analysis.get("reading") or "").strip()
        if not self._reading_ok(preferred):
            sound = candidate.phonetic or (
                candidate.parts[-1] if candidate.parts else ""
            )
            preferred = self._pinyin_of(sound)
        if self._reading_ok(preferred):
            old = candidate.reading
            candidate.reading = preferred
            if old:
                candidate.explanation = candidate.explanation.replace(
                    f"Reading: {old}", f"Reading: {preferred}"
                )
        return candidate

    def _positional_node(self, node, parent_op: str | None = None, index: int = 0, total: int = 1):
        if isinstance(node, str):
            if parent_op in {"⿰", "⿲"}:
                if index == 0:
                    return shape_for(node, "⿰", 0)
                if index == total - 1:
                    return shape_for(node, "⿰", 1)
            elif parent_op in {"⿱", "⿳"}:
                if index == 0:
                    return shape_for(node, "⿱", 0)
                if index == total - 1:
                    return shape_for(node, "⿱", 1)
            return node
        op = node[0]
        children = node[1:]
        return (op, *[
            self._positional_node(child, op, position, len(children))
            for position, child in enumerate(children)
        ])

    def _position_share(self, component: str, position: str) -> float:
        counts = self.position_counts.get(canonical(component))
        if not counts:
            return 0.0
        total = sum(counts.values())
        return counts.get(position, 0) / total if total else 0.0

    def _plan_two_part(
        self, semantic: str, phonetic: str
    ) -> tuple[str, tuple[str, str], tuple[str, str]] | None:
        left_only = set(LEFT_VARIANTS.values())
        top_only = {"宀", "艹", "雨", "穴", "竹", "罒", "冖", "亠", "覀"}
        options: list[tuple[float, str, tuple[str, str]]] = []

        def add(structure: str, parts: tuple[str, str]) -> None:
            if semantic in left_only and not (
                structure == "⿰" and canonical(parts[0]) == canonical(semantic)
            ):
                return
            if semantic in top_only and not (
                structure == "⿱" and canonical(parts[0]) == canonical(semantic)
            ):
                return
            if semantic in ENCLOSURES and not (
                canonical(parts[0]) == canonical(semantic)
                and structure == ENCLOSURES[semantic]
            ):
                return

            positions = {
                "⿰": ("left", "right"),
                "⿱": ("top", "bottom"),
                "⿴": ("outer", "inner"),
                "⿵": ("outer", "inner"),
                "⿶": ("outer", "inner"),
                "⿷": ("outer", "inner"),
                "⿸": ("outer", "inner"),
                "⿹": ("outer", "inner"),
                "⿺": ("outer", "inner"),
            }[structure]
            score = 0.0
            for part, position in zip(parts, positions):
                weight = 2.0 if canonical(part) == canonical(semantic) else 1.0
                share = self._position_share(part, position)
                score += weight * share
                if share == 0:
                    score -= weight * 0.5

            dominant, confidence, _ = self._dominant_position(semantic)
            semantic_position = (
                positions[0]
                if canonical(parts[0]) == canonical(semantic)
                else positions[1]
            )
            if confidence >= 0.5 and semantic_position == dominant:
                score += 0.4
            options.append((score, structure, parts))

        add("⿰", (semantic, phonetic))
        add("⿰", (phonetic, semantic))
        add("⿱", (semantic, phonetic))
        add("⿱", (phonetic, semantic))
        if semantic in ENCLOSURES:
            add(ENCLOSURES[semantic], (semantic, phonetic))
        if phonetic in ENCLOSURES:
            add(ENCLOSURES[phonetic], (phonetic, semantic))

        options.sort(key=lambda item: item[0], reverse=True)
        for score, structure, parts in options:
            if score < 0.8:
                continue
            shapes = tuple(
                shape_for(part, structure, index)
                for index, part in enumerate(parts)
            )
            if len(set(shapes)) < 2 or any(
                not self._component_ok(shape) for shape in shapes
            ):
                continue
            signature = structure + "".join(shapes)
            if self._is_real_signature(signature):
                continue
            return structure, parts, shapes
        return None

    def _is_compound(self, component: str) -> bool:
        record = self.records.get(component) or self.records.get(canonical(component))
        if record is None:
            return False
        node = parse_ids(record.get("decomposition"))
        return node is not None and not isinstance(node, str)

    def _heavy_compound(self, component: str) -> bool:
        return self._is_compound(component) and not self._is_simple(component)

    def _reduce_component(self, component: str) -> str | None:
        record = self.records.get(component) or self.records.get(canonical(component))
        if record is None:
            return None
        etymology = record.get("etymology") or {}
        for candidate in (etymology.get("semantic"), record.get("radical")):
            if (
                isinstance(candidate, str)
                and len(candidate) == 1
                and candidate != component
                and self._component_ok(candidate)
                and self._is_simple(candidate)
            ):
                return candidate
        return None

    def _reduce_ideographic_pair(self, node):
        if not (
            isinstance(node, tuple)
            and node[0] == "⿰"
            and len(node) == 3
        ):
            return [node]
        first, second = node[1], node[2]
        if not (isinstance(first, str) and isinstance(second, str)):
            return [node]
        if not (self._heavy_compound(first) and self._heavy_compound(second)):
            return [node]
        first_reduced = self._reduce_component(first)
        second_reduced = self._reduce_component(second)
        options: list = []
        if first_reduced:
            options.append(("⿰", first_reduced, second))
        if second_reduced:
            options.append(("⿰", first, second_reduced))
        if first_reduced and second_reduced:
            options.append(("⿰", first_reduced, second_reduced))
        return options or [node]

    def _compact_ideographic_options(self, node):
        if isinstance(node, str):
            return [node]
        leaves = all_leaves(node)
        if len(leaves) == 3:
            first, second, third = leaves
            raw = [
                ("⿱", first, ("⿰", second, third)),
                ("⿰", first, ("⿱", second, third)),
                ("⿰", ("⿱", first, second), third),
                ("⿱", ("⿰", first, second), third),
            ]
            options = []
            for option in raw:
                if option not in options:
                    options.append(option)
            return options
        if len(leaves) == 4:
            first, second, third, fourth = leaves
            raw = [
                node,
                ("⿰", ("⿱", first, second), ("⿱", third, fourth)),
                ("⿱", ("⿰", first, second), ("⿰", third, fourth)),
                ("⿰", first, ("⿱", second, ("⿰", third, fourth))),
                ("⿱", first, ("⿰", second, ("⿱", third, fourth))),
            ]
            options = []
            for option in raw:
                if option not in options:
                    options.append(option)
            return options

        op = node[0]
        children = node[1:]
        child_options = [
            self._compact_ideographic_options(child) for child in children
        ]
        results = [(op, *[options[0] for options in child_options])]
        for index, options in enumerate(child_options):
            for alternative in options[1:]:
                combination = [entry[0] for entry in child_options]
                combination[index] = alternative
                results.append((op, *combination))
        return results

    def _stroke_count(self, component: str) -> int:
        record = self.records.get(component) or self.records.get(canonical(component))
        if not record:
            return 1
        try:
            return max(1, int(record.get("stroke_count") or 1))
        except (TypeError, ValueError):
            return 1

    def _layout_penalty(self, signature: str, shapes: tuple[str, ...]) -> float:
        mask, trace = layout_trace(signature, shapes, 384, self.style)
        if not trace:
            return 99.0
        coverage = sum(mask.histogram()[1:]) / max(1, mask.width * mask.height)
        areas = [max(1, width * height) for _, width, height, _, _ in trace]
        strokes = [self._stroke_count(shape) for shape, _, _, _, _ in trace]
        total_area = sum(areas) or 1
        total_strokes = sum(strokes) or 1
        area_shares = [area / total_area for area in areas]
        stroke_shares = [stroke / total_strokes for stroke in strokes]
        balance = 0.5 * sum(
            abs(a - s) for a, s in zip(area_shares, stroke_shares)
        )
        anchor_penalty = max(0.0, stroke_shares[0] - area_shares[0])
        stretches: list[float] = []
        for _, width, height, natural_w, natural_h in trace:
            if not (width and height and natural_w and natural_h):
                continue
            slot = width / height
            natural = natural_w / natural_h
            stretches.append(max(slot / natural, natural / slot))
        stretch_penalty = (
            sum(max(0.0, value - 2.0) for value in stretches)
            / max(1, len(stretches))
        )
        depth = sum(signature.count(operator) for operator in ALL_OPS)
        return (
            2.2 * balance
            + 1.4 * abs(coverage - 0.17)
            + 0.25 * stretch_penalty
            + 0.20 * depth
            + 1.5 * anchor_penalty
        )

    def _repetition_component(self, analysis: dict | None) -> str | None:
        if not isinstance(analysis, dict):
            return None
        components = [
            char for char in analysis.get("component_candidates", [])
            if isinstance(char, str)
            and len(char) == 1
            and is_cjk(char)
            and self._component_ok(char)
        ]
        literal = str(analysis.get("literal_meaning") or "")
        match = re.search(r"\b([A-Za-z]+)\s+of\s+\1s?\b", literal, re.I)
        if match:
            stem = match.group(1).lower().rstrip("s")
            for char in components:
                if stem and stem in self._meaning(char).lower():
                    return char
        for term in analysis.get("chinese_terms") or []:
            text = str(term)
            for char in components:
                if text.count(char) >= 2 and self._is_simple(char):
                    return char
        return None

    def _repetition_candidate(
        self, analysis: dict | None, phrase: str
    ) -> Candidate | None:
        component = self._repetition_component(analysis)
        if component is None:
            return None
        ids = to_ids(("⿱", component, ("⿰", component, component)))
        literal = str((analysis or {}).get("literal_meaning") or "").strip()
        gloss = literal or str((analysis or {}).get("definition") or "").strip()
        design = {
            "type": "ideographic",
            "components": [component, component, component],
            "structure": ids,
            "semantic": [component, component, component],
            "phonetic": "",
            "reading": self._pinyin_of(component),
            "gloss": gloss,
            "rationale": "Three repeated components: the meaning repeated and stacked.",
        }
        return self._ideographic_candidate(design, phrase)

    def _ideographic_candidate(self, design: dict, phrase: str) -> Candidate | None:
        components = design.get("components")
        if not isinstance(components, list) or not (2 <= len(components) <= 4):
            return None
        if not all(
            isinstance(component, str)
            and len(component) == 1
            and is_cjk(component)
            for component in components
        ):
            return None
        if any(
            not self._is_valid_semantic(component)
            or not self._component_ok(component)
            for component in components
        ):
            return None

        redundant = self._redundant_radicals(components)
        if redundant:
            self._last_llm_rejected.append("redundant:" + "+".join(redundant))
            return None
        if len(components) >= 3:
            heavy = [
                component for component in components
                if self._is_compound(component)
                and not self._is_simple(component)
                and component not in COMPLEX_SEMANTIC_RADICALS
            ]
            if heavy:
                self._last_llm_rejected.append("heavy3:" + "+".join(heavy))
                return None

        raw = str(design.get("structure") or "").strip().replace(" ", "")
        if not raw or raw[0] not in ALL_OPS:
            return None
        ids = raw.split("[", 1)[0]
        node = parse_ids(ids)
        if node is None or isinstance(node, str):
            return None
        leaves = all_leaves(node)
        if not (2 <= len(leaves) <= 4):
            return None
        if Counter(leaves) != Counter(components):
            return None

        layout_options: list = []
        for option in self._compact_ideographic_options(node):
            layout_options.extend(self._reduce_ideographic_pair(option))
        deduped: list = []
        for option in layout_options:
            if option not in deduped:
                deduped.append(option)

        rank_templates = len(components) >= 3
        best: Candidate | None = None
        best_penalty = float("inf")
        for option in deduped:
            option_leaves = all_leaves(option)
            reduced_pair = (
                len(components) == 2
                and len(option_leaves) == 2
                and option_leaves[0] != components[0]
                and option_leaves[1] == components[1]
            )
            positional = self._positional_node(option)
            shapes = tuple(all_leaves(positional))
            signature = to_ids(positional)
            if self._is_real_signature(signature):
                continue
            if any(not self._component_ok(shape) for shape in shapes):
                continue

            reading = str(design.get("reading") or "").strip()[:80]
            if not self._one_syllable(reading):
                sound = components[-1]
                pinyin = self._pinyin_of(sound)
                base = pinyin_base(pinyin)
                if base:
                    rng = random.Random(
                        seed_for(f"{phrase}|ideo|{''.join(components)}|{signature}")
                    )
                    reading = apply_tone(base, rng.randint(1, 4))
            if not self._one_syllable(reading):
                reading = ""

            label_sources = (
                tuple(components) if reduced_pair else tuple(option_leaves)
            )
            labels = " + ".join(
                f"{leaf} ({self._meaning(leaf)})"
                if leaf == shape
                else f"{leaf} → {shape} ({self._meaning(leaf)})"
                for leaf, shape in zip(label_sources, shapes)
            )
            explanation = (
                f"Ideographic: {labels} — a forged character for “{phrase}”."
            )
            if reading:
                explanation += f" Reading: {reading}."
            gloss = sanitize_layout_text(str(design.get("gloss") or "")).rstrip(".")[:200]
            rationale = sanitize_layout_text(str(design.get("rationale") or ""))[:300]
            if gloss:
                explanation += f" Meaning: {gloss}."

            candidate = Candidate(
                structure=signature,
                parts=tuple(option_leaves),
                shapes=shapes,
                explanation=explanation,
                origin="llm",
                reading=reading,
                gloss=gloss,
                rationale=rationale,
                semantic="+".join(components),
                phonetic="",
            )
            if not rank_templates:
                return candidate
            penalty = self._layout_penalty(signature, shapes)
            if penalty < best_penalty:
                best = candidate
                best_penalty = penalty
        return best

    def _llm_candidates(
        self,
        phrase: str,
        count: int = 6,
        variation: int = 0,
        feedback: str = "",
        analysis: dict | None = None,
    ) -> list[Candidate]:
        designs = llm.design(
            phrase,
            count=count,
            variation=variation,
            feedback=feedback,
            analysis=analysis,
        )
        candidates: list[Candidate] = []
        self._last_llm_rejected = []

        for design in designs:
            design_type = str(design.get("type") or "phono_semantic").strip().lower()
            components = design.get("components")

            if design_type == "ideographic":
                candidate = self._ideographic_candidate(design, phrase)
                if candidate is None:
                    label = "".join(components) if isinstance(components, list) else "ideo"
                    self._last_llm_rejected.append(f"ideo:{label}")
                else:
                    candidates.append(self._prefer_reading(candidate, analysis))
                continue

            if not isinstance(components, list) or len(components) != 2:
                continue
            first, second = components
            if not (
                isinstance(first, str)
                and isinstance(second, str)
                and is_cjk(first)
                and is_cjk(second)
                and canonical(first) != canonical(second)
            ):
                continue
            semantic_value = design.get("semantic")
            if isinstance(semantic_value, list):
                semantic_value = semantic_value[0] if len(semantic_value) == 1 else None
            phonetic_value = design.get("phonetic")
            pair_label = f"{first}+{second}"
            if not (
                isinstance(semantic_value, str)
                and isinstance(phonetic_value, str)
                and semantic_value in (first, second)
                and phonetic_value in (first, second)
                and canonical(semantic_value) != canonical(phonetic_value)
                and self._is_valid_semantic(semantic_value)
                and self._is_atomic(phonetic_value)
            ):
                self._last_llm_rejected.append(pair_label)
                continue
            redundant = self._redundant_radicals((semantic_value, phonetic_value))
            if redundant:
                self._last_llm_rejected.append(
                    "redundant:" + "+".join(redundant)
                )
                continue

            planned = self._plan_two_part(semantic_value, phonetic_value)
            if planned is None:
                self._last_llm_rejected.append(pair_label)
                continue
            structure, parts, shapes = planned
            signature = structure + "".join(shapes)

            left_scale = ""
            if (
                structure == "⿰"
                and parts[0] == semantic_value
                and semantic_value not in LEFT_WIDTH_RATIOS
            ):
                hint = str(design.get("left_scale") or "").lower()
                if hint in {"narrow", "medium", "full"}:
                    left_scale = hint

            reading = str(design.get("reading") or "").strip()[:80]
            if not self._one_syllable(reading):
                reading = self._pinyin_of(phonetic_value) if phonetic_value else ""
            if not self._one_syllable(reading):
                reading = ""

            labels = " + ".join(
                f"{part} ({self._meaning(part)})"
                if part == shape
                else f"{part} → {shape} ({self._meaning(part)})"
                for part, shape in zip(parts, shapes)
            )
            explanation = (
                f"{labels}, {describe_layout(structure)} — a forged character "
                f"for “{phrase}”."
            )
            explanation += f" Semantic: {semantic_value}."
            explanation += f" Phonetic: {phonetic_value}."
            if reading:
                explanation += (
                    f" Reading: {reading} "
                    f"(phonetic {phonetic_value} is a sound hint)."
                )
            gloss = sanitize_layout_text(str(design.get("gloss") or "")).rstrip(".")[:200]
            rationale = sanitize_layout_text(str(design.get("rationale") or ""))[:300]
            if gloss:
                explanation += f" Meaning: {gloss}."

            candidate = Candidate(
                structure=signature,
                parts=tuple(parts),
                shapes=tuple(shapes),
                explanation=explanation,
                origin="llm",
                reading=reading,
                gloss=gloss,
                rationale=rationale,
                semantic=semantic_value,
                phonetic=phonetic_value,
                left_scale=left_scale,
            )
            candidates.append(self._prefer_reading(candidate, analysis))

        return candidates

    # ------------------------------------------------------------------ public
    def forge(
        self,
        phrase: str,
        count: int = 4,
        mode: str = "new",
        use_llm: bool | None = None,
        llm_variation: int = 0,
        seal: bool = False,
    ) -> ForgeResult:
        phrase = " ".join(phrase.split())
        if not phrase:
            raise ValueError("Please enter a phrase.")

        raw_matches = self.index.search(phrase, k=90)
        matches = [
            match for match in raw_matches if match.char not in FUNCTION_CHARS
        ]
        if not matches:
            matches = raw_matches

        warning = ""
        has_cjk = any(is_cjk(char) for char in phrase)
        if has_cjk:
            warning = ""
        elif not self.index.content_words(phrase):
            warning = (
                "This phrase is mostly function words, so the glyph is decorative "
                "rather than a translation."
                if any(char.isalpha() for char in phrase)
                else "No dictionary content words were found; the glyph is a "
                "decorative interpretation rather than a translation."
            )
        elif matches[0].score < 0.42:
            warning = (
                "No close dictionary match; the result is an imaginative "
                "interpretation rather than a confident translation."
            )

        if mode == "real":
            candidates: list[Candidate] = []
            for match in [item for item in matches if can_render(item.char, self.style)][:count]:
                mask = character_mask(match.char, 1024, self.style)
                candidate = Candidate(
                    structure=None,
                    parts=(match.char,),
                    shapes=(match.char,),
                    explanation=f"{match.char} — {match.record['definition']}",
                    source=match.char,
                )
                candidate.image = finish(
                    mask,
                    phrase,
                    seal_char=match.char,
                    seed=seed_for(f"{phrase}|{match.char}"),
                    reading=self._pinyin_of(match.char),
                    gloss=match.record.get("definition", ""),
                    show_seal=seal,
                )
                candidates.append(candidate)
            return ForgeResult(
                phrase=phrase, mode="real", candidates=candidates, matches=matches[:8]
            )

        llm_requested = llm.is_available() if use_llm is None else use_llm
        analysis = (
            llm.analyze(phrase)
            if mode == "new" and llm_requested
            else None
        )
        repeat_candidate = self._repetition_candidate(analysis, phrase)

        ranked = self._score_components(matches)
        hint_groups = self.index.hint_groups(phrase)
        hints = list(dict.fromkeys(char for chars in hint_groups.values() for char in chars))
        if analysis:
            extra = [
                char
                for char in analysis.get("component_candidates", [])
                if isinstance(char, str) and char in self.records
            ]
            if extra:
                hints = list(dict.fromkeys(hints + extra))
                ranked = self._merge_hints(ranked, extra)
        ranked = self._merge_hints(ranked, hints)
        candidates, ranked = self._build_candidates(
            ranked, matches, count, phrase, hints=hints, hint_groups=hint_groups
        )

        llm_used = False
        if mode == "new" and llm_requested:
            creative = self._llm_candidates(
                phrase,
                count=6,
                variation=llm_variation,
                analysis=analysis,
            )
            if repeat_candidate is not None:
                creative.insert(0, repeat_candidate)
            rejected = list(dict.fromkeys(self._last_llm_rejected))
            attempts = 0
            while len(creative) < count and attempts < 2:
                feedback = ", ".join(rejected[:12])
                more = self._llm_candidates(
                    phrase,
                    count=6,
                    variation=llm_variation + 1 + attempts,
                    feedback=feedback,
                    analysis=analysis,
                )
                rejected.extend(self._last_llm_rejected)
                rejected = list(dict.fromkeys(rejected))
                seen = {(c.structure or "", c.shapes) for c in creative}
                for candidate in more:
                    key = (candidate.structure or "", candidate.shapes)
                    if key not in seen:
                        seen.add(key)
                        creative.append(candidate)
                attempts += 1
            if analysis and len(creative) < count:
                components = [
                    char for char in analysis.get("component_candidates", [])
                    if isinstance(char, str) and char in self.records
                ]
                if len(components) >= 2:
                    first, second = components[0], components[1]
                    design = {
                        "type": "ideographic",
                        "components": [first, second],
                        "structure": "⿰" + first + second,
                        "semantic": [first, second],
                        "phonetic": "",
                        "reading": str(analysis.get("reading") or ""),
                        "gloss": str(analysis.get("definition") or ""),
                        "rationale": "A two-part design anchored on the concept analysis.",
                    }
                    anchored = self._ideographic_candidate(design, phrase)
                    if anchored is not None:
                        creative.insert(0, anchored)

            if creative:
                llm_used = True
                merged: list[Candidate] = []
                seen: set[tuple[str, tuple[str, ...]]] = set()
                for candidate in creative + candidates:
                    key = (candidate.structure or "", candidate.shapes)
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(candidate)
                candidates = merged[:count]
            else:
                detail = llm.last_error()
                note = "LLM creative director failed"
                if detail:
                    note += f": {detail}"
                note += ". Using deterministic designs."
                warning = f"{warning} {note}".strip()

        seal_char = matches[0].char
        if not candidates and matches:
            best = matches[0]
            candidate = Candidate(
                structure=None,
                parts=(best.char,),
                shapes=(best.char,),
                explanation=f"{best.char} — {best.record['definition']}",
                source=best.char,
            )
            candidate.image = finish(
                character_mask(best.char, 1024, self.style),
                phrase,
                seal_char=best.char,
                seed=seed_for(f"{phrase}|fallback"),
                reading=self._pinyin_of(best.char),
                gloss=best.record.get("definition", ""),
                show_seal=seal,
            )
            candidates = [candidate]

        fallback_gloss = phrase
        if analysis:
            definition = " ".join(str(analysis.get("definition") or "").split())
            if definition:
                fallback_gloss = definition[:140]
        for candidate in candidates:
            if not candidate.gloss:
                candidate.gloss = fallback_gloss

        for index, candidate in enumerate(candidates):
            if candidate.origin == "deterministic" and not candidate.reading:
                sound = (
                    candidate.parts[1]
                    if len(candidate.parts) > 1
                    else (candidate.parts[0] if candidate.parts else "")
                )
                pinyin = self._pinyin_of(sound)
                if not pinyin and candidate.parts:
                    sound = candidate.parts[0]
                    pinyin = self._pinyin_of(sound)
                base = pinyin_base(pinyin)
                if base:
                    rng = random.Random(
                        seed_for(
                            f"{phrase}|det|{index}|{''.join(candidate.parts)}"
                        )
                    )
                    tone = rng.randint(1, 4)
                    reading = apply_tone(base, tone)
                    candidate.reading = reading
                    candidate.phonetic = sound
                    candidate.explanation += (
                        f" Sound hint: {sound} ({pinyin}); reading: {reading}."
                    )

        for index, candidate in enumerate(candidates):
            if candidate.image is not None:
                continue
            if candidate.origin == "llm" or (
                candidate.structure and candidate.structure not in LAYOUTS
            ):
                mask = compose_ids(
                    candidate.structure or "",
                    candidate.shapes,
                    1024,
                    self.style,
                    left_scale=candidate.left_scale or None,
                )
            else:
                mask = compose_mask(
                    candidate.structure or "⿰", candidate.shapes, 1024, self.style
                )
            candidate.image = finish(
                mask,
                phrase,
                seal_char=seal_char,
                seed=seed_for(f"{phrase}|{index}|{candidate.structure}|{''.join(candidate.shapes)}"),
                reading=candidate.reading,
                gloss=candidate.gloss,
                show_seal=seal,
            )

        return ForgeResult(
            phrase=phrase,
            mode="new",
            candidates=candidates,
            matches=matches[:8],
            components=ranked[:8],
            warning=warning,
            llm_used=llm_used,
            llm_rejected=len(rejected) if mode == "new" and llm_requested else 0,
            analysis=analysis,
        )
