"""Load character meanings from Make Me a Hanzi and CC-CEDICT."""

from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path

CJK_FIRST = 0x4E00
CJK_LAST = 0x9FFF

_CEDICT_LINE = re.compile(r"^(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/(.*)/$")
_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-z]+")

_HINT_STOPWORDS = set(
    """
    a an the and or but if then than so such very really just about above after
    again against all am any are as at be because been before being below between
    both by can could did do does doing down during each few for from further had
    has have having he her here hers herself him himself his how i if in into is
    it its itself me more most my myself no nor not of off on once only or other
    our ours ourselves out over own same she should so some such than that the
    their theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why will
    with would you your yours yourself yourselves
    """.split()
)


def is_cjk(char: str) -> bool:
    return len(char) == 1 and CJK_FIRST <= ord(char) <= CJK_LAST


def clean(text: str | None) -> str:
    return _WHITESPACE.sub(" ", text or "").strip(" ;,.")


def load_makemeahanzi(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/fetch_assets.py first")

    records: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            char = item.get("character") or ""
            if not is_cjk(char):
                continue
            definition = clean(item.get("definition"))
            if not definition:
                continue

            etymology = item.get("etymology") or {}
            records[char] = {
                "char": char,
                "definition": definition,
                "pinyin": item.get("pinyin") or [],
                "radical": item.get("radical") or "",
                "decomposition": item.get("decomposition") or "？",
                "etymology": {
                    key: value
                    for key, value in etymology.items()
                    if key in {"type", "semantic", "phonetic", "hint"}
                },
                "cedict": [],
                "cedict_pinyin": "",
            }
    return records


def add_cedict(path: Path, records: dict[str, dict]) -> None:
    if not path.exists():
        print("CC-CEDICT not found; definitions will come from Make Me a Hanzi only.")
        return

    matched = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            match = _CEDICT_LINE.match(line.strip())
            if not match:
                continue
            traditional, simplified, pinyin, raw = match.groups()
            senses = [clean(part) for part in raw.split("/")]
            senses = [
                sense
                for sense in senses
                if sense and not sense.lower().startswith("variant of")
            ]
            if not senses:
                continue

            for char in {traditional, simplified}:
                record = records.get(char)
                if record is None:
                    continue
                matched += 1
                if not record["cedict_pinyin"]:
                    record["cedict_pinyin"] = pinyin
                for sense in senses:
                    if sense not in record["cedict"] and len(record["cedict"]) < 6:
                        record["cedict"].append(sense)

    print(f"CC-CEDICT enriched {len(records):,} characters ({matched:,} dictionary hits).")


def add_frequency(path: Path, records: dict[str, dict]) -> None:
    if not path.exists():
        print("hanziDB not found; frequency priors will be disabled.")
        return

    matched = 0
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            char = row.get("charcter") or row.get("character") or ""
            record = records.get(char)
            if record is None:
                continue
            try:
                record["rank"] = int(row.get("frequency_rank") or 0)
            except ValueError:
                continue
            try:
                record["stroke_count"] = int(row.get("stroke_count") or 0)
            except ValueError:
                record["stroke_count"] = 0
            record["hsk"] = row.get("hsk_level") or ""
            matched += 1
    print(f"Frequency data added for {matched:,} characters.")


def finalize(records: dict[str, dict]) -> dict[str, dict]:
    for record in records.values():
        cedict = "; ".join(record.get("cedict") or [])
        gloss = record["definition"]
        if cedict:
            gloss = f"{gloss}; {cedict}"
        record["gloss"] = gloss[:400]
        record["pinyin"] = ", ".join(record.get("pinyin") or []) or record.get(
            "cedict_pinyin", ""
        )
        record.pop("cedict_pinyin", None)
    return dict(sorted(records.items()))


def load_keyword_hints(
    path: Path, max_chars: int = 16, min_count: int = 2
) -> dict[str, list[str]]:
    """Map English dictionary keywords to the characters of matching words.

    CC-CEDICT contains multi-character words, so a keyword like "digital" points
    at 数字 / 数码 and therefore at 数, 字, 码.  This gives modern vocabulary a
    lexical bridge that the sentence embedding alone often misses.
    """
    if not path.exists():
        return {}

    from collections import Counter, defaultdict

    counts: dict[str, Counter] = defaultdict(Counter)
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            match = _CEDICT_LINE.match(line.strip())
            if not match:
                continue
            _, simplified, _, raw = match.groups()
            chars = [char for char in simplified if is_cjk(char)]
            if not chars:
                continue
            tokens: set[str] = set()
            for sense in raw.split("/"):
                for token in _WORD.findall(sense.lower()):
                    if len(token) >= 3 and token not in _HINT_STOPWORDS:
                        tokens.add(token)
            for token in tokens:
                for position, char in enumerate(dict.fromkeys(chars)):
                    counts[token][char] += 3 if position == 0 else 1

    hints: dict[str, list[str]] = {}
    for keyword, counter in counts.items():
        selected = [char for char, count in counter.most_common(max_chars) if count >= min_count]
        if selected:
            hints[keyword] = selected
    return hints


def load_records(dictionary: Path, cedict: Path, hanzidb: Path | None = None) -> dict[str, dict]:
    records = load_makemeahanzi(dictionary)
    add_cedict(cedict, records)
    if hanzidb is not None:
        add_frequency(hanzidb, records)
    return finalize(records)


def short_meaning(record: dict | None) -> str:
    if not record:
        return ""
    definition = record.get("definition") or ""
    return definition.split(";")[0].strip()
