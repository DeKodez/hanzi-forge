"""Optional LLM creative director for character designs.

The LLM decides *what* to combine and why; the deterministic composer decides
*whether* the result is renderable and genuinely new.  Any OpenAI-compatible
chat completion endpoint works; the defaults target DeepSeek.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .paths import DATA, ROOT

CACHE_PATH = DATA / "llm_cache.json"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT = 120.0
DEFAULT_REASONING = "none"
LAST_ERROR = ""

SYSTEM_PROMPT = """You are an avant-garde Chinese character designer. Invent ONE new
hanzi for an English phrase, using either a phono-semantic compound (形声字) or an
ideographic compound (会意字).

Before designing, interpret the phrase. If it is a title, name, or loanword, translate
its meaning: shahanshah = king of kings. If the meaning naturally repeats or stacks a
component (king of kings -> 王+王+王; a forest -> 木+木+木; gold -> 金+金+金), use an
ideographic compound. When the analyst's literal meaning is "X of X", include a
threefold repeated-component design before the phono-semantic alternatives.

Two design types:
1. "phono_semantic": exactly two components, a semantic radical plus a phonetic sound
   hint. The new character is read with ONE syllable inspired by the phonetic; the
   reading may drift from the phonetic's dictionary pinyin.
2. "ideographic": two to four semantic components, no phonetic. Use a full IDS structure
   such as ⿱王⿰王王 (a king above two kings). The reading is one invented syllable, or a
   one-syllable reading taken from one of the components.

If an analyst output is provided, its transliteration and component candidates are
authoritative. For a foreign name or tradition, anchor the glyph with the standard
Chinese transliteration (Vishnu -> 毗) plus a domain radical (示/心), not generic
king/field/cosmos imagery.

If the phrase names a concrete thing or creature (dragon, fish, bird, horse, fire,
rain...), the semantic radical must encode that core: a dragon phrase needs 龙/龍, a fish
needs 鱼/魚, a bird needs 鸟/鳥. Complex but indivisible radicals are allowed: 鹿 黑 鼓
鼠 鼻 麻 龙/龍 鱼/魚 鸟/鳥 马/馬 龟/龜. If the phrase is abstract (loneliness,
schadenfreude, vertigo...), choose an unexpected semantic angle; do not force a literal
translation.

Simple phonetics are atomic characters of roughly 4–9 strokes: 忄 氵 宀 目 足 门 疒 耳
心 手 火 水 木 口 女 子 山 土 日 月 田 石 金 言 糸 犬 雨 云 风 电 光 令 灵 双 呙 弟 相
青 包 丁 可 生 尧 乔 召 肖 少 方 仑. If an obvious phonetic would make an existing
character, use a less common atomic phonetic (夬, 睘, 翟, 彗, 尞, 翏, 睪) or a different
semantic angle. Do NOT use compound characters as components: 铃 涡 霜 梯 窗 潮 情 想
意 影 灯 钟 枕 尘 雾.

The rendering engine chooses the conventional placement; the "structure" field is a
suggestion and may be overridden. Do not describe spatial placement in the gloss or
rationale.

Format examples only. Do NOT reuse their content; invent fresh content for the user's
phrase.
Phono-semantic:
- phrase: "a shelter for a lost traveler", type "phono_semantic",
  components ["宀","灵"], forms ["auto","auto"], structure "⿱宀灵",
  semantic "宀", phonetic "灵", reading "líng",
  gloss "a roof over a spirit: shelter for someone still searching."
Ideographic:
- phrase: "king of kings", type "ideographic",
  components ["王","王","王"], forms ["王","王","王"],
  structure "⿱王⿰王王", semantic ["王","王","王"], phonetic "", reading "wáng",
  gloss "king of kings; sovereignty multiplied",
  rationale "Three kings: rule stacked upon rule."

Return JSON only:
{
  "designs": [
    {
      "type": "phono_semantic",
      "components": ["宀", "灵"],
      "forms": ["auto", "auto"],
      "structure": "⿱宀灵",
      "semantic": "宀",
      "phonetic": "灵",
      "reading": "líng",
      "left_scale": "narrow",
      "gloss": "a roof over a spirit: shelter for someone still searching",
      "rationale": "The roof frames a small interior; 灵 lends the sound líng."
    },
    {
      "type": "ideographic",
      "components": ["王", "王", "王"],
      "forms": ["王", "王", "王"],
      "structure": "⿱王⿰王王",
      "semantic": ["王", "王", "王"],
      "phonetic": "",
      "reading": "wáng",
      "gloss": "king of kings; sovereignty multiplied",
      "rationale": "Three kings: rule stacked upon rule."
    }
  ]
}

Rules:
- "type" is "phono_semantic" or "ideographic".
- "components": exactly two for phono_semantic; two to four for ideographic. Each is one
  CJK character; phono phonetics are simple atomic characters; ideographic components
  may be complex indivisible radicals.
- "forms" is optional and has one entry per component: a single CJK character to
  preserve that form, or "auto" to let the engine apply the conventional positional
  variant. Use full forms for ideographic components built from complete characters;
  use "auto" only for conventional radicals such as 水→氵, 心→忄, and 手→扌.
- "structure" is a full IDS string using the components exactly once, with operators
  ⿰ ⿱ ⿲ ⿳ ⿴ ⿵ ⿶ ⿷ ⿸ ⿹ ⿺. For three components use a nested 2D arrangement
  (⿱A⿰BC or ⿰A⿱BC), not a 1×3 ⿲ row of complex characters.
- Do not combine two full compound characters. Pair one semantic radical with one
  component; for example "lion city" should be 犭 + 城 or 土 + 狮, not 狮 + 城. The
  engine may reduce the first compound to its radical if needed.
- Never repeat a radical. Do not pair a component with a full compound that already
  contains its radical: 示 + 祆 (both altar), 心 + 念/忆, 氵 + 河, 亻 + 你. The
  engine rejects these as tautological.
- For three or four ideographic components, use only simple components (4–6 strokes
  or common radicals). A full compound character such as 祆, 念, 命, 神, 夜, 城, 学,
  空, 音 in a 3+ part design is crowded and is rejected. Prefer two parts.
- "semantic" is one component (phono) or a list (ideographic); "phonetic" is one
  component for phono and "" for ideographic.
- "reading" is exactly ONE pinyin syllable (tone mark optional). It may be invented and
  need not match the phonetic exactly. Never write two syllables.
- "left_scale" is optional for a phono semantic radical placed on the left: "narrow",
  "medium", or "full".
- "gloss" is one short, evocative English definition. Do not describe spatial placement.
- "rationale" is one sentence linking the image to the phrase. Do not describe spatial
  placement (left/right/top/bottom/above/below/enclosing).
- Do not use grammatical function characters (的, 是, 不, 了, 和, 我, 你, ...).
- Do not propose an existing character.
- Every design must be about the user's phrase.
- Give 6 designs. For a title, name, loanword, or a meaning that repeats/stacks a
  component, include at least one ideographic design; otherwise use mostly
  phono-semantic designs.
"""


_ENV_LOADED = False


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env without overriding real environment vars."""
    global _ENV_LOADED
    if path is None and _ENV_LOADED:
        return

    env_path = path or (ROOT / ".env")
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            key, separator, value = line.partition("=")
            if not separator:
                continue
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] in "\"'\"" and value[-1] == value[0]:
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)

    if path is None:
        _ENV_LOADED = True


def _config() -> tuple[str, str, str]:
    load_dotenv()
    key = os.environ.get("FORGE_LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    base = os.environ.get("FORGE_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("FORGE_LLM_MODEL", DEFAULT_MODEL)
    return key, base, model


PROMPT_VERSION = "phono-v10-forms1"
ANALYSIS_VERSION = "analysis-v1"

ANALYSIS_PROMPT = """You are a bilingual concept analyst for a Chinese character design
tool. Given an English term or phrase, produce a factual, compact analysis.

Return JSON only:
{
  "definition": "one or two plain-English sentences",
  "literal_meaning": "short literal gloss, or empty string",
  "key_concepts": ["3-6 short English concepts"],
  "chinese_terms": ["1-4 standard Chinese words or transliterations"],
  "transliteration": "standard Chinese transliteration for a foreign term, else empty",
  "component_candidates": ["2-5 single Chinese characters"],
  "reading": "one pinyin syllable",
  "confidence": "high | medium | low"
}

Rules:
- For a name, tradition, or loanword, use the conventional Chinese form. Example:
  Vishnu -> 毗湿奴; Vaishnavism -> 毗湿奴派 (devotion to Vishnu).
- If the literal meaning repeats or stacks a concept (king of kings, lord of lords,
  a forest), include the repeated character in component_candidates and list the
  repeated Chinese term (for example 王 and 万王之王).
- component_candidates must be single CJK characters usable as glyph components:
  prefer simple semantic radicals (示, 心, 水, 火, 人, 口, 教 is too heavy);
  include at most one 4-9 stroke transliteration anchor (e.g. 毗, 奴, 信, 梵).
  Avoid compound components over 9 strokes (虔, 湿, 教, 派).
- reading is exactly one pinyin syllable, normally the first syllable of the
  transliteration (毗湿奴 -> pí), otherwise a component syllable.
- confidence is low when the term is unfamiliar or the Chinese mapping is uncertain.
- Do not design a glyph, choose a structure, or invent etymology.
- Do not invent facts; if the term is obscure, give the standard dictionary sense.
"""


def _single_cjk(value) -> bool:
    if not isinstance(value, str):
        return False
    value = value.strip()
    if len(value) != 1:
        return False
    code = ord(value)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def _normalize_analysis(data) -> dict | None:
    if isinstance(data, dict) and isinstance(data.get("analysis"), dict):
        data = data["analysis"]
    if not isinstance(data, dict):
        return None

    def text(value, limit=240) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.split())[:limit]

    def string_list(value, limit=8) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            item = text(item, 120)
            if item and item not in result:
                result.append(item)
        return result[:limit]

    definition = text(data.get("definition"))
    literal = text(data.get("literal_meaning"))
    concepts = string_list(data.get("key_concepts"), 6)
    terms = [
        item for item in string_list(data.get("chinese_terms"), 4)
        if any(_single_cjk(char) for char in item)
    ]
    components: list[str] = []
    raw_components = data.get("component_candidates")
    if isinstance(raw_components, list):
        for item in raw_components:
            if _single_cjk(item) and item.strip() not in components:
                components.append(item.strip())
    components = components[:5]
    transliteration = text(data.get("transliteration"), 40)
    reading = text(data.get("reading"), 20)
    confidence = text(data.get("confidence"), 10).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    if not (definition or literal or concepts or components):
        return None
    return {
        "definition": definition,
        "literal_meaning": literal,
        "key_concepts": concepts,
        "chinese_terms": terms,
        "transliteration": transliteration,
        "component_candidates": components,
        "reading": reading,
        "confidence": confidence,
    }


def analyze(phrase: str) -> dict | None:
    """Ground a term: definition, Chinese equivalents, and radical candidates."""
    key, base, model = _config()
    if not key:
        return None

    cache = _load_cache()
    cache_key = f"{ANALYSIS_VERSION}::{model}::{phrase.strip().lower()}"
    if cache_key in cache:
        cached = cache[cache_key]
        return cached if isinstance(cached, dict) else None

    max_tokens, timeout = _limits()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": ANALYSIS_PROMPT},
            {"role": "user", "content": f'English term or phrase: "{phrase}"'},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": min(max_tokens, 1400),
    }
    reasoning = _reasoning_mode()
    if reasoning and reasoning not in {"none", "off", "false", "0"}:
        payload["reasoning_effort"] = reasoning

    for attempt in range(2):
        try:
            import httpx

            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{base}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                )
                if (
                    response.status_code == 400
                    and "reasoning_effort" in payload
                    and "reasoning_effort" in response.text
                ):
                    payload.pop("reasoning_effort", None)
                    response = client.post(
                        f"{base}/chat/completions",
                        headers={"Authorization": f"Bearer {key}"},
                        json=payload,
                    )
                if response.status_code >= 400:
                    print(
                        "LLM concept analyst unavailable: "
                        f"HTTP {response.status_code}"
                    )
                    return None
                body = response.json()
            content = body["choices"][0]["message"].get("content") or ""
            data = _parse_json(content)
            analysis = _normalize_analysis(data)
            if analysis is None:
                print("LLM concept analyst: invalid JSON")
                if attempt == 0:
                    payload["temperature"] = 0.0
                    continue
                return None
            cache[cache_key] = analysis
            _save_cache(cache)
            return analysis
        except Exception as exc:
            print(f"LLM concept analyst unavailable: {str(exc)[:180]}")
            return None
    return None



STRATEGY_VERSION = "strategy-v2-forms1"
STRATEGY_PROMPT = """You are a composition strategist for a Chinese-character design
tool. Given a phrase and its concept analysis, propose five distinct plans for ONE new
glyph. Think about both meaning and sound.

Use these strategies:
- repetition or stacking for repeated, plural, superlative, or "X of X" concepts;
- a semantic radical plus a phonetic or transliteration anchor, with the reading from
  the phonetic;
- an ideographic scene made from two to four semantic components;
- a metaphorical, enclosure, or contrast-based composition.

Use the analysis's Chinese terms, transliteration, and component candidates. For a
foreign term, use the standard Chinese sound as well as the meaning. Do not default to
generic heart, king, water, or animal symbols.

Every component must be one renderable CJK character and must not be a function
character. Use two to four components, each exactly once. Full compound components
must be at most nine strokes. The structure must be a full IDS string using the
operators ⿰ ⿱ ⿲ ⿳ ⿴ ⿵ ⿶ ⿷ ⿸ ⿹ ⿺. For three components use a nested
two-dimensional arrangement such as ⿱A⿰BC or ⿰A⿱BC, not a text description.

For repeated concepts, include a compact repeated pair and a stacked three- or
four-component variant. The engine rejects structures that are already existing
characters, so the variant matters.

For every plan, "forms" must have one entry per component. Use a full character to
preserve a complete component, or "auto" to apply a conventional positional variant.
For ideographic compositions built from complete characters, normally preserve the
full forms; use "auto" for established radicals such as 水→氵, 心→忄, and 手→扌.

Return JSON only:
{"plans": [
  {"mode": "repetition|semantic_phonetic|ideographic|metaphorical",
   "components": ["..."], "forms": ["..."], "structure": "...",
   "semantic": ["..."], "phonetic": "", "reading": "one syllable",
   "gloss": "short meaning", "rationale": "why this encodes meaning and/or sound"}
]}
"""


def _normalize_plans(data) -> list[dict]:
    if isinstance(data, dict) and isinstance(data.get("plans"), list):
        raw_plans = data["plans"]
    elif isinstance(data, list):
        raw_plans = data
    else:
        return []
    plans: list[dict] = []
    for raw in raw_plans:
        if not isinstance(raw, dict):
            continue
        components = [
            char.strip() for char in raw.get("components", [])
            if _single_cjk(char)
        ]
        if not (2 <= len(components) <= 4):
            continue
        structure = str(raw.get("structure") or "").strip()
        if not structure or structure[0] not in "⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺":
            continue
        raw_forms = raw.get("forms")
        forms: list[str] = []
        if isinstance(raw_forms, list) and len(raw_forms) == len(components):
            for form in raw_forms:
                if not isinstance(form, str):
                    continue
                form = form.strip()
                if form == "auto" or _single_cjk(form):
                    forms.append(form)
        if len(forms) != len(components):
            forms = ["auto"] * len(components)
        semantic = raw.get("semantic")
        if isinstance(semantic, list):
            semantic = [
                text for text in semantic if isinstance(text, str) and text.strip()
            ]
        elif isinstance(semantic, str):
            semantic = [semantic.strip()]
        else:
            semantic = []
        plans.append({
            "mode": str(raw.get("mode") or "ideographic").strip()[:40],
            "components": components,
            "forms": forms,
            "structure": structure[:120],
            "semantic": semantic[:4],
            "phonetic": str(raw.get("phonetic") or "").strip()[:10],
            "reading": str(raw.get("reading") or "").strip()[:20],
            "gloss": str(raw.get("gloss") or "").strip()[:240],
            "rationale": str(raw.get("rationale") or "").strip()[:300],
        })
    return plans[:8]


def strategize(
    phrase: str,
    analysis: dict | None,
    feedback: str = "",
) -> list[dict]:
    """Return composition plans for an analyzed phrase, or []."""
    if not isinstance(analysis, dict):
        return []
    key, base, model = _config()
    if not key:
        return []

    digest = __import__("hashlib").sha1(
        json.dumps(analysis, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    cache_key = f"{STRATEGY_VERSION}::{model}::{phrase.strip().lower()}::{digest}"
    if feedback:
        cache_key += "::f" + __import__("hashlib").sha1(
            feedback.encode("utf-8")
        ).hexdigest()[:8]
    cache = _load_cache()
    if cache_key in cache:
        cached = cache[cache_key]
        return cached if isinstance(cached, list) else []

    max_tokens, timeout = _limits()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": STRATEGY_PROMPT},
            {
                "role": "user",
                "content": (
                    f'Phrase: "{phrase}"\nConcept analysis:\n'
                    + json.dumps(analysis, ensure_ascii=False)
                    + (f"\nThe previous plans failed validation: {feedback}. "
                       "Propose different structures and component counts." if feedback else "")
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.7,
        "max_tokens": min(max_tokens, 2200),
        "reasoning_effort": "none",
    }

    for attempt in range(2):
        try:
            import httpx

            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{base}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                )
                if (
                    response.status_code == 400
                    and "reasoning_effort" in payload
                    and "reasoning_effort" in response.text
                ):
                    payload.pop("reasoning_effort", None)
                    response = client.post(
                        f"{base}/chat/completions",
                        headers={"Authorization": f"Bearer {key}"},
                        json=payload,
                    )
            if response.status_code >= 400:
                print(f"LLM strategist unavailable: HTTP {response.status_code}")
                return []
            content = response.json()["choices"][0]["message"].get("content") or ""
            plans = _normalize_plans(_parse_json(content))
            if plans:
                cache[cache_key] = plans
                _save_cache(cache)
                return plans
            print("LLM strategist: invalid or empty plans")
            if attempt == 0:
                payload["temperature"] = 0.2
        except Exception as exc:
            print(f"LLM strategist unavailable: {str(exc)[:180]}")
            return []
    return []


def _reasoning_mode() -> str:
    load_dotenv()
    return os.environ.get("FORGE_LLM_REASONING", DEFAULT_REASONING).strip().lower()


def is_likely_slow() -> bool:
    return _reasoning_mode() not in {"", "none", "off", "false", "0"}


def _limits() -> tuple[int, float]:
    load_dotenv()
    try:
        max_tokens = int(os.environ.get("FORGE_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    except ValueError:
        max_tokens = DEFAULT_MAX_TOKENS
    try:
        timeout = float(os.environ.get("FORGE_LLM_TIMEOUT", DEFAULT_TIMEOUT))
    except ValueError:
        timeout = DEFAULT_TIMEOUT
    return max_tokens, timeout


def _parse_json(content: str) -> dict | None:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def is_available() -> bool:
    return bool(_config()[0])


def last_error() -> str:
    return LAST_ERROR


def model_name() -> str:
    return _config()[2]


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        with open(CACHE_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False)
    tmp.replace(CACHE_PATH)


def _cache_key(
    phrase: str,
    count: int,
    model: str | None = None,
    variation: int = 0,
    feedback: str = "",
    analysis: dict | None = None,
) -> str:
    model = model or _config()[2]
    base = f"{PROMPT_VERSION}::{model}::{phrase.strip().lower()}::{count}"
    if variation:
        base += f"::v{variation}"
    if feedback:
        base += "::f" + __import__("hashlib").sha1(feedback.encode("utf-8")).hexdigest()[:8]
    if analysis:
        digest = __import__("hashlib").sha1(
            json.dumps(analysis, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:10]
        base += f"::a{digest}"
    return base


def is_cached(
    phrase: str,
    count: int = 4,
    variation: int = 0,
    feedback: str = "",
    analysis: dict | None = None,
) -> bool:
    return _cache_key(
        phrase,
        count,
        variation=variation,
        feedback=feedback,
        analysis=analysis,
    ) in _load_cache()


def design(
    phrase: str,
    count: int = 3,
    variation: int = 0,
    feedback: str = "",
    analysis: dict | None = None,
) -> list[dict]:
    """Return creative designs, or [] when the LLM is unavailable/invalid."""
    key, base, model = _config()
    if not key:
        return []

    global LAST_ERROR

    cache_key = _cache_key(
        phrase,
        count,
        model,
        variation=variation,
        feedback=feedback,
        analysis=analysis,
    )
    cache = _load_cache()
    if cache_key in cache:
        LAST_ERROR = ""
        return cache[cache_key]

    max_tokens, timeout = _limits()
    reasoning = _reasoning_mode()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f'English phrase: "{phrase}"\n'
                    f"Create {count} designs as JSON. Use an ideographic design when the "
                    "phrase is a title, loanword, or a repeated/stacked meaning."
                    + (
                        (
                            "\nAnalyst output (authoritative; use it):\n"
                            + json.dumps(analysis, ensure_ascii=False)
                            + "\nMandatory: at least one design must use a component "
                            "from component_candidates. For a foreign term, use the first "
                            "transliteration character as the phonetic with a semantic "
                            "radical and use the analyst's reading; do not fall back to "
                            "generic king/field/cosmos/animal imagery."
                        )
                        if analysis
                        else ""
                    )
                    + (
                        ""
                        if variation == 0
                        else "\nThe previous designs were rejected because they used "
                        "heavy compound components or existing characters. Use only "
                        "atomic components (simple radicals or 4–9 stroke characters)."
                    )
                    + (
                        f"\nRejected component pairs: {feedback}. Do not repeat these "
                        "pairs or their reverses. Choose different, less common "
                        "phonetics and different semantic angles."
                        if feedback
                        else ""
                    )
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 1.2,
        "max_tokens": max_tokens,
    }
    if reasoning:
        payload["reasoning_effort"] = (
            "none" if reasoning in {"none", "off", "false", "0"} else reasoning
        )

    try:
        import httpx

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
            )
            if (
                response.status_code == 400
                and "reasoning_effort" in payload
                and "reasoning_effort" in response.text
            ):
                payload.pop("reasoning_effort", None)
                response = client.post(
                    f"{base}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                )
            if response.status_code >= 400:
                try:
                    detail = response.json().get("error", {}).get("message", "")
                except Exception:
                    detail = ""
                LAST_ERROR = (detail or f"HTTP {response.status_code}")[:220]
                print(f"LLM creative director unavailable: {LAST_ERROR}")
                return []
            body = response.json()
        choice = body["choices"][0]
        if choice.get("finish_reason") == "length":
            LAST_ERROR = (
                "model hit the token limit before returning JSON; "
                "raise FORGE_LLM_MAX_TOKENS"
            )
            print(f"LLM creative director unavailable: {LAST_ERROR}")
            return []
        content = choice["message"].get("content") or ""
        data = _parse_json(content)
        if data is None:
            LAST_ERROR = "model did not return valid JSON"
            print(f"LLM creative director unavailable: {LAST_ERROR}")
            return []
        designs = data.get("designs", [])
        if not isinstance(designs, list):
            LAST_ERROR = "response did not contain a designs list"
            return []
        designs = [item for item in designs if isinstance(item, dict)]
        LAST_ERROR = ""
        cache[cache_key] = designs
        _save_cache(cache)
        return designs
    except Exception as exc:  # network, auth, malformed JSON
        LAST_ERROR = str(exc)[:220]
        print(f"LLM creative director unavailable: {LAST_ERROR}")
        return []


def clear_cache() -> None:
    if CACHE_PATH.exists():
        Path(CACHE_PATH).unlink()
