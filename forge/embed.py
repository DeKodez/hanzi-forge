"""Semantic index over character glosses using a small pretrained model."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

from .paths import INDEX_DIR, MODELS

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_CHARS = INDEX_DIR / "chars.json"
_VECTORS = INDEX_DIR / "embeddings.npy"
_META = INDEX_DIR / "meta.json"
_HINTS = INDEX_DIR / "keyword_hints.json"

_model = None

STOPWORDS = set(
    """
    a about above after again against all also am an and any are as at be because
    been before being below between both but by can could did do does doing down
    during each few for from further had has have having he her here hers herself
    him himself his how i if in into is it its itself just me more most my myself
    no nor not of off on once only or other our ours ourselves out over own
    same she should so some such than that the their theirs them themselves then
    there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours
    yourself yourselves go goes going went gone get gets got getting make makes
    made someone something anything everyone nobody
    """.split()
)

_QUERY_WORDS = re.compile(r"[a-z']+")


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODELS))
    return _model


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _gloss_hash(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def frequency_prior(rank: int | None) -> float:
    """A mild preference for characters people actually use."""
    if not rank:
        return 0.94
    if rank <= 500:
        return 1.12
    if rank <= 1500:
        return 1.08
    if rank <= 3000:
        return 1.04
    if rank <= 6000:
        return 1.01
    return 0.97


def build_index(
    records: dict[str, dict],
    batch_size: int = 64,
    force: bool = False,
    keyword_hints: dict[str, list[str]] | None = None,
) -> None:
    order = list(records)
    texts = [records[char]["gloss"] for char in order]
    digest = _gloss_hash(texts)

    meta: dict = {}
    if _META.exists():
        with open(_META, encoding="utf-8") as handle:
            meta = json.load(handle)

    reusable = (
        not force
        and _VECTORS.exists()
        and meta.get("model") == MODEL_NAME
        and meta.get("count") == len(texts)
        and meta.get("gloss_hash") == digest
    )

    if reusable:
        vectors = np.load(_VECTORS)
        print("Glosses unchanged; reusing the existing embeddings.")
    else:
        print(f"Embedding {len(texts):,} character glosses with {MODEL_NAME}...")
        model = _get_model()
        vectors = np.vstack([vector for vector in model.embed(texts, batch_size=batch_size)])
        vectors = _normalize(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_VECTORS, vectors)
    with open(_CHARS, "w", encoding="utf-8") as handle:
        json.dump([records[char] for char in order], handle, ensure_ascii=False)
    with open(_META, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": MODEL_NAME,
                "count": len(order),
                "dimensions": int(vectors.shape[1]),
                "gloss_hash": digest,
            },
            handle,
        )
    if keyword_hints is not None:
        with open(_HINTS, "w", encoding="utf-8") as handle:
            json.dump(keyword_hints, handle, ensure_ascii=False)
        print(f"Saved {len(keyword_hints):,} keyword hints.")
    print(f"Index written to {INDEX_DIR} ({vectors.shape[0]:,} x {vectors.shape[1]}).")


@dataclass(frozen=True)
class Match:
    char: str
    score: float
    record: dict


class SemanticIndex:
    def __init__(
        self,
        chars: list[dict],
        vectors: np.ndarray,
        keyword_hints: dict[str, list[str]] | None = None,
    ):
        self.chars = chars
        self.vectors = _normalize(vectors)
        self.by_char = {record["char"]: record for record in chars}
        self.index_by_char = {record["char"]: index for index, record in enumerate(chars)}
        self.keyword_hints = keyword_hints or {}
        self.priors = np.array(
            [frequency_prior(record.get("rank")) for record in chars], dtype=np.float32
        )

    @classmethod
    def load(cls) -> "SemanticIndex":
        if not (_CHARS.exists() and _VECTORS.exists()):
            raise FileNotFoundError(
                "Semantic index missing. Run: python scripts/build_index.py"
            )
        with open(_CHARS, encoding="utf-8") as handle:
            chars = json.load(handle)
        vectors = np.load(_VECTORS)
        hints = {}
        if _HINTS.exists():
            with open(_HINTS, encoding="utf-8") as handle:
                hints = json.load(handle)
        return cls(chars, vectors, hints)

    def hint_groups(self, phrase: str, per_word: int = 5) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for word in self.content_words(phrase):
            candidates = self.keyword_hints.get(word, [])
            if not candidates:
                continue
            query = self._embed_query(word)
            ranked: list[str] = []
            for char in candidates:
                index = self.index_by_char.get(char)
                if index is None:
                    continue
                if float(self.vectors[index] @ query) >= 0.47:
                    ranked.append(char)
            groups[word] = ranked[:per_word]
        return groups

    def hints_for(self, phrase: str, per_word: int = 5) -> list[str]:
        groups = self.hint_groups(phrase, per_word=per_word)
        found = [char for chars in groups.values() for char in chars]
        return list(dict.fromkeys(found))

    def _embed_query(self, text: str) -> np.ndarray:
        vector = next(iter(_get_model().embed([QUERY_PREFIX + text], batch_size=1)))
        return _normalize(vector)[0]

    @staticmethod
    def content_words(phrase: str) -> list[str]:
        words = _QUERY_WORDS.findall(phrase.lower())
        return [word for word in words if len(word) >= 3 and word not in STOPWORDS][:5]

    def search(self, phrase: str, k: int = 60) -> list[Match]:
        scores = self.vectors @ self._embed_query(phrase)

        words = self.content_words(phrase)
        if words:
            word_scores = [
                (self.vectors @ self._embed_query(word)) * self.priors for word in words
            ]
            scores = scores + 0.32 * np.vstack(word_scores).max(axis=0)

            # Make sure every concept in the phrase gets at least one representative,
            # even when one word dominates the sentence embedding.
            for word_score in word_scores:
                index = int(np.argmax(word_score))
                scores[index] = max(scores[index], float(word_score[index]) * 0.80)

        scores = scores * self.priors

        k = min(k, len(scores))
        best = np.argpartition(-scores, k - 1)[:k]
        best = best[np.argsort(-scores[best])]
        return [
            Match(self.chars[index]["char"], float(scores[index]), self.chars[index])
            for index in best
        ]
