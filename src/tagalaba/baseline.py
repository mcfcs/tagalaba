"""Baseline (retrieval/template) Tagalog clue generator.

This is the floor a trained model must beat. It produces no novel language --
it only retrieves and lightly templates the genuinely-Tagalog material already
present in the data (synonyms, riddles, place trivia), filters answer leakage,
ranks by how natural the source tends to be, and returns the top candidates.

Sources ranked by natural-Tagalog quality:
    bugtong (riddle)   > tagalog gloss > synonym > place trivia > english gloss
"""

from __future__ import annotations

from dataclasses import dataclass

from .normalize import leaks_answer
from .sources import ClueIndex, get_index


@dataclass
class Candidate:
    clue: str
    source: str        # bugtong | synonym | gloss_tl | place | gloss_en
    score: float
    language: str      # tl | en


# Base score per source: how natural-Tagalog the output tends to be.
_SOURCE_SCORE = {
    "bugtong": 1.00,
    "gloss_tl": 0.78,
    "synonym": 0.70,
    "place": 0.62,
    "gloss_en": 0.30,
}


def _candidates(word: str, index: ClueIndex) -> list[Candidate]:
    wr = index.get(word)
    if wr is None:
        return []

    out: list[Candidate] = []

    for riddle in wr.bugtong:
        out.append(Candidate(riddle, "bugtong", _SOURCE_SCORE["bugtong"], "tl"))

    for g in wr.glosses_tl:
        out.append(Candidate(g, "gloss_tl", _SOURCE_SCORE["gloss_tl"], "tl"))

    # Synonyms: present both the bare synonym (NYT-style) and a templated form.
    for syn in wr.synonyms:
        out.append(Candidate(syn, "synonym", _SOURCE_SCORE["synonym"], "tl"))
        out.append(
            Candidate(
                f"Kasingkahulugan ng '{syn}'",
                "synonym",
                _SOURCE_SCORE["synonym"] - 0.05,
                "tl",
            )
        )

    if wr.province:
        prov = wr.province
        out.append(
            Candidate(
                f"Lungsod o bayan sa lalawigan ng {prov}",
                "place",
                _SOURCE_SCORE["place"],
                "tl",
            )
        )

    for g in wr.glosses_en:
        out.append(Candidate(g, "gloss_en", _SOURCE_SCORE["gloss_en"], "en"))

    return out


def _dedupe(cands: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    keep: list[Candidate] = []
    for c in sorted(cands, key=lambda x: x.score, reverse=True):
        norm = c.clue.strip().lower()
        if norm in seen:
            continue
        seen.add(norm)
        keep.append(c)
    return keep


def generate(
    word: str,
    *,
    top_k: int = 5,
    allow_english: bool = False,
    index: ClueIndex | None = None,
) -> list[Candidate]:
    """Return up to `top_k` ranked clue candidates for `word`.

    `allow_english=False` (default) drops the English dictionary glosses so the
    output is natural Tagalog only.
    """
    index = index or get_index()
    cands = _candidates(word, index)
    cands = [c for c in cands if not leaks_answer(c.clue, word)]
    cands = [c for c in cands if len(c.clue.strip()) >= 2]
    if not allow_english:
        cands = [c for c in cands if c.language == "tl"]
    return _dedupe(cands)[:top_k]
