"""Load the four CSV sources into a unified per-word clue index.

Each source contributes differently-shaped raw material for a clue:
  - synonyms : terse, genuinely Tagalog          (main material)
  - bugtong  : rich, idiomatic Tagalog riddles    (best *voice*, scarce)
  - cities   : place -> province trivia           (templated knowledge)
  - glosses  : English meanings                    (knowledge only, not natural TL)
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field

from .normalize import answer_key, strip_headword, gloss_is_tagalog

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
)


@dataclass
class WordResources:
    """Everything we know about one answer, keyed by `answer_key`."""

    key: str
    display: str = ""                       # nicest surface form we've seen
    synonyms: list[str] = field(default_factory=list)
    bugtong: list[str] = field(default_factory=list)
    glosses_tl: list[str] = field(default_factory=list)
    glosses_en: list[str] = field(default_factory=list)
    province: str | None = None             # set if the answer is a place


class ClueIndex:
    """In-memory index: answer_key -> WordResources, built from the CSVs."""

    def __init__(self) -> None:
        self._by_key: dict[str, WordResources] = {}

    def _slot(self, raw_word: str) -> WordResources:
        k = answer_key(raw_word)
        wr = self._by_key.get(k)
        if wr is None:
            wr = WordResources(key=k, display=raw_word)
            self._by_key[k] = wr
        return wr

    # ---- loaders -------------------------------------------------------

    def load_all(self, data_dir: str = _DATA_DIR) -> "ClueIndex":
        self._load_synonyms(os.path.join(data_dir, "tagalogcrosswordcrosscheck.csv"))
        self._load_bugtong(os.path.join(data_dir, "tagalog_bugtong.csv"))
        self._load_cities(os.path.join(data_dir, "cities_regions_raw.csv"))
        self._load_glosses(os.path.join(data_dir, "tagalogdefinitionpure.csv"))
        return self

    def _load_synonyms(self, path: str) -> None:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                word, clue = row.get("Word", ""), (row.get("Clue") or "").strip()
                if not word or not clue:
                    continue
                wr = self._slot(word)
                if clue not in wr.synonyms:
                    wr.synonyms.append(clue)

    def _load_bugtong(self, path: str) -> None:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                word, clue = row.get("Word", ""), (row.get("Clue") or "").strip()
                if not word or not clue:
                    continue
                self._slot(word).bugtong.append(clue)

    def _load_cities(self, path: str) -> None:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                city = (row.get("Lungsod o bayan") or "").strip()
                prov = (row.get("Lalawigan") or "").strip()
                if not city:
                    continue
                wr = self._slot(city)
                wr.display = city
                wr.province = prov or wr.province

    def _load_glosses(self, path: str) -> None:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                word = (row.get("word") or "").strip()
                definition = (row.get("definition") or "").strip()
                if not word or not definition:
                    continue
                gloss = strip_headword(definition, word)
                if not gloss:
                    continue
                wr = self._slot(word)
                if len(word) > len(wr.display):  # prefer hyphenated surface form
                    wr.display = word
                bucket = wr.glosses_tl if gloss_is_tagalog(gloss) else wr.glosses_en
                if gloss not in bucket:
                    bucket.append(gloss)

    # ---- access --------------------------------------------------------

    def get(self, word: str) -> WordResources | None:
        return self._by_key.get(answer_key(word))

    def __len__(self) -> int:
        return len(self._by_key)

    def coverage(self) -> dict[str, int]:
        c = defaultdict(int)
        c["total_words"] = len(self._by_key)
        for wr in self._by_key.values():
            if wr.synonyms:
                c["has_synonym"] += 1
            if wr.bugtong:
                c["has_bugtong"] += 1
            if wr.province is not None:
                c["is_place"] += 1
            if wr.glosses_tl:
                c["has_tagalog_gloss"] += 1
            if wr.glosses_en:
                c["has_english_gloss"] += 1
        return dict(c)


_INDEX: ClueIndex | None = None


def get_index(data_dir: str = _DATA_DIR) -> ClueIndex:
    """Load (once) and return the shared index."""
    global _INDEX
    if _INDEX is None:
        _INDEX = ClueIndex().load_all(data_dir)
    return _INDEX
