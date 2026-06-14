"""Text normalization, answer-key matching, and leakage checks.

The four data sources spell the same word differently:
  - crosscheck answers are UPPERCASE and space-stripped:  MAGUTOSNANGMAGUTOS
  - dictionary headwords are lowercase with hyphens:       aalug-alog
  - bugtong answers are UPPERCASE:                         SUNGOT
  - city names are Titlecase:                              Bangued

`answer_key` collapses all of these to a single canonical join key so the
same concept can be matched across sources.
"""

from __future__ import annotations

import re
import unicodedata

# Keep Latin letters and the Tagalog enye; drop everything else (spaces,
# hyphens, periods, digits, punctuation).
_NON_LETTER = re.compile(r"[^A-Za-zÑñ]")

# Part-of-speech / register tags that lead a dictionary gloss, e.g.
#   "aam n. broth made from boiled rice"  ->  strip "n."
#   "abahin (inaaba, ...) v., inf. notify" -> strip "v., inf."
_POS_TAG = re.compile(
    r"^\s*(?:n|adj|adv|v|pron|prep|conj|intrj|interj|art|num|part|idiom)"
    r"\.?(?:\s*,\s*(?:inf|fig|lit|colloq|arch|slang|var)\.?)*\s*\.?\s*",
    re.IGNORECASE,
)

# A leading parenthetical of verb inflections: "(inaaba, inaba, aabahin) ..."
_LEADING_PARENS = re.compile(r"^\s*\([^)]*\)\s*")

# A leading numbered sense marker: "1. inconvenience; 2. interruption"
_LEADING_SENSE_NUM = re.compile(r"^\s*\d+\.\s*")


def answer_key(word: str) -> str:
    """Canonical cross-source join key: uppercase, letters only."""
    word = unicodedata.normalize("NFC", word or "")
    return _NON_LETTER.sub("", word).upper()


def strip_headword(definition: str, headword: str) -> str:
    """Remove the leaked headword + POS tag from a dictionary definition.

    The dictionary redundantly repeats the headword inside the gloss
    (``"aam n. broth..."``), which would leak the answer. This peels off the
    headword, any inflection parenthetical, and the part-of-speech tag,
    leaving just the gloss (``"broth made from boiled rice"``).
    """
    text = (definition or "").strip()
    hk = answer_key(headword)

    # Drop a leading literal copy of the headword (with or without hyphens).
    # Compare on the answer-key so "aalug-alog" matches "aalugalog".
    tokens = text.split()
    while tokens and answer_key(tokens[0]) and hk.startswith(answer_key(tokens[0])):
        # only consume if it actually advances the headword match
        consumed = answer_key(tokens[0])
        if not consumed:
            break
        tokens = tokens[1:]
        hk = hk[len(consumed):]
        if not hk:
            break
    text = " ".join(tokens)

    text = _LEADING_PARENS.sub("", text)
    text = _POS_TAG.sub("", text)
    text = _LEADING_PARENS.sub("", text)  # parens sometimes follow the POS tag
    text = _LEADING_SENSE_NUM.sub("", text)
    return text.strip(" .;,")


# `tagalogdefinitionpure` is overwhelmingly a Tagalog->English dictionary, so a
# gloss is assumed English UNLESS it carries Tagalog function words/markers.
# This is deliberately conservative: short English glosses like "folding fan"
# have no markers and are correctly treated as English (knowledge only), not as
# natural-Tagalog clue text.
_TL_MARKERS = {
    "ng", "mga", "sa", "ang", "na", "ay", "ko", "mo", "ito", "iyon", "iyan",
    "kung", "nang", "may", "mayroon", "wala", "walang", "isang", "para",
    "pang", "taga", "kay", "nila", "niya", "ka", "ako", "ikaw", "sila",
    "tayo", "kami", "hindi", "ginagamit", "ginagawa", "uri", "bagay",
    "tao", "lugar", "isa", "dalawa", "kapag", "upang", "dahil",
}


def gloss_is_tagalog(text: str) -> bool:
    words = re.findall(r"[A-Za-zÑñ]+", (text or "").lower())
    return any(w in _TL_MARKERS for w in words)


# --- Tagalog morphology (heuristic) -----------------------------------------
#
# The round-trip verifier rejects clues whose solver lands on a different
# *inflection* of the right word (DUMAMPI vs DAMPI, MAGTAAS vs TAAS). Tagalog is
# heavily affixing/reduplicating, so an exact-surface match is too strict. These
# helpers reduce a word to a SET of plausible roots; two words "share a lemma"
# if their root sets intersect on a stem of >= 4 letters. The set approach is
# deliberately generous on *recall* (try every reduction) but guarded on
# *precision* by the length-4 floor, so MAGANDA->{GANDA,ANDA,...} still matches
# GANDA without ANDA causing spurious hits.

# Applied longest-first so MAG- is tried before MA-, etc.
_PREFIXES = (
    "ipinagpa", "ipinaki", "ipinag", "ipina", "ipa",
    "nakikipag", "makikipag", "nakipag", "makipag",
    "nakapagpa", "makapagpa", "nakapag", "makapag",
    "nagpaka", "magpaka", "napaka", "pinaka",
    "nakaka", "makaka", "nagpa", "magpa", "pagpa",
    "pinag", "pina", "nagka", "magka", "pagka",
    "naka", "maka", "naki", "maki",
    "nang", "mang", "pang", "nag", "mag", "pag",
    "nam", "mam", "pam", "nan", "man", "pan",
    "na", "ma", "pa", "ka", "ki", "um", "in", "i",
)
_SUFFIXES = ("han", "hin", "ng", "an", "in")


def _reductions(w: str) -> set[str]:
    """One layer of plausible morphological reductions of an answer-key form."""
    out: set[str] = set()
    n = len(w)
    # leading doubled vowel reduplication: AALUG -> ALUG
    if n >= 3 and w[0] == w[1] and w[0] in "AEIOU":
        out.add(w[1:])
    # CV-syllable reduplication: SUSULAT -> SULAT, LALARO -> LARO
    if n >= 4 and w[0:2] == w[2:4]:
        out.add(w[2:])
    # exact full-word reduplication (hyphen already stripped): HALOHALO -> HALO
    if n % 2 == 0 and w[: n // 2] == w[n // 2:]:
        out.add(w[: n // 2])
    # -um-/-in- infix after an initial consonant: DUMAMPI -> DAMPI
    if n >= 4 and w[0] not in "AEIOU" and w[1:3] in ("UM", "IN"):
        out.add(w[0] + w[3:])
    # prefix strip (one prefix), keeping a >=3-letter remainder
    for p in _PREFIXES:
        pu = p.upper()
        if w.startswith(pu) and n - len(pu) >= 3:
            out.add(w[len(pu):])
    # suffix strip (one suffix)
    for s in _SUFFIXES:
        su = s.upper()
        if w.endswith(su) and n - len(su) >= 3:
            out.add(w[: -len(su)])
    return out


def stem_candidates(word: str) -> set[str]:
    """All plausible roots of `word` (incl. itself), via a fixed-point of
    `_reductions`. Generous by design; precision comes from the caller's
    >=4-letter match floor."""
    w = answer_key(word)
    if not w:
        return set()
    seen = {w}
    frontier = {w}
    for _ in range(5):  # bounded; layered affixes (NAG+redup) need a few passes
        nxt: set[str] = set()
        for cand in frontier:
            for r in _reductions(cand):
                if r not in seen:
                    seen.add(r)
                    nxt.add(r)
        if not nxt:
            break
        frontier = nxt
    return {s for s in seen if len(s) >= 3}


def same_lemma(a: str, b: str, *, min_len: int = 4) -> bool:
    """True if `a` and `b` plausibly share a Tagalog root of >= min_len letters."""
    shared = stem_candidates(a) & stem_candidates(b)
    return any(len(s) >= min_len for s in shared)


def reveal_pattern(word: str, every: int = 3) -> str:
    """Deterministic crossing-letter simulation: reveal every Nth letter, mask
    the rest with '_'. DUMAMPI -> 'D__A__I'. Mimics the partial constraints a
    solver gets from filled crossings in a real grid."""
    k = answer_key(word)
    return "".join(ch if i % every == 0 else "_" for i, ch in enumerate(k))


def leaks_answer(clue: str, answer: str) -> bool:
    """True if the clue reveals the answer (shared word stem)."""
    ak = answer_key(answer)
    if not ak:
        return False
    for tok in re.findall(r"[A-Za-zÑñ]+", clue or ""):
        tk = answer_key(tok)
        if not tk:
            continue
        # exact token match, or one contains the other for >=4-letter stems
        if tk == ak:
            return True
        if len(tk) >= 4 and len(ak) >= 4 and (tk in ak or ak in tk):
            return True
    return False
