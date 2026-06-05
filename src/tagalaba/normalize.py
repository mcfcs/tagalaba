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
