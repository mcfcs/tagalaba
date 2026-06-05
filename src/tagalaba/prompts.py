"""Prompt builders for clue generation (Gemini) and verification (Claude)."""

from __future__ import annotations

from .sources import WordResources

# A few real bugtong, embedded only to anchor the *voice* (natural, idiomatic
# Tagalog). They are style exemplars, never tied to the target word.
_STYLE_ANCHORS = [
    "Maitim na puwit, tangkay ay nakakabit.",
    "Di dapat na kulangin, di rin dapat pasobrahin.",
    "Heto na si Kaka, bubuka-bukaka.",
]

GEN_SYSTEM = (
    "Ikaw ay dalubhasa sa paggawa ng mga pahiwatig (clues) para sa Tagalog na "
    "crossword — kasing-husay ng New York Times ngunit ganap na nasa natural, "
    "idyomatikong Tagalog.\n\n"
    "Bibigyan kita ng isang SAGOT at ilang sanggunian (kasingkahulugan, "
    "kahulugan sa Ingles para sa konteksto, antas ng hirap). Gumawa ng mga "
    "pahiwatig na sumusunod sa MAHIGPIT na mga patakaran:\n"
    "1. Natural na Tagalog lamang (maliban sa mga karaniwang hiram na salita).\n"
    "2. HUWAG banggitin ang sagot o ang ugat-salita nito sa loob ng pahiwatig.\n"
    "3. Patas at malulutas — malinaw na tumuturo sa sagot.\n"
    "4. Iba't ibang istilo: kahulugan, kasingkahulugan, 'punan ang patlang' "
    "(gamit ang ___), at banayad na palaisipan.\n"
    "5. Iwasang maging tuyot o literal na salin mula sa Ingles.\n\n"
    "Halimbawa ng natural na tono (bugtong, istilo lamang):\n  - "
    + "\n  - ".join(_STYLE_ANCHORS)
    + "\n\nIbalik LAMANG bilang JSON:\n"
    '{"clues": [{"clue": "...", "difficulty": "madali|katamtaman|mahirap", '
    '"style": "kahulugan|kasingkahulugan|patlang|palaisipan"}]}'
)


def gen_user_prompt(wr: WordResources, n: int) -> str:
    syns = ", ".join(wr.synonyms[:8]) if wr.synonyms else "(wala)"
    en = "; ".join((wr.glosses_en or wr.glosses_tl)[:3]) or "(wala)"
    place = f"\nLugar: bayan/lungsod sa {wr.province}" if wr.province else ""
    return (
        f"SAGOT: {wr.display.upper()} ({len(wr.key)} titik)\n"
        f"Kasingkahulugan: {syns}\n"
        f"Kahulugan (Ingles, konteksto lamang): {en}{place}\n\n"
        f"Gumawa ng {n} magkakaibang pahiwatig na may iba't ibang antas ng hirap."
    )


VERIFY_SYSTEM = (
    "You are an expert Tagalog crossword solver. You are given a clue and the "
    "number of letters in the answer. Reply with your three best guesses as "
    "UPPERCASE Tagalog words (letters only, no spaces), best guess first. "
    'Reply ONLY as JSON: {"guesses": ["...", "...", "..."]}. No explanation.'
)


def verify_user_prompt(clue: str, length: int) -> str:
    return f"Pahiwatig: {clue}\nBilang ng titik: {length}"
