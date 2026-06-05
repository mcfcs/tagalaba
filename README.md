# tagalaba — natural-sounding Tagalog crossword clue generator

Goal: given a Tagalog answer word, generate a natural-sounding crossword clue
with the *craft* of a good NYT clue (fair, well-formed, unambiguous) but in
idiomatic Tagalog.

## Data reality (measured, not assumed)

| Source | Words | Clue language | Role |
|---|---|---|---|
| `tagalogcrosswordcrosscheck.csv` | 37,138 | **Tagalog** (synonyms) | Backbone — terse but natural |
| `tagalog_bugtong.csv` | 309 | **Tagalog** (riddles) | Best *voice*, scarce |
| `cities_regions_raw.csv` | 1,414 | place names | Templated trivia |
| `tagalogdefinitionpure.csv` | 42,306 EN / 174 TL | **English glosses** | Knowledge only — NOT natural TL |

Key finding: there is **no large gold corpus of well-crafted Tagalog clues**.
The dictionary is Tagalog→English. So a trained model for *natural* Tagalog
will need a **synthesized, round-trip-verified** Tagalog clue set to learn from.

## Why not RoBERTa / XLM-R

They are encoder-only (understanding) models and cannot generate text. Use them
only for auxiliary scoring/classification. For generation use an encoder-decoder
(**mT5 / mBART**) or a Filipino-capable decoder LLM (**SEA-LION, Sailor, Aya,
Qwen**) fine-tuned with LoRA/QLoRA.

## What's built now: the baseline (the floor to beat)

A pure retrieval/template generator over the genuinely-Tagalog material, with
answer-leakage filtering and source-quality ranking. No novel language yet.

```bash
# from repo root
PYTHONPATH=src python -m tagalaba            # dataset coverage stats
PYTHONPATH=src python -m tagalaba HINTO      # -> Tahan / Tumigil / ...
PYTHONPATH=src python -m tagalaba SUNGOT     # -> bugtong riddle + synonyms
PYTHONPATH=src python -m tagalaba ABANIKO --english   # allow EN glosses
```

Ranking (most→least natural): `bugtong > tagalog gloss > synonym > place > english gloss`.

## Roadmap

1. **[done]** Baseline retrieval generator + unified data index.
2. **Round-trip verifier** — generate a clue, solve it back with an LLM/solver,
   keep only clues that recover the answer. The single biggest quality lever.
3. **Synthesize a Tagalog clue set** — use a Filipino-capable LLM to write
   natural Tagalog clues (seeded by synonym + English meaning + bugtong style),
   filter by round-trip + leakage. This becomes the training data.
4. **Fine-tune** a self-hosted model (mT5 or QLoRA'd SEA-LION/Sailor) on the
   verified set, conditioned on difficulty once labels exist.
5. **Evaluate** — round-trip solve rate, leakage rate, held-out-by-word, human
   preference.

## Layout

```
data/                     # source CSVs (untracked)
src/tagalaba/
  normalize.py            # answer-key join, headword/POS stripping, leakage, lang
  sources.py              # load 4 CSVs -> unified ClueIndex
  baseline.py             # word -> ranked Tagalog clue candidates
  cli.py / __main__.py    # python -m tagalaba
```
