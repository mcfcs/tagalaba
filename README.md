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

## Synthesis pipeline (builds the training data)

`tagalaba.synthesize` produces a verified Tagalog clue set:
**Gemini generates** natural-Tagalog clues per word → leakage filter →
**Claude Haiku solves each clue back** to the answer (honest round-trip, a
*different* model than the generator) → keep only clues that recover the word.
Output is resumable JSONL, rate-limited for Gemini's free tier.

```bash
pip install -e .                          # makes `tagalaba` importable everywhere
                                          #   (no PYTHONPATH needed; Windows-friendly)
copy .env.example .env                     # then fill ANTHROPIC_API_KEY (+ GEMINI_API_KEY)

python -m tagalaba.synthesize --dry-run --limit 3   # prompts only, no cost
python -m tagalaba.synthesize --limit 50            # pilot, eyeball quality
python -m tagalaba.synthesize --limit 30000         # full set (resumable)
```

**Generator backend** (`--gen-provider`, default `anthropic`):
- `anthropic` — Claude **Sonnet** generates, **Haiku** verifies (two different
  models, so the solver isn't the author). Works with just your Anthropic key.
- `gemini` — Gemini Flash generates (cheapest), Haiku verifies. Needs Gemini
  free-tier access or prepaid credits.

Keys are read from `.env` (git-ignored). `--max-rank 1` (default) is strict
precision; raise it to keep more borderline clues. Cost: pilot < $1; full set
~$150 all-Claude, or ~$15 if you enable Gemini for generation.

## Training & deployment — sized for a 16 GB-RAM laptop

| Stage | Where | Notes |
|---|---|---|
| Build dataset (above) | the laptop, via APIs | light, API-bound |
| **Fine-tune** | **free cloud GPU** (Colab/Kaggle T4) | 16 GB RAM can't train an LLM |
| **Run the model** | the laptop, **CPU** | must be small |

Local-inference targets that fit 16 GB on CPU:
- **mT5-base (580M)** — purpose-built seq2seq for word→clue; trivial on CPU.
- **A 4-bit ≤3B decoder** (Gemma-2-2B / Qwen2.5-3B) via **Ollama** — ~2 GB,
  more natural fluency. Avoid 7B+ (runs but painfully slow on CPU).

## Roadmap

1. **[done]** Baseline retrieval generator + unified data index.
2. **[done]** Round-trip verifier + synthesis pipeline (`synthesize.py`).
3. **Fine-tune** mT5-base / a quantized ≤3B decoder on the verified set (cloud GPU).
4. **Evaluate** — round-trip solve rate, leakage rate, held-out-by-word, human pref.

## Layout

```
data/                     # source CSVs (untracked)
src/tagalaba/
  normalize.py            # answer-key join, headword/POS stripping, leakage, lang
  sources.py              # load 4 CSVs -> unified ClueIndex
  baseline.py             # word -> ranked Tagalog clue candidates
  cli.py / __main__.py    # python -m tagalaba
```
