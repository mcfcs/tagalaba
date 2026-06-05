"""Build a verified Tagalog clue training set.

For each answer word:
  1. Gemini writes N natural-Tagalog clue candidates (seeded by synonyms/meaning).
  2. Drop any clue that leaks the answer.
  3. Claude Haiku tries to SOLVE each clue back to the word (round-trip).
  4. Keep a clue only if Claude recovers the answer within --max-rank guesses.

Output is JSONL (one verified clue per line), resumable, free-tier rate-limited.

Run (after `pip install -r requirements.txt` and filling .env):
    PYTHONPATH=src python -m tagalaba.synthesize --limit 50            # pilot
    PYTHONPATH=src python -m tagalaba.synthesize --dry-run --limit 3   # no API calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import config, prompts
from .normalize import answer_key, leaks_answer
from .sources import WordResources, get_index


def _utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _pick_words(index, limit: int, require_synonym: bool) -> list[WordResources]:
    """Deterministic selection: words with the richest Tagalog material first."""
    words = list(index._by_key.values())  # noqa: SLF001 - internal access by design
    if require_synonym:
        words = [w for w in words if w.synonyms or w.bugtong]
    # Prefer words that have a synonym AND a meaning, then sort by key for
    # reproducibility across runs (so --limit always picks the same prefix).
    words.sort(key=lambda w: (-(bool(w.synonyms) + bool(w.glosses_en or w.glosses_tl)
                                + bool(w.bugtong)), w.key))
    return words[:limit]


def _done_keys(out_path: str) -> set[str]:
    done: set[str] = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except Exception:
                    continue
    return done


def _rank_of(word: str, guesses: list[str]) -> int:
    ak = answer_key(word)
    for i, g in enumerate(guesses, 1):
        if answer_key(g) == ak:
            return i
    return 0


def run(args: argparse.Namespace) -> int:
    _utf8()
    index = get_index()
    words = _pick_words(index, args.limit, not args.no_require_synonym)

    if args.dry_run:
        for wr in words[:3]:
            print("=" * 72)
            print(prompts.gen_user_prompt(wr, args.clues_per_word))
        print("\n[dry-run] No API calls made. Remove --dry-run to generate.")
        return 0

    from .llm import ClaudeVerifier, FatalAPIError, make_generator  # lazy

    gen = make_generator(args.gen_provider)
    if args.gen_model:
        gen.model = args.gen_model
    ver = ClaudeVerifier()
    print(f"generate: {gen.model}  |  verify: {ver.model}")

    done = _done_keys(args.out)
    todo = [w for w in words if w.key not in done]
    print(f"{len(done)} already done; {len(todo)} to process -> {args.out}")

    min_interval = 60.0 / max(args.rpm, 1)
    kept = attempted = 0

    with open(args.out, "a", encoding="utf-8") as out:
        for n, wr in enumerate(todo, 1):
            t0 = time.time()
            try:
                cands = gen.generate(prompts.GEN_SYSTEM,
                                     prompts.gen_user_prompt(wr, args.clues_per_word))
            except FatalAPIError as e:
                print(f"\nFATAL: generation API is unusable -> {e}\n"
                      f"Fix billing/quota, or switch provider with "
                      f"--gen-provider {'gemini' if gen.model.startswith('claude') else 'anthropic'}.")
                break
            except Exception as e:  # noqa: BLE001 - transient; skip this word
                print(f"  ! gen failed for {wr.display}: {e}")
                cands = []

            for c in cands:
                clue = (c.get("clue") or "").strip()
                if not clue or leaks_answer(clue, wr.display):
                    continue
                attempted += 1
                try:
                    guesses = ver.guesses(
                        prompts.VERIFY_SYSTEM,
                        prompts.verify_user_prompt(clue, len(wr.key)),
                    )
                except FatalAPIError as e:
                    print(f"\nFATAL: verification API is unusable -> {e}")
                    return 1
                except Exception as e:  # noqa: BLE001 - transient; skip clue
                    print(f"  ! verify failed: {e}")
                    continue
                rank = _rank_of(wr.display, guesses)
                if rank and rank <= args.max_rank:
                    rec = {
                        "key": wr.key,
                        "answer": wr.display.upper(),
                        "length": len(wr.key),
                        "clue": clue,
                        "difficulty": c.get("difficulty"),
                        "style": c.get("style"),
                        "verify_rank": rank,
                        "gen_model": gen.model,
                        "verify_model": ver.model,
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    kept += 1

            if n % 10 == 0 or n == len(todo):
                print(f"  [{n}/{len(todo)}] kept={kept} attempted={attempted} "
                      f"pass={kept/max(attempted,1):.0%}")

            dt = time.time() - t0
            if dt < min_interval and n < len(todo):
                time.sleep(min_interval - dt)

    print(f"\nDone. Kept {kept}/{attempted} clues "
          f"({kept/max(attempted,1):.0%} round-trip pass rate).")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tagalaba.synthesize")
    ap.add_argument("--limit", type=int, default=50, help="words to process")
    ap.add_argument("--clues-per-word", type=int, default=4)
    ap.add_argument("--max-rank", type=int, default=1,
                    help="keep clue if answer is within this guess rank (1=strict)")
    ap.add_argument("--rpm", type=int, default=config.RPM, help="requests/min cap")
    ap.add_argument("--gen-provider", default=config.GEN_PROVIDER,
                    choices=["anthropic", "gemini"], help="clue generator backend")
    ap.add_argument("--gen-model", default=None, help="override generator model id")
    ap.add_argument("--out", default=config.DEFAULT_OUT)
    ap.add_argument("--no-require-synonym", action="store_true",
                    help="also process words lacking synonyms/bugtong")
    ap.add_argument("--dry-run", action="store_true",
                    help="print prompts only; make no API calls")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
