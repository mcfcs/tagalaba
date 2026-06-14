"""Build a verified Tagalog clue training set.

For each answer word (processed concurrently across --workers threads):
  1. The generator model writes N natural-Tagalog clue candidates.
  2. Drop any clue that leaks the answer or exceeds --max-words.
  3. A DIFFERENT model tries to SOLVE each clue back to the word (round-trip).
  4. Keep a clue only if the solver recovers the answer (or a known synonym of
     it) within --max-rank guesses.

Output is JSONL (one verified clue per line), resumable (skips done words).

Run (after `pip install -e .` and filling .env):
    python -m tagalaba.synthesize --limit 200          # pilot
    python -m tagalaba.synthesize --dry-run --limit 3  # no API calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config, prompts
from .normalize import answer_key, leaks_answer, reveal_pattern, same_lemma
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


def _accept_rank(wr: WordResources, guesses: list[str]) -> tuple[int, str]:
    """(rank, tier) of the first guess that counts as solving the clue.

    Tiers, strictest first -- recorded per clue so training can filter by rigor:
      exact   : guess IS the answer.
      synonym : guess is a listed synonym (AKLAT<->LIBRO) -- right concept.
      lemma   : guess shares a Tagalog root with the answer or a synonym
                (DUMAMPI<->DAMPI) -- right word, different inflection.
    All three are fair because crossing letters disambiguate in a real grid.
    Returns (0, "") if no guess qualifies.
    """
    syns = {answer_key(s) for s in wr.synonyms} - {""}
    for i, g in enumerate(guesses, 1):
        gk = answer_key(g)
        if not gk:
            continue
        if gk == wr.key:
            return i, "exact"
        if gk in syns:
            return i, "synonym"
        if same_lemma(gk, wr.key) or any(same_lemma(gk, s) for s in syns):
            return i, "lemma"
    return 0, ""


def _process_word(wr, gen, ver, args):
    """Generate + verify all clues for one word (runs in a worker thread).

    Returns (out_records, dbg_records, attempted, dropped_long, warn). Re-raises
    FatalAPIError so the main loop can abort the whole run.
    """
    from .llm import FatalAPIError  # lazy

    out_records: list[dict] = []
    dbg_records: list[dict] = []
    attempted = dropped_long = 0

    try:
        cands = gen.generate(prompts.GEN_SYSTEM,
                             prompts.gen_user_prompt(wr, args.clues_per_word))
    except FatalAPIError:
        raise
    except Exception as e:  # noqa: BLE001 - transient; skip this word
        return out_records, dbg_records, 0, 0, f"gen failed for {wr.display}: {e}"

    for c in cands:
        clue = (c.get("clue") or "").strip()
        if not clue or leaks_answer(clue, wr.display):
            continue
        if len(clue.split()) > args.max_words:   # too long -> drop
            dropped_long += 1
            continue
        attempted += 1
        try:
            guesses = ver.guesses(prompts.VERIFY_SYSTEM,
                                  prompts.verify_user_prompt(clue, len(wr.key)))
        except FatalAPIError:
            raise
        except Exception:  # noqa: BLE001 - transient; skip clue
            continue
        rank, tier = _accept_rank(wr, guesses)
        keep = bool(rank and rank <= args.max_rank)

        # Pattern rescue: a fair clue can still miss when the solver lands on a
        # meaning-neighbour of the right length. Re-solve once with a few letters
        # revealed (as a real grid's crossings would), and keep it at the lower
        # "pattern" tier if that pins the answer.
        if not keep and getattr(args, "pattern_rescue", True):
            pat = reveal_pattern(wr.key)
            try:
                pguesses = ver.guesses(
                    prompts.VERIFY_SYSTEM,
                    prompts.verify_user_prompt(clue, len(wr.key), pat))
            except FatalAPIError:
                raise
            except Exception:  # noqa: BLE001 - transient; skip rescue
                pguesses = []
            prank, _ptier = _accept_rank(wr, pguesses)
            if prank and prank <= args.max_rank:
                keep, rank, tier = True, prank, "pattern"
                guesses = guesses + ["[pattern]"] + pguesses

        if args.debug_log:
            dbg_records.append({
                "answer": wr.display.upper(), "clue": clue,
                "difficulty": c.get("difficulty"), "style": c.get("style"),
                "rank": rank, "tier": tier, "guesses": guesses, "kept": keep,
            })
        if keep:
            out_records.append({
                "key": wr.key, "answer": wr.display.upper(), "length": len(wr.key),
                "clue": clue, "difficulty": c.get("difficulty"),
                "style": c.get("style"), "verify_rank": rank, "match": tier,
                "gen_model": gen.model, "verify_model": ver.model,
            })
    return out_records, dbg_records, attempted, dropped_long, None


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

    from .llm import FatalAPIError, make_generator, make_verifier  # lazy

    gen = make_generator(args.gen_provider)
    if args.gen_model:
        gen.model = args.gen_model
    ver = make_verifier(args.verify_provider)
    if args.verify_model:
        ver.model = args.verify_model
    print(f"generate: {gen.model}  |  verify: {ver.model}  |  workers: {args.workers}")

    done = _done_keys(args.out)
    todo = [w for w in words if w.key not in done]
    print(f"{len(done)} already done; {len(todo)} to process -> {args.out}")
    if not todo:
        return 0

    kept = attempted = dropped_long = 0
    dbg = open(args.debug_log, "a", encoding="utf-8") if args.debug_log else None
    fatal = None

    # Writes happen only here in the main thread, so no file lock is needed.
    with open(args.out, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_word, wr, gen, ver, args): wr for wr in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                recs, dbgs, att, dl, warn = fut.result()
            except FatalAPIError as e:
                fatal = e
                for f in futs:
                    f.cancel()
                break
            if warn:
                print("  !", warn)
            for rec in recs:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            out.flush()
            if dbg is not None and dbgs:
                for d in dbgs:
                    dbg.write(json.dumps(d, ensure_ascii=False) + "\n")
                dbg.flush()
            attempted += att
            dropped_long += dl
            if n % 10 == 0 or n == len(todo):
                print(f"  [{n}/{len(todo)}] kept={kept} attempted={attempted} "
                      f"pass={kept/max(attempted,1):.0%}")

    if dbg is not None:
        dbg.close()
    if fatal is not None:
        print(f"\nFATAL: API unusable -> {fatal}\nProgress saved to {args.out}.")
        return 1
    print(f"\nDone. Kept {kept}/{attempted} clues "
          f"({kept/max(attempted,1):.0%} round-trip pass rate); "
          f"dropped {dropped_long} for length.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tagalaba.synthesize")
    ap.add_argument("--limit", type=int, default=50, help="words to process")
    ap.add_argument("--clues-per-word", type=int, default=4)
    ap.add_argument("--max-rank", type=int, default=1,
                    help="keep clue if answer is within this guess rank (1=strict)")
    ap.add_argument("--max-words", type=int, default=12,
                    help="drop clues longer than this many words")
    ap.add_argument("--no-pattern-rescue", dest="pattern_rescue",
                    action="store_false",
                    help="disable the revealed-letter rescue re-solve")
    ap.add_argument("--debug-log", default=None,
                    help="write every attempt (clue, rank, guesses, kept) here")
    ap.add_argument("--workers", type=int, default=config.WORKERS,
                    help="words processed concurrently")
    ap.add_argument("--gen-provider", default=config.GEN_PROVIDER,
                    choices=["anthropic", "gemini"], help="clue generator backend")
    ap.add_argument("--gen-model", default=None, help="override generator model id")
    ap.add_argument("--verify-provider", default=config.VERIFY_PROVIDER,
                    choices=["anthropic", "gemini"], help="clue verifier backend")
    ap.add_argument("--verify-model", default=None, help="override verifier model id")
    ap.add_argument("--out", default=config.DEFAULT_OUT)
    ap.add_argument("--no-require-synonym", action="store_true",
                    help="also process words lacking synonyms/bugtong")
    ap.add_argument("--dry-run", action="store_true",
                    help="print prompts only; make no API calls")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
