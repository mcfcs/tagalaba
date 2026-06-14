"""Diagnose whether synthesis is biased toward common (easy) words.

The round-trip filter keeps a clue only if a DIFFERENT model can solve it back
to the answer. For rare/hard words the verifier often doesn't know the word, so
good clues get dropped -- biasing the corpus toward common vocabulary. This
script quantifies that bias by running the real gen+verify pipeline on two
strata and comparing pass rates:

  COMMON : words that have a crossword synonym (what the default run uses)
  RARE   : words with only an English dictionary gloss, no synonym/bugtong
           (excluded by default unless --no-require-synonym)

Run:
    python -m tagalaba.diagnose --per-bucket 12        # ~24 words, a few cents
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor

from .sources import get_index, WordResources
from .synthesize import _process_word


def _utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _is_clean(wr: WordResources) -> bool:
    """A plausible crossword answer: 4-10 letters, alphabetic, not a fragment."""
    return 4 <= len(wr.key) <= 10 and wr.key.isalpha()


def _spread(words: list[WordResources], n: int) -> list[WordResources]:
    """Deterministic, alphabet-spread sample of n words (not just A-words)."""
    words = sorted(words, key=lambda w: w.key)
    if len(words) <= n:
        return words
    step = len(words) / n
    return [words[int(i * step)] for i in range(n)]


def _buckets(index, n: int) -> dict[str, list[WordResources]]:
    common, rare = [], []
    for wr in index._by_key.values():  # noqa: SLF001 - internal access by design
        if not _is_clean(wr):
            continue
        if wr.synonyms:
            common.append(wr)
        elif wr.glosses_en and not wr.bugtong and not wr.glosses_tl:
            rare.append(wr)
    return {"COMMON": _spread(common, n), "RARE": _spread(rare, n)}


def _p(rep, *args) -> None:
    """Print to stdout (flushed) AND collect the line for a results file."""
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    rep.append(line)


def run(args: argparse.Namespace) -> int:
    _utf8()
    from .llm import make_generator, make_verifier  # lazy

    index = get_index()
    buckets = _buckets(index, args.per_bucket)
    gen = make_generator(args.gen_provider)
    ver = make_verifier(args.verify_provider)
    rep: list[str] = []
    _p(rep, f"generate: {gen.model}  |  verify: {ver.model}  |  max_rank={args.max_rank}\n")

    proc_args = Namespace(clues_per_word=args.clues_per_word, max_words=12,
                          max_rank=args.max_rank, debug_log=True)

    for name, words in buckets.items():
        kept = attempted = 0
        samples_kept, samples_dropped = [], []
        _p(rep, f"===== {name} ({len(words)} words) =====")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(
                lambda wr: _process_word(wr, gen, ver, proc_args), words))
        for recs, dbgs, att, _dl, warn in results:
            if warn:
                _p(rep, "  !", warn)
            attempted += att
            kept += len(recs)
            for d in dbgs:
                (samples_kept if d["kept"] else samples_dropped).append(d)
        pass_rate = kept / max(attempted, 1)
        _p(rep, f"  -> kept {kept}/{attempted}  ({pass_rate:.0%} round-trip pass rate)")
        for d in samples_kept[:3]:
            _p(rep, f"     KEPT  {d['answer']:12} {d['clue']}")
        for d in samples_dropped[:3]:
            _p(rep, f"     DROP  {d['answer']:12} {d['clue']}")
            _p(rep, f"           solver guessed: {', '.join(d['guesses'][:5])}")
        _p(rep, "")
    _p(rep, "If RARE's pass rate is much lower than COMMON's, the synthesis filter "
            "is biasing your corpus toward easy words -- mitigate before the full run.")
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print(f"\n[report written to {args.report}]", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tagalaba.diagnose")
    ap.add_argument("--per-bucket", type=int, default=12, help="words per stratum")
    ap.add_argument("--clues-per-word", type=int, default=4)
    ap.add_argument("--max-rank", type=int, default=2)
    ap.add_argument("--gen-provider", default=None, choices=[None, "anthropic", "gemini"])
    ap.add_argument("--verify-provider", default=None, choices=[None, "anthropic", "gemini"])
    ap.add_argument("--report", default="data/diagnose_report.txt")
    ap.add_argument("--workers", type=int, default=8)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
