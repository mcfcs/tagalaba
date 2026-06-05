"""CLI demo:  python -m tagalaba HINTO  [--english] [--k 5]
Or with no word, prints dataset coverage stats and a few examples.
"""

from __future__ import annotations

import argparse
import sys

from .baseline import generate
from .sources import get_index


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # py3.7+
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(prog="tagalaba")
    ap.add_argument("word", nargs="?", help="answer word to clue")
    ap.add_argument("--k", type=int, default=5, help="number of clues")
    ap.add_argument("--english", action="store_true", help="allow English glosses")
    args = ap.parse_args(argv)

    index = get_index()

    if not args.word:
        print("Loaded index:", len(index), "words")
        for k, v in index.coverage().items():
            print(f"  {k:22} {v:>8,}")
        print("\nTry:  python -m tagalaba HINTO")
        return 0

    cands = generate(args.word, top_k=args.k, allow_english=args.english, index=index)
    if not cands:
        print(f"No clue material found for {args.word!r}.")
        return 1

    print(f"Clues for {args.word.upper()}:")
    for i, c in enumerate(cands, 1):
        print(f"  {i}. [{c.source:8} {c.score:.2f} {c.language}] {c.clue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
