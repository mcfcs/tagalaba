"""Configuration + .env loading (no hard dependency on python-dotenv)."""

from __future__ import annotations

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def _load_dotenv(path: str) -> None:
    """Minimal .env parser: set keys into os.environ if not already present."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


_load_dotenv(os.path.join(_REPO_ROOT, ".env"))


def _get(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


# Keys
ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY")
GEMINI_API_KEY = _get("GEMINI_API_KEY")

# Generation backend: "gemini" (cheapest + cross-provider verify; needs credits)
# or "anthropic" (Claude Sonnet gen; works with just the Anthropic key).
GEN_PROVIDER = _get("TAGALABA_GEN_PROVIDER", "gemini")

# Models (overridable via env). Generator and verifier are DIFFERENT models on
# purpose, so the round-trip solver never just re-reads its own intent.
GEN_MODEL = _get("TAGALABA_GEN_MODEL", "gemini-2.5-flash")          # if provider=gemini
ANTHROPIC_GEN_MODEL = _get("TAGALABA_ANTHROPIC_GEN_MODEL", "claude-sonnet-4-6")

# Verifier. Haiku proved too weak at Tagalog; Sonnet works but is pricey at
# scale. Default: a DIFFERENT Gemini model than the generator (competent +
# cheap + reasonably independent). Sonnet stays available for max rigor.
VERIFY_PROVIDER = _get("TAGALABA_VERIFY_PROVIDER", "gemini")
VERIFY_MODEL = _get("TAGALABA_VERIFY_MODEL", "claude-sonnet-4-6")   # if provider=anthropic
GEMINI_VERIFY_MODEL = _get("TAGALABA_GEMINI_VERIFY_MODEL", "gemini-3.5-flash")

# Concurrency: words processed in parallel (paid Gemini handles high RPM).
WORKERS = int(_get("TAGALABA_WORKERS", "8"))

# Words/min cap. Paid Gemini allows far more than the old free-tier 15; the real
# limit is per-word latency (gen + a few verify calls run sequentially).
RPM = int(_get("TAGALABA_RPM", "30"))

# Paths
DATA_DIR = os.path.join(_REPO_ROOT, "data")
DEFAULT_OUT = os.path.join(DATA_DIR, "synthetic_clues.jsonl")
