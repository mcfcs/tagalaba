"""Thin clients: Gemini for generation, Claude (Haiku) for verification.

SDKs are imported lazily so the rest of the package (baseline, index) works
without them installed. Each call retries with exponential backoff so a free
Gemini key surviving the occasional 429 doesn't kill a long batch run.
"""

from __future__ import annotations

import json
import re
import time

from . import config


def extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object from model output."""
    if not text:
        return {}
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
    return {}


class FatalAPIError(RuntimeError):
    """Non-retryable: quota exhausted, no credits, or bad key."""


# Substrings that mean "retrying will never help" — stop immediately.
_FATAL_MARKERS = (
    "limit: 0", "depleted", "prepay", "billing", "quota exceeded for metric",
    "permission_denied", "unauthenticated", "api key not valid",
    "invalid_api_key", "authentication_error", "insufficient_quota",
)


def _is_fatal(e: Exception) -> bool:
    s = str(e).lower()
    return any(m in s for m in _FATAL_MARKERS)


def _retry(fn, *, tries: int = 4, base: float = 2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if _is_fatal(e):
                raise FatalAPIError(str(e)) from e
            last = e
            if i == tries - 1:
                break
            time.sleep(base * (2**i))
    raise last  # type: ignore[misc]


class GeminiGenerator:
    """Generates Tagalog clue candidates as structured JSON."""

    def __init__(self, model: str | None = None, temperature: float = 0.9) -> None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set (put it in .env).")
        from google import genai  # lazy
        from google.genai import types  # lazy

        self._types = types
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = model or config.GEN_MODEL
        self.temperature = temperature

    def generate(self, system: str, user: str) -> list[dict]:
        cfg = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            response_mime_type="application/json",
        )
        resp = _retry(
            lambda: self._client.models.generate_content(
                model=self.model, contents=user, config=cfg
            )
        )
        data = extract_json(getattr(resp, "text", "") or "")
        clues = data.get("clues", []) if isinstance(data, dict) else []
        return [c for c in clues if isinstance(c, dict) and c.get("clue")]


class AnthropicGenerator:
    """Generates Tagalog clue candidates with Claude (used when Gemini is
    unavailable). Assistant-prefilled with '{' to force a JSON object."""

    def __init__(self, model: str | None = None, temperature: float = 1.0) -> None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set (put it in .env).")
        import anthropic  # lazy

        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model = model or config.ANTHROPIC_GEN_MODEL
        self.temperature = temperature

    def generate(self, system: str, user: str) -> list[dict]:
        msg = _retry(
            lambda: self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": "{"},  # force JSON start
                ],
            )
        )
        text = "{" + "".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        )
        data = extract_json(text)
        clues = data.get("clues", []) if isinstance(data, dict) else []
        return [c for c in clues if isinstance(c, dict) and c.get("clue")]


def make_generator(provider: str | None = None):
    """Generator factory: 'anthropic' (default) or 'gemini'."""
    provider = (provider or config.GEN_PROVIDER).lower()
    if provider == "gemini":
        return GeminiGenerator()
    if provider == "anthropic":
        return AnthropicGenerator()
    raise ValueError(f"unknown gen provider: {provider!r}")


class ClaudeVerifier:
    """Solves a clue back to candidate answers (the round-trip check)."""

    def __init__(self, model: str | None = None) -> None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set (put it in .env).")
        import anthropic  # lazy

        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model = model or config.VERIFY_MODEL

    def guesses(self, system: str, user: str) -> list[str]:
        msg = _retry(
            lambda: self._client.messages.create(
                model=self.model,
                max_tokens=128,
                system=[  # cache the static system prompt (prompt caching)
                    {"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}
                ],
                messages=[{"role": "user", "content": user}],
            )
        )
        text = "".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        )
        data = extract_json(text)
        gs = data.get("guesses", []) if isinstance(data, dict) else []
        return [g for g in gs if isinstance(g, str)]
