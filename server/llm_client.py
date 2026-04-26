"""
Shared OpenAI-compatible LLM client.

Used by:
  - inference.py            (drives the central-coordinator agent)
  - server/ward_actor.py    (LLM-driven ward actors)

Single source of truth for endpoint, key, and model selection so the two
call sites never drift. Configurable entirely via env vars:

    API_BASE_URL  default https://router.huggingface.co/v1
    HF_TOKEN / API_KEY    auth (either)
    MODEL_NAME / MODEL    primary model id

Ward actors may use a separate (typically cheaper/faster) model:
    WARD_MODEL_NAME       default = MODEL_NAME

If no key is found, ``is_available()`` returns False — call sites should
fall back to deterministic scripted behaviour.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

try:
    from openai import BadRequestError, OpenAI, RateLimitError
except ImportError:  # pragma: no cover — server may run without openai during training
    OpenAI = None  # type: ignore[assignment]
    RateLimitError = Exception  # type: ignore[assignment,misc]
    BadRequestError = Exception  # type: ignore[assignment,misc]

_log = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL", "openai/gpt-oss-20b:groq")
WARD_MODEL_NAME = os.getenv("WARD_MODEL_NAME", MODEL_NAME)


def is_available() -> bool:
    """True iff we have both the SDK and an API key."""
    return OpenAI is not None and bool(API_KEY)


_client_singleton: Optional["OpenAI"] = None


def get_client() -> "OpenAI":
    """Lazy singleton — saves the per-process socket pool."""
    global _client_singleton
    if not is_available():
        raise RuntimeError(
            "LLM client unavailable: set HF_TOKEN or API_KEY (and `pip install openai`)."
        )
    if _client_singleton is None:
        _client_singleton = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return _client_singleton


def chat_completion(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    parallel_tool_calls: bool = True,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    max_retries: int = 4,
    backoff_base: float = 2.0,
) -> Any:
    """
    Single chat-completion call with rate-limit backoff. Returns the raw
    OpenAI ``ChatCompletion`` response so callers can inspect tool_calls,
    finish_reason, etc.
    """
    client = get_client()
    selected_model = model or MODEL_NAME

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice or "auto"
                kwargs["parallel_tool_calls"] = parallel_tool_calls
            return client.chat.completions.create(**kwargs)

        except RateLimitError as exc:
            last_exc = exc
            wait = min(30.0, backoff_base ** attempt)
            _log.warning("LLM rate-limit (attempt %d/%d) — sleeping %.1fs",
                         attempt + 1, max_retries, wait)
            time.sleep(wait)

        except BadRequestError as exc:
            # Don't retry — caller's request is malformed.
            _log.warning("LLM bad request: %s", exc)
            raise

    assert last_exc is not None
    raise last_exc


def chat_text(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """Convenience wrapper that returns assistant text only (no tools)."""
    resp = chat_completion(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()
