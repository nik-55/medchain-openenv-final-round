"""
Ward actors — LLM-driven (with deterministic scripted fallback).

A WardActor encapsulates a single ward's behaviour:
  - propose_request(true_need, history) -> (requested_qty, justification, padded_flag)
  - respond_to_message(body, ctx)              -> text

Determinism contract
--------------------
- WARD_ACTOR_MODE=scripted          → fully deterministic (legacy behaviour)
- WARD_ACTOR_MODE=llm or unset      → uses llm_client if available; falls back
                                     to scripted on missing key / API errors
- All scripted code paths are seeded by (episode_seed, ward_id, round_idx, sku)

Ground-truth `true_need` and `padded_flag` are owned by the simulation. The
actor only chooses how to *frame* the request — RLVR signals (allocation
accuracy, escalation_acc) read state directly, not actor dialogue.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from . import llm_client
except ImportError:  # pragma: no cover
    llm_client = None  # type: ignore[assignment]

from .tasks import WardActorConfig, WardConfig

_log = logging.getLogger(__name__)


@dataclass
class ProposedRequest:
    requested_qty: int
    justification: str
    padded_flag: bool


def _mode() -> str:
    """Resolve effective mode: scripted (default) | llm (opt-in only).

    Wards are scripted by default. The LLM code paths remain in place but
    are only activated when WARD_ACTOR_MODE=llm is exported AND a client
    is available — never auto-enabled by mere presence of an API key.
    Reason: the audit/escalation mechanic is the canonical multi-actor
    surface; ward LLMs are kept as optional flavour for users who want
    fully-LLM dialogue.
    """
    explicit = os.getenv("WARD_ACTOR_MODE", "").lower().strip()
    if explicit == "llm" and llm_client is not None and llm_client.is_available():
        return "llm"
    return "scripted"


def _seeded_rng(*key: Any) -> np.random.Generator:
    h = abs(hash(tuple(str(k) for k in key))) % (2**32)
    return np.random.default_rng(h)


# ─── Public class ─────────────────────────────────────────────────────────────

class WardActor:
    """Stateless actor — per-round state lives in SimState.ward_actor_state."""

    def __init__(self, ward: WardConfig, actor_cfg: WardActorConfig):
        self.ward = ward
        self.cfg = actor_cfg

    # ── Propose this round's request ──────────────────────────────────────

    def propose_request(
        self,
        product_id: str,
        true_need: float,
        round_idx: int,
        episode_seed: int,
        recent_stockouts: int,
        reputation: float,
        active_event_summary: str,
        history_text: str,
    ) -> ProposedRequest:
        """Decide how much to request this round, and how to phrase it."""
        if _mode() == "llm":
            try:
                return self._propose_llm(
                    product_id, true_need, round_idx, episode_seed,
                    recent_stockouts, reputation, active_event_summary,
                    history_text,
                )
            except Exception as exc:  # pragma: no cover — runtime safety net
                _log.warning("[%s] LLM propose failed (%s); falling back to scripted",
                             self.ward.ward_id, exc)
        return self._propose_scripted(product_id, true_need, round_idx, episode_seed)

    # Scripted: matches the legacy logic in MedchainSimulation._generate_ward_request
    def _propose_scripted(
        self, product_id: str, true_need: float, round_idx: int, episode_seed: int,
    ) -> ProposedRequest:
        rng = _seeded_rng(episode_seed, self.ward.ward_id, round_idx, product_id)
        padded = bool(rng.random() < self.ward.pad_prob)
        if padded:
            mult = float(rng.uniform(self.ward.pad_lo, self.ward.pad_hi))
            requested = max(1, int(round(true_need * mult)))
            template = self.ward.padded_justifications[
                int(rng.integers(0, len(self.ward.padded_justifications)))
            ]
        else:
            requested = max(1, int(round(true_need * float(rng.uniform(0.95, 1.05)))))
            template = self.ward.honest_justifications[
                int(rng.integers(0, len(self.ward.honest_justifications)))
            ]
        # Re-evaluate padded_flag against the 1.10× threshold so downstream
        # RLVR signals (escalation_acc) compute consistently.
        padded_flag = requested > true_need * 1.10
        justification = template.replace("{sku}", product_id)
        return ProposedRequest(requested, justification, padded_flag)

    def _propose_llm(
        self,
        product_id: str,
        true_need: float,
        round_idx: int,
        episode_seed: int,
        recent_stockouts: int,
        reputation: float,
        active_event_summary: str,
        history_text: str,
    ) -> ProposedRequest:
        """LLM picks an inflation factor in [0.95, 1.60]. The simulation owns true_need."""
        assert llm_client is not None
        sys_prompt = (
            f"{self.cfg.persona}\n\n"
            "PRIVATE GOALS (the central coordinator does NOT see these):\n"
            + "\n".join(f"  - {g}" for g in self.cfg.private_goals)
            + "\n\nYou are submitting a supply request for the next 2-day round. "
              "Decide an inflation factor and a one-sentence justification. "
              "Respond ONLY with strict JSON: "
              '{"factor": 1.00, "justification": "..."}\n'
              "Constraints: 0.95 <= factor <= 1.60. "
              "Higher factors increase the chance the central coordinator escalates you to clinical review. "
              "Justification should be plausible and brief (under 30 words)."
        )
        user_prompt = (
            f"SKU: {product_id} | true need (private to you): {true_need:.1f}\n"
            f"Round {round_idx} | hoarding pressure: {self.cfg.hoarding_pressure_init:.2f}\n"
            f"Your reputation with central: {reputation:.2f}  (1.0 = trusted)\n"
            f"Your recent stockouts: {recent_stockouts}\n"
            f"Active events: {active_event_summary or 'none'}\n\n"
            f"Recent history:\n{history_text or '(none)'}"
        )
        text = llm_client.chat_text(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=llm_client.WARD_MODEL_NAME,
            temperature=0.6,
            max_tokens=200,
        )
        factor, justification = _parse_json_decision(text, default_factor=1.0)
        factor = max(0.95, min(1.60, factor))
        requested = max(1, int(round(true_need * factor)))
        padded_flag = requested > true_need * 1.10
        return ProposedRequest(requested, justification, padded_flag)

    # ── Outbound message reply ────────────────────────────────────────────

    def respond_to_message(self, body: str, context: Dict[str, Any]) -> str:
        if _mode() == "llm":
            try:
                return self._reply_llm(body, context)
            except Exception:  # pragma: no cover
                pass
        return self._reply_scripted(body, context)

    def _reply_scripted(self, body: str, context: Dict[str, Any]) -> str:
        return (
            f"[{self.ward.ward_id}] Acknowledged: {body[:60]}. "
            "Will incorporate into our next round planning."
        )

    def _reply_llm(self, body: str, context: Dict[str, Any]) -> str:
        assert llm_client is not None
        sys_prompt = (
            f"{self.cfg.persona}\n\n"
            "Reply to the central coordinator's message in 1-2 sentences. "
            "Be in-character. Do not include JSON or quotes — plain text only."
        )
        ctx_lines = "\n".join(f"  {k}: {v}" for k, v in context.items())
        user_prompt = f"Coordinator message:\n{body}\n\nContext:\n{ctx_lines}"
        text = llm_client.chat_text(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=llm_client.WARD_MODEL_NAME,
            temperature=0.5,
            max_tokens=180,
        )
        return f"[{self.ward.ward_id}] {text.strip()}"


# ─── JSON parsing helpers (LLMs are messy) ────────────────────────────────────

def _extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Try whole-string parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fallback: first {...} substring
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _parse_json_decision(text: str, default_factor: float = 1.0) -> Tuple[float, str]:
    obj = _extract_json_blob(text) or {}
    try:
        factor = float(obj.get("factor", default_factor))
    except (TypeError, ValueError):
        factor = default_factor
    justification = str(obj.get("justification", "")).strip()
    if not justification:
        justification = "Standard request based on projected consumption."
    return factor, justification[:240]


