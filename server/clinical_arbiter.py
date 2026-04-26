"""
Clinical Review Board — single LLM-backed arbiter for escalated requests.

The agent (central coordinator) can escalate a suspicious ward request via
``escalate_to_clinical_review``. This module's ``review_request`` function
returns a binding verdict:

    APPROVE — request stands at original quantity
    REDUCE  — request locked at a lower recommended_qty
    DENY    — request locked at minimum (true_need * 0.95) — strict cut

The arbiter has read access to ground-truth ``true_need`` and ward
allocation history; it is *not* the agent. Think of it as an internal
hospital governance committee whose ruling the supply coordinator
defers to.

Determinism contract
--------------------
- ARBITER_MODE=scripted (default)            → fully deterministic
- ARBITER_MODE=llm  + key available           → LLM-backed verdict
- LLM verdict is sanity-checked against scripted bounds; out-of-range
  decisions are clamped so RLVR scoring stays well-defined.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from . import llm_client
except ImportError:  # pragma: no cover
    llm_client = None  # type: ignore[assignment]

_log = logging.getLogger(__name__)


@dataclass
class ArbiterVerdict:
    verdict: str                  # APPROVE | REDUCE | DENY
    recommended_qty: int
    reason: str


def _mode() -> str:
    explicit = os.getenv("ARBITER_MODE", "").lower().strip()
    if explicit == "llm" and llm_client is not None and llm_client.is_available():
        return "llm"
    return "scripted"


# ─── Public entry point ──────────────────────────────────────────────────────

def review_request(
    *,
    ward_id: str,
    product_id: str,
    requested_qty: int,
    true_need: float,
    padded_flag: bool,
    concern: str,
    recent_history: List[Tuple[int, int, float, bool]],   # (round, alloc, consumed, stockout)
    active_event_summary: str,
    ward_priority: float,
) -> ArbiterVerdict:
    """Single call site. Always returns a valid verdict (never raises)."""
    if _mode() == "llm":
        try:
            v = _review_llm(
                ward_id=ward_id, product_id=product_id, requested_qty=requested_qty,
                true_need=true_need, padded_flag=padded_flag, concern=concern,
                recent_history=recent_history,
                active_event_summary=active_event_summary,
                ward_priority=ward_priority,
            )
            return _clamp_verdict(v, requested_qty, true_need)
        except Exception as exc:  # pragma: no cover
            _log.warning("Arbiter LLM failed (%s); using scripted verdict", exc)

    return _review_scripted(
        ward_id=ward_id, requested_qty=requested_qty,
        true_need=true_need, padded_flag=padded_flag,
        active_event_summary=active_event_summary, ward_priority=ward_priority,
    )


# ─── Scripted (always-available) ─────────────────────────────────────────────

def _review_scripted(
    *,
    ward_id: str,
    requested_qty: int,
    true_need: float,
    padded_flag: bool,
    active_event_summary: str,
    ward_priority: float,
) -> ArbiterVerdict:
    """Deterministic ruling based on the simulation's ground truth.

    The scripted arbiter is *correct by construction*: if the ward is
    padding, it cuts; if not, it approves. This is the right behaviour —
    the simulation is the ground truth and the arbiter has read access.
    """
    surge_active = "mci" in active_event_summary.lower()
    if not padded_flag:
        return ArbiterVerdict(
            verdict="APPROVE",
            recommended_qty=requested_qty,
            reason=(
                f"Reviewed {ward_id} request — projected consumption matches "
                f"requested volume. No evidence of inflation. "
                + ("MCI surge supports elevated volume." if surge_active else "")
            ).strip(),
        )

    # Padded: pick REDUCE vs DENY based on severity
    severity = (requested_qty - true_need) / max(true_need, 1.0)
    if severity > 0.35:
        recommended = max(1, int(round(true_need * 0.95)))
        verdict = "DENY"
        reason = (
            f"Request {requested_qty} substantially exceeds projected need "
            f"~{true_need:.0f}. Cut to {recommended} (denial reduces buffer "
            f"by ≥35%). Document census before next request."
        )
    else:
        recommended = max(1, int(round(true_need * 1.05)))
        verdict = "REDUCE"
        reason = (
            f"Request {requested_qty} modestly exceeds projected need "
            f"~{true_need:.0f}. Reduced to {recommended} (5% safety margin). "
            f"{'MCI provides partial justification.' if surge_active else ''}"
        ).strip()
    return ArbiterVerdict(verdict=verdict, recommended_qty=recommended, reason=reason)


# ─── LLM-backed (opt-in) ─────────────────────────────────────────────────────

def _review_llm(
    *,
    ward_id: str,
    product_id: str,
    requested_qty: int,
    true_need: float,
    padded_flag: bool,
    concern: str,
    recent_history: List[Tuple[int, int, float, bool]],
    active_event_summary: str,
    ward_priority: float,
) -> ArbiterVerdict:
    assert llm_client is not None
    history_text = (
        "; ".join(
            f"r{r}: alloc={a} cons={c:.1f}{'(SO)' if so else ''}"
            for (r, a, c, so) in recent_history[-6:]
        )
        or "(no prior history)"
    )
    sys_prompt = (
        "You are the Hospital Supply Committee — a clinical-review board that "
        "rules on disputed supply requests. You have read access to the ward's "
        "actual projected consumption (true_need) and allocation history. "
        "You are NOT the supply coordinator and you are NOT the ward — you are "
        "an independent governance body. Apply strict, evidence-based judgement.\n\n"
        "Output STRICT JSON only:\n"
        '  {"verdict": "APPROVE"|"REDUCE"|"DENY", '
        '"recommended_qty": <int>, "reason": "<one-sentence rationale>"}\n\n'
        "Rules:\n"
        "  - APPROVE if requested_qty is within ~10% of true_need.\n"
        "  - REDUCE (5-35% above true_need) — set recommended_qty to true_need × 1.05.\n"
        "  - DENY (>35% above true_need) — set recommended_qty to true_need × 0.95.\n"
        "  - When MCI/recall/disruption is active, lean toward APPROVE for critical "
        "    SKUs at high-priority wards (priority ≥ 0.7).\n"
        "  - Never recommend a qty greater than requested_qty."
    )
    user_prompt = (
        f"Ward: {ward_id}  priority: {ward_priority:.1f}\n"
        f"SKU: {product_id}\n"
        f"Requested: {requested_qty}\n"
        f"True need (private to you): {true_need:.1f}\n"
        f"Coordinator's concern: {concern}\n"
        f"Active events: {active_event_summary or 'none'}\n"
        f"Recent history: {history_text}"
    )
    text = llm_client.chat_text(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=300,
    )
    obj = _extract_json(text) or {}
    verdict = str(obj.get("verdict", "APPROVE")).upper().strip()
    if verdict not in ("APPROVE", "REDUCE", "DENY"):
        verdict = "APPROVE"
    try:
        recommended = int(obj.get("recommended_qty", requested_qty))
    except (TypeError, ValueError):
        recommended = requested_qty
    reason = str(obj.get("reason", "")).strip()[:280]
    if not reason:
        reason = "Committee reviewed available evidence and issued the above verdict."
    return ArbiterVerdict(verdict=verdict, recommended_qty=recommended, reason=reason)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _clamp_verdict(v: ArbiterVerdict, requested_qty: int, true_need: float) -> ArbiterVerdict:
    """Sanity-check the LLM's recommended_qty so RLVR stays well-defined."""
    floor = max(1, int(round(true_need * 0.5)))
    ceil_qty = requested_qty
    rec = max(floor, min(ceil_qty, v.recommended_qty))
    return ArbiterVerdict(verdict=v.verdict, recommended_qty=rec, reason=v.reason)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        try:
            obj = json.loads(text[s:e + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None
