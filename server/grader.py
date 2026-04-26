"""
Terminal reward for the MedChain Finals environment.

Six-component formula:
    0.30 × network_service_level
  + 0.20 × critical_service_level
  + 0.20 × allocation_accuracy      ← the key new term
  + 0.15 × event_response
  + 0.10 × budget_efficiency
  + 0.05 × waste_control
  - justification_penalty (capped at 0.15)

allocation_accuracy is the average per-ward-round score from the simplified
surplus+stockout formula (not counterfactual):
    surplus_ratio  = max(0, allocated − true_need) / max(allocated, 1)
    stockout_flag  = 1 if consumed < true_need else 0
    shortage_pen   = priority_weight × stockout_flag
    surplus_pen    = surplus_ratio × (1 − priority_weight) × 0.5
    acc_score      = max(0, 1 − shortage_pen − surplus_pen)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Set

if TYPE_CHECKING:
    from .simulation import SimState
    from .tasks import TaskConfig


# ─── Public entry point ──────────────────────────────────────────────────────

def compute_reward(state: "SimState", task_config: "TaskConfig") -> float:
    """Compute the terminal episode reward in [0, 1]."""
    return compute_reward_breakdown(state, task_config)["score"]


def compute_reward_breakdown(state: "SimState", task_config: "TaskConfig") -> Dict[str, float]:
    """Same scoring formula as compute_reward, but returns every component
    so callers (eval, inference summaries, dashboards) can show where signal
    landed without recomputing.
    """
    w = _weights()

    components = {
        "network_sl":       _network_service_level(state),
        "critical_sl":      _critical_service_level(state, task_config),
        "alloc_acc":        _allocation_accuracy(state, task_config),
        "event_resp":       _event_response(state, task_config),
        "budget_eff":       _budget_efficiency(state, task_config),
        "waste_ctrl":       _waste_control(state),
        "audit_score":      _audit_score(state),
        "approval_score":   _approval_workflow_score(state),
        "tool_discovery":   _tool_discovery_score(state, task_config),
        "briefing_eff":     _briefing_efficiency(state),
    }

    incoherent = sum(1 for r in state.justification_log if not r.is_coherent)
    justif_pen = min(0.15, incoherent * 0.05)

    score = (
        w["service"]   * components["network_sl"]
        + w["critical"]  * components["critical_sl"]
        + w["alloc"]     * components["alloc_acc"]
        + w["event"]     * components["event_resp"]
        + w["budget"]    * components["budget_eff"]
        + w["waste"]     * components["waste_ctrl"]
        + w["audit"]     * components["audit_score"]
        + w["approval"]  * components["approval_score"]
        + w["discovery"] * components["tool_discovery"]
        + w["briefing"]  * components["briefing_eff"]
        - justif_pen
    )
    components["justif_pen"] = justif_pen
    components["score"] = max(0.0, min(1.0, score))
    return components


def _weights() -> Dict[str, float]:
    """Weights sum to 1.00 (not counting justification penalty)."""
    return {
        "service":   0.25,
        "critical":  0.18,
        "alloc":     0.18,
        "event":     0.12,
        "budget":    0.07,
        "waste":     0.04,
        "audit":     0.05,    # combined: challenge + evidence + escalation accuracy
        "approval":  0.05,    # finance workflow
        "discovery": 0.03,    # enterprise tool discovery
        "briefing":  0.03,    # context efficiency
    }


# ─── Components ──────────────────────────────────────────────────────────────

def _network_service_level(state: "SimState") -> float:
    total_need = 0.0
    total_consumed = 0.0
    for alloc in state.ward_allocation_log:
        if alloc.round_idx <= 0:
            continue
        total_need += alloc.true_need
        total_consumed += alloc.actual_consumed
    if total_need <= 1e-9:
        return 0.0
    return min(1.0, total_consumed / total_need)


def _critical_service_level(state: "SimState", task_config: "TaskConfig") -> float:
    critical_skus = {
        p.product_id for p in task_config.products if p.criticality == "CRITICAL"
    }
    total_need = 0.0
    total_consumed = 0.0
    for alloc in state.ward_allocation_log:
        if alloc.round_idx <= 0:
            continue
        if alloc.product_id not in critical_skus:
            continue
        if alloc.ward_id not in ("ward_icu", "ward_er"):
            continue
        total_need += alloc.true_need
        total_consumed += alloc.actual_consumed
    if total_need <= 1e-9:
        return 1.0   # no critical demand means no failure
    return min(1.0, total_consumed / total_need)


def _allocation_accuracy(state: "SimState", task_config: "TaskConfig") -> float:
    priority_by_ward = {w.ward_id: w.priority_weight for w in task_config.wards}
    scores: List[float] = []
    for alloc in state.ward_allocation_log:
        if alloc.round_idx <= 0:    # skip synthetic prior-round rows
            continue
        priority = priority_by_ward.get(alloc.ward_id, 0.5)
        allocated = max(alloc.allocated_qty, 1)
        surplus_ratio = max(0.0, alloc.allocated_qty - alloc.true_need) / allocated
        stockout = 1 if alloc.actual_consumed + 1e-6 < alloc.true_need else 0
        shortage_pen = priority * stockout
        surplus_pen = surplus_ratio * (1.0 - priority) * 0.5
        acc = max(0.0, 1.0 - shortage_pen - surplus_pen)
        scores.append(acc)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _event_response(state: "SimState", task_config: "TaskConfig") -> float:
    """Average over applicable event-response checks (0-4 checks)."""
    checks: List[float] = []
    event_types = {e.event_type for e in task_config.events}

    if "mci" in event_types or "mci_warning" in event_types:
        checks.append(1.0 if state.mci_prepositioned else 0.0)

    if "supplier_disruption" in event_types:
        checks.append(1.0 if state.supplier_switched else 0.0)

    if "product_recall" in event_types:
        if state.recall_quarantined_by_round is not None:
            # Find the recall event's round (approximate — trigger_day / round_length)
            recall_event = next(
                (e for e in task_config.events if e.event_type == "product_recall"),
                None,
            )
            if recall_event:
                expected_round = max(
                    1, recall_event.trigger_day // max(state.round_length_days, 1)
                )
                delay = max(0, state.recall_quarantined_by_round - expected_round)
                checks.append(max(0.0, 1.0 - 0.35 * delay))
            else:
                checks.append(1.0)
        else:
            checks.append(0.0)

    if "cold_chain_breach" in event_types:
        checks.append(1.0 if state.coldchain_replenished else 0.0)

    if not checks:
        return 1.0   # no applicable events → don't penalise
    return sum(checks) / len(checks)


def _budget_efficiency(state: "SimState", task_config: "TaskConfig") -> float:
    benchmark = task_config.benchmark_spend
    actual = state.total_spend
    if actual <= 1e-6:
        # If agent spent nothing but had real demand, don't reward that.
        need = sum(a.true_need for a in state.ward_allocation_log if a.round_idx > 0)
        return 0.2 if need > 0 else 1.0
    return min(1.0, benchmark / actual)


def _waste_control(state: "SimState") -> float:
    if state.total_spend <= 1e-6:
        return 1.0
    waste_fraction = min(1.0, state.total_wasted_value / state.total_spend)
    return max(0.0, 1.0 - waste_fraction)


# ─── Multi-actor / enterprise components ─────────────────────────────────────

def _audit_score(state: "SimState") -> float:
    """Combined audit-loop signal: evidence-use + escalation.

    Two sub-signals, each computed in [0, 1] and averaged with equal
    weight. Components missing entirely (e.g. agent never escalated)
    score 0; rewarded only when actually exercised correctly.

      evidence_use_rate  — fraction of disclosed evidence cited in rationale
      escalation_acc     — correct arbiter escalations
    """
    parts: list[float] = []

    # 1) Evidence-use rate — fraction of disclosed evidence rows that the
    #    agent cited in a rationale. Honest, defensible governance signal.
    ev = getattr(state, "evidence_log", None) or []
    if ev:
        cited = sum(1 for e in ev if e.used_in_allocation)
        parts.append(cited / len(ev))

    # 2) Escalation accuracy
    esc = getattr(state, "escalation_log", None) or []
    if esc:
        correct = sum(1 for e in esc if e.correct)
        # Penalise frivolous escalations harder than misses
        score = (correct - 0.5 * (len(esc) - correct)) / len(esc)
        parts.append(max(0.0, min(1.0, score)))

    if not parts:
        return 0.0
    return sum(parts) / len(parts)


def _approval_workflow_score(state: "SimState") -> float:
    """For each approval ticket the agent triggered, did it resolve cleanly?

    Approved-and-coherent and rejected-and-recovered both score 1.0.
    Pending-at-end-of-episode scores 0.0 (orphaned ticket).
    """
    log = getattr(state, "approval_log", None) or []
    pending = getattr(state, "pending_approvals", None) or {}

    total = len(log) + len(pending)
    if total == 0:
        return 1.0    # no approvals required → don't penalise

    correct = sum(1 for a in log if a.coherent and a.status == "approved")
    # Rejection counts as recovery if a later PO was placed for the same SKU
    rejected_recovered = 0
    for a in log:
        if a.status == "rejected":
            recovered = any(
                po.product_id == a.po.product_id
                and po.day_submitted >= a.po.day_submitted
                for po in state.pipeline_orders
            )
            if recovered:
                rejected_recovered += 1
    return (correct + rejected_recovered) / total


def _tool_discovery_score(state: "SimState", task_config: "TaskConfig") -> float:
    """Fraction of relevant systems the agent actually invoked."""
    relevant = set(task_config.relevant_systems)
    used = set(getattr(state, "systems_used", set()) or set())
    if not relevant:
        return 1.0
    return len(used & relevant) / len(relevant)


def _briefing_efficiency(state: "SimState") -> float:
    """1.0 if the agent used `get_round_briefing` exactly once per played round.
    Penalise zero-use (sequence-of-individual-calls anti-pattern) and over-use.
    """
    rounds_played = max(1, getattr(state, "round_idx", 0))
    briefings = getattr(state, "briefings_total", 0)
    if briefings == 0:
        return 0.0
    ratio = briefings / rounds_played
    if ratio <= 1.0:
        return ratio
    # Over-use: linearly penalised, floor at 0.3
    return max(0.3, 1.0 - 0.4 * (ratio - 1.0))


# ─── Justification grading (kept from Round 1) ────────────────────────────────

def grade_justification(reason: str, active_event_types: Set[str]) -> bool:
    """Deterministic keyword-based coherence check."""
    CRISIS_KEYWORDS: Dict[str, List[str]] = {
        "mci": ["mci", "mass casualty", "trauma", "incident", "accident",
                "emergency", "casualties", "blood", "critical patients"],
        "mci_warning": ["mci", "mass casualty", "trauma", "standby", "surge"],
        "supplier_disruption": ["disruption", "delay", "lead time", "supplier",
                                "shortage", "force majeure", "extended"],
        "product_recall": ["recall", "quarantine", "contamination", "lot",
                           "health authority", "batch", "defective", "compromised"],
        "cold_chain_breach": ["cold chain", "temperature", "breach",
                              "refriger", "spoilage", "compromised", "platelet"],
    }
    GENERIC_KEYWORDS = [
        "urgent", "critical", "shortage", "low stock",
        "stockout", "emergency", "insufficient", "surge",
    ]

    lower = reason.lower()

    if not active_event_types:
        return any(kw in lower for kw in GENERIC_KEYWORDS)

    for et in active_event_types:
        for kw in CRISIS_KEYWORDS.get(et, []):
            if kw in lower:
                return True
    return any(kw in lower for kw in GENERIC_KEYWORDS)
