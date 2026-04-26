"""
MedChain Finals — simulation engine.

Manages ward actors (ICU, ER, General) and the central-agent coordination
loop. Each round spans `round_length_days` simulated days. The agent acts at
round boundaries, allocates stock across wards, and calls advance_round to
move the simulation forward. Ward actors are scripted (not trained).

Core mechanics preserved from Round 1: FEFO lot-based inventory, supplier
lead times with jitter, lot expiry, event-driven inbox, justification flow.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .tasks import (
    Product,
    SimEvent,
    TaskConfig,
    WardConfig,
)
from .ward_actor import ProposedRequest, WardActor


# ─── Reused low-level dataclasses ─────────────────────────────────────────────

@dataclass
class Lot:
    lot_id: str
    qty: int
    expiry_day: Optional[int]
    cost_per_unit: float


@dataclass
class PurchaseOrder:
    po_id: str
    supplier_id: str
    product_id: str
    destination_id: str
    quantity: int
    priority: str
    day_submitted: int
    eta_day: int
    unit_cost: float
    total_cost: float
    status: str                      # "pending_justification", "in_transit", "delivered"
    lot_id: str


@dataclass
class PendingBudgetOverride:
    ticket_id: str
    po: PurchaseOrder


@dataclass
class InboxMessage:
    msg_id: str
    priority: str
    timestamp_str: str
    sender: str
    subject: str
    body: str
    read: bool
    flagged: bool
    event_id: str


@dataclass
class JustificationRecord:
    ticket_id: str
    po_id: str
    reason: str
    is_coherent: bool


# ─── Ward-round tracking ─────────────────────────────────────────────────────

@dataclass
class WardRequest:
    round_idx: int
    ward_id: str
    product_id: str
    day: int                         # day the request opens
    true_need: float                 # HIDDEN from agent: actual total need over round
    requested_qty: int               # visible — may be padded
    justification: str               # visible
    padded_flag: bool                # HIDDEN


@dataclass
class WardAllocation:
    round_idx: int
    ward_id: str
    product_id: str
    allocated_qty: int
    true_need: float                 # copied from request at allocation time
    actual_consumed: float = 0.0
    stockout_flag: bool = False
    resolved: bool = False


# ─── Multi-actor / enterprise additions ──────────────────────────────────────

@dataclass
class PendingApproval:
    approval_id: str
    po: PurchaseOrder
    submitted_round: int
    justification: str = ""
    status: str = "pending"          # pending | approved | rejected
    coherent: bool = False           # set when justification is filed


@dataclass
class PendingQuote:
    quote_id: str
    supplier_id: str
    product_id: str
    quantity: int
    submitted_day: int
    resolves_day: int
    lead_time: int                   # cached at request time
    unit_cost: float
    total_cost: float
    fulfilled: bool = False


@dataclass
class EvidenceDisclosure:
    round_idx: int
    ward_id: str
    product_id: str
    evidence_type: str               # census | acuity | recent_actuals | events
    disclosed: Dict[str, Any]        # what the ward returned
    full_disclosure: bool            # did the ward redact based on hoarding?
    used_in_allocation: bool = False # set when allocation_rationale references it


@dataclass
class EscalationRecord:
    round_idx: int
    ward_id: str
    product_id: str
    concern: str
    was_padded: bool                 # ground truth at escalation time
    original_qty: int
    recommended_qty: int             # arbiter's verdict
    verdict: str                     # APPROVE | REDUCE | DENY
    reason: str                      # arbiter's rationale
    correct: bool                    # T if (padded & verdict!=APPROVE) or (honest & verdict==APPROVE)


@dataclass
class AllocationRationale:
    round_idx: int
    ward_id: str
    text: str
    references_evidence: bool        # did it cite a disclosed evidence type?


# ─── SimState ─────────────────────────────────────────────────────────────────

@dataclass
class SimState:
    # Episode meta
    task: str
    episode_id: str
    seed: int
    rng: np.random.Generator

    # Time
    day: int
    max_days: int

    # Rounds
    round_idx: int
    max_rounds: int
    round_length_days: int
    current_round_trigger: str       # descriptive tag for the agent

    # Budget
    budget_used: float
    budget_limit: float

    # Inventory: (location_id, product_id) -> List[Lot]  (FEFO-sorted)
    inventory: Dict[Tuple[str, str], List[Lot]]

    # Orders & inbox
    pipeline_orders: List[PurchaseOrder]
    po_counter: int
    inbox: List[InboxMessage]
    msg_counter: int
    pending_overrides: Dict[str, PendingBudgetOverride]

    # Quarantine
    quarantined_lots: Set[str]

    # Ward tracking
    pending_requests: List[WardRequest]        # awaiting allocation this round
    ward_request_log: List[WardRequest]        # all requests (includes synthetic)
    ward_allocation_log: List[WardAllocation]  # all allocations (includes synthetic)

    # Active event effects: event_id -> last_day_active (inclusive)
    active_events: Dict[str, int]

    # Per-round shaping reward flags
    shaping_flags_this_round: Set[str]

    # Round/event response tracking
    mci_prepositioned: bool
    supplier_switched: bool
    recall_quarantined_by_round: Optional[int]
    coldchain_replenished: bool

    # Spend tracking
    total_spend: float
    total_wasted_value: float

    # Justification log
    justification_log: List[JustificationRecord]

    # ER surge state (one value per round)
    er_surge_state: float

    # ── Multi-actor / enterprise (Tier-1 upgrades) ──
    # Per-ward actor state: {ward_id: {reputation, recent_stockouts, hoarding_pressure}}
    ward_actor_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Finance approval queue
    pending_approvals: Dict[str, PendingApproval] = field(default_factory=dict)
    approval_log: List[PendingApproval] = field(default_factory=list)

    # Supplier-portal async quotes
    pending_quotes: Dict[str, PendingQuote] = field(default_factory=dict)
    quote_counter: int = 1
    approval_counter: int = 1

    # ERP-Oracle stale snapshot (1-round-old inventory totals)
    inventory_snapshot_prev_round: Dict[Tuple[str, str], int] = field(default_factory=dict)

    # Tool-discovery tracking
    systems_used: Set[str] = field(default_factory=set)
    briefing_calls_this_round: int = 0
    briefings_total: int = 0

    # Outbound messaging log
    outbound_messages: List[Dict[str, str]] = field(default_factory=list)

    # ── Audit / evidence / escalation ──
    evidence_log: List[EvidenceDisclosure] = field(default_factory=list)
    escalation_log: List[EscalationRecord] = field(default_factory=list)
    rationale_log: List[AllocationRationale] = field(default_factory=list)


# ─── Simulation ───────────────────────────────────────────────────────────────

class MedchainSimulation:
    """
    Central simulation engine. Tool implementations are called by the outer
    MedchainEnvironment via MCP. The main loop is:

        reset(seed)               → _open_round() → round 1 brief
        ... agent calls tools ... → submit_allocation_plan, optional submit_po
        advance_round()           → run round_length_days of sim → either
                                    terminal summary, or _open_round() again
    """

    def __init__(self, task_config: TaskConfig):
        self._task = task_config
        self._state: Optional[SimState] = None
        self._last_reward: float = 0.0
        self._done: bool = False
        self._actors: Dict[str, WardActor] = self._build_actors(task_config)

    @staticmethod
    def _build_actors(task_config: TaskConfig) -> Dict[str, WardActor]:
        out: Dict[str, WardActor] = {}
        for ward in task_config.wards:
            actor_cfg = task_config.ward_actor_configs.get(ward.ward_id)
            if actor_cfg is not None:
                out[ward.ward_id] = WardActor(ward, actor_cfg)
        return out

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self, seed: int, episode_id: str,
              difficulty: Optional[str] = None) -> str:
        """Initialise a new episode. Returns opening round brief text."""
        self._done = False
        self._last_reward = 0.0

        # Rebuild task config from seed (so that a fresh seed gives fresh events)
        from .tasks import make_task_config
        diff = difficulty or self._task.difficulty
        self._task = make_task_config(seed=seed, difficulty=diff)
        self._actors = self._build_actors(self._task)

        rng = np.random.default_rng(seed)
        self._state = SimState(
            task=self._task.name,
            episode_id=episode_id,
            seed=seed,
            rng=rng,
            day=1,
            max_days=self._task.max_days,
            round_idx=0,
            max_rounds=self._task.max_rounds,
            round_length_days=self._task.round_length_days,
            current_round_trigger="episode_start",
            budget_used=0.0,
            budget_limit=self._task.budget_limit,
            inventory={},
            pipeline_orders=[],
            po_counter=1,
            inbox=[],
            msg_counter=1,
            pending_overrides={},
            quarantined_lots=set(),
            pending_requests=[],
            ward_request_log=[],
            ward_allocation_log=[],
            active_events={},
            shaping_flags_this_round=set(),
            mci_prepositioned=False,
            supplier_switched=False,
            recall_quarantined_by_round=None,
            coldchain_replenished=False,
            total_spend=0.0,
            total_wasted_value=0.0,
            justification_log=[],
            er_surge_state=1.0,
            ward_actor_state={
                w.ward_id: {
                    "reputation": 0.5,
                    "recent_stockouts": 0,
                    "hoarding_pressure": (
                        self._task.ward_actor_configs.get(w.ward_id).hoarding_pressure_init
                        if w.ward_id in self._task.ward_actor_configs else 0.3
                    ),
                    "challenges_received": 0,
                }
                for w in self._task.wards
            },
        )

        self._initialize_inventory()
        self._inject_welcome_inbox()
        # Inject any day-1 events (warning messages / trigger_day == 1)
        for event in self._task.events:
            if event.trigger_day == 1:
                self._inject_event(event, 1)
                if event.event_type == "cold_chain_breach":
                    self._apply_cold_chain_breach(event)
                if event.event_type == "product_recall":
                    self._inject_recall_lots(event, 1)
            if event.warning_message and event.trigger_day - 1 == 1:
                self._inject_warning(event, 1)
        self._update_active_events(1)
        self._generate_synthetic_history()

        return self._open_round("episode_start")

    def _initialize_inventory(self):
        """Seed ward + central inventory per-SKU."""
        state = self._state
        for product in self._task.products:
            stock_days = self._task.initial_stock_days.get(product.product_id, 3.0)

            # Ward-local stock: per-ward base_demand × stock_days
            for loc_id in product.locations:
                qty = max(1, int(product.base_demand * stock_days))
                expiry = (
                    state.day + int(product.shelf_life_days * 0.7)
                    if product.shelf_life_days else None
                )
                lot = Lot(
                    lot_id=f"INIT-{product.product_id}-{loc_id}",
                    qty=qty, expiry_day=expiry, cost_per_unit=product.unit_cost,
                )
                state.inventory.setdefault((loc_id, product.product_id), []).append(lot)

            # Central pharmacy: ~4× total ward stock (replenishment reserve)
            central_qty = int(product.base_demand * len(product.locations) * stock_days * 1.5)
            if central_qty > 0:
                central_lot = Lot(
                    lot_id=f"INIT-{product.product_id}-central",
                    qty=central_qty,
                    expiry_day=(state.day + int((product.shelf_life_days or 365) * 0.8))
                    if product.shelf_life_days else None,
                    cost_per_unit=product.unit_cost,
                )
                state.inventory.setdefault(
                    ("central_pharmacy", product.product_id), []
                ).append(central_lot)

    def _inject_welcome_inbox(self):
        state = self._state
        welcome = InboxMessage(
            msg_id=f"MSG-{state.msg_counter:04d}",
            priority="LOW",
            timestamp_str="Day 1 08:00",
            sender="System",
            subject="Shift Handover",
            body=(
                "Welcome to the central supply coordinator role.\n"
                f"Episode runs up to {state.max_rounds} rounds (each round = "
                f"{state.round_length_days} sim days).\n"
                "Three wards (ICU, ER, General) will submit requests each round.\n"
                "Use view_requests, query_ward_history, and read_inbox for context.\n"
                "Submit one allocation plan per round, then call advance_round."
            ),
            read=False,
            flagged=False,
            event_id="system_welcome",
        )
        state.inbox.append(welcome)
        state.msg_counter += 1

    # ── Synthetic history ────────────────────────────────────────────────

    def _generate_synthetic_history(self):
        """
        Pre-seed query_ward_history with H rounds of plausible prior activity.
        Uses a deterministic sub-RNG per ward.
        """
        state = self._state
        H = self._task.synthetic_history_rounds
        for h in range(H):
            synth_round = -(H - h)   # negative round indices: -H, ..., -1
            for ward in self._task.wards:
                ward_rng = np.random.default_rng(state.seed ^ abs(hash(ward.ward_id)))
                # Advance RNG by h to desynchronise across rounds
                ward_rng.bit_generator.advance(h * 16)
                for product_id in ward.products_tracked:
                    product = self._product(product_id)
                    if product is None:
                        continue
                    # true_need = base × days_per_round × small noise
                    true_need = (
                        product.base_demand
                        * state.round_length_days
                        * ward_rng.uniform(0.85, 1.15)
                    )
                    padded = ward_rng.random() < ward.pad_prob
                    if padded:
                        requested = int(round(
                            true_need * ward_rng.uniform(ward.pad_lo, ward.pad_hi)
                        ))
                    else:
                        requested = int(round(true_need))

                    # Synthetic allocation: match true_need most of the time,
                    # but occasionally under-allocate to create a stockout history
                    stockout_prob = 0.15 if ward.ward_id == "ward_general" else 0.08
                    force_stockout = ward_rng.random() < stockout_prob
                    if force_stockout:
                        allocated = int(round(true_need * ward_rng.uniform(0.5, 0.85)))
                        consumed = float(allocated)
                        stockout = True
                    else:
                        allocated = max(requested, int(round(true_need)))
                        consumed = float(min(allocated, true_need))
                        stockout = False

                    req = WardRequest(
                        round_idx=synth_round,
                        ward_id=ward.ward_id,
                        product_id=product_id,
                        day=-1,
                        true_need=true_need,
                        requested_qty=max(1, requested),
                        justification="[synthetic prior-round record]",
                        padded_flag=padded,
                    )
                    alloc = WardAllocation(
                        round_idx=synth_round,
                        ward_id=ward.ward_id,
                        product_id=product_id,
                        allocated_qty=max(1, allocated),
                        true_need=true_need,
                        actual_consumed=consumed,
                        stockout_flag=stockout,
                        resolved=True,
                    )
                    state.ward_request_log.append(req)
                    state.ward_allocation_log.append(alloc)

    # ── Round opening ────────────────────────────────────────────────────

    def _open_round(self, trigger: str = "scheduled") -> str:
        """Open a new round: generate requests, reset per-round flags."""
        state = self._state
        # Capture stale ERP snapshot BEFORE generating new requests, so
        # erp_oracle_get_inventory always reflects the previous round.
        if state.round_idx >= 1:
            self._snapshot_inventory_for_oracle()

        state.round_idx += 1
        state.current_round_trigger = trigger
        state.shaping_flags_this_round = set()
        state.pending_requests = []
        state.briefing_calls_this_round = 0
        # Decay reputations toward 0.5
        for ward_id, rep_state in state.ward_actor_state.items():
            actor_cfg = self._task.ward_actor_configs.get(ward_id)
            decay = actor_cfg.reputation_decay if actor_cfg else 0.85
            rep_state["reputation"] = (
                decay * rep_state["reputation"] + (1.0 - decay) * 0.5
            )

        # Draw ER surge state for this round
        if self._mci_active(state.day):
            state.er_surge_state = float(state.rng.uniform(2.5, 3.0))
        else:
            er_ward = self._ward("ward_er")
            if er_ward and state.rng.random() < er_ward.spike_prob:
                state.er_surge_state = float(er_ward.spike_multiplier)
            else:
                state.er_surge_state = 1.0

        # Generate one WardRequest per ward × tracked product
        for ward in self._task.wards:
            for product_id in ward.products_tracked:
                req = self._generate_ward_request(ward, product_id)
                state.pending_requests.append(req)
                state.ward_request_log.append(req)

        return self._format_round_brief()

    def _generate_ward_request(self, ward: WardConfig, product_id: str) -> WardRequest:
        state = self._state
        product = self._product(product_id)
        if product is None:
            # shouldn't happen — return empty stub
            return WardRequest(
                round_idx=state.round_idx, ward_id=ward.ward_id,
                product_id=product_id, day=state.day,
                true_need=0.0, requested_qty=0,
                justification="(unknown SKU)", padded_flag=False,
            )

        # ── 1. Compute true_need (sim-owned, ground truth) ─────────────────
        base_need = product.base_demand * state.round_length_days
        if ward.ward_id == "ward_er":
            base_need *= state.er_surge_state
        if ward.ward_id in ("ward_icu", "ward_er"):
            for event_id in state.active_events:
                event = self._event(event_id)
                if event and event.event_type == "mci" and \
                        product.criticality in ("CRITICAL", "HIGH") and \
                        ward.ward_id in event.params.get("locations", []):
                    base_need *= event.params.get("demand_multiplier", 2.8)
        noise = float(state.rng.normal(1.0, 0.1))
        true_need = max(0.5, base_need * noise)

        # ── 2. Ask the actor how to frame it ───────────────────────────────
        actor = self._actors.get(ward.ward_id)
        if actor is None:
            # Legacy fallback path (no actor configured) — keep old logic verbatim
            return self._legacy_request(ward, product_id, true_need)

        rep_state = state.ward_actor_state.get(ward.ward_id, {})
        proposal: ProposedRequest = actor.propose_request(
            product_id=product_id,
            true_need=true_need,
            round_idx=state.round_idx,
            episode_seed=state.seed,
            recent_stockouts=int(rep_state.get("recent_stockouts", 0)),
            reputation=float(rep_state.get("reputation", 0.5)),
            active_event_summary=", ".join(
                self._event(eid).event_type
                for eid in state.active_events
                if self._event(eid)
            ) or "none",
            history_text=self._compact_history_for_actor(ward.ward_id, product_id),
        )

        return WardRequest(
            round_idx=state.round_idx,
            ward_id=ward.ward_id,
            product_id=product_id,
            day=state.day,
            true_need=true_need,
            requested_qty=proposal.requested_qty,
            justification=proposal.justification,
            padded_flag=proposal.padded_flag,
        )

    def _legacy_request(self, ward: WardConfig, product_id: str, true_need: float) -> WardRequest:
        """Pre-actor scripted path. Kept as a guaranteed fallback."""
        state = self._state
        padded = bool(state.rng.random() < ward.pad_prob)
        if padded:
            mult = float(state.rng.uniform(ward.pad_lo, ward.pad_hi))
            requested = max(1, int(round(true_need * mult)))
            template = ward.padded_justifications[
                int(state.rng.integers(0, len(ward.padded_justifications)))
            ]
        else:
            requested = max(1, int(round(true_need * float(state.rng.uniform(0.95, 1.05)))))
            template = ward.honest_justifications[
                int(state.rng.integers(0, len(ward.honest_justifications)))
            ]
        return WardRequest(
            round_idx=state.round_idx,
            ward_id=ward.ward_id,
            product_id=product_id,
            day=state.day,
            true_need=true_need,
            requested_qty=requested,
            justification=template.replace("{sku}", product_id),
            padded_flag=requested > true_need * 1.10,
        )

    def _compact_history_for_actor(self, ward_id: str, product_id: str, n: int = 3) -> str:
        state = self._state
        rows = [
            (a.round_idx, a.actual_consumed, a.allocated_qty, a.stockout_flag)
            for a in state.ward_allocation_log
            if a.ward_id == ward_id and a.product_id == product_id
            and a.round_idx > 0
        ][-n:]
        if not rows:
            return ""
        return "; ".join(
            f"r{r[0]}: alloc={r[2]} consumed={r[1]:.1f}{'(STOCKOUT)' if r[3] else ''}"
            for r in rows
        )

    @staticmethod
    def _er_event_tpls() -> List[str]:
        return [
            "Mass casualty incident in progress — actual consumption will be at full request volume.",
            "Active MCI response; blood and critical supplies needed at surge rate.",
            "Trauma bay at full load — request reflects real need during crisis.",
        ]

    def _format_round_brief(self) -> str:
        state = self._state
        active_event_names = [
            self._event(eid).event_type if self._event(eid) else eid
            for eid in state.active_events
        ]
        lines = [
            f"=== ROUND {state.round_idx} / {state.max_rounds} "
            f"(day {state.day}, trigger: {state.current_round_trigger}) ===",
            f"Active events: {', '.join(active_event_names) if active_event_names else 'none'}",
            f"Pending ward requests: {len(state.pending_requests)}",
            f"Unread inbox messages: {sum(1 for m in state.inbox if not m.read)}",
            f"Budget used: ${state.budget_used:,.0f} / ${state.budget_limit:,.0f}",
            "",
            "Available tools:",
            "  read_inbox, view_requests, query_ward_history, query_erp, query_supplier,",
            "  submit_po, file_justification, quarantine_lot, submit_allocation_plan, advance_round",
            "",
            "Call submit_allocation_plan once per round, then advance_round.",
        ]
        return "\n".join(lines)

    # ── Tool: read_inbox ─────────────────────────────────────────────────

    def read_inbox(self, filter: str = "unread") -> str:
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."

        messages = list(state.inbox)
        if filter == "unread":
            messages = [m for m in messages if not m.read]
        elif filter == "flagged":
            messages = [m for m in messages if m.flagged]

        for m in messages:
            m.read = True

        if not messages:
            return f"INBOX EMPTY\nFilter: {filter} | No messages matching filter."

        lines = []
        for m in messages:
            lines.append(
                f"\n[MSG {m.msg_id} | PRIORITY: {m.priority} | {m.timestamp_str}]"
            )
            lines.append(f"FROM: {m.sender}")
            lines.append(f"SUBJ: {m.subject}")
            lines.append("")
            lines.append(m.body)
            lines.append("")
        return "\n".join(lines)

    # ── Tool: view_requests ──────────────────────────────────────────────

    def view_requests(self) -> str:
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."
        if not state.pending_requests:
            return "PENDING REQUESTS: none. Call advance_round to move to the next round."

        header = (
            f"PENDING WARD REQUESTS — Round {state.round_idx} "
            f"(day {state.day}, spanning {state.round_length_days} sim days)\n"
        )
        sep = "-" * 98
        col = (
            f"{'WARD':<14} | {'SKU':<12} | {'QTY_REQ':>7} | "
            f"{'PRIO':>4} | JUSTIFICATION"
        )
        rows = []
        ward_priority = {w.ward_id: w.priority_weight for w in self._task.wards}
        for req in state.pending_requests:
            prio = ward_priority.get(req.ward_id, 0.5)
            rows.append(
                f"{req.ward_id:<14} | {req.product_id:<12} | "
                f"{req.requested_qty:>7} | {prio:>4.1f} | {req.justification[:60]}"
            )
        return "\n".join([header, sep, col, sep] + rows + [sep,
                f"{len(rows)} pending request(s)."])

    # ── Tool: query_ward_history ─────────────────────────────────────────

    def query_ward_history(self, ward_id: str,
                           product_id: Optional[str] = None,
                           n_rounds: int = 10) -> str:
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."

        valid_wards = {w.ward_id for w in self._task.wards}
        if ward_id not in valid_wards:
            return f"ERROR: Unknown ward '{ward_id}'. Valid: {sorted(valid_wards)}"

        # Join requests + allocations on (round_idx, ward_id, product_id)
        alloc_lookup: Dict[Tuple[int, str, str], WardAllocation] = {}
        for a in state.ward_allocation_log:
            alloc_lookup[(a.round_idx, a.ward_id, a.product_id)] = a

        rows: List[Tuple[int, str, int, int, float, bool]] = []
        for req in state.ward_request_log:
            if req.ward_id != ward_id:
                continue
            if product_id is not None and req.product_id != product_id:
                continue
            alloc = alloc_lookup.get((req.round_idx, req.ward_id, req.product_id))
            allocated = alloc.allocated_qty if alloc else 0
            consumed = alloc.actual_consumed if alloc else 0.0
            stockout = alloc.stockout_flag if alloc else False
            rows.append((
                req.round_idx,
                req.product_id,
                req.requested_qty,
                allocated,
                consumed,
                stockout,
            ))

        rows.sort(key=lambda r: (r[0], r[1]))
        # Keep the last n_rounds distinct round_idx values
        if n_rounds > 0 and rows:
            last_rounds = sorted({r[0] for r in rows})[-n_rounds * 10:]
            rows = [r for r in rows if r[0] in last_rounds]

        header = (
            f"WARD HISTORY — {ward_id} | last {n_rounds} round(s)"
            + (f" | sku={product_id}" if product_id else "")
        )
        sep = "-" * 78
        col = (
            f"{'RND':>4} | {'SKU':<12} | {'REQ':>5} | {'ALLOC':>5} | "
            f"{'CONS':>6} | STOCKOUT"
        )
        body = [
            f"{rnd:>4} | {sku:<12} | {req:>5} | {alloc:>5} | "
            f"{cons:>6.1f} | {'Y' if so else ' '}"
            for (rnd, sku, req, alloc, cons, so) in rows
        ]
        note = "\n(rows with negative RND = pre-episode synthetic history)"
        return "\n".join([header, sep, col, sep] + (body or ["(no history)"]) + [sep, note])

    # ── Tool: query_erp ──────────────────────────────────────────────────

    def query_erp(self, table: str, location: str = "all", sku: str = "all") -> str:
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."

        valid_tables = ["inventory", "expiry", "pipeline_orders"]
        if table not in valid_tables:
            return f"ERROR: Unknown table '{table}'. Valid: {valid_tables}"

        from .erp_formatter import (
            format_expiry_table, format_inventory_table, format_pipeline_table,
        )
        if table == "inventory":
            return format_inventory_table(state, self._task, location, sku)
        if table == "expiry":
            return format_expiry_table(state, self._task, location, sku)
        if table == "pipeline_orders":
            return format_pipeline_table(state, location, sku)
        return "ERROR: Unexpected table."

    # ── Tool: query_supplier ─────────────────────────────────────────────

    def query_supplier(self, supplier_id: str) -> str:
        state = self._state
        supplier = next(
            (s for s in self._task.suppliers if s.supplier_id == supplier_id), None
        )
        if not supplier:
            available = [s.supplier_id for s in self._task.suppliers]
            return f"ERROR: Supplier '{supplier_id}' not found. Available: {available}"

        effective_lead = supplier.base_lead_time
        note = "No active disruptions."
        for event_id in state.active_events:
            event = self._event(event_id)
            if (event and event.event_type == "supplier_disruption"
                    and event.params.get("supplier_id") == supplier_id):
                effective_lead = event.params["new_lead_time"]
                note = (
                    f"ACTIVE DISRUPTION: lead time extended to {effective_lead} days. "
                    f"Reason: {event.params['reason']}"
                )

        from .erp_formatter import format_supplier_info
        return format_supplier_info(supplier, effective_lead, note)

    # ── Tool: submit_po ──────────────────────────────────────────────────

    def submit_po(self, supplier_id: str, product_id: str, destination_id: str,
                  quantity: int, priority: str = "standard") -> str:
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."
        if priority not in ("standard", "expedited"):
            return "ERROR: priority must be 'standard' or 'expedited'."
        if quantity <= 0:
            return "ERROR: quantity must be positive."

        supplier = next(
            (s for s in self._task.suppliers if s.supplier_id == supplier_id), None
        )
        if not supplier:
            return f"ERROR: Supplier '{supplier_id}' not found."
        if product_id not in supplier.products:
            return f"ERROR: Supplier '{supplier_id}' does not supply '{product_id}'."

        valid_locs = [l.location_id for l in self._task.locations]
        if destination_id not in valid_locs:
            return f"ERROR: Destination '{destination_id}' not found. Valid: {valid_locs}"

        product = self._product(product_id)
        expedited_multiplier = 1.5 if priority == "expedited" else 1.0
        unit_cost = product.unit_cost * supplier.cost_multiplier * expedited_multiplier
        total_cost = unit_cost * quantity

        if state.budget_used + total_cost > state.budget_limit:
            overage = (state.budget_used + total_cost) - state.budget_limit
            return (
                f"ERROR: BUDGET_EXCEEDED\n"
                f"Order cost: ${total_cost:,.2f} | Outstanding: ${state.budget_used:,.2f} | "
                f"Limit: ${state.budget_limit:,.2f} | Overage: ${overage:,.2f}"
            )

        lead_time = supplier.base_lead_time
        for event_id in state.active_events:
            event = self._event(event_id)
            if (event and event.event_type == "supplier_disruption"
                    and event.params.get("supplier_id") == supplier_id):
                lead_time = event.params["new_lead_time"]

        if priority == "expedited":
            lead_time = max(1, lead_time - 2)
        if supplier.lead_time_std > 0:
            jitter = int(round(state.rng.normal(0, supplier.lead_time_std)))
            lead_time = max(1, lead_time + jitter)

        eta_day = state.day + lead_time
        po_id = f"POD-{state.po_counter:04d}"
        lot_id = f"LOT-{po_id}"
        state.po_counter += 1

        # MCI-prepositioning flag: PO for critical blood during warning/MCI window
        self._check_preposition_flag(product, destination_id)
        # Supplier-switch flag: PO during supplier disruption that picks a different supplier
        self._check_supplier_switch_flag(supplier_id)
        # Cold-chain-replenish flag
        self._check_coldchain_replenish_flag(product_id)

        po = PurchaseOrder(
            po_id=po_id, supplier_id=supplier_id, product_id=product_id,
            destination_id=destination_id, quantity=quantity, priority=priority,
            day_submitted=state.day, eta_day=eta_day,
            unit_cost=unit_cost, total_cost=total_cost,
            status="pending_justification" if priority == "expedited" else "pending_finance",
            lot_id=lot_id,
        )

        if priority == "expedited":
            ticket_id = f"BOT-{state.po_counter:04d}"
            state.po_counter += 1
            state.pending_overrides[ticket_id] = PendingBudgetOverride(
                ticket_id=ticket_id, po=po,
            )
            return (
                f"BUDGET_OVERRIDE_REQUIRED\n"
                f"Expedited PO {po_id} (${total_cost:,.2f}) requires justification.\n"
                f"Ticket: {ticket_id}. Call file_justification(ticket_id, reason)."
            )

        # Standard PO — gated on finance approval if above threshold
        if total_cost > self._task.approval_threshold:
            approval_id = f"APR-{state.approval_counter:04d}"
            state.approval_counter += 1
            state.pending_approvals[approval_id] = PendingApproval(
                approval_id=approval_id,
                po=po,
                submitted_round=state.round_idx,
            )
            return (
                f"APPROVAL_REQUIRED\n"
                f"PO {po_id} (${total_cost:,.2f}) exceeds the "
                f"${self._task.approval_threshold:,.0f} finance gate.\n"
                f"Approval ticket: {approval_id}. Call "
                f"finance_sap_request_approval(approval_id={approval_id!r}, justification=...)."
            )

        po.status = "in_transit"
        state.pipeline_orders.append(po)
        state.budget_used += total_cost
        return (
            f"OK — PO {po_id} submitted.\n"
            f"{product_id} × {quantity} → {destination_id} | ETA: Day {eta_day} | "
            f"Cost: ${total_cost:,.2f}"
        )

    # ── Tool: file_justification ─────────────────────────────────────────

    def file_justification(self, ticket_id: str, reason: str) -> str:
        state = self._state
        if ticket_id not in state.pending_overrides:
            return (
                f"ERROR: Ticket '{ticket_id}' not found or already processed.\n"
                f"Active tickets: {list(state.pending_overrides.keys())}"
            )
        override = state.pending_overrides.pop(ticket_id)
        po = override.po

        active_types: Set[str] = set()
        for event_id in state.active_events:
            event = self._event(event_id)
            if event:
                active_types.add(event.event_type)

        from .grader import grade_justification
        is_coherent = grade_justification(reason, active_types)
        state.justification_log.append(JustificationRecord(
            ticket_id=ticket_id, po_id=po.po_id, reason=reason, is_coherent=is_coherent,
        ))

        po.status = "in_transit"
        state.pipeline_orders.append(po)
        state.budget_used += po.total_cost

        audit = "" if is_coherent else (
            "\n[AUDIT FLAG] Justification does not reference active crisis conditions."
        )
        return (
            f"OK — justification {'accepted' if is_coherent else 'FLAGGED'}. "
            f"PO {po.po_id} released.\n"
            f"{po.product_id} × {po.quantity} → {po.destination_id} | "
            f"ETA: Day {po.eta_day}{audit}"
        )

    # ── Tool: quarantine_lot ─────────────────────────────────────────────

    def quarantine_lot(self, location_id: str, sku: str, lot_id: str) -> str:
        state = self._state
        valid_locs = {l.location_id for l in self._task.locations}
        if location_id not in valid_locs:
            return f"ERROR: Location '{location_id}' not found."

        key = (location_id, sku)
        lots = state.inventory.get(key, [])
        if lot_id == "all":
            target = list(lots)
        else:
            target = [l for l in lots if l.lot_id == lot_id]
            if not target:
                target = [l for l in lots if lot_id in l.lot_id]

        if not target:
            available = [l.lot_id for l in lots]
            return (
                f"ERROR: Lot '{lot_id}' not found at {location_id} for SKU {sku}. "
                f"Available: {available}"
            )

        qty = 0
        ids = []
        for lot in target:
            if lot.lot_id not in state.quarantined_lots:
                state.quarantined_lots.add(lot.lot_id)
                qty += lot.qty
                ids.append(lot.lot_id)

        # Event-response: track recall quarantine
        for event_id in list(state.active_events.keys()) + [
            e.event_id for e in self._task.events
            if e.event_type == "product_recall" and e.trigger_day <= state.day
        ]:
            event = self._event(event_id)
            if event and event.event_type == "product_recall":
                if sku == event.params.get("product_id") and (
                    lot_id == event.params.get("recall_lot_id")
                    or event.params.get("recall_lot_id") in [l.lot_id for l in target]
                ):
                    if state.recall_quarantined_by_round is None:
                        state.recall_quarantined_by_round = state.round_idx

        return (
            f"OK — quarantined {qty} unit(s) across lots {ids} at {location_id}."
        )

    # ── Tool: submit_allocation_plan ─────────────────────────────────────

    def submit_allocation_plan(
        self, plan_json: str, rationale_json: Optional[str] = None,
    ) -> str:
        """
        Submit the allocation plan for the current round.

        plan_json: JSON object {ward_id: {sku: qty}}.

        rationale_json (optional): JSON object {ward_id: "<rationale text>"}
        explaining the allocation. Rationales are scored on whether they
        reference disclosed evidence (census, acuity, recent_actuals,
        events, history, reputation). Wards that receive evidence-grounded
        rationale decay their hoarding_pressure slightly faster next round.
        """
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."
        if not state.pending_requests:
            return "ERROR: No pending requests this round. Call advance_round."

        try:
            plan = json.loads(plan_json) if isinstance(plan_json, str) else dict(plan_json)
        except Exception as exc:
            return f"ERROR: could not parse plan_json ({exc}). Expected JSON object."
        if not isinstance(plan, dict):
            return "ERROR: plan_json must be a JSON object."

        # Parse optional rationale
        rationale_map: Dict[str, str] = {}
        if rationale_json:
            try:
                parsed = (
                    json.loads(rationale_json)
                    if isinstance(rationale_json, str) else dict(rationale_json)
                )
                if isinstance(parsed, dict):
                    rationale_map = {str(k): str(v) for k, v in parsed.items()}
            except Exception:
                rationale_map = {}

        valid_wards = {w.ward_id for w in self._task.wards}
        pending_by_key: Dict[Tuple[str, str], WardRequest] = {
            (r.ward_id, r.product_id): r for r in state.pending_requests
        }

        lines_ok = []
        lines_err = []

        for ward_id, prod_plan in plan.items():
            if ward_id not in valid_wards:
                lines_err.append(f"  - unknown ward '{ward_id}' ignored.")
                continue
            if not isinstance(prod_plan, dict):
                lines_err.append(f"  - ward '{ward_id}' value must be an object.")
                continue
            for product_id, qty in prod_plan.items():
                try:
                    qty = int(qty)
                except (TypeError, ValueError):
                    lines_err.append(f"  - {ward_id}/{product_id}: qty must be int.")
                    continue
                if qty < 0:
                    lines_err.append(f"  - {ward_id}/{product_id}: negative qty.")
                    continue

                req = pending_by_key.get((ward_id, product_id))
                if req is None:
                    lines_err.append(
                        f"  - {ward_id}/{product_id}: no pending request."
                    )
                    continue

                requested = req.requested_qty
                if qty > requested:
                    lines_err.append(
                        f"  - {ward_id}/{product_id}: qty {qty} > requested "
                        f"{requested}; capping."
                    )
                    qty = requested

                moved = self._move_central_to_ward(product_id, ward_id, qty)
                if moved < qty:
                    lines_err.append(
                        f"  - {ward_id}/{product_id}: only {moved} units available "
                        f"at central_pharmacy (requested {qty})."
                    )
                alloc = WardAllocation(
                    round_idx=state.round_idx,
                    ward_id=ward_id,
                    product_id=product_id,
                    allocated_qty=moved,
                    true_need=req.true_need,
                )
                state.ward_allocation_log.append(alloc)
                lines_ok.append(f"  + {ward_id}/{product_id}: {moved} units allocated.")

        # Any pending requests without an explicit plan entry → zero allocation
        for (ward_id, product_id), req in pending_by_key.items():
            has_entry = (
                ward_id in plan and isinstance(plan[ward_id], dict)
                and product_id in plan[ward_id]
            )
            if not has_entry:
                alloc = WardAllocation(
                    round_idx=state.round_idx,
                    ward_id=ward_id,
                    product_id=product_id,
                    allocated_qty=0,
                    true_need=req.true_need,
                )
                state.ward_allocation_log.append(alloc)

        # Clear pending requests
        state.pending_requests = []

        # ── Process per-ward rationales (audit signal) ─────────────────
        EVIDENCE_KEYWORDS = (
            "census", "acuity", "actuals", "history", "consum",
            "events", "reputation", "stockout", "disclosed", "evidence",
        )
        for ward_id, rationale_text in rationale_map.items():
            if ward_id not in valid_wards:
                continue
            references = any(kw in rationale_text.lower() for kw in EVIDENCE_KEYWORDS)
            state.rationale_log.append(AllocationRationale(
                round_idx=state.round_idx,
                ward_id=ward_id,
                text=rationale_text[:280],
                references_evidence=references,
            ))
            # Mark the most-recent disclosed evidence rows for this ward as used
            if references:
                for ev in reversed(state.evidence_log):
                    if ev.ward_id == ward_id and ev.round_idx == state.round_idx:
                        ev.used_in_allocation = True
                        break
                # Lower next round's hoarding pressure (ward feels heard)
                rep_state = state.ward_actor_state.get(ward_id, {})
                if rep_state:
                    rep_state["hoarding_pressure"] = max(
                        0.0, rep_state.get("hoarding_pressure", 0.3) - 0.07
                    )

        lines = [
            f"ALLOCATION PLAN COMMITTED — round {state.round_idx}",
        ]
        if lines_ok:
            lines.append("Allocations:")
            lines.extend(lines_ok)
        if lines_err:
            lines.append("Warnings:")
            lines.extend(lines_err)
        lines.append("Call advance_round to resolve consumption.")
        return "\n".join(lines)

    def _move_central_to_ward(self, product_id: str, ward_id: str,
                              qty: int) -> int:
        """FEFO-move up to `qty` from central_pharmacy lots into ward lots."""
        state = self._state
        key_c = ("central_pharmacy", product_id)
        lots = sorted(
            [l for l in state.inventory.get(key_c, [])
             if l.lot_id not in state.quarantined_lots and l.qty > 0],
            key=lambda l: (l.expiry_day is None, l.expiry_day or 0),
        )
        remaining = qty
        key_w = (ward_id, product_id)
        state.inventory.setdefault(key_w, [])
        for lot in lots:
            if remaining <= 0:
                break
            take = min(remaining, lot.qty)
            lot.qty -= take
            remaining -= take
            state.inventory[key_w].append(Lot(
                lot_id=f"ALLOC-{state.round_idx}-{lot.lot_id}",
                qty=take, expiry_day=lot.expiry_day, cost_per_unit=lot.cost_per_unit,
            ))
        state.inventory[key_c] = [l for l in state.inventory.get(key_c, []) if l.qty > 0]
        return qty - remaining

    # ═══════════════════════════════════════════════════════════════════════
    # Tier-1 multi-actor + enterprise tools
    # ═══════════════════════════════════════════════════════════════════════

    # ── get_round_briefing — one-shot dashboard ───────────────────────────

    def get_round_briefing(self) -> str:
        """One-call situational briefing replacing the typical
        read_inbox + view_requests + query_erp×2 sequence."""
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."
        # Mark ALL inbox messages as read so re-issuing read_inbox won't double-pay
        for m in state.inbox:
            m.read = True
        state.briefing_calls_this_round += 1
        state.briefings_total += 1
        state.systems_used.add("messaging")  # briefing surfaces messaging info
        from .erp_formatter import format_briefing
        return format_briefing(state, self._task)

    # ── request_evidence — structured ward disclosure ────────────────────

    _EVIDENCE_TYPES = ("census", "acuity", "recent_actuals", "events", "all")

    def request_evidence(self, ward_id: str, sku: str, evidence_type: str = "all") -> str:
        """Ask a ward to substantiate its request with structured data.

        The data itself is pulled from SimState (ground truth) — the ward
        cannot fabricate numbers. But high-hoarding-pressure wards may
        REDACT one field (modelling reluctance to disclose). The redaction
        decision is deterministic per (seed, ward, sku, round, evidence).

        evidence_type ∈ {census, acuity, recent_actuals, events, all}
        """
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."
        valid_wards = {w.ward_id for w in self._task.wards}
        if ward_id not in valid_wards:
            return f"ERROR: Unknown ward '{ward_id}'."
        if evidence_type not in self._EVIDENCE_TYPES:
            return f"ERROR: evidence_type must be one of {list(self._EVIDENCE_TYPES)}."

        # Locate the active request for this ward × SKU (current round)
        req: Optional[WardRequest] = None
        for r in state.pending_requests:
            if r.ward_id == ward_id and r.product_id == sku:
                req = r
                break
        if req is None:
            for r in reversed(state.ward_request_log):
                if (r.ward_id == ward_id and r.product_id == sku
                        and r.round_idx == state.round_idx):
                    req = r
                    break
        if req is None:
            return f"ERROR: No current-round request for {ward_id}/{sku}."

        ward = self._ward(ward_id)
        product = self._product(sku)
        if ward is None or product is None:
            return f"ERROR: Unknown ward or SKU."

        # Deterministic disclosure decision based on hoarding pressure
        rep_state = state.ward_actor_state.get(ward_id, {})
        hoarding = float(rep_state.get("hoarding_pressure", 0.3))
        disclosure_seed = abs(hash((state.seed, ward_id, sku, state.round_idx, evidence_type))) % (2**32)
        disclose_rng = np.random.default_rng(disclosure_seed)
        will_redact = disclose_rng.random() < (hoarding * 0.6)   # ≤60% redaction even at max pressure

        # Compose evidence facts (all derived from sim state — no LLM)
        # Census: base_demand × round_length_days × small noise (seeded).
        census_seed = abs(hash((state.seed, ward_id, "census", state.round_idx))) % (2**32)
        census_rng = np.random.default_rng(census_seed)
        projected_census = max(1, int(round(
            ward.priority_weight * 18 + census_rng.normal(0, 2)
        )))   # ICU ~18, ER ~13, General ~5 baseline
        if ward_id == "ward_er":
            projected_census = int(projected_census * state.er_surge_state)

        acuity_score = round(ward.priority_weight * 0.85 + census_rng.uniform(-0.05, 0.10), 2)

        recent_actuals = []
        for a in state.ward_allocation_log:
            if (a.ward_id == ward_id and a.product_id == sku and a.round_idx > 0
                    and a.round_idx < state.round_idx):
                recent_actuals.append({
                    "round": a.round_idx,
                    "allocated": a.allocated_qty,
                    "consumed": round(a.actual_consumed, 1),
                    "stockout": a.stockout_flag,
                })
        recent_actuals = recent_actuals[-4:]

        events_text = ", ".join(
            self._event(eid).event_type for eid in state.active_events if self._event(eid)
        ) or "none"

        # Build response, redacting one field for high-pressure wards
        full = {
            "census": {"projected_census": projected_census, "unit_priority": ward.priority_weight},
            "acuity": {"acuity_score": acuity_score, "scale": "0.0 (light) - 1.0 (critical)"},
            "recent_actuals": recent_actuals or ["(no prior consumption — round 1)"],
            "events": {"active": events_text},
        }
        redacted_field: Optional[str] = None
        if will_redact:
            # Pick the field most likely to expose padding: census or recent_actuals
            redacted_field = ("census" if hoarding >= 0.6 else "recent_actuals")
            full[redacted_field] = "[REDACTED — not tracked at this granularity]"

        if evidence_type != "all":
            disclosed = {evidence_type: full.get(evidence_type, "[unknown]")}
            if evidence_type == redacted_field:
                full_disclosure = False
            else:
                full_disclosure = True
        else:
            disclosed = full
            full_disclosure = redacted_field is None

        state.evidence_log.append(EvidenceDisclosure(
            round_idx=state.round_idx,
            ward_id=ward_id,
            product_id=sku,
            evidence_type=evidence_type,
            disclosed=disclosed if isinstance(disclosed, dict) else {"value": disclosed},
            full_disclosure=full_disclosure,
        ))

        # Render response
        sep = "-" * 70
        lines = [
            f"EVIDENCE DISCLOSURE — {ward_id} / {sku}",
            f"Type: {evidence_type}  full_disclosure: {full_disclosure}",
            sep,
        ]
        for k, v in (disclosed.items() if isinstance(disclosed, dict) else []):
            lines.append(f"  {k}: {v}")
        lines.append(sep)
        if not full_disclosure:
            lines.append(
                f"NOTE: ward redacted '{redacted_field}'. Reputation will reflect this."
            )
            # Penalise reputation for refusing disclosure
            rep_state["reputation"] = max(0.0, rep_state["reputation"] - 0.05)
        return "\n".join(lines)

    # ── escalate_to_clinical_review — binding arbiter verdict ─────────────

    def escalate_to_clinical_review(
        self, ward_id: str, sku: str, concern: str,
    ) -> str:
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."
        if not isinstance(concern, str) or len(concern.strip()) < 8:
            return "ERROR: 'concern' must be a substantive string (8+ chars)."

        # Find the request to escalate (current-round only)
        req: Optional[WardRequest] = None
        for r in state.pending_requests:
            if r.ward_id == ward_id and r.product_id == sku:
                req = r
                break
        if req is None:
            return (
                f"ERROR: No pending request for {ward_id}/{sku} this round. "
                "Escalation must happen BEFORE submit_allocation_plan."
            )

        ward = self._ward(ward_id)
        if ward is None:
            return f"ERROR: Unknown ward '{ward_id}'."

        recent_history = [
            (a.round_idx, a.allocated_qty, a.actual_consumed, a.stockout_flag)
            for a in state.ward_allocation_log
            if a.ward_id == ward_id and a.product_id == sku and a.round_idx > 0
        ][-6:]

        active_event_summary = ", ".join(
            self._event(eid).event_type for eid in state.active_events if self._event(eid)
        ) or "none"

        from .clinical_arbiter import review_request as _review
        verdict = _review(
            ward_id=ward_id,
            product_id=sku,
            requested_qty=req.requested_qty,
            true_need=req.true_need,
            padded_flag=req.padded_flag,
            concern=concern,
            recent_history=recent_history,
            active_event_summary=active_event_summary,
            ward_priority=ward.priority_weight,
        )

        original_qty = req.requested_qty
        # Bind: lock request at recommended quantity
        req.requested_qty = verdict.recommended_qty

        # Score correctness:
        #   was_padded & verdict != APPROVE → correct (+1)
        #   was_padded & verdict == APPROVE → frivolous-let-go (-)
        #   was_honest & verdict == APPROVE → correct (+1)
        #   was_honest & verdict != APPROVE → frivolous escalation (-)
        approved = verdict.verdict == "APPROVE"
        correct = (req.padded_flag and not approved) or (not req.padded_flag and approved)

        state.escalation_log.append(EscalationRecord(
            round_idx=state.round_idx,
            ward_id=ward_id,
            product_id=sku,
            concern=concern[:240],
            was_padded=req.padded_flag,
            original_qty=original_qty,
            recommended_qty=verdict.recommended_qty,
            verdict=verdict.verdict,
            reason=verdict.reason[:280],
            correct=correct,
        ))

        # Reputation impact: padded-and-confirmed cuts reputation harder than challenge
        rep_state = state.ward_actor_state.setdefault(ward_id, {
            "reputation": 0.5, "recent_stockouts": 0,
            "hoarding_pressure": 0.3, "challenges_received": 0,
        })
        if req.padded_flag and not approved:
            rep_state["reputation"] = max(0.0, rep_state["reputation"] - 0.20)
        elif not req.padded_flag and approved:
            rep_state["reputation"] = min(1.0, rep_state["reputation"] + 0.05)
        elif not req.padded_flag and not approved:
            # frivolous escalation — refund the ward some trust
            rep_state["reputation"] = min(1.0, rep_state["reputation"] + 0.05)

        return (
            f"CLINICAL REVIEW BOARD VERDICT — {ward_id}/{sku}\n"
            f"Verdict:        {verdict.verdict}\n"
            f"Recommended:    {verdict.recommended_qty} (original: {original_qty})\n"
            f"Reason:         {verdict.reason}\n"
            f"Reviewer note:  request locked at recommended quantity for this round."
        )

    # ── ERP Oracle (stale-by-1-round) ─────────────────────────────────────

    def erp_oracle_get_inventory(self, location: str = "all", sku: str = "all") -> str:
        """Authoritative ERP inventory snapshot — refreshed at round open.
        Stale by up to 1 round vs the WMS scanner."""
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."
        state.systems_used.add("erp_oracle")
        sep = "-" * 78
        rows: list[str] = []
        loc_f = location.lower()
        sku_f = sku.lower()
        for (loc, prod), qty in sorted(state.inventory_snapshot_prev_round.items()):
            if loc_f != "all" and loc.lower() != loc_f:
                continue
            if sku_f != "all" and prod.lower() != sku_f:
                continue
            if qty <= 0:
                continue
            rows.append(f"  {loc:<18} {prod:<12} qty_snapshot={qty}")
        body = rows or ["  (no rows in last snapshot — round 1 has none)"]
        return "\n".join([
            f"ERP-ORACLE INVENTORY SNAPSHOT  [as of round {max(0, state.round_idx - 1)}]",
            sep,
            *body,
            sep,
            "NOTE: snapshot lags by one round. Use wms_scan_inventory for live data.",
        ])

    def erp_oracle_get_pipeline(self) -> str:
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."
        state.systems_used.add("erp_oracle")
        return self.query_erp("pipeline_orders")

    # ── WMS (live, noisy) ─────────────────────────────────────────────────

    def wms_scan_inventory(self, location: str = "all", sku: str = "all") -> str:
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."
        state.systems_used.add("wms")
        from .erp_formatter import format_wms_inventory
        return format_wms_inventory(
            state, self._task, location, sku, self._task.wms_noise_pct,
        )

    # ── Supplier Portal (async quotes) ────────────────────────────────────

    def supplier_portal_request_quote(
        self, supplier_id: str, product_id: str, quantity: int,
    ) -> str:
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."
        if quantity <= 0:
            return "ERROR: quantity must be positive."
        supplier = next(
            (s for s in self._task.suppliers if s.supplier_id == supplier_id), None
        )
        if not supplier:
            return f"ERROR: Supplier '{supplier_id}' not found."
        if product_id not in supplier.products:
            return f"ERROR: Supplier '{supplier_id}' does not supply '{product_id}'."
        product = self._product(product_id)
        if product is None:
            return f"ERROR: product '{product_id}' not found."

        state.systems_used.add("supplier_portal")
        lead_time = supplier.base_lead_time
        for eid in state.active_events:
            ev = self._event(eid)
            if (ev and ev.event_type == "supplier_disruption"
                    and ev.params.get("supplier_id") == supplier_id):
                lead_time = ev.params["new_lead_time"]
        unit_cost = product.unit_cost * supplier.cost_multiplier
        total = unit_cost * quantity

        quote_id = f"Q-{state.quote_counter:04d}"
        state.quote_counter += 1
        state.pending_quotes[quote_id] = PendingQuote(
            quote_id=quote_id,
            supplier_id=supplier_id,
            product_id=product_id,
            quantity=quantity,
            submitted_day=state.day,
            resolves_day=state.day + max(0, self._task.quote_resolution_turns),
            lead_time=lead_time,
            unit_cost=unit_cost,
            total_cost=total,
        )
        return (
            f"QUOTE REQUEST RECEIVED — {quote_id}\n"
            f"Resolves on day {state.pending_quotes[quote_id].resolves_day}.\n"
            f"Call supplier_portal_get_quote(quote_id={quote_id!r}) to retrieve."
        )

    def supplier_portal_get_quote(self, quote_id: str) -> str:
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."
        state.systems_used.add("supplier_portal")
        q = state.pending_quotes.get(quote_id)
        if q is None:
            return f"ERROR: Unknown quote_id '{quote_id}'."
        from .erp_formatter import format_supplier_quote
        if state.day < q.resolves_day:
            return format_supplier_quote({
                "quote_id": q.quote_id,
                "status": "pending",
                "submitted_day": q.submitted_day,
                "resolves_day": q.resolves_day,
            })
        q.fulfilled = True
        return format_supplier_quote({
            "quote_id": q.quote_id,
            "status": "ready",
            "supplier_id": q.supplier_id,
            "product_id": q.product_id,
            "quantity": q.quantity,
            "lead_time": q.lead_time,
            "unit_cost": q.unit_cost,
            "total_cost": q.total_cost,
        })

    # ── Finance SAP (budget + approvals) ──────────────────────────────────

    def finance_sap_get_budget(self) -> str:
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."
        state.systems_used.add("finance_sap")
        headroom = state.budget_limit - state.budget_used
        return "\n".join([
            "FINANCE-SAP BUDGET STATUS",
            "-" * 50,
            f"  Limit:           ${state.budget_limit:,.0f}",
            f"  Outstanding:     ${state.budget_used:,.0f}",
            f"  Headroom:        ${headroom:,.0f}",
            f"  Approval gate:   POs > ${self._task.approval_threshold:,.0f} require finance approval",
            f"  Pending appr.:   {len(state.pending_approvals)}",
            "-" * 50,
        ])

    def finance_sap_request_approval(self, approval_id: str, justification: str) -> str:
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."
        state.systems_used.add("finance_sap")
        appr = state.pending_approvals.get(approval_id)
        if appr is None:
            return (
                f"ERROR: Approval ticket '{approval_id}' not found. "
                f"Active approvals: {list(state.pending_approvals.keys())}"
            )
        if not isinstance(justification, str) or len(justification.strip()) < 8:
            return "ERROR: 'justification' must be a substantive string (8+ chars)."

        active_types: Set[str] = set()
        for eid in state.active_events:
            ev = self._event(eid)
            if ev:
                active_types.add(ev.event_type)
        from .grader import grade_justification
        coherent = grade_justification(justification, active_types)

        appr.justification = justification[:400]
        appr.coherent = coherent
        if coherent:
            appr.status = "approved"
            appr.po.status = "in_transit"
            state.pipeline_orders.append(appr.po)
            state.budget_used += appr.po.total_cost
            state.approval_log.append(appr)
            del state.pending_approvals[approval_id]
            return (
                f"APPROVAL GRANTED — {approval_id}\n"
                f"PO {appr.po.po_id} released ({appr.po.product_id} × "
                f"{appr.po.quantity} → {appr.po.destination_id})."
            )

        appr.status = "rejected"
        state.approval_log.append(appr)
        del state.pending_approvals[approval_id]
        return (
            f"APPROVAL DENIED — {approval_id}\n"
            "Justification did not reference active operational context. "
            f"PO {appr.po.po_id} cancelled. Resubmit with stronger evidence "
            "(e.g. cite an active MCI, recall, or supplier disruption) or pick "
            "a cheaper supplier."
        )

    # ── Messaging (outbound) ──────────────────────────────────────────────

    def messaging_send_to_ward(self, ward_id: str, body: str) -> str:
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."
        valid = {w.ward_id for w in self._task.wards}
        if ward_id not in valid:
            return f"ERROR: Unknown ward '{ward_id}'. Valid: {sorted(valid)}"
        if not isinstance(body, str) or len(body.strip()) < 4:
            return "ERROR: 'body' must be non-trivial text."
        state.systems_used.add("messaging")
        actor = self._actors.get(ward_id)
        rep_state = state.ward_actor_state.get(ward_id, {})
        ctx = {
            "round": state.round_idx,
            "reputation": f"{rep_state.get('reputation', 0.5):.2f}",
            "recent_stockouts": rep_state.get("recent_stockouts", 0),
        }
        if actor is None:
            reply = f"[{ward_id}] (auto) acknowledged: {body[:60]}"
        else:
            reply = actor.respond_to_message(body, ctx)
        state.outbound_messages.append({
            "round": str(state.round_idx),
            "ward_id": ward_id,
            "body": body[:240],
            "reply": reply[:240],
        })
        return f"MESSAGE DELIVERED to {ward_id}.\nReply: {reply}"

    # ── Stale-snapshot capture (called from _open_round) ──────────────────

    def _snapshot_inventory_for_oracle(self):
        state = self._state
        snapshot: Dict[Tuple[str, str], int] = {}
        for (loc, prod), lots in state.inventory.items():
            qty = sum(l.qty for l in lots if l.lot_id not in state.quarantined_lots)
            if qty > 0:
                snapshot[(loc, prod)] = qty
        state.inventory_snapshot_prev_round = snapshot

    # ═══════════════════════════════════════════════════════════════════════

    # ── advance_round ─────────────────────────────────────────────────────

    def advance_round(self) -> str:
        """Run round_length_days of simulation forward, then open next round
        or finish the episode."""
        state = self._state
        if state is None:
            return "ERROR: Environment not initialised."
        # If the agent calls advance_round without submitting a plan, assume
        # zero allocation (this is already handled in submit_allocation_plan
        # fallback logic — but if they skipped entirely we still need to
        # record zero allocations).
        if state.pending_requests:
            # Record zero allocations for all unplanned pending requests
            for req in state.pending_requests:
                state.ward_allocation_log.append(WardAllocation(
                    round_idx=state.round_idx,
                    ward_id=req.ward_id,
                    product_id=req.product_id,
                    allocated_qty=0,
                    true_need=req.true_need,
                ))
            state.pending_requests = []

        events_fired: List[str] = []
        for _ in range(state.round_length_days):
            if state.day > state.max_days:
                break
            day_events = self._advance_one_day(state.day)
            events_fired.extend(day_events)
            state.day += 1

        self._resolve_round_consumption_window()

        if state.round_idx >= state.max_rounds or state.day > state.max_days:
            self._done = True
            from .grader import compute_reward
            final = compute_reward(state, self._task)
            self._last_reward = final
            return self._format_terminal_summary(final)

        trigger = ", ".join(events_fired) if events_fired else "scheduled"
        return self._open_round(trigger)

    def _advance_one_day(self, day: int) -> List[str]:
        """Deliveries, expiry, ward consumption, event injection for a day."""
        state = self._state
        events_today: List[str] = []

        # 1. Deliver arriving POs into destination inventory
        for po in list(state.pipeline_orders):
            if po.eta_day <= day and po.status == "in_transit":
                product = self._product(po.product_id)
                key = (po.destination_id, po.product_id)
                expiry = (day + product.shelf_life_days) if product.shelf_life_days else None
                state.inventory.setdefault(key, []).append(Lot(
                    lot_id=po.lot_id, qty=po.quantity,
                    expiry_day=expiry, cost_per_unit=po.unit_cost,
                ))
                state.budget_used -= po.total_cost
                state.total_spend += po.total_cost
                po.status = "delivered"

        state.pipeline_orders = [
            po for po in state.pipeline_orders if po.status != "delivered"
        ]

        # 2. Expire old lots
        for key in list(state.inventory.keys()):
            fresh, expired = [], []
            for lot in state.inventory[key]:
                if lot.expiry_day is not None and lot.expiry_day <= day:
                    expired.append(lot)
                else:
                    fresh.append(lot)
            for lot in expired:
                state.total_wasted_value += lot.qty * lot.cost_per_unit
            state.inventory[key] = fresh

        # 3. Inject events for this day
        for event in self._task.events:
            if event.trigger_day == day:
                self._inject_event(event, day)
                events_today.append(event.event_type)

                if event.event_type == "cold_chain_breach":
                    self._apply_cold_chain_breach(event)
                if event.event_type == "product_recall":
                    self._inject_recall_lots(event, day)
            if event.warning_message and event.trigger_day - 1 == day:
                self._inject_warning(event, day)

        # 4. Update active events dict
        self._update_active_events(day)

        # 5. Ward consumption for the day (actual_need = same formula, per-day slice)
        self._resolve_ward_day_consumption(day)

        return events_today

    def _resolve_ward_day_consumption(self, day: int):
        """For each ward × product, consume one day's actual_need FEFO from
        ward-local inventory. Accumulates on the pending-round WardAllocation
        so that stockout/consumption are known when the round ends."""
        state = self._state
        for ward in self._task.wards:
            for product_id in ward.products_tracked:
                product = self._product(product_id)
                if product is None:
                    continue

                # Actual need for this single day
                base_daily = product.base_demand
                if ward.ward_id == "ward_er":
                    base_daily *= state.er_surge_state
                if ward.ward_id in ("ward_icu", "ward_er"):
                    for eid in state.active_events:
                        event = self._event(eid)
                        if (event and event.event_type == "mci"
                                and product.criticality in ("CRITICAL", "HIGH")
                                and ward.ward_id in event.params.get("locations", [])):
                            base_daily *= event.params.get("demand_multiplier", 2.8)
                daily_need = max(0.0, base_daily * float(state.rng.normal(1.0, 0.08)))

                consumed = self._fefo_consume(ward.ward_id, product_id, daily_need)

                # Track actual_consumed + stockout on current-round WardAllocation
                alloc = self._latest_allocation(state.round_idx, ward.ward_id, product_id)
                if alloc is not None:
                    alloc.actual_consumed += consumed
                    if consumed < daily_need:
                        alloc.stockout_flag = True

    def _fefo_consume(self, location_id: str, product_id: str, demand: float) -> float:
        """Consume `demand` units FEFO from location's inventory. Returns consumed qty."""
        state = self._state
        key = (location_id, product_id)
        lots = sorted(
            [l for l in state.inventory.get(key, [])
             if l.lot_id not in state.quarantined_lots and l.qty > 0],
            key=lambda l: (l.expiry_day is None, l.expiry_day or 0),
        )
        remaining = demand
        consumed = 0.0
        for lot in lots:
            if remaining <= 0:
                break
            take = min(remaining, float(lot.qty))
            lot.qty = int(lot.qty - math.floor(take))
            # Handle fractional remainder by converting floor then tracking leftover
            consumed += take
            remaining -= take
        state.inventory[key] = [l for l in state.inventory.get(key, []) if l.qty > 0]
        return consumed

    def _resolve_round_consumption_window(self):
        """Mark all unresolved current-round WardAllocations as resolved.
        Also update per-ward recent_stockouts for the actor state."""
        state = self._state
        per_ward_stockouts: Dict[str, int] = {}
        for alloc in state.ward_allocation_log:
            if alloc.round_idx == state.round_idx and not alloc.resolved:
                alloc.resolved = True
                if alloc.stockout_flag:
                    per_ward_stockouts[alloc.ward_id] = (
                        per_ward_stockouts.get(alloc.ward_id, 0) + 1
                    )
        for ward_id, rep_state in state.ward_actor_state.items():
            rep_state["recent_stockouts"] = per_ward_stockouts.get(ward_id, 0)

    def _latest_allocation(self, round_idx: int, ward_id: str,
                           product_id: str) -> Optional[WardAllocation]:
        for alloc in reversed(self._state.ward_allocation_log):
            if (alloc.round_idx == round_idx
                    and alloc.ward_id == ward_id
                    and alloc.product_id == product_id):
                return alloc
        return None

    # ── Event helpers ────────────────────────────────────────────────────

    def _inject_event(self, event: SimEvent, day: int):
        state = self._state
        msg = InboxMessage(
            msg_id=f"MSG-{state.msg_counter:04d}",
            priority=event.message.priority,
            timestamp_str=f"Day {day} 06:00",
            sender=event.message.sender,
            subject=event.message.subject,
            body=event.message.body,
            read=False,
            flagged=(event.message.priority == "CRITICAL"),
            event_id=event.event_id,
        )
        state.inbox.append(msg)
        state.msg_counter += 1

    def _inject_warning(self, event: SimEvent, day: int):
        state = self._state
        wm = event.warning_message
        msg = InboxMessage(
            msg_id=f"MSG-{state.msg_counter:04d}",
            priority=wm.priority,
            timestamp_str=f"Day {day} 18:00",
            sender=wm.sender,
            subject=wm.subject,
            body=wm.body,
            read=False,
            flagged=False,
            event_id=f"{event.event_id}_warning",
        )
        state.inbox.append(msg)
        state.msg_counter += 1

    def _update_active_events(self, day: int):
        state = self._state
        state.active_events = {
            eid: last_day
            for eid, last_day in state.active_events.items()
            if last_day >= day
        }
        for event in self._task.events:
            if event.trigger_day == day and event.duration_days > 0:
                state.active_events[event.event_id] = day + event.duration_days - 1
        # Also include zero-duration events as "active today"
        for event in self._task.events:
            if event.trigger_day == day and event.duration_days == 0:
                state.active_events[event.event_id] = day

    def _apply_cold_chain_breach(self, event: SimEvent):
        state = self._state
        loc = event.params.get("location_id")
        prod = event.params.get("product_id")
        key = (loc, prod)
        for lot in state.inventory.get(key, []):
            state.quarantined_lots.add(lot.lot_id)

    def _inject_recall_lots(self, event: SimEvent, day: int):
        """Inject the recall lot across the listed locations."""
        state = self._state
        product_id = event.params["product_id"]
        recall_lot_id = event.params["recall_lot_id"]
        qty = event.params["qty_per_location"]
        product = self._product(product_id)
        if product is None:
            return
        for loc_id in event.params["locations_with_lot"]:
            key = (loc_id, product_id)
            state.inventory.setdefault(key, []).append(Lot(
                lot_id=recall_lot_id, qty=qty,
                expiry_day=None, cost_per_unit=product.unit_cost,
            ))

    def _check_preposition_flag(self, product: Product, destination_id: str):
        state = self._state
        if product.criticality != "CRITICAL":
            return
        # MCI warning active (warning message fired) OR MCI currently active
        mci_warning_seen = any(
            m for m in state.inbox if "mci" in m.subject.lower() or
            "mci" in m.body.lower()
        )
        mci_now = any(
            self._event(eid) and self._event(eid).event_type == "mci"
            for eid in state.active_events
        )
        if (mci_warning_seen or mci_now) and destination_id in ("ward_icu", "ward_er"):
            state.mci_prepositioned = True

    def _check_supplier_switch_flag(self, supplier_id: str):
        state = self._state
        # Active disruption on a DIFFERENT supplier
        for eid in state.active_events:
            event = self._event(eid)
            if (event and event.event_type == "supplier_disruption"
                    and event.params.get("supplier_id") != supplier_id):
                state.supplier_switched = True
                return

    def _check_coldchain_replenish_flag(self, product_id: str):
        state = self._state
        for eid in state.active_events:
            event = self._event(eid)
            if (event and event.event_type == "cold_chain_breach"
                    and event.params.get("product_id") == product_id):
                state.coldchain_replenished = True

    def _mci_active(self, day: int) -> bool:
        state = self._state
        for eid in state.active_events:
            event = self._event(eid)
            if event and event.event_type == "mci":
                return True
        return False

    # ── Accessors ────────────────────────────────────────────────────────

    def _product(self, product_id: str) -> Optional[Product]:
        return next(
            (p for p in self._task.products if p.product_id == product_id), None
        )

    def _ward(self, ward_id: str) -> Optional[WardConfig]:
        return next((w for w in self._task.wards if w.ward_id == ward_id), None)

    def _event(self, event_id: str) -> Optional[SimEvent]:
        return next(
            (e for e in self._task.events if e.event_id == event_id), None
        )

    def get_last_reward(self) -> float:
        return self._last_reward

    def is_done(self) -> bool:
        return self._done

    # ── Terminal summary ─────────────────────────────────────────────────

    def _format_terminal_summary(self, final_score: float) -> str:
        state = self._state
        # Aggregate service-level-ish numbers
        total_true = sum(a.true_need for a in state.ward_allocation_log
                         if a.round_idx > 0)
        total_consumed = sum(a.actual_consumed for a in state.ward_allocation_log
                             if a.round_idx > 0)
        sl = total_consumed / max(total_true, 1e-6)

        stockouts_by_ward: Dict[str, int] = {}
        for a in state.ward_allocation_log:
            if a.round_idx > 0 and a.stockout_flag:
                stockouts_by_ward[a.ward_id] = stockouts_by_ward.get(a.ward_id, 0) + 1

        lines = [
            "=== EPISODE COMPLETE ===",
            f"Final Score: {final_score:.3f}",
            f"Rounds played: {state.round_idx} / {state.max_rounds}",
            f"Network service level: {sl * 100:.1f}%",
            f"Total spend: ${state.total_spend:,.2f}",
            f"Waste: ${state.total_wasted_value:,.2f}",
            "Stockouts by ward:",
        ]
        for ward_id in sorted(stockouts_by_ward):
            lines.append(f"  {ward_id}: {stockouts_by_ward[ward_id]}")
        if not stockouts_by_ward:
            lines.append("  (none)")
        return "\n".join(lines)
