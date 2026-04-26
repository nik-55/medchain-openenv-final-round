"""
Task configuration for MedChain Finals — single seeded multi-actor coordination task.

The agent coordinates stock allocation across three scripted ward actors
(ICU, ER, General) who submit requests that may be inflated. Episodes are
5-8 event-based rounds (each round = 2 simulated days).

Exposed factory: make_task_config(seed, difficulty) -> TaskConfig.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Core simulation dataclasses (reused by simulation.py) ────────────────────

@dataclass
class Product:
    product_id: str
    name: str
    shelf_life_days: Optional[int]
    criticality: str                 # "CRITICAL", "HIGH", "NORMAL"
    unit_cost: float
    locations: List[str]             # which wards stock this SKU
    base_demand: float               # per-ward per-day baseline consumption
    demand_std: float
    seasonal_amplitude: float = 0.0
    seasonal_period: int = 0
    seasonal_phase: float = 0.0


@dataclass
class Supplier:
    supplier_id: str
    name: str
    base_lead_time: int
    lead_time_std: float
    cost_multiplier: float
    products: List[str]


@dataclass
class Location:
    location_id: str
    name: str
    capacity: Optional[int] = None


@dataclass(frozen=True)
class InboxMessageTemplate:
    priority: str
    sender: str
    subject: str
    body: str


@dataclass
class SimEvent:
    event_id: str
    event_type: str                  # "mci", "mci_warning", "supplier_disruption",
                                     # "product_recall", "cold_chain_breach"
    trigger_day: int
    duration_days: int
    params: Dict
    message: InboxMessageTemplate
    warning_message: Optional[InboxMessageTemplate] = None


# ─── Ward personality ────────────────────────────────────────────────────────

@dataclass
class WardConfig:
    """Scripted ward actor parameters. Wards are NOT trained."""
    ward_id: str                     # e.g. "ward_icu"
    name: str
    priority_weight: float           # used in allocation_accuracy scoring
    pad_prob: float                  # probability of inflating this round's request
    pad_lo: float                    # padding multiplier lower bound (e.g. 1.25)
    pad_hi: float                    # upper bound
    products_tracked: List[str]      # SKUs this ward requests
    honest_justifications: List[str]
    padded_justifications: List[str]
    spike_prob: float = 0.0          # prob of non-event spike (ER only)
    spike_multiplier: float = 2.0    # magnitude when spike fires


@dataclass
class WardActorConfig:
    """LLM-driven ward actor parameters (Theme #1 multi-actor upgrade).

    Each ward has private goals that the central coordinator can't see and
    must infer from behaviour. Ground-truth `true_need` is still computed
    by the simulation — actors only choose how to *frame* the request.
    """
    ward_id: str
    persona: str                     # short LLM-prompt persona (1-2 sentences)
    private_goals: List[str]         # bullet list, hidden from agent
    hoarding_pressure_init: float    # 0.0 honest .. 1.0 chronically padding
    reputation_decay: float = 0.85   # rep moves toward 0.5 each round


# ─── Per-difficulty actor personas ────────────────────────────────────────────

_ICU_PERSONA = (
    "You are the ICU charge nurse. You are precise, evidence-driven, and you "
    "trust the central coordinator. You very rarely inflate requests because "
    "every padded order erodes the trust your unit needs in a real crisis."
)
_ER_PERSONA = (
    "You are the ER duty manager. You handle unpredictable surges and care "
    "about resilience. You sometimes pad blood and consumables because the "
    "downside of a stockout during an MCI is patient death — but you can "
    "back that up with real operational reasons if asked for evidence."
)
_GENERAL_PERSONA = (
    "You are the General Ward supervisor. Your unit has the lowest priority "
    "and you have learned that overordering is the only way to get adequate "
    "stock. You routinely add a buffer and rationalise it post-hoc."
)

_ICU_GOALS = ["never stock out on critical SKUs", "preserve trust capital"]
_ER_GOALS = [
    "be ready for tail-risk surge",
    "win pre-positioning battles before MCIs",
    "don't get caught padding in calm rounds",
]
_GENERAL_GOALS = [
    "secure comfortable buffer stock",
    "avoid being deprioritised next round",
]


def _make_ward_actor_configs(difficulty: str) -> Dict[str, WardActorConfig]:
    """Hoarding pressure scales with difficulty for ER and General."""
    er_pressure = {"light": 0.20, "medium": 0.45, "heavy": 0.65}[difficulty]
    gen_pressure = {"light": 0.55, "medium": 0.70, "heavy": 0.85}[difficulty]
    return {
        "ward_icu": WardActorConfig(
            ward_id="ward_icu",
            persona=_ICU_PERSONA,
            private_goals=_ICU_GOALS,
            hoarding_pressure_init=0.10,
        ),
        "ward_er": WardActorConfig(
            ward_id="ward_er",
            persona=_ER_PERSONA,
            private_goals=_ER_GOALS,
            hoarding_pressure_init=er_pressure,
        ),
        "ward_general": WardActorConfig(
            ward_id="ward_general",
            persona=_GENERAL_PERSONA,
            private_goals=_GENERAL_GOALS,
            hoarding_pressure_init=gen_pressure,
        ),
    }


# ─── Task config ──────────────────────────────────────────────────────────────

@dataclass
class TaskConfig:
    name: str
    seed: int
    difficulty: str                  # "light" | "medium" | "heavy"
    max_rounds: int
    round_length_days: int
    max_days: int
    budget_limit: float
    products: List[Product]
    suppliers: List[Supplier]
    locations: List[Location]
    wards: List[WardConfig]
    events: List[SimEvent]
    initial_stock_days: Dict[str, float]     # product_id -> days of stock to seed
    synthetic_history_rounds: int            # pre-seeded history length H
    benchmark_spend: float                   # for budget_efficiency term

    # ── Multi-actor / enterprise-system additions ──
    ward_actor_configs: Dict[str, WardActorConfig] = field(default_factory=dict)
    approval_threshold: float = 10_000.0     # POs above this cost need finance approval
    quote_resolution_turns: int = 1          # supplier_portal async quote latency
    wms_noise_pct: float = 0.05              # WMS scan noise as fraction of qty
    relevant_systems: List[str] = field(default_factory=lambda: [
        "erp_oracle", "wms", "supplier_portal", "finance_sap", "messaging",
    ])


# ─── Deterministic catalogs ──────────────────────────────────────────────────

_PRODUCTS: List[Product] = [
    # Life-critical (scored under crit_sl)
    Product("BLOOD-RBC",  "Packed Red Blood Cells",       42,  "CRITICAL", 220.0,
            ["ward_icu", "ward_er"], 6.0, 1.2),
    Product("BLOOD-PLT",  "Platelets",                    5,   "CRITICAL", 450.0,
            ["ward_icu", "ward_er"], 3.0, 0.8),
    Product("BLOOD-FFP",  "Fresh Frozen Plasma",          365, "CRITICAL", 180.0,
            ["ward_icu", "ward_er"], 4.0, 1.0),

    # Important consumables
    Product("IV-SAL-500", "IV Saline 500ml",              540, "HIGH",     3.5,
            ["ward_icu", "ward_er", "ward_general"], 20.0, 3.5),
    Product("ANTIBIO-01", "Broad-spectrum Antibiotics",   365, "HIGH",     15.0,
            ["ward_icu", "ward_er", "ward_general"], 12.0, 2.0),
    Product("OXY-MASK",   "Oxygen Masks",                 None,"HIGH",     4.0,
            ["ward_icu", "ward_er"], 8.0, 1.5),

    # Routine
    Product("SYR-10",     "10ml Syringes",                None,"NORMAL",   0.5,
            ["ward_icu", "ward_er", "ward_general"], 40.0, 5.0),
    Product("GLOVE-001",  "Surgical Gloves (box)",        None,"NORMAL",   5.0,
            ["ward_icu", "ward_er", "ward_general"], 25.0, 3.0),
    Product("MASK-001",   "Surgical Masks",               None,"NORMAL",   1.0,
            ["ward_icu", "ward_er", "ward_general"], 30.0, 4.0),
    Product("GAUZE-01",   "Sterile Gauze Pads",           None,"NORMAL",   0.8,
            ["ward_icu", "ward_er", "ward_general"], 35.0, 4.5),
]

_ALL_SKUS = [p.product_id for p in _PRODUCTS]

_SUPPLIERS: List[Supplier] = [
    Supplier("MEDLINE",  "MedLine Primary",     base_lead_time=2, lead_time_std=0.0,
             cost_multiplier=1.0, products=_ALL_SKUS),
    Supplier("BACKUP-B", "BackupCo",            base_lead_time=3, lead_time_std=0.0,
             cost_multiplier=1.3, products=_ALL_SKUS),
    Supplier("FASTMED",  "FastMed Expedited",   base_lead_time=1, lead_time_std=0.0,
             cost_multiplier=1.8, products=_ALL_SKUS),
]

_LOCATIONS: List[Location] = [
    Location("central_pharmacy", "Central Pharmacy"),
    Location("ward_icu",         "Intensive Care Unit"),
    Location("ward_er",          "Emergency Room"),
    Location("ward_general",     "General Ward"),
]

# Justification template banks
_ICU_HONEST = [
    "ICU patient census steady; requesting standard replenishment to cover next 2 days.",
    "Running low on {sku}; projected consumption matches current census.",
    "Replenishing safety stock after recent ventilated-patient transfers.",
]
_ICU_PADDED = [
    "Buffering against possible critical admits; request covers worst-case.",
    "Want a small safety margin on top of projected need given volatile census.",
]
_ER_HONEST = [
    "ER volume normal; standard request.",
    "Restock request covers expected walk-ins for the next cycle.",
    "Trauma bay needs replenishing after recent cases.",
]
_ER_PADDED = [
    "Anticipating possible weekend surge; requesting above baseline.",
    "Request includes contingency for potential multi-casualty arrivals.",
    "Precautionary buffer for unpredictable surge volumes.",
]
_ER_EVENT = [
    "Mass casualty incident in progress — actual consumption will be at full request volume.",
    "Active MCI response; blood and critical supplies needed at surge rate.",
    "Trauma bay at full load — request reflects real need during crisis.",
]
_GEN_HONEST = [
    "General ward steady; standard replenishment request.",
    "Routine request to top up par levels.",
]
_GEN_PADDED = [
    "Request includes safety buffer to avoid stockouts.",
    "Ordering slightly above baseline for operational headroom.",
    "Want to maintain comfortable stock levels through the weekend.",
    "Standard order with contingency buffer for admissions variability.",
]


def _make_wards(difficulty: str) -> List[WardConfig]:
    """Difficulty tunes General's pad aggression and ER's non-event spike rate."""
    general_pad_prob = {"light": 0.65, "medium": 0.75, "heavy": 0.85}[difficulty]
    er_spike_prob   = {"light": 0.10, "medium": 0.25, "heavy": 0.40}[difficulty]
    return [
        WardConfig(
            ward_id="ward_icu", name="ICU",
            priority_weight=1.0, pad_prob=0.10, pad_lo=1.05, pad_hi=1.15,
            products_tracked=["BLOOD-RBC", "BLOOD-PLT", "BLOOD-FFP",
                              "IV-SAL-500", "ANTIBIO-01", "OXY-MASK",
                              "SYR-10", "GLOVE-001"],
            honest_justifications=_ICU_HONEST,
            padded_justifications=_ICU_PADDED,
        ),
        WardConfig(
            ward_id="ward_er", name="ER",
            priority_weight=0.7, pad_prob=0.20, pad_lo=1.20, pad_hi=1.50,
            products_tracked=["BLOOD-RBC", "BLOOD-PLT", "BLOOD-FFP",
                              "IV-SAL-500", "ANTIBIO-01", "OXY-MASK",
                              "SYR-10", "GLOVE-001", "GAUZE-01"],
            honest_justifications=_ER_HONEST,
            padded_justifications=_ER_PADDED,
            spike_prob=er_spike_prob,
            spike_multiplier=2.0,
        ),
        WardConfig(
            ward_id="ward_general", name="General Ward",
            priority_weight=0.3, pad_prob=general_pad_prob,
            pad_lo=1.25, pad_hi=1.60,
            products_tracked=["IV-SAL-500", "ANTIBIO-01",
                              "SYR-10", "GLOVE-001", "MASK-001", "GAUZE-01"],
            honest_justifications=_GEN_HONEST,
            padded_justifications=_GEN_PADDED,
        ),
    ]


# ─── Event bank ──────────────────────────────────────────────────────────────

def _make_mci_event(trigger_day: int) -> List[SimEvent]:
    """MCI warning (trigger_day-1) + activation (trigger_day) that spikes blood demand."""
    return [
        SimEvent(
            event_id=f"mci_warn_{trigger_day}",
            event_type="mci_warning",
            trigger_day=max(1, trigger_day - 1),
            duration_days=0,
            params={"expected_activation_day": trigger_day},
            message=InboxMessageTemplate(
                priority="HIGH", sender="Trauma Command",
                subject="MCI STANDBY — Mass Casualty Warning",
                body=(
                    f"Multi-vehicle incident reported. Expect elevated trauma load starting Day {trigger_day}.\n"
                    "Recommend pre-positioning blood products and critical consumables at ICU/ER.\n"
                    "ACTION SUGGESTED: place expedited orders for BLOOD-RBC, BLOOD-PLT, BLOOD-FFP."
                ),
            ),
        ),
        SimEvent(
            event_id=f"mci_{trigger_day}",
            event_type="mci",
            trigger_day=trigger_day,
            duration_days=3,
            params={
                "locations": ["ward_icu", "ward_er"],
                "demand_multiplier": 2.8,
            },
            message=InboxMessageTemplate(
                priority="CRITICAL", sender="Trauma Command",
                subject="MCI ACTIVE — Surge Demand Confirmed",
                body=(
                    "Mass casualty incident active. Blood products and critical SKUs\n"
                    "will run at ~2.8× normal consumption at ICU/ER for the next 3 days."
                ),
            ),
        ),
    ]


def _make_supplier_disruption(trigger_day: int, supplier_id: str = "MEDLINE") -> SimEvent:
    return SimEvent(
        event_id=f"supdis_{supplier_id}_{trigger_day}",
        event_type="supplier_disruption",
        trigger_day=trigger_day,
        duration_days=4,
        params={
            "supplier_id": supplier_id,
            "new_lead_time": 5,
            "reason": "Warehouse fire at primary distribution center.",
        },
        message=InboxMessageTemplate(
            priority="HIGH", sender="Procurement",
            subject=f"Supplier Disruption — {supplier_id}",
            body=(
                f"{supplier_id} reports 5-day lead time for the next 4 days.\n"
                "Consider switching urgent orders to BACKUP-B or FASTMED."
            ),
        ),
    )


def _make_recall(trigger_day: int) -> SimEvent:
    return SimEvent(
        event_id=f"recall_iv_{trigger_day}",
        event_type="product_recall",
        trigger_day=trigger_day,
        duration_days=0,
        params={
            "product_id": "IV-SAL-500",
            "recall_lot_id": "RECALL-LOT-IV-9821",
            "locations_with_lot": ["ward_icu", "ward_er", "ward_general"],
            "qty_per_location": 30,
        },
        message=InboxMessageTemplate(
            priority="CRITICAL", sender="Quality Authority",
            subject="MANDATORY RECALL — IV-SAL-500 Lot RECALL-LOT-IV-9821",
            body=(
                "Voluntary recall of IV Saline 500ml, lot RECALL-LOT-IV-9821.\n"
                "ACTION REQUIRED: quarantine this lot at every ward and submit replacement orders."
            ),
        ),
    )


def _make_cold_chain(trigger_day: int) -> SimEvent:
    return SimEvent(
        event_id=f"coldchain_{trigger_day}",
        event_type="cold_chain_breach",
        trigger_day=trigger_day,
        duration_days=0,
        params={
            "location_id": "ward_icu",
            "product_id": "BLOOD-PLT",
        },
        message=InboxMessageTemplate(
            priority="CRITICAL", sender="Quality Authority",
            subject="Cold Chain Breach — BLOOD-PLT at ward_icu",
            body=(
                "Refrigerator alarm at ICU. Platelet supply compromised.\n"
                "Affected lots auto-quarantined. Replenishment required urgently."
            ),
        ),
    )


_EVENT_BUILDERS: List[Tuple[str, callable]] = [
    ("mci", _make_mci_event),
    ("supplier_disruption", _make_supplier_disruption),
    ("product_recall", _make_recall),
    ("cold_chain_breach", _make_cold_chain),
]


def _sample_events(rng: random.Random, difficulty: str, max_days: int) -> List[SimEvent]:
    """Sample 1-5 events; higher difficulty → more events."""
    n_events = {
        "light":  rng.randint(1, 2),
        "medium": rng.randint(2, 3),
        "heavy":  rng.randint(3, 5),
    }[difficulty]

    builders = list(_EVENT_BUILDERS)
    rng.shuffle(builders)

    chosen: List[SimEvent] = []
    used_types: set = set()
    # Ensure events spread across the episode
    candidate_days = list(range(2, max_days - 1))
    rng.shuffle(candidate_days)

    idx = 0
    while len(chosen) < n_events and idx < len(builders) and candidate_days:
        event_type, builder = builders[idx % len(builders)]
        idx += 1
        if event_type in used_types and difficulty != "heavy":
            continue
        used_types.add(event_type)
        day = candidate_days.pop()
        produced = builder(day)
        if isinstance(produced, list):
            chosen.extend(produced)
        else:
            chosen.append(produced)

    chosen.sort(key=lambda e: e.trigger_day)
    return chosen


# ─── Main factory ─────────────────────────────────────────────────────────────

def make_task_config(seed: int = 0, difficulty: str = "medium") -> TaskConfig:
    """
    Build a TaskConfig for the multi_actor_coordination task.

    The seed controls: event sampling, initial stock levels, and (downstream in
    simulation) ward padding RNG + synthetic history RNG.

    difficulty ∈ {"light", "medium", "heavy"} controls event count and
    General ward padding aggression.

    Training seeds: 0-49. Held-out eval seeds: 50-59.
    """
    if difficulty not in ("light", "medium", "heavy"):
        raise ValueError(f"Invalid difficulty: {difficulty!r}")

    rng = random.Random(seed * 7919 + 101)

    max_rounds = 8
    round_length_days = 2
    max_days = max_rounds * round_length_days  # 16

    events = _sample_events(rng, difficulty, max_days)

    # Initial stock: loose base × 2-5 days depending on seed
    stock_mult_by_sku: Dict[str, float] = {}
    for p in _PRODUCTS:
        stock_mult_by_sku[p.product_id] = rng.uniform(2.0, 5.0)

    # Budget scales with event count and ward count
    base_budget = 25_000.0 + 5_000.0 * len(events)
    budget_limit = round(base_budget, -2)

    # Benchmark spend is a back-of-envelope estimate used by budget_efficiency
    total_base_demand = sum(p.base_demand * len(p.locations) for p in _PRODUCTS)
    avg_unit_cost = (
        sum(p.unit_cost * p.base_demand for p in _PRODUCTS)
        / max(sum(p.base_demand for p in _PRODUCTS), 1)
    )
    benchmark_spend = total_base_demand * avg_unit_cost * max_days * 1.1

    return TaskConfig(
        name="multi_actor_coordination",
        seed=seed,
        difficulty=difficulty,
        max_rounds=max_rounds,
        round_length_days=round_length_days,
        max_days=max_days,
        budget_limit=budget_limit,
        products=list(_PRODUCTS),
        suppliers=list(_SUPPLIERS),
        locations=list(_LOCATIONS),
        wards=_make_wards(difficulty),
        events=events,
        initial_stock_days=stock_mult_by_sku,
        synthetic_history_rounds=8,
        benchmark_spend=benchmark_spend,
        ward_actor_configs=_make_ward_actor_configs(difficulty),
        approval_threshold=10_000.0,
        quote_resolution_turns=1,
        wms_noise_pct=0.05,
    )


# ─── Backwards-compat alias (used by existing MedchainEnvironment.__init__) ──

def get_task_config(task_name: str = "multi_actor_coordination",
                    seed: int = 0,
                    difficulty: str = "medium") -> TaskConfig:
    """Single-task factory. `task_name` is ignored — there is only one task."""
    return make_task_config(seed=seed, difficulty=difficulty)
