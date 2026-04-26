"""
ASCII table formatters for the MEDSUPPLY ERP v2.1 interface.

All functions receive SimState + TaskConfig and return plain text strings
for display in an LLM chat interface.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import SimState
    from .tasks import Product, Supplier, TaskConfig

_W = 70  # default box width


def _box(lines: list[str], width: int = _W) -> str:
    """Wrap a list of content strings in a simple Unicode box."""
    inner_width = width - 2
    top    = "╔" + "═" * inner_width + "╗"
    mid    = "╠" + "═" * inner_width + "╣"
    bottom = "╚" + "═" * inner_width + "╝"

    rows = [top]
    for i, line in enumerate(lines):
        if line == "---DIVIDER---":
            rows.append(mid)
        else:
            padded = ("║ " + line).ljust(inner_width + 1) + "║"
            rows.append(padded)
    rows.append(bottom)
    return "\n".join(rows)


def _sep(width: int = _W) -> str:
    return "-" * width


def _status(days_left) -> str:
    if days_left is None:
        return "NON-PERISH"
    if days_left <= 0:
        return "*** EXPIRED ***"
    if days_left <= 3:
        return "WARN_CRITICAL"
    if days_left <= 7:
        return "WARN_LOW"
    return "OK"


# ─── Dashboard ───────────────────────────────────────────────────────────────

def format_dashboard(state: "SimState", task_config: "TaskConfig") -> str:
    """
    Returns the ERP dashboard — kept as a helper even though round briefs
    are now the primary round-opening text.
    """
    unread = sum(1 for m in state.inbox if not m.read)
    in_transit = sum(1 for po in state.pipeline_orders if po.status == "in_transit")

    supplier_lines = [
        f"  {s.supplier_id} → base lead {s.base_lead_time}d, ×{s.cost_multiplier:.1f}"
        for s in task_config.suppliers
    ]

    lines = [
        "  MEDSUPPLY ERP — CENTRAL HOSPITAL NETWORK",
        f"  Task: {task_config.name} (difficulty={task_config.difficulty}, seed={task_config.seed})",
        f"  Round: {state.round_idx}/{state.max_rounds} | Day {state.day}/{state.max_days}",
        f"  Budget used: ${state.budget_used:,.0f} / ${state.budget_limit:,.0f}",
        f"  Inbox: {unread} unread | Pipeline: {in_transit} in transit",
        "---DIVIDER---",
        "  SUPPLIERS:",
    ] + supplier_lines
    return _box(lines)


# ─── Inventory Table ─────────────────────────────────────────────────────────

def format_inventory_table(
    state: "SimState",
    task_config: "TaskConfig",
    location: str,
    sku: str,
) -> str:
    loc_filter = location.lower()
    sku_filter = sku.lower()

    loc_label = location.upper() if loc_filter != "all" else "ALL"
    sku_label = sku.upper() if sku_filter != "all" else "ALL"

    header = [
        f"SYSTEM QUERY RESULT [TABLE: INVENTORY] [LOC: {loc_label}] [SKU: {sku_label}]",
        f"[TIMESTAMP: Day {state.day}]",
    ]
    sep = _sep()
    col_header = f"{'LOT_ID':<22} | {'DESC':<24} | {'QTY':>5} | {'EXP_DAY':>7} | {'DAYS_LEFT':>9} | STATUS"
    rows = []

    product_map = {p.product_id: p for p in task_config.products}

    for (loc_id, product_id), lots in sorted(state.inventory.items()):
        if loc_filter != "all" and loc_id.lower() != loc_filter:
            continue
        if sku_filter != "all" and product_id.lower() != sku_filter:
            continue

        product = product_map.get(product_id)
        desc = product.name if product else product_id

        for lot in lots:
            if lot.qty == 0:
                continue
            is_quarantined = lot.lot_id in state.quarantined_lots
            days_left = (lot.expiry_day - state.day) if lot.expiry_day is not None else None
            status = "[QUARANTINED]" if is_quarantined else _status(days_left)
            exp_str = f"{lot.expiry_day:07d}" if lot.expiry_day is not None else "N/A    "
            dl_str  = f"{days_left:09d}" if days_left is not None else "N/A      "
            row = f"{lot.lot_id:<22} | {desc[:24]:<24} | {lot.qty:>5} | {exp_str:>7} | {dl_str:>9} | {status}"
            rows.append(row)

    lines = header + [sep, col_header, sep]
    if rows:
        lines += rows
    else:
        lines.append("(no stock found)")
    lines += [sep, f"QUERY OK | {len(rows)} row(s) returned"]
    return "\n".join(lines)


# ─── Expiry Table ─────────────────────────────────────────────────────────────

def format_expiry_table(
    state: "SimState",
    task_config: "TaskConfig",
    location: str,
    sku: str,
) -> str:
    loc_filter = location.lower()
    sku_filter = sku.lower()

    header = [
        "SYSTEM QUERY RESULT [TABLE: EXPIRY] [Lots expiring within 14 days]",
        f"[TIMESTAMP: Day {state.day}]",
    ]
    sep = _sep()
    col_header = f"{'LOT_ID':<22} | {'LOC':<16} | {'SKU':<12} | {'QTY':>5} | {'EXP_DAY':>7} | {'DAYS_LEFT':>9} | STATUS"
    rows = []

    product_map = {p.product_id: p for p in task_config.products}

    for (loc_id, product_id), lots in sorted(state.inventory.items()):
        if loc_filter != "all" and loc_id.lower() != loc_filter:
            continue
        if sku_filter != "all" and product_id.lower() != sku_filter:
            continue
        for lot in lots:
            if lot.qty == 0:
                continue
            if lot.expiry_day is None:
                continue
            days_left = lot.expiry_day - state.day
            if days_left > 14:
                continue
            is_quarantined = lot.lot_id in state.quarantined_lots
            status = "[QUARANTINED]" if is_quarantined else _status(days_left)
            row = (
                f"{lot.lot_id:<22} | {loc_id:<16} | {product_id:<12} | "
                f"{lot.qty:>5} | {lot.expiry_day:>7} | {days_left:>9} | {status}"
            )
            rows.append(row)

    lines = header + [sep, col_header, sep]
    if rows:
        lines += rows
    else:
        lines.append("(no expiry warnings)")
    lines += [sep, f"QUERY OK | {len(rows)} row(s) returned"]
    return "\n".join(lines)


# ─── Pipeline Orders Table ────────────────────────────────────────────────────

def format_pipeline_table(
    state: "SimState",
    location: str,
    sku: str,
) -> str:
    loc_filter = location.lower()
    sku_filter = sku.lower()

    header = ["SYSTEM QUERY RESULT [TABLE: PIPELINE_ORDERS]", f"[TIMESTAMP: Day {state.day}]"]
    sep = _sep()
    col_header = f"{'PO_ID':<9} | {'SUPPLIER':<12} | {'SKU':<12} | {'DESTINATION':<16} | {'QTY':>5} | {'PRIORITY':<10} | {'ETA':>5} | STATUS"
    rows = []

    for po in state.pipeline_orders:
        if loc_filter != "all" and po.destination_id.lower() != loc_filter:
            continue
        if sku_filter != "all" and po.product_id.lower() != sku_filter:
            continue
        status = po.status.upper().replace("_", " ")
        row = (
            f"{po.po_id:<9} | {po.supplier_id:<12} | {po.product_id:<12} | "
            f"{po.destination_id:<16} | {po.quantity:>5} | {po.priority:<10} | "
            f"D-{po.eta_day:02d} | {status}"
        )
        rows.append(row)

    lines = header + [sep, col_header, sep]
    if rows:
        lines += rows
    else:
        lines.append("(no orders in pipeline)")
    lines += [sep, f"QUERY OK | {len(rows)} row(s) returned"]
    return "\n".join(lines)


# ─── Supplier Info ────────────────────────────────────────────────────────────

def format_supplier_info(
    supplier: "Supplier",
    effective_lead_time: int,
    disruption_note: str,
) -> str:
    sep = _sep()
    products_str = ", ".join(supplier.products)
    lines = [
        f"SUPPLIER INFO: {supplier.supplier_id}",
        sep,
        f"Name:          {supplier.name}",
        f"Status:        ACTIVE",
        f"Lead Time:     {effective_lead_time} days (effective)",
        f"Base Lead:     {supplier.base_lead_time} days",
        f"Cost Mult:     {supplier.cost_multiplier:.1f}× base price",
        f"Products:      {products_str}",
        f"Notes:         {disruption_note}",
        sep,
    ]
    return "\n".join(lines)


# ─── Round briefing (one-shot dashboard) ─────────────────────────────────────

def format_briefing(state: "SimState", task_config: "TaskConfig") -> str:
    """One-shot situational briefing — replaces the typical 4-tool sequence
    of read_inbox + view_requests + query_erp(inventory) + query_erp(pipeline).
    Designed for a single tool call at the top of each round.
    """
    sep = _sep()

    # Active events
    active_events = []
    for eid, last_day in state.active_events.items():
        ev = next((e for e in task_config.events if e.event_id == eid), None)
        if ev:
            active_events.append(f"{ev.event_type} (ends day {last_day})")

    # Unread inbox subjects
    unread = [m for m in state.inbox if not m.read]
    inbox_lines = [
        f"  [{m.priority}] {m.subject} — {m.sender}"
        for m in unread[:8]
    ] or ["  (no unread messages)"]

    # Pending requests (compact)
    ward_priority = {w.ward_id: w.priority_weight for w in task_config.wards}
    req_lines = []
    for r in state.pending_requests:
        prio = ward_priority.get(r.ward_id, 0.5)
        req_lines.append(
            f"  {r.ward_id:<14} {r.product_id:<12} qty={r.requested_qty:>4} "
            f"prio={prio:.1f}  '{r.justification[:50]}'"
        )
    if not req_lines:
        req_lines = ["  (no pending requests)"]

    # Central pharmacy on-hand (top SKUs by qty)
    central_lines: list[str] = []
    central_totals: dict[str, int] = {}
    for (loc, sku), lots in state.inventory.items():
        if loc != "central_pharmacy":
            continue
        qty = sum(l.qty for l in lots if l.lot_id not in state.quarantined_lots)
        if qty > 0:
            central_totals[sku] = qty
    for sku, qty in sorted(central_totals.items(), key=lambda x: -x[1])[:10]:
        central_lines.append(f"  {sku:<12} {qty:>5} units on-hand")
    if not central_lines:
        central_lines = ["  (central pharmacy empty)"]

    # In-transit pipeline
    pipe_lines = [
        f"  PO {po.po_id} {po.product_id:<12} → {po.destination_id:<14} "
        f"qty={po.quantity:>4} ETA day {po.eta_day} ({po.status})"
        for po in state.pipeline_orders if po.status == "in_transit"
    ][:8] or ["  (no in-transit orders)"]

    # Ward reputations + recent stockouts (Theme #1 signal)
    reputation_lines = []
    rep_state = getattr(state, "ward_actor_state", {}) or {}
    for w in task_config.wards:
        s = rep_state.get(w.ward_id, {})
        rep = s.get("reputation", 0.5)
        recent = s.get("recent_stockouts", 0)
        reputation_lines.append(
            f"  {w.ward_id:<14} reputation={rep:.2f}  recent_stockouts={recent}"
        )
    if not reputation_lines:
        reputation_lines = ["  (no reputation data — round 1)"]

    return "\n".join([
        f"=== ROUND {state.round_idx}/{state.max_rounds} BRIEFING (day {state.day}) ===",
        f"Budget: ${state.budget_used:,.0f} used / ${state.budget_limit:,.0f} limit",
        f"Active events: {', '.join(active_events) if active_events else 'none'}",
        sep,
        "INBOX (unread):",
        *inbox_lines,
        sep,
        "PENDING WARD REQUESTS:",
        *req_lines,
        sep,
        "CENTRAL PHARMACY (top SKUs):",
        *central_lines,
        sep,
        "PIPELINE (in-transit):",
        *pipe_lines,
        sep,
        "WARD REPUTATIONS:",
        *reputation_lines,
        sep,
        f"Pending finance approvals: {len(getattr(state, 'pending_approvals', {}) or {})}",
        f"Pending supplier quotes:   {len(getattr(state, 'pending_quotes', {}) or {})}",
        sep,
    ])


# ─── WMS noisy inventory ─────────────────────────────────────────────────────

def format_wms_inventory(
    state: "SimState",
    task_config: "TaskConfig",
    location: str,
    sku: str,
    noise_pct: float,
) -> str:
    """Real-time scan with deterministic ±noise_pct per (round, lot) pair.
    Useful for cross-checking against erp_oracle (stale snapshot) — agent
    that reconciles both gets a more accurate picture.
    """
    loc_filter = location.lower()
    sku_filter = sku.lower()

    header = [
        f"WMS WAREHOUSE SCAN [LOC: {location.upper()}] [SKU: {sku.upper()}]",
        f"[TIMESTAMP: Day {state.day} {state.round_idx}r live]  noise=±{noise_pct*100:.0f}%",
    ]
    sep = _sep()
    col = f"{'LOT_ID':<22} | {'LOC':<16} | {'SKU':<12} | {'SCANNED_QTY':>11} | STATUS"
    rows: list[str] = []

    for (loc_id, product_id), lots in sorted(state.inventory.items()):
        if loc_filter != "all" and loc_id.lower() != loc_filter:
            continue
        if sku_filter != "all" and product_id.lower() != sku_filter:
            continue
        for lot in lots:
            if lot.qty == 0:
                continue
            is_q = lot.lot_id in state.quarantined_lots
            seed = abs(hash((state.round_idx, lot.lot_id))) % (2**32)
            rng = math.sin(seed) * 10_000
            noise_factor = 1.0 + ((rng - math.floor(rng)) - 0.5) * 2 * noise_pct
            scanned = max(0, int(round(lot.qty * noise_factor)))
            status = "[QUARANTINED]" if is_q else "OK"
            rows.append(
                f"{lot.lot_id:<22} | {loc_id:<16} | {product_id:<12} | "
                f"{scanned:>11} | {status}"
            )

    lines = header + [sep, col, sep]
    if rows:
        lines += rows
    else:
        lines.append("(no lots scanned)")
    lines += [sep, f"WMS OK | {len(rows)} lot(s) | reconcile against erp_oracle for ground truth"]
    return "\n".join(lines)


# ─── Supplier-portal quote ───────────────────────────────────────────────────

def format_supplier_quote(quote: dict) -> str:
    sep = _sep()
    if quote.get("status") == "pending":
        return "\n".join([
            f"QUOTE {quote['quote_id']} — STATUS: PENDING",
            sep,
            f"Submitted on day {quote['submitted_day']} | resolves on day {quote['resolves_day']}",
            "Call supplier_portal_get_quote again next round.",
            sep,
        ])
    return "\n".join([
        f"QUOTE {quote['quote_id']} — STATUS: READY",
        sep,
        f"Supplier:     {quote['supplier_id']}",
        f"Product:      {quote['product_id']}  ×  {quote['quantity']}",
        f"Lead Time:    {quote['lead_time']} days",
        f"Unit Cost:    ${quote['unit_cost']:.2f}",
        f"Total Cost:   ${quote['total_cost']:,.2f}",
        "Pass these terms to submit_po to commit.",
        sep,
    ])


