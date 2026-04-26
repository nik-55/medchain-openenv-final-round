"""
MedChain — Rubber-stamp Rollout Collection (local, CPU-only)

Drives MedchainSimulation forward with a rubber-stamp allocation policy
to collect per-round training rows. Pushes the result to HF Hub so
train_hf_offline.py can load it.

Usage:
    python collect_rollouts.py
    python collect_rollouts.py --seeds 0-49 --difficulties light medium heavy
    python collect_rollouts.py --output-file rollouts.jsonl   # skip HF push
"""

import argparse
import json
import os
import uuid

from datasets import Dataset
from huggingface_hub import login as hf_login
from server.simulation import MedchainSimulation
from server.tasks import make_task_config

HF_DATASET_ID = "nik-55/medchain-rollouts"

WARDS = ["ward_icu", "ward_er", "ward_general"]

ALLOC_SYSTEM_PROMPT = """\
You are the central supply-chain coordinator for a hospital network.
Three wards submit allocation requests each round. Some requests are inflated.
Use the round context to output the allocation plan.

OUTPUT: One line — a JSON object {"ward_id": {"product_id": qty}}.
Include ALL THREE wards and ALL their SKUs. qty = non-negative integer.

Valid ward IDs:    ward_icu  ward_er  ward_general
ward_icu SKUs:     BLOOD-RBC  BLOOD-PLT  BLOOD-FFP  IV-SAL-500  ANTIBIO-01  OXY-MASK  SYR-10  GLOVE-001
ward_er SKUs:      BLOOD-RBC  BLOOD-PLT  BLOOD-FFP  IV-SAL-500  ANTIBIO-01  OXY-MASK  SYR-10  GLOVE-001  GAUZE-01
ward_general SKUs: IV-SAL-500  ANTIBIO-01  SYR-10  GLOVE-001  MASK-001  GAUZE-01

Rules:
- Never allocate more than requested. Sum per SKU ≤ central pharmacy stock.
- Padding patterns: ICU ~10%, ER ~20-50%, General ~25-60%.
  Target: ICU ≈ request, ER × 0.75-0.85, General × 0.55-0.75.
- Blood products at ICU+ER are CRITICAL — never stockout.
- Active MCI event ([CRITICAL] inbox): trust ER fully for blood products.
- Output ONLY the JSON. No explanation, no markdown."""


def _central_stock(sim: MedchainSimulation) -> dict[str, int]:
    state = sim._state
    stock: dict[str, int] = {}
    for (loc, sku), lots in state.inventory.items():
        if loc == "central_pharmacy":
            stock[sku] = stock.get(sku, 0) + sum(
                l.qty for l in lots if l.lot_id not in state.quarantined_lots
            )
    return stock


def rubber_stamp(sim: MedchainSimulation) -> dict[str, dict[str, int]]:
    """Approve full request capped by central stock, then reorder what was allocated.

    Submitting a MEDLINE PO for each allocated SKU keeps central stock levels
    stable across rounds (lead time 2 days = arrives next 2-day round), so
    later rounds have realistic stock and meaningful training signal.
    """
    remaining = _central_stock(sim)
    plan: dict[str, dict[str, int]] = {}
    sku_allocated: dict[str, int] = {}

    for req in sim._state.pending_requests:
        take = min(req.requested_qty, remaining.get(req.product_id, 0))
        plan.setdefault(req.ward_id, {})[req.product_id] = take
        remaining[req.product_id] = remaining.get(req.product_id, 0) - take
        sku_allocated[req.product_id] = sku_allocated.get(req.product_id, 0) + take

    # Reorder exactly what was allocated so the next round starts at similar stock
    for sku, qty in sku_allocated.items():
        if qty > 0:
            try:
                sim.submit_po(
                    supplier_id="MEDLINE",
                    product_id=sku,
                    destination_id="central_pharmacy",
                    quantity=qty,
                    priority="standard",
                )
            except Exception:
                pass  # skip SKUs unavailable from MEDLINE

    return plan


def collect_episode(
    seed: int,
    difficulty: str,
    rows: list[dict],
    next_idx: int,
) -> int:
    sim = MedchainSimulation(make_task_config(seed=seed, difficulty=difficulty))
    sim.reset(seed=seed, episode_id=str(uuid.uuid4()), difficulty=difficulty)

    while not sim._done:
        state = sim._state
        if not state.pending_requests:
            break

        inbox_text    = sim.read_inbox(filter="unread")
        requests_text = sim.view_requests()
        history_parts = [sim.query_ward_history(w, n_rounds=5) for w in WARDS]
        stock_text    = sim.query_erp(table="inventory", location="central_pharmacy")
        pipeline_text = sim.query_erp(table="pipeline_orders")

        true_needs: dict[str, dict] = {}
        requested:  dict[str, dict] = {}
        for req in state.pending_requests:
            true_needs.setdefault(req.ward_id, {})[req.product_id] = req.true_need
            requested.setdefault(req.ward_id,  {})[req.product_id] = req.requested_qty

        priority = {w.ward_id: w.priority_weight for w in sim._task.wards}
        cs       = _central_stock(sim)

        user_msg = (
            f"=== ROUND {state.round_idx}/{state.max_rounds}  day {state.day} ===\n\n"
            f"INBOX (unread):\n{inbox_text}\n\n"
            f"WARD REQUESTS:\n{requests_text}\n\n"
            f"CENTRAL PHARMACY STOCK:\n{stock_text}\n\n"
            f"PIPELINE ORDERS (in-transit):\n{pipeline_text}\n\n"
            "WARD HISTORY (last 5 rounds):\n"
            + "\n\n".join(history_parts)
            + "\n\nYour allocation plan (single-line JSON):"
        )

        rows.append({
            "_rollout_idx":  next_idx,
            "prompt": [
                {"role": "system", "content": ALLOC_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "true_needs":    json.dumps(true_needs),
            "requested":     json.dumps(requested),
            "priority":      json.dumps(priority),
            "central_stock": json.dumps(cs),
            "seed":          seed,
            "difficulty":    difficulty,
            "round_idx":     state.round_idx,
        })
        next_idx += 1

        plan = rubber_stamp(sim)
        sim.submit_allocation_plan(json.dumps(plan))
        sim.advance_round()

    return next_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", default="0-49",
        help="Seed range as 'start-end' (inclusive) or comma-separated list",
    )
    parser.add_argument(
        "--difficulties", nargs="+", default=["light", "medium", "heavy"],
        choices=["light", "medium", "heavy"],
    )
    parser.add_argument(
        "--output-file", default=None,
        help="Save to local JSONL file instead of pushing to HF Hub",
    )
    parser.add_argument(
        "--hf-dataset-id", default=HF_DATASET_ID,
        help=f"HF dataset repo to push to (default: {HF_DATASET_ID})",
    )
    args = parser.parse_args()

    # Parse seed spec
    if "-" in args.seeds and "," not in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    total = len(seeds) * len(args.difficulties)
    print(f"Collecting: {len(seeds)} seeds × {len(args.difficulties)} difficulties "
          f"= {total} episodes  (~{total * 8} rows)")

    rows: list[dict] = []
    idx  = 0
    done = 0

    for seed in seeds:
        for diff in args.difficulties:
            before = len(rows)
            idx = collect_episode(seed, diff, rows, idx)
            done += 1
            if done % 15 == 0 or done == total:
                print(f"  [{done:3d}/{total}]  rows={len(rows)}  "
                      f"last_episode={len(rows)-before} rounds")

    print(f"\nDone: {len(rows)} rows collected")

    if args.output_file:
        with open(args.output_file, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"Saved → {args.output_file}")
    else:
        hf_token = os.environ.get("HF_TOKEN", "")
        hf_login(token=hf_token)
        ds = Dataset.from_list(rows)
        print(f"Pushing {len(ds)} rows → {args.hf_dataset_id} ...")
        ds.push_to_hub(args.hf_dataset_id, token=hf_token)
        print(f"Done → https://huggingface.co/datasets/{args.hf_dataset_id}")


if __name__ == "__main__":
    main()
