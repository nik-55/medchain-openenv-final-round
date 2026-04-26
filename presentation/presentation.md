---
marp: true
theme: default
paginate: true
---

# Hospital Supply Chain Coordination
### The Domain

<!--
Voice note: Start with the real-world problem — not the simulation. This is a job that exists in every hospital. Set the scene before introducing MedChain.
-->

---

## The Job

It's 2 AM. You're the central pharmacy coordinator for a three-ward hospital.

- **ICU** needs packed red blood cells for a patient going into surgery in 4 hours
- **ER** just submitted a 3× epinephrine request — real mass-casualty surge, or padding?
- **General ward** wants antibiotics again, 40% above actual consumption, as it does every week

You have **five siloed systems** open in front of you:
- ERP shows *yesterday's* inventory
- Warehouse scanner is live but gives ±5% noise per lot
- Supplier portal has quotes pending — from *last round's* request
- Finance will auto-reject any PO above $10k without a pre-approved justification
- Three wards are waiting on your allocation decision, which you must justify in writing

**This is a governance problem, not a logistics problem.** Wards don't negotiate — you audit, review, and decide. Disputed requests go to a binding clinical-review committee.

<!--
Voice note: Emphasize that the core difficulty is the governance layer — distinguishing legitimate clinical need from strategic inflation — on top of the already-hard operational problem of inventory management across siloed systems.
-->

---

## What We're Simulating — MedChain

**An OpenEnv-compliant simulation of this exact job**

- **3 wards**: ICU (high priority, honest), ER (volatile, legitimately surges), General (chronic padding)
- **5 enterprise systems**: Legacy ERP · Live WMS · Supplier Portal · Finance SAP · Ward Messaging
- **8 rounds** per episode, each = 2 simulated days
- Agent's job: allocate supplies, catch padding, manage procurement, justify every decision in writing

Ward actors are **persistent scripted agents** with distinct personalities and memory. Reputations carry across rounds — catch a ward padding and it pads less aggressively next round; let it slide and hoarding pressure builds.

<!--
Voice note: Now connect the domain to the simulation. The key point is that the wards are not passive inputs — they adapt. The agent's decisions in round 3 directly shape what round 4 looks like.
-->

---

## Why It's Hard to Reason About

Five distinct failure modes — each one a known gap in current LLMs:

| Failure mode | What goes wrong |
|---|---|
| **Multi-source reconciliation** | ERP stale −1 round · WMS live ±5% noise. Models pick one and act on it. |
| **Strategic actors** | ER pads on normal rounds AND during real surges. Distinguishing them requires cross-referencing census + acuity. |
| **Multi-step governance** | `request_evidence → review → escalate → cite in rationale` — models start the loop and don't finish it. |
| **Async temporal planning** | Supplier quotes take 1 round. An agent that requests at round 5 for a round-6 need creates the stockout itself. |
| **Finance gate sequencing** | POs > $10k need pre-approval. Models submit optimistically and wonder why it bounced. |

<!--
Voice note: These aren't artificial constraints. Stale ERP vs. live-but-noisy WMS is exactly how real hospital systems work. The finance gate sequence is standard procurement governance. The asymmetric information problem with ER is the actual challenge ward managers face.
-->

---

## Tools Available to the Agent

**21 MCP tools across 5 enterprise silos**

| System | Tools | Key constraint |
|---|---|---|
| **ERP Oracle** | `erp_oracle_get_inventory`, `erp_oracle_get_pipeline` | Stale by 1 round |
| **WMS** | `wms_scan_inventory` | Live, ±5% noise per lot |
| **Supplier Portal** | `supplier_portal_request_quote`, `supplier_portal_get_quote` | Async — response next round |
| **Finance SAP** | `finance_sap_get_budget`, `finance_sap_request_approval` | Gate for POs > $10k |
| **Ward Messaging** | `messaging_send_to_ward` | Free-form dialogue |

**Coordination tools:** `get_round_briefing` · `view_requests` · `read_inbox` · `submit_allocation_plan` · `advance_round`

**Audit tools:** `request_evidence` · `escalate_to_clinical_review`

**Procurement:** `submit_po` · `file_justification` · `quarantine_lot`

### Scripted Ward Actors

Each ward has a fixed persona, pad probability, and hoarding pressure — but **reputation drifts across rounds**. Catch a ward padding → its pressure drops. Let it slide → it pads harder next round.

<!--
Voice note: The scripted actors are deterministic by default (WARD_ACTOR_MODE=scripted). This is deliberate — it makes the RLVR contract clean. Two episodes with identical seed produce byte-identical reward decompositions. The clinical arbiter is also scripted and returns binding verdicts on escalation.
-->

---

## A Full Episode: Round 3 Walk-through

**Setup:** `seed=5, difficulty=medium`. General ward requests 180 units of ANTIBIO-01 (8-round average: 115 units).

```
Agent: get_round_briefing()
→ 3 pending requests, finance queue empty, 1 inbox message

Agent: view_requests()
→ ward_general: ANTIBIO-01 × 180  [flagged: +57% above history]

Agent: query_ward_history(ward_id="ward_general", n_rounds=8)
→ avg consumption 115/round, no documented surge

Agent: erp_oracle_get_inventory()  +  wms_scan_inventory()
→ ERP: 210 units on hand (as of yesterday)
→ WMS: 198 units (live, noisy) + 0 pipeline

Agent: request_evidence(ward_id="ward_general", sku="ANTIBIO-01", evidence_type="all")
→ census: +12% patients  |  acuity: unchanged  |  recent_actuals: [REDACTED]

Agent: escalate_to_clinical_review(ward_id="ward_general", sku="ANTIBIO-01",
         concern="Request 57% above history; recent actuals redacted")

Clinical Review Board: → REDUCE to 130 units  (binding)

Agent: submit_allocation_plan(plan_json={...}, rationale_json={
  "ward_general.ANTIBIO-01": "Reduced per clinical review: census +12% justifies
   115→130, not 180. Actuals redaction treated as adverse signal."
})

Agent: advance_round()
```

**Result:** audit_score +1, General ward hoarding pressure −0.08 into round 4.

<!--
Voice note: Walk through this slowly. The redacted `recent_actuals` is the key signal — a ward that knows its numbers are bad hides them. The agent has to notice the redaction, treat it as evidence, and cite it in the rationale. Just calling request_evidence and then writing a generic rationale scores zero.
-->

---

## Architecture & Workflow

![Architecture diagram](media/architecture.png)

**Episode shape:** 8 rounds × ~15 tool calls/round · Context reset after each `advance_round`

**Determinism contract:** `WARD_ACTOR_MODE=scripted` + fixed seed → byte-identical rewards. No LLM judge anywhere in the reward loop.

<!--
Voice note: The context reset is operationally important — without it, an 8-round episode accumulates 3000–8000 tokens and each generate() call takes 2–4 minutes on a T4. After reset, the round brief carries everything the model needs to continue; per-round context stays under ~1000 tokens.
-->

---

## How Rewards Are Given

**Fully deterministic — computed from `SimState`, no LLM judge**

```
0.25 × network_service_level       were actual consumption needs met?
0.18 × critical_service_level      ICU + ER blood products specifically
0.18 × allocation_accuracy         per-ward surplus + stockout penalty
0.12 × event_response              MCI · product recall · cold-chain · supplier disruption
0.07 × budget_efficiency
0.04 × waste_control
0.05 × audit_score                 mean(evidence_use_rate, escalation_accuracy)
0.05 × approval_workflow_score     finance gates resolved cleanly
0.03 × tool_discovery_score        used all 5 enterprise systems at least once
0.03 × briefing_efficiency         one dashboard call per round, not four
−      justification_penalty       vague or unsupported rationales (cap 0.15)
```

**Per-step shaping** (to guide early exploration):
`read_inbox +0.01` · `correct escalation +0.05` · `frivolous escalation −0.03`
`valid submit_allocation_plan +0.03` · `first use of any enterprise system +0.005`

> **`audit_score` quirk:** it measures whether evidence was *cited in the rationale*, not just retrieved. Call `request_evidence`, receive census data, write "allocated based on ward request" → score = 0.

<!--
Voice note: The decomposed reward is deliberate. Each of the five failure modes maps to at least one reward component — so the gradient is always informative even when the agent is only partially competent. A model that learns to handle finance gates correctly gets a signal even if it's still bad at the audit loop.
-->

---

## Improving the Agent with GRPO

**Model:** Qwen3.5-2B · 4-bit NF4 + LoRA (`r=16`) · trained on T4 via Google Colab

**Why GRPO?** Prompting can describe the audit workflow. Only RL can teach the model to *value* actions whose consequences arrive 2 rounds later.

### Training setup
- Group size G=4, batch B=8 → **32 live simulation episodes per step**
- Context reset to `[system, round_brief]` after each `advance_round`
- Custom `medchain_rollout` function — no static dataset, trains on real environment

### Results (~23 steps, compute-budget limited)

| Policy | Score |
|---|---|
| Rubber-stamp (allocate exactly requested) | ~0.41 |
| Discount-General heuristic (cut General 30%) | ~0.47 |
| **Trained Qwen3.5-2B (GRPO, ~23 steps)** | **~0.28–0.35** |
| Frontier model (full audit loop) | ~0.65–0.75 |

Reward mean: 0.13 → 0.28–0.35 over 23 steps. Still early in the training curve — clear headroom.

**What's next:** per-turn reward signals · batch grouping by episode length · full Tier-1 tool surface in training prompt

<!--
Voice note: Be honest about the numbers. The trained model is behind the heuristic baseline after 23 steps — that's expected. The trajectory is upward and the reward curve is clean. The bigger point is that the environment has a well-defined gradient and the training infrastructure is production-ready; more compute = more steps = better model.
-->

---

## Thank You

**MedChain** — OpenEnv India 2026, Finals

For full details:

- HuggingFace Space / model card: [submission link]
- GitHub: [repo link]
- WandB training run: https://api.wandb.ai/links/nikm5502-nikhil-mahajna/5pri4ooa

*Built with OpenEnv · HuggingFace TRL (GRPO) · Qwen3.5-2B*

<!--
Voice note: End cleanly. The environment is production-ready, the training pipeline works end-to-end on consumer hardware, and the reward decomposition gives a clear signal for every capability gap. Happy to go deeper on any of the five failure modes, the reward formula, or the Qwen3.5 architecture quirks.
-->
