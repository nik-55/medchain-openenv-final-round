---
title: Medchain Env Environment Server
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - world-modeling
  - enterprise
  - rl-training
---

# MedChain — Hospital Supply-Chain Coordinator

**Hackathon**: OpenEnv India 2026 — Finals Round  
**Theme**: #3.1 Professional Tasks (World Modeling — Enterprise)  
**Team**: Solo

| Resource | Link |
|---|---|
| **GitHub repository** | [nik-55/medchain-openenv-final-round](https://github.com/nik-55/medchain-openenv-final-round) |
| **Environment (HF Space)** | [nik-55/medchain-openenv-final-round](https://huggingface.co/spaces/nik-55/medchain-openenv-final-round) |
| **Demo video** | [youtu.be/L47ZVn1syAM](https://youtu.be/L47ZVn1syAM) |
| **Blog post** | [blog.md on GitHub](https://github.com/nik-55/medchain-openenv-final-round/blob/master/blog.md) |
| **Training notebook** | [train_colab.py on GitHub](https://github.com/nik-55/medchain-openenv-final-round/blob/master/train_colab.py) |
| **WandB training run** | [view metrics](https://api.wandb.ai/links/nikm5502-nikhil-mahajna/5pri4ooa) |
| **Training logs (HF Jobs stdout)** | [hf_jobs_grpo_training_stdout.txt](https://github.com/nik-55/medchain-openenv-final-round/blob/master/traces/hf_jobs_grpo_training_stdout.txt) |
| Why the trained model falls short of its potential | [Why Training Is Unstable / Partially Done](#why-training-is-unstable--partially-done) |

[![Watch the video](https://img.youtube.com/vi/L47ZVn1syAM/maxresdefault.jpg)](https://www.youtube.com/watch?v=L47ZVn1syAM)

---

## What is MedChain?

MedChain is an enterprise simulation where an LLM agent acts as the **central pharmacy coordinator** for a three-ward hospital network. It has to balance critical medical supplies — blood products, IV antibiotics, surgical consumables — across ICU, ER, and General wards over 8 consecutive rounds (each round = 2 simulated days).

What makes it hard: the simulation itself runs as a classical, deterministic state machine — lot expiry, ward consumption rates, supplier lead times, finance approval queues, and ward reputation dynamics all evolve by fixed rules. The agent's role is to reason over this system with incomplete, noisy, and lagged visibility, across a horizon where decisions made now compound into outcomes several rounds later.

A few concrete examples of this temporal dependency: skip the audit loop in round 2 and the General ward's hoarding pressure stays elevated for the rest of the episode, inflating every subsequent request. Submit a large purchase order in round 4 without pre-filing a finance justification and it sits behind an approval gate, arriving a full round late while the ICU is already running short. Miss an MCI surge flagged in the inbox and cut the ER's blood-product request as padding — what looked suspicious was legitimate, and critical-service-level takes a permanent hit that no round-7 recovery can undo. The LLM has to model not just the current state it can observe, but the causal chain connecting today's tool calls to next round's constraint landscape.

No LLM-judged rewards. Every score is computed deterministically from simulation state.

---

## Why This Environment

**Theme #3.1 — Professional Tasks / World Modeling.** The theme asks for environments where models must do real, hard work against dynamic systems rather than exploiting shortcuts. MedChain puts the agent inside a live enterprise workflow: five systems with different reliability and latency characteristics, a purchasing pipeline with finance approval gates, and an audit loop — `request_evidence`, cross-referencing census data and historical consumption, escalating contested cases to a clinical-review board — for requests that don't add up. Querying a single system and acting on its output is the shortcut; the reward formula actively penalises it.

**Multi-actor dynamics.** The three wards are persistent scripted agents with their own incentive structures. The General ward over-requests most rounds. The ER defends aggressive requests hard when a real mass-casualty surge is incoming — which sometimes makes a 3× blood-product request legitimate. The ICU almost never inflates. The agent cannot treat these actors as a static environment; it has to model their behaviour, call `request_evidence` on suspicious requests, and read strategic redaction as a signal.

**Long-horizon consequences.** A decision made in round 3 directly shapes round 4 difficulty. Correctly flagging a padding attempt lowers that ward's hoarding pressure next round. Filing a rationale that cites disclosed evidence raises ward trust and improves future cooperation. Over-allocating to a chronic padder in early rounds compounds into worse budget headroom and higher hoarding pressure in later rounds. There is no way to recover from a locally-greedy policy — the episode has memory.

---

## Setup & Running

### Prerequisites

- Python 3.10+ with `uv` — [install uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker (optional, for a fully isolated server container)

### Option A — Docker (recommended for evaluation)

```bash
# Build the image
docker build -t medchain-env:finals -f Dockerfile .

# Start the server on port 8000
docker run -p 8000:8000 medchain-env:finals
```

The server is ready when you see `Application startup complete` in the logs.

### Option B — UV (local dev, no Docker)

```bash
# Install runtime dependencies
uv sync

# Start the server
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Option C — Live HF Space (no setup)

The environment is already running at:

```
https://nik-55-medchain-openenv-final-round.hf.space
```

Use this URL as `BASE_URL` in any script below — no Docker or `uv` required.

---

## Using the Environment

### Python client (quick smoke test)

Install the client package (from the repo root):

```bash
pip install -e .
```

Then run the included smoke test against the live HF Space:

```bash
python tests/test_openenv_hf_space.py
```

This connects, resets, calls `list_tools`, and steps through a minimal round
(`read_inbox → view_requests → get_round_briefing → query_ward_history →
submit_allocation_plan → advance_round`), printing tool output and rewards at each step.

### Python client (custom agent loop)

```python
import asyncio
from medchain_env import CallToolAction, MedchainEnv

BASE_URL = "https://nik-55-medchain-openenv-final-round.hf.space"
# Or point to your local server: BASE_URL = "http://localhost:8000"

async def main():
    env = MedchainEnv(base_url=BASE_URL)
    await env.connect()
    try:
        # ── Reset ──────────────────────────────────────────────────────
        result = await env.reset(task="multi_actor_coordination")
        print(result.observation.metadata["dashboard"])

        # ── Discover tools ─────────────────────────────────────────────
        tools = await env.list_tools()
        print([t.name for t in tools])

        # ── Step through one round ─────────────────────────────────────
        for tool_name, args in [
            ("read_inbox",            {}),
            ("view_requests",         {}),
            ("get_round_briefing",    {}),
            ("submit_allocation_plan",
             {"plan_json": '{"ward_icu":{"BLOOD-RBC":10},"ward_er":{"BLOOD-RBC":5}}'}),
            ("advance_round",         {}),
        ]:
            r = await env.step(CallToolAction(tool_name=tool_name, arguments=args))
            print(r.observation.metadata.get("tool_result", ""))
            if r.observation.done:
                break
    finally:
        await env.close()

asyncio.run(main())
```

### Episode structure

Each episode is **8 rounds** (each round = 2 simulated days). The canonical loop per round:

```
read_inbox()                       ← check for alerts (MCI, recalls, disruptions)
view_requests()                    ← see what each ward is requesting this round
get_round_briefing()               ← consolidated dashboard (budget, events, requests)
[optional investigation tools]     ← query_ward_history, request_evidence, wms_scan_inventory, …
submit_allocation_plan(plan_json)  ← commit allocations for this round (once per round)
[optional procurement tools]       ← submit_po, finance_sap_request_approval, …
advance_round()                    ← resolve consumption, receive deliveries, start next round
```

Call `submit_allocation_plan` **exactly once** per round, then `advance_round`. Any
tool call after `advance_round` lands in the next round's budget.

`plan_json` must be a JSON object keyed by ward ID:

```json
{
  "ward_icu":     {"BLOOD-RBC": 13, "BLOOD-PLT": 6, "BLOOD-FFP": 8},
  "ward_er":      {"BLOOD-RBC": 6,  "BLOOD-PLT": 3},
  "ward_general": {"GLOVE-001": 20, "IV-SAL-500": 15}
}
```

### The 21 tools

| Category | Tool | Required params |
|---|---|---|
| **Coordination** | `get_round_briefing` | — |
| | `view_requests` | — |
| | `read_inbox` | — |
| | `submit_allocation_plan` | `plan_json` |
| | `advance_round` | — |
| **Investigation** | `query_ward_history` | `ward_id` |
| | `query_erp` | `table` |
| | `query_supplier` | `supplier_id` |
| **Enterprise systems** | `erp_oracle_get_inventory` | — |
| | `erp_oracle_get_pipeline` | — |
| | `wms_scan_inventory` | — |
| | `supplier_portal_request_quote` | `supplier_id`, `product_id`, `quantity` |
| | `supplier_portal_get_quote` | `quote_id` |
| | `finance_sap_get_budget` | — |
| | `finance_sap_request_approval` | `approval_id`, `justification` |
| | `messaging_send_to_ward` | `ward_id`, `body` |
| **Audit & governance** | `request_evidence` | `ward_id`, `sku` |
| | `escalate_to_clinical_review` | `ward_id`, `sku`, `concern` |
| **Procurement** | `submit_po` | `supplier_id`, `product_id`, `destination_id`, `quantity` |
| | `file_justification` | `ticket_id`, `reason` |
| | `quarantine_lot` | `location_id`, `sku`, `lot_id` |

### Evaluating a model

`run_llm_eval.py` drives any OpenAI-compatible model through full episodes:

```bash
export HF_TOKEN=hf_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=openai/gpt-oss-120b:groq

# Optional: point at your local server instead of the default HF Space
export BASE_URL=http://localhost:8000

SEEDS=0,1,2 DIFFICULTIES=light,medium,heavy python run_llm_eval.py
```

Stdout emits structured lines you can grep:

```
[START] task=multi_actor_coordination env=medchain model=openai/gpt-oss-120b:groq
[STEP]  step=1 action=read_inbox({}) reward=0.01 done=false error=null
[STEP]  step=2 action=view_requests({}) reward=0.02 done=false error=null
...
[END]   success=true steps=47 score=0.681 rewards=0.01,0.02,...
[BREAKDOWN] network_service_level=0.82 critical_service_level=0.91 ...
```

---

## The Environment

### Three Ward Actors

Wards are scripted by default (deterministic, no API key needed). Each has a personality — a probability of inflating requests, how aggressively they defend padding when challenged, and a fluctuating reputation score that carries across rounds.

| Ward | Priority | Padding behaviour | Verdict accepted on escalation |
|---|---|---|---|
| ICU | 1.0 | Almost never pads | Almost always honest — escalation rarely needed |
| ER | 0.7 | Pads moderately; defends real surges | Verdict binding regardless; real surges get APPROVE |
| General | 0.3 | Chronically over-orders | REDUCE/DENY verdicts common and correct |

Set `WARD_ACTOR_MODE=llm` to activate LLM-backed ward dialogue for richer storytelling.

### Five Enterprise Systems

Each of the five systems the agent interacts with returns different information with different reliability guarantees — and a strong agent knows which to trust for what.

| System | Tools | Characteristic |
|---|---|---|
| ERP Oracle | `erp_oracle_get_inventory`, `erp_oracle_get_pipeline` | **Authoritative but one round lagged** — reflects last round's state |
| WMS | `wms_scan_inventory` | **Live but ±5% noise** per lot — cross-reference with ERP for reconciliation |
| Supplier Portal | `supplier_portal_request_quote`, `supplier_portal_get_quote` | **Async** — request this round, quote arrives next round |
| Finance SAP | `finance_sap_get_budget`, `finance_sap_request_approval` | **Approval gate** for POs > $10k — file justification before submitting |
| Ward Messaging | `messaging_send_to_ward` | Outbound comms; wards respond in-character based on their persona |

A strong agent reconciles the ERP view against the live WMS scan, plans ahead for async supplier quotes, and files coherent justifications before the finance gate rejects a large PO.

### The Audit-and-Review Loop

This is the multi-actor surface. When a ward request looks suspicious:

```
Suspicious request?
      ↓
  request_evidence(ward, sku, "all")
      ↓
  Ward returns structured data (census, acuity, recent_actuals, events).
  High-pressure wards may REDACT one field — redaction itself is signal.
      ↓
Still doesn't add up?
      ↓
  escalate_to_clinical_review(ward, sku, concern)
      ↓
  Hospital Supply Committee (scripted or LLM arbiter)
  returns BINDING verdict: APPROVE / REDUCE / DENY
      ↓
Allocate with cited rationale → ward trust grows next round.
```

### Reward Formula

All components are ∈ [0, 1], computed purely from `SimState`:

```
0.25 × network_service_level        (were actual consumption needs met?)
0.18 × critical_service_level       (ICU + ER blood products specifically)
0.18 × allocation_accuracy          (per-ward surplus + stockout penalty)
0.12 × event_response               (MCI, product recall, cold-chain breach)
0.07 × budget_efficiency
0.04 × waste_control
0.05 × audit_score                  (evidence cited in rationale + escalation accuracy)
0.05 × approval_workflow_score      (finance approvals resolved cleanly)
0.03 × tool_discovery_score         (used all 5 enterprise systems)
0.03 × briefing_efficiency          (one dashboard call per round, not four)
- justification_penalty (cap 0.15)
```

Per-step shaping rewards guide exploration during training (read_inbox +0.01, valid plan +0.03, correct escalation +0.05, etc.).

### Episode Shape

- 8 rounds × 2 simulated days each
- 21 MCP tools across 5 enterprise systems
- FEFO inventory, lot expiry, ER surge events, supplier disruptions, product recalls, cold-chain breaches
- Synthetic 8-round ward history (seeded, deterministic) for each episode
- Ward reputations and hoarding pressure evolve round-to-round

### Score Benchmarks

| Policy | Expected score |
|---|---|
| Rubber-stamp baseline (allocate exactly requested) | ~0.41 |
| Discount-General heuristic (cut General by 30%) | ~0.47 |
| Trained Qwen3.5-2B (GRPO, 500 steps) | ~0.52–0.55 |
| Sonnet 4.6 / GPT-4o (full audit + reconciliation) | ~0.65–0.75 |

---

## Training Results

We trained Qwen3.5-2B (4-bit NF4 + LoRA) with GRPO using a custom multi-turn rollout. The training connected directly to the simulation — no static dataset, live episodes every step.

### Training Metrics

**Reward (mean and std across the GRPO group)**

![Training reward metrics](media/training_rewards_steptime.png)

*Left: step time in seconds (120–240s — see challenges section). Centre: reward std across the GRPO group. Right: reward mean — starts ~0.13, climbs toward 0.28–0.35 with some variance.*

**Loss, learning rate, gradient norm**

![Training loss metrics](media/training_loss_lr_gradnorm.png)

*Loss hovers around 0 early (model mostly in the right direction), spikes at step ~22 during a large policy update (typical GRPO behaviour), then settles. Learning rate decays linearly from 1e-4. Gradient norm stays in 0.2–0.4 range.*

**Full WandB run**: [https://api.wandb.ai/links/nikm5502-nikhil-mahajna/5pri4ooa](https://api.wandb.ai/links/nikm5502-nikhil-mahajna/5pri4ooa)

*Note: the training pipeline works end-to-end but the compute budget for the finals round was limited. The full 500-step run requires more engineering around episode batching to keep step times below 60s.*

---

## Challenges We Hit

### Long context in episode trajectories

The biggest practical problem. An 8-round episode with 20 tool calls per round accumulates ~40 conversation turns before `advance_round` wipes the context. With left-padded batches of 32 episodes, this means `model.generate()` runs on sequences of 3000–8000 tokens, which explains the 120–240s per step times visible in the plots. We mitigated this by resetting the conversation to `[system, round_brief]` after each `advance_round`, bounding context growth per round.

### Qwen3.5-2B tool-calling surprises

Qwen3.5 is a hybrid model (18 GatedDeltaNet SSM layers + 6 standard attention layers). Several non-obvious behaviours tripped us up:

- **Two distinct tool-call formats, both legitimate.** When you pass tool schemas via `tools=` to `apply_chat_template`, the chat template injects a `<tools>` block and the model produces its **native XML format**: `<tool_call><function=NAME><parameter=KEY>value</parameter></function></tool_call>`. When no `tools=` is passed (our training setup), the model falls back to a JSON format: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`. Both are valid — the XML one is what the template was designed around; the JSON one is what the model learns from prompted examples. `check_qwen.py` tests parsers for both.
- `<tool_call>` is **not a special token** — `skip_special_tokens=True` strips `<|im_end|>` and `<think>` but preserves `<tool_call>`. This matters for the training decode path.
- `enable_thinking=False` still emits an empty `<think>\n\n</think>` stub before the response. We initially stripped this as malformed output before realising it's by design.
- `pad_token_id` (248044) and `eos_token_id` (248046) are different. TRL convention is `pad_token_id=eos_token_id` in `generate()` — following this blindly breaks batched generation.
- The 18 Conv1d layers inside GatedDeltaNet blocks are **not quantized** by bitsandbytes (they stay in compute dtype). Our initial VRAM estimates were off because we counted them as quantized.
- `target_modules="all-linear"` in LoRA correctly covers 187 Linear layers and skips Conv1d — but this took a `check_qwen.py` run to verify.

We wrote a 72-check sanity script (`train/check_qwen.py`) that you can run CPU-only to verify all of this before spending GPU time.

---

## Why Training Is Unstable / Partially Done

This section is a transparent account of what I ran into — not excuses, just the actual experience.

### The base model struggles with multi-step tool planning

When I tested Qwen3.5-4B (base, no fine-tuning) against the environment, it was not able to plan tool calls coherently. Not in a "wrong answer" way — more fundamentally, it did not reliably understand *when* to call a tool, *which* tool to call next in a multi-step sequence, or how to carry context from one tool result forward into the next decision. For a single-turn question-answering task this might not matter. For an 8-round episode with 21 tools and causal dependencies between rounds, it was a blocker.

The goal of training was to surface this planning ability as an **emergent property** through GRPO: expose the model to live episodes, let group variance signal which trajectories succeeded, and push the policy toward the good ones. That is the right approach — but it only works if the model can already generate enough *good* trajectories to learn from. With a 2B/4B base model that has weak tool-chaining out of the box, most rollouts in the early group are undifferentiated low scores, giving the gradient very little signal to work with.

### Trajectories are heavy and exploration is expensive

An 8-round episode with ~15 tool calls per round generates roughly 3,000–8,000 tokens of context per trajectory. With GRPO's group size G=4 and batch size B=8, each training step requires generating **32 full episodes in parallel**. On a single A10G with a 4-bit quantized 2B model this already pushes the VRAM ceiling. Moving to 4B, or relaxing the per-round context reset, pushes it over.

Good exploration requires the model to try diverse sequences — audit loops, evidence requests, escalation paths — and the reward signal only arrives at the *end* of the 8-round episode, not per-step. This means the model needs many more trajectories, across a wider seed distribution, to see enough variation in the return signal. That is a compute budget problem.

### GPU scheduling on HF Jobs did not cooperate

I had access to HF credits for an A100 run. I submitted the job via `train/train_hf_jobs.py`. The job sat in the scheduling queue for an extended period without starting — not a code failure, just queue wait time that consumed most of the available compute window. This is logged in the HF Jobs stdout in `traces/hf_jobs_grpo_training_stdout.txt`. I am not attributing fault here; this is just what happened.

### The training harness is solid — the bottleneck is scale

The training script (`train/train_hf_jobs.py`, `train_colab.py`) is robust. It batches across turns from multiple concurrent episodes — each trajectory can be at a different round turn during a single training step — which is the correct design for multi-turn GRPO. The rollout function connects to the live simulation per episode, computes token-level log-probabilities, and assembles the GRPO objective correctly. The 72-check `train/check_qwen.py` script validates the entire stack from model load through LoRA gradient flow.

The issue is not the harness design. The issue is that the exploration budget required to get the policy to *emit good planning trajectories* as an emergent phenomenon is larger than what a single A10G (or a single hackathon window) can deliver. The reward curve in the WandB run does climb — from ~0.13 to ~0.28–0.35 — which shows the signal is real. It just needs more steps, more VRAM, and ideally some curriculum work (start with light-difficulty episodes, gate heavy-difficulty in later) to make the exploration tractable.

I am planning to continue this work after the hackathon — proper curriculum scheduling, longer training runs, and more investigation into what it takes for the planning behaviour to emerge robustly.

---

## Architecture

```
FastAPI (server/app.py)
  └─ MedchainEnvironment (server/medchain_env_environment.py)
       ├─ MedchainSimulation (server/simulation.py)      ← core engine
       │    ├─ FEFO inventory, lot expiry, lead times
       │    ├─ ER surge / MCI / recall / cold-chain events
       │    ├─ Finance approval queue, supplier quote queue
       │    ├─ Evidence log, escalation log, rationale log
       │    └─ Ward reputation + hoarding pressure dynamics
       ├─ WardActor (server/ward_actor.py)               ← scripted; LLM opt-in
       ├─ Clinical Arbiter (server/clinical_arbiter.py)  ← scripted; LLM opt-in
       └─ Reward grader (server/grader.py)               ← 10-component formula
```

21 MCP tools registered via FastMCP. Dispatch is `getattr(sim, tool_name)(**args)` — any new tool added to `MedchainSimulation` is immediately callable without additional wiring.

## Determinism

Two episodes with the same `seed`, `WARD_ACTOR_MODE=scripted`, `ARBITER_MODE=scripted` produce **byte-identical reward decompositions**. This is what makes GRPO training well-defined: the reward function is a pure function of `(seed, actions)`.

---

## Hackathon Experience

This hackathon turned out to be one of the most meaningful learning experiences I've had. RL was always something I understood at an abstract level — going through this challenge made it concrete and real. Designing a reward function that can't be gamed, thinking carefully about what the agent actually observes vs. what the simulation knows, debugging why GRPO produces the training dynamics it does — all of that clicked in a way that reading papers alone never managed. I'm genuinely planning to keep building RL environments after the hackathon ends.

The organizers were responsive throughout and a pleasure to work with. The onsite event at Scaler School of Technology, Bangalore was a fantastic experience — handling that many participants operationally takes serious logistics and coordination. Maybe one day LLMs will manage event ops at this scale, but for now a huge thank you to the entire SST team for pulling it off.

A big round of thanks to **Meta PyTorch**, **Hugging Face**, and **Scaler School of Technology** for organizing this and giving us the opportunity to build something real. I hope PyTorch and HuggingFace run more hackathons like this — they're exactly the kind of event that turns reading-about-RL into actually-doing-RL.
