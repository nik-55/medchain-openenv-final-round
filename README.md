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
| **Environment (HF Space)** | [nik-55/medchain-openenv-final-round](https://huggingface.co/spaces/nik-55/medchain-openenv-final-round) |
| **Training notebook** | [train_colab.py on GitHub](https://github.com/nik-55/medchain-openenv-final-round/blob/master/train_colab.py) |
| **Blog post** | [blog.md on GitHub](https://github.com/nik-55/medchain-openenv-final-round/blob/master/blog.md) |
| **WandB training run** | [view metrics](https://api.wandb.ai/links/nikm5502-nikhil-mahajna/5pri4ooa) |

---

## What is MedChain?

MedChain is an enterprise simulation where an LLM agent acts as the **central pharmacy coordinator** for a three-ward hospital network. It has to balance critical medical supplies — blood products, IV antibiotics, surgical consumables — across ICU, ER, and General wards over 8 consecutive rounds (each round = 2 simulated days).

What makes it hard: **every information source disagrees with every other one**. The legacy ERP is stale by one round. The warehouse scanner is live but noisy. The supplier portal is async — quotes arrive next round. Wards submit inflated requests (we call this "padding"), and the only way to know if a request is legitimate is to dig into census data, patient acuity scores, and historical consumption — then escalate contested cases to a binding clinical-review board.

No LLM-judged rewards. Every score is computed deterministically from simulation state.

---

## Why This Environment

**Theme #3.1 — Professional Tasks / World Modeling.** The theme asks for environments where models must do real, hard work against dynamic systems rather than exploiting shortcuts. MedChain puts the agent inside a live enterprise workflow: five siloed data sources with different reliability guarantees, a purchasing pipeline with approval gates, and a governance process for challenging ward requests. Querying a single system and acting on its output is the shortcut — the reward formula actively penalises it.

**Multi-actor dynamics.** The three wards are persistent scripted agents with their own incentive structures. The General ward over-requests most rounds. The ER defends aggressive requests hard when a real mass-casualty surge is incoming — which sometimes makes a 3× blood-product request legitimate. The ICU almost never inflates. The agent cannot treat these actors as a static environment; it has to model their behaviour, issue challenges, and read strategic redaction as a signal.

**Long-horizon consequences.** A decision made in round 3 directly shapes round 4 difficulty. Correctly flagging a padding attempt lowers that ward's hoarding pressure next round. Filing a rationale that cites disclosed evidence raises ward trust and improves future cooperation. Over-allocating to a chronic padder in early rounds compounds into worse budget headroom and higher hoarding pressure in later rounds. There is no way to recover from a locally-greedy policy — the episode has memory.

---

## Quick Start

```bash
# Run the environment server
docker build -t medchain-env:finals -f Dockerfile .
docker run -p 8000:8000 medchain-env:finals

# Or with UV (local dev)
uv run server --host 0.0.0.0 --port 8000

# Evaluate any OpenAI-compatible model against the environment
export HF_TOKEN=hf_...
SEEDS=0,1,2 DIFFICULTIES=light,medium,heavy python run_llm_eval.py
```

**Live environment**: [huggingface.co/spaces/nik-55/medchain-openenv-final-round](https://huggingface.co/spaces/nik-55/medchain-openenv-final-round)

**Training notebook**: download [`train_colab.py`](https://github.com/nik-55/medchain-openenv-final-round/blob/master/train_colab.py) and open in Google Colab (File → Upload). Needs a T4 GPU runtime. Uses `# %%` cell markers.

**WandB training run**: [view live metrics →](https://api.wandb.ai/links/nikm5502-nikhil-mahajna/5pri4ooa)

---

## The Environment

### Three Ward Actors

Wards are scripted by default (deterministic, no API key needed). Each has a personality — a probability of inflating requests, how aggressively they defend padding when challenged, and a fluctuating reputation score that carries across rounds.

| Ward | Priority | Padding behaviour | Backs down when challenged |
|---|---|---|---|
| ICU | 1.0 | Almost never pads | ~95% |
| ER | 0.7 | Pads moderately; defends real surges | ~55% |
| General | 0.3 | Chronically over-orders | ~90% |

Set `WARD_ACTOR_MODE=llm` to activate LLM-backed ward dialogue for richer storytelling.

### Five Siloed Enterprise Systems

This is the core of the World Modeling challenge. Each system returns different information with different reliability guarantees — exactly like real enterprise ops.

| System | Tools | Quirk |
|---|---|---|
| ERP Oracle | `erp_oracle_get_inventory`, `erp_oracle_get_pipeline` | **Stale by 1 round** — authoritative but lagged |
| WMS | `wms_scan_inventory` | **Live but ±5% noise** per lot |
| Supplier Portal | `supplier_portal_request_quote`, `supplier_portal_get_quote` | **Async** — quote resolves next round |
| Finance SAP | `finance_sap_get_budget`, `finance_sap_request_approval` | **Approval gate** for POs > $10k |
| Ward Messaging | `messaging_send_to_ward` | Outbound comms; ward replies in-character |

A strong agent reconciles stale ERP against noisy WMS, plans ahead for async quotes, and files coherent justifications for large POs before the finance gate rejects them.

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
- 21 MCP tools across 5 enterprise silos
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

- `enable_thinking=False` still emits an empty `<think>\n\n</think>` stub before the response. We initially stripped this as malformed output before realising it's by design.
- `pad_token_id` (248044) and `eos_token_id` (248046) are different. TRL convention is `pad_token_id=eos_token_id` in `generate()` — following this blindly breaks batched generation.
- The 18 Conv1d layers inside GatedDeltaNet blocks are **not quantized** by bitsandbytes (they stay in compute dtype). Our initial VRAM estimates were off because we counted them as quantized.
- `target_modules="all-linear"` in LoRA correctly covers 187 Linear layers and skips Conv1d — but this took a `check_qwen.py` run to verify.

We wrote a 72-check sanity script (`train/check_qwen.py`) that you can run CPU-only to verify all of this before spending GPU time.

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

---

## File Structure

```
openenv-hack/
├── train_colab.py            ← MAIN training notebook (run in Colab/Kaggle)
├── run_llm_eval.py           ← benchmark any OpenAI-compatible model
├── train.py                  ← shared training helpers (SYSTEM_PROMPT, parse_tool_calls, etc.)
├── client.py                 ← OpenEnv client (MedchainEnv)
├── models.py                 ← Pydantic Action/Observation/State models
├── config.py                 ← shared hyperparameters (GPU target, batch size, LoRA)
├── openenv.yaml              ← OpenEnv manifest
├── Dockerfile                ← server container
│
├── server/
│   ├── simulation.py         ← core simulation engine, all 21 tool methods
│   ├── grader.py             ← 10-component deterministic reward formula
│   ├── tasks.py              ← task configs, ward personas, events
│   ├── ward_actor.py         ← scripted/LLM ward actors
│   ├── clinical_arbiter.py   ← binding-verdict clinical review board
│   ├── medchain_env_environment.py  ← OpenEnv MCP wrapper
│   ├── prompts.py            ← inference system prompt + full tool schemas
│   └── llm_client.py         ← shared OpenAI-compatible client
│
├── train/
│   ├── train_hf_jobs.py      ← HF Jobs cloud training (A100, UV PEP 723)
│   ├── train_hf_jobs_offline.py  ← offline variant (pre-collected rollouts)
│   ├── collect_rollouts.py   ← CPU-only rollout collection for offline training
│   ├── evaluate.py           ← held-out eval (seeds 50–59, 90 episodes)
│   ├── run_trained_model.py  ← run a checkpoint via the OpenEnv server
│   └── check_qwen.py         ← 72-check Qwen3.5 sanity script (CPU-safe)
│
└── media/
    ├── training_rewards_steptime.png
    └── training_loss_lr_gradnorm.png
```

---

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `WARD_ACTOR_MODE` | `scripted` | Set `llm` to activate LLM ward dialogue |
| `ARBITER_MODE` | `scripted` | Set `llm` to activate LLM clinical arbiter |
| `HF_TOKEN` / `API_KEY` | — | Required for any LLM mode or inference eval |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible endpoint |
| `MODEL_NAME` | `openai/gpt-oss-20b:groq` | Model for `run_llm_eval.py` |
| `WARD_MODEL_NAME` | inherits `MODEL_NAME` | Ward actor model in LLM mode |

## Determinism

Two episodes with the same `seed`, `WARD_ACTOR_MODE=scripted`, `ARBITER_MODE=scripted` produce **byte-identical reward decompositions**. This is what makes GRPO training well-defined: the reward function is a pure function of `(seed, actions)`.

---

## Resources

- **Environment (HF Space)**: [huggingface.co/spaces/nik-55/medchain-openenv-final-round](https://huggingface.co/spaces/nik-55/medchain-openenv-final-round)
- **Blog post**: [blog.md on GitHub](https://github.com/nik-55/medchain-openenv-final-round/blob/master/blog.md)
- **Training notebook**: [train_colab.py on GitHub](https://github.com/nik-55/medchain-openenv-final-round/blob/master/train_colab.py)
- **WandB training run**: [https://api.wandb.ai/links/nikm5502-nikhil-mahajna/5pri4ooa](https://api.wandb.ai/links/nikm5502-nikhil-mahajna/5pri4ooa)
- **OpenEnv framework**: [github.com/openenv/openenv](https://github.com/openenv/openenv)
