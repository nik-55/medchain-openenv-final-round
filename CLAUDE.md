# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

Hackathon submission for the **OpenEnv India 2026 Hackathon — Finals round**.
Theme addressed: **#3.1 Professional Tasks (World Modeling — Enterprise)**.
Also touches on multi-actor dynamics (scripted ward actors + binding clinical arbiter).

Environment: **MedChain** — central supply-chain coordinator for a three-ward
hospital network, navigating five siloed enterprise systems (legacy ERP, live WMS,
supplier portal, finance SAP, ward messaging), running an evidence-based audit loop
on potentially-padded ward requests, and shepherding large purchase orders through
finance approvals. Disputed requests escalate to a binding clinical-review arbiter.

**Domain framing**: this is a *governance* simulation (audit + review), not a
marketplace. Wards do not negotiate; central pharmacy runs evidence-based reviews
and escalates to a clinical-review committee — that's what real hospital ops looks like.

**RLVR contract**: every reward signal is computed deterministically from `SimState`.
No LLM-judged rewards. Ward actors and the arbiter have optional LLM modes for
storytelling, but ground-truth (`true_need`, `padded_flag`, evidence facts) is
always sim-owned.

## Development Environment

**Local machine: CPU only** (`cuda=False`). Do not run GPU-dependent commands
locally. If a task requires GPU execution (training, 4-bit quant checks, eval),
ask the user to run the command — they have access to a **T4 GPU** via Google
Colab or Kaggle.

Standard pattern when GPU is needed:
> "Please run this on Colab/Kaggle: `python train_colab.py`"

## Dependencies

All training dependencies are in **`requirements-train.txt`** (repo root of `openenv-hack/`).
```bash
pip install -r requirements-train.txt
```

Key version constraints:
- `transformers>=5.2.0` — **minimum required for Qwen3.5 (`model_type=qwen3_5`) support**.
  Versions below 5.2.0 (all 4.x releases) raise `ValueError: does not recognize this architecture`.
- `trl>=0.29.0` — required for `rollout_func` kwarg in `GRPOTrainer`.

---

## Directory Layout

```
openenv-hack/
├── train_colab.py            ← MAIN training notebook (Colab/Kaggle, # %% cells)
├── run_llm_eval.py           ← benchmark any OpenAI-compatible model (no GPU needed)
├── train.py                  ← shared training helpers: SYSTEM_PROMPT, parse_tool_calls,
│                                build_dataset, TOOL_SCHEMAS — imported by all train scripts
├── client.py                 ← OpenEnv client (MedchainEnv — async + sync)
├── models.py                 ← Pydantic Action / Observation / State models
├── config.py                 ← shared hyperparameters (GPU target, batch size, LoRA)
├── openenv.yaml              ← OpenEnv manifest
├── Dockerfile                ← server container (root-level, used by HF Spaces)
├── requirements-train.txt    ← training dependencies
├── blog.md                   ← HF blog post for submission
├── README.md                 ← submission README with training results + links
│
├── server/
│   ├── simulation.py         ← core sim engine; all 21 tool methods live here
│   ├── grader.py             ← 10-component deterministic reward formula
│   ├── tasks.py              ← TaskConfig, WardConfig, ward personas, events
│   ├── ward_actor.py         ← scripted/LLM ward actors (scripted by default)
│   ├── clinical_arbiter.py   ← binding-verdict clinical review board
│   ├── medchain_env_environment.py  ← OpenEnv MCP wrapper (21 FastMCP tools)
│   ├── prompts.py            ← INFERENCE_SYSTEM_PROMPT + INFERENCE_TOOL_SCHEMAS
│   ├── llm_client.py         ← shared OpenAI-compatible client
│   ├── erp_formatter.py      ← ERP Oracle response formatting helpers
│   ├── app.py                ← FastAPI app entry point
│   └── requirements.txt      ← server-only runtime deps
│
├── train/
│   ├── train_hf_jobs.py      ← HF Jobs cloud training (A100, UV PEP 723 self-contained)
│   ├── train_hf_jobs_offline.py  ← offline variant (pre-collected rollouts, no live sim)
│   ├── collect_rollouts.py   ← CPU-only rubber-stamp rollout collection for offline training
│   ├── evaluate.py           ← held-out eval: seeds 50-59, 90 episodes, greedy decoding
│   ├── run_trained_model.py  ← run a checkpoint through the OpenEnv WebSocket server
│   ├── check_qwen.py         ← 72-check Qwen3.5 sanity script (CPU-safe; --4bit needs GPU)
│   └── training_and_eval_strategy.md  ← design notes on training strategy
│
├── traces/
│   ├── golden_log.txt        ← full episode trace from a good run (reference)
│   └── gd_2.txt              ← another episode trace
│
├── artifacts/
│   └── rollouts.jsonl        ← pre-collected rollout dataset (for offline training)
│
└── media/
    ├── training_rewards_steptime.png  ← step_time, reward std, reward mean plots
    └── training_loss_lr_gradnorm.png  ← loss, learning_rate, grad_norm plots
```

---

## Key Commands

```bash
# Build + run the environment server
docker build -t medchain-env:finals -f Dockerfile .
docker run -p 8000:8000 medchain-env:finals

# Run server with UV (local dev, no Docker)
uv run server --host 0.0.0.0 --port 8000
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000

# Benchmark any OpenAI-compatible model (no GPU, no server needed)
HF_TOKEN=... SEEDS=0,1,2 DIFFICULTIES=light,medium python run_llm_eval.py

# Sanity-check Qwen 3.5 (CPU-safe, no GPU needed)
python train/check_qwen.py --model-id Qwen/Qwen3.5-2B
python train/check_qwen.py --skip-generation   # config/parse checks only

# Full quant + LoRA check (needs GPU — ask user to run on Colab/Kaggle)
python train/check_qwen.py --model-id Qwen/Qwen3.5-2B --4bit

# Run a trained checkpoint through the OpenEnv server (needs GPU + running server)
SEED=0 DIFFICULTY=medium python train/run_trained_model.py
SEED=42 DIFFICULTY=heavy CHECKPOINT=checkpoints/checkpoint-200 python train/run_trained_model.py

# Eval a checkpoint (needs GPU)
python train/evaluate.py --base
python train/evaluate.py --checkpoint checkpoints/checkpoint-100

# Cloud training via HF Jobs (A100)
# Submit train/train_hf_jobs.py via the HF Jobs API — see its header for details

# Collect rubber-stamp rollouts for offline training (CPU-only)
python train/collect_rollouts.py --seeds 0-49 --difficulties light medium heavy
```

---

## Architecture

### Request flow (server-side)

```
FastAPI (server/app.py)
  └─ MedchainEnvironment (server/medchain_env_environment.py)
       ├─ MedchainSimulation (server/simulation.py)
       │    ├─ FEFO inventory, lot expiry, supplier lead times
       │    ├─ ER surge / MCI / recall / coldchain events
       │    ├─ Finance approval queue, supplier quote queue
       │    ├─ Evidence log, escalation log, rationale log
       │    └─ Ward reputation + hoarding pressure dynamics
       ├─ WardActor (server/ward_actor.py)
       │    └─ Scripted by default; LLM dialogue opt-in via WARD_ACTOR_MODE=llm
       ├─ Clinical Arbiter (server/clinical_arbiter.py)
       │    └─ Scripted by default; LLM verdict opt-in via ARBITER_MODE=llm
       └─ _MedchainMCPDelegate (MCPEnvironment) — FastMCP dispatch (21 tools)
```

Reward survives WebSocket serialization because `MedchainEnvironment` (outer class)
computes the final float and packs it into `MedchainToolObservation.reward`.

### Package layout

`pyproject.toml` maps:
- `medchain_env` → repo root (`client.py`, `models.py`)
- `medchain_env.server` → `server/` subdirectory

### Episode shape

- **Single task**: `multi_actor_coordination`, parameterised by `seed` and
  `difficulty ∈ {light, medium, heavy}`.
- **Round-based**: 8 rounds per episode, each 2 simulated days.
- **Wards**: `ward_icu`, `ward_er`, `ward_general`. Scripted by default.
  Each has priority, pad_prob, pad_range, tracked SKUs, justification bank,
  plus a `WardActorConfig` (persona, hoarding pressure).
- **ER surge**: hidden `er_surge_state` drawn once per round — spikes to
  2.5–3.0× during MCI events; otherwise mostly 1.0 with occasional 2.0.
- **Pre-seeded history**: 8 synthetic prior rounds per ward, deterministic
  from seed; visible via `query_ward_history`.
- **Reputation drift**: each round, ward reputations decay toward 0.5.
  Confirmed padding via escalation lowers it; evidence-grounded rationales
  raise it (and lower next-round hoarding pressure).

### Tools (21 MCP)

**Coordination** — `get_round_briefing`, `view_requests`, `read_inbox`,
`submit_allocation_plan(plan_json, rationale_json?)`, `advance_round`

**Investigation** — `query_ward_history`, `query_supplier`, `query_erp` *(legacy alias)*

**Enterprise systems** (5 silos) —
`erp_oracle_get_inventory` (stale by 1 round),
`erp_oracle_get_pipeline`,
`wms_scan_inventory` (live, ±5% noise),
`supplier_portal_request_quote` (async),
`supplier_portal_get_quote`,
`finance_sap_get_budget`,
`finance_sap_request_approval` (gate for POs > $10k),
`messaging_send_to_ward`

**Audit & governance** —
`request_evidence(ward_id, sku, evidence_type)`,
`escalate_to_clinical_review(ward_id, sku, concern)`

**Procurement** — `submit_po`, `file_justification`, `quarantine_lot`

### Reward formula

```
0.25 × network_service_level
+ 0.18 × critical_service_level     # ICU+ER blood products
+ 0.18 × allocation_accuracy        # per-ward-round surplus+stockout
+ 0.12 × event_response             # MCI / supplier / recall / coldchain
+ 0.07 × budget_efficiency
+ 0.04 × waste_control
+ 0.05 × audit_score                # mean(evidence_use_rate, escalation_acc)
+ 0.05 × approval_workflow_score    # finance approvals resolved cleanly
+ 0.03 × tool_discovery_score       # used N/5 enterprise systems
+ 0.03 × briefing_efficiency        # one briefing per round
- justification_penalty (cap 0.15)
```

Per-step shaping (resets each round): read_inbox +0.01, view_requests +0.02,
valid submit_allocation_plan +0.03, valid submit_po +0.02,
recall-window quarantine +0.03, get_round_briefing +0.02 (once),
request_evidence +0.01 (+0.02 if redacted),
escalate_to_clinical_review +0.05 / -0.03,
finance_sap_request_approval +0.03 / -0.02,
first-use of any enterprise system +0.005.

### Audit loop scoring quirks

- `audit_score` averages only sub-signals that were **exercised**. An agent that
  never escalates gets 0 there — the cost lands on `alloc_acc` instead.
- `evidence_use_rate` requires the agent to **cite** evidence in `rationale_json`.
  Calling `request_evidence` then writing a generic rationale = 0 use rate.
- `escalation_acc`: -0.5 per frivolous escalation, +1 per correct. 50/50 agent → 0.25.

---

## Training Notes

### `train.py` — shared module

`train.py` exports `SYSTEM_PROMPT`, `TOOL_SCHEMAS`, `parse_tool_calls`, `build_dataset`,
plus constants `MAX_TURNS`, `MAX_ROUND_TURNS`. **All training scripts import from here.**
Do not rename or move it — it must stay at the repo root to be importable by Colab
scripts that `sys.path.insert(0, WORK_DIR)`.

The `SYSTEM_PROMPT` and `TOOL_SCHEMAS` in `train.py` reflect the legacy 10-tool surface.
For training against the full Tier-1 surface (audit loop, enterprise systems, briefing):
1. Swap `SYSTEM_PROMPT` for `INFERENCE_SYSTEM_PROMPT` from `server/prompts.py`
2. Swap `TOOL_SCHEMAS` for `INFERENCE_TOOL_SCHEMAS`
3. Increase `MAX_TURNS` — the audit loop adds 2-3 calls per round

### Model: Qwen3.5-2B / 4B

- **Hybrid architecture**: 18 `GatedDeltaNet` (SSM-style) + 6 standard attention layers.
  18 `Conv1d` layers inside GatedDeltaNet blocks — **NOT quantized by bitsandbytes**
  (stays in compute dtype, expected). VRAM estimates must account for this.
- `enable_thinking=False` still emits an empty `<think>\n\n</think>` stub — by design.
- `pad_token_id` (248044) ≠ `eos_token_id` (248046); TRL convention uses
  `pad_token_id=tok.eos_token_id` in `generate()` — intentional.
- LoRA `target_modules="all-linear"` covers 187 Linear layers, skips Conv1d — correct.

### Rollout design (`medchain_rollout`)

- Called with B prompts; expands to B×G episodes (32 on T4 with B=8, G=4).
- Context **reset after every `advance_round`** — `[system, round_brief]` only.
  This is critical: without it, 8-round episodes accumulate 3000–8000 token sequences
  which makes each `model.generate()` call take 2–4 minutes on T4.
- `skip_special_tokens=True` when decoding into `ep["messages"]` — strips `<|im_end|>`,
  `<think>` before storing. `<tool_call>` tags are preserved (they are NOT special tokens).
- Logprobs: `output_scores=True` → `log_softmax` + `gather`.

### Key constants (config.py)
| Constant | A10G | T4 (Colab) | Notes |
|---|---|---|---|
| `B` | 8 | 4–8 | Seeds per step |
| `G` | 4 | 4 | GRPO group size |
| `MAX_NEW_TOKENS` | 256 | 256 | Tool calls <100 tokens |
| `MAX_TURNS` | 60 | 60 | Total loop guard across 8 rounds |
| `MAX_ROUND_TURNS` | 12 | 12 | Force advance_round if model loops |

### LoRA config
```python
LoraConfig(r=16, lora_alpha=16, target_modules="all-linear",
           lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
```
Do NOT call `get_peft_model()` manually — TRL applies it via `peft_config`.

---

## LLM Eval (`run_llm_eval.py`)

OpenAI-compatible client driving multi-episode runs against the simulation directly
(no HTTP server needed). Use this to benchmark frontier models before training.

Key env vars: `HF_TOKEN`/`API_KEY`, `API_BASE_URL`, `MODEL_NAME`, `SEEDS`, `DIFFICULTIES`.

Stdout: `[START]`, `[STEP]`, `[END]`, `[BREAKDOWN]`, `[SUMMARY]` lines — grep-friendly.

---

## Eval (`train/evaluate.py`)

- Held-out seeds 50–59 (never seen during training).
- 3 rollouts × 10 seeds × 3 difficulties = **90 episodes**, all batched.
- Greedy decoding (`do_sample=False`) — no `output_scores` needed.
- Imports `SYSTEM_PROMPT`, `parse_tool_calls`, `MAX_TURNS` from `train.py`.
- sys.path patched at top to find parent modules.

---

## Sanity-check (`train/check_qwen.py`)

Runs **72 checks** across 10 sections. All CPU-safe except `--4bit`.

| Section | What it checks |
|---|---|
| 1 | Model & tokenizer load, pad/eos tokens |
| 2 | Chat template, `enable_thinking=False`, empty `<think>` stub |
| 3 | Generation + logprob shape/sign alignment |
| 4 | `parse_tool_calls` — simple & hard cases + model generation |
| 5 | Context reset: round-reset produces shorter sequence |
| 6 | Batch generation with left-padding |
| 7 | BitsAndBytesConfig / LoraConfig / GRPOConfig fields |
| 8 | `seed=N;diff=X` embed/parse regex |
| 9 | Architecture inspection (Linear count, Conv1d count, vocab alignment) |
| 10 | 4-bit quant: Linear4bit present, no NaN logits, LoRA gradient flow (**GPU only**) |

---

## Determinism contract

| Source | Default | Opt-in |
|---|---|---|
| Ward actor dialogue | scripted | `WARD_ACTOR_MODE=llm` |
| Clinical arbiter verdict | scripted | `ARBITER_MODE=llm` |
| WMS noise | seeded by `(round_idx, lot_id)` | always deterministic |
| Synthetic ward history | seeded by `(seed, ward_id)` | always deterministic |

Two episodes with identical `seed`, `WARD_ACTOR_MODE=scripted`, `ARBITER_MODE=scripted`
produce **byte-identical** reward decompositions. This is what makes GRPO well-defined.
