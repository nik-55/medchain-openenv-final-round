# MedChain Finals — Training & Evaluation Strategy

## Environment Summary

**Task**: Multi-actor hospital supply chain coordination (`multi_actor_coordination`).  
**Episode shape**: 8 rounds × 2 sim days = 16 simulated days.  
**Agent role**: Central supply coordinator — reads ward requests, queries inventory/history, submits POs, allocates stock across ICU / ER / General.  
**Core challenge**: Ward requests are padded (General: 65–85% of rounds, ER: 20%, ICU: ~10%). `true_need` is never shown. Agent must infer real demand from history and justifications.

**Reward** (terminal scalar, 0–1):
```
0.30 × network_service_level
0.20 × critical_service_level    # BLOOD-* at ICU + ER
0.20 × allocation_accuracy       # surplus + stockout penalty per ward-round
0.15 × event_response            # MCI preposition, supplier switch, quarantine, coldchain
0.10 × budget_efficiency
0.05 × waste_control
−     justification_penalty      # up to −0.15 for incoherent expedited justifications
```

No per-step shaping rewards used in training.

---

## Model

| Property | Choice |
|---|---|
| Model | Qwen 3.5 4B |
| Fine-tuning | QLoRA (4-bit NF4 base, bf16 LoRA adapters) |
| Rationale | 4B gives sufficient baseline capability for tool-use + reasoning; QLoRA fits comfortably on A10G/A100 with the batch sizes needed |

---

## Algorithm: Episode-Level GRPO

**Why GRPO**: No critic network needed, lower memory overhead, works well with sparse terminal rewards.  
**Why episode-level**: The reward is a single terminal scalar reflecting overall episode performance across all 8 rounds — service levels, event responses, budget efficiency are all outcomes of the cumulative sequence of decisions, not any individual round. Round-level GRPO would require attributing credit to individual rounds, which demands either a value function (defeating GRPO's purpose) or heuristic decomposition of a reward that is inherently holistic. Episode-level is the correct unit: run G=4 full episodes from the same seed, get G terminal rewards, normalize within the group, broadcast that single advantage signal to every token generated across all 8 rounds. The model learns which full episode strategies lead to better outcomes relative to its own other rollouts on the same starting state.

Basically it is multi turn GRPO.

```
What is GRPO (from wikipedia):
Multi-Turn GRPO is an extension of Group Relative Policy Optimization (GRPO) designed for conversational or agentic settings where a model must complete a task over multiple back-and-forth exchanges rather than a single response. Instead of scoring one output per prompt, it rolls out full multi-turn trajectories, assigns a reward at the end (or per turn), and uses the relative performance across a group of sampled trajectories as the baseline — eliminating the need for a separate critic model. This makes it well-suited for training LLMs on complex sequential tasks like tool use, coding agents, and multi-step reasoning.
```


### Group Size

- **G = 4** rollouts per seed (same initial environment state, different model samples due to temperature > 0)
- G=4 is tractable on a single GPU with QLoRA 4B; gives enough group variance for stable advantage estimates

### GRPO Loss

For each episode group sharing seed `s`:
```
rewards    = [r1, r2, r3, r4]
mean_r     = mean(rewards)
std_r      = std(rewards) + ε
advantage_i = (r_i − mean_r) / std_r
```

Loss sums over all generated tokens across all 8 rounds of the episode, weighted by the group-relative advantage:
```
loss = −Σ_rounds Σ_tokens [ advantage × log_prob(token) × clip_mask ] + β × KL_penalty
```

Reference model (frozen base) used for KL term. QLoRA base is shared between policy and reference — only LoRA adapters are updated.

---

## Context Structure (Key Design Decision)

The environment is **stateful**. Each round the agent receives a fresh round brief and queries the simulation for current ground truth (inventory, ward history, pipeline orders, active events). The environment state carries all cross-round continuity — no conversation history needs to pass between rounds.

```
Round N context:
  [round_brief]
  → tool call: read_inbox
  ← inbox results
  → tool call: view_requests
  ← pending requests
  → tool call: query_ward_history / query_erp / query_supplier
  ← query results
  → tool call: submit_po (optional)
  → tool call: submit_allocation_plan
  ← allocation confirmation
  → tool call: advance_round
  ← next round brief (or terminal summary)

Round N+1 context: fresh window, same pattern
```

**Context length per round**: ~500–2000 tokens. No accumulation across rounds. No context length concerns.

**Episode for backprop**: 8 separate forward passes (one per round), all sharing the same episode-level advantage scalar.

---

## Parallel Episode Architecture

```
B = 8 seeds sampled from training pool
G = 4 rollouts per seed
Total = 32 parallel episodes per training step
```

All 32 episodes run synchronously:
1. All active episodes generate observations (round briefs)
2. Batch all 32 observations into a single model forward pass
3. Route tool call responses back to each environment instance
4. Continue within-round tool use until `advance_round` is called
5. Collect terminal rewards at episode end
6. Group by seed → compute GRPO advantages within each group of 4
7. Compute loss, update LoRA weights

### What Synchronous Means

All 32 episodes share a **single inference loop** — no threads, no async queues, no independent processes. Each iteration of the loop:

1. Collect observations from all episodes currently waiting for a model response
2. Batch them into one forward pass
3. Distribute responses back, execute tool calls against each environment instance
4. Repeat

```
iteration 1:  active=32  →  batch_size=32  →  forward pass  →  distribute
iteration 2:  active=32  →  batch_size=32  →  forward pass  →  distribute
...
iteration N:  active=19  →  batch_size=19  →  ...  (some episodes finished)
...
iteration M:  active=2   →  batch_size=2   →  ...
              all done → collect 32 terminal rewards → GRPO update
```

Episodes that finish early (after their round 8 `advance_round`) simply stop contributing to the batch. The loop runs until all 32 are done. This is fine — no synchronization needed, no padding needed, late episodes don't block anything.

The reason async is unnecessary: tool calls are pure Python method calls on the simulation (microseconds). The only real latency is model inference, which is already maximally utilized by batching all active episodes together each iteration.

**Direct environment import** — `MedchainSimulation` instantiated in-process. No Docker, no WebSocket, no network calls. Pure Python method calls.

```python
from server.simulation import MedchainSimulation
from server.tasks import make_task_config
```

---

## Training Seeds

| Split | Seeds | Count |
|---|---|---|
| Training | 0–49 | 50 |
| Held-out eval | 50–59 | 10 |

Per-step: sample 8 seeds without replacement from the training pool per training step. Rotate through the full pool across steps.

---

## Evaluation Strategy

### Setup
- **Seeds**: 50–59 (held-out, never seen during training)
- **Difficulties**: `light` / `medium` / `heavy` — all three evaluated
- **Configs**: 10 seeds × 3 difficulties = 30 environment configurations
- **Rollouts per config**: 3–5 with greedy sampling (temperature ≈ 0)
- **Total eval episodes**: 90–150

### Batching During Eval

Eval uses the **same batched episode loop as training** — all eval episodes run in parallel through a single inference loop, observations batched into one forward pass per iteration. The only differences from training:
- `torch.no_grad()` throughout — no gradient computation
- Greedy / near-greedy sampling (temperature ≈ 0) instead of temperature > 0
- No GRPO grouping — each episode is independent, just collecting metrics
- Can push batch sizes larger since no optimizer states or gradient graph in memory

This makes eval fast enough to run 90–150 episodes in a reasonable wall-clock time, and the implementation is a clean subset of the training loop code.

### Metrics

| Metric | Purpose |
|---|---|
| Mean episode score | Primary improvement signal |
| Score std dev | Consistency, not just average |
| `critical_service_level` | Non-negotiable floor — never stock out blood |
| `allocation_accuracy` | Core skill — learning to discount padded requests |
| `event_response` | Emergent behavior — MCI preposition, quarantine |
| `budget_efficiency` | Did it over-order? |
| Score by difficulty | light / medium / heavy breakdown |
| Baseline delta | Δ vs untuned Qwen 3.5 4B |

### Baseline
Run untuned Qwen 3.5 4B on all 30 eval configs before training begins. This is the ground truth comparison point. Expected baseline behavior:
- Reasonable `network_service_level` (model knows not to let things run out)
- Poor `event_response` (won't proactively quarantine or preposition blood on MCI warning)
- Poor `allocation_accuracy` on General ward (will trust padded requests)

### Eval Frequency
- Eval before training (baseline)
- Eval every K training steps (e.g., every 50–100 steps)
- Eval on final checkpoint

### Eval Script
Separate `eval.py`, checkpoint-agnostic:
```
python eval.py --base                        # untuned baseline
python eval.py --checkpoint checkpoints/step_200
```

Reports all metrics above, outputs to W&B and local JSON.

---

## Infrastructure

- **Platform**: AWS EC2 (single GPU instance)
- **Target GPU**: A100 40GB+ preferred; A10G 24GB workable with QLoRA batch tuning
- **Logging**: Weights & Biases — track reward components separately, not just total score
- **Checkpointing**: Save LoRA adapter weights every K steps

we will use HuggingFace Libraries

---

## Key Risks & What to Watch

| Risk | Signal | Mitigation |
|---|---|---|
| Group variance collapses | All 4 rollouts get same reward → zero gradient | Ensure temperature > 0 (0.6–0.8 range) |
| Model ignores inbox | `event_response` stays near zero | Check tool call logs — is it calling `read_inbox`? |
| Over-allocates to General | `allocation_accuracy` low, `budget_efficiency` low | Core learning signal — should improve with training |
| Never stocks out ICU/ER blood | `critical_service_level` drops | Alarm threshold — if this degrades, something is wrong |
| Malformed tool calls | Episodes fail early | Return error strings from simulation (already handled), model learns from them |

---

## Qwen 3.5 4B — Usage Notes

### Thinking Mode Decision

**Start with non-thinking mode** (`enable_thinking=False`).

Qwen 3.5 4B has thinking ON by default — must be disabled explicitly.

- The task is structured enough (tool calls → JSON plans) that the GRPO reward signal can teach the right patterns without explicit CoT
- Non-thinking recommended params (`temperature=0.7`) fit naturally into GRPO's exploration range

**When to reconsider**: If `allocation_accuracy` and `event_response` plateau early and the model can't learn padding detection or event response, switch to thinking mode. The explicit reasoning chain helps with those two specific skills.

### Loading with QLoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
)
model = get_peft_model(model, lora_config)
```

### Generation (Training Rollouts)

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    chat_template_kwargs={"enable_thinking": False},
)

outputs = model.generate(
    **tokenizer(text, return_tensors="pt").to(model.device),
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    do_sample=True,
)
```

### Generation (Eval — greedy)

```python
outputs = model.generate(
    **tokenizer(text, return_tensors="pt").to(model.device),
    max_new_tokens=512,
    do_sample=False,   # greedy — valid only in non-thinking mode
)
```

### Key Gotcha

`transformers >= 4.51.0` required — older versions throw an import error on Qwen 3.5.
