# /// script
# dependencies = [
#   "torch>=2.1.0",
#   "transformers>=5.2.0",
#   "peft>=0.12.0",
#   "bitsandbytes>=0.43.0",
#   "trl>=0.29.0",
#   "datasets>=2.18.0",
#   "accelerate>=0.30.0",
#   "wandb>=0.17.0",
#   "openenv-core[core]>=0.2.1",
#   "fastmcp>=2.0.0",
#   "numpy>=1.24.0",
#   "fastapi>=0.115.0",
#   "uvicorn>=0.24.0",
#   "pydantic>=2.0.0",
# ]
# ///
#
# MedChain GRPO — Single-turn Offline Training (HF Jobs)
#
# Step 1 (collect_rollouts_hf.py):
#   Rubber-stamp policy drives the simulation through all 8 rounds per episode,
#   collecting (round_context → true_need) pairs — no GPU needed.
#
# Step 2 (this script):
#   Load the pre-collected dataset. Each training example is a single turn:
#   "given this round context, output allocation JSON."
#   Reward = per-round allocation accuracy scored against hidden true_need.
#   No custom rollout_func — TRL generates completions internally.
#   Dense reward signal (per round, not terminal) → faster learning.
#
# Submit via:
#   hf_jobs("uv", {
#       "script": open("train_hf_offline.py").read(),
#       "flavor": "a100-large",
#       "timeout": "6h",
#       "secrets": {"HF_TOKEN": "$HF_TOKEN", "WANDB_API_KEY": "$WANDB_API_KEY"},
#   })

import os, sys, subprocess

REPO_URL = "https://github.com/nik-55/sst-final.git"
REPO_DIR = "/tmp/sst-final"
WORK_DIR = os.path.join(REPO_DIR, "openenv-hack")

print(f"Cloning {REPO_URL} (master) ...")
subprocess.run(
    ["git", "clone", "--depth=1", "--branch", "master", REPO_URL, REPO_DIR],
    check=True,
)
os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)
print(f"cwd: {os.getcwd()}")

from config import (
    MODEL_ID, G, B, MAX_STEPS, LR, SAVE_STEPS,
    MAX_COMPLETION_LENGTH, LOGGING_STEPS, GRADIENT_CHECKPOINTING,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, LORA_BIAS,
    CKPT_DIR, HF_REPO_ID,
)

r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "causal-conv1d"],
    capture_output=True,
)
print("causal-conv1d:", "OK" if r.returncode == 0 else "unavailable (torch fallback is fine)")

import re, json, warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from huggingface_hub import login as hf_login, HfApi

warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")

HF_DATASET_ID = "nik-55/medchain-rollouts"

WANDB_API_KEY = ""
HF_TOKEN      = ""

hf_login(token=HF_TOKEN)
try:
    HfApi(token=HF_TOKEN).whoami()
    print(f"HF login OK → pushing to {HF_REPO_ID}")
except Exception as e:
    print(f"HF whoami FAILED: {e}")

if WANDB_API_KEY:
    import wandb
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.login(key=WANDB_API_KEY, relogin=True)
    print("WandB login OK")
    REPORT_TO = "wandb"
else:
    print("WANDB_API_KEY not set — metrics will not be logged")
    REPORT_TO = "none"

os.makedirs(CKPT_DIR, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected.")

props = torch.cuda.get_device_properties(0)
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
USE_BF16      = torch.cuda.is_bf16_supported()
USE_FP16      = not USE_BF16

print(f"\nGPU  : {torch.cuda.get_device_name(0)}")
print(f"VRAM : {props.total_memory / 1e9:.1f} GB")
print(f"Model: {MODEL_ID}  (4-bit NF4 + LoRA)  B={B}  G={G}")
print("Mode : single-turn offline GRPO — dense per-round allocation reward\n")

# ── Model + tokenizer ─────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto",
)
model = prepare_model_for_kbit_training(model)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Load + prepare rollout dataset ───────────────────────────────────────────
# Priority: local file (ROLLOUTS_FILE env var or rollouts.jsonl in cwd) → HF Hub
_local_file = os.environ.get("ROLLOUTS_FILE", "rollouts.jsonl")
if os.path.exists(_local_file):
    print(f"Loading rollout dataset from local file: {_local_file} ...")
    raw_ds = load_dataset("json", data_files=_local_file, split="train")
else:
    print(f"Loading rollout dataset from {HF_DATASET_ID} ...")
    raw_ds = load_dataset(HF_DATASET_ID, split="train")
print(f"Loaded {len(raw_ds)} rows  ({len(raw_ds) // B} steps per epoch at B={B})")

# Build fast reward lookup: _rollout_idx → hidden state row
rollout_lookup: dict[int, dict] = {
    int(row["_rollout_idx"]): row for row in raw_ds
}

# Pre-format prompts using Qwen3.5 chat template with thinking disabled.
# Storing the formatted string means TRL tokenises once without re-applying
# the template, which avoids the default enable_thinking=True behaviour.
MAX_PROMPT_TOKENS = 5000  # keeps ~75% of context; tail-truncation drops early ward history

def _format_prompt(example: dict) -> dict:
    text = tokenizer.apply_chat_template(
        example["prompt"],
        tools=None,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > MAX_PROMPT_TOKENS:
        # System message is at the START of the formatted text; tail truncation
        # drops it, causing the model to ignore "Output ONLY the JSON" and
        # generate verbose explanations that hit max_completion_length before
        # closing '}' → parse failure for all completions → reward_std=0 → loss=0.
        # Fix: always keep system message + tail of user content.
        # apply_chat_template requires a user turn; format system manually.
        sys_content = example["prompt"][0]["content"]
        sys_text    = f"<|im_start|>system\n{sys_content}<|im_end|>\n"
        sys_ids     = tokenizer.encode(sys_text, add_special_tokens=False)
        tail_ids = ids[-(MAX_PROMPT_TOKENS - len(sys_ids)):]
        ids  = sys_ids + tail_ids
        text = tokenizer.decode(ids, skip_special_tokens=False)
    return {"prompt": text}

print("Pre-formatting prompts (applying chat template, enable_thinking=False) ...")
train_dataset = raw_ds.map(_format_prompt, desc="format prompts")
# Keep only the columns TRL needs; hidden-state fields travel via rollout_lookup
train_dataset = train_dataset.select_columns(["prompt", "_rollout_idx"])
print(f"Dataset ready: {len(train_dataset)} rows")

# ── Reward function ───────────────────────────────────────────────────────────

def _parse_allocation(text: str) -> dict | None:
    """Parse model output to {ward_id: {sku: qty}}, tolerating <think> blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    lo, hi = text.find("{"), text.rfind("}")
    if lo < 0 or hi <= lo:
        return None
    try:
        plan = json.loads(text[lo : hi + 1])
    except Exception:
        return None
    if not isinstance(plan, dict):
        return None
    out: dict = {}
    for ward_id, inner in plan.items():
        if not isinstance(inner, dict):
            continue
        out[ward_id] = {}
        for sku, qty in inner.items():
            try:
                out[ward_id][sku] = max(0, int(qty))
            except (TypeError, ValueError):
                continue
    return out or None


def _score_allocation(
    plan: dict | None,
    true_needs: dict[str, dict],
    requested: dict[str, dict],
    priority: dict[str, float],
    central_stock: dict[str, int],
) -> float:
    """
    Per-round allocation accuracy in [-0.2, 1.0].
    Mirrors the grader's _allocation_accuracy formula, with a small penalty
    for over-stock (sum across wards exceeds central stock).
    """
    if plan is None:
        return -0.1  # parse failure

    # Over-stock penalty: model shouldn't allocate more than central stock
    sku_total: dict[str, int] = {}
    for ward_id, inner in plan.items():
        for sku, qty in inner.items():
            sku_total[sku] = sku_total.get(sku, 0) + qty
    over_stock_pen = sum(
        0.03 for sku, total in sku_total.items()
        if total > central_stock.get(sku, 0)
    )

    scores: list[float] = []
    for ward_id, req_ward in requested.items():
        prio      = priority.get(ward_id, 0.5)
        true_ward = true_needs.get(ward_id, {})
        for sku, req_q in req_ward.items():
            alloc_q   = min(plan.get(ward_id, {}).get(sku, 0), req_q)
            true_need = true_ward.get(sku, float(req_q))
            consumed  = min(alloc_q, true_need)
            surplus_r = max(0.0, alloc_q - true_need) / max(alloc_q, 1)
            stockout  = 1 if consumed + 1e-6 < true_need else 0
            shortage_pen = prio * stockout
            surplus_pen  = surplus_r * (1.0 - prio) * 0.5
            scores.append(max(0.0, 1.0 - shortage_pen - surplus_pen))

    if not scores:
        return 0.0
    return max(-0.2, sum(scores) / len(scores) - over_stock_pen)


def reward_func(completions, prompts, _rollout_idx, **kwargs) -> list[float]:
    scores: list[float] = []
    for completion, idx in zip(completions, _rollout_idx):
        row        = rollout_lookup[int(idx)]
        plan       = _parse_allocation(completion)
        true_needs = json.loads(row["true_needs"])
        requested  = json.loads(row["requested"])
        priority   = json.loads(row["priority"])
        cs         = json.loads(row["central_stock"])
        scores.append(_score_allocation(plan, true_needs, requested, priority, cs))
    return scores

# ── Sanity check: baseline rubber-stamp score ─────────────────────────────────
print("\nBaseline check (rubber-stamp policy on first 30 rows) ...")
import uuid as _uuid
from server.simulation import MedchainSimulation
from server.tasks import make_task_config

_sample   = [rollout_lookup[i] for i in range(min(30, len(rollout_lookup)))]
_baseline = []
for row in _sample:
    tn = json.loads(row["true_needs"])
    rq = json.loads(row["requested"])
    pr = json.loads(row["priority"])
    cs = json.loads(row["central_stock"])
    _remaining = dict(cs)
    plan: dict = {}
    for ward_id, reqs in rq.items():
        plan[ward_id] = {}
        for sku, qty in reqs.items():
            take = min(qty, _remaining.get(sku, 0))
            plan[ward_id][sku] = take
            _remaining[sku] = _remaining.get(sku, 0) - take
    _baseline.append(_score_allocation(plan, tn, rq, pr, cs))
print(f"  Rubber-stamp mean score: {sum(_baseline)/len(_baseline):.4f}  "
      f"(target: model beats this)")

# ── LoRA + GRPO config ────────────────────────────────────────────────────────
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    task_type="CAUSAL_LM",
)

# A10G 24GB: B=1, G=4 → 4 seqs × 6024 tok; attention peak ~1.5GB/layer → fits comfortably.
# GRAD_ACCUM=B keeps effective seeds-per-step equal to config B (8 for a10g).
B_OFFLINE  = 1
GRAD_ACCUM = B  # = 8 on a10g; compensates for B_OFFLINE=1

# Help the CUDA allocator handle the large, variable-length sequences
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

grpo_config = GRPOConfig(
    num_generations             = G,
    per_device_train_batch_size = B_OFFLINE,
    gradient_accumulation_steps = GRAD_ACCUM,
    max_completion_length       = 1024,  # 23-SKU JSON ~350 tok + think stub; 512 clips too early
    learning_rate               = LR,
    max_steps                   = MAX_STEPS,
    bf16                        = USE_BF16,
    fp16                        = USE_FP16,
    gradient_checkpointing      = GRADIENT_CHECKPOINTING,
    output_dir                  = CKPT_DIR,
    save_steps                  = SAVE_STEPS,
    logging_steps               = LOGGING_STEPS,
    report_to                   = REPORT_TO,
    push_to_hub                 = True,
    hub_model_id                = HF_REPO_ID,
    hub_strategy                = "every_save",
)

# ── WandB ─────────────────────────────────────────────────────────────────────
if REPORT_TO == "wandb":
    wandb.init(
        project="medchain-grpo",
        name=f"single-turn-offline-B{B}G{G}",
        config=dict(
            model_id=MODEL_ID, B=B, G=G,
            mode="single_turn_offline",
            rollout_dataset=HF_DATASET_ID,
            n_rollouts=len(raw_ds),
            lr=LR, max_steps=MAX_STEPS,
            gpu=torch.cuda.get_device_name(0),
        ),
        resume="allow",
    )

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model            = model,
    processing_class = tokenizer,
    reward_funcs     = reward_func,
    args             = grpo_config,
    train_dataset    = train_dataset,
    peft_config      = peft_config,
    # No rollout_func — TRL generates completions internally
)

print(f"\nTraining: steps={MAX_STEPS}  lr={LR}  B={B_OFFLINE}  G={G}  "
      f"grad_accum={GRAD_ACCUM}  effective_seeds={B_OFFLINE*GRAD_ACCUM}  "
      f"save_every={SAVE_STEPS}  bf16={USE_BF16}")
print(f"Checkpoints → Hub ({HF_REPO_ID}) every {SAVE_STEPS} steps")
print(f"VRAM before train: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

trainer.train()
print("\nTraining complete.")

# ── Push final model ──────────────────────────────────────────────────────────
print(f"\nPushing final model to {HF_REPO_ID} ...")
try:
    HfApi(token=HF_TOKEN).create_repo(HF_REPO_ID, exist_ok=True, private=False)
    trainer.model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    print(f"Uploaded → https://huggingface.co/{HF_REPO_ID}")
except Exception as e:
    print(f"HF upload failed: {e}")

if REPORT_TO == "wandb":
    wandb.finish()
print("Done.")
