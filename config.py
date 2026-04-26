# Shared hyperparameters for train.py, train_hf.py, and eval.py.
# Switch GPU target by changing GPU below — no other edits needed.

# ── GPU target ────────────────────────────────────────────────────────────────
GPU = "a10g"   # "a10g" | "a100"

# ── Common ────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3.5-2B"

# Episode / rollout shape (not VRAM-sensitive)
G               = 4       # GRPO group size (rollouts per seed)
MAX_NEW_TOKENS  = 256     # per-turn generation cap (tool calls are <100 tok)
MAX_TURNS       = 60      # total inference loop guard across all rounds
MAX_ROUND_TURNS = 12      # force advance_round if model loops within a round

# Training
MAX_STEPS              = 500
LR                     = 1e-4
SAVE_STEPS             = 25
MAX_COMPLETION_LENGTH  = 16384
LOGGING_STEPS          = 1
GRADIENT_CHECKPOINTING = True

# Sampling (rollout generation)
TEMPERATURE = 0.7
TOP_P       = 0.8
TOP_K       = 20

# LoRA
LORA_R              = 16
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = "all-linear"
LORA_BIAS           = "none"

# Paths / Hub
CKPT_DIR   = "/tmp/medchain_checkpoints"
HF_REPO_ID = "nik-55/medchain-grpo-qwen35-2b"

# Eval
EVAL_SEEDS   = [2]
DIFFICULTIES = ["light", "medium", "heavy"]
ROLLOUTS     = 1    # rollouts per seed×difficulty

# ── GPU-specific (VRAM-sensitive) ─────────────────────────────────────────────
# A10G-large (24 GB) — NF4 weights ≈ 1.5 GB | scores(B=8 ,G=4,256tok) ≈  8.1 GB | total ≈ 12 GB (50%)
# A100-large (80 GB) — NF4 weights ≈ 1.5 GB | scores(B=32,G=4,256tok) ≈ 32.5 GB | total ≈ 36 GB (45%)
if GPU == "a10g":
    B               = 8
    EVAL_BATCH_SIZE = 4
elif GPU == "a100":
    B               = 32
    EVAL_BATCH_SIZE = 16
else:
    raise ValueError(f"Unknown GPU={GPU!r}; use 'a10g' or 'a100'")
