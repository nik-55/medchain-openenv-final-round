# %% [markdown]
# # MedChain GRPO Training — Colab / Kaggle Notebook
#
# **Model**: Qwen3.5-2B · **Quantisation**: 4-bit NF4 · **Adapter**: LoRA (r=16) · **Algorithm**: GRPO
#
# This notebook trains a Qwen3.5-2B model to coordinate hospital supply chains
# using Group Relative Policy Optimisation (GRPO) with live simulation rollouts.
#
# ## What is being trained?
# The model learns to act as a central pharmacy coordinator over 8 rounds:
# - Navigate 5 siloed enterprise systems (ERP, WMS, Supplier Portal, Finance SAP, Messaging)
# - Audit ward requests (ICU, ER, General) for strategic inflation
# - Escalate contested requests to a clinical-review board
# - File purchase orders, track lot expiry, respond to supply-chain events
# - All rewards are deterministic — no LLM judge in the loop
#
# ## Quick links
# - **WandB run**: https://api.wandb.ai/links/nikm5502-nikhil-mahajna/5pri4ooa
# - **HF Model**: https://huggingface.co/nik-55/medchain-grpo-qwen35-2b
# - **Environment repo**: https://github.com/nik-55/sst-final
#
# ## GPU requirements
# - **T4 (16 GB)**: B=8, G=4, MAX_NEW_TOKENS=256 → ~11 GB peak VRAM (fp16)
# - **A10G (24 GB)**: increase B to 16 (see train/train_hf_jobs.py)
# - **A100 (80 GB)**: increase B to 32
#
# ## How to run
# Run cells top-to-bottom. Re-running any cell is safe — guards are in place.
# Set WANDB_API_KEY and HF_TOKEN in Cell 4 before running Cell 11 (training).

# %% ── Cell 1: Detect environment · GPU check · clone repo ──────────────────
import os, sys, subprocess

# ── Detect runtime environment ───────────────────────────────────────────────
if os.path.exists("/content"):
    ENV, COLAB_ROOT = "colab",  "/content"
elif os.path.exists("/kaggle/working"):
    ENV, COLAB_ROOT = "kaggle", "/kaggle/working"
else:
    ENV, COLAB_ROOT = "local",  os.getcwd()

print(f"Environment : {ENV}")
print(f"Root        : {COLAB_ROOT}")

# ── GPU check (fail fast if no GPU) ──────────────────────────────────────────
import torch
print(f"\nPyTorch     : {torch.__version__}")
print(f"CUDA        : {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected — attach a GPU runtime before running.")

props = torch.cuda.get_device_properties(0)
print(f"GPU         : {torch.cuda.get_device_name(0)}")
print(f"VRAM        : {props.total_memory / 1e9:.1f} GB")
print(f"CUDA cap    : sm_{props.major}{props.minor}")
print(f"BF16 hw     : {torch.cuda.is_bf16_supported()}")

# ── Clone / update repo ───────────────────────────────────────────────────────
REPO_URL = "https://github.com/nik-55/sst-final.git"
REPO_DIR = os.path.join(COLAB_ROOT, "sst-final")
WORK_DIR = os.path.join(REPO_DIR, "openenv-hack")

if not os.path.exists(REPO_DIR):
    print(f"\nCloning {REPO_URL} ...")
    r = subprocess.run(["git", "clone", "--depth=1", REPO_URL, REPO_DIR],
                       capture_output=True, text=True)
    print(r.stdout or "(no stdout)")
    if r.returncode != 0:
        print("STDERR:", r.stderr)
        raise RuntimeError("git clone failed")
    print("Clone OK")
else:
    print(f"\nRepo exists at {REPO_DIR} — pulling latest ...")
    r = subprocess.run(["git", "-C", REPO_DIR, "pull"], capture_output=True, text=True)
    print(r.stdout.strip() or "(already up to date)")

# Guard: only chdir once per session
if os.getcwd() != WORK_DIR:
    os.chdir(WORK_DIR)
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

print(f"\ncwd  : {os.getcwd()}")
print(f"path : {sys.path[0]}")

# %% ── Cell 2: Install / upgrade dependencies ────────────────────────────────
print("Installing dependencies from requirements-train.txt ...")
req_path = os.path.join(WORK_DIR, "requirements-train.txt")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "-r", req_path,
], check=True)

# causal-conv1d speeds up Qwen3.5 GatedDeltaNet Conv1d layers on GPU.
# If the CUDA build fails (T4 Turing wheels can be missing), torch fallback
# is numerically identical — training still works, just slightly slower.
print("\nTrying causal-conv1d ...")
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "causal-conv1d"],
    capture_output=True, text=True,
)
if r.returncode == 0:
    print("  causal-conv1d installed — fast Conv1d path enabled")
else:
    print("  causal-conv1d unavailable — using torch fallback (OK)")

print("\nDependencies ready.")

# %% ── Cell 3: Imports + version table ───────────────────────────────────────
import re, json, uuid, random, warnings
import torch
import torch.nn.functional as F
import transformers, peft, trl, bitsandbytes, accelerate
import datasets as _ds
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

# Suppress the torch_dtype deprecation noise from older transformers
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")

print(f"{'Library':<18} Version")
print("─" * 32)
for name, lib in [
    ("torch",          torch),
    ("transformers",   transformers),
    ("peft",           peft),
    ("trl",            trl),
    ("bitsandbytes",   bitsandbytes),
    ("datasets",       _ds),
    ("accelerate",     accelerate),
    ("wandb",          wandb),
]:
    print(f"  {name:<16} {lib.__version__}")

print(f"\nPython : {sys.version.split()[0]}")

# %% ── Cell 4: Auth — WandB + HuggingFace ────────────────────────────────────
# NOTE: test credentials — rotate after the hackathon.
WANDB_API_KEY = "<WANDB_KEY>"
HF_TOKEN      = "<HF_TOKEN>"   # drop trailing % (copy-paste artifact)
HF_REPO_ID    = "nik-55/medchain-grpo-qwen35-2b"

os.environ["WANDB_API_KEY"] = WANDB_API_KEY
wandb.login(key=WANDB_API_KEY, relogin=True)
print("WandB  login OK")

from huggingface_hub import login as hf_login, HfApi
hf_login(token=HF_TOKEN)
print("HF     login OK")

# Verify HF token can reach the API before spending GPU time
try:
    api = HfApi(token=HF_TOKEN)
    api.whoami()
    print(f"HF     whoami OK → will push to {HF_REPO_ID}")
except Exception as e:
    print(f"HF     whoami FAILED: {e}")
    print("       Training will continue but HF upload at the end may fail.")

# %% ── Cell 5: Checkpoint directory ─────────────────────────────────────────
# Colab: persist to Google Drive so checkpoints survive runtime resets.
# Kaggle: /kaggle/working/ is persisted across cell runs within a session.

if ENV == "colab":
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        CKPT_DIR = "/content/drive/MyDrive/medchain_checkpoints"
        print("Checkpoints → Google Drive")
    except Exception as e:
        CKPT_DIR = os.path.join(COLAB_ROOT, "medchain_checkpoints")
        print(f"Drive mount failed ({e})\nCheckpoints → local: {CKPT_DIR}")
elif ENV == "kaggle":
    CKPT_DIR = "/kaggle/working/medchain_checkpoints"
    print("Checkpoints → /kaggle/working/")
else:
    CKPT_DIR = os.path.join(COLAB_ROOT, "medchain_checkpoints")
    print(f"Checkpoints → {CKPT_DIR}")

os.makedirs(CKPT_DIR, exist_ok=True)
print(f"CKPT_DIR = {CKPT_DIR}")

# %% ── Cell 6: Constants + dtype auto-detection ──────────────────────────────
# Import shared constants from train.py (SYSTEM_PROMPT, parse_tool_calls, etc.)
from train import SYSTEM_PROMPT, parse_tool_calls, MAX_TURNS, MAX_ROUND_TURNS, build_dataset, TOOL_SCHEMAS
from server.simulation import MedchainSimulation
from server.tasks import make_task_config

MODEL_ID       = "Qwen/Qwen3.5-2B"
G              = 4    # rollouts per seed — keep at 4 for GRPO advantage quality
B              = 8    # seeds per step — 8×4=32 active episodes; output_scores peak ~8 GB, total ~11 GB on T4
MAX_NEW_TOKENS = 256  # tool calls are <100 tokens; 256 is safe headroom

# T4 (Turing sm_75) has no hardware BF16 — use FP16.
# Ampere+ (A100, L4, A10) support BF16 natively — auto-selected.
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
USE_BF16      = torch.cuda.is_bf16_supported()
USE_FP16      = not USE_BF16

print(f"Model          : {MODEL_ID}")
print(f"Compute dtype  : {COMPUTE_DTYPE}  (bf16={USE_BF16}  fp16={USE_FP16})")
print(f"B={B}  G={G}  MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
print(f"Max rollout batch : {B * G} active episodes")

# VRAM estimate (worst-case, all episodes generating simultaneously):
# output_scores peak = MAX_NEW_TOKENS × B×G × vocab(248077) × 4 bytes (float32 scores)
score_peak_gb = MAX_NEW_TOKENS * B * G * 248077 * 4 / 1e9
print(f"\nVRAM estimate:")
print(f"  Model weights (4-bit 2B)  : ~1.5 GB")
print(f"  output_scores peak        : ~{score_peak_gb:.1f} GB   ← biggest variable")
print(f"  KV cache + activations    : ~1-2 GB")
print(f"  Total peak (rough)        : ~{1.5 + score_peak_gb + 1.5:.1f} GB")
print(f"\n  T4 has 16 GB → {'OK' if 1.5 + score_peak_gb + 1.5 < 14 else 'TIGHT — reduce B'}")

# %% ── Cell 7: Load model + tokenizer ────────────────────────────────────────
import torch.nn as nn
from collections import Counter

print(f"Loading {MODEL_ID} with 4-bit NF4 ...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,  # fp16 on T4, bf16 on Ampere+
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto",
)
model = prepare_model_for_kbit_training(model)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ── Architecture debug ────────────────────────────────────────────────────────
layer_counts = Counter(type(m).__name__ for m in model.modules())
print(f"\nModel class    : {model.__class__.__name__}")
print("Top layer types:")
for name, cnt in layer_counts.most_common(10):
    print(f"  {cnt:4d}  {name}")

total_p    = sum(p.numel() for p in model.parameters())
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal params   : {total_p/1e6:.1f}M")
print(f"Trainable      : {trainable_p/1e6:.1f}M  (before LoRA)")

# Qwen3.5 is a GatedDeltaNet hybrid — 18 Conv1d + 6 Attention layers.
# Conv1d layers are NOT quantized by bitsandbytes (expected — they stay in COMPUTE_DTYPE).
conv1d_n  = layer_counts.get("Conv1d", 0)
linear4b_n = sum(1 for _, m in model.named_modules()
                 if type(m).__name__ == "Linear4bit")
print(f"\nLinear4bit     : {linear4b_n}  (quantized)")
print(f"Conv1d         : {conv1d_n}  (stays in {COMPUTE_DTYPE}, not quantized — expected)")

print(f"\nVRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated  "
      f"/ {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")

# %% ── Cell 8: Rollout + reward functions ────────────────────────────────────
# Identical logic to train.py::medchain_rollout.
# Only differences: MAX_NEW_TOKENS=256 (was 512) and VRAM debug prints.

_current_rewards: list[float] = []


def reward_func(completions, prompts, **kwargs) -> list[float]:
    return list(_current_rewards)


def medchain_rollout(prompts, trainer) -> dict:
    global _current_rewards
    _model = trainer.model
    tok    = trainer.processing_class
    _model.eval()

    # ── Init B*G episodes ────────────────────────────────────────────────────
    episodes = []
    for prompt in prompts:
        meta = prompt[-1]["content"]
        seed = int(re.search(r"seed=(\d+)", meta).group(1))
        diff = re.search(r"diff=(\w+)", meta).group(1)
        for _ in range(G):
            sim   = MedchainSimulation(make_task_config(seed=seed, difficulty=diff))
            brief = sim.reset(seed=seed, episode_id=str(uuid.uuid4()))
            episodes.append({
                "sim":              sim,
                "messages":         [{"role": "system", "content": SYSTEM_PROMPT},
                                     {"role": "user",   "content": brief}],
                "comp_ids":         [],
                "logprobs":         [],
                "first_prompt_ids": None,
                "done":             False,
                "reward":           0.0,
                "round_turns":      0,
            })
    print(f"[rollout] init {len(episodes)} episodes ({len(prompts)} seeds × G={G})")

    # ── Batched inference loop ───────────────────────────────────────────────
    for step in range(MAX_TURNS):
        active = [ep for ep in episodes if not ep["done"]]
        if not active:
            break
        if step % 10 == 0:
            n_done = len(episodes) - len(active)
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"[rollout turn {step:3d}] active={len(active)}  done={n_done}/{len(episodes)}  VRAM={vram:.2f}GB")

        tok.padding_side = "left"
        texts = [
            tok.apply_chat_template(
                ep["messages"], tools=TOOL_SCHEMAS, tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            for ep in active
        ]
        inputs     = tok(texts, return_tensors="pt", padding=True).to(_model.device)
        padded_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen = _model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tok.eos_token_id,
            )

        for i, ep in enumerate(active):
            comp    = gen.sequences[i, padded_len:].cpu()
            eos_pos = (comp == tok.eos_token_id).nonzero()
            if len(eos_pos):
                comp = comp[:eos_pos[0].item() + 1]

            T = len(comp)
            if T > 0 and gen.scores:
                scores   = torch.stack([gen.scores[t][i].cpu() for t in range(T)])
                lp       = F.log_softmax(scores.float(), dim=-1)
                token_lp = lp.gather(1, comp.unsqueeze(-1)).squeeze(-1)
            else:
                token_lp = torch.zeros(T)

            ep["comp_ids"].append(comp)
            ep["logprobs"].append(token_lp)

            if ep["first_prompt_ids"] is None:
                raw     = gen.sequences[i, :padded_len].cpu()
                non_pad = (raw != tok.pad_token_id).nonzero()
                ep["first_prompt_ids"] = raw[non_pad[0].item():] if len(non_pad) else raw

            # skip_special_tokens=True: strips <|im_end|>/<think> before storing
            # in messages so they don't corrupt the chat template on the next turn
            text = tok.decode(comp, skip_special_tokens=True)
            ep["messages"].append({"role": "assistant", "content": text})

            tool_calls = parse_tool_calls(text)
            if not tool_calls:
                print(f"  [ep{i}] turn={step} round_turn={ep['round_turns']} — no tool call  text={text[:80]!r}")
                ep["messages"].append({
                    "role":    "user",
                    "content": "Use a tool. Call advance_round when done with this round.",
                })
            else:
                for tc in tool_calls:
                    name, args, sim = tc.get("name", ""), tc.get("arguments", {}), ep["sim"]
                    try:
                        result = (getattr(sim, name)(**args)
                                  if hasattr(sim, name)
                                  else f"ERROR: Unknown tool '{name}'")
                    except Exception as e:
                        result = f"ERROR: {e}"
                        print(f"  [ep{i}] TOOL ERROR {name}({args}): {e}")

                    ep["messages"].append({"role": "user", "content": f"<tool_response>{result}</tool_response>"})

                    if name == "advance_round":
                        if sim._done:
                            ep["done"]   = True
                            ep["reward"] = sim._last_reward
                            print(f"  [ep{i}] DONE  reward={ep['reward']:.4f}")
                        else:
                            ep["messages"]    = [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user",   "content": result},
                            ]
                            ep["round_turns"] = 0
                            print(f"  [ep{i}] advance_round → next round  round_turns reset")
                        break

            ep["round_turns"] += 1
            if not ep["done"] and ep["round_turns"] >= MAX_ROUND_TURNS:
                print(f"  [ep{i}] FORCE advance_round (stuck {ep['round_turns']} turns in round)")
                result = ep["sim"].advance_round()
                if ep["sim"]._done:
                    ep["done"]   = True
                    ep["reward"] = ep["sim"]._last_reward
                    print(f"  [ep{i}] DONE (forced)  reward={ep['reward']:.4f}")
                else:
                    ep["messages"]    = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": result},
                    ]
                    ep["round_turns"] = 0

        # Periodic VRAM + progress log (every 25 steps)
        if step % 25 == 0:
            n_done = sum(1 for ep in episodes if ep["done"])
            vram   = torch.cuda.memory_allocated() / 1e9
            print(f"  [rollout step {step:3d}] active={len(active):2d}  "
                  f"done={n_done:2d}/{len(episodes)}  VRAM={vram:.2f} GB")

    _current_rewards = [ep["reward"] for ep in episodes]

    n_done  = sum(1 for ep in episodes if ep["done"])
    rewards = _current_rewards
    print(f"[rollout] done={n_done}/{len(episodes)}  "
          f"reward mean={sum(rewards)/len(rewards):.4f}  "
          f"min={min(rewards):.4f}  max={max(rewards):.4f}  "
          f"zeros={rewards.count(0.0)}/{len(rewards)}  "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")

    return {
        "prompt_ids": [
            ep["first_prompt_ids"] if ep["first_prompt_ids"] is not None
            else torch.tensor([], dtype=torch.long)
            for ep in episodes
        ],
        "completion_ids": [
            torch.cat(ep["comp_ids"]) if ep["comp_ids"]
            else torch.tensor([], dtype=torch.long)
            for ep in episodes
        ],
        "logprobs": [
            torch.cat(ep["logprobs"]) if ep["logprobs"]
            else torch.tensor([], dtype=torch.float)
            for ep in episodes
        ],
    }

# %% ── Cell 9: LoRA config + dataset + quick sanity check ────────────────────
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",   # adapts all 187 Linear layers; Conv1d skipped (fine)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

train_dataset = build_dataset()
print(f"Dataset  : {len(train_dataset)} rows")
print(f"Sample 0 : {train_dataset[0]['prompt'][-1]}")

# ── Quick sanity check: one generation before committing to full training ─────
# Catches VRAM OOM, import errors, and chat-template issues cheaply.
print("\nSanity check: single generation pass ...")
_msgs = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": "Call the read_inbox tool now with filter=unread."},
]
_text = tokenizer.apply_chat_template(
    _msgs, tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True,
    chat_template_kwargs={"enable_thinking": False},
)
_ids = tokenizer([_text], return_tensors="pt").to(model.device)
with torch.no_grad():
    _gen = model.generate(
        **_ids, max_new_tokens=64, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
_comp    = _gen[0, _ids["input_ids"].shape[1]:]
_decoded = tokenizer.decode(_comp, skip_special_tokens=True)
_calls   = parse_tool_calls(_decoded)

print(f"  Response     : {_decoded[:120]!r}")
print(f"  Tool calls   : {_calls}")
print(f"  VRAM         : {torch.cuda.memory_allocated()/1e9:.2f} GB")

if not _calls:
    print("  NOTE: model did not produce a tool call on this prompt — "
          "this is expected before training; it will learn to do so.")

torch.cuda.empty_cache()
print("Sanity check passed — starting training setup.")

# %% ── Cell 10: Base eval (untrained model) ──────────────────────────────────
# Run before GRPOTrainer is created — LoRA has NOT been applied yet.
# Gives a baseline score to compare against the trained model.
# evaluate.py lives in train/ — add that subdirectory to sys.path
import sys as _sys
_train_dir = os.path.join(WORK_DIR, "train")
if _train_dir not in _sys.path:
    _sys.path.insert(0, _train_dir)
from evaluate import run_eval

print("Running BASE eval (untrained model, 90 episodes, greedy) ...")
torch.cuda.empty_cache()
base_summary = run_eval(model, tokenizer, step_label="base", batch_size=16)
print("Base eval done → eval_results/base.json")

# %% ── Cell 11: WandB init + train ───────────────────────────────────────────
MAX_STEPS  = 500
LR         = 1e-4
SAVE_STEPS = 25   # more frequent than train.py default (50) — Colab may disconnect

wandb.init(
    project = "medchain-grpo",
    name    = f"qwen35-2b-{props.name.replace(' ', '_')}-B{B}G{G}",
    config  = dict(
        model_id       = MODEL_ID,
        B              = B,
        G              = G,
        max_new_tokens = MAX_NEW_TOKENS,
        compute_dtype  = str(COMPUTE_DTYPE),
        lr             = LR,
        max_steps      = MAX_STEPS,
        gpu            = torch.cuda.get_device_name(0),
    ),
    resume  = "allow",  # safe to re-run cell if training crashes mid-way
)

config = GRPOConfig(
    num_generations             = G,
    per_device_train_batch_size = B,
    max_completion_length       = 16384,
    learning_rate               = LR,
    max_steps                   = MAX_STEPS,
    bf16                        = USE_BF16,
    fp16                        = USE_FP16,
    output_dir                  = CKPT_DIR,
    save_steps                  = SAVE_STEPS,
    logging_steps               = 1,
    report_to                   = "wandb",
    dataloader_num_workers      = 0,   # avoid multiprocessing deadlocks in Colab
)

trainer = GRPOTrainer(
    model            = model,
    processing_class = tokenizer,
    reward_funcs     = reward_func,
    args             = config,
    train_dataset    = train_dataset,
    peft_config      = peft_config,
    rollout_func     = medchain_rollout,
)

print(f"Training config:")
print(f"  steps={MAX_STEPS}  lr={LR}  B={B}  G={G}  save_every={SAVE_STEPS}")
print(f"  bf16={USE_BF16}  fp16={USE_FP16}")
print(f"  output_dir={CKPT_DIR}")
print(f"  VRAM before train: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print()

trainer.train()
print("\nTraining complete.")
print(f"VRAM after train : {torch.cuda.memory_allocated()/1e9:.2f} GB")

# %% ── Cell 12: Post-training eval ───────────────────────────────────────────
print("Running POST-TRAINING eval (trained model, 90 episodes, greedy) ...")
torch.cuda.empty_cache()
trained_summary = run_eval(trainer.model, tokenizer, step_label="post_training", batch_size=16)

# Log both evals to wandb for easy comparison
wandb.log({f"eval/base/{k}": v for k, v in base_summary.items()})
wandb.log({f"eval/trained/{k}": v for k, v in trained_summary.items()})
print(f"\nScore delta : {trained_summary['score'] - base_summary['score']:+.4f}")
print("Post-training eval done → eval_results/post_training.json")

# %% ── Cell 13: Save final model + upload to HF Hub ──────────────────────────
FINAL_CKPT = os.path.join(CKPT_DIR, "final")
trainer.save_model(FINAL_CKPT)
tokenizer.save_pretrained(FINAL_CKPT)
print(f"Model saved locally → {FINAL_CKPT}")

print(f"\nPushing LoRA adapter to HF Hub: {HF_REPO_ID} ...")
try:
    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_REPO_ID, exist_ok=True, private=False)
    trainer.model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    print(f"Uploaded → https://huggingface.co/{HF_REPO_ID}")
except Exception as e:
    print(f"HF upload failed: {e}")
    print(f"Adapter is still saved locally at {FINAL_CKPT}")

wandb.finish()
print("Done.")
