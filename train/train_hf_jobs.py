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
# train_hf_jobs.py — MedChain GRPO cloud training via Hugging Face Jobs
# Qwen3.5-2B + 4-bit NF4 + LoRA + GRPO on A100-80GB
#
# This is a self-contained UV PEP 723 inline script that:
#   1. Clones the repo to /tmp/sst-final
#   2. Installs dependencies from the inline block above
#   3. Runs full GRPO training with a custom episode rollout
#   4. Pushes the final LoRA adapter to HuggingFace Hub
#
# Submit via HF Jobs API:
#   hf_jobs("uv", {
#       "script": open("train/train_hf_jobs.py").read(),
#       "flavor": "a100-large",
#       "timeout": "6h",
#       "secrets": {"HF_TOKEN": "$HF_TOKEN", "WANDB_API_KEY": "$WANDB_API_KEY"},
#   })
#
# For Colab/Kaggle (T4 GPU), use train_colab.py in the repo root instead.

import os, sys, subprocess

# ── Clone repo ────────────────────────────────────────────────────────────────
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
    MODEL_ID, G, B, MAX_NEW_TOKENS, MAX_TURNS, MAX_ROUND_TURNS,
    MAX_STEPS, LR, SAVE_STEPS, EVAL_BATCH_SIZE,
    MAX_COMPLETION_LENGTH, LOGGING_STEPS, GRADIENT_CHECKPOINTING,
    TEMPERATURE, TOP_P, TOP_K,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, LORA_BIAS,
    CKPT_DIR, HF_REPO_ID,
)


# Try causal-conv1d for faster Qwen3.5 Conv1d layers — optional, torch fallback is identical
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "causal-conv1d"],
    capture_output=True,
)
print("causal-conv1d:", "OK" if r.returncode == 0 else "unavailable (torch fallback is fine)")

# ── Imports ───────────────────────────────────────────────────────────────────
import re, uuid, warnings
import torch
import torch.nn.functional as F
import transformers, peft, trl, bitsandbytes, accelerate
import datasets as _ds

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer
from collections import Counter

warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")

print(f"{'Library':<18} Version")
print("─" * 32)
for _name, _lib in [
    ("torch",        torch),
    ("transformers", transformers),
    ("peft",         peft),
    ("trl",          trl),
    ("bitsandbytes", bitsandbytes),
    ("datasets",     _ds),
    ("accelerate",   accelerate),
]:
    print(f"  {_name:<16} {_lib.__version__}")
print(f"\nPython : {sys.version.split()[0]}")

# ── GPU check ─────────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected.")

props = torch.cuda.get_device_properties(0)
print(f"\nGPU  : {torch.cuda.get_device_name(0)}")
print(f"VRAM : {props.total_memory / 1e9:.1f} GB")
print(f"BF16 : {torch.cuda.is_bf16_supported()}")

# ── Auth ──────────────────────────────────────────────────────────────────────
WANDB_API_KEY = ""
HF_TOKEN      = ""

from huggingface_hub import login as hf_login, HfApi
hf_login(token=HF_TOKEN)
try:
    HfApi(token=HF_TOKEN).whoami()
    print(f"HF login OK → pushing to {HF_REPO_ID}")
except Exception as e:
    print(f"HF whoami FAILED: {e} — push may fail later")

if WANDB_API_KEY:
    import wandb
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.login(key=WANDB_API_KEY, relogin=True)
    print("WandB login OK")
    REPORT_TO = "wandb"
else:
    print("WANDB_API_KEY not set — training metrics will not be logged")
    REPORT_TO = "none"

os.makedirs(CKPT_DIR, exist_ok=True)

from train import SYSTEM_PROMPT, parse_tool_calls, MAX_TURNS, MAX_ROUND_TURNS, build_dataset, TOOL_SCHEMAS
from server.simulation import MedchainSimulation
from server.tasks import make_task_config

COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
USE_BF16      = torch.cuda.is_bf16_supported()
USE_FP16      = not USE_BF16

print(f"\nModel         : {MODEL_ID}  (4-bit NF4)")
print(f"Compute dtype : {COMPUTE_DTYPE}  (bf16={USE_BF16}  fp16={USE_FP16})")
print(f"B={B}  G={G}  active_episodes={B*G}  MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
score_peak_gb = MAX_NEW_TOKENS * B * G * 248320 * 4 / 1e9
vram_total_gb = props.total_memory / 1e9
print(f"VRAM estimate — weights≈1.5GB  scores_peak≈{score_peak_gb:.1f}GB  total≈{1.5+score_peak_gb+2:.1f}GB  ({(1.5+score_peak_gb+2)/vram_total_gb*100:.0f}% of {vram_total_gb:.0f}GB)")

# ── Load model + tokenizer ────────────────────────────────────────────────────
print(f"\nLoading {MODEL_ID} with 4-bit NF4 ...")
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

layer_counts = Counter(type(m).__name__ for m in model.modules())
total_p = sum(p.numel() for p in model.parameters())
linear4b_n = sum(1 for _, m in model.named_modules() if type(m).__name__ == "Linear4bit")
print(f"Total params : {total_p/1e6:.1f}M")
print(f"Linear4bit   : {linear4b_n}  (quantized)")
print(f"Conv1d       : {layer_counts.get('Conv1d', 0)}  (GatedDeltaNet hybrid — not quantized, expected)")
print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Quant sanity check ────────────────────────────────────────────────────────
print("\n=== Quant sanity check ===")
if linear4b_n == 0:
    raise RuntimeError("QUANT FAILED: zero Linear4bit layers — model loaded in full precision!")
print(f"  Linear4bit count: {linear4b_n}  ✓")

_qc_text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello"}],
    tokenize=False, add_generation_prompt=True,
    chat_template_kwargs={"enable_thinking": False},
)
_qc_ids = tokenizer(_qc_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    _qc_logits = model(**_qc_ids).logits
assert not torch.isnan(_qc_logits).any(), "QUANT FAILED: NaN in logits!"
assert not torch.isinf(_qc_logits).any(), "QUANT FAILED: Inf in logits!"
print(f"  Forward pass: logits shape={tuple(_qc_logits.shape)}  dtype={_qc_logits.dtype}  no NaN/Inf  ✓")
# logits dtype is float32 regardless of compute_dtype — normal with bnb 4-bit
print(f"  VRAM after quant check: {torch.cuda.memory_allocated()/1e9:.2f} GB")
del _qc_ids, _qc_logits
torch.cuda.empty_cache()
print("=== Quant sanity check PASSED ===\n")

# ── Rollout + reward functions ────────────────────────────────────────────────
_current_rewards: list[float] = []


def reward_func(completions, prompts, **kwargs) -> list[float]:
    return list(_current_rewards)


def medchain_rollout(prompts, trainer) -> dict:
    global _current_rewards
    _model = trainer.model
    tok    = trainer.processing_class
    _model.eval()

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
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
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
                            print(f"  [ep{i}] advance_round → next round")
                        break

            ep["round_turns"] += 1
            if not ep["done"] and ep["round_turns"] >= MAX_ROUND_TURNS:
                print(f"  [ep{i}] FORCE advance_round (stuck {ep['round_turns']} turns)")
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

    _current_rewards = [ep["reward"] for ep in episodes]
    rewards = _current_rewards
    n_done  = sum(1 for ep in episodes if ep["done"])
    print(f"[rollout] done={n_done}/{len(episodes)}  "
          f"mean={sum(rewards)/len(rewards):.4f}  "
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

# ── LoRA config + dataset ─────────────────────────────────────────────────────
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    task_type="CAUSAL_LM",
)

train_dataset = build_dataset()
print(f"\nDataset  : {len(train_dataset)} rows")
print(f"Sample 0 : {train_dataset[0]['prompt'][-1]}")

# ── Sanity check ──────────────────────────────────────────────────────────────
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
print(f"  Response   : {_decoded[:120]!r}")
print(f"  Tool calls : {_calls}")
print(f"  VRAM       : {torch.cuda.memory_allocated()/1e9:.2f} GB")
torch.cuda.empty_cache()
print("Sanity check passed.")

# ── Base eval (before LoRA is applied) ───────────────────────────────────────
# from eval import run_eval
# print("\nRunning BASE eval (untrained model, 15 episodes, greedy) ...")
# torch.cuda.empty_cache()
# base_summary = run_eval(model, tokenizer, step_label="base", batch_size=EVAL_BATCH_SIZE)
# print("Base eval done → eval_results/base.json")

# ── WandB init ────────────────────────────────────────────────────────────────
if REPORT_TO == "wandb":
    wandb.init(
        project="medchain-grpo",
        name=f"qwen35-2b-a10g-B{B}G{G}",
        config=dict(
            model_id=MODEL_ID, B=B, G=G, max_new_tokens=MAX_NEW_TOKENS,
            compute_dtype=str(COMPUTE_DTYPE), lr=LR, max_steps=MAX_STEPS,
            gpu=torch.cuda.get_device_name(0),
        ),
        resume="allow",
    )

# ── Train ─────────────────────────────────────────────────────────────────────
grpo_config = GRPOConfig(
    num_generations             = G,
    per_device_train_batch_size = B,
    max_completion_length       = MAX_COMPLETION_LENGTH,
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

trainer = GRPOTrainer(
    model            = model,
    processing_class = tokenizer,
    reward_funcs     = reward_func,
    args             = grpo_config,
    train_dataset    = train_dataset,
    peft_config      = peft_config,
    rollout_func     = medchain_rollout,
)

print(f"\nTraining: steps={MAX_STEPS}  lr={LR}  B={B}  G={G}  save_every={SAVE_STEPS}  bf16={USE_BF16}  fp16={USE_FP16}")
print(f"Checkpoints → Hub ({HF_REPO_ID}) every {SAVE_STEPS} steps")
print(f"VRAM before train: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

trainer.train()
print("\nTraining complete.")
print(f"VRAM after train: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Post-training eval ────────────────────────────────────────────────────────
print("\nRunning POST-TRAINING eval (15 episodes, greedy) ...")
torch.cuda.empty_cache()
trained_summary = run_eval(trainer.model, tokenizer, step_label="post_training", batch_size=EVAL_BATCH_SIZE)

if REPORT_TO == "wandb":
    wandb.log({f"eval/base/{k}": v for k, v in base_summary.items()})
    wandb.log({f"eval/trained/{k}": v for k, v in trained_summary.items()})

print(f"\nScore delta : {trained_summary['score'] - base_summary['score']:+.4f}")
print("Post-training eval done → eval_results/post_training.json")

# ── Push final model ──────────────────────────────────────────────────────────
print(f"\nPushing final model to {HF_REPO_ID} ...")
try:
    HfApi(token=HF_TOKEN).create_repo(HF_REPO_ID, exist_ok=True, private=False)
    trainer.model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    print(f"Uploaded → https://huggingface.co/{HF_REPO_ID}")
except Exception as e:
    print(f"HF upload failed: {e}")
    print("Model was saved to checkpoint dir but NOT pushed to Hub.")

if REPORT_TO == "wandb":
    wandb.finish()
print("Done.")
