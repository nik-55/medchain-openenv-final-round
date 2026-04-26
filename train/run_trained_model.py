"""
run_trained_model.py — run a trained (or base) Qwen3.5 checkpoint against the
OpenEnv WebSocket server and print per-turn results.

Replaces the old infer.py. Run from the repo root or from train/:
    python train/run_trained_model.py
    SEED=42 DIFFICULTY=heavy CHECKPOINT=checkpoints/checkpoint-200 python train/run_trained_model.py

Original file: infer.py

Environment variables (all optional, defaults shown):
    SERVER_URL   http://localhost:8000   Running OpenEnv server endpoint
    SEED         0                       Episode seed
    DIFFICULTY   medium                  light | medium | heavy
    CHECKPOINT                           Path to LoRA adapter dir; empty = base model
    MODEL_ID     Qwen/Qwen3.5-4B        HuggingFace model ID

Usage:
    # Server must already be running:
    #   uv run server --host 0.0.0.0 --port 8000

    python infer.py
    SEED=42 DIFFICULTY=heavy CHECKPOINT=checkpoints/checkpoint-200 python infer.py
"""

# Add repo root to path so sibling modules (train, client, config) are importable
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))

import asyncio
import os
import uuid

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from openenv.core.env_server.mcp_types import CallToolAction

from client import MedchainEnv
from train import MAX_ROUND_TURNS, MAX_TURNS, SYSTEM_PROMPT, parse_tool_calls

# ── Config from environment variables ────────────────────────────────────────
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")
SEED = int(os.environ.get("SEED", "0"))
DIFFICULTY = os.environ.get("DIFFICULTY", "medium")
CHECKPOINT = os.environ.get("CHECKPOINT", "").strip() or None
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-4B")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(base, CHECKPOINT) if CHECKPOINT else base
    model.eval()
    return model, tok


# ── Single-turn greedy generation ─────────────────────────────────────────────

def generate(model, tok, messages: list[dict]) -> str:
    tok.padding_side = "left"
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    comp = gen[0, prompt_len:]
    eos_pos = (comp == tok.eos_token_id).nonzero()
    if len(eos_pos):
        comp = comp[: eos_pos[0].item() + 1]
    return tok.decode(comp, skip_special_tokens=False)


# ── Episode loop ──────────────────────────────────────────────────────────────

async def run_episode(model, tok) -> float:
    episode_id = str(uuid.uuid4())
    print(f"\n{'='*60}")
    print(f"Episode : {episode_id}")
    print(f"Server  : {SERVER_URL}")
    print(f"Seed    : {SEED}  Difficulty: {DIFFICULTY}")
    print(f"Model   : {MODEL_ID}" + (f"  +LoRA: {CHECKPOINT}" if CHECKPOINT else "  (base)"))
    print(f"{'='*60}\n")

    async with MedchainEnv(base_url=SERVER_URL) as env:
        # Reset — returns the round-1 brief
        result = await env.reset(seed=SEED, episode_id=episode_id, difficulty=DIFFICULTY)
        brief = result.observation.metadata.get("dashboard", "")
        print(f"[BRIEF]\n{brief}\n")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": brief},
        ]

        done = False
        final_reward = 0.0
        turn = 0
        round_turns = 0
        current_round = 1

        while not done and turn < MAX_TURNS:
            turn += 1

            # Generate model response
            response = generate(model, tok, messages)
            messages.append({"role": "assistant", "content": response})

            tool_calls = parse_tool_calls(response)

            if not tool_calls:
                print(f"  [turn {turn}] no tool call — prompting again")
                messages.append({
                    "role": "user",
                    "content": "Use a tool. Call advance_round when done with this round.",
                })
            else:
                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})

                    result = await env.step(CallToolAction(tool_name=name, arguments=args))
                    tool_result = result.observation.metadata.get("tool_result", "")

                    messages.append({"role": "tool", "content": tool_result})
                    print(f"  [turn {turn}] {name}({args}) → {tool_result[:120].strip()}")

                    if name == "advance_round":
                        if result.done:
                            done = True
                            final_reward = result.reward or 0.0
                        else:
                            current_round += 1
                            print(f"\n--- Round {current_round} ---")
                            # Fresh context window for next round
                            messages = [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": tool_result},
                            ]
                            round_turns = 0
                        break

            round_turns += 1

            # Force advance if model loops within a round
            if not done and round_turns >= MAX_ROUND_TURNS:
                print(f"  [force advance_round after {round_turns} round turns]")
                result = await env.step(CallToolAction(tool_name="advance_round", arguments={}))
                tool_result = result.observation.metadata.get("tool_result", "")
                if result.done:
                    done = True
                    final_reward = result.reward or 0.0
                else:
                    current_round += 1
                    print(f"\n--- Round {current_round} ---")
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": tool_result},
                    ]
                    round_turns = 0

    print(f"\n{'='*60}")
    print(f"Final reward : {final_reward:.4f}")
    print(f"Total turns  : {turn}")
    print(f"{'='*60}\n")
    return final_reward


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    model, tok = load_model()
    asyncio.run(run_episode(model, tok))


if __name__ == "__main__":
    main()
