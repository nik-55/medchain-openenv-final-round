# Add repo root to path so sibling modules (train, server, config) are importable
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import uuid
import argparse
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from server.simulation import MedchainSimulation
from server.tasks import make_task_config
from server.grader import (
    _network_service_level,
    _critical_service_level,
    _allocation_accuracy,
    _event_response,
    _budget_efficiency,
    _waste_control,
)
from train import SYSTEM_PROMPT, parse_tool_calls, TOOL_SCHEMAS
from config import (
    MODEL_ID, MAX_TURNS, MAX_ROUND_TURNS,
    EVAL_SEEDS, DIFFICULTIES, ROLLOUTS, EVAL_BATCH_SIZE,
)


def load_eval_model(checkpoint: str | None = None):
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
    model = PeftModel.from_pretrained(base, checkpoint) if checkpoint else base
    model.eval()
    return model, tok


def run_eval(model, tok, step_label: str, batch_size: int = EVAL_BATCH_SIZE) -> dict:
    """Batched greedy eval — episodes processed in mini-batches to bound VRAM."""
    # ── Create eval episodes ─────────────────────────────────────────────────
    episodes = []
    for seed in EVAL_SEEDS:
        for diff in DIFFICULTIES:
            for _ in range(ROLLOUTS):
                sim = MedchainSimulation(make_task_config(seed=seed, difficulty=diff))
                brief = sim.reset(seed=seed, episode_id=str(uuid.uuid4()))
                episodes.append({
                    "seed": seed,
                    "diff": diff,
                    "sim": sim,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": brief},
                    ],
                    "done": False,
                    "reward": 0.0,
                    "round_turns": 0,
                })

    print(f"[eval] {len(episodes)} episodes  batch_size={batch_size}  label={step_label}")

    # ── Batched greedy loop — mirrors rollout_func, no_grad, do_sample=False ─
    with torch.no_grad():
        for turn in range(MAX_TURNS):
            active = [ep for ep in episodes if not ep["done"]]
            if not active:
                break
            n_done = len(episodes) - len(active)
            vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"[eval turn {turn:3d}] active={len(active)}  done={n_done}/{len(episodes)}  VRAM={vram:.2f}GB")

            tok.padding_side = "left"

            for b_start in range(0, len(active), batch_size):
                batch = active[b_start : b_start + batch_size]
                print(f"  [eval turn {turn} batch {b_start//batch_size + 1}/{-(-len(active)//batch_size)}] generating {len(batch)} eps  VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")

                texts = [
                    tok.apply_chat_template(
                        ep["messages"],
                        tools=TOOL_SCHEMAS,
                        tokenize=False,
                        add_generation_prompt=True,
                        chat_template_kwargs={"enable_thinking": False},
                    )
                    for ep in batch
                ]

                for i, ep in enumerate(batch):
                    print(f"\n{'='*80}")
                    print(f"[INPUT] turn={turn} seed={ep['seed']} diff={ep['diff']} round_turn={ep['round_turns']} n_messages={len(ep['messages'])}")
                    print(f"{'='*80}")
                    print(texts[i])
                    print(f"{'='*80}\n")

                inputs = tok(texts, return_tensors="pt", padding=True).to(model.device)
                padded_len = inputs["input_ids"].shape[1]
                print(f"  [batch] input_ids shape={tuple(inputs['input_ids'].shape)}  padded_len={padded_len}  VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")

                gen = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )

                for i, ep in enumerate(batch):
                    comp = gen[i, padded_len:]
                    eos_pos = (comp == tok.eos_token_id).nonzero()
                    if len(eos_pos):
                        comp = comp[: eos_pos[0].item() + 1]

                    text = tok.decode(comp, skip_special_tokens=True)

                    print(f"\n{'─'*80}")
                    print(f"[OUTPUT] turn={turn} seed={ep['seed']} diff={ep['diff']}  tokens={len(comp)}")
                    print(f"{'─'*80}")
                    print(text)
                    print(f"{'─'*80}\n")

                    ep["messages"].append({"role": "assistant", "content": text})

                    tool_calls = parse_tool_calls(text)
                    if not tool_calls:
                        print(f"  [eval ep seed={ep['seed']} {ep['diff']}] turn={turn} — NO TOOL CALL PARSED")
                        ep["messages"].append({"role": "user", "content": "Use a tool."})
                    else:
                        print(f"  [eval ep seed={ep['seed']} {ep['diff']}] parsed {len(tool_calls)} tool call(s): {[tc.get('name') for tc in tool_calls]}")
                        for tc in tool_calls:
                            name = tc.get("name", "")
                            args = tc.get("arguments", {})
                            sim = ep["sim"]
                            print(f"  [TOOL CALL] {name}({args})")
                            try:
                                result = (
                                    getattr(sim, name)(**args)
                                    if hasattr(sim, name)
                                    else f"ERROR: Unknown tool '{name}'"
                                )
                            except Exception as e:
                                result = f"ERROR: {e}"
                                print(f"  [TOOL ERROR] {name}({args}): {e}")

                            print(f"  [TOOL RESULT] {name} →\n{result}\n")
                            ep["messages"].append({"role": "user", "content": f"<tool_response>{result}</tool_response>"})

                            if name == "advance_round":
                                if sim._done:
                                    ep["done"] = True
                                    ep["reward"] = sim._last_reward
                                    print(f"  [DONE] seed={ep['seed']} {ep['diff']}  reward={ep['reward']:.4f}")
                                else:
                                    ep["messages"] = [
                                        {"role": "system", "content": SYSTEM_PROMPT},
                                        {"role": "user", "content": result},
                                    ]
                                    ep["round_turns"] = 0
                                    print(f"  [ADVANCE ROUND] seed={ep['seed']} {ep['diff']} → next round  context reset")
                                break

                    ep["round_turns"] += 1

                    if not ep["done"] and ep["round_turns"] >= MAX_ROUND_TURNS:
                        print(f"  [FORCE advance_round] seed={ep['seed']} {ep['diff']} stuck {ep['round_turns']} turns")
                        result = ep["sim"].advance_round()
                        print(f"  [FORCE RESULT] →\n{result}\n")
                        if ep["sim"]._done:
                            ep["done"] = True
                            ep["reward"] = ep["sim"]._last_reward
                            print(f"  [DONE forced] seed={ep['seed']} {ep['diff']}  reward={ep['reward']:.4f}")
                        else:
                            ep["messages"] = [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": result},
                            ]
                            ep["round_turns"] = 0

    # ── Collect per-component metrics ────────────────────────────────────────
    metrics: dict[str, list] = defaultdict(list)
    for ep in episodes:
        s = ep["sim"]._state
        t = ep["sim"]._task
        metrics["score"].append(ep["reward"])
        metrics[f"score_{ep['diff']}"].append(ep["reward"])
        metrics["network_sl"].append(_network_service_level(s))
        metrics["critical_sl"].append(_critical_service_level(s, t))
        metrics["alloc_acc"].append(_allocation_accuracy(s, t))
        metrics["event_resp"].append(_event_response(s, t))
        metrics["budget_eff"].append(_budget_efficiency(s, t))
        metrics["waste_ctrl"].append(_waste_control(s))

    summary = {k: sum(v) / len(v) for k, v in metrics.items()}
    summary["score_std"] = torch.tensor(metrics["score"]).std().item()
    summary["n_episodes"] = len(episodes)

    os.makedirs("eval_results", exist_ok=True)
    out_path = f"eval_results/{step_label}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Eval: {step_label} ===")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true", help="Evaluate untuned base model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to LoRA checkpoint dir")
    args = parser.parse_args()

    if not args.base and not args.checkpoint:
        parser.error("Provide --base or --checkpoint <path>")

    model, tok = load_eval_model(None if args.base else args.checkpoint)
    label = "base" if args.base else os.path.basename(args.checkpoint.rstrip("/"))
    run_eval(model, tok, step_label=label)


if __name__ == "__main__":
    main()
