import re
import json
import uuid
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from server.simulation import MedchainSimulation
from server.tasks import make_task_config
from config import MODEL_ID, G, B, MAX_NEW_TOKENS, MAX_TURNS, MAX_ROUND_TURNS, MAX_STEPS, LR, SAVE_STEPS

SYSTEM_PROMPT = """You are the central supply coordinator for a hospital network running 8 rounds per episode.
Each round lasts 2 simulated days. After advance_round your conversation history is wiped — only the round brief carries over.

═══ MANDATORY ROUND SEQUENCE ═══
1. read_inbox(filter="unread")                         ← ALWAYS first; catches crises and recall alerts
2. view_requests()                                      ← see what wards are asking for (may be padded)
3. query_erp(table="pipeline_orders")                  ← CHECK IN-TRANSIT ORDERS BEFORE ORDERING MORE
4. query_erp(table="inventory", location="central_pharmacy")  ← current on-hand stock
   [steps 3+4 can be called in parallel in a single turn]
5. query_ward_history / query_supplier                  ← additional context if needed (1-2 calls max)
   [multiple query_ward_history calls can be parallelised across wards in one turn]
6. submit_po(...)  [+ file_justification if expedited]  ← order ONLY the net gap (see PO rules below)
7. quarantine_lot(...)  ← ONLY if inbox contains a recall or cold-chain breach alert
8. submit_allocation_plan(plan_json=...)                ← REQUIRED every round; see rules below
9. advance_round()                                      ← LAST call; ends the round

═══ REFERENCE — VALID IDs (use EXACTLY these strings) ═══
Locations (for query_erp and submit_po destination_id):
  central_pharmacy   ward_icu   ward_er   ward_general

Suppliers (for query_supplier and submit_po):
  MEDLINE   (lead 2d, cost 1.0×, all SKUs)
  BACKUP-B  (lead 3d, cost 1.3×, all SKUs — fallback when MEDLINE disrupted)
  FASTMED   (lead 1d, cost 1.8×, all SKUs — use only for life-critical emergencies)

═══ PURCHASE ORDER RULES (critical for budget score) ═══
• ALWAYS query pipeline_orders BEFORE placing any PO. Your context is wiped after each round —
  last round's orders are still in transit and will appear in pipeline_orders.
• Derive consumption from query_ward_history — do NOT invent or memorise fixed quantities.
• Compute for each SKU: net_qty = history_avg_consumption - in_transit_qty - surplus_on_hand.
  *** If net_qty ≤ 0: DO NOT call submit_po. Calling submit_po with quantity ≤ 0 is FORBIDDEN. ***
  *** If net_qty > 0: call submit_po ONCE with quantity = net_qty. ONE call per SKU total. ***
• Sum need across all wards before calling submit_po — never submit per-ward separately.
  All stock routes through central_pharmacy regardless of destination.
• Do NOT call view_requests() more than once per round.

═══ WARD SKU REFERENCE ═══
ward_icu     : BLOOD-RBC  BLOOD-PLT  BLOOD-FFP  IV-SAL-500  ANTIBIO-01  OXY-MASK  SYR-10  GLOVE-001
ward_er      : BLOOD-RBC  BLOOD-PLT  BLOOD-FFP  IV-SAL-500  ANTIBIO-01  OXY-MASK  SYR-10  GLOVE-001  GAUZE-01
ward_general : IV-SAL-500  ANTIBIO-01  SYR-10  GLOVE-001  MASK-001  GAUZE-01

═══ ALLOCATION PLAN RULES (critical for score) ═══
• Include ALL THREE wards — ward_icu, ward_er, ward_general — every single round.
• Include EVERY SKU listed above for each ward. Omitting a SKU is treated as 0 (stockout penalty).
• Never allocate 0 for any SKU. Use ward_history average consumption when uncertain.
• Start from request quantities, then discount by the inflation factor (wards pad 10-60%).
  ICU pads ~10%, ER pads ~20-50%, General pads ~25-60% — allocate closer to true need.
• Blood products (BLOOD-RBC, BLOOD-PLT, BLOOD-FFP) are CRITICAL for ICU and ER — never stockout.
• Allocation comes from central_pharmacy stock. Submit POs first if stock is low.

═══ QUERY RULES ═══
• query_erp: location and sku are optional — but NEVER pass null or empty string.
  Omit the field entirely when you want all rows: query_erp(table="inventory", location="central_pharmacy")
• query_supplier: use only MEDLINE, BACKUP-B, or FASTMED — no other IDs exist.
• file_justification ticket_id: use the PO ticket ID returned by submit_po response, not a guessed ID.

═══ EVENTS TO WATCH FOR ═══
• MCI (mass casualty): blood demand ×2.8 at ICU+ER — pre-position via expedited FASTMED orders.
• Supplier disruption: switch to BACKUP-B for urgent items; FASTMED for life-critical.
• Product recall: quarantine_lot at every listed location using the EXACT lot_id string from the inbox message body — never invent or guess it.
• Cold-chain breach: lot is auto-quarantined by the system; place emergency replenishment order — no quarantine_lot call needed."""


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "read_inbox",
        "description": "Read inbox messages. Always call first each round.",
        "parameters": {"type": "object", "properties": {
            "filter": {"type": "string", "enum": ["unread", "all", "flagged"]}
        }, "required": ["filter"]},
    }},
    {"type": "function", "function": {
        "name": "view_requests",
        "description": "View pending ward supply requests. Requests may be inflated.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "query_ward_history",
        "description": "Query historical requests and allocations for a ward.",
        "parameters": {"type": "object", "properties": {
            "ward_id": {"type": "string"},
            "product_id": {"type": "string"},
            "n_rounds": {"type": "integer"},
        }, "required": ["ward_id"]},
    }},
    {"type": "function", "function": {
        "name": "query_erp",
        "description": (
            "Query ERP system for inventory, expiry, or pipeline orders. "
            "location must be one of: central_pharmacy, ward_icu, ward_er, ward_general. "
            "sku must be a valid SKU string. OMIT location or sku entirely to get all rows — "
            "never pass null or an empty string."
        ),
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string", "enum": ["inventory", "expiry", "pipeline_orders"]},
            "location": {"type": "string", "enum": ["central_pharmacy", "ward_icu", "ward_er", "ward_general"]},
            "sku": {"type": "string"},
        }, "required": ["table"]},
    }},
    {"type": "function", "function": {
        "name": "query_supplier",
        "description": "Query supplier lead time and disruption status. Valid supplier_id values: MEDLINE, BACKUP-B, FASTMED.",
        "parameters": {"type": "object", "properties": {
            "supplier_id": {"type": "string", "enum": ["MEDLINE", "BACKUP-B", "FASTMED"]},
        }, "required": ["supplier_id"]},
    }},
    {"type": "function", "function": {
        "name": "submit_po",
        "description": "Submit a purchase order to a supplier. destination_id must be central_pharmacy (the central hub — stock is allocated from there to wards). Use FASTMED only for life-critical emergencies (1d lead, 1.8× cost).",
        "parameters": {"type": "object", "properties": {
            "supplier_id": {"type": "string", "enum": ["MEDLINE", "BACKUP-B", "FASTMED"]},
            "product_id": {"type": "string"},
            "destination_id": {"type": "string", "enum": ["central_pharmacy", "ward_icu", "ward_er", "ward_general"]},
            "quantity": {"type": "integer"},
            "priority": {"type": "string", "enum": ["standard", "expedited"]},
        }, "required": ["supplier_id", "product_id", "destination_id", "quantity"]},
    }},
    {"type": "function", "function": {
        "name": "file_justification",
        "description": "File justification for an expedited purchase order.",
        "parameters": {"type": "object", "properties": {
            "ticket_id": {"type": "string"},
            "reason": {"type": "string"},
        }, "required": ["ticket_id", "reason"]},
    }},
    {"type": "function", "function": {
        "name": "quarantine_lot",
        "description": "Quarantine a lot due to recall or cold-chain breach.",
        "parameters": {"type": "object", "properties": {
            "location_id": {"type": "string"},
            "sku": {"type": "string"},
            "lot_id": {"type": "string"},
        }, "required": ["location_id", "sku", "lot_id"]},
    }},
    {"type": "function", "function": {
        "name": "submit_allocation_plan",
        "description": "Submit stock allocation plan across all wards as JSON string {ward_id: {sku: qty}}.",
        "parameters": {"type": "object", "properties": {
            "plan_json": {"type": "string"},
        }, "required": ["plan_json"]},
    }},
    {"type": "function", "function": {
        "name": "advance_round",
        "description": "Close the current round and advance to the next. Call after all actions are complete.",
        "parameters": {"type": "object", "properties": {}},
    }},
]

# Integer params that need type coercion (native format gives all values as strings)
_INT_PARAMS: dict[str, set[str]] = {
    tool["function"]["name"]: {
        k for k, v in tool["function"]["parameters"].get("properties", {}).items()
        if v.get("type") == "integer"
    }
    for tool in TOOL_SCHEMAS
}


def parse_tool_calls(text: str) -> list[dict]:
    """Parse Qwen3.5 native XML tool call format."""
    results = []
    for block in re.finditer(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        inner = block.group(1)
        fn = re.search(r'<function=([^>]+)>', inner)
        if not fn:
            continue
        name = fn.group(1).strip()
        args = {}
        for p in re.finditer(r'<parameter=([^>]+)>\n?(.*?)\n?</parameter>', inner, re.DOTALL):
            k, v = p.group(1).strip(), p.group(2).strip()
            if k in _INT_PARAMS.get(name, set()):
                try:
                    v = int(v)
                except (ValueError, TypeError):
                    pass
            args[k] = v
        results.append({"name": name, "arguments": args})
    return results


# Reward closure — rollout_func populates this, reward_func reads it
_current_rewards: list[float] = []


def reward_func(completions, prompts, **kwargs) -> list[float]:
    return list(_current_rewards)


def medchain_rollout(prompts, trainer) -> dict:
    """
    Custom rollout with fresh context windows per round.

    Receives B=8 prompts from TRL. Creates G=4 episodes per seed = 32 total.
    Each episode resets its context window after every advance_round() call.
    All active episodes are batched into one forward pass per loop iteration.
    Returns 32 results; TRL groups consecutive G=4 as one GRPO group.
    """
    global _current_rewards
    model = trainer.model
    tok = trainer.processing_class
    model.eval()

    # ── Initialise 32 episodes (B=8 seeds × G=4 rollouts) ──────────────────
    episodes = []
    for prompt in prompts:
        # seed and difficulty embedded in the last user message by the dataset
        meta = prompt[-1]["content"]
        seed = int(re.search(r"seed=(\d+)", meta).group(1))
        diff = re.search(r"diff=(\w+)", meta).group(1)
        for _ in range(G):
            sim = MedchainSimulation(make_task_config(seed=seed, difficulty=diff))
            brief = sim.reset(seed=seed, episode_id=str(uuid.uuid4()))
            episodes.append({
                "sim": sim,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": brief},
                ],
                "comp_ids": [],           # per-turn cpu tensors
                "logprobs": [],           # per-turn cpu tensors
                "first_prompt_ids": None,
                "done": False,
                "reward": 0.0,
                "round_turns": 0,
            })
    print(f"[rollout] init {len(episodes)} episodes ({len(prompts)} seeds × G={G})")

    # ── Batched fresh-window inference loop ─────────────────────────────────
    for turn in range(MAX_TURNS):
        active = [ep for ep in episodes if not ep["done"]]
        if not active:
            break
        if turn % 10 == 0:
            n_done = len(episodes) - len(active)
            vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"[rollout turn {turn:3d}] active={len(active)}  done={n_done}/{len(episodes)}  VRAM={vram:.2f}GB")

        tok.padding_side = "left"
        texts = [
            tok.apply_chat_template(
                ep["messages"],
                tools=TOOL_SCHEMAS,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            for ep in active
        ]
        inputs = tok(texts, return_tensors="pt", padding=True).to(model.device)
        padded_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(
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
            comp = gen.sequences[i, padded_len:].cpu()

            # Trim at first EOS
            eos_pos = (comp == tok.eos_token_id).nonzero()
            if len(eos_pos):
                comp = comp[: eos_pos[0].item() + 1]

            # Per-token logprobs from generation scores
            T = len(comp)
            if T > 0 and gen.scores:
                scores = torch.stack([gen.scores[t][i].cpu() for t in range(T)])
                lp = F.log_softmax(scores.float(), dim=-1)
                token_lp = lp.gather(1, comp.unsqueeze(-1)).squeeze(-1)
            else:
                token_lp = torch.zeros(T)

            ep["comp_ids"].append(comp)
            ep["logprobs"].append(token_lp)

            # Store first-round prompt_ids for TRL's loss mask
            if ep["first_prompt_ids"] is None:
                raw = gen.sequences[i, :padded_len].cpu()
                non_pad = (raw != tok.pad_token_id).nonzero()
                ep["first_prompt_ids"] = raw[non_pad[0].item():] if len(non_pad) else raw

            # Decode and parse tool calls
            text = tok.decode(comp, skip_special_tokens=True)
            ep["messages"].append({"role": "assistant", "content": text})

            tool_calls = parse_tool_calls(text)
            if not tool_calls:
                print(f"  [ep{i}] turn={turn} round_turn={ep['round_turns']} — no tool call parsed  text={text[:80]!r}")
                ep["messages"].append({
                    "role": "user",
                    "content": "Use a tool. Call advance_round when done with this round.",
                })
            else:
                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    sim = ep["sim"]
                    try:
                        result = (
                            getattr(sim, name)(**args)
                            if hasattr(sim, name)
                            else f"ERROR: Unknown tool '{name}'"
                        )
                    except Exception as e:
                        result = f"ERROR: {e}"
                        print(f"  [ep{i}] TOOL ERROR {name}({args}): {e}")

                    ep["messages"].append({"role": "user", "content": f"<tool_response>{result}</tool_response>"})

                    if name == "advance_round":
                        if sim._done:
                            ep["done"] = True
                            ep["reward"] = sim._last_reward
                            print(f"  [ep{i}] DONE  reward={ep['reward']:.4f}")
                        else:
                            ep["messages"] = [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": result},
                            ]
                            ep["round_turns"] = 0
                            print(f"  [ep{i}] advance_round → next round  round_turns reset")
                        break

            ep["round_turns"] += 1

            # Force advance if model gets stuck within a round
            if not ep["done"] and ep["round_turns"] >= MAX_ROUND_TURNS:
                print(f"  [ep{i}] FORCE advance_round (stuck {ep['round_turns']} turns in round)")
                result = ep["sim"].advance_round()
                if ep["sim"]._done:
                    ep["done"] = True
                    ep["reward"] = ep["sim"]._last_reward
                    print(f"  [ep{i}] DONE (forced)  reward={ep['reward']:.4f}")
                else:
                    ep["messages"] = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": result},
                    ]
                    ep["round_turns"] = 0

    # ── Package results for TRL ─────────────────────────────────────────────
    _current_rewards = [ep["reward"] for ep in episodes]
    n_done = sum(1 for ep in episodes if ep["done"])
    rewards = _current_rewards
    print(f"[rollout] done={n_done}/{len(episodes)}  "
          f"reward mean={sum(rewards)/len(rewards):.4f}  "
          f"min={min(rewards):.4f}  max={max(rewards):.4f}  "
          f"zeros={rewards.count(0.0)}/{len(rewards)}")

    return {
        "prompt_ids": [
            ep["first_prompt_ids"]
            if ep["first_prompt_ids"] is not None
            else torch.tensor([], dtype=torch.long)
            for ep in episodes
        ],
        "completion_ids": [
            torch.cat(ep["comp_ids"])
            if ep["comp_ids"]
            else torch.tensor([], dtype=torch.long)
            for ep in episodes
        ],
        "logprobs": [
            torch.cat(ep["logprobs"])
            if ep["logprobs"]
            else torch.tensor([], dtype=torch.float)
            for ep in episodes
        ],
    }


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer


peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# Do NOT call get_peft_model here — TRL applies it via peft_config
# and uses disable_adapter() for the KL reference model.


def build_dataset() -> Dataset:
    random.seed(42)
    seeds_list = list(range(50)) * 20  # 1000 rows; seeds 0-49 each ×20
    diffs_list = [random.choice(["light", "medium", "heavy"]) for _ in seeds_list]
    return Dataset.from_dict({
        "prompt": [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                # Embed seed/difficulty so rollout_func can extract them
                {"role": "user", "content": f"seed={s};diff={d}"},
            ]
            for s, d in zip(seeds_list, diffs_list)
        ],
    }).shuffle(seed=42)


def main():
    import argparse
    import wandb

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-project", default="medchain-grpo")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--save-steps", type=int, default=SAVE_STEPS)
    args = parser.parse_args()

    wandb.init(project=args.wandb_project)

    model, tokenizer = load_model()
    train_dataset = build_dataset()

    config = GRPOConfig(
        num_generations=G,                   # TRL groups consecutive G=4 results as one GRPO group
        per_device_train_batch_size=B,       # B=8 seeds per step; rollout_func returns B*G=32 results
        max_completion_length=16384,         # ceiling on concatenated completion_ids per episode
        learning_rate=args.lr,
        max_steps=args.max_steps,
        bf16=True,
        output_dir="checkpoints",
        save_steps=args.save_steps,
        logging_steps=1,
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=config,
        train_dataset=train_dataset,
        peft_config=peft_config,
        rollout_func=medchain_rollout,
    )
    trainer.train()


if __name__ == "__main__":
    main()
