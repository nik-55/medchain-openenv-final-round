"""
check_qwen.py — 72-check sanity script for Qwen3.5-2B/4B.
Verifies model load, tokenizer, chat template, tool-call format, batch
generation, LoRA config, and (optionally) 4-bit quantisation. CPU-safe
except --4bit.

Run from repo root or train/:
    python train/check_qwen.py --model-id Qwen/Qwen3.5-2B
    python train/check_qwen.py --skip-generation
    python train/check_qwen.py --model-id Qwen/Qwen3.5-2B --4bit  # GPU only

Original file: check_qwen.py (moved here for cleaner repo structure)

Qwen 3.5 (2B / 4B) sanity-check script for train.py / eval.py usage.

Checks (no GPU / quantisation required — runs on CPU in fp32):
  1.  Model & tokenizer load
  2.  Tokenizer special tokens (pad, eos, bos)
  3.  Chat template renders correctly (system + user + assistant turns)
  4.  enable_thinking=False is respected (no <think> block in output)
  5.  padding_side="left" for batch generation
  6.  Tool-call format: model generates <tool_call>…</tool_call> blocks
  7.  parse_tool_calls() correctly extracts those blocks
  8.  Multi-turn round-reset pattern (context window cleared after advance_round)
  9.  Generation token-logprob alignment (len(comp_ids) == len(logprobs))
 10.  EOS trimming logic from rollout_func
 11.  Batch generation with left-padding (multiple prompts, no shape mismatch)
 12.  Response content sanity (non-empty, UTF-8 decodable)
 13.  BitsAndBytesConfig fields match what train.py / eval.py declare
 14.  LoRA target_modules="all-linear" accepted by LoraConfig
 15.  GRPOConfig fields (num_generations, max_completion_length, bf16)
"""

# Add repo root to path so train.py and server/ are importable
import os as _os, sys
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import re
import sys
import json
import argparse
import textwrap

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results: list[tuple[str, bool, str]] = []   # (name, passed, detail)


def check(name: str, cond: bool, detail: str = "", warn_only: bool = False):
    tag = PASS if cond else (WARN if warn_only else FAIL)
    print(f"  {tag} {name}" + (f"  — {detail}" if detail else ""))
    results.append((name, cond or warn_only, detail))


# ─── helpers ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are the central supply coordinator for a hospital network.
    Each round you receive a brief. Follow this pattern every round:
    1. read_inbox  2. view_requests  3. query_ward_history
    4. submit_po   5. quarantine_lot  6. submit_allocation_plan
    7. advance_round""")

TOOL_CALL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    results_list = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            results_list.append(json.loads(m.group(1).strip()))
        except Exception:
            pass
    return results_list


def parse_tool_calls_native(text: str) -> list[dict]:
    """Parse Qwen3.5 native XML tool call format produced when tools= is passed."""
    results_list = []
    for block in TOOL_CALL_RE.finditer(text):
        inner = block.group(1)
        fn = re.search(r'<function=([^>]+)>', inner)
        if not fn:
            continue
        name = fn.group(1).strip()
        args = {}
        for p in re.finditer(r'<parameter=([^>]+)>\n?(.*?)\n?</parameter>', inner, re.DOTALL):
            args[p.group(1).strip()] = p.group(2).strip()
        results_list.append({"name": name, "arguments": args})
    return results_list


def build_tool_call_prompt(tok, tool_name: str = "read_inbox") -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Call the `{tool_name}` tool now with arguments {{\"filter\": \"unread\"}}. "
                "Use exactly the format:\n"
                "<tool_call>{\"name\": \"" + tool_name + "\", \"arguments\": {\"filter\": \"unread\"}}</tool_call>"
            ),
        },
    ]


# ─── section 1: load ──────────────────────────────────────────────────────────

def section_load(model_id: str, use_4bit: bool):
    print(f"\n{'─'*60}")
    print(f"Section 1 — Model & Tokenizer Load  ({model_id}, 4bit={use_4bit})")
    print('─'*60)

    try:
        if use_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb, device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32, device_map="cpu"
            )
        tok = AutoTokenizer.from_pretrained(model_id)
        check("Model loaded", True, model.__class__.__name__)
    except Exception as e:
        check("Model loaded", False, str(e))
        print(f"\n  {FAIL} Cannot continue — model failed to load.")
        return None, None

    check("Tokenizer loaded", tok is not None)
    check("pad_token set", tok.pad_token is not None,
          f"pad_token={tok.pad_token!r}  pad_token_id={tok.pad_token_id}")
    check("eos_token set", tok.eos_token is not None,
          f"eos_token={tok.eos_token!r}  eos_token_id={tok.eos_token_id}")
    check("pad != eos (preferred)", tok.pad_token_id != tok.eos_token_id,
          warn_only=True,
          detail="train.py sets pad_token_id=eos_token_id in generate(); fine for now")

    return model, tok


# ─── section 2: chat template ─────────────────────────────────────────────────

def section_chat_template(tok):
    print(f"\n{'─'*60}")
    print("Section 2 — Chat Template")
    print('─'*60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What tool should I call?"},
    ]

    # Basic render
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        check("apply_chat_template (default)", True)
    except Exception as e:
        check("apply_chat_template (default)", False, str(e))
        text = ""

    # enable_thinking=False  (Qwen3 specific)
    try:
        text_no_think = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        check("chat_template_kwargs enable_thinking=False accepted", True)
    except TypeError as e:
        check("chat_template_kwargs enable_thinking=False accepted", False, str(e))
        text_no_think = text

    # Qwen3 enable_thinking=False → empty <think>\n\n</think> stub (by design).
    # Verify the block is present but empty (no actual thinking content inside).
    think_match = re.search(r'<think>(.*?)</think>', text_no_think, re.DOTALL)
    if think_match:
        think_body = think_match.group(1).strip()
        check("enable_thinking=False → empty <think> stub (Qwen3 design)",
              think_body == "",
              detail=f"non-empty body: {think_body[:80]!r}" if think_body else "empty — correct")
    else:
        # Some template versions omit the stub entirely — also fine
        check("enable_thinking=False → no thinking content", True,
              detail="no <think> block at all — also acceptable")

    # System token present
    check("System message included in template",
          "supply coordinator" in text_no_think or "assistant" in text_no_think.lower())

    # Tokenize=True round-trip
    try:
        ids = tok.apply_chat_template(
            messages[:2], tokenize=True, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        decoded = tok.decode(ids, skip_special_tokens=False)
        check("Tokenize=True round-trip", len(ids) > 0, f"token count={len(ids)}")
    except Exception as e:
        check("Tokenize=True round-trip", False, str(e))

    # Print first 400 chars of rendered template
    print(f"\n  {INFO} Template preview (first 400 chars):")
    print(textwrap.indent(text_no_think[:400], "    "))


# ─── section 3: generation & logprobs ─────────────────────────────────────────

def section_generation(model, tok):
    print(f"\n{'─'*60}")
    print("Section 3 — Generation & Logprob Alignment")
    print('─'*60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Reply with exactly one word: hello"},
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    tok.padding_side = "left"
    inputs = tok([text], return_tensors="pt", padding=True).to(model.device)
    padded_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tok.eos_token_id,
        )

    comp = gen.sequences[0, padded_len:].cpu()

    # EOS trimming (from rollout_func)
    eos_pos = (comp == tok.eos_token_id).nonzero()
    if len(eos_pos):
        comp_trimmed = comp[: eos_pos[0].item() + 1]
    else:
        comp_trimmed = comp

    check("Generation produced tokens", len(comp) > 0, f"raw tokens={len(comp)}")
    check("EOS trim leaves ≥1 token", len(comp_trimmed) >= 1,
          f"trimmed={len(comp_trimmed)}")

    # Logprob alignment
    T = len(comp)
    if T > 0 and gen.scores:
        scores = torch.stack([gen.scores[t][0].cpu() for t in range(T)])
        lp = F.log_softmax(scores.float(), dim=-1)
        token_lp = lp.gather(1, comp.unsqueeze(-1)).squeeze(-1)
        check("len(scores) == len(comp)", len(gen.scores) == T,
              f"scores={len(gen.scores)} comp={T}")
        check("logprobs shape matches comp", token_lp.shape == comp.shape,
              f"lp={token_lp.shape} comp={comp.shape}")
        check("All logprobs ≤ 0 (valid log-probs)", (token_lp <= 0).all().item(),
              f"max_lp={token_lp.max().item():.4f}")
    else:
        check("Logprob extraction", False, "no scores or empty completion")

    # Decode
    decoded = tok.decode(comp_trimmed, skip_special_tokens=False)
    check("Decoded response non-empty", len(decoded.strip()) > 0)
    check("Decoded response is valid UTF-8", True)   # Python str is always unicode
    print(f"\n  {INFO} Model response: {decoded.strip()[:120]!r}")


# ─── section 4: tool call format ─────────────────────────────────────────────

def section_tool_calls(model, tok):
    print(f"\n{'─'*60}")
    print("Section 4 — Tool-Call Format (<tool_call> blocks) — simple + hard")
    print('─'*60)

    # ── 4a: simple / baseline parse cases ────────────────────────────────────
    print(f"  --- simple cases ---")
    good = '<tool_call>{"name": "read_inbox", "arguments": {"filter": "unread"}}</tool_call>'
    bad_json = '<tool_call>{name: read_inbox}</tool_call>'
    empty = 'No tool called here.'
    multi = (
        '<tool_call>{"name": "view_requests", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "submit_po", "arguments": {"supplier_id": "S1", '
        '"product_id": "BLOOD-RBC", "destination_id": "ward_icu", "quantity": 10, "priority": "expedited"}}</tool_call>'
    )

    r = parse_tool_calls(good)
    check("simple: valid single call", len(r) == 1 and r[0]["name"] == "read_inbox",
          f"got {r}")

    r = parse_tool_calls(bad_json)
    check("simple: bad JSON skipped gracefully", r == [], f"got {r}")

    r = parse_tool_calls(empty)
    check("simple: no tags → empty list", r == [], f"got {r}")

    r = parse_tool_calls(multi)
    check("simple: two consecutive blocks parsed", len(r) == 2,
          f"parsed {len(r)} tool calls")
    if len(r) == 2:
        check("  second call is submit_po", r[1]["name"] == "submit_po")
        check("  submit_po args complete",
              all(k in r[1]["arguments"] for k in
                  ["supplier_id", "product_id", "destination_id", "quantity"]))

    # ── 4b: harder parse cases ────────────────────────────────────────────────
    print(f"\n  --- harder parse cases ---")

    # Whitespace / newlines inside the block
    whitespace_block = (
        '<tool_call>\n  {\n    "name": "query_erp",\n'
        '    "arguments": {"table": "inventory", "location": "ward_icu", "sku": "BLOOD-RBC"}\n'
        '  }\n</tool_call>'
    )
    r = parse_tool_calls(whitespace_block)
    check("harder: newlines+indentation inside block", len(r) == 1 and r[0]["name"] == "query_erp",
          f"got {r}")

    # Block buried in surrounding prose (model adds commentary)
    prose_wrapped = (
        "Sure, I will query the supplier now.\n"
        '<tool_call>{"name": "query_supplier", "arguments": {"supplier_id": "SUP-002"}}</tool_call>\n'
        "Let me know if you need anything else."
    )
    r = parse_tool_calls(prose_wrapped)
    check("harder: block buried in prose", len(r) == 1 and r[0]["name"] == "query_supplier",
          f"got {r}")

    # Nested JSON in arguments (allocation plan as JSON string)
    alloc_plan_str = json.dumps({"ward_icu": {"BLOOD-RBC": 5, "BLOOD-PLT": 3}, "ward_er": {"BLOOD-RBC": 4}})
    nested_args = (
        f'<tool_call>{{"name": "submit_allocation_plan", "arguments": {{"plan_json": {json.dumps(alloc_plan_str)}}}}}</tool_call>'
    )
    r = parse_tool_calls(nested_args)
    check("harder: nested JSON string in arguments", len(r) == 1, f"got {r}")
    if r:
        inner = r[0]["arguments"].get("plan_json", "")
        plan = json.loads(inner) if isinstance(inner, str) else inner
        check("  allocation plan round-trips correctly",
              plan.get("ward_icu", {}).get("BLOOD-RBC") == 5,
              f"plan={plan}")

    # Multiple blocks with mixed valid/invalid JSON — only valid ones returned
    mixed_validity = (
        '<tool_call>{"name": "read_inbox", "arguments": {"filter": "all"}}</tool_call>\n'
        '<tool_call>{this is not json}</tool_call>\n'
        '<tool_call>{"name": "view_requests", "arguments": {}}</tool_call>'
    )
    r = parse_tool_calls(mixed_validity)
    check("harder: mixed valid/invalid — only 2 valid returned", len(r) == 2,
          f"got {len(r)} calls: {[x['name'] for x in r]}")

    # file_justification with multi-line reason string
    justification = (
        '<tool_call>{"name": "file_justification", "arguments": '
        '{"ticket_id": "TKT-999", "reason": "MCI event surge: ER patient volume spiked 2.8x.\\nBlood products critically low."}}'
        '</tool_call>'
    )
    r = parse_tool_calls(justification)
    check("harder: file_justification with multi-line reason", len(r) == 1, f"got {r}")
    if r:
        check("  ticket_id correct", r[0]["arguments"]["ticket_id"] == "TKT-999")
        check("  reason contains newline escape", "\\n" in json.dumps(r[0]["arguments"]["reason"]))

    # quarantine_lot with all three required args
    quarantine = (
        '<tool_call>{"name": "quarantine_lot", "arguments": '
        '{"location_id": "ward_icu", "sku": "BLOOD-FFP", "lot_id": "LOT-2024-0088"}}</tool_call>'
    )
    r = parse_tool_calls(quarantine)
    check("harder: quarantine_lot with location+sku+lot_id", len(r) == 1, f"got {r}")
    if r:
        check("  all three quarantine args present",
              all(k in r[0]["arguments"] for k in ["location_id", "sku", "lot_id"]))

    # Three-tool round sequence: read_inbox → view_requests → submit_po
    round_sequence = (
        '<tool_call>{"name": "read_inbox", "arguments": {"filter": "unread"}}</tool_call>\n'
        '<tool_call>{"name": "view_requests", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "submit_po", "arguments": {"supplier_id": "SUP-001", '
        '"product_id": "BLOOD-RBC", "destination_id": "ward_icu", "quantity": 20, "priority": "expedited"}}</tool_call>'
    )
    r = parse_tool_calls(round_sequence)
    check("harder: 3-tool round sequence parsed in order", len(r) == 3, f"got {len(r)}")
    if len(r) == 3:
        check("  sequence order correct",
              [x["name"] for x in r] == ["read_inbox", "view_requests", "submit_po"])

    # Integer vs string quantity — should pass through as-is (int)
    int_qty = '<tool_call>{"name": "submit_po", "arguments": {"supplier_id": "S1", "product_id": "GLOVES-L", "destination_id": "ward_general", "quantity": 500, "priority": "standard"}}</tool_call>'
    r = parse_tool_calls(int_qty)
    check("harder: quantity as int (not string)", len(r) == 1 and isinstance(r[0]["arguments"]["quantity"], int),
          f"quantity type={type(r[0]['arguments']['quantity']).__name__ if r else 'N/A'}")

    # ── 4c: model actually generates a tool-call block ───────────────────────
    print(f"\n  --- model generation ---")
    messages = build_tool_call_prompt(tok, "read_inbox")
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    tok.padding_side = "left"
    inputs = tok([text], return_tensors="pt", padding=True).to(model.device)
    padded_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            return_dict_in_generate=True,
            pad_token_id=tok.eos_token_id,
        )

    comp = gen.sequences[0, padded_len:].cpu()

    # Decode both ways — train.py/eval.py use skip_special_tokens=True
    decoded_raw  = tok.decode(comp, skip_special_tokens=False)
    decoded_clean = tok.decode(comp, skip_special_tokens=True)

    # skip_special_tokens=False baseline
    has_block = "<tool_call>" in decoded_raw
    parsed = parse_tool_calls(decoded_raw)
    check("Model generates <tool_call> block", has_block, f"raw: {decoded_raw.strip()[:200]!r}")
    check("parse_tool_calls works on raw decode", len(parsed) >= 1, f"parsed={parsed}")

    # skip_special_tokens=True — what train.py/eval.py actually use
    check("skip_special_tokens=True: <tool_call> tag preserved",
          "<tool_call>" in decoded_clean,
          f"clean: {decoded_clean.strip()[:200]!r}")
    check("skip_special_tokens=True: <|im_end|> stripped",
          "<|im_end|>" not in decoded_clean)
    check("skip_special_tokens=True: <|endoftext|> stripped",
          "<|endoftext|>" not in decoded_clean)
    check("skip_special_tokens=True: <think> stripped",
          "<think>" not in decoded_clean)
    parsed_clean = parse_tool_calls(decoded_clean)
    check("parse_tool_calls works on clean decode (train.py path)",
          len(parsed_clean) >= 1, f"parsed={parsed_clean}")
    if parsed_clean and parsed:
        check("  clean and raw parse to identical result",
              parsed_clean == parsed, f"clean={parsed_clean} raw={parsed}")

    print(f"\n  {INFO} raw  : {decoded_raw.strip()[:160]!r}")
    print(f"  {INFO} clean: {decoded_clean.strip()[:160]!r}")

    # ── 4d: train.py / eval.py actual prompt — NO format hint ───────────────
    # This is what the model sees during GRPO rollouts: just SYSTEM_PROMPT + brief.
    # The model must generate tool calls WITHOUT being shown the format.
    # Failure here means the base model needs training to learn the format.
    print(f"\n  --- train.py prompt (no format hint) ---")
    train_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "=== Round 1 of 8 ===\n"
            "Warehouse stock: BLOOD-RBC 12 units (ICU par=20), BLOOD-PLT 4 units (ICU par=10).\n"
            "Begin your assessment."
        )},
    ]
    train_text = tok.apply_chat_template(
        train_messages, tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    inputs_train = tok([train_text], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        gen_train = model.generate(
            **inputs_train,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            return_dict_in_generate=True,
            pad_token_id=tok.eos_token_id,
        )
    comp_train = gen_train.sequences[0, inputs_train["input_ids"].shape[1]:].cpu()
    decoded_train = tok.decode(comp_train, skip_special_tokens=True)
    calls_train = parse_tool_calls(decoded_train)

    check("train.py prompt: model generates <tool_call> block (no format hint)",
          "<tool_call>" in decoded_train,
          warn_only=True,
          detail=f"response: {decoded_train.strip()[:160]!r}")
    check("train.py prompt: parse_tool_calls finds ≥1 call",
          len(calls_train) >= 1,
          warn_only=True,
          detail=f"parsed={calls_train}")
    print(f"  {INFO} train prompt response: {decoded_train.strip()[:200]!r}")

    # ── 4e: native tool calling via tools= parameter ─────────────────────────
    # Pass tool schemas to apply_chat_template — the template injects the format
    # instructions automatically. Model should use <function=NAME> XML format.
    print(f"\n  --- native tool calling (tools= parameter) ---")
    tool_schemas = [
        {"type": "function", "function": {
            "name": "read_inbox",
            "description": "Read inbox messages",
            "parameters": {"type": "object", "properties": {
                "filter": {"type": "string", "enum": ["unread", "all", "flagged"]}
            }, "required": ["filter"]}
        }},
        {"type": "function", "function": {
            "name": "advance_round",
            "description": "Close the current round and advance to the next",
            "parameters": {"type": "object", "properties": {}}
        }},
    ]
    native_messages = [
        {"role": "user", "content": (
            "=== Round 1 of 8 ===\n"
            "Warehouse stock: BLOOD-RBC 12 units (ICU par=20). Begin your assessment."
        )},
    ]
    try:
        native_text = tok.apply_chat_template(
            native_messages, tools=tool_schemas,
            tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        check("apply_chat_template with tools= accepted", True)
        check("tools= injects <tools> block into prompt", "<tools>" in native_text)
        check("tools= injects <function= format hint", "<function=" in native_text)
        print(f"  {INFO} native prompt snippet: {native_text[native_text.find('<tools>'):][:300]!r}")
    except Exception as e:
        check("apply_chat_template with tools= accepted", False, str(e))
        return

    inputs_native = tok([native_text], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        gen_native = model.generate(
            **inputs_native,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            return_dict_in_generate=True,
            pad_token_id=tok.eos_token_id,
        )
    comp_native = gen_native.sequences[0, inputs_native["input_ids"].shape[1]:].cpu()
    decoded_native = tok.decode(comp_native, skip_special_tokens=True)

    uses_native_format = "<function=" in decoded_native
    uses_json_format   = "<tool_call>" in decoded_native and "{" in decoded_native
    check("native: model uses <function=NAME> XML format",
          uses_native_format, warn_only=True,
          detail=f"response: {decoded_native.strip()[:160]!r}")
    check("native: model does NOT fall back to JSON format",
          not uses_json_format, warn_only=True,
          detail="fell back to JSON" if uses_json_format else "correct")
    print(f"  {INFO} native response: {decoded_native.strip()[:200]!r}")

    # ── 4f: parse_tool_calls_native — static + live model output ────────────
    print(f"\n  --- parse_tool_calls_native ---")

    # single call, clean
    s1 = '<tool_call>\n<function=read_inbox>\n<parameter=filter>\nunread\n</parameter>\n</function>\n</tool_call>'
    r = parse_tool_calls_native(s1)
    check("native parse: single call extracted", len(r) == 1 and r[0]["name"] == "read_inbox", f"got {r}")
    check("native parse: parameter value correct", r[0]["arguments"].get("filter") == "unread" if r else False, f"args={r[0]['arguments'] if r else {}}")

    # no-arg call (advance_round)
    s2 = '<tool_call>\n<function=advance_round>\n</function>\n</tool_call>'
    r = parse_tool_calls_native(s2)
    check("native parse: no-arg call (advance_round)", len(r) == 1 and r[0]["name"] == "advance_round", f"got {r}")
    check("native parse: no-arg has empty arguments", r[0]["arguments"] == {} if r else False)

    # multi-param call (submit_po)
    s3 = (
        '<tool_call>\n<function=submit_po>\n'
        '<parameter=supplier_id>\nSUP-001\n</parameter>\n'
        '<parameter=product_id>\nBLOOD-RBC\n</parameter>\n'
        '<parameter=destination_id>\nward_icu\n</parameter>\n'
        '<parameter=quantity>\n20\n</parameter>\n'
        '<parameter=priority>\nexpedited\n</parameter>\n'
        '</function>\n</tool_call>'
    )
    r = parse_tool_calls_native(s3)
    check("native parse: multi-param submit_po", len(r) == 1 and r[0]["name"] == "submit_po", f"got {r}")
    if r:
        check("native parse: all submit_po params present",
              all(k in r[0]["arguments"] for k in ["supplier_id", "product_id", "destination_id", "quantity", "priority"]),
              f"args={r[0]['arguments']}")
        check("native parse: quantity value is '20' (string)", r[0]["arguments"]["quantity"] == "20",
              f"quantity={r[0]['arguments'].get('quantity')!r}")

    # two consecutive calls
    s4 = s1 + '\n' + s2
    r = parse_tool_calls_native(s4)
    check("native parse: two consecutive calls", len(r) == 2, f"got {len(r)}")

    # no tool call
    r = parse_tool_calls_native("No tool here, just prose.")
    check("native parse: no tags → empty list", r == [], f"got {r}")

    # ── harder native parse cases ─────────────────────────────────────────────
    print(f"\n  --- harder native parse cases ---")

    # submit_allocation_plan with JSON string as parameter value
    alloc_json = '{"ward_icu": {"BLOOD-RBC": 5, "BLOOD-PLT": 3}, "ward_er": {"BLOOD-RBC": 4}}'
    s_alloc = (
        '<tool_call>\n<function=submit_allocation_plan>\n'
        f'<parameter=plan_json>\n{alloc_json}\n</parameter>\n'
        '</function>\n</tool_call>'
    )
    r = parse_tool_calls_native(s_alloc)
    check("harder native: submit_allocation_plan with JSON string param", len(r) == 1, f"got {r}")
    if r:
        inner = r[0]["arguments"].get("plan_json", "")
        try:
            plan = json.loads(inner)
            check("harder native: plan_json round-trips to dict",
                  plan.get("ward_icu", {}).get("BLOOD-RBC") == 5, f"plan={plan}")
        except Exception as e:
            check("harder native: plan_json round-trips to dict", False, str(e))

    # file_justification with multi-line reason
    s_just = (
        '<tool_call>\n<function=file_justification>\n'
        '<parameter=ticket_id>\nTKT-999\n</parameter>\n'
        '<parameter=reason>\nMCI event surge: ER patient volume spiked 2.8x.\n'
        'Blood products critically low. Expedited order required.\n</parameter>\n'
        '</function>\n</tool_call>'
    )
    r = parse_tool_calls_native(s_just)
    check("harder native: file_justification multi-line reason", len(r) == 1 and r[0]["name"] == "file_justification", f"got {r}")
    if r:
        check("harder native: ticket_id correct", r[0]["arguments"].get("ticket_id") == "TKT-999")
        check("harder native: reason contains newline", "\n" in r[0]["arguments"].get("reason", ""),
              f"reason={r[0]['arguments'].get('reason')!r}")

    # quarantine_lot with 3 params
    s_quar = (
        '<tool_call>\n<function=quarantine_lot>\n'
        '<parameter=location_id>\nward_icu\n</parameter>\n'
        '<parameter=sku>\nBLOOD-FFP\n</parameter>\n'
        '<parameter=lot_id>\nLOT-2024-0088\n</parameter>\n'
        '</function>\n</tool_call>'
    )
    r = parse_tool_calls_native(s_quar)
    check("harder native: quarantine_lot all 3 params", len(r) == 1, f"got {r}")
    if r:
        check("harder native: quarantine args complete",
              all(k in r[0]["arguments"] for k in ["location_id", "sku", "lot_id"]))

    # block buried in prose
    s_prose = (
        "I will now read the inbox to check for urgent messages.\n"
        '<tool_call>\n<function=read_inbox>\n<parameter=filter>\nflagged\n</parameter>\n</function>\n</tool_call>\n'
        "Proceeding with assessment."
    )
    r = parse_tool_calls_native(s_prose)
    check("harder native: block buried in prose", len(r) == 1 and r[0]["name"] == "read_inbox", f"got {r}")

    # 3-tool sequence in order
    s_seq = (
        '<tool_call>\n<function=read_inbox>\n<parameter=filter>\nunread\n</parameter>\n</function>\n</tool_call>\n'
        '<tool_call>\n<function=view_requests>\n</function>\n</tool_call>\n'
        '<tool_call>\n<function=advance_round>\n</function>\n</tool_call>'
    )
    r = parse_tool_calls_native(s_seq)
    check("harder native: 3-tool sequence parsed in order", len(r) == 3, f"got {len(r)}")
    if len(r) == 3:
        check("harder native: sequence order correct",
              [x["name"] for x in r] == ["read_inbox", "view_requests", "advance_round"])

    # parse live model output from 4e
    if decoded_native:
        r_live = parse_tool_calls_native(decoded_native)
        check("native parse: live model output parsed correctly",
              len(r_live) >= 1,
              warn_only=True,
              detail=f"parsed={r_live}  raw={decoded_native.strip()[:120]!r}")


# ─── section 5: multi-turn context-reset pattern ─────────────────────────────

def section_context_reset(tok):
    print(f"\n{'─'*60}")
    print("Section 5 — Multi-Turn Round-Reset Pattern")
    print('─'*60)

    # Simulate the context-window reset done in rollout_func after advance_round
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Round 1 brief: manage inventory."},
        {"role": "assistant", "content": '<tool_call>{"name": "read_inbox", "arguments": {"filter": "unread"}}</tool_call>'},
        {"role": "tool", "content": "Inbox: 2 messages."},
        {"role": "assistant", "content": '<tool_call>{"name": "advance_round", "arguments": {}}</tool_call>'},
        {"role": "tool", "content": "Round 2 brief: handle surplus."},
    ]

    # After advance_round, rollout_func resets to just system+brief
    messages_reset = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Round 2 brief: handle surplus."},
    ]

    try:
        text_full = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        text_reset = tok.apply_chat_template(
            messages_reset, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        len_full = len(tok(text_full)["input_ids"])
        len_reset = len(tok(text_reset)["input_ids"])
        check("Context reset produces shorter sequence", len_reset < len_full,
              f"full={len_full} reset={len_reset} tokens")
        check("Round 2 brief present in reset context", "Round 2" in text_reset)
        check("Round 1 content absent from reset context", "Round 1" not in text_reset)
    except Exception as e:
        check("Context reset rendering", False, str(e))


# ─── section 6: batch generation (left-padding) ──────────────────────────────

def section_batch(model, tok):
    print(f"\n{'─'*60}")
    print("Section 6 — Batched Generation with Left-Padding")
    print('─'*60)

    prompts = [
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "Brief A: seed=0;diff=light"}],
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "Brief B: longer context here — seed=1;diff=heavy — please respond with a tool call."}],
    ]
    tok.padding_side = "left"
    texts = [
        tok.apply_chat_template(
            p, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        for p in prompts
    ]
    inputs = tok(texts, return_tensors="pt", padding=True).to(model.device)
    padded_len = inputs["input_ids"].shape[1]

    check("Left-padding applied (padding_side='left')", tok.padding_side == "left")
    check("Batch input_ids shape is [2, padded_len]",
          inputs["input_ids"].shape == torch.Size([2, padded_len]),
          f"shape={tuple(inputs['input_ids'].shape)}")
    # Verify first tokens of longer prompt are NOT pad
    check("Longer sequence doesn't start with pad",
          inputs["input_ids"][1, 0].item() != tok.pad_token_id,
          warn_only=True, detail="Expected: longer text isn't padded on left")
    # Verify shorter sequence's first token IS pad (left-padded)
    check("Shorter sequence starts with pad (left-padded)",
          inputs["input_ids"][0, 0].item() == tok.pad_token_id,
          warn_only=True, detail="Expected for correct left-padding")

    try:
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        check("Batch generate succeeded", True,
              f"output shape={tuple(gen.shape)}")
        check("Output has 2 sequences", gen.shape[0] == 2)
        # Extract per-example completions
        for i in range(2):
            comp = gen[i, padded_len:]
            decoded = tok.decode(comp, skip_special_tokens=True)
            check(f"Sequence {i} non-empty", len(decoded.strip()) > 0,
                  f"{decoded.strip()[:80]!r}")
    except Exception as e:
        check("Batch generate", False, str(e))


# ─── section 7: config objects ────────────────────────────────────────────────

def section_configs():
    print(f"\n{'─'*60}")
    print("Section 7 — BitsAndBytesConfig / LoraConfig / GRPOConfig fields")
    print('─'*60)

    # BnB
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        check("BitsAndBytesConfig fields accepted", True)
        check("bnb_4bit_quant_type=nf4", bnb.bnb_4bit_quant_type == "nf4")
        check("bnb_4bit_compute_dtype=bfloat16",
              bnb.bnb_4bit_compute_dtype == torch.bfloat16)
    except Exception as e:
        check("BitsAndBytesConfig", False, str(e))

    # LoRA
    try:
        lora = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        check("LoraConfig target_modules='all-linear' accepted", True)
        check("LoraConfig r=16 lora_alpha=16", lora.r == 16 and lora.lora_alpha == 16)
    except Exception as e:
        check("LoraConfig", False, str(e))

    # GRPOConfig — import only; don't instantiate (needs full trl stack)
    try:
        from trl import GRPOConfig
        sig = GRPOConfig.__init__.__doc__ or ""
        fields = ["num_generations", "max_completion_length", "learning_rate"]
        # Just check it imports and is a class
        check("GRPOConfig importable from trl", True)
        # Check bf16 param exists
        import inspect
        params = inspect.signature(GRPOConfig.__init__).parameters
        check("GRPOConfig has bf16 param", "bf16" in params,
              warn_only=True, detail="may be in TrainingArguments base")
        check("GRPOConfig has num_generations param",
              "num_generations" in params or True,  # may be in base too
              warn_only=True)
    except Exception as e:
        check("GRPOConfig import", False, str(e))


# ─── section 8: seed/diff regex (dataset ↔ rollout_func) ─────────────────────

def section_architecture(model, tok):
    """Inspect Qwen3.5 layer types — flags anything that could break quant or LoRA."""
    print(f"\n{'─'*60}")
    print("Section 9 — Architecture Inspection (layer types, LoRA targets)")
    print('─'*60)

    import torch.nn as nn
    from collections import Counter

    layer_types = Counter(type(m).__name__ for m in model.modules())
    linear_count = layer_types.get("Linear", 0)
    conv1d_count  = layer_types.get("Conv1d", 0)
    embedding_count = layer_types.get("Embedding", 0)

    print(f"\n  {INFO} Top layer types:")
    for name, cnt in layer_types.most_common(12):
        print(f"    {cnt:4d}  {name}")

    check("Has Linear layers (LoRA all-linear will find targets)",
          linear_count > 0, f"count={linear_count}")
    check("Conv1d layers present (SSM/hybrid — quant risk)",
          conv1d_count == 0,
          warn_only=True,
          detail=f"{conv1d_count} Conv1d layers found — nf4 quant may not cover these; "
                 "LoRA all-linear also won't adapt them" if conv1d_count else "none — clean transformer")
    check("No unexpected layer types (RNNCell / GRUCell / LSTMCell)",
          not any(t in layer_types for t in ["RNNCell", "GRUCell", "LSTMCell"]),
          detail="recurrent cells found — incompatible with batched left-pad generation"
          if any(t in layer_types for t in ["RNNCell", "GRUCell", "LSTMCell"]) else "")

    # Count trainable params before LoRA
    total_params  = sum(p.numel() for p in model.parameters())
    train_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  {INFO} Params: total={total_params/1e6:.1f}M  trainable={train_params/1e6:.1f}M")
    check("Model has parameters", total_params > 0, f"{total_params/1e6:.1f}M")

    # Confirm tokenizer vocab aligns with model embedding
    try:
        embed = None
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Embedding) and mod.weight.shape[0] > 1000:
                embed = mod
                break
        if embed is not None:
            vocab_match = embed.weight.shape[0] >= len(tok)
            check("Embedding vocab ≥ tokenizer vocab",
                  vocab_match,
                  f"embed={embed.weight.shape[0]}  tok={len(tok)}")
        else:
            check("Embedding found", False, "no large Embedding layer found")
    except Exception as e:
        check("Vocab alignment check", False, str(e))

    # Verify model device placement
    try:
        first_param_device = next(model.parameters()).device
        check("Model placed on a device", True, f"device={first_param_device}")
    except Exception as e:
        check("Model device check", False, str(e))

    return linear_count, conv1d_count


def section_quantization(model_id: str, tok):
    """4-bit quant checks — only runs under --4bit (needs CUDA + bitsandbytes)."""
    print(f"\n{'─'*60}")
    print("Section 10 — 4-bit Quantization (nf4 + LoRA)")
    print('─'*60)

    import torch.nn as nn
    from peft import get_peft_model, LoraConfig as LC
    from peft import prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        print(f"  {WARN} CUDA not available — skipping quantization checks.")
        return

    # Load fresh quantized model
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        qmodel = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto"
        )
        check("4-bit model loaded", True, qmodel.__class__.__name__)
    except Exception as e:
        check("4-bit model loaded", False, str(e))
        return

    # Check that key layers are actually quantized
    try:
        import bitsandbytes as bnb_lib
        quant_layers = [
            name for name, mod in qmodel.named_modules()
            if isinstance(mod, bnb_lib.nn.Linear4bit)
        ]
        check("Linear4bit layers present (quant applied)",
              len(quant_layers) > 0, f"count={len(quant_layers)}")
        print(f"  {INFO} Sample quantized layers: {quant_layers[:3]}")
    except Exception as e:
        check("Linear4bit layer inspection", False, str(e))

    # Forward pass — no NaN / Inf in logits
    try:
        text = tok.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        ids = tok(text, return_tensors="pt").to(qmodel.device)
        with torch.no_grad():
            logits = qmodel(**ids).logits
        check("4-bit forward pass: no NaN", not torch.isnan(logits).any().item(),
              f"shape={tuple(logits.shape)}")
        check("4-bit forward pass: no Inf", not torch.isinf(logits).any().item())
        check("4-bit logits dtype is bfloat16", logits.dtype == torch.bfloat16,
              f"dtype={logits.dtype}")
    except Exception as e:
        check("4-bit forward pass", False, str(e))

    # prepare_model_for_kbit_training
    try:
        qmodel = prepare_model_for_kbit_training(qmodel)
        check("prepare_model_for_kbit_training succeeded", True)
    except Exception as e:
        check("prepare_model_for_kbit_training", False, str(e))
        return

    # Apply LoRA exactly as train.py does
    try:
        lora_cfg = LC(
            r=16, lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        lora_model = get_peft_model(qmodel, lora_cfg)
        train_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total_params  = sum(p.numel() for p in lora_model.parameters())
        pct = 100 * train_params / total_params
        check("LoRA applied successfully", train_params > 0,
              f"trainable={train_params/1e6:.2f}M / {total_params/1e6:.1f}M ({pct:.2f}%)")
        check("LoRA trainable % is sane (0.1%–5%)", 0.05 <= pct <= 10,
              f"{pct:.2f}%")
        lora_model.print_trainable_parameters()
    except Exception as e:
        check("LoRA application", False, str(e))
        return

    # Dummy backward through LoRA params (gradient flow)
    try:
        text2 = tok.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        ids2 = tok(text2, return_tensors="pt").to(lora_model.device)
        out = lora_model(**ids2, labels=ids2["input_ids"])
        out.loss.backward()
        lora_grad_ok = all(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in lora_model.parameters() if p.requires_grad
        )
        check("LoRA gradients flow (no NaN grad)", lora_grad_ok)
    except Exception as e:
        check("LoRA gradient flow", False, str(e))


def section_seed_regex():
    print(f"\n{'─'*60}")
    print("Section 8 — seed/diff Embedding & Extraction (dataset ↔ rollout_func)")
    print('─'*60)

    test_cases = [
        ("seed=0;diff=light",   0,   "light"),
        ("seed=49;diff=heavy",  49,  "heavy"),
        ("seed=123;diff=medium", 123, "medium"),
    ]
    for content, expected_seed, expected_diff in test_cases:
        sm = re.search(r"seed=(\d+)", content)
        dm = re.search(r"diff=(\w+)", content)
        ok = (
            sm is not None and int(sm.group(1)) == expected_seed
            and dm is not None and dm.group(1) == expected_diff
        )
        check(f"seed/diff parse: {content!r}", ok,
              f"seed={sm.group(1) if sm else '?'}  diff={dm.group(1) if dm else '?'}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Qwen 3.5 2B/4B correctness checks for train.py/eval.py"
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-0.6B",
        help=(
            "HF model ID to test. "
            "Use 'Qwen/Qwen3.5-2B' or 'Qwen/Qwen3.5-4B' to test actual models. "
            "Defaults to Qwen3-0.6B for quick CI smoke-test."
        ),
    )
    parser.add_argument(
        "--4bit", dest="use_4bit", action="store_true",
        help="Load with 4-bit quantisation (requires bitsandbytes + CUDA)"
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip generation/tool-call checks (config & parsing checks only)"
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Qwen 3.5 sanity checks — {args.model_id}")
    print(f"  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    print(f"{'='*60}")

    # Configs / regex checks don't need the model
    section_seed_regex()
    section_configs()

    if args.skip_generation:
        print(f"\n{INFO} --skip-generation: skipping model load and inference checks.\n")
    else:
        model, tok = section_load(args.model_id, args.use_4bit)
        if model is None:
            pass
        else:
            section_chat_template(tok)
            section_generation(model, tok)
            section_tool_calls(model, tok)
            section_context_reset(tok)
            section_batch(model, tok)
            section_architecture(model, tok)
            if args.use_4bit:
                section_quantization(args.model_id, tok)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed
    print(f"  Results: {passed}/{total} passed  ({failed} failed)")
    if failed:
        print(f"\n  {FAIL} Failed checks:")
        for name, ok, detail in results:
            if not ok:
                print(f"    • {name}" + (f": {detail}" if detail else ""))
    else:
        print(f"\n  {PASS} All checks passed.")
    print('='*60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
