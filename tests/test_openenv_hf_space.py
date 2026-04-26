"""
Smoke test for the MedChain Finals HF Space environment.

Connects to https://nik-55-medchain-openenv-final-round.hf.space, resets,
discovers tools, then exercises a minimal multi-actor coordination episode:
  read_inbox → view_requests → get_round_briefing → submit_allocation_plan → advance_round

Usage:
    python tests/test_openenv_hf_space.py

No API key required — the environment server is public.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from medchain_env import CallToolAction, MedchainEnv

HF_SPACE_URL = "https://nik-55-medchain-openenv-final-round.hf.space"

# Minimal allocation plan: {ward_id: {sku: qty}} — the format the server expects
_SAMPLE_PLAN = json.dumps(
    {
        "ward_icu": {"BLOOD-RBC": 13, "BLOOD-PLT": 6, "BLOOD-FFP": 8},
        "ward_er": {"BLOOD-RBC": 6, "BLOOD-PLT": 3},
        "ward_general": {"GLOVE-001": 20, "IV-SAL-500": 15},
    }
)

_SAMPLE_RATIONALE = json.dumps(
    {
        "ward_icu": "ICU request within historical baseline; no padding flag.",
        "ward_er": "ER request elevated but within surge tolerance.",
        "ward_general": "Standard restock; no anomalies detected.",
    }
)


def _sep(label: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print("─" * 60)


async def main() -> None:
    print(f"Connecting to: {HF_SPACE_URL}")
    env = MedchainEnv(base_url=HF_SPACE_URL)
    await env.connect()

    try:
        # ── 1. Reset ──────────────────────────────────────────────────────
        _sep("RESET")
        reset_result = await env.reset(task="multi_actor_coordination")
        obs = reset_result.observation
        dashboard = obs.metadata.get("dashboard", "<no dashboard>")
        print(f"done={obs.done}  reward={obs.reward}")
        print("\n[Dashboard]\n")
        print(dashboard[:1500])
        if len(dashboard) > 1500:
            print(f"... (truncated, total {len(dashboard)} chars)")

        # ── 2. List tools ─────────────────────────────────────────────────
        _sep("LIST TOOLS")
        tools = await env.list_tools(use_cache=False)
        print(f"Discovered {len(tools)} tools:\n")
        for t in tools:
            req = list((t.input_schema or {}).get("required", []))
            print(f"  • {t.name:<40s}  required={req}")

        assert len(tools) > 0, "Expected at least one tool"

        tool_names = {t.name for t in tools}

        # ── 3. read_inbox ─────────────────────────────────────────────────
        _sep("STEP: read_inbox")
        r = await env.step(CallToolAction(tool_name="read_inbox", arguments={"filter": "unread"}))
        result = r.observation.metadata.get("tool_result", "")
        print(result[:800])
        print(f"\nreward={r.reward}  done={r.observation.done}")

        # ── 4. view_requests ──────────────────────────────────────────────
        _sep("STEP: view_requests")
        r = await env.step(CallToolAction(tool_name="view_requests", arguments={}))
        result = r.observation.metadata.get("tool_result", "")
        print(result[:800])
        print(f"\nreward={r.reward}  done={r.observation.done}")

        # ── 5. get_round_briefing (if available) ──────────────────────────
        if "get_round_briefing" in tool_names:
            _sep("STEP: get_round_briefing")
            r = await env.step(
                CallToolAction(tool_name="get_round_briefing", arguments={})
            )
            result = r.observation.metadata.get("tool_result", "")
            print(result[:800])
            print(f"\nreward={r.reward}  done={r.observation.done}")

        # ── 6. query_ward_history (if available) ──────────────────────────
        if "query_ward_history" in tool_names:
            _sep("STEP: query_ward_history(ward_id=ward_icu)")
            r = await env.step(
                CallToolAction(
                    tool_name="query_ward_history",
                    arguments={"ward_id": "ward_icu"},
                )
            )
            result = r.observation.metadata.get("tool_result", "")
            print(result[:800])
            print(f"\nreward={r.reward}  done={r.observation.done}")

        # ── 7. submit_allocation_plan ─────────────────────────────────────
        _sep("STEP: submit_allocation_plan")
        alloc_args: dict = {"plan_json": _SAMPLE_PLAN}
        if "rationale_json" in {
            p
            for t in tools
            if t.name == "submit_allocation_plan"
            for p in (t.input_schema or {}).get("properties", {})
        }:
            alloc_args["rationale_json"] = _SAMPLE_RATIONALE

        r = await env.step(
            CallToolAction(tool_name="submit_allocation_plan", arguments=alloc_args)
        )
        result = r.observation.metadata.get("tool_result", "")
        print(result[:800])
        print(f"\nreward={r.reward}  done={r.observation.done}")

        # ── 8. advance_round ──────────────────────────────────────────────
        _sep("STEP: advance_round")
        r = await env.step(CallToolAction(tool_name="advance_round", arguments={}))
        result = r.observation.metadata.get("tool_result", "")
        print(result[:800])
        print(f"\nreward={r.reward}  done={r.observation.done}")

        # ── Summary ───────────────────────────────────────────────────────
        _sep("SUMMARY")
        print(f"All smoke actions completed successfully.")
        print(f"episode done={r.observation.done}  final_reward={r.reward}")

    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
