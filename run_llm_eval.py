"""
run_llm_eval.py — MedChain LLM evaluation driver
=================================================
Runs multi-episode, multi-round evaluations of ANY OpenAI-compatible model
against the MedChain OpenEnv server.
Use this to benchmark frontier models (GPT-4o, Claude Sonnet, Llama 3.3)
on the full 21-tool MedChain surface before or alongside RL training.

Environment variables:
    HF_TOKEN / API_KEY      Auth token (required)
    API_BASE_URL            API endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME / MODEL      Model to use  (default: openai/gpt-oss-20b:groq)
    BASE_URL                MedChain server URL (default: HF Space)
    SEEDS                   Comma-separated seeds, e.g. "0,1,2"  (default: 0,1,2)
    DIFFICULTIES            Comma-separated difficulties         (default: light,medium,heavy)
    ROLLOUTS                Rollouts per (seed, difficulty) pair (default: 1)
    SLEEP                   Seconds between API calls            (default: 0.5)
    LOG_LEVEL               INFO (default) | DEBUG

Stdout format — one line per event, in order:
    [START] episode=<n> seed=<s> difficulty=<d> env=medchain model=<model>
    [STEP]  step=<n> action=<action> reward=<r> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
    [SUMMARY] episodes=<n> completed=<n> avg_score=<s>
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

from openai import BadRequestError, OpenAI, RateLimitError

sys.path.insert(0, str(Path(__file__).parent))         # for server.* imports
sys.path.insert(0, str(Path(__file__).parent.parent))  # for medchain_env package import

from medchain_env import MedchainEnv, CallToolAction
from server.prompts import INFERENCE_SYSTEM_PROMPT

SYSTEM_PROMPT = INFERENCE_SYSTEM_PROMPT

# ── Configuration ──────────────────────────────────────────────────────────────

LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO").upper()
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME") or os.getenv("MODEL", "openai/gpt-oss-20b:groq")
BASE_URL     = os.getenv("BASE_URL", "https://nik-55-medchain-openenv-final-round.hf.space")

SEEDS        = [int(s.strip()) for s in os.getenv("SEEDS", "0,1,2").split(",") if s.strip()]
DIFFICULTIES = [d.strip() for d in os.getenv("DIFFICULTIES", "light,medium,heavy").split(",") if d.strip()]
ROLLOUTS     = int(os.getenv("ROLLOUTS", "1"))
SLEEP        = float(os.getenv("SLEEP", "0.5"))

MAX_STEPS             = 220    # 8 rounds × ~25 turns + buffer
MAX_TOKENS            = 8000
TEMPERATURE           = 0.1
MAX_CONSECUTIVE_ERRS  = 5

_429_WINDOW      = 60
_429_THRESHOLD   = 3
_429_BACKOFF     = 5
_429_MAX_BACKOFF = 30

BENCHMARK = "medchain"

# ── Logging ────────────────────────────────────────────────────────────────────

_log_fmt = logging.Formatter(
    "[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S"
)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_log_fmt)
_handlers: list = [_stream_handler]

if LOG_LEVEL == "DEBUG":
    os.makedirs("logs", exist_ok=True)
    _log_filename = datetime.now().strftime("logs/inference_%Y%m%d_%H%M%S.log")
    _file_handler = logging.FileHandler(_log_filename)
    _file_handler.setFormatter(_log_fmt)
    _handlers.append(_file_handler)
    print(f"[DEBUG] Logging to file: {_log_filename}", flush=True)

logging.basicConfig(level=logging.WARNING, handlers=_handlers)
log = logging.getLogger(__name__)
log.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# ── Tool discovery ─────────────────────────────────────────────────────────────

def _tools_to_chat_format(tools) -> list[dict]:
    """Convert MCP Tool objects to OpenAI function-calling format."""
    result = []
    for t in tools:
        schema = t.input_schema or {}
        result.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        })
    return result


async def _discover_tools() -> list[dict]:
    env = MedchainEnv(base_url=BASE_URL)
    await env.connect()
    mcp_tools = await env.list_tools()
    await env.close()
    return _tools_to_chat_format(mcp_tools)


TOOL_SCHEMAS: list[dict] = asyncio.run(_discover_tools())
log.info("Discovered %d tools from %s", len(TOOL_SCHEMAS), BASE_URL)

# ── Structured output helpers ──────────────────────────────────────────────────

def _log_start(episode: int, seed: int, difficulty: str) -> None:
    print(
        f"[START] episode={episode} seed={seed} difficulty={difficulty} "
        f"env={BENCHMARK} model={MODEL_NAME}",
        flush=True,
    )


def _log_step(step: int, action: str, reward: float, done: bool,
              error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )

# ── Episode runner ─────────────────────────────────────────────────────────────

async def run_episode(
    client: OpenAI,
    seed: int,
    difficulty: str,
    episode_num: int,
) -> Dict[str, Any]:
    """Run one full MedChain episode via the OpenEnv client and return summary metrics."""
    env = MedchainEnv(base_url=BASE_URL)
    await env.connect()

    reset_result = await env.reset(seed=seed, difficulty=difficulty, episode_id=str(uuid.uuid4()))
    brief = reset_result.observation.metadata.get("dashboard", "")

    messages: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": brief},
    ]

    _log_start(episode=episode_num, seed=seed, difficulty=difficulty)
    log.info("[ep%d] seed=%d diff=%s started", episode_num, seed, difficulty)

    step_count        = 0
    done              = False
    final_reward      = 0.0
    rewards: List[float] = []
    consecutive_errs  = 0
    rate_limit_times: List[float] = []
    backoff_count     = 0
    ep_start          = time.monotonic()

    try:
        while not done and step_count < MAX_STEPS:
            step_count += 1
            log.debug("[ep%d] step %d/%d  msgs=%d", episode_num, step_count, MAX_STEPS, len(messages))

            # ── LLM call ────────────────────────────────────────────────────
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="required",
                    parallel_tool_calls=True,
                    max_completion_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                consecutive_errs = 0
                backoff_count    = 0

            except RateLimitError as exc:
                now = time.monotonic()
                rate_limit_times.append(now)
                rate_limit_times[:] = [t for t in rate_limit_times if now - t <= _429_WINDOW]
                backoff_count += 1
                wait = min(_429_BACKOFF * (2 ** (backoff_count - 1)), _429_MAX_BACKOFF)
                log.warning("[ep%d] step %d — 429 (count=%d  wait=%.1fs): %s",
                            episode_num, step_count, len(rate_limit_times), wait, exc)
                await asyncio.sleep(wait)
                step_count -= 1
                continue

            except BadRequestError as exc:
                consecutive_errs += 1
                log.warning("[ep%d] step %d — BadRequest (%d/%d): %s",
                            episode_num, step_count, consecutive_errs, MAX_CONSECUTIVE_ERRS, exc)
                if consecutive_errs >= MAX_CONSECUTIVE_ERRS:
                    log.error("[ep%d] aborting — too many consecutive errors", episode_num)
                    break
                messages.append({
                    "role": "user",
                    "content": f"Your previous call was rejected: {exc}. Retry with a valid tool call.",
                })
                step_count -= 1
                continue

            # ── Parse tool calls from response ───────────────────────────────
            message    = response.choices[0].message
            step_error: Optional[str] = None

            if message.tool_calls:
                raw_calls = []
                for tc in message.tool_calls:
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, AttributeError):
                        tool_args = {}
                    raw_calls.append((tc.function.name, tool_args, tc.id))
            else:
                log.warning("[ep%d] step %d — no tool_call; defaulting to advance_round",
                            episode_num, step_count)
                raw_calls = [("advance_round", {}, "fallback")]

            # ── Dispatch each tool call via OpenEnv env.step() ───────────────
            executed: List[tuple] = []   # (tool_name, tool_args, tool_call_id, result_text)
            advance_result: Optional[str] = None

            for tool_name, tool_args, tool_call_id in raw_calls:
                log.debug("[ep%d] step %d — %s(%s)", episode_num, step_count, tool_name, tool_args)

                sr  = await env.step(CallToolAction(tool_name=tool_name, arguments=tool_args))
                obs = sr.observation
                result_text = obs.metadata.get("tool_result", "")

                if not result_text:
                    result_text = f"ERROR: {obs.metadata}"
                    step_error  = str(obs.metadata)
                    log.warning("[ep%d] tool error %s(%s): %s",
                                episode_num, tool_name, tool_args, obs.metadata)

                executed.append((tool_name, tool_args, tool_call_id, result_text))

                if tool_name == "advance_round":
                    if obs.done:
                        done         = True
                        final_reward = obs.reward if obs.reward is not None else 0.0
                    else:
                        advance_result = result_text
                    break   # advance_round always ends the turn

            # ── Step reward ──────────────────────────────────────────────────
            step_reward = final_reward if done else 0.0
            rewards.append(step_reward)

            for tool_name, tool_args, _, _ in executed:
                action_str = f"{tool_name}({json.dumps(tool_args, separators=(',', ':'))})"
                _log_step(step=step_count, action=action_str,
                          reward=step_reward if tool_name == "advance_round" else 0.0,
                          done=done, error=step_error)

            # ── Append assistant message + tool results to history ───────────
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id":   tool_call_id,
                        "type": "function",
                        "function": {
                            "name":      tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                    for tool_name, tool_args, tool_call_id, _ in executed
                ],
            })
            for tool_name, tool_args, tool_call_id, result_text in executed:
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call_id,
                    "content":      result_text[:2700] if result_text else "OK",
                })

            # ── Context reset after round advance ────────────────────────────
            if advance_result is not None and not done:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": advance_result},
                ]
                log.info("[ep%d] step %d — round advanced; context reset", episode_num, step_count)

            if not done:
                await asyncio.sleep(SLEEP)

    finally:
        await env.close()

    elapsed = time.monotonic() - ep_start
    log.info("[ep%d] seed=%d diff=%s — done=%s reward=%.4f steps=%d elapsed=%.1fs",
             episode_num, seed, difficulty, done, final_reward, step_count, elapsed)

    _log_end(success=done, steps=step_count, score=final_reward, rewards=rewards)

    return {
        "episode":    episode_num,
        "seed":       seed,
        "difficulty": difficulty,
        "reward":     final_reward,
        "steps":      step_count,
        "done":       done,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    if not API_KEY:
        raise SystemExit("Set HF_TOKEN or API_KEY before running.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    manifest: List[Tuple[int, str, int]] = []
    for seed in SEEDS:
        for diff in DIFFICULTIES:
            for _ in range(ROLLOUTS):
                manifest.append((seed, diff, len(manifest) + 1))

    total = len(manifest)
    log.info(
        "Starting %d episode(s)  seeds=%s  difficulties=%s  rollouts=%d  model=%s  server=%s",
        total, SEEDS, DIFFICULTIES, ROLLOUTS, MODEL_NAME, BASE_URL,
    )

    results: List[Dict[str, Any]] = []
    for seed, diff, ep_num in manifest:
        log.info("─── Episode %d / %d  seed=%d  diff=%s ───", ep_num, total, seed, diff)
        try:
            results.append(await run_episode(client, seed=seed, difficulty=diff, episode_num=ep_num))
        except Exception as exc:
            log.error("[ep%d] seed=%d diff=%s — unhandled: %s", ep_num, seed, diff, exc)

    if not results:
        return

    avg   = sum(r["reward"] for r in results) / len(results)
    done_ = sum(1 for r in results if r["done"])
    print(
        f"\n[SUMMARY] episodes={len(results)} completed={done_} avg_score={avg:.4f}",
        flush=True,
    )
    for diff in DIFFICULTIES:
        sub = [r for r in results if r["difficulty"] == diff]
        if sub:
            print(
                f"  {diff:8s}: n={len(sub)}  avg={sum(r['reward'] for r in sub)/len(sub):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
