"""
run_llm_eval.py — MedChain LLM evaluation driver
=================================================
Runs multi-episode, multi-round evaluations of ANY OpenAI-compatible model
against the MedchainSimulation directly (no HTTP server required).
Use this to benchmark frontier models (GPT-4o, Claude Sonnet, Llama 3.3)
on the full 21-tool MedChain surface before or alongside RL training.

Environment variables:
    HF_TOKEN / API_KEY      Auth token (required)
    API_BASE_URL            API endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME / MODEL      Model to use  (default: openai/gpt-oss-20b:groq)
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

sys.path.insert(0, str(Path(__file__).parent))

from server.simulation import MedchainSimulation
from server.tasks import make_task_config
from server.prompts import INFERENCE_SYSTEM_PROMPT, INFERENCE_TOOL_SCHEMAS
from server.grader import compute_reward_breakdown

SYSTEM_PROMPT = INFERENCE_SYSTEM_PROMPT
TOOL_SCHEMAS = INFERENCE_TOOL_SCHEMAS

# ── Configuration ──────────────────────────────────────────────────────────────

LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO").upper()
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME") or os.getenv("MODEL", "openai/gpt-oss-20b:groq")

SEEDS        = [int(s.strip()) for s in os.getenv("SEEDS", "0,1,2").split(",") if s.strip()]
DIFFICULTIES = [d.strip() for d in os.getenv("DIFFICULTIES", "light,medium,heavy").split(",") if d.strip()]
ROLLOUTS     = int(os.getenv("ROLLOUTS", "1"))
SLEEP        = float(os.getenv("SLEEP", "0.5"))

MAX_STEPS             = 220    # 8 rounds × ~25 turns + buffer (more tools, more parallel calls)
MAX_TOKENS            = 8000   # bigger tool surface; parallel tool_calls inflate completion size
TEMPERATURE           = 0.1    # matches reference inference.py
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


def _log_breakdown(breakdown: Dict[str, float]) -> None:
    """Emit per-component reward breakdown so judges can see signal location."""
    keys = [
        "network_sl", "critical_sl", "alloc_acc", "event_resp", "budget_eff",
        "waste_ctrl", "challenge_score", "approval_score",
        "tool_discovery", "briefing_eff", "justif_pen", "score",
    ]
    parts = " ".join(
        f"{k}={breakdown.get(k, 0.0):.3f}" for k in keys if k in breakdown
    )
    print(f"[BREAKDOWN] {parts}", flush=True)

# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(
    client: OpenAI,
    seed: int,
    difficulty: str,
    episode_num: int,
) -> Dict[str, Any]:
    """Run one full MedChain episode and return summary metrics."""
    sim   = MedchainSimulation(make_task_config(seed=seed, difficulty=difficulty))
    brief = sim.reset(seed=seed, episode_id=str(uuid.uuid4()))

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

    while not done and step_count < MAX_STEPS:
        step_count += 1
        log.debug("[ep%d] step %d/%d  msgs=%d", episode_num, step_count, MAX_STEPS, len(messages))

        # ── LLM call ────────────────────────────────────────────────────────
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
            time.sleep(wait)
            step_count -= 1   # retry doesn't consume a step
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

        # ── Parse tool calls from response (parallel tool calls supported) ────
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

        # ── Execute all tool calls in this turn ──────────────────────────────
        executed: List[tuple] = []   # (tool_name, tool_args, tool_call_id, result)
        for tool_name, tool_args, tool_call_id in raw_calls:
            log.debug("[ep%d] step %d — %s(%s)", episode_num, step_count, tool_name, tool_args)
            if tool_name == "submit_po" and tool_args.get("quantity", 1) <= 0:
                result     = "ERROR: quantity must be positive — PO skipped."
                step_error = "zero_qty_po"
                executed.append((tool_name, tool_args, tool_call_id, result))
                continue
            try:
                if hasattr(sim, tool_name):
                    result: str = getattr(sim, tool_name)(**tool_args)
                else:
                    result     = f"ERROR: Unknown tool '{tool_name}'"
                    step_error = f"unknown_tool:{tool_name}"
            except Exception as exc:
                result     = f"ERROR: {exc}"
                step_error = str(exc)
                log.warning("[ep%d] tool error %s(%s): %s",
                            episode_num, tool_name, tool_args, exc)
            executed.append((tool_name, tool_args, tool_call_id, result))

        # ── Determine step reward (advance_round sets sim._done) ─────────────
        step_reward = 0.0
        advance_result: Optional[str] = None
        for tool_name, tool_args, tool_call_id, result in executed:
            if tool_name == "advance_round" and sim._done:
                done          = True
                final_reward  = sim._last_reward
                step_reward   = final_reward
                advance_result = result
                break
            if tool_name == "advance_round" and not sim._done:
                advance_result = result

        rewards.append(step_reward)
        for tool_name, tool_args, _, _ in executed:
            action_str = f"{tool_name}({json.dumps(tool_args, separators=(',', ':'))})"
            _log_step(step=step_count, action=action_str,
                      reward=step_reward if tool_name == "advance_round" else 0.0,
                      done=done, error=step_error)

        # ── Append assistant message (all tool_calls) + tool results ─────────
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
        for tool_name, tool_args, tool_call_id, result in executed:
            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call_id,
                "content":      result[:2700] if result else "OK",
            })

        # ── Context reset after each round advance (mirrors training loop) ───
        if advance_result is not None and not done:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": advance_result},
            ]
            log.info("[ep%d] step %d — round advanced; context reset", episode_num, step_count)

        if not done:
            time.sleep(SLEEP)

    elapsed = time.monotonic() - ep_start
    log.info("[ep%d] seed=%d diff=%s — done=%s reward=%.4f steps=%d elapsed=%.1fs",
             episode_num, seed, difficulty, done, final_reward, step_count, elapsed)

    _log_end(success=done, steps=step_count, score=final_reward, rewards=rewards)

    # Reward decomposition — visible per-component signal so judges and
    # sanity-checking humans can see where the score landed.
    breakdown: Dict[str, float] = {}
    try:
        if sim._state is not None:
            breakdown = compute_reward_breakdown(sim._state, sim._task)
            _log_breakdown(breakdown)
    except Exception as exc:  # pragma: no cover
        log.warning("[ep%d] breakdown computation failed: %s", episode_num, exc)

    return {
        "episode":    episode_num,
        "seed":       seed,
        "difficulty": difficulty,
        "reward":     final_reward,
        "steps":      step_count,
        "done":       done,
        "breakdown":  breakdown,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        raise SystemExit("Set HF_TOKEN or API_KEY before running.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Build episode manifest: seeds × difficulties × rollouts
    manifest: List[Tuple[int, str, int]] = []
    for seed in SEEDS:
        for diff in DIFFICULTIES:
            for _ in range(ROLLOUTS):
                manifest.append((seed, diff, len(manifest) + 1))

    total = len(manifest)
    log.info(
        "Starting %d episode(s)  seeds=%s  difficulties=%s  rollouts=%d  model=%s",
        total, SEEDS, DIFFICULTIES, ROLLOUTS, MODEL_NAME,
    )

    results: List[Dict[str, Any]] = []
    for seed, diff, ep_num in manifest:
        log.info("─── Episode %d / %d  seed=%d  diff=%s ───", ep_num, total, seed, diff)
        try:
            results.append(run_episode(client, seed=seed, difficulty=diff, episode_num=ep_num))
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

    # Aggregate per-component averages across all episodes
    breakdown_keys = [
        "network_sl", "critical_sl", "alloc_acc", "event_resp", "budget_eff",
        "waste_ctrl", "challenge_score", "approval_score",
        "tool_discovery", "briefing_eff",
    ]
    aggregated: Dict[str, float] = {k: 0.0 for k in breakdown_keys}
    counted = 0
    for r in results:
        bd = r.get("breakdown") or {}
        if bd:
            counted += 1
            for k in breakdown_keys:
                aggregated[k] += bd.get(k, 0.0)
    if counted:
        print(f"\n[REWARD COMPONENTS — mean over {counted} episode(s)]", flush=True)
        for k in breakdown_keys:
            print(f"  {k:18s} {aggregated[k]/counted:.4f}", flush=True)


if __name__ == "__main__":
    main()
