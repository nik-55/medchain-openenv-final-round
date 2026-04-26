"""MedChain Env Environment Client."""

import logging
import re
from typing import Any, Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    ListToolsAction,
    ListToolsObservation,
    Tool,
)
from openenv.core.env_server.types import Observation, State

from .models import MedchainState

_log = logging.getLogger(__name__)


class MedchainEnv(EnvClient[CallToolAction, Observation, MedchainState]):
    """
    Client for the MedChain Env hospital supply chain environment.

    Inherits from EnvClient and communicates via the standard OpenEnv
    WebSocket protocol (simulation mode).

    Example:
        >>> async with MedchainEnv(base_url="http://localhost:8000") as env:
        ...     obs = await env.reset()
        ...     print(obs.observation.metadata["dashboard"])
        ...     tools = await env.list_tools()
        ...     result = await env.step(CallToolAction(tool_name="read_inbox", arguments={}))

    Example with Docker:
        >>> env = await MedchainEnv.from_docker_image("medchain_env-env:latest")
        >>> obs = await env.reset()
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("message_timeout_s", 1500.0)
        super().__init__(**kwargs)
        self._tools_cache: Optional[List[Tool]] = None

    # ── EnvClient abstract methods ─────────────────────────────────────────

    def _step_payload(self, action: Any) -> Dict[str, Any]:
        if isinstance(action, ListToolsAction):
            return {"type": "list_tools"}
        if isinstance(action, CallToolAction):
            return {
                "type": "call_tool",
                "tool_name": action.tool_name,
                "arguments": action.arguments,
            }
        raise ValueError(f"Unsupported action type: {type(action).__name__}")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Observation]:
        obs_data = payload.get("observation", {})
        reward = payload.get("reward")
        done = payload.get("done", False) or obs_data.get("done", False)

        # ── List-tools response ──────────────────────────────────────────
        if "tools" in obs_data:
            tools = [
                Tool(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    input_schema=t.get("input_schema", t.get("inputSchema", {})),
                )
                for t in obs_data.get("tools", [])
            ]
            observation = ListToolsObservation(
                tools=tools,
                done=done,
                reward=reward,
            )
            return StepResult(observation=observation, reward=reward, done=done)

        # ── Reset response (has "dashboard" field) ───────────────────────
        if "dashboard" in obs_data:
            observation = Observation(done=done, reward=reward, metadata=obs_data)
            return StepResult(observation=observation, reward=reward, done=done)

        # ── Tool-call response (has "tool_name" and "tool_result") ───────
        if "tool_name" in obs_data:
            result_text = obs_data.get("tool_result", "")

            # Safety net: if reward is still None (should not happen after the
            # serialization fix), fall back to parsing the Final Score from text.
            if reward is None and result_text:
                m = re.search(r"Final Score:\s*([\d.]+)", result_text)
                if m:
                    reward = float(m.group(1))

            observation = Observation(
                done=done,
                reward=reward,
                metadata={"tool_result": result_text},
            )
            return StepResult(observation=observation, reward=reward, done=done)

        # ── Generic fallback ─────────────────────────────────────────────
        observation = Observation(done=done, reward=reward, metadata=obs_data)
        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> MedchainState:
        return MedchainState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", "multi_actor_coordination"),
            round_idx=payload.get("round_idx", 0),
            max_rounds=payload.get("max_rounds", 0),
            day=payload.get("day", 0),
            budget_used=payload.get("budget_used", 0.0),
            budget_limit=payload.get("budget_limit", 0.0),
            unread_messages=payload.get("unread_messages", 0),
            orders_in_transit=payload.get("orders_in_transit", 0),
            pending_request_count=payload.get("pending_request_count", 0),
            active_event_count=payload.get("active_event_count", 0),
        )

    # ── Tool discovery ─────────────────────────────────────────────────────

    async def list_tools(self, use_cache: bool = True) -> List[Tool]:
        """
        Discover the MCP tools available in this environment.

        Args:
            use_cache: Return cached tools if available (default True).

        Returns:
            List of Tool objects with name, description, and input_schema.
        """
        if use_cache and self._tools_cache is not None:
            return self._tools_cache

        result = await self.step(ListToolsAction())
        if isinstance(result.observation, ListToolsObservation):
            self._tools_cache = result.observation.tools
            return self._tools_cache

        self._tools_cache = []
        return self._tools_cache

    # ── Resource cleanup ───────────────────────────────────────────────────

    async def close(self) -> None:
        """Close client, tolerating Docker stop timeouts gracefully."""
        try:
            await super().close()
        except Exception as e:
            # docker stop can time out (10 s) when the container is slow to exit.
            # Log and swallow so the inference script doesn't crash.
            _log.warning("MedchainEnv.close() suppressed error during shutdown: %s", e)
