"""
MedChain Finals — OpenEnv Environment implementation.

Tool dispatch is delegated to an internal _MedchainMCPDelegate (MCPEnvironment)
for FastMCP schema generation. MedchainEnvironment itself handles reward
computation and returns MedchainToolObservation with a plain float reward.
"""

import uuid
from typing import Any, Optional

from fastmcp import FastMCP
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
)
from openenv.core.env_server.types import Action, Observation

try:
    from ..models import AVAILABLE_TOOLS, MedchainState, MedObservation, MedchainToolObservation
    from .simulation import MedchainSimulation
    from .tasks import make_task_config
except ImportError:
    from models import AVAILABLE_TOOLS, MedchainState, MedObservation, MedchainToolObservation
    from server.simulation import MedchainSimulation
    from server.tasks import make_task_config


class _MedchainMCPDelegate(MCPEnvironment):
    """Thin MCPEnvironment wrapper holding the FastMCP server for tool dispatch."""

    def reset(self, **kwargs: Any) -> Observation:  # type: ignore[override]
        return Observation(done=False, reward=0.0)

    def _step_impl(self, action: Action, **kwargs: Any) -> Observation:
        return Observation(done=False, reward=0.0)

    @property
    def state(self) -> MedchainState:
        return MedchainState()


class MedchainEnvironment(Environment):
    """
    Multi-actor hospital supply coordination environment.

    The agent is the central procurement coordinator. Three scripted wards
    (ICU, ER, General) submit requests each round. The agent allocates
    scarce stock, places purchase orders, and calls advance_round to
    progress. Reward is computed terminally (final round's advance_round).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, seed: int = 0, difficulty: str = "medium"):
        self._default_seed = seed
        self._default_difficulty = difficulty
        self._sim = MedchainSimulation(
            make_task_config(seed=seed, difficulty=difficulty)
        )
        self._step_count = 0

        # ── Register MCP tools ────────────────────────────────────────────

        mcp = FastMCP("medchain_env")

        @mcp.tool
        def read_inbox(filter: str = "unread") -> str:
            """
            Read messages from the central operations inbox.

            Args:
                filter: 'unread' (default), 'all', or 'flagged'.
            """
            return self._sim.read_inbox(filter)

        @mcp.tool
        def view_requests() -> str:
            """
            View all pending ward allocation requests for the current round.
            Returns ward ID, SKU, requested quantity, priority weight, and
            justification text. True need and padding information are hidden.
            """
            return self._sim.view_requests()

        @mcp.tool
        def query_ward_history(
            ward_id: str,
            product_id: Optional[str] = None,
            n_rounds: int = 10,
        ) -> str:
            """
            Look up a ward's historical requests and actual consumption.
            Includes pre-seeded synthetic prior rounds (round_idx < 0) so
            history is available from round 1.

            Args:
                ward_id: 'ward_icu', 'ward_er', or 'ward_general'.
                product_id: optional SKU filter.
                n_rounds: how many recent distinct rounds to show (default 10).
            """
            return self._sim.query_ward_history(ward_id, product_id, n_rounds)

        @mcp.tool
        def query_erp(table: str, location: str = "all", sku: str = "all") -> str:
            """
            Query the legacy ERP database.

            Args:
                table: 'inventory', 'expiry', or 'pipeline_orders'.
                location: location_id or 'all'.
                sku: product_id or 'all'.
            """
            return self._sim.query_erp(table, location, sku)

        @mcp.tool
        def query_supplier(supplier_id: str) -> str:
            """Query supplier lead time and active disruption status."""
            return self._sim.query_supplier(supplier_id)

        @mcp.tool
        def submit_po(
            supplier_id: str,
            product_id: str,
            destination_id: str,
            quantity: int,
            priority: str = "standard",
        ) -> str:
            """
            Submit a purchase order to a supplier. 'expedited' orders require a
            subsequent call to file_justification with the returned ticket_id.

            Args:
                supplier_id: MEDLINE, BACKUP-B, or FASTMED.
                product_id: SKU.
                destination_id: central_pharmacy or ward_*.
                quantity: positive integer.
                priority: 'standard' (default) or 'expedited'.
            """
            return self._sim.submit_po(
                supplier_id, product_id, destination_id, quantity, priority
            )

        @mcp.tool
        def file_justification(ticket_id: str, reason: str) -> str:
            """
            File a justification for an expedited order's budget override.
            Justifications are audited against active event context.
            """
            return self._sim.file_justification(ticket_id, reason)

        @mcp.tool
        def quarantine_lot(location_id: str, sku: str, lot_id: str) -> str:
            """
            Quarantine an inventory lot (e.g. following a recall).
            Use lot_id='all' to quarantine every lot of the SKU at this location.
            """
            return self._sim.quarantine_lot(location_id, sku, lot_id)

        @mcp.tool
        def submit_allocation_plan(plan_json: str) -> str:
            """
            Submit the allocation plan for the current round.

            plan_json is a JSON object: {ward_id: {product_id: qty}}.

            Example:
                {"ward_icu": {"BLOOD-RBC": 10, "BLOOD-PLT": 4},
                 "ward_er": {"BLOOD-RBC": 6},
                 "ward_general": {"SYR-10": 60}}

            Any pending request not explicitly in the plan receives zero
            allocation. Quantities above the requested amount are capped.
            """
            return self._sim.submit_allocation_plan(plan_json)

        @mcp.tool
        def advance_round() -> str:
            """
            Close the current round and advance the simulation by
            round_length_days. Returns the next round's brief, or the
            terminal summary if the episode ends.
            """
            return self._sim.advance_round()

        # ═══════════════════════════════════════════════════════════════════
        # Tier-1 multi-actor + enterprise tools
        # ═══════════════════════════════════════════════════════════════════

        @mcp.tool
        def get_round_briefing() -> str:
            """
            One-shot situational briefing for this round. Replaces the
            common four-tool sequence (read_inbox + view_requests +
            query_erp(inventory) + query_erp(pipeline_orders)). Always call
            this first each round — it is the most efficient way to
            establish situational awareness.
            """
            return self._sim.get_round_briefing()

        @mcp.tool
        def erp_oracle_get_inventory(location: str = "all", sku: str = "all") -> str:
            """
            Authoritative legacy ERP inventory snapshot. **Stale by 1 round.**
            Use for cross-checking against wms_scan_inventory.
            """
            return self._sim.erp_oracle_get_inventory(location, sku)

        @mcp.tool
        def erp_oracle_get_pipeline() -> str:
            """ERP view of all in-transit purchase orders."""
            return self._sim.erp_oracle_get_pipeline()

        @mcp.tool
        def wms_scan_inventory(location: str = "all", sku: str = "all") -> str:
            """
            Real-time warehouse-management-system scan. Live but noisy
            (~5% per-lot noise). Reconcile against erp_oracle for accuracy.
            """
            return self._sim.wms_scan_inventory(location, sku)

        @mcp.tool
        def supplier_portal_request_quote(
            supplier_id: str, product_id: str, quantity: int,
        ) -> str:
            """
            Submit a non-binding quote request to a supplier portal.
            Returns a quote_id; the actual quote resolves after one round.
            Use supplier_portal_get_quote(quote_id) to retrieve once ready.
            """
            return self._sim.supplier_portal_request_quote(supplier_id, product_id, quantity)

        @mcp.tool
        def supplier_portal_get_quote(quote_id: str) -> str:
            """Retrieve a previously requested quote by id (returns pending or terms)."""
            return self._sim.supplier_portal_get_quote(quote_id)

        @mcp.tool
        def finance_sap_get_budget() -> str:
            """
            Read the finance system: outstanding budget, headroom,
            approval threshold, count of pending approvals.
            """
            return self._sim.finance_sap_get_budget()

        @mcp.tool
        def finance_sap_request_approval(approval_id: str, justification: str) -> str:
            """
            File a justification for a pending finance approval.
            Justifications are graded against active operational events
            (MCI, supplier disruption, recall, cold-chain breach).
            """
            return self._sim.finance_sap_request_approval(approval_id, justification)

        @mcp.tool
        def messaging_send_to_ward(ward_id: str, body: str) -> str:
            """
            Send a direct message to a ward's actor. The ward actor replies
            in-character. Useful for clarifications, soft pushback, or
            policy signalling without using the formal challenge flow.
            """
            return self._sim.messaging_send_to_ward(ward_id, body)

        @mcp.tool
        def request_evidence(ward_id: str, sku: str, evidence_type: str = "all") -> str:
            """
            Ask a ward to substantiate a request with structured data.
            Evidence is pulled from operational records (true to ground state)
            but high-pressure wards may REDACT one field — that itself is a
            signal worth acting on.

            evidence_type ∈ {census, acuity, recent_actuals, events, all}.
            """
            return self._sim.request_evidence(ward_id, sku, evidence_type)

        @mcp.tool
        def escalate_to_clinical_review(
            ward_id: str, sku: str, concern: str,
        ) -> str:
            """
            Escalate a disputed request to the Hospital Supply Committee.
            The committee returns a binding verdict (APPROVE / REDUCE / DENY)
            with a written rationale; the request is locked at the recommended
            quantity for this round.

            Use after evidence-gathering when you believe a request is
            substantively inflated. Frivolous escalations cost reward.
            """
            return self._sim.escalate_to_clinical_review(ward_id, sku, concern)

        super().__init__()
        self._mcp_env = _MedchainMCPDelegate(mcp)

    # ── OpenEnv interface ──────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        self._step_count = 0
        ep_id = episode_id or str(uuid.uuid4())
        seed_val = seed if seed is not None else self._default_seed
        diff = difficulty or self._default_difficulty
        # Rebuild task with given seed/difficulty
        self._sim = MedchainSimulation(make_task_config(seed=seed_val, difficulty=diff))
        brief = self._sim.reset(seed=seed_val, episode_id=ep_id, difficulty=diff)

        state = self._sim._state
        return MedObservation(
            dashboard=brief,
            available_tools=AVAILABLE_TOOLS,
            episode_id=ep_id,
            round_idx=state.round_idx if state else 0,
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._step_count += 1

        if isinstance(action, ListToolsAction):
            return self._mcp_env._handle_list_tools()

        if isinstance(action, CallToolAction):
            return self._handle_call_tool(action, timeout_s=timeout_s)

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": (
                    f"Unknown action type: {type(action).__name__}. "
                    f"Use CallToolAction with one of: {AVAILABLE_TOOLS}"
                )
            },
        )

    def _handle_call_tool(
        self, action: CallToolAction, timeout_s: Optional[float] = None,
    ) -> MedchainToolObservation:
        call_obs: CallToolObservation = self._mcp_env._handle_call_tool(
            action, timeout_s=timeout_s
        )
        result_text = self._extract_result_text(call_obs)
        is_error = (call_obs.error is not None) or result_text.startswith("ERROR")

        if action.tool_name == "advance_round":
            # If the simulation reports done, the reward is the final score.
            done = self._sim.is_done()
            reward = self._sim.get_last_reward() if done else 0.0
        else:
            reward = self._shaping_reward(action.tool_name, result_text, is_error)
            done = False

        error_msg: Optional[str] = None
        if call_obs.error is not None:
            err = call_obs.error
            error_msg = err.message if hasattr(err, "message") else str(err)

        return MedchainToolObservation(
            tool_name=action.tool_name,
            tool_result=result_text,
            error_msg=error_msg,
            done=done,
            reward=reward,
        )

    @staticmethod
    def _extract_result_text(call_obs: CallToolObservation) -> str:
        if call_obs.error is not None:
            err = call_obs.error
            msg = err.message if hasattr(err, "message") else str(err)
            return f"ERROR: {msg}"
        r = call_obs.result
        if r is None:
            return ""
        if hasattr(r, "data") and r.data is not None:
            return str(r.data)
        if hasattr(r, "content") and r.content:
            first = r.content[0]
            return first.text if hasattr(first, "text") else str(first)
        return str(r)

    # ── Shaping rewards ────────────────────────────────────────────────────

    def _shaping_reward(self, tool_name: str, result_text: str, is_error: bool) -> float:
        """Small per-call shaping signals. Resets each round."""
        state = self._sim._state
        if state is None:
            return 0.0

        flags = state.shaping_flags_this_round

        if is_error and tool_name != "file_justification":
            return 0.0

        if tool_name == "read_inbox":
            if "read_inbox" not in flags and "INBOX EMPTY" not in result_text:
                flags.add("read_inbox")
                return 0.01
            return 0.0

        if tool_name == "view_requests":
            if "view_requests" not in flags and "no pending" not in result_text.lower():
                flags.add("view_requests")
                return 0.02
            return 0.0

        if tool_name == "submit_allocation_plan":
            if "submit_allocation_plan" not in flags and "COMMITTED" in result_text:
                flags.add("submit_allocation_plan")
                return 0.03
            return 0.0

        if tool_name == "submit_po":
            if "OK — PO" in result_text and not is_error:
                return 0.02
            return 0.0

        if tool_name == "quarantine_lot" and not is_error:
            # extra bonus when quarantining within the active recall window
            recall_active = any(
                e.event_type == "product_recall" and e.trigger_day <= state.day
                for e in self._sim._task.events
            )
            return 0.03 if recall_active else 0.005

        if tool_name == "file_justification":
            if "FLAGGED" in result_text:
                return -0.05
            if "accepted" in result_text:
                return 0.01
            return 0.0

        if tool_name == "get_round_briefing":
            if "briefing_used" not in flags:
                flags.add("briefing_used")
                return 0.02
            return 0.0

        if tool_name == "request_evidence" and not is_error:
            # Small reward; redaction signals get the agent its real value
            if "REDACTED" in result_text:
                return 0.02
            return 0.01

        if tool_name == "escalate_to_clinical_review" and not is_error:
            if "REDUCE" in result_text or "DENY" in result_text:
                return 0.05
            if "APPROVE" in result_text:
                return -0.03   # frivolous
            return 0.0

        if tool_name == "finance_sap_request_approval":
            if "GRANTED" in result_text:
                return 0.03
            if "DENIED" in result_text:
                return -0.02
            return 0.0

        if tool_name in (
            "erp_oracle_get_inventory", "erp_oracle_get_pipeline",
            "wms_scan_inventory", "supplier_portal_request_quote",
            "supplier_portal_get_quote", "finance_sap_get_budget",
            "messaging_send_to_ward",
        ) and not is_error:
            # Tiny shaping for first-use of each enterprise system per round
            sentinel = f"used:{tool_name}"
            if sentinel not in flags:
                flags.add(sentinel)
                return 0.005
            return 0.0

        return 0.0

    # ── Generic dispatch (used by inference.py and external drivers) ──────

    def dispatch(self, tool_name: str, **kwargs: Any) -> str:
        """Route a tool name to the underlying simulation method.

        All MedChain tools are implemented as methods on MedchainSimulation;
        this helper is what external drivers (inference.py) use instead of
        re-implementing the dispatch table. Returns the tool's text result
        or a string error message.
        """
        sim = self._sim
        if not hasattr(sim, tool_name):
            return f"ERROR: Unknown tool '{tool_name}'."
        try:
            return getattr(sim, tool_name)(**kwargs)
        except TypeError as exc:
            return f"ERROR: bad arguments to {tool_name}: {exc}"
        except Exception as exc:
            return f"ERROR: {tool_name} raised: {exc}"

    # ── State property ─────────────────────────────────────────────────────

    @property
    def state(self) -> MedchainState:
        s = self._sim._state
        if s is None:
            return MedchainState()
        return MedchainState(
            episode_id=s.episode_id,
            step_count=self._step_count,
            task=s.task,
            round_idx=s.round_idx,
            max_rounds=s.max_rounds,
            day=s.day,
            budget_used=s.budget_used,
            budget_limit=s.budget_limit,
            unread_messages=sum(1 for m in s.inbox if not m.read),
            orders_in_transit=sum(
                1 for po in s.pipeline_orders if po.status == "in_transit"
            ),
            pending_request_count=len(s.pending_requests),
            active_event_count=len(s.active_events),
            pending_approvals=len(getattr(s, "pending_approvals", {}) or {}),
            pending_quotes=len(getattr(s, "pending_quotes", {}) or {}),
            systems_used=len(getattr(s, "systems_used", set()) or set()),
        )
