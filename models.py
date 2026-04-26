"""Client-side state and observation types for the MedChain Env (finals)."""

from openenv.core.env_server import State
from openenv.core.env_server.types import Observation
from pydantic import Field
from typing import List, Optional

AVAILABLE_TOOLS = [
    # Legacy single-system tools
    "read_inbox",
    "view_requests",
    "query_ward_history",
    "query_erp",
    "query_supplier",
    "submit_po",
    "file_justification",
    "quarantine_lot",
    "submit_allocation_plan",
    "advance_round",
    # Tier-1 multi-actor + enterprise additions
    "get_round_briefing",
    "erp_oracle_get_inventory",
    "erp_oracle_get_pipeline",
    "wms_scan_inventory",
    "supplier_portal_request_quote",
    "supplier_portal_get_quote",
    "finance_sap_get_budget",
    "finance_sap_request_approval",
    "messaging_send_to_ward",
    "request_evidence",
    "escalate_to_clinical_review",
]


class MedchainState(State):
    """Runtime state exposed by the environment server."""

    task: str = Field(
        default="multi_actor_coordination",
        description="Task name (always multi_actor_coordination)",
    )
    round_idx: int = Field(default=0, description="Current round index (1-indexed)")
    max_rounds: int = Field(default=0, description="Total rounds for this episode")
    day: int = Field(default=0, description="Current simulation day (1-indexed)")
    budget_used: float = Field(default=0.0, description="Outstanding committed PO budget ($)")
    budget_limit: float = Field(default=0.0, description="Budget ceiling ($)")
    unread_messages: int = Field(default=0, description="Unread inbox messages")
    orders_in_transit: int = Field(default=0, description="POs currently in transit")
    pending_request_count: int = Field(default=0, description="Ward requests awaiting allocation")
    active_event_count: int = Field(default=0, description="Active crisis events")
    pending_approvals: int = Field(default=0, description="Finance approvals awaiting justification")
    pending_quotes: int = Field(default=0, description="Supplier portal quotes awaiting resolution")
    systems_used: int = Field(default=0, description="Distinct enterprise systems accessed")


class MedObservation(Observation):
    """Initial observation returned by reset(). Contains round 1 brief text."""

    dashboard: str = Field(default="", description="Round 1 briefing")
    available_tools: List[str] = Field(default_factory=list, description="Available tools")
    episode_id: str = Field(default="", description="Episode ID")
    round_idx: int = Field(default=0, description="Current round index")


class MedchainToolObservation(Observation):
    """Observation returned for every tool-call step."""

    tool_name: str = Field(default="", description="Name of the tool that was called")
    tool_result: str = Field(default="", description="Text result from the tool")
    error_msg: Optional[str] = Field(default=None, description="Error message if call failed")
