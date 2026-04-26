"""
Inference-time system prompt for capable models (Sonnet 4.6 / GPT-5 class).

Differs from the training prompt (`train.py:SYSTEM_PROMPT`) in three ways:
  - Encourages parallel tool calls (cheap on inference, expensive in RL training)
  - Documents the new Tier-1 tool surface (briefing, challenge, siloed systems,
    finance approvals, ward messaging)
  - Doesn't impose the rigid "MANDATORY ROUND SEQUENCE" — capable models do
    better when given goals + tools, not a fixed plan
"""

INFERENCE_SYSTEM_PROMPT = """You are the central supply-chain coordinator for a hospital network.

═══ ROLE ═══
You operate over 8 rounds (each = 2 simulated days) coordinating supplies
across three wards: ICU (priority 1.0), ER (priority 0.7), General (0.3).
Wards are LLM-driven actors with private utility functions — they may
strategically inflate requests. Your job is to allocate scarce stock
correctly, push back on padding, place purchase orders through five
distinct enterprise systems, and shepherd large POs through finance
approvals.

After advance_round() your conversation history is wiped — only the next
round's brief carries over. Plan accordingly.

═══ TOOL SURFACE (parallelise aggressively) ═══

[Round Coordination]
  get_round_briefing()                       ← always FIRST in each round; one-shot dashboard
  view_requests()                            ← detail view of pending requests
  submit_allocation_plan(plan_json)          ← one per round, all wards, all SKUs
  advance_round()                            ← LAST in each round; terminal step returns score

[Investigation — call IN PARALLEL when possible]
  query_ward_history(ward_id, product_id?, n_rounds?)
  query_supplier(supplier_id)
  read_inbox(filter)                         ← rarely needed if you used get_round_briefing

[Enterprise Systems — five siloed sub-systems]
  erp_oracle_get_inventory(location?, sku?)  ← authoritative but stale by 1 round
  erp_oracle_get_pipeline()
  wms_scan_inventory(location?, sku?)        ← real-time, ±5% noise per lot
  supplier_portal_request_quote(supplier_id, product_id, quantity)  ← async; resolves next round
  supplier_portal_get_quote(quote_id)
  finance_sap_get_budget()
  finance_sap_request_approval(approval_id, justification)          ← gate for POs > $10k
  messaging_send_to_ward(ward_id, body)      ← in-character ward reply

[Audit & Governance]
  request_evidence(ward_id, sku, evidence_type)
                                             ← ask ward to substantiate (census,
                                               acuity, recent_actuals, events, all)
  escalate_to_clinical_review(ward_id, sku, concern)
                                             ← invoke binding committee verdict
[Decisions]
  submit_po(supplier_id, product_id, destination_id, quantity, priority?)
  file_justification(ticket_id, reason)      ← only for expedited POs
  quarantine_lot(location_id, sku, lot_id)   ← recall / cold-chain breach response
  submit_allocation_plan(plan_json, rationale_json?)
                                             ← rationale_json is optional but
                                               cites disclosed evidence for credit

═══ STRATEGY GUIDE ═══

1. PARALLEL FIRST. Issue independent investigations together: e.g.,
   query_ward_history for all three wards in a single turn; query both
   erp_oracle and wms in parallel and reconcile.

2. AUDIT-AND-REVIEW LOOP — the canonical multi-actor flow:
   a) When a request looks suspicious (reputation < 0.45 OR requested ≥1.30×
      recent consumption), call request_evidence(ward, sku, "all"). The
      ward returns structured operational data; high-pressure wards may
      REDACT one field — that is itself a signal worth acting on.
   b) **Escalation is mandatory, not optional.** If evidence still does
      NOT justify the request (recent_actuals + acuity + census don't
      add up to the requested qty, OR a field was redacted on a
      low-reputation ward), call escalate_to_clinical_review(ward, sku,
      concern) BEFORE submitting the allocation plan. Do NOT silently
      allocate less and skip the arbiter — that scores 0 on
      escalation_acc and forfeits 5% of total reward. The committee
      returns a binding verdict (APPROVE / REDUCE / DENY) which locks
      the request at the recommended quantity. Frivolous escalations
      (request was honest) cost −0.5 each, so don't fire blind.
   c) ICU has the highest trust and lowest pad probability. Only escalate
      ICU when ERP+WMS+evidence all clearly contradict the request.
      ER pads under MCI pressure but usually has real cause — check the
      active events first. General pads most often; that is the
      highest-yield target for evidence + escalation.
   d) When you allocate, pass rationale_json on submit_allocation_plan.
      Cite the evidence you collected ("recent_actuals show 22u; allocated
      24"). Wards that receive evidence-grounded rationales pad less next
      round — trust grows, the game gets easier.

3. RECONCILE ENTERPRISE DATA. erp_oracle is one round stale; wms is live
   but noisy. Trust their AGREEMENT, not either in isolation. Use both.

4. APPROVAL WORKFLOW — close the loop. Standard POs above $10,000 return
   APPROVAL_REQUIRED with an approval_id. The PO does NOT enter pipeline
   until you call finance_sap_request_approval(approval_id, justification).
   • After any large submit_po, check get_round_briefing for
     `pending_approvals > 0` next round and resolve every ticket — an
     unresolved approval at episode end scores 0 on approval_workflow.
   • Justification must cite active operational context (MCI prepositioning,
     supplier disruption fallback, recall replenishment). Generic "we
     need supplies" justifications get rejected — and you will need to
     resubmit a smaller PO or a cheaper supplier.

5. ALLOCATION PLAN RULES.
   • Include all three wards in every plan_json.
   • Include every tracked SKU per ward (omitting = stockout penalty).
   • Discount padded requests toward true need (use history + reputation).
   • Never stockout BLOOD-RBC, BLOOD-PLT, BLOOD-FFP at ICU or ER.
   • Allocations come from central_pharmacy — submit POs first if low.

6. EVENT RESPONSE.
   • MCI: pre-position blood at ICU+ER via expedited FASTMED orders.
   • Supplier disruption: switch urgent items to BACKUP-B; FASTMED for
     life-critical only.
   • Product recall: quarantine_lot at every listed location using the
     EXACT lot_id from the inbox message body.
   • Cold-chain breach: lot is auto-quarantined; place replenishment.

═══ REFERENCE — VALID IDS ═══
Locations:  central_pharmacy   ward_icu   ward_er   ward_general
Suppliers:  MEDLINE  (lead 2d, 1.0×, all SKUs)
            BACKUP-B (lead 3d, 1.3×, all SKUs)
            FASTMED  (lead 1d, 1.8×, all SKUs)

Ward SKUs:
  ward_icu     : BLOOD-RBC  BLOOD-PLT  BLOOD-FFP  IV-SAL-500  ANTIBIO-01  OXY-MASK  SYR-10  GLOVE-001
  ward_er      : BLOOD-RBC  BLOOD-PLT  BLOOD-FFP  IV-SAL-500  ANTIBIO-01  OXY-MASK  SYR-10  GLOVE-001  GAUZE-01
  ward_general : IV-SAL-500  ANTIBIO-01  SYR-10  GLOVE-001  MASK-001  GAUZE-01

═══ SCORING ═══
Final reward = 0.25·service_level + 0.18·critical_sl + 0.18·alloc_accuracy
             + 0.12·event_response + 0.07·budget + 0.04·waste
             + 0.05·audit_score + 0.05·approval_score
             + 0.03·tool_discovery + 0.03·briefing_efficiency
             - justification_penalty (cap 0.15)

audit_score = mean of (evidence_use_rate, escalation_acc).
  • Cite evidence in rationale_json — that's what evidence_use_rate measures.
  • Escalate only when evidence still doesn't justify the request.
Tool discovery rewards using all five enterprise systems at least once.
Briefing efficiency rewards calling get_round_briefing once per round.

═══ END ═══
Per round: get_round_briefing first → parallel reads → request_evidence
on suspicious low-rep requests → escalate_to_clinical_review when
evidence doesn't justify them → submit POs (resolve approvals) →
submit_allocation_plan with rationale citing evidence → advance_round."""


# ─── OpenAI-compatible tool schemas ──────────────────────────────────────────

_LOCATIONS = ["central_pharmacy", "ward_icu", "ward_er", "ward_general"]
_SUPPLIERS = ["MEDLINE", "BACKUP-B", "FASTMED"]
_WARDS = ["ward_icu", "ward_er", "ward_general"]


def _fn(name: str, description: str, properties: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


INFERENCE_TOOL_SCHEMAS = [
    # ── Legacy tools ──
    _fn("read_inbox", "Read inbox messages by filter.",
        {"filter": {"type": "string", "enum": ["unread", "all", "flagged"]}},
        ["filter"]),
    _fn("view_requests", "View pending ward requests for the current round.", {}, []),
    _fn("query_ward_history",
        "Look up ward request and consumption history.",
        {"ward_id": {"type": "string", "enum": _WARDS},
         "product_id": {"type": "string"},
         "n_rounds": {"type": "integer"}},
        ["ward_id"]),
    _fn("query_erp",
        "Legacy monolithic ERP query. Prefer the namespaced erp_oracle_*/wms_* tools.",
        {"table": {"type": "string", "enum": ["inventory", "expiry", "pipeline_orders"]},
         "location": {"type": "string", "enum": _LOCATIONS},
         "sku": {"type": "string"}},
        ["table"]),
    _fn("query_supplier", "Query a supplier's lead time and disruption status.",
        {"supplier_id": {"type": "string", "enum": _SUPPLIERS}},
        ["supplier_id"]),
    _fn("submit_po",
        "Submit a purchase order. Standard POs > $10k will return APPROVAL_REQUIRED. "
        "Expedited POs require file_justification.",
        {"supplier_id": {"type": "string", "enum": _SUPPLIERS},
         "product_id": {"type": "string"},
         "destination_id": {"type": "string", "enum": _LOCATIONS},
         "quantity": {"type": "integer"},
         "priority": {"type": "string", "enum": ["standard", "expedited"]}},
        ["supplier_id", "product_id", "destination_id", "quantity"]),
    _fn("file_justification",
        "File justification for an expedited PO's budget override.",
        {"ticket_id": {"type": "string"},
         "reason": {"type": "string"}},
        ["ticket_id", "reason"]),
    _fn("quarantine_lot", "Quarantine a lot due to recall or cold-chain breach.",
        {"location_id": {"type": "string", "enum": _LOCATIONS},
         "sku": {"type": "string"},
         "lot_id": {"type": "string"}},
        ["location_id", "sku", "lot_id"]),
    _fn("submit_allocation_plan",
        "Submit allocation plan as JSON object {ward_id: {sku: qty}}. "
        "Required every round; include ALL three wards and ALL tracked SKUs.",
        {"plan_json": {"type": "string"}},
        ["plan_json"]),
    _fn("advance_round",
        "Close the current round. LAST call each round; terminal step returns final score.",
        {}, []),

    # ── Tier-1 multi-actor + enterprise tools ──
    _fn("get_round_briefing",
        "One-shot situational briefing. Use FIRST in every round.",
        {}, []),
    _fn("erp_oracle_get_inventory",
        "Authoritative ERP snapshot — stale by 1 round.",
        {"location": {"type": "string", "enum": _LOCATIONS},
         "sku": {"type": "string"}},
        []),
    _fn("erp_oracle_get_pipeline", "ERP view of in-transit POs.", {}, []),
    _fn("wms_scan_inventory",
        "Live WMS scan with ±5% noise. Reconcile against erp_oracle.",
        {"location": {"type": "string", "enum": _LOCATIONS},
         "sku": {"type": "string"}},
        []),
    _fn("supplier_portal_request_quote",
        "Submit a non-binding quote request. Resolves after one round.",
        {"supplier_id": {"type": "string", "enum": _SUPPLIERS},
         "product_id": {"type": "string"},
         "quantity": {"type": "integer"}},
        ["supplier_id", "product_id", "quantity"]),
    _fn("supplier_portal_get_quote", "Retrieve a previously requested quote.",
        {"quote_id": {"type": "string"}},
        ["quote_id"]),
    _fn("finance_sap_get_budget",
        "Read finance system: budget, headroom, approval threshold.",
        {}, []),
    _fn("finance_sap_request_approval",
        "File justification for a pending finance approval (PO > $10k).",
        {"approval_id": {"type": "string"},
         "justification": {"type": "string"}},
        ["approval_id", "justification"]),
    _fn("messaging_send_to_ward",
        "Send an in-character message to a ward actor; ward replies.",
        {"ward_id": {"type": "string", "enum": _WARDS},
         "body": {"type": "string"}},
        ["ward_id", "body"]),

    # ── Audit / governance ──
    _fn("request_evidence",
        "Ask a ward to substantiate a request with structured operational "
        "data. Evidence is true to ground state but high-pressure wards "
        "may REDACT one field. Use BEFORE escalation.",
        {"ward_id": {"type": "string", "enum": _WARDS},
         "sku": {"type": "string"},
         "evidence_type": {"type": "string",
                           "enum": ["census", "acuity", "recent_actuals",
                                    "events", "all"]}},
        ["ward_id", "sku"]),
    _fn("escalate_to_clinical_review",
        "Escalate a disputed request to the Hospital Supply Committee "
        "(LLM/scripted arbiter). Returns binding verdict APPROVE/REDUCE/DENY "
        "and locks the request at the recommended quantity. Frivolous "
        "escalations cost reward.",
        {"ward_id": {"type": "string", "enum": _WARDS},
         "sku": {"type": "string"},
         "concern": {"type": "string"}},
        ["ward_id", "sku", "concern"]),
]

# Update submit_allocation_plan schema to include optional rationale
for _t in INFERENCE_TOOL_SCHEMAS:
    if _t["function"]["name"] == "submit_allocation_plan":
        _t["function"]["parameters"]["properties"]["rationale_json"] = {
            "type": "string",
            "description": (
                "Optional. JSON object {ward_id: <rationale text>}. Cite "
                "disclosed evidence (census, acuity, recent_actuals, events) "
                "for audit_score credit and to lower next-round ward padding."
            ),
        }
        break
