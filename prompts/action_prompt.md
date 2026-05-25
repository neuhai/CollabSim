You are participating in a collaboration simulation.
Choose one action from the allowed actions and output JSON only.

Allowed actions: {allowed_actions}

Status update:
{status_update}

<Response Format>
Return JSON with the following format:
{
  "action": {"type": "<one_of_allowed_actions>", "payload": {...}},
  "rationale": "optional short rationale"
}

Action payload references:
- message: {"channel":"broadcast|direct","content":"...","content_type":"text","recipients":["B"] (required for direct; direct channel allows exactly ONE recipient only)}
- decide: {"decision_id":"plan_selection","choice":"...","reveal":"{decide_reveal}"}
- produce_shape: {"shape":"<choose_from_task_state>","quantity":1}
- propose_trade_offer: {"offer_type":"buy|sell","shape":"<shape_from_task_state>","price_per_unit":20,"target_id":"B","quantity":1}
- trade_response: {"transaction_id":"offer_2_1","response_type":"accept|decline"}
- cancel_trade_offer: {"transaction_id":"offer_2_1"}
- fulfill_order: {"order_indices":[0,1]}
- make_individual_investment: {"invest_price":30}
- make_group_investment: {"invest_price":30}
- draw: {"cells":[[19,53],[20,53],[21,53]]}
- erase: {"cells":[[19,53],[20,53]]}
- undo: {}
- reset: {}
- do_nothing: {"reason":"..."}

Important:
- Match payload keys exactly.
- Use standard JSON with single braces `{` and `}` — never double braces like `{{` or `}}`.
- If direct communication is enforced, never use broadcast.
- For direct messages, recipients must contain exactly one participant id. To reach multiple people, send separate direct messages.
- If a previous action was rejected, change strategy and do not repeat the same invalid payload.
- In shapefactory, do not default every turn to the same shape. Use your own `observation.agent_id` row in `task_state.participants`.
- Task-specific action constraints (per `task_type`) are defined in the task instructions block earlier in the prompt, not here.
