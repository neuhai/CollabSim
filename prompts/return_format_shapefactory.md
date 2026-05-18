<Response Format>
Return JSON with the following format:
{
  "action": {"type": "<one_enabled_action>", "payload": {...}},
  "rationale": "optional short rationale"
}

Allowed payload examples:
- message: {"channel":"direct","recipients":["B"],"content":"...","content_type":"text"}
- produce_shape: {"shape":"<choose_from_task_state>","quantity":1}
- propose_trade_offer: {"offer_type":"sell","shape":"square","price_per_unit":25,"target_id":"A","quantity":1}
- trade_response: {"transaction_id":"offer_1_1","response_type":"accept"}
- cancel_trade_offer: {"transaction_id":"offer_1_1"}
- fulfill_order: {"order_indices":[0]}
- do_nothing: {"reason":"No valid high-value move this turn."}

Output strictness:
- Return JSON only. No markdown, no prose outside JSON.
- Use exactly one action per response.
