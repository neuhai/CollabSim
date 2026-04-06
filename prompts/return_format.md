Return JSON with the following shape:
{
  "action": {"type": "<one_enabled_action>", "payload": {...}},
  "rationale": "optional short rationale"
}

Allowed payload examples:
- communicate: {"channel":"direct","recipients":["B"],"content":"...","content_type":"text"}
- decide: {"decision_id":"plan_selection","choice":"...","reveal":"{decide_reveal}"}
- produce_shape: {"shape":"<choose_from_task_state>","quantity":1}
- propose_trade_offer: {"offer_type":"sell","shape":"square","price_per_unit":25,"target_id":"A","quantity":1}
- trade_response: {"transaction_id":"offer_1_1","response_type":"accept"}
- cancel_trade_offer: {"transaction_id":"offer_1_1"}
- fulfill_order: {"order_indices":[0]}
- make_investment: {"invest_price":40,"invest_decision_type":"individual"}
- update_map_progress: {"map_progress":{"segment":"bridge_to_tower","status":"done"}}
- do_nothing: {"reason":"No valid high-value move this turn."}

Output strictness:
- Return JSON only. No markdown, no prose outside JSON.
- Use exactly one action per response.
