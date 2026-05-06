Return JSON with one of the following shapes:
{
  "action": {"type": "<one_enabled_action>", "payload": {...}},
  "rationale": "optional short rationale"
}
or
{
  "actions": [
    {"type": "<enabled_action_1>", "payload": {...}},
    {"type": "<enabled_action_2>", "payload": {...}}
  ],
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
- make_individual_investment: {"invest_price":40}
- make_group_investment: {"invest_price":40}
- update_map_progress: {"map_progress":{"segment":"bridge_to_tower","status":"done","drawn_points":[[21,53],[21,52],[21,51]]}}
- do_nothing: {"reason":"No valid high-value move this turn."}

Output strictness:
- Return JSON only. No markdown, no prose outside JSON.
- Use `action` for a single action, or `actions` for ordered multi-action output.
- If you use `actions`, each action must still be valid under current task rules.
- For maptask communication content, do not use coordinates (row/col, x/y, cell indices).
- Use landmark names plus relative directions to describe movement and position.
- Coordinate arrays are allowed only in `update_map_progress.map_progress.drawn_points`.
