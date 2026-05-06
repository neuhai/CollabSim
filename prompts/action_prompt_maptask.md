You are participating in a collaboration simulation.
Choose one or more actions from the allowed actions and output JSON only.

Allowed actions: {allowed_actions}

Observation:
{observation_json}

Return JSON with one of the following shapes:
{{
  "action": {{"type": "<one_of_allowed_actions>", "payload": {{...}}}},
  "rationale": "optional short rationale"
}}
or
{{
  "actions": [
    {{"type": "<one_of_allowed_actions>", "payload": {{...}}}},
    {{"type": "<one_of_allowed_actions>", "payload": {{...}}}}
  ],
  "rationale": "optional short rationale"
}}

Action payload references:
- communicate: {{"channel":"broadcast|direct","content":"...","content_type":"text","recipients":["B"] (required for direct)}}
- decide: {{"decision_id":"plan_selection","choice":"...","reveal":"{decide_reveal}"}}
- produce_shape: {{"shape":"<choose_from_task_state>","quantity":1}}
- propose_trade_offer: {{"offer_type":"buy|sell","shape":"<shape_from_task_state>","price_per_unit":20,"target_id":"B","quantity":1}}
- trade_response: {{"transaction_id":"offer_2_1","response_type":"accept|decline"}}
- cancel_trade_offer: {{"transaction_id":"offer_2_1"}}
- fulfill_order: {{"order_indices":[0,1]}}
- make_individual_investment: {{"invest_price":30}}
- make_group_investment: {{"invest_price":30}}
- update_map_progress: {{"map_progress":{{"segment":"start_to_bridge","status":"confirmed","drawn_points":[[19,53],[20,53],[21,53]]}}}}
- do_nothing: {{"reason":"..."}}

Important:
- Match payload keys exactly.
- If direct communication is enforced, never use broadcast.
- If a previous action was rejected, change strategy and do not repeat the same invalid payload.
- In maptask follower turns, you may combine `update_map_progress` and `communicate` in one ordered `actions` array.
- In maptask `communicate.content`, never describe positions with coordinates (row/col, x/y, cell indices).
- Use landmark names and relative directions for spatial descriptions.
- Coordinates are only valid inside `update_map_progress.map_progress.drawn_points`.
- **Follower drawing:** Before `update_map_progress`, every `[row,col]` must match `map_text` with that cell **not** `#`, the path must be **4-connected** from `S` or existing route, and the **first** step must be a **legal neighbor of `S`**. Use shorter `drawn_points` lists when unsure.
- Task-specific action constraints (per `task_type`) are defined in the task instructions block earlier in the prompt, not here.
