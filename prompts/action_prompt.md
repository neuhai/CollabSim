You are participating in a collaboration simulation.
Choose one action from the allowed actions and output JSON only.

Allowed actions: {allowed_actions}

Observation:
{observation_json}

Return JSON with the following shape:
{{
  "action": {{"type": "<one_of_allowed_actions>", "payload": {{...}}}},
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
- make_investment: {{"invest_price":30,"invest_decision_type":"individual|group"}}
- update_map_progress: {{"map_progress":{{"segment":"start_to_bridge","status":"confirmed"}}}}
- do_nothing: {{"reason":"..."}}

Important:
- Match payload keys exactly.
- If direct communication is enforced, never use broadcast.
- If a previous action was rejected, change strategy and do not repeat the same invalid payload.
- In shapefactory, do not default every turn to the same shape. Use your own `observation.agent_id` row in `task_state.participants`.
- Task-specific action constraints (per `task_type`) are defined in the task instructions block earlier in the prompt, not here.
