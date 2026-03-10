You are participating in a Hidden Profile collaboration task.
Choose one action from the allowed actions and output JSON only.

Allowed actions: {allowed_actions}

Observation:
{observation_json}

Return JSON with the following shape:
{{
  "action": {{"type": "communicate|decide", "payload": {{...}}}},
  "rationale": "optional short rationale"
}}

Communicate payload: {{"channel":"broadcast","content":"...","content_type":"text"}}
Decide payload: {{"decision_id":"plan_selection","choice":"...","reveal":"{decide_reveal}"}}
