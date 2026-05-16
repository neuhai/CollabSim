Return JSON with the following shape:
{
  "action": {"type": "<one_enabled_action>", "payload": {...}},
  "rationale": "optional short rationale"
}

Allowed payload examples:
- communicate: {"channel":"broadcast","content":"...","content_type":"text"}
- decide: {"decision_id":"initial_vote","choice":"Candidate X","reveal":"aggregated"}
- do_nothing: {"reason":"Waiting for others to respond."}

Output strictness:
- Return JSON only. No markdown, no prose outside JSON.
- Use exactly one action per response.
