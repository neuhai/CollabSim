Return JSON with the following shape:
{
  "action": {"type": "communicate|decide|do_nothing", "payload": {...}},
  "rationale": "optional short rationale"
}

Communicate payload: {"channel":"broadcast","content":"...","content_type":"text"}
Decide payload: {"decision_id":"plan_selection","choice":"...","reveal":"{decide_reveal}"}
Do nothing payload: {"reason":"..."}
