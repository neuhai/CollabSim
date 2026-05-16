Return JSON with the following shape:
{
  "action": {"type": "<one_enabled_action>", "payload": {...}},
  "rationale": "optional short rationale"
}

Allowed payload examples:
- communicate: {"channel":"broadcast","content":"...","content_type":"text"}
- make_individual_investment: {"invest_price":40}
- make_group_investment: {"invest_price":40}
- do_nothing: {"reason":"No valid high-value move this turn."}

Output strictness:
- Return JSON only. No markdown, no prose outside JSON.
- Use exactly one action per response.
