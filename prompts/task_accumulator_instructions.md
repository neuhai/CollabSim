<EXPERIMENT RULES>
- Use observation state to track `current_value`, `target_value`, and increment-related progress.

<EXPERIMENT GOALS>
- Reach target value with stable progress.

<ACTION PLANNING AND RESPONSES>
- Choose exactly one action each response.

<VALID ACTION SPACES>
- message
- decide
- do_nothing

<RESPONSE FORMAT>
{
  "action": {
    "type": "<one_enabled_action>",
    "payload": { ... }
  },
  "rationale": "short strategic reason"
}

<INSTRUCTIONS ON GENERATING VALID ACTIONS>
- If using `decide`, keep payload schema-valid and aligned with accumulation progress.
- If rejected, adjust and do not resend the same invalid payload.
