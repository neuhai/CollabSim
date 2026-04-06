<EXPERIMENT RULES>
- Use observation state to track `target_steps`, progress, and completion.

<EXPERIMENT GOALS>
- Reach target steps efficiently.

<ACTION PLANNING AND RESPONSES>
- Choose exactly one action each response.

<VALID ACTION SPACES>
- communicate
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
- If using `decide`, keep payload schema-valid and focused on progress.
- If rejected, change payload and do not repeat invalid output.
