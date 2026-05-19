FOLLOWER PRE-SUBMIT CHECK (mandatory before writing any JSON when drawing):
  1. Locate your last confirmed drawn point (or S if nothing drawn yet) — this is your anchor.
  2. For EVERY [r, c] in your planned cells: read the character at line r, position c of map_text in your observation.
  3. If that character is '#' → remove that point and everything after it from the list.
  4. Connectivity rule: the FIRST point in cells must be 4-connected (up/down/left/right, no diagonals) to the anchor OR to any previously accepted drawn point. Each SUBSEQUENT point must be 4-connected to the point immediately before it in the list.
  5. Only emit draw.cells after this check passes. If fewer than 2 valid points remain after removing bad cells, try a perpendicular bypass before falling back to `message`.

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
- message: {"channel":"direct","recipients":["B"],"content":"...","content_type":"text"}
- draw: {"cells":[[21,53],[21,52],[21,51]]}
- erase: {"cells":[[21,53],[21,52]]}
- undo: {}
- reset: {}
- do_nothing: {"reason":"No valid high-value move this turn."}

Output strictness:
- Return JSON only. No markdown, no prose outside JSON.
- Use `action` for a single action, or `actions` for ordered multi-action output.
- If you use `actions`, each action must still be valid under current task rules.
- For maptask communication content, do not use coordinates (row/col, x/y, cell indices).
- Use landmark names plus relative directions to describe movement and position.
- Coordinate arrays are allowed only in draw.cells or erase.cells.
