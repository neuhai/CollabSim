FOLLOWER PRE-SUBMIT CHECK (mandatory before writing any JSON when drawing):
  1. Read **Your route state** and **Recent action rejections** in the Status update — anchor there, not on rejected cells.
  2. Locate your last **confirmed** drawn point (or S if nothing drawn yet) — this is your anchor.
  3. For EVERY [r, c] in your planned cells: read the character at line r, position c of map_text when available.
  4. If that character is '#' → remove that point and everything after it from the list.
  5. **4-connected only (no diagonals):** each step changes exactly ONE of row or col by ±1, never both.
     - INVALID diagonal SW: [[6,34],[7,33],[8,32]]
     - VALID stair-step SW: [[6,34],[7,34],[7,33],[8,33],[8,32]]
  6. The FIRST point in cells must be 4-connected to the anchor or any previously accepted drawn point. Each SUBSEQUENT point must be 4-connected to the point immediately before it.
  7. Only emit draw.cells after this check passes. If fewer than 2 valid points remain, try a perpendicular stair-step before falling back to `message`.
  8. After a draw/erase rejection, do **not** resubmit the same cells — fix connectivity first (see Status update rejections).

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
