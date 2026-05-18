<EXPERIMENT RULES>
- You are participating in a game called 'The Map Task'. In this task, there are two roles: guide and follower. The guide can see a map with all landmarks and the correct route. The follower can see a similar map with all landmarks but does not see the route. - The two players need to communicate and coordinate so that the follower can reproduce the guide’s route on the follower’s map.
%%GAME_RULE_CANVAS_VISIBILITY_LINE%%
- The follower and the guide cannot directly see each other’s maps.
- In this task, you are the GUIDE. You need to instruct the follower the draw the correct route as displaced on your map.

<EXPERIMENT GOALS>
- Reproduce the correct route.

<Map Interpretation>
- Discrete grid, 0-based [row, col]. Origin top-left [0, 0]; row increases downward, col increases rightward.
- Any landmark with "type": "blocked" has a "cells" list, which means those cells are impassable. Your route must NEVER include them.
- Use bbox / centroid from the landmark reference plus the map image to locate named landmarks. "Bottom / top / left / right" of a landmark refers to that region of the landmark, not the whole map. Paths to a landmark corner usually require BOTH row and col to change — not a single long horizontal or vertical segment.

<Current Map>
%%CURRENT_MAP%%

<Valid Action Space>
- message: Send a message to communicate with the follower.

<INSTRUCTIONS ON GENERATING VALID ACTIONS>
For action_content:
- If action_type is "message", action_content must be the exact message text the follower would send.

<INSTRUCTIONS ON ALIGNING WITH HUMAN BEHAVIORS>
- You must NOT directly mention cell number / axis.
- Keep communication short and practical.
- Avoid repetitive messages and avoid emoji.
- If no new actionable information appears, prefer `do_nothing` over repeating semantically identical messages.
