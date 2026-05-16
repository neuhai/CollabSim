<EXPERIMENT RULES>
- Participants will make investment decisions based on market conditions and their own strategies.
- Each participant starts with a fixed amount of money and can make investments at different price points.
- In each experiment session, participants need to make investment decisions within the allowed price range.
- There is limited money allocated to each participant at the beginning of each round, and a time constraint for the experiment.
- DayTrader follows round-based phases:
  - `decision` phase: submit `make_individual_investment`, `make_group_investment`, or `do_nothing`.
  - `group_chat` phase: use `communicate` for coordination (or `do_nothing`).
  - Then the next round starts and phase returns to `decision`.
- In `group_chat`, the controller queries all participants in turn (A, B, C, …) for an initial sequence.
- After that initial sequence, further thinking is message-triggered only (no periodic all-agent re-trigger).
- Free chat ends when both participants choose `do_nothing`, or when `task.phase_rules.group_chat_max_turns` is reached.
- Message-trigger thinking is enabled: if agent A sends a message to selected recipients, recipients are triggered for an extra think cycle.
- During `group_chat`, both direct and broadcast communication are valid, including selected recipient subsets.

<INVESTMENT RULES>
- `make_individual_investment`: invest $X, receive $2X immediately (net gain $X). Only you benefit.
  - `invest_price` must be within configured min/max bounds and positive.
- `make_group_investment`: invest $X into a shared pool. At end of the decision phase, the total pooled amount is distributed equally to ALL participants — each person receives the full pool total.
  - Example (3 participants each invest $10 group): pool = $30, each receives $30 (net +$20 each).
  - Example (1 invests $30 group, 2 invest $0 group): pool = $30, each still receives $30 (net 0 for the investor, +$30 for the others).
  - `invest_price` must be non-negative (0 is valid — you join the group pool without contributing).
  - `invest_price` must not exceed the configured max bound.

<ROUND BONUS RULE>
- Starting from round 2, after each decision phase settles, the participant who earned the most that round receives a $90 bonus.
- "Earned this round" = your money after settlement minus your money at the start of that round.
- If two or more participants tie for the highest earnings, the $90 is split equally among them.
- The bonus is awarded automatically and is visible to everyone.
- This means: if you earn more than the others in a round (e.g. by investing individual while others invest group), you pocket the $90 on top of your regular return.

<EXPERIMENT GOALS>
- Maximize your own monetary balance through strategic investment choices.

<ACTION PLANNING AND RESPONSES>
- Based on your persona, perception of previous and current situations, and experiment objectives, plan for your investment strategies and decide what actions to take at the moment.
- You can choose to perform one or more actions from the available action spaces shown in VALID ACTION SPACES.
- You can choose to wait and take no action if you believe it is the best strategic decision. If you decide to wait, return an empty "actions" array.
- You MUST respond with your planned actions following the JSON structure template below in RESPONSE FORMAT. Instructions in $$ are placeholders for the actual content.

<INSTRUCTIONS ON GENERATING VALID ACTIONS>
- `make_individual_investment`: `invest_price` must be within configured min/max bounds and positive; ensure available funds can cover the investment.
- `make_group_investment`: `invest_price` must be non-negative (0 allowed) and within max bound; ensure available funds can cover the investment.
- Only one investment action per decision phase (individual OR group, not both).
- Use `task_state.phase` and `task_state.round_index` from observation to pick valid actions for the current phase.
- If an action is rejected, change strategy and do not repeat the same invalid payload.

<INSTRUCTIONS ON ALIGNING WITH HUMAN BEHAVIORS>
- Keep communication concise and practical.
- Avoid repetitive messages and avoid emoji.
