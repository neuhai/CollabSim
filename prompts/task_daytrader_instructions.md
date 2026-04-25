<EXPERIMENT RULES>
- Participants will make investment decisions based on market conditions and their own strategies.
- Each participant starts with a fixed amount of money and can make investments at different price points.
- In each experiment session, participants need to make investment decisions within the allowed price range.
- Each investment decision can be made individually or as part of a group decision.
- There is limited money allocated to each participant at the beginning of each round, and a time constraint for the experiment.
- Participants can obtain returns through strategic investment decisions based on market conditions.
- Your investment decisions should reflect your risk tolerance and personality traits.
- DayTrader follows round-based phases:
  - `decision` phase: submit `make_investment` (or `do_nothing`) only.
  - `group_chat` phase: use `communicate` for coordination (or `do_nothing`).
  - Then the next round starts and phase returns to `decision`.
- In `group_chat` with two participants, controller forces one initial sequence: query `A` first, then query `B`.
- After that initial sequence, further thinking is message-triggered only (no periodic all-agent re-trigger).
- Free chat ends when both participants choose `do_nothing`, or when `task.phase_rules.group_chat_max_turns` is reached.
- Message-trigger thinking is enabled: if agent A sends a message to selected recipients, recipients are triggered for an extra think cycle.
- During `group_chat`, both direct and broadcast communication are valid, including selected recipient subsets.


<EXPERIMENT GOALS>
- Maximize monetary balance through strategic investment choices.

<ACTION PLANNING AND RESPONSES>
- Based on your persona, perception of previous and current situations, and experiment objectives, plan for your investment strategies and decide what actions to take at the moment.
- You can choose to perform one or more actions from the available action spaces shown in VALID ACTION SPACES.
- You can choose to wait and take no action if you believe it is the best strategic decision. If you decide to wait, return an empty "actions" array.
- You MUST respond with your planned actions following the JSON structure template below in RESPONSE FORMAT. Instructions in $$ are placeholders for the actual content.

<INSTRUCTIONS ON GENERATING VALID ACTIONS>
- `make_investment`:
  - `invest_price` must be within configured min/max bounds
  - `invest_decision_type` must be `individual` or `group`
  - ensure available funds can cover the investment
- Use `task_state.phase` and `task_state.round_index` from observation to pick valid actions for the current phase.
- If an action is rejected, change strategy and do not repeat the same invalid payload.

<INSTRUCTIONS ON ALIGNING WITH HUMAN BEHAVIORS>
- Keep communication concise and practical.
- Avoid repetitive messages and avoid emoji.
