## Event Types

This document lists the core event types emitted by the controller loop. Event
payloads are defined by the controller and task implementations.

Core events:
- `action_submitted`: raw action received from an agent
- `action_validated`: action passed schema and precondition checks
- `action_rejected`: action failed validation
- `context_update`: agent proposal (pre-validation)
- `observation_built`: per-agent observation snapshot
- `state_updated`: task or controller state updated
- `message_delivered`: communication delivered to recipients
- `decision_buffered`: decision stored pending reveal
- `decision_revealed`: decision revealed to eligible agents
- `decision_timed_out`: decision reveal forced after timeout
- `proposal_created`: proposal recorded in controller buffers
- `proposal_responded`: response recorded against a proposal
- `proposal_expired`: proposal removed after expiry threshold
- `resource_transferred`: resource ledger updated from transfer action
- `probe_asked`: interviewer probe issued
- `probe_answered`: probe response recorded

Payload highlights:
- `message_delivered`: `message_id`, `recipients`, `channel`, `content`, `content_type`
- `decision_timed_out`: `decision_id`
- `proposal_created`: `proposal_id`, `target_ids`, `terms`, `expires_at`
- `proposal_responded`: `proposal_id`, `response`, `counter_terms`
- `proposal_expired`: `proposal_id`
- `resource_transferred`: `resource_id`, `from`, `to`, `amount`
- `context_update`: `action`, `rationale`, `alternatives`
- `observation_built`: `observation` (state, visible_events, memory), `context` (persona, protocol, actions)
- `probe_asked`: `probe_id`, `template_id`, `construct`, `prompt`

Notes:
- Tasks may emit additional custom events; document them alongside task specs.
- Event visibility rules: `public` visible to all agents; `private` visible to `payload.recipients`; `system` logs only.
- `context_update` and `observation_built` are emitted only when `logging.observation_events` is true.
- Proposal resolution: `accept`/`reject` responses remove the proposal from buffers; `counter` keeps it active.
- Proposal expiry: proposals expire when `step_index >= expires_at`; controller emits `proposal_expired`.
- Decision timeouts: decisions reveal when `step_index >= created_step + decision_timeout_steps` and emit `decision_timed_out`.
- Decision quorum: aggregated reveals can trigger once `decision_quorum` submissions are reached.
- Decision ordering: decision choice lists are sorted by `actor_id` for deterministic logs.
