## Config Fields

Field-level guidance for experiment configuration.

### experiment
- `id`: unique experiment identifier
- `seed`: random seed for reproducibility
- `max_steps`: maximum controller steps (positive integer; used when controller.run omits max_steps)
- `seeds`: optional map of library-specific seeds

### prompts
- optional prompt path overrides (`system`, `persona`, `task`, `protocol`, `action_space`, `return_format`, `action`, `probe`, `persona_profiles`)

### agents
- `id`: agent identifier
- `role`: agent role label
- `model`: provider/name/decoding params
- `persona_profile`: optional persona string to control agent personality
- `persona_profile_path`: optional path to persona profile prompt

### action_space
- `enabled`: list of enabled action primitives (communicate, decide, propose, respond, transfer)
- `decide.reveal`: default decision reveal strategy (sequential/aggregated/simultaneous)

### controls
- `communication`: communication constraints (`mode`: broadcast/direct, `max_messages_per_turn`: positive integer)
- `information_distribution`: visibility controls (`visibility`: non-empty string)
- `interdependence`: task interdependence structure (`structure`: non-empty string)
- `visibility_map`: per-agent visible state fields by section (shape: `{agent_id: {task_state: [fields], resources: [fields], turn_state: [fields], buffers: [fields]}}`)
- `visibility_defaults`: default visible state fields by section (shape: `{task_state: [fields], resources: [fields], turn_state: [fields], buffers: [fields]}`)
- `visibility_defaults_by_agent`: per-agent default visible fields map (shape: `{agent_id: {task_state: [fields], resources: [fields], turn_state: [fields], buffers: [fields]}}`)
- `visibility_overrides`: additional visible fields applied on top of defaults (shape: `{task_state: [fields], resources: [fields], turn_state: [fields], buffers: [fields]}`)
- `visibility_overrides_by_agent`: per-agent visibility override map (shape: `{agent_id: {task_state: [fields], resources: [fields], turn_state: [fields], buffers: [fields]}}`)
- `visibility_excludes`: excluded visible fields (shape: `{task_state: [fields], resources: [fields], turn_state: [fields], buffers: [fields]}`)
- `visibility_excludes_by_agent`: per-agent excluded visible fields (shape: `{agent_id: {task_state: [fields], resources: [fields], turn_state: [fields], buffers: [fields]}}`)

### task
- `type`: registered task identifier
- task-specific fields documented in `docs/task_schema.md`
- hidden_profile fields: `target_steps`, `shared_facts`, `private_facts`

### protocol
- `turn_taking`: turn-taking strategy (`sequential` or `simultaneous`)
- `termination`: termination settings (`condition`: `max_steps` or `task_complete`)
- `step_mode`: controller stepping (`event` or `tick`)
- `proposal_expiry_steps`: integer step threshold for proposal expiry (relative to creation)
- `decision_timeout_steps`: integer step threshold before decision reveal
- `decision_quorum`: integer number of decisions required for aggregated reveal
- `memory_turn_limit`: number of recent turns to keep in agent memory (default 7)
- `visible_event_window`: integer number of recent events to include per observation
- `visible_event_types`: list of event types included per observation
- `visible_event_types_exclude`: list of event types excluded per observation
- `visible_event_types_by_agent`: per-agent event type filter map
- `visible_event_types_exclude_by_agent`: per-agent excluded event type map (precedence: per-agent includes override global excludes unless per-agent excludes are set)
- `visible_event_window_by_agent`: per-agent event window map
- `agent_memory_key_limit`: max number of agent memory keys to persist per step
- `observation_memory_key_limit`: max number of memory keys included in observations
- `agent_memory_key_limit_by_agent`: per-agent memory key limit map
- `observation_memory_key_limit_by_agent`: per-agent observation memory limit map

### probe
- `cadence`: probe cadence (per_action, per_turn, on_event)
- `templates`: list of probe template ids
- `events`: list of event types to trigger probes (when cadence=on_event)
- `questions_path`: path to JSON list of interview questions
- `questions`: inline list of interview questions

### logging
- `trace_schema_version`: schema version tag
- `output_dir`: output directory for run logs
- `observation_events`: when true, emit observation/action proposal events
