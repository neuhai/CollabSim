## Observations

Controllers build per-agent observations once per step and cache them in the
controller state. Cached observations are reused within the same step to
avoid repeated visibility filtering and event filtering.

The optional `protocol.visible_event_window` config limits each observation's
`visible_events` to the most recent N events.

The optional `protocol.visible_event_types` config filters observations to only
include matching event types.

The optional `protocol.visible_event_types_exclude` config removes matching
event types after filters are applied.

The optional `protocol.visible_event_types_by_agent` map overrides visible event
types per agent id.

The optional `protocol.visible_event_types_exclude_by_agent` map overrides event
exclusions per agent id.

Event filter precedence:
- Apply per-agent include/exclude filters first when present.
- Apply global include/exclude filters when per-agent filters are absent.

When per-agent includes are set, global excludes are ignored unless per-agent
excludes are provided.

The optional `protocol.visible_event_window_by_agent` map overrides event
windows per agent id.

The optional `protocol.observation_memory_key_limit` config limits the number
of memory keys included in each observation.

The optional `protocol.observation_memory_key_limit_by_agent` map overrides
observation memory limits per agent id.

The optional `protocol.memory_turn_limit` config controls how many recent
context turns are retained in each agent's observation memory (default: 7).

The optional `controls.visibility_defaults` config defines default visible
state fields when no per-agent visibility map is specified.

The optional `controls.visibility_defaults_by_agent` map overrides defaults
per agent id.

The optional `controls.visibility_overrides` config adds additional visible
fields on top of defaults.

The optional `controls.visibility_overrides_by_agent` map overrides visibility
overrides per agent id.

Visibility overrides apply even when a per-agent visibility map is provided.

The optional `controls.visibility_excludes` config removes visible fields after
defaults and overrides are applied.

The optional `controls.visibility_excludes_by_agent` map overrides excludes per
agent id.

Visibility filtering precedence:
1. Base visibility (`visibility_map` if present, else `visibility_defaults`).
2. Apply overrides (`visibility_overrides`, `visibility_overrides_by_agent`).
3. Apply excludes (`visibility_excludes`, `visibility_excludes_by_agent`).

When a visibility map omits a section, defaults are used for that section if
configured.
