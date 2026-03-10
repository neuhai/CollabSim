# Probe Templates

This document provides example probe prompt templates and expected response
fields for the mental state probes. The templates are meant to be stored in
configuration and registered by the probe registry before execution.

## Template Fields
- `template_id`: Unique identifier for the prompt template.
- `construct`: One of `grounding`, `situation_awareness`, or `coordination`.
- `version`: Version tag for the prompt content.
- `prompt`: Natural language question shown to the agent.

## Response Fields
- Required: `answer`
- Optional: `confidence` (0 to 1), `structured_fields`
- Optional metadata: `actor_id` (agent identifier), `template_id`, `construct`

`structured_fields` is an object used for construct-specific slots such as
`partner_intent`, `predicted_action`, or `obstacle`.

Construct-specific guidance (suggested `structured_fields` keys):
- Grounding: `partner_intent`, `shared_goal`, `misalignment`
- Situation awareness: `predicted_action`, `info_gaps`
- Coordination: `obstacle`, `proposed_next_step`

Example `structured_fields` payloads by construct:
```json
{
  "grounding": {"partner_intent": "share_resources", "shared_goal": "maximize_total"},
  "situation_awareness": {"predicted_action": "transfer", "info_gaps": ["inventory_B"]},
  "coordination": {"obstacle": "missing_resource", "proposed_next_step": "request_transfer"}
}
```

Example probe response record (as logged):
```json
{
  "probe_id": "probe_12",
  "template_id": "grounding_v1",
  "construct": "grounding",
  "actor_id": "agent_A",
  "answer": "I believe the partner intends to share resources.",
  "confidence": 0.72,
  "structured_fields": {"partner_intent": "share_resources"},
  "timestamp": "2026-03-09T12:00:00Z"
}
```

## Configuration Example
See `configs/probe_templates.yml` for example templates that align with the
default construct set and minimal response constraints.

## CLI Integration
The CLI loads probe templates from `configs/probe_templates.yml` by default.
Override via `--probe-templates` when running `python -m src.cli`.
