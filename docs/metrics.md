## Metrics

This document summarizes run-level and per-agent metrics output in
`metrics.json`. Metrics are derived from event and probe logs to support
reproducible experiment analysis.

### Per-run Metrics
- `steps_taken`, `target_steps`, `efficiency`, `completed` (counter task only)
- `steps_taken`, `target_value`, `current_value`, `increment`, `efficiency`, `completed` (accumulator task only)
- `steps_taken`, `target_steps`, `efficiency`, `completed` (hidden_profile task only)
- `event_count_<event_type>`: total event count per type
- `action_submitted_<action_type>_count`: submitted action count per type
- `messages_sent`: total delivered messages
- `messages_received`: total delivered messages summed across recipients
- `transfers_count`: total resource transfer events
- `transfer_amount_total`: sum of transfer amounts
- `transfer_amount_mean`: mean transfer amount
- `proposal_created`: count of proposals created
- `proposal_responded`: count of proposal responses
- `proposal_response_rate`: responded / created
- `decision_choice_count`: total decision choices revealed
- `probe_records`: number of probe records
- `probe_answered`: probes with non-null answers
- `probe_unanswered`: probes with null answers
- `probe_response_rate`: answered / total
- `probe_confidence_mean`: mean confidence across responses
- `probe_construct_<construct>`: count per construct (e.g., `probe_construct_grounding`)

### Per-agent Metrics
- `messages_sent`: number of delivered messages authored by the agent
- `messages_received`: number of delivered messages addressed to the agent
- `transfers_sent`: number of transfers initiated by the agent
- `transfer_amount_sent`: total amount sent by the agent
- `transfers_received`: number of transfers received by the agent
- `transfer_amount_received`: total amount received by the agent
- `probe_records`: number of probe records for the agent
- `probe_answered`: probes with non-null answers
- `probe_unanswered`: probes with null answers
- `probe_response_rate`: answered / total
- `probe_confidence_mean`: mean confidence across responses
- `probe_construct_<construct>`: count per construct

### Versioning
Metrics schema versioning is tracked via `trace_schema_version` in the run
manifest and logs.

### Example Output
```json
{
  "per_agent": {
    "agent_A": {
      "probe_records": 2,
      "probe_answered": 1,
      "probe_unanswered": 1,
      "probe_response_rate": 0.5,
      "probe_confidence_mean": 0.8,
      "probe_construct_grounding": 1
    }
  },
  "per_run": {
    "steps_taken": 3,
    "target_steps": 3,
    "efficiency": 1.0,
    "completed": 1.0,
    "probe_records": 2,
    "probe_answered": 1,
    "probe_unanswered": 1,
    "probe_response_rate": 0.5,
    "probe_confidence_mean": 0.8,
    "probe_construct_grounding": 1
  }
}
```
