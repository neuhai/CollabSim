## Event Log Example

Example records from `events.jsonl` (one JSON object per line):

```json
{"event_id":"step_1","event_type":"state_updated","timestamp":"2026-03-09T12:00:00Z","actor_id":"system","visibility":"system","payload":{"step_index":1,"state_delta":{"step_index":1},"resulting_state_hash":"abc123"}}
{"event_id":"step_2","event_type":"state_updated","timestamp":"2026-03-09T12:00:01Z","actor_id":"system","visibility":"system","payload":{"step_index":2,"state_delta":{"step_index":2},"resulting_state_hash":"def456"}}
{"event_id":"event_3","event_type":"resource_transferred","timestamp":"2026-03-09T12:00:02Z","actor_id":"agent_A","visibility":"public","payload":{"resource_id":"credits","from":"agent_A","to":"agent_B","amount":2.0,"from_balance":3.0,"to_balance":5.0}}
```
