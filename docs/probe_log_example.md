## Probe Log Example

Example records from `probes.jsonl` (one JSON object per line):

```json
{"probe_id":"probe_1","template_id":"grounding_v1","construct":"grounding","prompt":"State your partner's current intent in one sentence.","actor_id":"agent_A","answer":"Partner intends to share resources.","confidence":0.7,"structured_fields":{"partner_intent":"share_resources"},"timestamp":"2026-03-09T12:00:00Z"}
{"probe_id":"probe_2","template_id":"coordination_v1","construct":"coordination","prompt":"Name the main coordination obstacle right now.","actor_id":"agent_B","answer":null,"confidence":null,"structured_fields":null,"timestamp":"2026-03-09T12:00:01Z"}
```
