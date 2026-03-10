## Event Log Schema

This document captures the expected structure of event records stored in
`events.jsonl`. It mirrors the event taxonomy used by the controller and is
intended for downstream analysis.

```json
{
  "type": "object",
  "required": ["event_id", "event_type", "timestamp", "actor_id", "visibility", "payload"],
  "properties": {
    "event_id": {"type": "string", "minLength": 1},
    "event_type": {"type": "string"},
    "timestamp": {"type": "string"},
    "actor_id": {"type": "string"},
    "visibility": {"type": "string"},
    "payload": {"type": "object"},
    "meta": {"type": "object"}
  },
  "additionalProperties": true
}
```

Notes:
- `payload` is event-type specific and may be extended per task.
- `meta` is reserved for schema versions, trace ids, and additional metadata.
