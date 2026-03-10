## Probe Response Schema

This schema describes the expected shape of probe response records stored in
`probes.jsonl`. The schema is minimal and intended to be extended per study.

```json
{
  "type": "object",
  "required": ["probe_id", "answer", "timestamp"],
  "properties": {
    "probe_id": {"type": "string", "minLength": 1},
    "template_id": {"type": "string"},
    "construct": {"type": "string"},
    "actor_id": {"type": "string"},
    "answer": {},
    "prompt": {"type": "string"},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "structured_fields": {"type": "object"},
    "timestamp": {"type": "string"}
  },
  "additionalProperties": true
}
```

Notes:
- `answer` is intentionally untyped to allow string or structured responses.
- `structured_fields` is reserved for construct-specific data.
