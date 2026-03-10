## Metrics Schema

This schema describes the structure of `metrics.json`. The metrics payload is
split into per-agent and per-run summaries.

```json
{
  "type": "object",
  "required": ["per_agent", "per_run"],
  "properties": {
    "per_agent": {"type": "object"},
    "per_run": {"type": "object"}
  },
  "additionalProperties": true
}
```

Notes:
- Metric keys are dynamic and depend on task type and enabled probes.
- See `docs/metrics.md` for field definitions and examples.
