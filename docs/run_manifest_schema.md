## Run Manifest Schema

This schema describes the structure of `run_manifest.json`, which captures
reproducibility metadata for each run.

```json
{
  "type": "object",
  "required": [
    "run_id",
    "config_hash",
    "model_metadata",
    "prompt_versions",
    "timestamp",
    "seeds",
    "trace_schema_version",
    "code_version"
  ],
  "properties": {
    "run_id": {"type": "string"},
    "config_hash": {"type": "string"},
    "model_metadata": {"type": "array", "items": {"type": "object"}},
    "prompt_versions": {"type": "array", "items": {"type": "object"}},
    "timestamp": {"type": "string"},
    "seeds": {"type": "object"},
    "trace_schema_version": {"type": "string"},
    "code_version": {"type": ["string", "null"]}
  },
  "additionalProperties": true
}
```

Notes:
- `model_metadata` entries include agent id, provider, name, version, and decoding params.
- `prompt_versions` entries include template id, version, and prompt hash.
