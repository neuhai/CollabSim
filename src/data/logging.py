"""Data pipeline logging utilities."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Iterable

from src.probe.registry import ProbeRegistry

@dataclass(frozen=True)
class RunPaths:
    """Filesystem locations for a single run's outputs."""

    run_dir: Path

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def probes_path(self) -> Path:
        return self.run_dir / "probes.jsonl"

    @property
    def probe_raw_path(self) -> Path:
        return self.run_dir / "probe_raw.jsonl"

    @property
    def actions_path(self) -> Path:
        return self.run_dir / "actions.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "run_summary.json"

    @property
    def trace_path(self) -> Path:
        return self.run_dir / "trace.jsonl"


def init_run_dir(run_dir: str | Path) -> RunPaths:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_dir=path)


def append_trace(path: Path, record: dict[str, Any]) -> None:
    """Append one agent-turn trace record to trace.jsonl."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=2))
        handle.write("\n")


def append_jsonl(path: Path, records: Iterable[dict[str, Any]], *, indent: int | None = 2) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, indent=indent))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records (compact one-line or pretty-printed multi-line per record).

    Also accepts a single pretty-printed JSON array (as written by write_events_json).
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    records: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    while idx < length:
        while idx < length and text[idx] in " \t\r\n":
            idx += 1
        if idx >= length:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            records.append(obj)
        elif isinstance(obj, list):
            records.extend(item for item in obj if isinstance(item, dict))
            break
        idx = end
    return records


def write_events_json(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write all events as a single pretty-printed JSON array.

    This is used instead of JSONL when a structured JSON file is preferred.
    """
    payload = list(records)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def log_probe_records(paths: RunPaths, records: Iterable[dict[str, Any]]) -> None:
    """Append probe records to the probes.jsonl file."""

    append_jsonl(paths.probes_path, records)


def log_probe_raw_records(paths: RunPaths, records: Iterable[dict[str, Any]]) -> None:
    """Append probe debug records (model raw text + parse outcome) to probe_raw.jsonl."""

    append_jsonl(paths.probe_raw_path, records, indent=None)


def write_run_manifest(paths: RunPaths, manifest: dict[str, Any]) -> None:
    """Write the run manifest for reproducibility metadata."""

    write_json(paths.manifest_path, manifest)


def build_run_manifest(
    config: dict[str, Any],
    run_id: str,
    probe_registry: ProbeRegistry | None = None,
) -> dict[str, Any]:
    """Build a minimal run manifest from config and run id."""

    config_bytes = json.dumps(config, sort_keys=True).encode("utf-8")
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    model_meta = _extract_model_metadata(config)
    seed_value = config.get("experiment", {}).get("seed")
    seeds = _extract_seeds(config, seed_value)
    trace_schema_version = config.get("logging", {}).get("trace_schema_version")
    enabled_templates = _extract_enabled_templates(config)
    return {
        "run_id": run_id,
        "config_hash": config_hash,
        "model_metadata": model_meta,
        "prompt_versions": _extract_prompt_versions(probe_registry, enabled_templates),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "trace_schema_version": trace_schema_version,
        "code_version": _get_code_version(),
    }


def _extract_model_metadata(config: dict[str, Any]) -> list[dict[str, Any]]:
    agents = config.get("agents", [])
    results: list[dict[str, Any]] = []
    for agent in agents:
        model = agent.get("model", {})
        results.append(
            {
                "agent_id": agent.get("id"),
                "provider": model.get("provider"),
                "name": model.get("name"),
                "version": model.get("version"),
                "temperature": model.get("temperature"),
                "top_p": model.get("top_p"),
                "max_tokens": model.get("max_tokens"),
            }
        )
    return results


def _extract_prompt_versions(
    probe_registry: ProbeRegistry | None,
    enabled_templates: set[str],
) -> list[dict[str, Any]]:
    if probe_registry is None:
        return []
    prompts: list[dict[str, Any]] = []
    for template in probe_registry.list():
        if enabled_templates and template.template_id not in enabled_templates:
            continue
        prompt_hash = hashlib.sha256(template.prompt.encode("utf-8")).hexdigest()
        prompts.append(
            {
                "template_id": template.template_id,
                "version": template.version,
                "prompt_hash": prompt_hash,
            }
        )
    return prompts


def _extract_enabled_templates(config: dict[str, Any]) -> set[str]:
    probe_cfg = config.get("probe", {})
    if not isinstance(probe_cfg, dict):
        return set()
    templates = probe_cfg.get("templates", [])
    if not isinstance(templates, list):
        return set()
    return {item for item in templates if isinstance(item, str) and item}


def _extract_seeds(config: dict[str, Any], random_seed: Any) -> dict[str, Any]:
    seeds: dict[str, Any] = {}
    if random_seed is not None:
        seeds["random"] = random_seed
    experiment_cfg = config.get("experiment", {})
    if not isinstance(experiment_cfg, dict):
        return seeds
    extra_seeds = experiment_cfg.get("seeds")
    if isinstance(extra_seeds, dict):
        for key, value in extra_seeds.items():
            if isinstance(key, str) and key:
                seeds[key] = value
    return seeds


def _get_code_version() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def write_metrics(paths: RunPaths, metrics: dict[str, Any]) -> None:
    """Write run-level metrics to metrics.json."""

    write_json(paths.metrics_path, metrics)


def write_run_summary(paths: RunPaths, summary: dict[str, Any]) -> None:
    """Write the human-readable per-agent run summary to run_summary.json."""

    write_json(paths.summary_path, summary)
