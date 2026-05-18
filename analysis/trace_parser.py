"""Load and index experiment trace files from a run directory.

A run directory contains:
  events.jsonl   — one JSON object per line, each an emitted event
  probes.jsonl   — one JSON object per line, each a probe response record
  actions.jsonl  — one JSON object per line, each an action submission record
  metrics.json   — final MetricsResult snapshot (optional, may not exist)
  run_manifest.json — run metadata (config hash, model info, seed, …)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.logging import read_jsonl


@dataclass
class Trace:
    """All records from a single experiment run."""

    run_dir: Path
    events: list[dict[str, Any]] = field(default_factory=list)
    probes: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Derived convenience views (populated by load())
    # ------------------------------------------------------------------ #
    @property
    def task_type(self) -> str | None:
        config = self.manifest.get("config") or {}
        task = config.get("task") or {}
        t = task.get("type")
        if isinstance(t, str) and t:
            return t
        summary_type = self.summary.get("task_type")
        if isinstance(summary_type, str) and summary_type:
            return summary_type
        return None

    @property
    def agent_ids(self) -> list[str]:
        config = self.manifest.get("config") or {}
        agents = config.get("agents") or []
        return [a["id"] for a in agents if isinstance(a, dict) and "id" in a]

    def events_of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event_type") == event_type]

    def probes_of_construct(self, construct: str) -> list[dict[str, Any]]:
        return [p for p in self.probes if p.get("construct") == construct]

    def events_by_actor(self, actor_id: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("actor_id") == actor_id]

    def probes_by_actor(self, actor_id: str) -> list[dict[str, Any]]:
        return [p for p in self.probes if p.get("actor_id") == actor_id]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            obj = json.load(fh)
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def load_trace(run_dir: str | Path) -> Trace:
    """Load all trace files from *run_dir* and return a :class:`Trace`."""
    root = Path(run_dir)
    trace = Trace(run_dir=root)
    trace.events = read_jsonl(root / "events.jsonl")
    trace.probes = read_jsonl(root / "probes.jsonl")
    trace.actions = read_jsonl(root / "actions.jsonl")
    trace.manifest = _read_json(root / "run_manifest.json")
    trace.summary = _read_json(root / "run_summary.json")
    return trace


def find_run_dirs(experiments_root: str | Path) -> list[Path]:
    """Return all subdirectories under *experiments_root* that look like run dirs."""
    root = Path(experiments_root)
    dirs: list[Path] = []
    for candidate in sorted(root.rglob("run_manifest.json")):
        dirs.append(candidate.parent)
    return dirs
