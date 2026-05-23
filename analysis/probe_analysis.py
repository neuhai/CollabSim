"""Probe response analysis grouped by actual probe constructs.

Probe templates used across experiments include constructs such as:
    grounding            structured_fields: partner_intent, beliefs
    situation_awareness  structured_fields: partner_intent, beliefs
    coordination         structured_fields: obstacle, next_step

For each construct this module computes:
  - mean confidence and its standard deviation over all responses
  - response rate (answered / total probes)
  - confidence trend: divide the session into thirds and track mean confidence
  - per-agent breakdowns of all of the above
  - structured-field value frequencies (partner_intent, obstacle categories)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.trace_parser import Trace


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = _mean(values)
    assert m is not None
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


def _confidence_trend(probes: list[dict[str, Any]]) -> dict[str, float | None]:
    """Split probes into early / mid / late thirds and return mean confidence."""
    n = len(probes)
    if n == 0:
        return {"confidence_early": None, "confidence_mid": None, "confidence_late": None}
    third = max(1, n // 3)
    early = [float(p["confidence"]) for p in probes[:third] if isinstance(p.get("confidence"), (int, float))]
    mid = [float(p["confidence"]) for p in probes[third: 2 * third] if isinstance(p.get("confidence"), (int, float))]
    late = [float(p["confidence"]) for p in probes[2 * third:] if isinstance(p.get("confidence"), (int, float))]
    return {
        "confidence_early": _mean(early),
        "confidence_mid": _mean(mid),
        "confidence_late": _mean(late),
    }


def _field_frequencies(probes: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for p in probes:
        sf = p.get("structured_fields")
        if not isinstance(sf, dict):
            continue
        val = sf.get(field)
        if isinstance(val, str) and val.strip():
            counts[val.strip().lower()] += 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ------------------------------------------------------------------ #
# Per-construct analysis
# ------------------------------------------------------------------ #

def _analyze_construct(
    probes: list[dict[str, Any]],
    structured_keys: list[str],
) -> dict[str, Any]:
    total = len(probes)
    answered = sum(1 for p in probes if p.get("answer") is not None)
    confidences = [float(p["confidence"]) for p in probes if isinstance(p.get("confidence"), (int, float))]

    result: dict[str, Any] = {
        "probe_count": total,
        "answered": answered,
        "response_rate": answered / total if total > 0 else None,
        "confidence_mean": _mean(confidences),
        "confidence_std": _std(confidences),
    }
    result.update(_confidence_trend(probes))

    for key in structured_keys:
        result[f"field_{key}_frequencies"] = _field_frequencies(probes, key)

    return result


def analyze_probes(trace: Trace) -> dict[str, Any]:
    """Return probe analysis grouped by concrete construct names.

    Returns a dict with keys:
      overall.<construct> — run-level construct metrics
      per_agent           — {agent_id: {<construct>: ...}}
    """
    structured_keys_by_construct: dict[str, list[str]] = {
        "grounding": ["partner_intent", "beliefs"],
        "situation_awareness": ["partner_intent", "beliefs"],
        "coordination": ["obstacle", "next_step"],
    }

    constructs = sorted(
        {
            c
            for c in (p.get("construct") for p in trace.probes)
            if isinstance(c, str) and c.strip()
        }
    )

    overall: dict[str, Any] = {}
    probes_by_construct: dict[str, list[dict[str, Any]]] = {}
    for construct in constructs:
        probes = trace.probes_of_construct(construct)
        probes_by_construct[construct] = probes
        overall[construct] = _analyze_construct(
            probes,
            structured_keys_by_construct.get(construct, []),
        )

    per_agent: dict[str, dict[str, Any]] = {}
    all_ids = {p.get("actor_id") for p in trace.probes if isinstance(p.get("actor_id"), str)}
    for agent_id in sorted(all_ids):
        per_agent[agent_id] = {}
        for construct in constructs:
            agent_probes = [p for p in probes_by_construct[construct] if p.get("actor_id") == agent_id]
            per_agent[agent_id][construct] = _analyze_construct(
                agent_probes,
                structured_keys_by_construct.get(construct, []),
            )

    return {
        "overall": overall,
        "per_agent": per_agent,
    }


def probe_summary_rows(trace: Trace) -> list[dict[str, Any]]:
    """Flatten probe analysis into one row per (agent, construct) for CSV export."""
    analysis = analyze_probes(trace)
    rows: list[dict[str, Any]] = []
    run_id = trace.manifest.get("run_id", str(trace.run_dir.name))
    task_type = trace.task_type or "unknown"

    for construct in sorted(analysis["overall"].keys()):
        overall = analysis["overall"][construct]
        rows.append({
            "run_id": run_id,
            "task_type": task_type,
            "agent_id": "_all_",
            "construct": construct,
            **{k: v for k, v in overall.items() if not isinstance(v, dict)},
        })

    for agent_id, constructs in analysis["per_agent"].items():
        for construct, metrics in constructs.items():
            rows.append({
                "run_id": run_id,
                "task_type": task_type,
                "agent_id": agent_id,
                "construct": construct,
                **{k: v for k, v in metrics.items() if not isinstance(v, dict)},
            })

    return rows
