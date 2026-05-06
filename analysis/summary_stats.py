"""Flatten run_summary.json into CSV rows for statistical analysis.

Produces one row per agent per run with:
  - final_balance / final_wealth
  - probe confidence mean / std / count
  - per-construct confidence breakdown (grounding, coordination)
  - probe answer text for each question (serialised as JSON string)
  - task-specific fields (votes for hidden_profile, investment totals for daytrader)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

from analysis.trace_parser import Trace


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = _mean(values)
    assert m is not None
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def summary_stat_rows(trace: Trace) -> list[dict[str, Any]]:
    """Return one row per agent from run_summary.json, or empty list if unavailable."""
    summary = trace.summary
    if not summary:
        return []

    run_id = summary.get("run_id") or str(trace.run_dir.name)
    task_type = summary.get("task_type", "unknown")
    task_summary = summary.get("task_summary") or {}
    per_agent_summary = summary.get("per_agent") or {}

    rows: list[dict[str, Any]] = []

    for agent_id, agent_data in per_agent_summary.items():
        if not isinstance(agent_data, dict):
            continue

        probe_responses: list[dict[str, Any]] = agent_data.get("probe_responses") or []

        # Overall confidence stats
        all_confidences = [
            float(p["confidence"])
            for p in probe_responses
            if isinstance(p.get("confidence"), (int, float))
        ]

        # Per-construct confidence
        construct_confidences: dict[str, list[float]] = defaultdict(list)
        for p in probe_responses:
            c = p.get("construct")
            conf = p.get("confidence")
            if isinstance(c, str) and isinstance(conf, (int, float)):
                construct_confidences[c].append(float(conf))

        # Probe answers as a flat JSON blob
        answers_by_construct: dict[str, list[str | None]] = defaultdict(list)
        for p in probe_responses:
            c = p.get("construct") or "other"
            answers_by_construct[c].append(p.get("answer"))

        row: dict[str, Any] = {
            "run_id": run_id,
            "task_type": task_type,
            "agent_id": agent_id,
            "final_balance": agent_data.get("final_balance"),
            "probe_count": agent_data.get("probe_count", len(probe_responses)),
            "confidence_mean": _mean(all_confidences),
            "confidence_std": _std(all_confidences),
        }

        # Per-construct confidence mean
        for construct, confs in construct_confidences.items():
            row[f"confidence_mean_{construct}"] = _mean(confs)
            row[f"probe_count_{construct}"] = len(confs)

        # Serialise probe answers per construct
        for construct, answers in answers_by_construct.items():
            row[f"answers_{construct}"] = json.dumps(answers, ensure_ascii=False)

        # Task-specific fields
        if task_type == "hidden_profile":
            initial_votes = task_summary.get("initial_votes") or {}
            final_votes = task_summary.get("final_votes") or {}
            row["initial_vote"] = initial_votes.get(agent_id)
            row["final_vote"] = final_votes.get(agent_id)
            iv = initial_votes.get(agent_id)
            fv = final_votes.get(agent_id)
            row["vote_changed"] = 1 if (iv and fv and iv != fv) else 0

        if task_type == "daytrader":
            histories = (task_summary.get("investment_histories") or {}).get(agent_id) or []
            total_individual = sum(
                h.get("invest_price", 0)
                for h in histories
                if isinstance(h, dict) and h.get("investment_type") == "individual"
            )
            total_group = sum(
                h.get("invest_price", 0)
                for h in histories
                if isinstance(h, dict) and h.get("investment_type") == "group"
            )
            row["total_individual_invested"] = total_individual
            row["total_group_invested"] = total_group
            row["investment_count"] = len(histories)
            row["group_investment_rate"] = (
                total_group / (total_individual + total_group)
                if (total_individual + total_group) > 0
                else None
            )

        rows.append(row)

    return rows
