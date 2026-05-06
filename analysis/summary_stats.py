"""Flatten run_summary.json into CSV rows for statistical analysis.

Produces one row per agent per run with:
  - final_balance (daytrader / shapefactory)
  - probe confidence mean / std / count, per-construct breakdown
  - probe answer text per construct (JSON string)
  - task-specific fields per experiment type
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

        # Per-construct confidence and answer collection
        construct_confidences: dict[str, list[float]] = defaultdict(list)
        answers_by_construct: dict[str, list[str | None]] = defaultdict(list)
        for p in probe_responses:
            c = p.get("construct") or "other"
            conf = p.get("confidence")
            if isinstance(conf, (int, float)):
                construct_confidences[c].append(float(conf))
            answers_by_construct[c].append(p.get("answer"))

        row: dict[str, Any] = {
            "run_id": run_id,
            "task_type": task_type,
            "agent_id": agent_id,
            # task-agnostic financial / performance
            "final_balance": agent_data.get("final_balance"),
            # probe stats
            "probe_count": agent_data.get("probe_count", len(probe_responses)),
            "confidence_mean": _mean(all_confidences),
            "confidence_std": _std(all_confidences),
            # run-level completion context
            "steps_taken": task_summary.get("steps_taken"),
            "target_steps": task_summary.get("target_steps"),
            "run_complete": summary.get("complete"),
        }

        # Per-construct confidence mean and probe count
        for construct, confs in construct_confidences.items():
            row[f"confidence_mean_{construct}"] = _mean(confs)
            row[f"probe_count_{construct}"] = len(confs)

        # Probe answers per construct as JSON blob
        for construct, answers in answers_by_construct.items():
            row[f"answers_{construct}"] = json.dumps(answers, ensure_ascii=False)

        # ---- task-specific fields ----

        if task_type == "hidden_profile":
            initial_votes = task_summary.get("initial_votes") or {}
            final_votes = task_summary.get("final_votes") or {}
            row["initial_vote"] = initial_votes.get(agent_id)
            row["final_vote"] = final_votes.get(agent_id)
            iv, fv = initial_votes.get(agent_id), final_votes.get(agent_id)
            row["vote_changed"] = 1 if (iv and fv and iv != fv) else 0
            row["consensus_reached"] = int(bool(task_summary.get("consensus_reached")))
            row["phase_at_end"] = task_summary.get("phase")

        elif task_type == "daytrader":
            row["rounds_completed"] = task_summary.get("rounds_completed")
            row["target_rounds"] = task_summary.get("target_rounds")
            row["starting_money"] = task_summary.get("starting_money")
            histories = (task_summary.get("investment_histories") or {}).get(agent_id) or []
            ind_list = [h for h in histories if isinstance(h, dict) and h.get("investment_type") == "individual"]
            grp_list = [h for h in histories if isinstance(h, dict) and h.get("investment_type") == "group"]
            total_ind = sum(float(h.get("investment_amount", 0)) for h in ind_list)
            total_grp = sum(float(h.get("investment_amount", 0)) for h in grp_list)
            row["total_individual_invested"] = total_ind
            row["total_group_invested"] = total_grp
            row["individual_investment_count"] = len(ind_list)
            row["group_investment_count"] = len(grp_list)
            total_inv = total_ind + total_grp
            row["group_investment_rate"] = total_grp / total_inv if total_inv > 0 else None

        elif task_type == "shapefactory":
            row["specialty"] = agent_data.get("specialty")
            row["production_number"] = agent_data.get("production_number")
            row["order_progress"] = agent_data.get("order_progress")
            row["starting_money"] = task_summary.get("starting_money")
            row["completed_trades_run"] = task_summary.get("completed_trades")

        elif task_type == "maptask":
            row["role"] = agent_data.get("role")
            row["map_progress_updates"] = agent_data.get("map_progress_updates")
            row["drawn_points_count"] = agent_data.get("drawn_points_count")
            row["route_score"] = task_summary.get("route_score")
            row["route_score_max"] = task_summary.get("route_score_max")
            row["route_similarity"] = task_summary.get("route_similarity")

        rows.append(row)

    return rows
