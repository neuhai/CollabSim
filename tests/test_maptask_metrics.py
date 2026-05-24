"""Tests for MapTask metrics (score_board-based final and step-wise scores)."""

from __future__ import annotations

from pathlib import Path

from analysis.task_metrics import compute_task_metrics
from analysis.trace_parser import Trace


def _maptask_trace(events: list[dict], summary: dict | None = None, *, roles: dict[str, str] | None = None) -> Trace:
    manifest: dict = {
        "config": {
            "task": {"type": "maptask"},
            "agents": [{"id": "A"}, {"id": "B"}],
        }
    }
    if roles:
        manifest["config"]["task"]["roles"] = roles
    return Trace(
        run_dir=Path("/tmp/maptask_test"),
        events=events,
        manifest=manifest,
        summary=summary or {},
    )


def test_maptask_metrics_final_and_stepwise_from_events() -> None:
    events = [
        {
            "event_type": "map_progress_updated",
            "actor_id": "B",
            "step": 1,
            "payload": {
                "drawing_accuracy": {
                    "score_board_sum_drawn_cells": 3,
                    "max_route_score_board_sum": 300,
                    "ratio_vs_ground_truth_route": 0.01,
                    "drawn_cell_count": 1,
                    "route_cells_hit_count": 1,
                },
            },
        },
        {
            "event_type": "map_progress_updated",
            "actor_id": "B",
            "step": 5,
            "payload": {
                "drawing_accuracy": {
                    "score_board_sum_drawn_cells": 144,
                    "max_route_score_board_sum": 300,
                    "ratio_vs_ground_truth_route": 0.48,
                    "drawn_cell_count": 80,
                    "route_cells_hit_count": 40,
                },
            },
        },
        {
            "event_type": "message_delivered",
            "actor_id": "A",
            "payload": {"content": "go north"},
        },
    ]
    summary = {
        "task_summary": {
            "route_score": 144.0,
            "route_score_max": 300.0,
            "route_similarity": 0.48,
        },
    }
    per_run, per_agent = compute_task_metrics(_maptask_trace(events, summary))

    assert per_run["route_score"] == 144.0
    assert per_run["route_score_max"] == 300.0
    assert per_run["route_similarity"] == 0.48
    assert per_run["follower_accuracy"] == 0.48
    assert per_run["drawing_score_step_count"] == 2.0
    assert per_run["drawing_score_peak_ratio"] == 0.48
    assert per_run["drawing_score_final_ratio"] == 0.48
    assert len(per_run["drawing_score_steps"]) == 2
    assert per_run["drawing_score_final"]["score_board_sum_drawn_cells"] == 144
    assert per_agent["B"]["map_progress_updates"] == 2.0


def test_maptask_metrics_fallback_to_steps_without_summary() -> None:
    events = [
        {
            "event_type": "map_progress_updated",
            "actor_id": "B",
            "payload": {
                "drawing_accuracy": {
                    "score_board_sum_drawn_cells": 9,
                    "max_route_score_board_sum": 90,
                    "ratio_vs_ground_truth_route": 0.1,
                    "drawn_cell_count": 3,
                    "route_cells_hit_count": 2,
                },
            },
        },
    ]
    per_run, _ = compute_task_metrics(_maptask_trace(events))
    assert per_run["route_score"] == 9.0
    assert per_run["route_score_max"] == 90.0
    assert per_run["route_similarity"] == 0.1


def test_maptask_revision_rate_from_follower_drawing_actions() -> None:
    events = [
        {
            "event_type": "action_validated",
            "actor_id": "B",
            "payload": {"action": {"type": "draw"}},
        },
        {
            "event_type": "action_validated",
            "actor_id": "B",
            "payload": {"action": {"type": "draw"}},
        },
        {
            "event_type": "action_validated",
            "actor_id": "B",
            "payload": {"action": {"type": "draw"}},
        },
        {
            "event_type": "action_validated",
            "actor_id": "B",
            "payload": {"action": {"type": "erase"}},
        },
        {
            "event_type": "action_validated",
            "actor_id": "B",
            "payload": {"action": {"type": "undo"}},
        },
        {
            "event_type": "action_validated",
            "actor_id": "A",
            "payload": {"action": {"type": "message"}},
        },
    ]
    per_run, per_agent = compute_task_metrics(
        _maptask_trace(events, roles={"A": "guider", "B": "follower"}),
    )

    assert per_run["follower_action_draw_count"] == 3.0
    assert per_run["follower_action_erase_count"] == 1.0
    assert per_run["follower_action_undo_count"] == 1.0
    assert per_run["follower_action_reset_count"] == 0.0
    assert per_run["revision_rate"] == 0.4
    assert per_agent["B"]["revision_rate"] == 0.4
    assert per_agent["B"]["action_draw_count"] == 3.0
    assert "revision_rate" not in per_agent["A"]
