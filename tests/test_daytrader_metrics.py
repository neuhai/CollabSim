"""Tests for DayTrader task metrics."""

from __future__ import annotations

from pathlib import Path

from analysis.task_metrics import compute_task_metrics
from analysis.trace_parser import Trace


def _daytrader_trace(events: list[dict]) -> Trace:
    return Trace(
        run_dir=Path("/tmp/daytrader_test"),
        events=events,
        manifest={
            "config": {
                "task": {"type": "daytrader", "starting_money": 200.0},
                "agents": [{"id": "A"}, {"id": "B"}],
            }
        },
        summary={
            "per_agent": {
                "A": {"final_balance": 250.0},
                "B": {"final_balance": 230.0},
            }
        },
    )


def test_daytrader_avg_investment_metrics_per_agent_and_run() -> None:
    events = [
        {
            "event_type": "individual_investment_made",
            "actor_id": "A",
            "payload": {"invest_price": 30.0},
        },
        {
            "event_type": "individual_investment_made",
            "actor_id": "A",
            "payload": {"invest_price": 50.0},
        },
        {
            "event_type": "group_investment_contributed",
            "actor_id": "A",
            "payload": {"invest_price": 20.0},
        },
        {
            "event_type": "group_investment_contributed",
            "actor_id": "B",
            "payload": {"invest_price": 40.0},
        },
        {
            "event_type": "observation_built",
            "actor_id": "A",
            "payload": {
                "observation": {
                    "state": {
                        "task_state": {
                            "task_type": "daytrader",
                            "rounds_completed": 2,
                        }
                    }
                }
            },
        },
    ]
    per_run, per_agent = compute_task_metrics(_daytrader_trace(events))

    assert per_agent["A"]["individual_investment_count"] == 2.0
    assert per_agent["A"]["total_individual_invested"] == 80.0
    assert per_agent["A"]["avg_individual_investment_money"] == 40.0
    assert per_agent["A"]["avg_individual_investment_count"] == 1.0
    assert per_agent["A"]["group_investment_count"] == 1.0
    assert per_agent["A"]["total_group_invested"] == 20.0
    assert per_agent["A"]["avg_group_investment_money"] == 20.0
    assert per_agent["A"]["avg_group_investment_count"] == 0.5

    assert per_agent["B"]["group_investment_count"] == 1.0
    assert per_agent["B"]["avg_group_investment_money"] == 40.0
    assert per_agent["B"]["avg_individual_investment_money"] == 0.0

    assert per_run["avg_individual_investment_count_per_agent"] == 0.5
    assert per_run["avg_individual_investment_money_per_agent"] == 20.0
    assert per_run["avg_group_investment_count_per_agent"] == 0.5
    assert per_run["avg_group_investment_money_per_agent"] == 30.0
