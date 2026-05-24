"""Tests for ShapeFactory task metrics (trade prices on accept events)."""

from __future__ import annotations

from pathlib import Path

from analysis.task_metrics import compute_task_metrics
from analysis.trace_parser import Trace


def _shapefactory_trace(events: list[dict]) -> Trace:
    return Trace(
        run_dir=Path("/tmp/shapefactory_test"),
        events=events,
        manifest={
            "config": {
                "task": {"type": "shapefactory", "starting_money": 200.0},
                "agents": [{"id": "A"}, {"id": "B"}],
            }
        },
        summary={
            "per_agent": {
                "A": {"final_balance": 210.0},
                "B": {"final_balance": 190.0},
            }
        },
    )


def test_shapefactory_trade_price_metrics_from_accept_payload() -> None:
    events = [
        {
            "event_type": "trade_offer_created",
            "actor_id": "A",
            "payload": {
                "id": "offer_1_1",
                "from": "A",
                "to": "B",
                "offer_type": "buy",
                "shape": "square",
                "quantity": 1,
                "price_per_unit": 25.0,
                "status": "pending",
            },
        },
        {
            "event_type": "trade_offer_responded",
            "actor_id": "B",
            "payload": {
                "transaction_id": "offer_1_1",
                "response_type": "accept",
                "initiator_id": "A",
                "target_id": "B",
                "price_per_unit": 25.0,
            },
        },
    ]
    per_run, per_agent = compute_task_metrics(_shapefactory_trace(events))

    assert per_run["avg_trade_price"] == 25.0
    assert per_run["min_trade_price"] == 25.0
    assert per_run["max_trade_price"] == 25.0
    assert per_agent["A"]["avg_trade_price"] == 25.0
    assert per_agent["B"]["avg_trade_price"] == 25.0


def test_shapefactory_messages_per_successful_trade() -> None:
    events = [
        {
            "event_type": "message_delivered",
            "actor_id": "A",
            "payload": {"content": "want to trade"},
        },
        {
            "event_type": "message_delivered",
            "actor_id": "B",
            "payload": {"content": "sure"},
        },
        {
            "event_type": "message_delivered",
            "actor_id": "A",
            "payload": {"content": "thanks"},
        },
        {
            "event_type": "trade_offer_responded",
            "actor_id": "B",
            "payload": {
                "response_type": "accept",
                "initiator_id": "A",
                "target_id": "B",
                "price_per_unit": 25.0,
            },
        },
    ]
    per_run, per_agent = compute_task_metrics(_shapefactory_trace(events))

    assert per_run["messages_sent_total"] == 3.0
    assert per_run["total_successful_trades"] == 1.0
    assert per_run["messages_per_successful_trade"] == 3.0
    assert per_agent["A"]["messages_per_successful_trade"] == 2.0
    assert per_agent["B"]["messages_per_successful_trade"] == 1.0


def test_shapefactory_fulfill_order_aggregates_and_full_completion() -> None:
    trace = Trace(
        run_dir=Path("/tmp/shapefactory_fulfill"),
        events=[],
        manifest={
            "config": {
                "task": {"type": "shapefactory", "starting_money": 200.0, "shapes_order": 3},
                "agents": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
            }
        },
        summary={
            "per_agent": {
                "A": {"final_balance": 300.0, "order_progress": 3},
                "B": {"final_balance": 250.0, "order_progress": 1},
                "C": {"final_balance": 200.0, "order_progress": 2},
            }
        },
    )
    per_run, per_agent = compute_task_metrics(trace)

    assert per_run["agents_fully_fulfilled_own_order_count"] == 1.0
    assert per_run["avg_fulfilled_order_slots_per_agent"] == 2.0
    assert per_run["min_fulfilled_order_slots_per_agent"] == 1.0
    assert per_run["max_fulfilled_order_slots_per_agent"] == 3.0
    assert per_agent["A"]["fulfilled_order_slots"] == 3.0
    assert per_agent["A"]["order_fully_fulfilled"] == 1.0
    assert per_agent["B"]["fulfilled_order_slots"] == 1.0
    assert per_agent["B"]["order_fully_fulfilled"] == 0.0
    assert per_agent["C"]["order_fully_fulfilled"] == 0.0


def test_shapefactory_fulfilled_slots_fallback_from_order_events() -> None:
    events = [
        {"event_type": "order_fulfilled", "actor_id": "A", "payload": {"fulfilled_count": 2}},
        {"event_type": "order_fulfilled", "actor_id": "A", "payload": {"fulfilled_count": 1}},
    ]
    trace = Trace(
        run_dir=Path("/tmp/shapefactory_fulfill_ev"),
        events=events,
        manifest={
            "config": {
                "task": {"type": "shapefactory", "starting_money": 200.0, "shapes_order": 4},
                "agents": [{"id": "A"}],
            }
        },
        summary={"per_agent": {"A": {"final_balance": 220.0}}},
    )
    _, per_agent = compute_task_metrics(trace)
    assert per_agent["A"]["fulfilled_order_slots"] == 3.0
    assert per_agent["A"]["order_fully_fulfilled"] == 0.0
