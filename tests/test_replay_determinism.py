"""Deterministic replay tests for controller event logs."""

from __future__ import annotations

import copy
import unittest

from src.controller.controller import ExperimentState, compute_state_hash
from src.controller.factory import build_controller


def _base_config() -> dict[str, object]:
    return {
        "experiment": {"id": "replay_counter", "seed": 17},
        "agents": [
            {
                "id": "A",
                "role": "tester",
                "model": {"provider": "local", "name": "dummy"},
            }
        ],
        "action_space": {"enabled": ["decide"]},
        "controls": {},
        "task": {"type": "counter", "target_steps": 3},
        "protocol": {"turn_taking": "sequential", "termination": {"condition": "max_steps"}},
        "probe": {"cadence": "per_action"},
        "logging": {"trace_schema_version": "v0"},
    }


def _base_state() -> ExperimentState:
    return ExperimentState(
        agents={},
        task_state={},
        resources={},
        turn_state={},
        buffers={},
    )


def _task_config(
    task_type: str,
    *,
    enabled_actions: list[str],
    task_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    task_cfg: dict[str, object] = {"type": task_type}
    if task_fields:
        task_cfg.update(task_fields)
    return {
        "experiment": {"id": f"replay_{task_type}", "seed": 11, "max_steps": 5},
        "agents": [
            {"id": "A", "role": "tester", "model": {"provider": "local", "name": "dummy"}},
            {"id": "B", "role": "tester", "model": {"provider": "local", "name": "dummy"}},
        ],
        "action_space": {"enabled": enabled_actions},
        "controls": {},
        "task": task_cfg,
        "protocol": {"turn_taking": "simultaneous", "step_mode": "event", "termination": {"condition": "max_steps"}},
        "probe": {"cadence": "per_action"},
        "logging": {"trace_schema_version": "v0"},
    }


def _normalize_event(event: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(event)
    normalized.pop("timestamp", None)
    return normalized


class DeterministicReplayTests(unittest.TestCase):
    """Validate deterministic event log replay for identical configs."""

    def test_counter_replay_event_log_stability(self) -> None:
        config = _base_config()
        controller_a = build_controller(_base_state(), config=config)
        controller_b = build_controller(_base_state(), config=config)

        state_a = controller_a.run(max_steps=5)
        state_b = controller_b.run(max_steps=5)

        log_a = [_normalize_event(event) for event in state_a.event_log]
        log_b = [_normalize_event(event) for event in state_b.event_log]

        self.assertEqual(log_a, log_b)
        self.assertEqual(state_a.task_state, state_b.task_state)

    def test_state_hash_matches_snapshot(self) -> None:
        config = _base_config()
        controller = build_controller(_base_state(), config=config)

        for _ in range(3):
            controller.step()
            last_event = controller.state.event_log[-1]
            payload = last_event.get("payload", {})
            self.assertEqual(payload.get("resulting_state_hash"), compute_state_hash(controller.state))

    def test_shapefactory_action_replay_stability(self) -> None:
        config = _task_config(
            "shapefactory",
            enabled_actions=["produce_shape"],
            task_fields={"target_steps": 3, "shape_options": ["circle", "square"], "specialties": {"A": "circle", "B": "square"}},
        )
        controller_a = build_controller(_base_state(), config=config)
        controller_b = build_controller(_base_state(), config=config)
        produce_action = {
            "type": "produce_shape",
            "actor_id": "A",
            "timestamp": 1,
            "payload": {"shape": "circle", "quantity": 1},
        }
        controller_a.state.pending_actions.append(copy.deepcopy(produce_action))
        controller_b.state.pending_actions.append(copy.deepcopy(produce_action))
        controller_a.step()
        controller_b.step()
        log_a = [_normalize_event(event) for event in controller_a.state.event_log]
        log_b = [_normalize_event(event) for event in controller_b.state.event_log]
        self.assertEqual(log_a, log_b)
        a_money = controller_a.state.task_state["participants"]["A"]["money"]
        b_money = controller_b.state.task_state["participants"]["A"]["money"]
        self.assertEqual(a_money, b_money)
        self.assertLess(a_money, 200.0)

    def test_daytrader_action_replay_stability(self) -> None:
        config = _task_config(
            "daytrader",
            enabled_actions=["make_investment"],
            task_fields={"target_steps": 3, "starting_money": 200},
        )
        controller_a = build_controller(_base_state(), config=config)
        controller_b = build_controller(_base_state(), config=config)
        invest_action = {
            "type": "make_investment",
            "actor_id": "A",
            "timestamp": 1,
            "payload": {"invest_price": 20, "invest_decision_type": "individual"},
        }
        controller_a.state.pending_actions.append(copy.deepcopy(invest_action))
        controller_b.state.pending_actions.append(copy.deepcopy(invest_action))
        controller_a.step()
        controller_b.step()
        log_a = [_normalize_event(event) for event in controller_a.state.event_log]
        log_b = [_normalize_event(event) for event in controller_b.state.event_log]
        self.assertEqual(log_a, log_b)
        history = controller_a.state.task_state["participants"]["A"]["investment_history"]
        self.assertEqual(len(history), 1)

    def test_maptask_action_replay_stability(self) -> None:
        config = _task_config(
            "maptask",
            enabled_actions=["update_map_progress"],
            task_fields={"target_steps": 3, "roles": {"A": "follower", "B": "guider"}},
        )
        controller_a = build_controller(_base_state(), config=config)
        controller_b = build_controller(_base_state(), config=config)
        progress_action = {
            "type": "update_map_progress",
            "actor_id": "A",
            "timestamp": 1,
            "payload": {"map_progress": {"x": 12, "y": 5}},
        }
        controller_a.state.pending_actions.append(copy.deepcopy(progress_action))
        controller_b.state.pending_actions.append(copy.deepcopy(progress_action))
        controller_a.step()
        controller_b.step()
        log_a = [_normalize_event(event) for event in controller_a.state.event_log]
        log_b = [_normalize_event(event) for event in controller_b.state.event_log]
        self.assertEqual(log_a, log_b)
        progress = controller_a.state.task_state["participants"]["A"]["map_progress"]
        self.assertEqual(progress.get("x"), 12)
        self.assertEqual(progress.get("y"), 5)


if __name__ == "__main__":
    unittest.main()
