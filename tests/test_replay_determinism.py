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
        "protocol": {"turn_taking": "sequential"},
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


if __name__ == "__main__":
    unittest.main()
