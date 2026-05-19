"""Hidden-profile step-scheduled phase tests."""

from __future__ import annotations

import unittest

from src.controller.controller import ExperimentState
from src.controller.factory import build_controller


def _hidden_profile_config(*, max_steps: int = 90) -> dict[str, object]:
    return {
        "experiment": {"id": "hp_phases", "seed": 1, "max_steps": max_steps},
        "agents": [
            {"id": "A", "role": "planner", "model": {"provider": "local", "name": "dummy"}},
            {"id": "B", "role": "analyst", "model": {"provider": "local", "name": "dummy"}},
            {"id": "C", "role": "planner", "model": {"provider": "local", "name": "dummy"}},
        ],
        "action_space": {"enabled": ["message", "decide", "do_nothing"]},
        "controls": {"communication": {"mode": "broadcast"}},
        "task": {
            "type": "hidden_profile",
            "target_steps": 3,
            "phase_rules": {
                "initial_vote_steps": 1,
                "final_vote_steps": 1,
            },
            "shared_facts": ["shared"],
            "private_facts": {"A": ["private-a"], "B": ["private-b"], "C": ["private-c"]},
        },
        "protocol": {
            "turn_taking": "simultaneous",
            "step_mode": "step",
            "termination": {"condition": "task_complete"},
        },
        "probe": {"cadence": "per_action"},
        "logging": {"trace_schema_version": "v0"},
    }


class HiddenProfilePhaseTests(unittest.TestCase):
    def test_phase_follows_step_index_schedule(self) -> None:
        controller = build_controller(
            ExperimentState(agents={}, task_state={}, resources={}, turn_state={}, buffers={}),
            config=_hidden_profile_config(),
        )
        for step, expected in [
            (1, "initial"),
            (2, "discussion"),
            (50, "discussion"),
            (89, "discussion"),
            (90, "final"),
        ]:
            controller.state.step_index = step
            self.assertEqual(controller._hidden_profile_phase(), expected, msg=f"step={step}")

    def test_discussion_messages_do_not_change_phase(self) -> None:
        controller = build_controller(
            ExperimentState(agents={}, task_state={}, resources={}, turn_state={}, buffers={}),
            config=_hidden_profile_config(),
        )
        controller.state.step_index = 10
        controller._advance_hidden_profile_phase_from_action(
            "A",
            {
                "type": "message",
                "payload": {"channel": "broadcast", "content": "Thoughts on Candidate C?"},
            },
        )
        self.assertEqual(controller._hidden_profile_phase(), "discussion")
        ts = controller.state.task_state
        self.assertFalse(ts.get("discussion_force_final"))

    def test_forced_final_vote_overrides_step_schedule(self) -> None:
        controller = build_controller(
            ExperimentState(agents={}, task_state={}, resources={}, turn_state={}, buffers={}),
            config=_hidden_profile_config(),
        )
        controller.state.step_index = 20
        ts = controller.state.task_state
        ts["forced_final_vote"] = True
        self.assertEqual(controller._hidden_profile_phase(), "final")

    def test_initial_vote_uses_action_timestamp_when_applied_next_step(self) -> None:
        controller = build_controller(
            ExperimentState(agents={}, task_state={}, resources={}, turn_state={}, buffers={}),
            config=_hidden_profile_config(),
        )
        controller.state.step_index = 2
        self.assertEqual(controller._hidden_profile_phase(), "discussion")
        vote_action = {
            "type": "decide",
            "timestamp": 1,
            "payload": {"decision_id": "initial_vote", "choice": "Candidate A"},
        }
        self.assertIsNone(controller._check_hidden_profile_phase_preconditions(vote_action, "A"))
        self.assertIsNotNone(
            controller._check_hidden_profile_phase_preconditions(
                {
                    "type": "decide",
                    "timestamp": 2,
                    "payload": {"decision_id": "initial_vote", "choice": "Candidate A"},
                },
                "A",
            )
        )


if __name__ == "__main__":
    unittest.main()
