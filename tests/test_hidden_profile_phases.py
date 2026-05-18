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
                "initial_vote_steps": 3,
                "final_vote_steps": 3,
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
            (3, "initial"),
            (4, "discussion"),
            (50, "discussion"),
            (87, "discussion"),
            (88, "final"),
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

    def test_initial_votes_do_not_force_discussion_end(self) -> None:
        controller = build_controller(
            ExperimentState(agents={}, task_state={}, resources={}, turn_state={}, buffers={}),
            config=_hidden_profile_config(),
        )
        controller.state.step_index = 2
        ts = controller.state.task_state
        participants = ts["participants"]
        for entry in participants.values():
            entry["initial_vote"] = "Candidate A"
        controller._advance_hidden_profile_phase_from_action(
            "C",
            {
                "type": "decide",
                "payload": {"decision_id": "initial_vote", "choice": "Candidate A"},
            },
        )
        self.assertEqual(controller._hidden_profile_phase(), "initial")


if __name__ == "__main__":
    unittest.main()
