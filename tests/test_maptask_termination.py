"""MapTask termination: finish cell, step budget, and controller early exit."""

from __future__ import annotations

import unittest

from src.controller.controller import ExperimentState
from src.controller.factory import build_controller
from src.tasks.maptask import maptask_init_state, maptask_step


def _empty_state() -> ExperimentState:
    return ExperimentState(agents={}, task_state={}, resources={}, turn_state={}, buffers={})


def _maptask_config(*, termination: str = "task_complete", max_steps: int = 120) -> dict[str, object]:
    return {
        "experiment": {"id": "maptask_term", "type": "maptask", "seed": 1, "max_steps": max_steps},
        "agents": [
            {"id": "A", "role": "guider", "model": {"provider": "local", "name": "dummy"}},
            {"id": "B", "role": "follower", "model": {"provider": "local", "name": "dummy"}},
        ],
        "action_space": {
            "enabled": ["message", "draw", "erase", "undo", "reset", "do_nothing"],
            "enabled_by_role": {
                "guider": ["message", "do_nothing"],
                "follower": ["message", "draw", "erase", "undo", "reset", "do_nothing"],
            },
        },
        "controls": {"communication": {"mode": "direct"}},
        "task": {
            "type": "maptask",
            "target_steps": max_steps,
            "canvas_visibility": True,
            "map_material": {
                "map_json": "configs/map_task_material/map.json",
                "map_route_json": "configs/map_task_material/map_route.json",
                "map_route_txt": "configs/map_task_material/map_route.txt",
                "score_board_txt": "configs/map_task_material/score_board.txt",
            },
            "roles": {"A": "guider", "B": "follower"},
        },
        "protocol": {
            "turn_taking": "simultaneous",
            "step_mode": "event",
            "termination": {"condition": termination},
        },
        "probe": {"cadence": "per_turn", "templates": ["situation_awareness_v1"]},
        "logging": {"trace_schema_version": "v0", "output_dir": "experiments/test_maptask_term"},
    }


class MapTaskTerminationTests(unittest.TestCase):
    def test_target_steps_defaults_to_experiment_max_steps(self) -> None:
        config = {
            "__config_dir": "configs/study_conditions/maptask",
            "experiment": {"max_steps": 120},
            "agents": [{"id": "A"}, {"id": "B"}],
            "task": {
                "type": "maptask",
                "canvas_visibility": True,
                "map_material": {
                    "map_json": "configs/map_task_material/map.json",
                    "map_route_json": "configs/map_task_material/map_route.json",
                    "map_route_txt": "configs/map_task_material/map_route.txt",
                },
                "roles": {"A": "guider", "B": "follower"},
            },
        }
        state = maptask_init_state(config)
        self.assertEqual(state["target_steps"], 120)

    def test_maptask_step_marks_complete_at_target_steps(self) -> None:
        state = {"complete": False, "steps_taken": 0, "target_steps": 5}
        for _ in range(5):
            maptask_step(state)
        self.assertTrue(state["complete"])
        self.assertEqual(state["steps_taken"], 5)

    def test_should_terminate_for_maptask_complete_with_max_steps_condition(self) -> None:
        controller = build_controller(_empty_state(), config=_maptask_config(termination="max_steps"))
        controller.state.task_state["complete"] = True
        self.assertTrue(controller._should_terminate_for_task_complete())

    def test_step_budget_marks_maptask_complete(self) -> None:
        controller = build_controller(_empty_state(), config=_maptask_config(termination="task_complete", max_steps=5))
        for _ in range(5):
            controller.step()
        self.assertTrue(controller.state.task_state.get("complete"))
        self.assertTrue(controller._should_terminate_for_task_complete())
        self.assertEqual(controller.state.step_index, 5)


if __name__ == "__main__":
    unittest.main()
