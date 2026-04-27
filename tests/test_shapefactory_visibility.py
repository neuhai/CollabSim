"""ShapeFactory observation visibility: peer economic state is private."""

from __future__ import annotations

import unittest

from src.controller.controller import ExperimentState
from src.controller.factory import build_controller


def _shapefactory_config() -> dict[str, object]:
    return {
        "experiment": {"id": "sf_vis", "seed": 1, "max_steps": 5},
        "agents": [
            {"id": "A", "role": "t", "model": {"provider": "local", "name": "dummy"}},
            {"id": "B", "role": "t", "model": {"provider": "local", "name": "dummy"}},
        ],
        "action_space": {"enabled": ["produce_shape", "do_nothing"]},
        "controls": {},
        "task": {
            "type": "shapefactory",
            "target_steps": 10,
            "shape_options": ["circle", "square"],
            "specialties": {"A": "circle", "B": "square"},
        },
        "protocol": {"turn_taking": "simultaneous", "step_mode": "event", "termination": {"condition": "max_steps"}},
        "probe": {"cadence": "per_action"},
        "logging": {"trace_schema_version": "v0"},
    }


class ShapeFactoryVisibilityTests(unittest.TestCase):
    def test_participants_redacted_for_peers(self) -> None:
        state = ExperimentState(
            agents={},
            task_state={},
            resources={},
            turn_state={},
            buffers={},
        )
        controller = build_controller(state, config=_shapefactory_config())
        participants = controller.state.task_state["participants"]
        assert isinstance(participants, dict)
        participants["A"]["money"] = 111.0
        participants["A"]["inventory"] = ["circle"]
        participants["A"]["production_number"] = 2
        participants["B"]["money"] = 222.0
        participants["B"]["tasks"] = ["square", "circle"]

        obs_a = controller._build_observation_for_agent("A")
        ts = obs_a.state["task_state"]
        assert isinstance(ts, dict)
        pa = ts["participants"]["A"]
        pb = ts["participants"]["B"]
        self.assertEqual(pa.get("money"), 111.0)
        self.assertEqual(pa.get("inventory"), ["circle"])
        self.assertEqual(pb.get("specialty"), "square")
        self.assertNotIn("money", pb)
        self.assertNotIn("inventory", pb)
        self.assertNotIn("tasks", pb)
        self.assertNotIn("production_number", pb)

    def test_shape_produced_hidden_from_non_actor(self) -> None:
        state = ExperimentState(
            agents={},
            task_state={},
            resources={},
            turn_state={},
            buffers={},
        )
        controller = build_controller(state, config=_shapefactory_config())
        controller.state.event_log.append(
            {
                "event_id": "e1",
                "event_type": "shape_produced",
                "timestamp": "t",
                "actor_id": "B",
                "visibility": "public",
                "payload": {"shape": "square", "quantity": 1, "money_after": 50.0},
            }
        )
        visible_a = controller._filter_visible_events_for_agent("A")
        self.assertEqual(visible_a, [])
        visible_b = controller._filter_visible_events_for_agent("B")
        self.assertEqual(len(visible_b), 1)
        self.assertEqual(visible_b[0].get("event_type"), "shape_produced")


if __name__ == "__main__":
    unittest.main()
