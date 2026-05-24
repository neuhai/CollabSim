"""DayTrader observation visibility: peer balances and investment history stay private."""

from __future__ import annotations

import unittest

from src.controller.controller import ExperimentState
from src.controller.factory import build_controller


def _daytrader_config() -> dict[str, object]:
    return {
        "experiment": {"id": "dt_vis", "seed": 1},
        "agents": [
            {"id": "A", "role": "investor", "model": {"provider": "local", "name": "dummy"}},
            {"id": "B", "role": "investor", "model": {"provider": "local", "name": "dummy"}},
        ],
        "action_space": {"enabled": ["make_individual_investment", "do_nothing"]},
        "controls": {},
        "task": {
            "type": "daytrader",
            "target_rounds": 5,
            "starting_money": 200,
        },
        "protocol": {
            "turn_taking": "simultaneous",
            "step_mode": "step",
            "termination": {"condition": "task_complete"},
        },
        "probe": {"cadence": "per_action"},
        "logging": {"trace_schema_version": "v0"},
    }


class DayTraderVisibilityTests(unittest.TestCase):
    def test_buffers_redacted_for_peers(self) -> None:
        state = ExperimentState(
            agents={},
            task_state={},
            resources={},
            turn_state={},
            buffers={},
        )
        controller = build_controller(state, config=_daytrader_config())
        controller.state.buffers["daytrader_private_participants"] = {
            "A": {
                "money": 320.0,
                "investment_history": [
                    {"investment_amount": 40, "investment_type": "individual", "money_after": 240.0},
                ],
            },
            "B": {
                "money": 180.0,
                "investment_history": [
                    {"investment_amount": 50, "investment_type": "group", "money_after": 150.0},
                ],
            },
        }
        controller.state.buffers["daytrader_group_pool"] = {"A": 10.0, "B": 25.0}
        controller.state.buffers["daytrader_round_start_money"] = {"A": 200.0, "B": 200.0}

        obs_a = controller._build_observation_for_agent("A")
        buffers_a = obs_a.state.get("buffers")
        self.assertIsInstance(buffers_a, dict)
        private_a = buffers_a.get("daytrader_private_participants")
        self.assertIsInstance(private_a, dict)
        self.assertEqual(set(private_a.keys()), {"A"})
        self.assertEqual(private_a["A"]["money"], 320.0)
        self.assertEqual(len(private_a["A"]["investment_history"]), 1)
        self.assertEqual(buffers_a.get("daytrader_group_pool"), {"A": 10.0})
        self.assertEqual(buffers_a.get("daytrader_round_start_money"), {"A": 200.0})

        obs_b = controller._build_observation_for_agent("B")
        private_b = obs_b.state["buffers"]["daytrader_private_participants"]
        self.assertEqual(set(private_b.keys()), {"B"})
        self.assertEqual(private_b["B"]["money"], 180.0)
        self.assertNotIn("A", private_b)


if __name__ == "__main__":
    unittest.main()
