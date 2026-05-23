"""DayTrader round/discussion phase scheduling tests."""

from __future__ import annotations

import unittest

from src.controller.controller import ExperimentState
from src.controller.factory import build_controller


def _daytrader_config(
    *,
    target_rounds: int = 6,
    discussion_every_n_rounds: int = 2,
) -> dict[str, object]:
    return {
        "experiment": {"id": "dt_phases", "seed": 1},
        "agents": [
            {"id": "A", "role": "investor", "model": {"provider": "local", "name": "dummy"}},
            {"id": "B", "role": "investor", "model": {"provider": "local", "name": "dummy"}},
        ],
        "action_space": {"enabled": ["do_nothing", "message"]},
        "controls": {},
        "task": {
            "type": "daytrader",
            "target_rounds": target_rounds,
            "starting_money": 200,
            "phase_rules": {
                "discussion_every_n_rounds": discussion_every_n_rounds,
                "group_chat_max_turns": 2,
            },
        },
        "protocol": {
            "turn_taking": "simultaneous",
            "step_mode": "step",
            "termination": {"condition": "task_complete"},
        },
        "probe": {"cadence": "per_action"},
        "logging": {"trace_schema_version": "v0"},
    }


def _noop(agent_id: str, timestamp: int) -> dict[str, object]:
    return {
        "type": "do_nothing",
        "actor_id": agent_id,
        "timestamp": timestamp,
        "payload": {},
    }


class DayTraderPhaseTests(unittest.TestCase):
    def test_discussion_only_after_every_n_rounds(self) -> None:
        controller = build_controller(
            ExperimentState(agents={}, task_state={}, resources={}, turn_state={}, buffers={}),
            config=_daytrader_config(target_rounds=6, discussion_every_n_rounds=2),
        )
        ts = controller.state.task_state
        self.assertEqual(ts.get("round_index"), 1)
        self.assertEqual(ts.get("phase"), "decision")
        self.assertEqual(ts["phase_rules"]["discussion_every_n_rounds"], 2)

        # Round 1 decision: no group chat after all act.
        for agent_id in ("A", "B"):
            controller.state.pending_actions.append(_noop(agent_id, 1))
        controller.step()
        self.assertEqual(ts.get("round_index"), 2)
        self.assertEqual(ts.get("phase"), "decision")
        self.assertEqual(ts.get("rounds_completed"), 1)

        # Round 2 decision: group chat follows.
        for agent_id in ("A", "B"):
            controller.state.pending_actions.append(_noop(agent_id, 2))
        controller.step()
        self.assertEqual(ts.get("round_index"), 2)
        self.assertEqual(ts.get("phase"), "group_chat")

        # End group chat (group_chat_max_turns=2); do_nothing advances turns in this phase.
        for agent_id, ts_val in (("A", 3), ("B", 4)):
            controller.state.pending_actions.append(_noop(agent_id, ts_val))
            controller.step()
        self.assertEqual(ts.get("round_index"), 3)
        self.assertEqual(ts.get("phase"), "decision")
        self.assertEqual(ts.get("rounds_completed"), 2)

    def test_thirty_rounds_with_discussion_every_five(self) -> None:
        controller = build_controller(
            ExperimentState(agents={}, task_state={}, resources={}, turn_state={}, buffers={}),
            config=_daytrader_config(target_rounds=30, discussion_every_n_rounds=5),
        )
        ts = controller.state.task_state
        self.assertEqual(ts.get("target_rounds"), 30)

        for decision_round in range(1, 31):
            self.assertEqual(ts.get("round_index"), decision_round)
            self.assertEqual(ts.get("phase"), "decision")
            for agent_id in ("A", "B"):
                controller.state.pending_actions.append(_noop(agent_id, decision_round))
            controller.step()
            if decision_round % 5 == 0:
                self.assertEqual(
                    ts.get("phase"),
                    "group_chat",
                    f"round {decision_round} should enter group_chat",
                )
                for agent_id in ("A", "B"):
                    controller.state.pending_actions.append(_noop(agent_id, 100 + decision_round))
                    controller.step()
                if decision_round < 30:
                    self.assertEqual(ts.get("phase"), "decision")
                    self.assertEqual(ts.get("round_index"), decision_round + 1)
            else:
                self.assertEqual(
                    ts.get("phase"),
                    "decision",
                    f"round {decision_round} should skip group_chat",
                )
                self.assertEqual(ts.get("round_index"), decision_round + 1)

        self.assertTrue(ts.get("complete"))
        self.assertEqual(ts.get("rounds_completed"), 30)

    def test_group_chat_completes_after_rejected_message_drains_queue(self) -> None:
        config = _daytrader_config(target_rounds=30, discussion_every_n_rounds=5)
        config["agents"] = [
            {"id": "A", "role": "investor", "model": {"provider": "local", "name": "dummy"}},
            {"id": "B", "role": "investor", "model": {"provider": "local", "name": "dummy"}},
            {"id": "C", "role": "investor", "model": {"provider": "local", "name": "dummy"}},
        ]
        config["controls"] = {"communication": {"min_agent_actions_between_communicate": 5}}

        controller = build_controller(
            ExperimentState(agents={}, task_state={}, resources={}, turn_state={}, buffers={}),
            config=config,
        )
        ts = controller.state.task_state
        ts["round_index"] = 30
        ts["phase"] = "group_chat"
        ts["group_chat_turns"] = 11
        ts["rounds_completed"] = 29

        comm_state = controller._communication_controls_state()
        comm_state["actions_since_comm"] = {"C": 2}

        controller.state.pending_actions.append(
            {
                "type": "message",
                "actor_id": "C",
                "timestamp": 66,
                "payload": {
                    "channel": "broadcast",
                    "content": "Great game everyone!",
                    "content_type": "text",
                    "recipients": ["A", "B"],
                },
            }
        )
        controller.step()

        self.assertTrue(ts.get("complete"))
        self.assertEqual(ts.get("rounds_completed"), 30)


if __name__ == "__main__":
    unittest.main()
