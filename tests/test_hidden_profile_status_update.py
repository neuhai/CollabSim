"""Hidden Profile status-update phase hints."""

from __future__ import annotations

import unittest

from src.agents.interface import Observation
from src.agents.status_update_format import format_agent_status_update


class HiddenProfileStatusUpdateTests(unittest.TestCase):
    def test_final_phase_shows_vote_stage_and_submitted(self) -> None:
        obs = Observation(
            state={
                "task_state": {
                    "task_type": "hidden_profile",
                    "phase": "final",
                    "phase_rules": {
                        "initial_vote_decision_id": "initial_vote",
                        "final_vote_decision_id": "final_vote",
                    },
                }
            },
            visible_events=[],
            step_index=88,
            game_status={
                "hidden_profile_phase": "final",
                "hidden_profile_final_vote_submitted": True,
                "hidden_profile_initial_vote_submitted": True,
                "hidden_profile_initial_vote_step_range": "1-3",
                "hidden_profile_discussion_step_range": "4-87",
                "hidden_profile_final_vote_step_range": "88-90",
                "max_steps": 90,
                "remaining_steps": 78,
                "step_index": 88,
            },
        )
        text = format_agent_status_update("A", obs)
        self.assertIn("FINAL VOTE", text)
        self.assertIn("Your final vote: already submitted", text)
        self.assertIn("do_nothing is allowed", text)

    def test_discussion_phase_blocks_decide(self) -> None:
        obs = Observation(
            state={
                "task_state": {
                    "task_type": "hidden_profile",
                    "phase": "discussion",
                    "phase_rules": {"discussion_action_types": ["message"]},
                }
            },
            visible_events=[],
            step_index=5,
            game_status={
                "hidden_profile_phase": "discussion",
                "hidden_profile_initial_vote_step_range": "1-3",
                "hidden_profile_discussion_step_range": "4-87",
                "hidden_profile_final_vote_step_range": "88-90",
                "step_index": 10,
                "max_steps": 90,
            },
        )
        text = format_agent_status_update("B", obs)
        self.assertIn("DISCUSSION", text)
        self.assertIn("decide / voting is not allowed", text)


if __name__ == "__main__":
    unittest.main()
