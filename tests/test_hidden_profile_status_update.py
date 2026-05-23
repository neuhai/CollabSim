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
            step_index=90,
            game_status={
                "hidden_profile_phase": "final",
                "hidden_profile_final_vote_submitted": True,
                "hidden_profile_initial_vote_submitted": True,
                "hidden_profile_initial_vote_step_range": "1-1",
                "hidden_profile_discussion_step_range": "2-89",
                "hidden_profile_final_vote_step_range": "90-90",
                "max_steps": 90,
                "remaining_steps": 78,
                "step_index": 90,
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
                "hidden_profile_initial_vote_step_range": "1-1",
                "hidden_profile_discussion_step_range": "2-89",
                "hidden_profile_final_vote_step_range": "90-90",
                "step_index": 10,
                "max_steps": 90,
            },
        )
        text = format_agent_status_update("B", obs)
        self.assertIn("DISCUSSION", text)
        self.assertIn("decide / voting is not allowed", text)

    def test_discussion_shows_message_content_in_visible_events(self) -> None:
        obs = Observation(
            state={
                "task_state": {
                    "task_type": "hidden_profile",
                    "phase": "discussion",
                }
            },
            visible_events=[
                {
                    "event_type": "message_delivered",
                    "actor_id": "B",
                    "payload": {
                        "message_id": "msg_2",
                        "channel": "broadcast",
                        "content": "Candidate C handles stress well.",
                        "content_type": "text",
                        "recipients": ["A", "C"],
                    },
                }
            ],
            step_index=3,
            game_status={"step_index": 3, "max_steps": 90},
        )
        text = format_agent_status_update("A", obs)
        self.assertIn("Candidate C handles stress well", text)
        self.assertIn("message_delivered", text)

    def test_discussion_shows_full_message_content_not_truncated(self) -> None:
        long_content = "A" * 120
        obs = Observation(
            state={"task_state": {"task_type": "hidden_profile", "phase": "discussion"}},
            visible_events=[
                {
                    "event_type": "message_delivered",
                    "actor_id": "B",
                    "payload": {
                        "message_id": "msg_9",
                        "channel": "broadcast",
                        "content": long_content,
                        "content_type": "text",
                        "recipients": ["A", "C"],
                    },
                }
            ],
            step_index=3,
            game_status={"step_index": 3},
        )
        text = format_agent_status_update("A", obs)
        self.assertIn(long_content, text)
        self.assertNotIn("...", text)


if __name__ == "__main__":
    unittest.main()
