"""Probe context isolation from the action chat thread."""

from __future__ import annotations

import unittest

from src.agents.llm_conversation import (
    clear_llm_chat_thread,
    commit_llm_turn,
    finalize_probe_turn,
    init_llm_chat_thread,
    prepare_probe_messages,
)


class _StubAgent:
    def __init__(self, *, probe_context_mode: str = "ephemeral") -> None:
        self.probe_context_mode = probe_context_mode
        init_llm_chat_thread(self)


class LLMProbeContextTests(unittest.TestCase):
    def test_ephemeral_probe_reads_action_thread_but_does_not_commit(self) -> None:
        agent = _StubAgent(probe_context_mode="ephemeral")
        commit_llm_turn(agent, "action prompt 1", '{"action":{"type":"do_nothing"}}')
        commit_llm_turn(agent, "action prompt 2", '{"action":{"type":"do_nothing"}}')

        messages = prepare_probe_messages(agent, "probe question")
        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[-1]["content"], "probe question")

        finalize_probe_turn(agent, "probe question", '{"answer":"ok"}')
        self.assertEqual(len(agent._llm_chat_messages), 4)

        messages_again = prepare_probe_messages(agent, "probe question 2")
        self.assertEqual(len(messages_again), 5)
        self.assertEqual(messages_again[-1]["content"], "probe question 2")

    def test_shared_probe_commits_into_action_thread(self) -> None:
        agent = _StubAgent(probe_context_mode="shared")
        commit_llm_turn(agent, "action prompt", '{"action":{"type":"do_nothing"}}')

        prepare_probe_messages(agent, "probe question")
        finalize_probe_turn(agent, "probe question", '{"answer":"ok"}')
        self.assertEqual(len(agent._llm_chat_messages), 4)

    def test_clear_thread_resets_action_history(self) -> None:
        agent = _StubAgent()
        commit_llm_turn(agent, "action", "response")
        clear_llm_chat_thread(agent)
        messages = prepare_probe_messages(agent, "probe only")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "probe only")


if __name__ == "__main__":
    unittest.main()
