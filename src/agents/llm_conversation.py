"""Per-agent OpenAI-style chat history for multi-turn API calls.

Each agent keeps a single thread (action + probe exchanges in order). The full
``messages`` list is sent on every inference so the model sees prior turns.

Optional cap: set env ``COLLABSIM_LLM_CHAT_MAX_MESSAGES`` to a positive integer to
drop oldest messages when the thread exceeds that length (0 or unset = unlimited).
"""

from __future__ import annotations

import json
import os
from typing import Any

ChatMessage = dict[str, str]


def init_llm_chat_thread(agent: Any) -> None:
    if not hasattr(agent, "_llm_chat_messages"):
        agent._llm_chat_messages: list[ChatMessage] = []


def clear_llm_chat_thread(agent: Any) -> None:
    init_llm_chat_thread(agent)
    agent._llm_chat_messages.clear()


def max_llm_chat_messages_cap() -> int:
    raw = os.environ.get("COLLABSIM_LLM_CHAT_MAX_MESSAGES", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def trim_llm_chat_thread(messages: list[ChatMessage]) -> None:
    cap = max_llm_chat_messages_cap()
    if cap <= 0:
        return
    while len(messages) > cap:
        messages.pop(0)


def build_messages_for_request(agent: Any, user_content: str) -> list[ChatMessage]:
    init_llm_chat_thread(agent)
    return [*agent._llm_chat_messages, {"role": "user", "content": user_content}]


def commit_llm_turn(agent: Any, user_content: str, assistant_content: str) -> None:
    init_llm_chat_thread(agent)
    agent._llm_chat_messages.append({"role": "user", "content": user_content})
    agent._llm_chat_messages.append({"role": "assistant", "content": assistant_content})
    trim_llm_chat_thread(agent._llm_chat_messages)


def flatten_messages_for_responses_input(messages: list[ChatMessage]) -> str:
    """Serialize chat turns for APIs that only accept a single ``input`` string."""

    parts: list[str] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            parts.append(f"User:\n{content}")
        elif role == "assistant":
            parts.append(f"Assistant:\n{content}")
        else:
            parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def chat_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content
    text = payload.get("text")
    if isinstance(text, str):
        return text
    return json.dumps(payload)
