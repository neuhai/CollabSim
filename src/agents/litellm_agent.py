"""LiteLLM-backed agent: routes chat completions via the litellm library.

Supports any LiteLLM-recognized ``model.name`` string (OpenAI, Azure, Anthropic,
Bedrock, local OpenAI-compatible endpoints, proxies, etc.). Credentials are read
from the usual provider env vars and/or optional ``OPENAI_API_KEY`` / LiteLLM env.
Bedrock (``bedrock/...`` models) requires ``boto3`` and AWS credentials.

Environment (optional overrides):
    LITELLM_API_BASE  - forwarded as ``api_base`` when ``model.api_base`` is unset
    LITELLM_API_KEY   - forwarded as ``api_key`` when unset in config
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION_NAME - Bedrock auth

Config YAML (per agent, under ``model``):
    provider: litellm
    name: <litellm model id, e.g. gpt-4o, azure/<deployment>, openai/gpt-4o>
    api_base: <optional OpenAI-compatible base URL or proxy>
    api_key: <optional; prefer env for secrets>
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import litellm

from src.agents.interface import ActionProposal, AgentMetadata, Observation
from src.agents.action_prompt_compose import compose_action_prompt, compose_probe_prompt
from src.agents.llm_conversation import (
    build_messages_for_request,
    clear_llm_chat_thread,
    commit_llm_turn,
    init_llm_chat_thread,
)
from src.probe.probe import ProbeResponse
from src.utils.env import load_env_file

# Reuse JSON parsing aligned with other agents.
from src.agents.sglang_agent import _parse_json


@dataclass
class LiteLLMAgent:
    """Agent that calls models through LiteLLM (sync chat completion)."""

    metadata: AgentMetadata
    allowed_actions: list[str]
    system_prompt: str
    persona_prompt_template: str
    task_prompt_template: str
    protocol_prompt_template: str
    action_space_prompt_template: str
    return_format_prompt_template: str
    action_prompt_template: str
    probe_prompt_template: str
    persona_profile: str | None = None
    protocol_context: dict[str, Any] | None = None
    decide_reveal: str | None = None
    litellm_api_base: str | None = None
    litellm_api_key: str | None = None
    communication_limits: str = ""

    def __post_init__(self) -> None:
        load_env_file()
        self._llm_action_invocation = 0
        self._llm_probe_invocation = 0
        init_llm_chat_thread(self)

    def reset(self, seed: int | None = None) -> None:
        _ = seed
        self._llm_action_invocation = 0
        self._llm_probe_invocation = 0
        clear_llm_chat_thread(self)

    def serialize(self) -> dict[str, Any]:
        return {}

    def load(self, state: dict[str, Any]) -> None:
        _ = state

    def context_update(self, observation: Observation) -> ActionProposal:
        inv = self._llm_action_invocation
        use_full = inv == 0
        prompt, prompt_static, prompt_update = self._build_action_prompt(observation, use_full_prompt=use_full)
        messages = build_messages_for_request(self, prompt)
        text = self._call_litellm(messages)
        commit_llm_turn(self, prompt, text)
        self._llm_action_invocation = inv + 1
        parsed = _parse_json(text)
        if not parsed:
            action = self._fallback_action("parse_failure")
            return ActionProposal(
                action=action,
                rationale="Model output was not valid JSON.",
                prompt_text=prompt,
                raw_response=text,
                prompt_static=prompt_static,
                prompt_update=prompt_update,
            )
        rationale = parsed.get("rationale") if isinstance(parsed, dict) else None
        if isinstance(parsed, dict):
            raw_actions = parsed.get("actions")
            if isinstance(raw_actions, list):
                actions = self._normalize_actions(raw_actions)
                if len(actions) == 1:
                    return ActionProposal(
                        action=actions[0],
                        rationale=rationale if isinstance(rationale, str) else None,
                        prompt_text=prompt,
                        raw_response=text,
                        prompt_static=prompt_static,
                        prompt_update=prompt_update,
                    )
                if len(actions) > 1:
                    return {
                        "actions": actions,
                        "rationale": rationale if isinstance(rationale, str) else None,
                        "prompt_text": prompt,
                        "raw_response": text,
                        "prompt_static": prompt_static,
                        "prompt_update": prompt_update,
                    }
        action = parsed.get("action") if isinstance(parsed, dict) else None
        if not isinstance(action, dict):
            fallback = self._fallback_action("missing_action")
            return ActionProposal(
                action=fallback,
                rationale=rationale if isinstance(rationale, str) else "No action field in JSON.",
                prompt_text=prompt,
                raw_response=text,
                prompt_static=prompt_static,
                prompt_update=prompt_update,
            )
        action = self._normalize_action(action)
        return ActionProposal(
            action=action,
            rationale=rationale if isinstance(rationale, str) else None,
            prompt_text=prompt,
            raw_response=text,
            prompt_static=prompt_static,
            prompt_update=prompt_update,
        )

    def propose_action(self, observation: Observation) -> ActionProposal:
        return self.context_update(observation)

    def respond_probe(
        self,
        probe_id: str,
        prompt: str,
        construct: str | None,
        observation: Observation,
    ) -> ProbeResponse:
        pinv = self._llm_probe_invocation
        use_full = pinv == 0
        query = self._build_probe_prompt(prompt, construct, observation, use_full_prompt=use_full)
        messages = build_messages_for_request(self, query)
        text = self._call_litellm(messages)
        commit_llm_turn(self, query, text)
        self._llm_probe_invocation = pinv + 1
        parsed = _parse_json(text)
        answer = parsed.get("answer") if isinstance(parsed, dict) else text.strip()
        confidence = parsed.get("confidence") if isinstance(parsed, dict) else None
        structured_fields = parsed.get("structured_fields") if isinstance(parsed, dict) else None
        return ProbeResponse(
            probe_id=probe_id,
            answer=answer,
            confidence=confidence if isinstance(confidence, (int, float)) else None,
            structured_fields=structured_fields if isinstance(structured_fields, dict) else None,
        )

    def _call_litellm(self, messages: list[dict[str, str]]) -> str:
        model = self.metadata.model_name
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model.name is required for LiteLLMAgent (LiteLLM model id).")

        api_base = self.litellm_api_base or os.environ.get("LITELLM_API_BASE")
        api_key = self.litellm_api_key or os.environ.get("LITELLM_API_KEY")

        kwargs: dict[str, Any] = {
            "model": model.strip(),
            "messages": messages,
        }
        if self.metadata.temperature is not None:
            kwargs["temperature"] = self.metadata.temperature
        if self.metadata.top_p is not None:
            kwargs["top_p"] = self.metadata.top_p
        if self.metadata.max_tokens is not None:
            kwargs["max_tokens"] = self.metadata.max_tokens
        if api_base:
            kwargs["api_base"] = api_base.strip()
        if api_key:
            kwargs["api_key"] = api_key.strip()

        response = litellm.completion(**kwargs)
        choice = response.choices[0]
        message = choice.message
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("LiteLLM returned no string content in the first choice.")
        return content

    # ------------------------------------------------------------------
    # Prompt building (aligned with SGLangAgent)
    # ------------------------------------------------------------------

    def _build_action_prompt(
        self, observation: Observation, *, use_full_prompt: bool = True
    ) -> tuple[str, str, dict[str, Any]]:
        return compose_action_prompt(
            observation,
            agent_id=self.metadata.agent_id,
            role=self.metadata.role,
            system_prompt=self.system_prompt,
            persona_prompt_template=self.persona_prompt_template,
            persona_profile=self.persona_profile,
            task_prompt_template=self.task_prompt_template,
            protocol_prompt_template=self.protocol_prompt_template,
            protocol_context=self.protocol_context,
            communication_limits=self.communication_limits,
            action_space_prompt_template=self.action_space_prompt_template,
            return_format_prompt_template=self.return_format_prompt_template,
            action_prompt_template=self.action_prompt_template,
            allowed_actions=self.allowed_actions,
            decide_reveal=self.decide_reveal,
            use_full_prompt=use_full_prompt,
        )

    def _build_probe_prompt(
        self,
        prompt: str,
        construct: str | None,
        observation: Observation,
        *,
        use_full_prompt: bool = True,
    ) -> str:
        return compose_probe_prompt(
            observation,
            agent_id=self.metadata.agent_id,
            role=self.metadata.role,
            system_prompt=self.system_prompt,
            persona_prompt_template=self.persona_prompt_template,
            persona_profile=self.persona_profile,
            task_prompt_template=self.task_prompt_template,
            protocol_prompt_template=self.protocol_prompt_template,
            protocol_context=self.protocol_context,
            communication_limits=self.communication_limits,
            action_space_prompt_template=self.action_space_prompt_template,
            allowed_actions=self.allowed_actions,
            decide_reveal=self.decide_reveal,
            prompt=prompt,
            construct=construct,
            use_full_prompt=use_full_prompt,
        )

    # ------------------------------------------------------------------
    # Action helpers (aligned with SGLangAgent)
    # ------------------------------------------------------------------

    def _fallback_action(self, reason_code: str = "default") -> dict[str, Any]:
        if "do_nothing" in self.allowed_actions:
            reason_map = {
                "parse_failure": "fallback: parse_failure",
                "missing_action": "fallback: missing_action",
                "normalize_failure": "fallback: normalize_failure",
                "default": "fallback: default",
            }
            return {"type": "do_nothing", "payload": {"reason": reason_map.get(reason_code, "fallback: default")}}
        if "message" in self.allowed_actions:
            return {
                "type": "message",
                "payload": {
                    "channel": "broadcast",
                    "content": "Unable to parse model output; defaulting to status update.",
                    "content_type": "text",
                },
            }
        return {"type": "decide", "payload": {"decision_id": "default_decision", "choice": "option_A"}}

    def _normalize_action(self, action: Any) -> dict[str, Any]:
        if not isinstance(action, dict):
            return self._fallback_action("normalize_failure")
        action_type = action.get("type")
        payload = action.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        if not isinstance(action_type, str):
            action_type = None

        if action_type == "communicate":
            action_type = "message"

        if action_type is None:
            if "message" in payload or "content" in payload:
                action_type = "message"
            elif "decision" in payload or "choice" in payload or "plan" in payload:
                action_type = "decide"
            elif "do_nothing" in self.allowed_actions:
                action_type = "do_nothing"

        if action_type == "message":
            content = payload.get("content")
            if not isinstance(content, str) or not content:
                message = payload.get("message")
                content = message if isinstance(message, str) and message else "Status update."
            channel = payload.get("channel")
            if channel not in ("broadcast", "direct"):
                channel = "broadcast"
            content_type = payload.get("content_type")
            if content_type not in ("text", "json"):
                content_type = "text"
            normalized: dict[str, Any] = {
                "type": "message",
                "payload": {"channel": channel, "content": content, "content_type": content_type},
            }
            recipients = payload.get("recipients")
            if channel == "direct" and isinstance(recipients, list):
                normalized["payload"]["recipients"] = recipients
            return normalized

        if action_type == "decide":
            decision_id = payload.get("decision_id")
            if not isinstance(decision_id, str) or not decision_id:
                decision_id = "plan_selection"
            choice = payload.get("choice")
            if not isinstance(choice, str) or not choice:
                choice = payload.get("decision") or payload.get("plan")
            if not isinstance(choice, str) or not choice:
                choice = "undecided"
            normalized_payload: dict[str, Any] = {"decision_id": decision_id, "choice": choice}
            if self.decide_reveal:
                normalized_payload["reveal"] = self.decide_reveal
            return {"type": "decide", "payload": normalized_payload}

        if isinstance(action_type, str) and action_type in self.allowed_actions:
            return {"type": action_type, "payload": payload}

        return self._fallback_action()

    def _normalize_actions(self, actions: Any) -> list[dict[str, Any]]:
        if not isinstance(actions, list):
            return []
        return [self._normalize_action(a) for a in actions if isinstance(a, dict)]
