"""OpenAI-backed agent implementation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
import urllib.request
import urllib.error
from urllib.parse import quote

from src.agents.interface import ActionProposal, AgentMetadata, Observation
from src.agents.action_prompt_compose import compose_action_prompt, compose_probe_prompt
from src.agents.llm_conversation import (
    build_messages_for_request,
    chat_completion_text,
    clear_llm_chat_thread,
    commit_llm_turn,
    finalize_probe_turn,
    flatten_messages_for_responses_input,
    init_llm_chat_thread,
    prepare_probe_messages,
)
from src.agents.parse_model_json import parse_json_dict
from src.probe.probe import ProbeResponse
from src.utils.env import load_env_file, load_text_file
import os


@dataclass
class OpenAIAgent:
    """Agent that calls the OpenAI Responses API."""

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
    communication_limits: str = ""
    probe_context_mode: str = "ephemeral"

    def __post_init__(self) -> None:
        load_env_file()
        provider = (self.metadata.model_provider or "").lower()
        if provider == "azure_openai":
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            if not endpoint or not api_key:
                raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are required for Azure OpenAI.")
        else:
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is required for OpenAIAgent.")
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
        text = self._call_openai(messages)
        commit_llm_turn(self, prompt, text)
        self._llm_action_invocation = inv + 1
        parsed = parse_json_dict(text)
        rationale = parsed.get("rationale") if isinstance(parsed, dict) else None
        if isinstance(parsed, dict):
            raw_actions = parsed.get("actions")
            if isinstance(raw_actions, list):
                actions = self._normalize_actions(raw_actions)
                if len(actions) == 1:
                    return ActionProposal(action=actions[0], rationale=rationale if isinstance(rationale, str) else None, prompt_text=prompt, raw_response=text, prompt_static=prompt_static, prompt_update=prompt_update)
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
        action = self._normalize_action(action)
        return ActionProposal(action=action, rationale=rationale if isinstance(rationale, str) else None, prompt_text=prompt, raw_response=text, prompt_static=prompt_static, prompt_update=prompt_update)

    def propose_action(self, observation: Observation) -> ActionProposal:
        """Backward-compatible alias for context_update."""
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
        messages = prepare_probe_messages(self, query)
        text = self._call_openai(messages)
        finalize_probe_turn(self, query, text)
        self._llm_probe_invocation = pinv + 1
        parsed = parse_json_dict(text)
        answer = parsed.get("answer") if isinstance(parsed, dict) else text.strip()
        confidence = parsed.get("confidence") if isinstance(parsed, dict) else None
        structured_fields = parsed.get("structured_fields") if isinstance(parsed, dict) else None
        return ProbeResponse(
            probe_id=probe_id,
            answer=answer,
            confidence=confidence if isinstance(confidence, (int, float)) else None,
            structured_fields=structured_fields if isinstance(structured_fields, dict) else None,
        )

    def _build_action_prompt(
        self, observation: Observation, *, use_full_prompt: bool = True
    ) -> tuple[str, str, dict[str, Any]]:
        """Returns (full_prompt, static_prefix, observation_payload).

        full_prompt is sent to the API unchanged.
        static_prefix captures agent-level instructions (logged once per agent).
        observation_payload is the per-turn dynamic data (logged every turn).
        """
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

    def _call_openai(self, messages: list[dict[str, str]]) -> str:
        provider = (self.metadata.model_provider or "").lower()

        if provider == "azure_openai":
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            if not endpoint or not api_key:
                raise RuntimeError("Azure OpenAI endpoint or API key is missing.")
            deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or self.metadata.model_name
            payload: dict[str, Any] = {
                "model": deployment,
                "input": flatten_messages_for_responses_input(messages),
            }
            if self.metadata.temperature is not None:
                payload["temperature"] = self.metadata.temperature
            if self.metadata.top_p is not None:
                payload["top_p"] = self.metadata.top_p
            if self.metadata.max_tokens is not None:
                payload["max_output_tokens"] = self.metadata.max_tokens
            headers = {
                "api-key": api_key,
                "Content-Type": "application/json",
            }
            versions = _candidate_api_versions(os.environ.get("AZURE_OPENAI_API_VERSION"))
            urls = _candidate_azure_urls(endpoint, deployment, versions)
            last_error = ""
            for url in urls:
                try:
                    body = _post_json(url=url, headers=headers, payload=payload)
                    parsed = json.loads(body)
                    return _extract_response_text(parsed)
                except urllib.error.HTTPError as exc:
                    error_body = exc.read().decode("utf-8")
                    last_error = f"{exc.code} {error_body}"
                    if exc.code in (400, 404):
                        continue
                    raise RuntimeError(f"OpenAI API error: {error_body}") from exc
            raise RuntimeError(f"OpenAI API error: {last_error or 'Azure request failed after trying fallback URLs.'}")

        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.metadata.model_name,
            "messages": messages,
        }
        if self.metadata.temperature is not None:
            payload["temperature"] = self.metadata.temperature
        if self.metadata.top_p is not None:
            payload["top_p"] = self.metadata.top_p
        if self.metadata.max_tokens is not None:
            payload["max_tokens"] = self.metadata.max_tokens
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        }
        try:
            body = _post_json(url=url, headers=headers, payload=payload)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"OpenAI API error: {exc.read().decode('utf-8')}") from exc
        parsed = json.loads(body)
        return chat_completion_text(parsed)

    def _fallback_action(self) -> dict[str, Any]:
        if "do_nothing" in self.allowed_actions:
            return {"type": "do_nothing", "payload": {"reason": "No action needed."}}
        if "message" in self.allowed_actions:
            return {
                "type": "message",
                "payload": {
                    "channel": "broadcast",
                    "content": "Unable to parse model output; defaulting to status update.",
                    "content_type": "text",
                },
            }
        return {
            "type": "decide",
            "payload": {"decision_id": "default_decision", "choice": "option_A"},
        }

    def _normalize_action(self, action: Any) -> dict[str, Any]:
        if not isinstance(action, dict):
            return self._fallback_action()
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
            normalized = {
                "type": "message",
                "payload": {
                    "channel": channel,
                    "content": content,
                    "content_type": content_type,
                },
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
            normalized_payload = {
                "decision_id": decision_id,
                "choice": choice,
            }
            if self.decide_reveal:
                normalized_payload["reveal"] = self.decide_reveal
            return {"type": "decide", "payload": normalized_payload}

        if isinstance(action_type, str) and action_type in self.allowed_actions:
            # Preserve domain-specific actions when explicitly enabled by config.
            return {"type": action_type, "payload": payload}

        return self._fallback_action()

    def _normalize_actions(self, actions: Any) -> list[dict[str, Any]]:
        if not isinstance(actions, list):
            return []
        normalized: list[dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            normalized.append(self._normalize_action(action))
        return normalized


def _extract_response_text(payload: dict[str, Any]) -> str:
    output = payload.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content", [])
            if isinstance(content, list):
                texts = []
                for chunk in content:
                    if isinstance(chunk, dict) and chunk.get("type") == "output_text":
                        text = chunk.get("text")
                        if isinstance(text, str):
                            texts.append(text)
                if texts:
                    return "\n".join(texts)
    text = payload.get("output_text")
    if isinstance(text, str):
        return text
    return json.dumps(payload)


def _render_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any]) -> str:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def _candidate_api_versions(preferred: str | None) -> list[str]:
    ordered: list[str] = []
    if isinstance(preferred, str) and preferred.strip():
        ordered.append(preferred.strip())
    for candidate in ("preview", "2025-04-01-preview", "2025-03-01-preview", "2025-01-01-preview"):
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _candidate_azure_urls(endpoint: str, deployment: str, versions: list[str]) -> list[str]:
    base = endpoint.rstrip("/")
    encoded_deployment = quote(deployment, safe="")
    urls: list[str] = []

    if base.endswith("/openai/v1/responses"):
        v1_url = base
    elif base.endswith("/openai/v1"):
        v1_url = f"{base}/responses"
    else:
        v1_url = f"{base}/openai/v1/responses"

    openai_url = f"{base}/openai/responses"
    deployment_url = f"{base}/openai/deployments/{encoded_deployment}/responses"

    def _append(url: str) -> None:
        if url not in urls:
            urls.append(url)

    _append(v1_url)
    for version in versions:
        _append(f"{v1_url}?api-version={version}")
    for version in versions:
        _append(f"{openai_url}?api-version={version}")
    for version in versions:
        _append(f"{deployment_url}?api-version={version}")
    return urls
