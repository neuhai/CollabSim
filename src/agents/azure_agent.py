"""Azure OpenAI-backed agent implementation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
import urllib.request
import urllib.error

from src.agents.interface import ActionProposal, AgentMetadata, Observation
from src.probe.probe import ProbeResponse
from src.utils.env import load_env_file, load_text_file
import os


@dataclass
class AzureOpenAIAgent:
    """Agent that calls the Azure OpenAI API."""

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

    def __post_init__(self) -> None:
        load_env_file()
        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            raise ValueError("AZURE_OPENAI_API_KEY is required for AzureOpenAIAgent.")
        if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
            raise ValueError("AZURE_OPENAI_ENDPOINT is required for AzureOpenAIAgent.")

    def reset(self, seed: int | None = None) -> None:
        _ = seed

    def serialize(self) -> dict[str, Any]:
        return {}

    def load(self, state: dict[str, Any]) -> None:
        _ = state

    def context_update(self, observation: Observation) -> ActionProposal:
        prompt = self._build_action_prompt(observation)
        text = self._call_azure_openai(prompt)
        parsed = _parse_json(text)
        action = parsed.get("action") if isinstance(parsed, dict) else None
        rationale = parsed.get("rationale") if isinstance(parsed, dict) else None
        action = self._normalize_action(action)
        return ActionProposal(action=action, rationale=rationale if isinstance(rationale, str) else None)

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
        query = self._build_probe_prompt(prompt, construct, observation)
        text = self._call_azure_openai(query)
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

    def _build_action_prompt(self, observation: Observation) -> str:
        allowed = ", ".join(self.allowed_actions) if self.allowed_actions else "communicate, decide"
        decide_reveal = self.decide_reveal or "aggregated"
        observation_json = json.dumps(
            {
                "state": observation.state,
                "visible_events": observation.visible_events,
                "memory": observation.memory,
            },
            ensure_ascii=False,
        )
        persona_profile = self.persona_profile or "Default collaborative persona."
        persona_prompt = _render_template(self.persona_prompt_template, {"persona_profile": persona_profile})
        protocol_prompt = _render_template(
            self.protocol_prompt_template,
            {"protocol_json": json.dumps(self.protocol_context or {}, ensure_ascii=False)},
        )
        action_space_prompt = _render_template(self.action_space_prompt_template, {"allowed_actions": allowed})
        return_format_prompt = _render_template(self.return_format_prompt_template, {"decide_reveal": decide_reveal})
        action_prompt = _render_template(
            self.action_prompt_template,
            {
                "allowed_actions": allowed,
                "observation_json": observation_json,
                "decide_reveal": decide_reveal,
            },
        )
        return (
            f"{self.system_prompt}\n\n"
            f"{persona_prompt}\n\n"
            f"{self.task_prompt_template}\n\n"
            f"{protocol_prompt}\n\n"
            f"{action_space_prompt}\n\n"
            f"Observation:\n{observation_json}\n\n"
            f"{return_format_prompt}\n\n"
            f"{action_prompt}"
        )

    def _build_probe_prompt(
        self,
        prompt: str,
        construct: str | None,
        observation: Observation,
    ) -> str:
        construct_line = f"Construct: {construct}\n" if construct else ""
        observation_json = json.dumps(
            {
                "state": observation.state,
                "visible_events": observation.visible_events,
                "memory": observation.memory,
            },
            ensure_ascii=False,
        )
        persona_profile = self.persona_profile or "Default collaborative persona."
        persona_prompt = _render_template(self.persona_prompt_template, {"persona_profile": persona_profile})
        protocol_prompt = _render_template(
            self.protocol_prompt_template,
            {"protocol_json": json.dumps(self.protocol_context or {}, ensure_ascii=False)},
        )
        probe_prompt = _render_template(
            self.probe_prompt_template,
            {
                "construct_line": construct_line.rstrip(),
                "prompt": prompt,
                "observation_json": observation_json,
            },
        )
        return (
            f"{self.system_prompt}\n\n"
            f"{persona_prompt}\n\n"
            f"{self.task_prompt_template}\n\n"
            f"{protocol_prompt}\n\n"
            f"{probe_prompt}"
        )

    def _call_azure_openai(self, prompt: str) -> str:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required.")
        
        deployment_name = self.metadata.model_name
        url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment_name}/chat/completions?api-version=2024-02-15-preview"
        
        payload: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }
        if self.metadata.temperature is not None:
            payload["temperature"] = self.metadata.temperature
        if self.metadata.top_p is not None:
            payload["top_p"] = self.metadata.top_p
        if self.metadata.max_tokens is not None:
            payload["max_tokens"] = self.metadata.max_tokens
        
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {os.environ.get('AZURE_OPENAI_API_KEY')}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Azure OpenAI API error: {exc.read().decode('utf-8')}") from exc
        payload = json.loads(body)
        return _extract_response_text(payload)

    def _fallback_action(self) -> dict[str, Any]:
        if "do_nothing" in self.allowed_actions:
            return {"type": "do_nothing", "payload": {"reason": "No action needed."}}
        if "communicate" in self.allowed_actions:
            return {
                "type": "communicate",
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

        if action_type is None:
            if "message" in payload or "content" in payload:
                action_type = "communicate"
            elif "decision" in payload or "choice" in payload or "plan" in payload:
                action_type = "decide"
            elif "do_nothing" in self.allowed_actions:
                action_type = "do_nothing"

        if action_type == "communicate":
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
                "type": "communicate",
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

        return self._fallback_action()


def _extract_response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content
    text = payload.get("text")
    if isinstance(text, str):
        return text
    return json.dumps(payload)


def _parse_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return {}
    return {}


def _render_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered