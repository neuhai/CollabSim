"""SGLang-backed agent implementation.

Connects to a running SGLang inference server via its OpenAI-compatible
/v1/chat/completions endpoint.

Environment variables:
    SGLANG_HOST  - server host (default: localhost)
    SGLANG_PORT  - server port (default: 30000)
    SGLANG_HTTP_RETRIES - max retry attempts (default: 6)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
import time
from typing import Any

import httpx

from src.agents.interface import ActionProposal, AgentMetadata, Observation
from typing import ClassVar
from src.probe.probe import ProbeResponse
from src.utils.env import load_env_file


_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass
class SGLangAgent:
    """Agent that calls a local SGLang inference server."""

    # ClassVar so the controller can detect async support via getattr without
    # this being treated as a dataclass field.
    supports_async: ClassVar[bool] = True

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
        self._host = os.environ.get("SGLANG_HOST", "localhost")
        self._port = int(os.environ.get("SGLANG_PORT", "30000"))
        raw = os.environ.get("SGLANG_HTTP_RETRIES", "6")
        try:
            self._max_retries = max(1, min(int(raw), 20))
        except ValueError:
            self._max_retries = 6

    @property
    def _base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    # ------------------------------------------------------------------
    # Public sync interface (called by ThreadPoolExecutor path)
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        _ = seed

    def serialize(self) -> dict[str, Any]:
        return {}

    def load(self, state: dict[str, Any]) -> None:
        _ = state

    def context_update(self, observation: Observation) -> ActionProposal:
        return asyncio.run(self.context_update_async(observation))

    def propose_action(self, observation: Observation) -> ActionProposal:
        return self.context_update(observation)

    def respond_probe(
        self,
        probe_id: str,
        prompt: str,
        construct: str | None,
        observation: Observation,
    ) -> ProbeResponse:
        return asyncio.run(self.respond_probe_async(probe_id, prompt, construct, observation))

    # ------------------------------------------------------------------
    # Async interface (called by asyncio.gather path in controller)
    # ------------------------------------------------------------------

    async def context_update_async(self, observation: Observation) -> ActionProposal:
        prompt, prompt_static, prompt_update = self._build_action_prompt(observation)
        text = await self._call_sglang_async(prompt)
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

    async def respond_probe_async(
        self,
        probe_id: str,
        prompt: str,
        construct: str | None,
        observation: Observation,
    ) -> ProbeResponse:
        query = self._build_probe_prompt(prompt, construct, observation)
        text = await self._call_sglang_async(query)
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

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    async def _call_sglang_async(self, prompt: str) -> str:
        url = f"{self._base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.metadata.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.metadata.temperature is not None:
            payload["temperature"] = self.metadata.temperature
        if self.metadata.top_p is not None:
            payload["top_p"] = self.metadata.top_p
        if self.metadata.max_tokens is not None:
            payload["max_tokens"] = self.metadata.max_tokens

        async with httpx.AsyncClient(timeout=120.0) as client:
            for attempt in range(self._max_retries):
                try:
                    response = await client.post(url, json=payload)
                    if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries - 1:
                        await asyncio.sleep(min(60.0, 1.0 * (2 ** attempt)))
                        continue
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries - 1:
                        await asyncio.sleep(min(60.0, 1.0 * (2 ** attempt)))
                        continue
                    raise RuntimeError(
                        f"SGLang server error {exc.response.status_code}: {exc.response.text}"
                    ) from exc
                except httpx.RequestError as exc:
                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(min(60.0, 1.0 * (2 ** attempt)))
                        continue
                    raise RuntimeError(
                        f"SGLang connection failed after {self._max_retries} attempts: {exc}"
                    ) from exc
        raise RuntimeError("SGLang: exhausted retries without returning a response.")

    # ------------------------------------------------------------------
    # Prompt building (mirrors AzureOpenAIAgent)
    # ------------------------------------------------------------------

    def _build_action_prompt(self, observation: Observation) -> tuple[str, str, dict[str, Any]]:
        allowed = ", ".join(self.allowed_actions) if self.allowed_actions else "communicate, decide"
        decide_reveal = self.decide_reveal or "aggregated"
        observation_payload: dict[str, Any] = {
            "agent_id": self.metadata.agent_id,
            "state": observation.state,
            "visible_events": observation.visible_events,
            "memory": observation.memory,
        }
        observation_json = json.dumps(observation_payload, ensure_ascii=False)
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
        static_prefix = (
            f"{self.system_prompt}\n\n"
            f"Your participant id is: {self.metadata.agent_id}\n\n"
            f"{persona_prompt}\n\n"
            f"{self.task_prompt_template}\n\n"
            f"{protocol_prompt}\n\n"
            f"{action_space_prompt}\n\n"
            f"{return_format_prompt}"
        )
        full_prompt = (
            f"{self.system_prompt}\n\n"
            f"Your participant id is: {self.metadata.agent_id}\n\n"
            f"{persona_prompt}\n\n"
            f"{self.task_prompt_template}\n\n"
            f"{protocol_prompt}\n\n"
            f"{action_space_prompt}\n\n"
            f"Observation:\n{observation_json}\n\n"
            f"{return_format_prompt}\n\n"
            f"{action_prompt}"
        )
        return full_prompt, static_prefix, observation_payload

    def _build_probe_prompt(self, prompt: str, construct: str | None, observation: Observation) -> str:
        allowed = ", ".join(self.allowed_actions) if self.allowed_actions else "communicate, decide"
        decide_reveal = self.decide_reveal or "aggregated"
        observation_json = json.dumps(
            {
                "agent_id": self.metadata.agent_id,
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
        probe_context = f"Construct: {construct}\n\n" if construct else ""
        probe_prompt = (
            "Instead of selecting an action, answer the probe items below.\n"
            f"{probe_context}"
            f"{prompt}\n\n"
            "Return strict JSON only."
        )
        return (
            f"{self.system_prompt}\n\n"
            f"Your participant id is: {self.metadata.agent_id}\n\n"
            f"{persona_prompt}\n\n"
            f"{self.task_prompt_template}\n\n"
            f"{protocol_prompt}\n\n"
            f"{action_space_prompt}\n\n"
            f"Observation:\n{observation_json}\n\n"
            f"Action payload references:\n"
            f"- communicate: {{\"channel\":\"broadcast|direct\",\"content\":\"...\",\"content_type\":\"text\",\"recipients\":[\"B\"] (required for direct)}}\n"
            f"- decide: {{\"decision_id\":\"plan_selection\",\"choice\":\"...\",\"reveal\":\"{decide_reveal}\"}}\n"
            f"- produce_shape: {{\"shape\":\"<choose_from_task_state>\",\"quantity\":1}}\n"
            f"- propose_trade_offer: {{\"offer_type\":\"buy|sell\",\"shape\":\"<shape_from_task_state>\",\"price_per_unit\":20,\"target_id\":\"B\",\"quantity\":1}}\n"
            f"- trade_response: {{\"transaction_id\":\"offer_2_1\",\"response_type\":\"accept|decline\"}}\n"
            f"- cancel_trade_offer: {{\"transaction_id\":\"offer_2_1\"}}\n"
            f"- fulfill_order: {{\"order_indices\":[0,1]}}\n"
            f"- make_individual_investment: {{\"invest_price\":30}}\n"
            f"- make_group_investment: {{\"invest_price\":30}}\n"
            f"- update_map_progress: {{\"map_progress\":{{\"segment\":\"start_to_bridge\",\"status\":\"confirmed\"}}}}\n"
            f"- do_nothing: {{\"reason\":\"...\"}}\n\n"
            f"{probe_prompt}"
        )

    # ------------------------------------------------------------------
    # Action helpers (mirrors AzureOpenAIAgent)
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
        if "communicate" in self.allowed_actions:
            return {
                "type": "communicate",
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
            normalized: dict[str, Any] = {
                "type": "communicate",
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


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

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
