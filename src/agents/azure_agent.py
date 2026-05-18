"""Azure OpenAI-backed agent implementation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import ssl
import time
from typing import Any
import urllib.request
import urllib.error

from src.agents.interface import ActionProposal, AgentMetadata, Observation
from src.agents.action_prompt_compose import compose_action_prompt, compose_probe_prompt
from src.agents.llm_conversation import (
    build_messages_for_request,
    clear_llm_chat_thread,
    commit_llm_turn,
    flatten_messages_for_responses_input,
    init_llm_chat_thread,
)
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
    communication_limits: str = ""

    def __post_init__(self) -> None:
        load_env_file()
        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            raise ValueError("AZURE_OPENAI_API_KEY is required for AzureOpenAIAgent.")
        if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
            raise ValueError("AZURE_OPENAI_ENDPOINT is required for AzureOpenAIAgent.")
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
        text = self._call_azure_openai(messages)
        commit_llm_turn(self, prompt, text)
        self._llm_action_invocation = inv + 1
        parsed = _parse_json(text)
        if not parsed:
            action = self._fallback_action("parse_failure")
            return ActionProposal(action=action, rationale="Model output was not valid JSON.", prompt_text=prompt, raw_response=text, prompt_static=prompt_static, prompt_update=prompt_update)
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
        if not isinstance(action, dict):
            fallback = self._fallback_action("missing_action")
            return ActionProposal(action=fallback, rationale=rationale if isinstance(rationale, str) else "No action field in JSON.", prompt_text=prompt, raw_response=text, prompt_static=prompt_static, prompt_update=prompt_update)
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
        messages = build_messages_for_request(self, query)
        text = self._call_azure_openai(messages)
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
        return {
            "type": "decide",
            "payload": {"decision_id": "default_decision", "choice": "option_A"},
        }

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

    def _call_azure_openai(self, messages: list[dict[str, str]]) -> str:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required.")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required.")
        deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or self.metadata.model_name
        if not deployment_name:
            raise ValueError("Azure deployment name is required (AZURE_OPENAI_DEPLOYMENT or model.name).")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        chat_url = (
            f"{endpoint.rstrip('/')}/openai/deployments/{deployment_name}/chat/completions"
            f"?api-version={api_version}"
        )
        chat_payload: dict[str, Any] = {
            "messages": messages,
        }
        skip_sampling = _azure_skip_sampling_params(deployment_name) or _azure_skip_sampling_params(
            self.metadata.model_name
        )
        if self.metadata.temperature is not None and not skip_sampling:
            chat_payload["temperature"] = self.metadata.temperature
        if self.metadata.top_p is not None and not skip_sampling:
            chat_payload["top_p"] = self.metadata.top_p
        if self.metadata.max_tokens is not None:
            chat_payload["max_tokens"] = self.metadata.max_tokens

        chat_request = urllib.request.Request(
            chat_url,
            data=json.dumps(chat_payload).encode("utf-8"),
            headers={
                "api-key": api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            response_body = _http_post_with_ssl_retry(chat_request, timeout=60)
            response_payload = json.loads(response_body)
            return _extract_response_text(response_payload)
        except RuntimeError as exc:
            # Newer Azure models (e.g. GPT-5 Codex deployments) may reject the
            # chat/completions operation and require the v1 responses API.
            if "unsupported" not in str(exc).lower():
                raise

        responses_url = f"{endpoint.rstrip('/')}/openai/v1/responses"
        responses_payload: dict[str, Any] = {
            "model": deployment_name,
            "input": flatten_messages_for_responses_input(messages),
        }
        if self.metadata.temperature is not None and not skip_sampling:
            responses_payload["temperature"] = self.metadata.temperature
        if self.metadata.top_p is not None and not skip_sampling:
            responses_payload["top_p"] = self.metadata.top_p
        if self.metadata.max_tokens is not None:
            # v1 Responses uses max_output_tokens.
            responses_payload["max_output_tokens"] = self.metadata.max_tokens
        responses_request = urllib.request.Request(
            responses_url,
            data=json.dumps(responses_payload).encode("utf-8"),
            headers={
                "api-key": api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            responses_body = _http_post_with_ssl_retry(responses_request, timeout=60)
        except RuntimeError as exc:
            err_text = str(exc)
            # Always retry without sampling params when the API rejects temperature, even if
            # heuristics thought sampling was safe (opaque deployment names vs logical model.name).
            if not _azure_error_is_unsupported_temperature(err_text):
                raise
            retry_payload: dict[str, Any] = {
                "model": deployment_name,
                "input": flatten_messages_for_responses_input(messages),
            }
            if self.metadata.max_tokens is not None:
                retry_payload["max_output_tokens"] = self.metadata.max_tokens
            retry_request = urllib.request.Request(
                responses_url,
                data=json.dumps(retry_payload).encode("utf-8"),
                headers={
                    "api-key": api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            responses_body = _http_post_with_ssl_retry(retry_request, timeout=60)
        responses_payload_json = json.loads(responses_body)
        return _extract_responses_v1_text(responses_payload_json)


def _azure_skip_sampling_params(deployment_name: str) -> bool:
    """Whether to omit temperature/top_p for this deployment.

    Many GPT-5 / Codex / o-series Azure deployments return 400 if ``temperature`` is sent
    on the Responses API. Override with env ``AZURE_OPENAI_SKIP_SAMPLING_PARAMS``:
    ``1``/``true`` to always skip, ``0``/``false`` to never skip (use heuristics only when unset).
    """

    raw = os.environ.get("AZURE_OPENAI_SKIP_SAMPLING_PARAMS")
    if raw is not None and raw.strip() != "":
        low = raw.strip().lower()
        if low in ("1", "true", "yes"):
            return True
        if low in ("0", "false", "no"):
            return False
    dn = (deployment_name or "").lower()
    return any(
        key in dn
        for key in (
            "gpt-5",
            "gpt5",
            "codex",
            "o1-preview",
            "o1-mini",
            "o1-pro",
            "o3-mini",
            "o3-pro",
            "o3",
            "reasoning",
        )
    )


def _azure_error_is_unsupported_temperature(body_text: str) -> bool:
    b = body_text.lower()
    if "temperature" not in b:
        return False
    if "not supported" in b or "unsupported" in b:
        return True
    # JSON-style bodies sometimes omit the literal phrase "not supported".
    return '"param"' in b and '"temperature"' in b


def _azure_ssl_context() -> ssl.SSLContext:
    verify = os.environ.get("AZURE_OPENAI_SSL_VERIFY", "true").strip().lower()
    if verify in ("0", "false", "no"):
        return ssl._create_unverified_context()
    ctx = ssl.create_default_context()
    if hasattr(ssl, "TLSVersion"):
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


def _azure_http_error_retryable(status_code: int, body: str) -> bool:
    if status_code in (408, 425, 429, 500, 502, 503, 504):
        return True
    if "NoCapacity" in body or "rate limit" in body.lower():
        return True
    return False


def _http_post_with_ssl_retry(request: urllib.request.Request, *, timeout: int) -> str:
    raw_attempts = os.environ.get("AZURE_OPENAI_HTTP_RETRIES", "10")
    try:
        attempts = max(1, min(int(raw_attempts), 24))
    except ValueError:
        attempts = 10
    for attempt in range(attempts):
        try:
            ctx = _azure_ssl_context()
            with urllib.request.urlopen(request, timeout=timeout, context=ctx) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body_bytes = exc.read()
            try:
                body_text = body_bytes.decode("utf-8")
            except Exception:
                body_text = repr(body_bytes)
            code = getattr(exc, "code", None) or 0
            if _azure_http_error_retryable(int(code), body_text) and attempt < attempts - 1:
                delay = min(60.0, 1.0 * (2**attempt))
                time.sleep(delay)
                continue
            raise RuntimeError(f"Azure OpenAI API error: {body_text}") from exc
        except (ssl.SSLError, urllib.error.URLError, OSError) as exc:
            if attempt < attempts - 1:
                # TLS EOF/network flaps often recover with a longer exponential backoff.
                delay = min(60.0, 1.0 * (2**attempt))
                time.sleep(delay)
                continue
            raise RuntimeError(
                "HTTPS to Azure OpenAI failed after retries (network/TLS). "
                "Try another network or VPN, confirm AZURE_OPENAI_ENDPOINT, "
                "or set AZURE_OPENAI_SSL_VERIFY=0 only for debugging. "
                f"Last error: {exc!r}"
            ) from exc


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


def _extract_responses_v1_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for chunk in content:
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("type") not in {"output_text", "text"}:
                    continue
                text = chunk.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)
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


def _task_type_from_observation(observation: Observation) -> str | None:
    task_state = observation.state.get("task_state") if isinstance(observation.state, dict) else None
    if not isinstance(task_state, dict):
        return None
    task_type = task_state.get("task_type")
    return task_type if isinstance(task_type, str) and task_type else None


def _summarize_shapefactory_observation(payload: dict[str, Any]) -> str:
    agent_id = payload.get("agent_id")
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    task_state = state.get("task_state") if isinstance(state.get("task_state"), dict) else {}
    participants = task_state.get("participants") if isinstance(task_state.get("participants"), dict) else {}
    self_row = participants.get(agent_id) if isinstance(agent_id, str) else None
    if not isinstance(self_row, dict):
        self_row = {}
    rules = task_state.get("rules") if isinstance(task_state.get("rules"), dict) else {}
    pending_offers = task_state.get("pending_offers") if isinstance(task_state.get("pending_offers"), list) else []
    lines = [f"You are agent {agent_id} in ShapeFactory."]
    specialty = self_row.get("specialty")
    if isinstance(specialty, str) and specialty:
        lines.append(f"Your specialty shape is {specialty}.")
    money = self_row.get("money")
    if isinstance(money, (int, float)):
        lines.append(f"Your money: {money}.")
    inventory = self_row.get("inventory")
    if isinstance(inventory, dict):
        inv_items = ", ".join(f"{k}:{v}" for k, v in sorted(inventory.items()) if isinstance(k, str))
        lines.append(f"Your inventory: {inv_items or 'empty'}.")
    tasks = self_row.get("tasks")
    if isinstance(tasks, list):
        if tasks:
            task_lines = []
            for i, t in enumerate(tasks):
                if not isinstance(t, dict):
                    continue
                shapes = t.get("shapes", {})
                reward = t.get("reward")
                shape_str = ", ".join(f"{s}x{q}" for s, q in shapes.items()) if isinstance(shapes, dict) else str(shapes)
                task_lines.append(f"  order[{i}]: needs {shape_str}, reward={reward}")
            lines.append(f"Your pending orders ({len(tasks)}):\n" + "\n".join(task_lines))
        else:
            lines.append("Your pending orders: none.")
    production_number = self_row.get("production_number")
    max_production = rules.get("max_production_num")
    if isinstance(production_number, int) and isinstance(max_production, int):
        lines.append(f"Production used: {production_number}/{max_production}.")
    peer_bits: list[str] = []
    for pid, row in sorted(participants.items()):
        if not isinstance(pid, str) or pid == agent_id or not isinstance(row, dict):
            continue
        peer_specialty = row.get("specialty")
        if isinstance(peer_specialty, str) and peer_specialty:
            peer_bits.append(f"{pid}:{peer_specialty}")
    if peer_bits:
        lines.append("Other participants (public specialties only): " + ", ".join(peer_bits) + ".")
    # Show offers targeting me (actionable: I need to respond to these)
    offers_targeting_me = [o for o in pending_offers if isinstance(o, dict) and o.get("to") == agent_id]
    if offers_targeting_me:
        lines.append(f"Offers targeting YOU (respond with trade_response):")
        for o in offers_targeting_me:
            lines.append(f"  id={o.get('id')} type={o.get('offer_type')} shape={o.get('shape')} qty={o.get('quantity')} price={o.get('price_per_unit')} from={o.get('from')}")
    else:
        lines.append("Offers targeting you: none.")
    # Show my own pending offers
    my_offers = [o for o in pending_offers if isinstance(o, dict) and o.get("from") == agent_id]
    if my_offers:
        lines.append(f"Your open offers (cancel with cancel_trade_offer if stale):")
        for o in my_offers:
            lines.append(f"  id={o.get('id')} type={o.get('offer_type')} shape={o.get('shape')} qty={o.get('quantity')} price={o.get('price_per_unit')} to={o.get('to')}")
    else:
        lines.append("Your open offers: none.")
    # Show other market offers (context only)
    other_offers = [o for o in pending_offers if isinstance(o, dict) and o.get("from") != agent_id and o.get("to") != agent_id]
    if other_offers:
        lines.append(f"Other market offers ({len(other_offers)} total — for context only).")
    recent_events = payload.get("visible_events")
    if isinstance(recent_events, list) and recent_events:
        tail = recent_events[-5:]
        event_types = [str(evt.get("event_type")) for evt in tail if isinstance(evt, dict)]
        if event_types:
            lines.append("Recent visible events: " + ", ".join(event_types) + ".")
    return "\n".join(lines)