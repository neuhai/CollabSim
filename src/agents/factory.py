"""Agent factory helpers."""

from __future__ import annotations

from typing import Any
import json
import random

from src.agents.interface import AgentMetadata
from src.agents.openai_agent import OpenAIAgent
from src.agents.azure_agent import AzureOpenAIAgent
from src.agents.litellm_agent import LiteLLMAgent
from src.agents.sglang_agent import SGLangAgent
from src.utils.env import load_text_file


def build_agents(config: dict[str, Any]) -> dict[str, Any]:
    agents_cfg = config.get("agents", [])
    if not isinstance(agents_cfg, list):
        return {}
    action_space = config.get("action_space", {})
    enabled = action_space.get("enabled", []) if isinstance(action_space, dict) else []
    allowed_actions = [item for item in enabled if isinstance(item, str)]
    if "do_nothing" not in allowed_actions:
        allowed_actions.append("do_nothing")
    decide_cfg = action_space.get("decide", {}) if isinstance(action_space, dict) else {}
    decide_reveal = decide_cfg.get("reveal") if isinstance(decide_cfg, dict) else None
    prompt_cfg = config.get("prompts", {})
    if not isinstance(prompt_cfg, dict):
        prompt_cfg = {}
    system_path = prompt_cfg.get("system", "prompts/system_instructions.md")
    persona_path = prompt_cfg.get("persona", "prompts/persona_profile.md")
    task_path = prompt_cfg.get("task", "prompts/task_instructions.md")
    experiment_cfg = config.get("experiment", {})
    task_cfg = config.get("task", {})
    exp_type = experiment_cfg.get("type") if isinstance(experiment_cfg, dict) else None
    task_type = task_cfg.get("type") if isinstance(task_cfg, dict) else None
    split_maptask_prompts = exp_type == "maptask" or task_type == "maptask"
    protocol_path = prompt_cfg.get("protocol", "prompts/protocol.md")
    action_space_path = prompt_cfg.get("action_space", "prompts/action_space.md")
    return_format_path = prompt_cfg.get("return_format", "prompts/return_format.md")
    action_path = prompt_cfg.get("action", "prompts/action_prompt.md")
    probe_path = prompt_cfg.get("probe", "prompts/probe_prompt.md")
    system_prompt = load_text_file(system_path) or "You are a collaborative agent in a research simulation."
    collaboration_enabled = (
        isinstance(experiment_cfg, dict) and experiment_cfg.get("collaboration") is True
    )
    if collaboration_enabled:
        collaboration_path = prompt_cfg.get("collaboration", "prompts/collaboration_module.md")
        collaboration_text = load_text_file(
            collaboration_path if isinstance(collaboration_path, str) else "prompts/collaboration_module.md"
        )
        if isinstance(collaboration_text, str) and collaboration_text.strip():
            system_prompt = f"{system_prompt.rstrip()}\n\n{collaboration_text.strip()}"
    persona_prompt = load_text_file(persona_path) or "Persona profile:\n{persona_profile}"
    protocol_prompt = load_text_file(protocol_path) or "{protocol_json}{communication_limits}"
    action_space_prompt = load_text_file(action_space_path) or "Allowed actions: {allowed_actions}"
    return_format_prompt = load_text_file(return_format_path) or "Return JSON with action + rationale."
    action_prompt = load_text_file(action_path) or ""
    probe_prompt = load_text_file(probe_path) or "Question: {prompt}"
    persona_profiles = _load_persona_profiles(prompt_cfg.get("persona_profiles", "prompts/persona_profiles.json"))
    seed = config.get("experiment", {}).get("seed")
    protocol_context = _build_protocol_context(config)
    communication_limits = _communication_limits_for_prompt(config)
    agents: dict[str, Any] = {}
    for agent in agents_cfg:
        if not isinstance(agent, dict):
            continue
        agent_id = agent.get("id")
        role = agent.get("role")
        model = agent.get("model", {})
        if not isinstance(agent_id, str) or not agent_id:
            continue
        if not isinstance(role, str) or not role:
            role = "participant"
        role_key = role.strip().lower()
        per_role_enabled = None
        if isinstance(action_space, dict):
            by_role = action_space.get("enabled_by_role")
            if isinstance(by_role, dict):
                cand = by_role.get(role_key)
                if isinstance(cand, list) and cand:
                    per_role_enabled = [x for x in cand if isinstance(x, str)]
        agent_allowed = per_role_enabled if per_role_enabled is not None else allowed_actions
        if not isinstance(model, dict):
            continue
        provider = model.get("provider")
        name = model.get("name")
        if provider not in ("openai", "azure_openai", "azure", "sglang", "litellm"):
            continue
        metadata = AgentMetadata(
            agent_id=agent_id,
            role=role,
            model_provider=provider,
            model_name=name,
            model_version=model.get("version"),
            temperature=model.get("temperature"),
            top_p=model.get("top_p"),
            max_tokens=model.get("max_tokens"),
            prompt_version=model.get("prompt_version"),
        )
        persona_profile = _resolve_persona_profile(agent, persona_profiles, seed, agent_id)
        task_prompt_resolved = task_path
        if split_maptask_prompts:
            if role_key == "guider":
                task_prompt_resolved = prompt_cfg.get("task_guider") or task_path
            elif role_key == "follower":
                task_prompt_resolved = prompt_cfg.get("task_follower") or task_path
        task_prompt_loaded = load_text_file(task_prompt_resolved) or load_text_file(task_path)
        task_prompt_final = (
            task_prompt_loaded.strip()
            if isinstance(task_prompt_loaded, str) and task_prompt_loaded.strip()
            else "Task instructions: collaborate to solve the task."
        )
        system_prompt_trimmed = system_prompt.strip() or "You are a collaborative agent in a research simulation."
        agent_allowed_final = list(agent_allowed)
        if "do_nothing" not in agent_allowed_final:
            agent_allowed_final.append("do_nothing")
        kwargs = dict(
            metadata=metadata,
            allowed_actions=agent_allowed_final,
            system_prompt=system_prompt_trimmed,
            persona_prompt_template=persona_prompt,
            task_prompt_template=task_prompt_final,
            protocol_prompt_template=protocol_prompt,
            action_space_prompt_template=action_space_prompt,
            return_format_prompt_template=return_format_prompt,
            action_prompt_template=action_prompt,
            probe_prompt_template=probe_prompt,
            persona_profile=persona_profile,
            protocol_context=protocol_context,
            decide_reveal=decide_reveal if isinstance(decide_reveal, str) else None,
            communication_limits=communication_limits,
        )
        if provider in ("azure_openai", "azure"):
            agents[agent_id] = AzureOpenAIAgent(**kwargs)
        elif provider == "sglang":
            sg_kwargs = dict(kwargs)
            host_override = model.get("sglang_host")
            port_override = model.get("sglang_port")
            if isinstance(host_override, str) and host_override.strip():
                sg_kwargs["sglang_host"] = host_override.strip()
            if isinstance(port_override, int) or (
                isinstance(port_override, str) and str(port_override).isdigit()
            ):
                sg_kwargs["sglang_port"] = (
                    port_override if isinstance(port_override, int) else int(str(port_override).strip())
                )
            agents[agent_id] = SGLangAgent(**sg_kwargs)
        elif provider == "litellm":
            lm_kwargs = dict(kwargs)
            api_base = model.get("api_base")
            api_key = model.get("api_key")
            if isinstance(api_base, str) and api_base.strip():
                lm_kwargs["litellm_api_base"] = api_base.strip()
            if isinstance(api_key, str) and api_key.strip():
                lm_kwargs["litellm_api_key"] = api_key.strip()
            agents[agent_id] = LiteLLMAgent(**lm_kwargs)
        else:
            agents[agent_id] = OpenAIAgent(**kwargs)
    return agents


def _load_persona_profiles(path: str) -> list[str]:
    text = load_text_file(path)
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    profiles: list[str] = []
    for item in payload:
        if isinstance(item, str) and item:
            profiles.append(item)
        elif isinstance(item, dict):
            desc = item.get("description")
            if isinstance(desc, str) and desc:
                profiles.append(desc)
    return profiles


def _resolve_persona_profile(
    agent_cfg: dict[str, Any],
    persona_profiles: list[str],
    seed: Any,
    agent_id: str,
) -> str | None:
    if not isinstance(agent_cfg, dict):
        return None
    explicit = agent_cfg.get("persona_profile")
    if isinstance(explicit, str) and explicit:
        return explicit
    persona_path = agent_cfg.get("persona_profile_path")
    if isinstance(persona_path, str) and persona_path:
        loaded = load_text_file(persona_path)
        if loaded:
            return loaded.strip()
    if not persona_profiles:
        return None
    rng_seed = f"{seed}-{agent_id}"
    rng = random.Random(rng_seed)
    return rng.choice(persona_profiles)


def _build_protocol_context(config: dict[str, Any]) -> dict[str, Any]:
    protocol = config.get("protocol", {})
    protocol_context = dict(protocol) if isinstance(protocol, dict) else {}
    controls = config.get("controls", {})
    if isinstance(controls, dict):
        communication = controls.get("communication", {})
        if isinstance(communication, dict):
            mode = communication.get("mode")
            if isinstance(mode, str) and mode:
                protocol_context["communication_mode"] = mode
                if mode == "direct":
                    protocol_context["allowed_message_channels"] = ["direct"]
                elif mode == "broadcast":
                    # Broadcast mode allows both broadcast and direct channels.
                    protocol_context["allowed_message_channels"] = ["broadcast", "direct"]

    summary = _communication_level_summary(config)
    if summary:
        protocol_context["communication_level"] = summary

    return protocol_context


def _communication_level_summary(config: dict[str, Any]) -> str:
    """Human-readable communication rules for task prompts (e.g. Shape Factory)."""

    controls = config.get("controls", {})
    communication = controls.get("communication", {}) if isinstance(controls, dict) else {}
    if not isinstance(communication, dict):
        communication = {}

    mode = communication.get("mode")
    if not isinstance(mode, str) or not mode:
        mode = "broadcast"

    parts: list[str] = []
    if mode == "direct":
        parts.append(
            "Direct channel only: each message action must use payload channel \"direct\" "
            "with a non-empty recipients list (other participant ids only; no self)."
        )
    else:
        parts.append(
            "Broadcast-capable messaging: you may use payload channel \"broadcast\" (all or a listed subset) "
            "or \"direct\" to specific recipients."
        )

    mw = communication.get("max_message_words")
    if isinstance(mw, int) and mw > 0:
        parts.append(f"Text messages are capped at {mw} words (whitespace-separated).")

    interval = communication.get("min_sim_interval_between_communicate_sec")
    if isinstance(interval, (int, float)) and float(interval) > 0:
        parts.append(f"Wait at least ~{float(interval):g} simulated seconds between accepted messages (sim clock).")

    gap = communication.get("min_agent_actions_between_communicate")
    if isinstance(gap, int) and gap > 0:
        parts.append(
            f"After a message is accepted, complete at least {gap} other accepted actions "
            "before sending the next message (first message has no prior gap)."
        )

    protocol = config.get("protocol", {})
    step_mode = protocol.get("step_mode") if isinstance(protocol, dict) else None

    per_turn = communication.get("max_messages_per_turn")
    if isinstance(per_turn, int) and per_turn > 0 and step_mode != "time":
        parts.append(f"At most {per_turn} message action(s) per controller turn in this stepping mode.")

    return " ".join(parts)


def _communication_limits_for_prompt(config: dict[str, Any]) -> str:
    """Plain-text bullets appended via prompts/protocol.md {communication_limits}."""

    controls = config.get("controls", {})
    if not isinstance(controls, dict):
        return ""
    communication = controls.get("communication", {})
    if not isinstance(communication, dict):
        return ""

    protocol = config.get("protocol", {})
    step_mode = protocol.get("step_mode") if isinstance(protocol, dict) else None

    bullets: list[str] = []

    mw = communication.get("max_message_words")
    if isinstance(mw, int) and mw > 0:
        bullets.append(
            f'Text messages (message with content_type "text"): at most {mw} words '
            f"(whitespace-separated); longer bodies are rejected."
        )

    interval = communication.get("min_sim_interval_between_communicate_sec")
    if isinstance(interval, (int, float)) and float(interval) > 0:
        sec = float(interval)
        bullets.append(
            f"Wait at least about {sec:g} simulated seconds after your last accepted message "
            f"before sending another message (simulator clock)."
        )

    gap = communication.get("min_agent_actions_between_communicate")
    if isinstance(gap, int) and gap > 0:
        bullets.append(
            f"After each accepted message, complete at least {gap} simulator-accepted actions "
            f"that are NOT message before sending another; your first message has no prior gap."
        )

    per_turn = communication.get("max_messages_per_turn")
    if isinstance(per_turn, int) and per_turn > 0 and step_mode != "time":
        bullets.append(f"At most {per_turn} message action(s) per controller turn (this stepping mode).")
    elif isinstance(per_turn, int) and per_turn > 0 and step_mode == "time":
        bullets.append(
            "max_messages_per_turn is set but not enforced in simultaneous real-time (time) mode; "
            "follow the limits above instead."
        )

    if not bullets:
        return ""
    body = "\n".join(f"- {line}" for line in bullets)
    return (
        "\n\nCommunication bandwidth (enforced by the simulator — violations reject message actions):\n" + body
    )
