"""Agent factory helpers."""

from __future__ import annotations

from typing import Any
import json
import random

from src.agents.interface import AgentMetadata
from src.agents.openai_agent import OpenAIAgent
from src.agents.azure_agent import AzureOpenAIAgent
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
    protocol_path = prompt_cfg.get("protocol", "prompts/protocol.md")
    action_space_path = prompt_cfg.get("action_space", "prompts/action_space.md")
    return_format_path = prompt_cfg.get("return_format", "prompts/return_format.md")
    action_path = prompt_cfg.get("action", "prompts/action_prompt.md")
    probe_path = prompt_cfg.get("probe", "prompts/probe_prompt.md")
    system_prompt = load_text_file(system_path) or "You are a collaborative agent in a research simulation."
    persona_prompt = load_text_file(persona_path) or "Persona profile:\n{persona_profile}"
    task_prompt = load_text_file(task_path) or "Task instructions: collaborate to solve the task."
    protocol_prompt = load_text_file(protocol_path) or "Experiment protocol:\n{protocol_json}"
    action_space_prompt = load_text_file(action_space_path) or "Allowed actions: {allowed_actions}"
    return_format_prompt = load_text_file(return_format_path) or "Return JSON with action + rationale."
    action_prompt = load_text_file(action_path) or ""
    probe_prompt = load_text_file(probe_path) or "Question: {prompt}"
    persona_profiles = _load_persona_profiles(prompt_cfg.get("persona_profiles", "prompts/persona_profiles.json"))
    seed = config.get("experiment", {}).get("seed")
    protocol_context = _build_protocol_context(config)
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
        if not isinstance(model, dict):
            continue
        provider = model.get("provider")
        name = model.get("name")
        if provider not in ("openai", "azure_openai", "azure"):
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
        system_prompt_trimmed = system_prompt.strip() or "You are a collaborative agent in a research simulation."
        kwargs = dict(
            metadata=metadata,
            allowed_actions=allowed_actions,
            system_prompt=system_prompt_trimmed,
            persona_prompt_template=persona_prompt,
            task_prompt_template=task_prompt,
            protocol_prompt_template=protocol_prompt,
            action_space_prompt_template=action_space_prompt,
            return_format_prompt_template=return_format_prompt,
            action_prompt_template=action_prompt,
            probe_prompt_template=probe_prompt,
            persona_profile=persona_profile,
            protocol_context=protocol_context,
            decide_reveal=decide_reveal if isinstance(decide_reveal, str) else None,
        )
        if provider in ("azure_openai", "azure"):
            agents[agent_id] = AzureOpenAIAgent(**kwargs)
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
    if not isinstance(controls, dict):
        return protocol_context
    communication = controls.get("communication", {})
    if not isinstance(communication, dict):
        return protocol_context
    mode = communication.get("mode")
    if isinstance(mode, str) and mode:
        protocol_context["communication_mode"] = mode
        if mode == "direct":
            protocol_context["allowed_message_channels"] = ["direct"]
        elif mode == "broadcast":
            # Broadcast mode allows both broadcast and direct channels.
            protocol_context["allowed_message_channels"] = ["broadcast", "direct"]
    return protocol_context
