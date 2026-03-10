"""Configuration schema validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.core.actions import ACTION_TYPES


@dataclass(frozen=True)
class ConfigSchemaError(Exception):
    """Raised when a configuration schema validation fails."""

    message: str

    def __str__(self) -> str:
        return self.message


TOP_LEVEL_KEYS: tuple[str, ...] = (
    "experiment",
    "agents",
    "action_space",
    "controls",
    "task",
    "protocol",
    "probe",
    "logging",
)


def validate_config_schema(config: Mapping[str, Any]) -> None:
    """Validate basic structure and types for an experiment configuration."""

    if not isinstance(config, Mapping):
        raise ConfigSchemaError("Config must be a mapping/object.")

    for key in TOP_LEVEL_KEYS:
        if key not in config:
            raise ConfigSchemaError(f"Config missing required key: {key}")

    _require_mapping(config, "experiment")
    _require_sequence(config, "agents")
    _require_mapping(config, "action_space")
    _require_mapping(config, "controls")
    _require_mapping(config, "task")
    _require_mapping(config, "protocol")
    _require_mapping(config, "probe")
    _require_mapping(config, "logging")

    experiment = config["experiment"]
    _require_field(experiment, "id", str)
    if "seeds" in experiment and not isinstance(experiment.get("seeds"), Mapping):
        raise ConfigSchemaError("experiment.seeds must be an object when provided.")
    if "max_steps" in experiment:
        max_steps = experiment.get("max_steps")
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ConfigSchemaError("experiment.max_steps must be a positive integer.")
    if "prompts" in experiment:
        prompts = experiment.get("prompts")
        if not isinstance(prompts, Mapping):
            raise ConfigSchemaError("experiment.prompts must be an object when provided.")
        for key, value in prompts.items():
            if not isinstance(key, str) or not key:
                raise ConfigSchemaError("experiment.prompts keys must be non-empty strings.")
            if not isinstance(value, str) or not value:
                raise ConfigSchemaError("experiment.prompts values must be non-empty strings.")

    protocol = config["protocol"]
    _require_field(protocol, "turn_taking", str)
    turn_taking = protocol.get("turn_taking")
    if isinstance(turn_taking, str) and turn_taking:
        allowed_turns = {"sequential", "simultaneous"}
        if turn_taking not in allowed_turns:
            raise ConfigSchemaError("protocol.turn_taking must be sequential or simultaneous.")
    _require_field(protocol, "termination", dict)
    termination = protocol.get("termination")
    if isinstance(termination, Mapping):
        condition = termination.get("condition")
        if condition is not None and not isinstance(condition, str):
            raise ConfigSchemaError("protocol.termination.condition must be a string when provided.")
        if isinstance(condition, str) and condition:
            allowed_conditions = {"max_steps", "task_complete"}
            if condition not in allowed_conditions:
                raise ConfigSchemaError("protocol.termination.condition must be max_steps or task_complete.")
    if "proposal_expiry_steps" in protocol:
        expiry_steps = protocol.get("proposal_expiry_steps")
        if not isinstance(expiry_steps, int) or expiry_steps <= 0:
            raise ConfigSchemaError("protocol.proposal_expiry_steps must be a positive integer.")
    if "decision_timeout_steps" in protocol:
        timeout_steps = protocol.get("decision_timeout_steps")
        if not isinstance(timeout_steps, int) or timeout_steps <= 0:
            raise ConfigSchemaError("protocol.decision_timeout_steps must be a positive integer.")
    if "decision_quorum" in protocol:
        quorum = protocol.get("decision_quorum")
        if not isinstance(quorum, int) or quorum <= 0:
            raise ConfigSchemaError("protocol.decision_quorum must be a positive integer.")
    if "step_mode" in protocol:
        step_mode = protocol.get("step_mode")
        if step_mode not in ("tick", "event"):
            raise ConfigSchemaError("protocol.step_mode must be 'tick' or 'event'.")
    if "memory_turn_limit" in protocol:
        limit = protocol.get("memory_turn_limit")
        if not isinstance(limit, int) or limit <= 0:
            raise ConfigSchemaError("protocol.memory_turn_limit must be a positive integer.")
    if "visible_event_window" in protocol:
        window = protocol.get("visible_event_window")
        if not isinstance(window, int) or window <= 0:
            raise ConfigSchemaError("protocol.visible_event_window must be a positive integer.")
    if "visible_event_window_by_agent" in protocol:
        per_agent = protocol.get("visible_event_window_by_agent")
        if not isinstance(per_agent, Mapping) or not per_agent:
            raise ConfigSchemaError("protocol.visible_event_window_by_agent must be an object.")
        for agent_id, window in per_agent.items():
            if not isinstance(agent_id, str) or not agent_id:
                raise ConfigSchemaError("protocol.visible_event_window_by_agent keys must be non-empty strings.")
            if not isinstance(window, int) or window <= 0:
                raise ConfigSchemaError("protocol.visible_event_window_by_agent values must be positive integers.")
    if "agent_memory_key_limit" in protocol:
        limit = protocol.get("agent_memory_key_limit")
        if not isinstance(limit, int) or limit <= 0:
            raise ConfigSchemaError("protocol.agent_memory_key_limit must be a positive integer.")
    if "agent_memory_key_limit_by_agent" in protocol:
        per_agent = protocol.get("agent_memory_key_limit_by_agent")
        if not isinstance(per_agent, Mapping) or not per_agent:
            raise ConfigSchemaError("protocol.agent_memory_key_limit_by_agent must be an object.")
        for agent_id, limit in per_agent.items():
            if not isinstance(agent_id, str) or not agent_id:
                raise ConfigSchemaError("protocol.agent_memory_key_limit_by_agent keys must be non-empty strings.")
            if not isinstance(limit, int) or limit <= 0:
                raise ConfigSchemaError("protocol.agent_memory_key_limit_by_agent values must be positive integers.")
    if "observation_memory_key_limit" in protocol:
        limit = protocol.get("observation_memory_key_limit")
        if not isinstance(limit, int) or limit <= 0:
            raise ConfigSchemaError("protocol.observation_memory_key_limit must be a positive integer.")
    if "observation_memory_key_limit_by_agent" in protocol:
        per_agent = protocol.get("observation_memory_key_limit_by_agent")
        if not isinstance(per_agent, Mapping) or not per_agent:
            raise ConfigSchemaError("protocol.observation_memory_key_limit_by_agent must be an object.")
        for agent_id, limit in per_agent.items():
            if not isinstance(agent_id, str) or not agent_id:
                raise ConfigSchemaError("protocol.observation_memory_key_limit_by_agent keys must be non-empty strings.")
            if not isinstance(limit, int) or limit <= 0:
                raise ConfigSchemaError("protocol.observation_memory_key_limit_by_agent values must be positive integers.")
    if "visible_event_types" in protocol:
        types = protocol.get("visible_event_types")
        if not isinstance(types, list) or not types:
            raise ConfigSchemaError("protocol.visible_event_types must be a non-empty list.")
        if not all(isinstance(item, str) and item for item in types):
            raise ConfigSchemaError("protocol.visible_event_types must contain non-empty strings.")
    if "visible_event_types_exclude" in protocol:
        types = protocol.get("visible_event_types_exclude")
        if not isinstance(types, list) or not types:
            raise ConfigSchemaError("protocol.visible_event_types_exclude must be a non-empty list.")
        if not all(isinstance(item, str) and item for item in types):
            raise ConfigSchemaError("protocol.visible_event_types_exclude must contain non-empty strings.")
    if "visible_event_types_exclude_by_agent" in protocol:
        per_agent = protocol.get("visible_event_types_exclude_by_agent")
        if not isinstance(per_agent, Mapping) or not per_agent:
            raise ConfigSchemaError("protocol.visible_event_types_exclude_by_agent must be an object.")
        for agent_id, types in per_agent.items():
            if not isinstance(agent_id, str) or not agent_id:
                raise ConfigSchemaError("protocol.visible_event_types_exclude_by_agent keys must be non-empty strings.")
            if not isinstance(types, list) or not types:
                raise ConfigSchemaError("protocol.visible_event_types_exclude_by_agent values must be non-empty lists.")
            if not all(isinstance(item, str) and item for item in types):
                raise ConfigSchemaError("protocol.visible_event_types_exclude_by_agent values must contain non-empty strings.")
    if "visible_event_types_by_agent" in protocol:
        per_agent = protocol.get("visible_event_types_by_agent")
        if not isinstance(per_agent, Mapping) or not per_agent:
            raise ConfigSchemaError("protocol.visible_event_types_by_agent must be an object.")
        for agent_id, types in per_agent.items():
            if not isinstance(agent_id, str) or not agent_id:
                raise ConfigSchemaError("protocol.visible_event_types_by_agent keys must be non-empty strings.")
            if not isinstance(types, list) or not types:
                raise ConfigSchemaError("protocol.visible_event_types_by_agent values must be non-empty lists.")
            if not all(isinstance(item, str) and item for item in types):
                raise ConfigSchemaError("protocol.visible_event_types_by_agent values must contain non-empty strings.")

    action_space = config["action_space"]
    _require_field(action_space, "enabled", list)
    _require_list_of_strings(action_space, "enabled")
    enabled_actions = action_space.get("enabled", [])
    if isinstance(enabled_actions, list):
        invalid = [item for item in enabled_actions if item not in ACTION_TYPES]
        if invalid:
            raise ConfigSchemaError(f"action_space.enabled contains unsupported actions: {', '.join(invalid)}")
    decide_cfg = action_space.get("decide")
    if decide_cfg is not None:
        if not isinstance(decide_cfg, Mapping):
            raise ConfigSchemaError("action_space.decide must be an object when provided.")
        reveal = decide_cfg.get("reveal")
        if reveal is not None:
            if not isinstance(reveal, str) or not reveal:
                raise ConfigSchemaError("action_space.decide.reveal must be a non-empty string.")
            allowed_reveal = {"sequential", "aggregated", "simultaneous"}
            if reveal not in allowed_reveal:
                raise ConfigSchemaError(
                    "action_space.decide.reveal must be sequential, aggregated, or simultaneous."
                )

    probe = config["probe"]
    _require_field(probe, "cadence", str)
    cadence = probe.get("cadence")
    if isinstance(cadence, str) and cadence:
        allowed_cadence = {"per_action", "per_turn", "on_event"}
        if cadence not in allowed_cadence:
            raise ConfigSchemaError("probe.cadence must be per_action, per_turn, or on_event.")
        if cadence == "on_event":
            events = probe.get("events")
            if not isinstance(events, list) or not events:
                raise ConfigSchemaError("probe.events must be a non-empty list when cadence is on_event.")
            if not all(isinstance(item, str) and item for item in events):
                raise ConfigSchemaError("probe.events must contain non-empty strings.")
    if "templates" in probe:
        _require_list_of_strings(probe, "templates")
    questions_path = probe.get("questions_path")
    if questions_path is not None and (not isinstance(questions_path, str) or not questions_path):
        raise ConfigSchemaError("probe.questions_path must be a non-empty string when provided.")
    questions = probe.get("questions")
    if questions is not None:
        if not isinstance(questions, list) or not questions:
            raise ConfigSchemaError("probe.questions must be a non-empty list when provided.")
        if not all(isinstance(item, str) and item for item in questions):
            raise ConfigSchemaError("probe.questions must contain non-empty strings.")

    logging_cfg = config["logging"]
    _require_field(logging_cfg, "trace_schema_version", str)
    output_dir = logging_cfg.get("output_dir")
    if output_dir is not None and (not isinstance(output_dir, str) or not output_dir):
        raise ConfigSchemaError("logging.output_dir must be a non-empty string when provided.")
    observation_events = logging_cfg.get("observation_events")
    if observation_events is not None and not isinstance(observation_events, bool):
        raise ConfigSchemaError("logging.observation_events must be a boolean when provided.")

    prompts_cfg = config.get("prompts")
    if prompts_cfg is not None:
        if not isinstance(prompts_cfg, Mapping):
            raise ConfigSchemaError("prompts must be an object when provided.")
        for key, value in prompts_cfg.items():
            if not isinstance(key, str) or not key:
                raise ConfigSchemaError("prompts keys must be non-empty strings.")
            if not isinstance(value, str) or not value:
                raise ConfigSchemaError("prompts values must be non-empty strings.")

    task_cfg = config["task"]
    _require_field(task_cfg, "type", str)
    if task_cfg.get("type") == "counter" and "target_steps" in task_cfg:
        target_steps = task_cfg.get("target_steps")
        if not isinstance(target_steps, int) or target_steps <= 0:
            raise ConfigSchemaError("task.target_steps must be a positive integer.")
    if task_cfg.get("type") == "accumulator":
        target_value = task_cfg.get("target_value")
        increment = task_cfg.get("increment")
        if target_value is not None and (not isinstance(target_value, int) or target_value <= 0):
            raise ConfigSchemaError("task.target_value must be a positive integer.")
        if increment is not None and (not isinstance(increment, int) or increment <= 0):
            raise ConfigSchemaError("task.increment must be a positive integer.")
    if task_cfg.get("type") == "hidden_profile":
        target_steps = task_cfg.get("target_steps")
        if target_steps is not None and (not isinstance(target_steps, int) or target_steps <= 0):
            raise ConfigSchemaError("task.target_steps must be a positive integer.")
        shared_facts = task_cfg.get("shared_facts")
        if shared_facts is not None and not isinstance(shared_facts, list):
            raise ConfigSchemaError("task.shared_facts must be a list when provided.")
        private_facts = task_cfg.get("private_facts")
        if private_facts is not None and not isinstance(private_facts, Mapping):
            raise ConfigSchemaError("task.private_facts must be an object when provided.")
        if isinstance(private_facts, Mapping):
            for agent_id, facts in private_facts.items():
                if not isinstance(agent_id, str) or not agent_id:
                    raise ConfigSchemaError("task.private_facts keys must be non-empty strings.")
                if not isinstance(facts, list):
                    raise ConfigSchemaError("task.private_facts values must be lists.")
    controls = config.get("controls", {})
    if isinstance(controls, Mapping):
        communication = controls.get("communication")
        if communication is not None:
            if not isinstance(communication, Mapping):
                raise ConfigSchemaError("controls.communication must be an object when provided.")
            mode = communication.get("mode")
            if mode is not None:
                if not isinstance(mode, str) or not mode:
                    raise ConfigSchemaError("controls.communication.mode must be a non-empty string.")
                if mode not in {"broadcast", "direct"}:
                    raise ConfigSchemaError("controls.communication.mode must be broadcast or direct.")
            limit = communication.get("max_messages_per_turn")
            if limit is not None and (not isinstance(limit, int) or limit <= 0):
                raise ConfigSchemaError("controls.communication.max_messages_per_turn must be a positive integer.")
        information_distribution = controls.get("information_distribution")
        if information_distribution is not None:
            if not isinstance(information_distribution, Mapping):
                raise ConfigSchemaError("controls.information_distribution must be an object when provided.")
            visibility = information_distribution.get("visibility")
            if visibility is not None and (not isinstance(visibility, str) or not visibility):
                raise ConfigSchemaError("controls.information_distribution.visibility must be a non-empty string.")
        interdependence = controls.get("interdependence")
        if interdependence is not None:
            if not isinstance(interdependence, Mapping):
                raise ConfigSchemaError("controls.interdependence must be an object when provided.")
            structure = interdependence.get("structure")
            if structure is not None and (not isinstance(structure, str) or not structure):
                raise ConfigSchemaError("controls.interdependence.structure must be a non-empty string.")
        visibility_map = controls.get("visibility_map")
        if visibility_map is not None:
            _validate_visibility_map(visibility_map)
        visibility_defaults = controls.get("visibility_defaults")
        if visibility_defaults is not None:
            _validate_visibility_defaults(visibility_defaults)
        visibility_defaults_by_agent = controls.get("visibility_defaults_by_agent")
        if visibility_defaults_by_agent is not None:
            _validate_visibility_defaults_by_agent(visibility_defaults_by_agent)
        visibility_overrides = controls.get("visibility_overrides")
        if visibility_overrides is not None:
            _validate_visibility_overrides(visibility_overrides)
        visibility_overrides_by_agent = controls.get("visibility_overrides_by_agent")
        if visibility_overrides_by_agent is not None:
            _validate_visibility_overrides_by_agent(visibility_overrides_by_agent)
        visibility_excludes = controls.get("visibility_excludes")
        if visibility_excludes is not None:
            _validate_visibility_excludes(visibility_excludes)
        visibility_excludes_by_agent = controls.get("visibility_excludes_by_agent")
        if visibility_excludes_by_agent is not None:
            _validate_visibility_excludes_by_agent(visibility_excludes_by_agent)


def _require_mapping(config: Mapping[str, Any], key: str) -> None:
    if not isinstance(config.get(key), Mapping):
        raise ConfigSchemaError(f"{key} must be an object.")


def _require_sequence(config: Mapping[str, Any], key: str) -> None:
    if not isinstance(config.get(key), Sequence) or isinstance(config.get(key), (str, bytes)):
        raise ConfigSchemaError(f"{key} must be a list.")


def _require_field(mapping: Mapping[str, Any], key: str, expected_type: type) -> None:
    if key not in mapping:
        raise ConfigSchemaError(f"Missing required field: {key}")
    if not isinstance(mapping[key], expected_type):
        raise ConfigSchemaError(f"{key} must be of type {expected_type.__name__}.")


def _require_list_of_strings(mapping: Mapping[str, Any], key: str) -> None:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigSchemaError(f"{key} must be a non-empty list of strings.")
    if not all(isinstance(item, str) and item for item in value):
        raise ConfigSchemaError(f"{key} must contain only non-empty strings.")


def _validate_visibility_map(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ConfigSchemaError("controls.visibility_map must be an object.")
    for agent_id, sections in value.items():
        if not isinstance(agent_id, str) or not agent_id:
            raise ConfigSchemaError("controls.visibility_map keys must be non-empty strings.")
        if not isinstance(sections, Mapping):
            raise ConfigSchemaError("controls.visibility_map entries must be objects.")
        for section, fields in sections.items():
            if not isinstance(section, str) or not section:
                raise ConfigSchemaError("controls.visibility_map section keys must be non-empty strings.")
            if not isinstance(fields, list) or not fields:
                raise ConfigSchemaError("controls.visibility_map fields must be non-empty lists.")
            if not all(isinstance(field, str) and field for field in fields):
                raise ConfigSchemaError("controls.visibility_map fields must be non-empty strings.")


def _validate_visibility_defaults(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ConfigSchemaError("controls.visibility_defaults must be an object.")
    for section, fields in value.items():
        if not isinstance(section, str) or not section:
            raise ConfigSchemaError("controls.visibility_defaults section keys must be non-empty strings.")
        if not isinstance(fields, list) or not fields:
            raise ConfigSchemaError("controls.visibility_defaults fields must be non-empty lists.")
        if not all(isinstance(field, str) and field for field in fields):
            raise ConfigSchemaError("controls.visibility_defaults fields must be non-empty strings.")


def _validate_visibility_defaults_by_agent(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ConfigSchemaError("controls.visibility_defaults_by_agent must be an object.")
    for agent_id, sections in value.items():
        if not isinstance(agent_id, str) or not agent_id:
            raise ConfigSchemaError("controls.visibility_defaults_by_agent keys must be non-empty strings.")
        if not isinstance(sections, Mapping):
            raise ConfigSchemaError("controls.visibility_defaults_by_agent entries must be objects.")
        for section, fields in sections.items():
            if not isinstance(section, str) or not section:
                raise ConfigSchemaError("controls.visibility_defaults_by_agent section keys must be non-empty strings.")
            if not isinstance(fields, list) or not fields:
                raise ConfigSchemaError("controls.visibility_defaults_by_agent fields must be non-empty lists.")
            if not all(isinstance(field, str) and field for field in fields):
                raise ConfigSchemaError("controls.visibility_defaults_by_agent fields must be non-empty strings.")


def _validate_visibility_overrides(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ConfigSchemaError("controls.visibility_overrides must be an object.")
    for section, fields in value.items():
        if not isinstance(section, str) or not section:
            raise ConfigSchemaError("controls.visibility_overrides section keys must be non-empty strings.")
        if not isinstance(fields, list) or not fields:
            raise ConfigSchemaError("controls.visibility_overrides fields must be non-empty lists.")
        if not all(isinstance(field, str) and field for field in fields):
            raise ConfigSchemaError("controls.visibility_overrides fields must be non-empty strings.")


def _validate_visibility_overrides_by_agent(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ConfigSchemaError("controls.visibility_overrides_by_agent must be an object.")
    for agent_id, sections in value.items():
        if not isinstance(agent_id, str) or not agent_id:
            raise ConfigSchemaError("controls.visibility_overrides_by_agent keys must be non-empty strings.")
        if not isinstance(sections, Mapping):
            raise ConfigSchemaError("controls.visibility_overrides_by_agent entries must be objects.")
        for section, fields in sections.items():
            if not isinstance(section, str) or not section:
                raise ConfigSchemaError("controls.visibility_overrides_by_agent section keys must be non-empty strings.")
            if not isinstance(fields, list) or not fields:
                raise ConfigSchemaError("controls.visibility_overrides_by_agent fields must be non-empty lists.")
            if not all(isinstance(field, str) and field for field in fields):
                raise ConfigSchemaError("controls.visibility_overrides_by_agent fields must be non-empty strings.")


def _validate_visibility_excludes(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ConfigSchemaError("controls.visibility_excludes must be an object.")
    for section, fields in value.items():
        if not isinstance(section, str) or not section:
            raise ConfigSchemaError("controls.visibility_excludes section keys must be non-empty strings.")
        if not isinstance(fields, list) or not fields:
            raise ConfigSchemaError("controls.visibility_excludes fields must be non-empty lists.")
        if not all(isinstance(field, str) and field for field in fields):
            raise ConfigSchemaError("controls.visibility_excludes fields must be non-empty strings.")


def _validate_visibility_excludes_by_agent(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ConfigSchemaError("controls.visibility_excludes_by_agent must be an object.")
    for agent_id, sections in value.items():
        if not isinstance(agent_id, str) or not agent_id:
            raise ConfigSchemaError("controls.visibility_excludes_by_agent keys must be non-empty strings.")
        if not isinstance(sections, Mapping):
            raise ConfigSchemaError("controls.visibility_excludes_by_agent entries must be objects.")
        for section, fields in sections.items():
            if not isinstance(section, str) or not section:
                raise ConfigSchemaError("controls.visibility_excludes_by_agent section keys must be non-empty strings.")
            if not isinstance(fields, list) or not fields:
                raise ConfigSchemaError("controls.visibility_excludes_by_agent fields must be non-empty lists.")
            if not all(isinstance(field, str) and field for field in fields):
                raise ConfigSchemaError("controls.visibility_excludes_by_agent fields must be non-empty strings.")
