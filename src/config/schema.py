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
    if "duration_sec" in experiment:
        duration_sec = experiment.get("duration_sec")
        if not isinstance(duration_sec, (int, float)) or duration_sec <= 0:
            raise ConfigSchemaError("experiment.duration_sec must be a positive number.")
    if "duration_ms" in experiment:
        duration_ms = experiment.get("duration_ms")
        if not isinstance(duration_ms, (int, float)) or duration_ms <= 0:
            raise ConfigSchemaError("experiment.duration_ms must be a positive number.")
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
            allowed_conditions = {"max_steps", "task_complete", "time_elapsed"}
            if condition not in allowed_conditions:
                raise ConfigSchemaError(
                    "protocol.termination.condition must be max_steps, task_complete, or time_elapsed."
                )
        if "noop_consecutive_cycles" in termination:
            noop_cycles = termination.get("noop_consecutive_cycles")
            if not isinstance(noop_cycles, int) or noop_cycles <= 0:
                raise ConfigSchemaError("protocol.termination.noop_consecutive_cycles must be a positive integer.")
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
        if step_mode not in ("tick", "event", "time", "step"):
            raise ConfigSchemaError("protocol.step_mode must be 'tick', 'event', 'time', or 'step'.")
    if "agent_trigger_interval_sec" in protocol:
        interval_sec = protocol.get("agent_trigger_interval_sec")
        if not isinstance(interval_sec, (int, float)) or float(interval_sec) <= 0:
            raise ConfigSchemaError("protocol.agent_trigger_interval_sec must be a positive number.")
    if "produce_shape_delay_sec" in protocol:
        delay_sec = protocol.get("produce_shape_delay_sec")
        if not isinstance(delay_sec, (int, float)) or float(delay_sec) <= 0:
            raise ConfigSchemaError("protocol.produce_shape_delay_sec must be a positive number.")
    if "proactive_wakeup_interval_ms" in protocol:
        interval = protocol.get("proactive_wakeup_interval_ms")
        if not isinstance(interval, int) or interval <= 0:
            raise ConfigSchemaError("protocol.proactive_wakeup_interval_ms must be a positive integer.")
    if "max_concurrent_requests" in protocol:
        max_workers = protocol.get("max_concurrent_requests")
        if not isinstance(max_workers, int) or max_workers <= 0:
            raise ConfigSchemaError("protocol.max_concurrent_requests must be a positive integer.")
    # Backward compatibility: old key is still accepted.
    if "max_concurrent_thinking" in protocol:
        max_workers = protocol.get("max_concurrent_thinking")
        if not isinstance(max_workers, int) or max_workers <= 0:
            raise ConfigSchemaError("protocol.max_concurrent_thinking must be a positive integer.")
    if "think_time_ms" in protocol:
        think = protocol.get("think_time_ms")
        if isinstance(think, int):
            if think < 0:
                raise ConfigSchemaError("protocol.think_time_ms must be non-negative.")
        elif isinstance(think, Mapping):
            lo = think.get("min")
            hi = think.get("max")
            if not isinstance(lo, int) or not isinstance(hi, int) or lo < 0 or hi < lo:
                raise ConfigSchemaError("protocol.think_time_ms must be an int or {min,max} with 0 <= min <= max.")
        else:
            raise ConfigSchemaError("protocol.think_time_ms must be an int or object.")
    if "action_durations_ms" in protocol:
        durations = protocol.get("action_durations_ms")
        if not isinstance(durations, Mapping) or not durations:
            raise ConfigSchemaError("protocol.action_durations_ms must be a non-empty object.")
        for action_type, duration in durations.items():
            if not isinstance(action_type, str) or not action_type:
                raise ConfigSchemaError("protocol.action_durations_ms keys must be non-empty strings.")
            if not isinstance(duration, int) or duration < 0:
                raise ConfigSchemaError("protocol.action_durations_ms values must be non-negative integers.")
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
    enabled_by_role = action_space.get("enabled_by_role")
    if enabled_by_role is not None:
        if not isinstance(enabled_by_role, Mapping):
            raise ConfigSchemaError("action_space.enabled_by_role must be an object when provided.")
        for role_key, actions in enabled_by_role.items():
            if not isinstance(role_key, str) or not role_key.strip():
                raise ConfigSchemaError("action_space.enabled_by_role keys must be non-empty strings.")
            if not isinstance(actions, list) or not actions:
                raise ConfigSchemaError(
                    f"action_space.enabled_by_role['{role_key}'] must be a non-empty list when provided."
                )
            invalid_role = [item for item in actions if item not in ACTION_TYPES]
            if invalid_role:
                raise ConfigSchemaError(
                    f"action_space.enabled_by_role['{role_key}'] unsupported actions: {', '.join(invalid_role)}"
                )
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
        allowed_cadence = {"per_action", "per_turn", "on_event", "per_agent_n_actions"}
        if cadence not in allowed_cadence:
            raise ConfigSchemaError(
                "probe.cadence must be per_action, per_turn, on_event, or per_agent_n_actions."
            )
        if cadence == "on_event":
            events = probe.get("events")
            if not isinstance(events, list) or not events:
                raise ConfigSchemaError("probe.events must be a non-empty list when cadence is on_event.")
            if not all(isinstance(item, str) and item for item in events):
                raise ConfigSchemaError("probe.events must contain non-empty strings.")
        if cadence == "per_agent_n_actions":
            every_n_actions = probe.get("every_n_actions")
            if not isinstance(every_n_actions, int) or every_n_actions <= 0:
                raise ConfigSchemaError(
                    "probe.every_n_actions must be a positive integer when cadence is per_agent_n_actions."
                )
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
    task_type = task_cfg.get("type")
    if task_type not in {"counter", "accumulator", "hidden_profile", "shapefactory", "daytrader", "maptask", "noop"}:
        raise ConfigSchemaError(
            "task.type must be one of counter, accumulator, hidden_profile, shapefactory, daytrader, maptask, noop."
        )
    if task_type == "counter" and "target_steps" in task_cfg:
        target_steps = task_cfg.get("target_steps")
        if not isinstance(target_steps, int) or target_steps <= 0:
            raise ConfigSchemaError("task.target_steps must be a positive integer.")
    if task_type == "accumulator":
        target_value = task_cfg.get("target_value")
        increment = task_cfg.get("increment")
        if target_value is not None and (not isinstance(target_value, int) or target_value <= 0):
            raise ConfigSchemaError("task.target_value must be a positive integer.")
        if increment is not None and (not isinstance(increment, int) or increment <= 0):
            raise ConfigSchemaError("task.increment must be a positive integer.")
    if task_type == "hidden_profile":
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
        phase_rules = task_cfg.get("phase_rules")
        if phase_rules is not None:
            if not isinstance(phase_rules, Mapping):
                raise ConfigSchemaError("task.phase_rules must be an object when provided.")
            initial_vote_decision_id = phase_rules.get("initial_vote_decision_id")
            if initial_vote_decision_id is not None and (
                not isinstance(initial_vote_decision_id, str) or not initial_vote_decision_id
            ):
                raise ConfigSchemaError(
                    "task.phase_rules.initial_vote_decision_id must be a non-empty string when provided."
                )
            final_vote_decision_id = phase_rules.get("final_vote_decision_id")
            if final_vote_decision_id is not None and (
                not isinstance(final_vote_decision_id, str) or not final_vote_decision_id
            ):
                raise ConfigSchemaError(
                    "task.phase_rules.final_vote_decision_id must be a non-empty string when provided."
                )
            discussion_action_types = phase_rules.get("discussion_action_types")
            if discussion_action_types is not None:
                if not isinstance(discussion_action_types, list):
                    raise ConfigSchemaError("task.phase_rules.discussion_action_types must be a list when provided.")
                if not all(isinstance(item, str) and item for item in discussion_action_types):
                    raise ConfigSchemaError(
                        "task.phase_rules.discussion_action_types must contain non-empty strings."
                    )
            discussion_max_turns = phase_rules.get("discussion_max_turns")
            if discussion_max_turns is not None and (
                not isinstance(discussion_max_turns, int) or discussion_max_turns <= 0
            ):
                raise ConfigSchemaError(
                    "task.phase_rules.discussion_max_turns must be a positive integer when provided."
                )
            initial_vote_steps = phase_rules.get("initial_vote_steps")
            if initial_vote_steps is not None and (
                not isinstance(initial_vote_steps, int) or initial_vote_steps <= 0
            ):
                raise ConfigSchemaError(
                    "task.phase_rules.initial_vote_steps must be a positive integer when provided."
                )
            final_vote_steps = phase_rules.get("final_vote_steps")
            if final_vote_steps is not None and (
                not isinstance(final_vote_steps, int) or final_vote_steps <= 0
            ):
                raise ConfigSchemaError(
                    "task.phase_rules.final_vote_steps must be a positive integer when provided."
                )
    if task_type == "shapefactory":
        _require_positive_number_if_present(task_cfg, "starting_money", allow_zero=True)
        _require_positive_number_if_present(task_cfg, "regular_cost")
        _require_positive_number_if_present(task_cfg, "specialty_cost")
        _require_positive_number_if_present(task_cfg, "min_trade_price")
        _require_positive_number_if_present(task_cfg, "max_trade_price")
        _require_positive_number_if_present(task_cfg, "incentive_money")
        _require_positive_number_if_present(task_cfg, "production_time")
        _require_positive_int_if_present(task_cfg, "max_production_num")
        _require_positive_int_if_present(task_cfg, "shapes_order")
        _require_positive_int_if_present(task_cfg, "shapes_types")
        if "shape_options" in task_cfg:
            shape_options = task_cfg.get("shape_options")
            if not isinstance(shape_options, list) or not shape_options:
                raise ConfigSchemaError("task.shape_options must be a non-empty list when provided.")
            if not all(isinstance(item, str) and item for item in shape_options):
                raise ConfigSchemaError("task.shape_options must contain non-empty strings.")
        if "peer_economic_dashboard" in task_cfg:
            peer_dash = task_cfg.get("peer_economic_dashboard")
            if not isinstance(peer_dash, bool):
                raise ConfigSchemaError("task.peer_economic_dashboard must be a boolean when provided.")
    if task_type == "daytrader":
        if "target_steps" in task_cfg:
            raise ConfigSchemaError("daytrader uses task.target_rounds; remove task.target_steps.")
        target_rounds = task_cfg.get("target_rounds")
        if not isinstance(target_rounds, int) or target_rounds <= 0:
            raise ConfigSchemaError("task.target_rounds must be a positive integer for daytrader.")
        phase_rules = task_cfg.get("phase_rules")
        if phase_rules is not None:
            if not isinstance(phase_rules, Mapping):
                raise ConfigSchemaError("task.phase_rules must be an object when provided for daytrader.")
            group_chat_max_turns = phase_rules.get("group_chat_max_turns")
            if group_chat_max_turns is not None and (
                not isinstance(group_chat_max_turns, int) or group_chat_max_turns <= 0
            ):
                raise ConfigSchemaError(
                    "task.phase_rules.group_chat_max_turns must be a positive integer when provided."
                )
            discussion_every = phase_rules.get("discussion_every_n_rounds")
            if discussion_every is not None and (
                not isinstance(discussion_every, int) or discussion_every <= 0
            ):
                raise ConfigSchemaError(
                    "task.phase_rules.discussion_every_n_rounds must be a positive integer when provided."
                )
        _require_positive_number_if_present(task_cfg, "starting_money", allow_zero=True)
        _require_positive_number_if_present(task_cfg, "min_trade_price")
        _require_positive_number_if_present(task_cfg, "max_trade_price")
        _require_positive_number_if_present(task_cfg, "incentive_money")
    if task_type == "maptask":
        landmarks_path = task_cfg.get("landmarks_path")
        if landmarks_path is not None and (not isinstance(landmarks_path, str) or not landmarks_path):
            raise ConfigSchemaError("task.landmarks_path must be a non-empty string when provided.")
        if "roles" in task_cfg:
            roles = task_cfg.get("roles")
            if not isinstance(roles, Mapping):
                raise ConfigSchemaError("task.roles must be an object when provided.")
            for agent_id, role in roles.items():
                if not isinstance(agent_id, str) or not agent_id:
                    raise ConfigSchemaError("task.roles keys must be non-empty strings.")
                if role not in {"guider", "follower"}:
                    raise ConfigSchemaError("task.roles values must be guider or follower.")
        if "maps" in task_cfg:
            maps = task_cfg.get("maps")
            if not isinstance(maps, Mapping):
                raise ConfigSchemaError("task.maps must be an object when provided.")
            for agent_id, value in maps.items():
                if not isinstance(agent_id, str) or not agent_id:
                    raise ConfigSchemaError("task.maps keys must be non-empty strings.")
                if not isinstance(value, Mapping):
                    raise ConfigSchemaError("task.maps values must be objects.")
                map_id = value.get("map_id")
                if map_id is not None and (not isinstance(map_id, str) or not map_id):
                    raise ConfigSchemaError("task.maps.<agent_id>.map_id must be a non-empty string when provided.")
                map_file = value.get("map_file")
                if map_file is not None and (not isinstance(map_file, str) or not map_file):
                    raise ConfigSchemaError("task.maps.<agent_id>.map_file must be a non-empty string when provided.")
                if "landmarks" in value:
                    landmarks = value.get("landmarks")
                    if not isinstance(landmarks, list):
                        raise ConfigSchemaError("task.maps.<agent_id>.landmarks must be a list when provided.")
                    if not all(isinstance(item, str) and item for item in landmarks):
                        raise ConfigSchemaError("task.maps.<agent_id>.landmarks must contain non-empty strings.")
        if "canvas_visibility" in task_cfg:
            canvas_visibility = task_cfg.get("canvas_visibility")
            if not isinstance(canvas_visibility, bool):
                raise ConfigSchemaError("task.canvas_visibility must be a boolean when provided for maptask.")
        map_material = task_cfg.get("map_material")
        if map_material is not None:
            if not isinstance(map_material, Mapping):
                raise ConfigSchemaError("task.map_material must be an object when provided.")
            for key in ("map_json", "map_route_json", "map_route_txt"):
                path_val = map_material.get(key)
                if path_val is None or not isinstance(path_val, str) or not path_val.strip():
                    raise ConfigSchemaError(
                        f"task.map_material.{key} must be a non-empty string when map_material is provided."
                    )
            sb_opt = map_material.get("score_board_txt")
            if sb_opt is not None and (not isinstance(sb_opt, str) or not sb_opt.strip()):
                raise ConfigSchemaError("task.map_material.score_board_txt must be a non-empty string when provided.")
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
            max_words = communication.get("max_message_words")
            if max_words is not None and (not isinstance(max_words, int) or max_words <= 0):
                raise ConfigSchemaError("controls.communication.max_message_words must be a positive integer when provided.")
            min_interval = communication.get("min_sim_interval_between_communicate_sec")
            if min_interval is not None and (
                not isinstance(min_interval, (int, float)) or float(min_interval) <= 0
            ):
                raise ConfigSchemaError(
                    "controls.communication.min_sim_interval_between_communicate_sec must be positive when provided."
                )
            min_between = communication.get("min_agent_actions_between_communicate")
            if min_between is not None and (not isinstance(min_between, int) or min_between <= 0):
                raise ConfigSchemaError(
                    "controls.communication.min_agent_actions_between_communicate must be a positive integer when provided."
                )
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


def _require_positive_int_if_present(mapping: Mapping[str, Any], key: str) -> None:
    value = mapping.get(key)
    if value is None:
        return
    if not isinstance(value, int) or value <= 0:
        raise ConfigSchemaError(f"task.{key} must be a positive integer.")


def _require_positive_number_if_present(
    mapping: Mapping[str, Any],
    key: str,
    *,
    allow_zero: bool = False,
) -> None:
    value = mapping.get(key)
    if value is None:
        return
    if not isinstance(value, (int, float)):
        raise ConfigSchemaError(f"task.{key} must be a number when provided.")
    if allow_zero:
        if float(value) < 0:
            raise ConfigSchemaError(f"task.{key} must be >= 0.")
        return
    if float(value) <= 0:
        raise ConfigSchemaError(f"task.{key} must be > 0.")


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
