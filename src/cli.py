"""Minimal CLI entrypoint for CollabSim experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.utils.env import load_env_file
from src.config.loader import ConfigError, load_experiment_config
from src.config.schema import ConfigSchemaError, validate_config_schema
from src.controller.controller import ExperimentState
from src.controller.factory import build_controller
from src.data.logging import build_run_manifest, write_json
from src.probe.loader import ProbeTemplateError, load_probe_templates
from src.agents.factory import build_agents
from src.tasks.registry import build_default_registry
from src.config.model_overrides import apply_agent_model_overrides


_LAST_SIM_TIME_MS: int | None = None


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for validating experiment configurations."""

    parser = argparse.ArgumentParser(description="CollabSim CLI")
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument("--run-id", default="cli_run", help="Run identifier")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write run logs (events, probes, metrics, manifest)",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps to run")
    parser.add_argument("--validate-only", action="store_true", help="Validate config and exit")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing logs")
    parser.add_argument(
        "--print-actions",
        action="store_true",
        help="Print per-agent action progress to stdout during run",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Override run manifest output path (JSON)",
    )
    parser.add_argument(
        "--probe-templates",
        default="configs/probe_templates.yml",
        help="Path to probe template YAML for registry initialization",
    )
    parser.add_argument(
        "--model-provider",
        default=None,
        metavar="NAME",
        help="Override model.provider for all agents (overrides YAML). Env: COLLABSIM_MODEL_PROVIDER.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        metavar="MODEL",
        help="Override model.name for all agents. Env: COLLABSIM_MODEL_NAME.",
    )
    parser.add_argument(
        "--model-temperature",
        type=float,
        default=None,
        metavar="T",
        help="Override model.temperature for all agents. Env: COLLABSIM_MODEL_TEMPERATURE.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        default="collabsim",
        metavar="PROJECT",
        help="WandB project name (default: collabsim).",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        metavar="NAME",
        help="WandB run name (defaults to run_id).",
    )
    parser.add_argument(
        "--collaboration",
        nargs="?",
        const=True,
        default=False,
        metavar="BOOL",
        help="Append prompts/collaboration_module.md to each agent's initial prompt (default path; override via prompts.collaboration in YAML).",
    )
    args = parser.parse_args(argv)

    load_env_file()

    provider_override, name_override, temp_override = _resolve_model_cli_env(args)

    try:
        config = load_experiment_config(args.config)
        validate_config_schema(config)
    except (ConfigError, ConfigSchemaError) as exc:
        print(f"Config validation failed: {exc}", file=sys.stderr)
        return 1

    n_ov = apply_agent_model_overrides(
        config,
        provider=provider_override,
        name=name_override,
        temperature=temp_override,
    )
    if _model_override_requested(provider_override, name_override, temp_override) and n_ov > 0:
        print(f"Applied model overrides to {n_ov} agent(s).", file=sys.stderr)
    elif _model_override_requested(provider_override, name_override, temp_override) and n_ov == 0:
        print("Model overrides requested but no agents found in config.", file=sys.stderr)

    if not args.run_id.strip():
        print("run_id must be a non-empty string.", file=sys.stderr)
        return 1
    if args.output_dir is not None and not args.run_id:
        print("Output directory requires a non-empty run_id.", file=sys.stderr)
        return 1
    if args.max_steps is not None and args.max_steps <= 0:
        print("max_steps must be a positive integer.", file=sys.stderr)
        return 1
    if args.manifest_path and args.dry_run:
        print("manifest_path cannot be used with --dry-run.", file=sys.stderr)
        return 1

    if _coerce_bool(args.collaboration):
        experiment = config.setdefault("experiment", {})
        if isinstance(experiment, dict):
            experiment["collaboration"] = True

    if args.validate_only:
        print("Config validation passed.")
        return 0

    output_dir = args.output_dir
    if output_dir is None and not args.dry_run:
        output_dir = config.get("logging", {}).get("output_dir")

    agents = build_agents(config)
    state = ExperimentState(
        agents=agents,
        task_state={},
        resources={},
        turn_state={},
        buffers={},
    )
    registry = build_default_registry()
    try:
        probe_registry = load_probe_templates(args.probe_templates)
    except ProbeTemplateError as exc:
        print(f"Probe template load failed: {exc}", file=sys.stderr)
        return 1
    if args.wandb:
        _wandb_init(args, config)

    try:
        controller = build_controller(
            state,
            config=config,
            run_id=args.run_id,
            run_paths=None,
            output_dir=None if args.dry_run else output_dir,
            task_registry=registry,
            probe_registry=probe_registry,
        )
        controller.runtime_listener = _runtime_step_printer
        if args.print_actions:
            controller.runtime_listener = lambda name, payload: _runtime_progress_printer(name, payload, config)
            controller.event_listener = _action_event_printer
        if _is_time_mode(config):
            print("[t=0ms] simulation started", flush=True)
        controller.run(max_steps=args.max_steps)
    except (KeyError, ValueError) as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        _wandb_finish()
        return 1

    _wandb_finish()

    if args.manifest_path:
        manifest = build_run_manifest(
            config,
            run_id=args.run_id,
            probe_registry=probe_registry,
        )
        write_json(Path(args.manifest_path), manifest)

    print("Config validation passed.")
    return 0


def _runtime_step_printer(name: str, payload: dict[str, object]) -> None:
    if name != "step_start":
        return
    _update_last_sim_time(payload)
    print(f"[{_runtime_prefix(payload)}] starting", flush=True)


def _runtime_progress_printer(name: str, payload: dict[str, object], config: dict[str, object]) -> None:
    _update_last_sim_time(payload)
    if name == "step_start":
        round_phase = _daytrader_round_phase_suffix(payload, config)
        print(f"[{_runtime_prefix(payload)}] starting{round_phase}", flush=True)
        return
    if name == "probe_start":
        prefix = _runtime_prefix(payload)
        record_count = payload.get("record_count")
        request_count = payload.get("request_count")
        cadence = payload.get("cadence")
        print(
            f"[{prefix}] probe start cadence={cadence} records={record_count} requests={request_count}",
            flush=True,
        )
        return
    if name == "probe_progress":
        prefix = _runtime_prefix(payload)
        processed = payload.get("processed_count")
        request_count = payload.get("request_count")
        success = payload.get("success_count")
        error = payload.get("error_count")
        print(
            f"[{prefix}] probe progress {processed}/{request_count} success={success} error={error}",
            flush=True,
        )
        return
    if name == "probe_end":
        prefix = _runtime_prefix(payload)
        processed = payload.get("processed_count")
        request_count = payload.get("request_count")
        success = payload.get("success_count")
        error = payload.get("error_count")
        print(
            f"[{prefix}] probe end {processed}/{request_count} success={success} error={error}",
            flush=True,
        )
        return
    agent_id = payload.get("agent_id")
    prefix = _runtime_prefix(payload)
    if name == "context_update_start":
        print(f"[{prefix}] requesting action from agent={agent_id}", flush=True)
    elif name == "context_update_end":
        print(f"[{prefix}] received action proposal from agent={agent_id}", flush=True)


def _runtime_prefix(payload: dict[str, object]) -> str:
    sim_time_ms = payload.get("sim_time_ms")
    if isinstance(sim_time_ms, int):
        return f"t={sim_time_ms}ms"
    step_index = payload.get("step_index")
    return f"step={step_index}"


def _action_event_printer(event: dict[str, object]) -> None:
    event_type = event.get("event_type")
    actor_id = event.get("actor_id")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return
    if event_type not in {"action_validated", "action_rejected"}:
        return
    action = payload.get("action")
    if not isinstance(action, dict):
        return
    action_type = action.get("type")
    action_payload = action.get("payload", {})
    details = {
        "event_type": event_type,
        "actor_id": actor_id,
        "action_type": action_type,
        "action_payload": action_payload if isinstance(action_payload, dict) else {},
    }
    if event_type == "action_rejected":
        details["error_message"] = payload.get("error_message")
    prefix = _action_time_prefix()
    if prefix:
        print(f"[{prefix}] {json.dumps(details, ensure_ascii=False)}", flush=True)
    else:
        print(json.dumps(details, ensure_ascii=False), flush=True)


def _update_last_sim_time(payload: dict[str, object]) -> None:
    global _LAST_SIM_TIME_MS
    sim_time_ms = payload.get("sim_time_ms")
    if isinstance(sim_time_ms, int):
        _LAST_SIM_TIME_MS = sim_time_ms


def _action_time_prefix() -> str | None:
    if _LAST_SIM_TIME_MS is None:
        return None
    return f"t={_LAST_SIM_TIME_MS}ms"


def _daytrader_round_phase_suffix(payload: dict[str, object], config: dict[str, object]) -> str:
    round_index = payload.get("daytrader_round_index")
    phase = payload.get("daytrader_phase")
    if not isinstance(round_index, int) or round_index <= 0:
        task_cfg = config.get("task")
        protocol = config.get("protocol")
        if not isinstance(task_cfg, dict) or not isinstance(protocol, dict):
            return ""
        if task_cfg.get("type") != "daytrader" or protocol.get("step_mode") != "time":
            return ""
        sim_time_ms = payload.get("sim_time_ms")
        if not isinstance(sim_time_ms, int) or sim_time_ms < 0:
            return ""
        phase_rules = task_cfg.get("phase_rules")
        if not isinstance(phase_rules, dict):
            phase_rules = {}
        decision_duration_sec = phase_rules.get("decision_duration_sec", 15.0)
        group_chat_duration_sec = phase_rules.get("group_chat_duration_sec", 45.0)
        if not isinstance(decision_duration_sec, (int, float)) or float(decision_duration_sec) <= 0:
            decision_duration_sec = 15.0
        if not isinstance(group_chat_duration_sec, (int, float)) or float(group_chat_duration_sec) <= 0:
            group_chat_duration_sec = 45.0
        round_duration_sec = float(decision_duration_sec) + float(group_chat_duration_sec)
        if round_duration_sec <= 0:
            return ""
        elapsed_sec = float(sim_time_ms) / 1000.0
        round_index = int(elapsed_sec // round_duration_sec) + 1
        offset_sec = elapsed_sec % round_duration_sec
        phase = "decision" if offset_sec < float(decision_duration_sec) else "group_chat"
    if phase in {"decision", "group_chat"}:
        return f" round={round_index} phase={phase}"
    return f" round={round_index}"


def _is_time_mode(config: dict[str, object]) -> bool:
    protocol = config.get("protocol")
    if not isinstance(protocol, dict):
        return False
    return protocol.get("step_mode") == "time"


def _coerce_bool(value: object) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


def _nonempty_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    stripped = value.strip()
    return stripped if stripped else None


def _env_optional_str(key: str) -> str | None:
    raw = os.environ.get(key)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped else None


def _env_optional_float(key: str) -> float | None:
    raw = os.environ.get(key)
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        print(f"Warning: ignoring invalid float in {key}={raw!r}.", file=sys.stderr)
        return None


def _resolve_model_cli_env(args: argparse.Namespace) -> tuple[str | None, str | None, float | None]:
    provider = _nonempty_str(getattr(args, "model_provider", None)) or _env_optional_str(
        "COLLABSIM_MODEL_PROVIDER"
    )
    name = _nonempty_str(getattr(args, "model_name", None)) or _env_optional_str("COLLABSIM_MODEL_NAME")
    temp_override: float | None = getattr(args, "model_temperature", None)
    if temp_override is None:
        temp_override = _env_optional_float("COLLABSIM_MODEL_TEMPERATURE")
    return provider, name, temp_override


def _model_override_requested(
    provider: str | None,
    name: str | None,
    temperature: float | None,
) -> bool:
    return provider is not None or name is not None or temperature is not None


def _wandb_init(args: argparse.Namespace, config: dict) -> None:
    try:
        import wandb  # type: ignore[import-untyped]
    except ImportError:
        print("wandb not installed — skipping WandB logging.", file=sys.stderr)
        return
    task_cfg = config.get("task") or {}
    protocol_cfg = config.get("protocol") or {}
    agents_cfg = config.get("agents") or []
    agent_model = next(
        (a.get("model", {}).get("name") for a in agents_cfg if isinstance(a, dict)), None
    )
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or args.run_id,
        config={
            "run_id": args.run_id,
            "task_type": task_cfg.get("type"),
            "step_mode": protocol_cfg.get("step_mode"),
            "agent_model": agent_model,
            "num_agents": len(agents_cfg),
            "config_path": args.config,
            **{k: v for k, v in task_cfg.items() if isinstance(v, (str, int, float, bool))},
        },
    )
    print(f"WandB run initialized: {wandb.run.url}", flush=True)


def _wandb_finish() -> None:
    try:
        import wandb  # type: ignore[import-untyped]
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
