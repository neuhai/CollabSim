"""Minimal CLI entrypoint for CollabSim experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config.loader import ConfigError, load_experiment_config
from src.config.schema import ConfigSchemaError, validate_config_schema
from src.controller.controller import ExperimentState
from src.controller.factory import build_controller
from src.data.logging import build_run_manifest, write_json
from src.probe.loader import ProbeTemplateError, load_probe_templates
from src.agents.factory import build_agents
from src.tasks.registry import build_default_registry


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
        "--manifest-path",
        default=None,
        help="Override run manifest output path (JSON)",
    )
    parser.add_argument(
        "--probe-templates",
        default="configs/probe_templates.yml",
        help="Path to probe template YAML for registry initialization",
    )
    args = parser.parse_args(argv)

    try:
        config = load_experiment_config(args.config)
        validate_config_schema(config)
    except (ConfigError, ConfigSchemaError) as exc:
        print(f"Config validation failed: {exc}", file=sys.stderr)
        return 1

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
        controller.run(max_steps=args.max_steps)
    except (KeyError, ValueError) as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    if args.manifest_path:
        manifest = build_run_manifest(
            config,
            run_id=args.run_id,
            probe_registry=probe_registry,
        )
        write_json(Path(args.manifest_path), manifest)

    print("Config validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
