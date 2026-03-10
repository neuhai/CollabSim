"""Experiment configuration loader for reproducible simulations.

This module centralizes YAML parsing and minimal validation to ensure that
experiment configurations are structurally complete before execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConfigError(Exception):
    """Raised when a configuration file is invalid."""

    message: str

    def __str__(self) -> str:
        return self.message


REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "experiment",
    "agents",
    "action_space",
    "controls",
    "task",
    "protocol",
    "probe",
    "logging",
)


def load_experiment_config(path: str) -> dict[str, Any]:
    """Load and minimally validate an experiment YAML configuration.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        ConfigError: If the YAML cannot be parsed or required keys are missing.
    """

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ConfigError(
            "PyYAML is required to load configuration files. "
            "Install with `pip install pyyaml`."
        ) from exc

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"Failed to read config file: {exc}") from exc
    except yaml.YAMLError as exc:  # type: ignore[attr-defined]
        raise ConfigError(f"Failed to parse YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Config root must be a mapping/object.")

    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in data]
    if missing:
        raise ConfigError(f"Config missing required keys: {', '.join(missing)}")

    probe_cfg = data.get("probe")
    if not isinstance(probe_cfg, dict):
        raise ConfigError("probe must be an object.")
    templates = probe_cfg.get("templates")
    if templates is not None:
        if not isinstance(templates, list) or not templates:
            raise ConfigError("probe.templates must be a non-empty list when provided.")
        if not all(isinstance(item, str) and item for item in templates):
            raise ConfigError("probe.templates must contain only non-empty strings.")

    return data
