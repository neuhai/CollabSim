"""Probe template loader for registry initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.probe.registry import ProbeRegistry, ProbeTemplate


@dataclass(frozen=True)
class ProbeTemplateError(Exception):
    """Raised when probe template loading or validation fails."""

    message: str

    def __str__(self) -> str:
        return self.message


def load_probe_templates(path: str) -> ProbeRegistry:
    """Load probe templates from a YAML file into a registry.

    Args:
        path: Path to a YAML file containing probe templates.

    Returns:
        ProbeRegistry with templates registered.

    Raises:
        ProbeTemplateError: If parsing fails or template data is invalid.

    Research rationale:
        Centralizes prompt versioning and validation to ensure probe
        prompts are reproducible across runs and models.
    """

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProbeTemplateError(
            "PyYAML is required to load probe templates. Install dependencies with `uv sync`."
        ) from exc

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except OSError as exc:
        raise ProbeTemplateError(f"Failed to read probe templates: {exc}") from exc
    except yaml.YAMLError as exc:  # type: ignore[attr-defined]
        raise ProbeTemplateError(f"Failed to parse probe templates YAML: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ProbeTemplateError("Probe templates root must be an object.")
    templates = payload.get("templates")
    if not isinstance(templates, list) or not templates:
        raise ProbeTemplateError("Probe templates must define a non-empty templates list.")

    registry = ProbeRegistry()
    for item in templates:
        _register_template(item, registry)
    return registry


def _register_template(item: Any, registry: ProbeRegistry) -> None:
    if not isinstance(item, Mapping):
        raise ProbeTemplateError("Each template entry must be an object.")
    template_id = item.get("template_id")
    construct = item.get("construct")
    version = item.get("version")
    prompt = item.get("prompt")
    if not isinstance(template_id, str) or not template_id:
        raise ProbeTemplateError("template_id must be a non-empty string.")
    if not isinstance(construct, str) or not construct:
        raise ProbeTemplateError("construct must be a non-empty string.")
    if not isinstance(version, str) or not version:
        raise ProbeTemplateError("version must be a non-empty string.")
    if not isinstance(prompt, str) or not prompt:
        raise ProbeTemplateError("prompt must be a non-empty string.")
    registry.register(
        ProbeTemplate(
            template_id=template_id,
            construct=construct,
            prompt=prompt,
            version=version,
        )
    )
