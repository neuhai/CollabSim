"""Probe template registry and response validators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from src.probe.probe import ProbeResponse


@dataclass(frozen=True)
class ProbeTemplate:
    """Metadata for a probe prompt template."""

    template_id: str
    construct: str
    prompt: str
    version: str


@dataclass(frozen=True)
class ProbeValidationError(Exception):
    """Raised when probe response validation fails."""

    message: str

    def __str__(self) -> str:
        return self.message


class ProbeRegistry:
    """Registry for probe templates and versions."""

    def __init__(self) -> None:
        self._templates: dict[str, ProbeTemplate] = {}

    def register(self, template: ProbeTemplate) -> None:
        if not template.template_id:
            raise ProbeValidationError("template_id must be non-empty.")
        self._templates[template.template_id] = template

    def get(self, template_id: str) -> ProbeTemplate:
        if template_id not in self._templates:
            raise KeyError(f"Probe template not registered: {template_id}")
        return self._templates[template_id]

    def list(self) -> list[ProbeTemplate]:
        return list(self._templates.values())

    def validate_response(self, template_id: str, response: ProbeResponse) -> None:
        """Validate a probe response against the template's construct.

        Args:
            template_id: Registered template identifier.
            response: Structured probe response from an agent.

        Raises:
            KeyError: If the template_id is not registered.
            ProbeValidationError: If the response violates constraints.
        """

        template = self.get(template_id)
        validate_probe_response(response, template.construct)


def validate_probe_response(response: ProbeResponse, construct: str) -> None:
    """Validate a probe response against minimal structural rules.

    Args:
        response: Structured probe response from an agent.
        construct: Construct label (e.g., grounding).

    Raises:
        ProbeValidationError: If the response violates constraints.
    """

    if not response.probe_id:
        raise ProbeValidationError("probe_id must be non-empty.")
    if response.answer is None:
        raise ProbeValidationError("answer must be provided.")
    if response.confidence is not None and not (0.0 <= response.confidence <= 1.0):
        raise ProbeValidationError("confidence must be within [0, 1].")
    if response.structured_fields is not None and not isinstance(response.structured_fields, dict):
        raise ProbeValidationError("structured_fields must be an object when provided.")
    _validate_construct(construct)


def _validate_construct(construct: str) -> None:
    allowed = {"grounding", "situation_awareness", "coordination"}
    if construct not in allowed:
        raise ProbeValidationError(f"Unsupported construct: {construct}")
