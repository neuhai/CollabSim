"""Interviewer probe interface and minimal implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProbeRequest:
    """Structured probe request for mental state elicitation."""

    probe_id: str
    prompt_version: str
    construct: str
    question: str
    context: dict[str, Any]


@dataclass(frozen=True)
class ProbeResponse:
    """Structured probe response from an agent."""

    probe_id: str
    answer: Any
    confidence: float | None = None
    structured_fields: dict[str, Any] | None = None


class ProbeInterface:
    """Minimal probe interface for constructing requests and responses."""

    def build_request(
        self,
        probe_id: str,
        prompt_version: str,
        construct: str,
        question: str,
        context: dict[str, Any],
    ) -> ProbeRequest:
        return ProbeRequest(
            probe_id=probe_id,
            prompt_version=prompt_version,
            construct=construct,
            question=question,
            context=context,
        )

    def build_response(
        self,
        probe_id: str,
        answer: Any,
        confidence: float | None = None,
        structured_fields: dict[str, Any] | None = None,
    ) -> ProbeResponse:
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be within [0, 1]")
        return ProbeResponse(
            probe_id=probe_id,
            answer=answer,
            confidence=confidence,
            structured_fields=structured_fields,
        )
