"""Extract JSON object dicts from LLM text output (markdown fences, prose wrappers, etc.)."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

_CODE_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\r?\n?(.*?)\r?\n?```",
    re.DOTALL,
)

_PREFERRED_KEYS = ("action", "actions", "answer", "structured_fields", "rationale")


def parse_json_dict(text: str) -> dict[str, Any]:
    """Return the best matching dict parsed from *text*, or ``{}`` if none found."""

    if not isinstance(text, str) or not text.strip():
        return {}

    seen: set[str] = set()
    candidates: list[str] = []

    def _add(candidate: str) -> None:
        stripped = candidate.strip()
        if not stripped or stripped in seen:
            return
        seen.add(stripped)
        candidates.append(stripped)

    stripped = text.strip()
    _add(stripped)

    for block in _CODE_FENCE_RE.findall(text):
        _add(block)

    for snippet in _brace_delimited_snippets(text):
        _add(snippet)

    parsed_dicts: list[dict[str, Any]] = []
    for candidate in candidates:
        parsed = _loads_dict(candidate)
        if isinstance(parsed, dict):
            parsed_dicts.append(parsed)

    if not parsed_dicts:
        return {}

    for key in _PREFERRED_KEYS:
        for parsed in reversed(parsed_dicts):
            if key in parsed:
                return parsed

    return parsed_dicts[-1]


def _loads_dict(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return parsed
    except (SyntaxError, ValueError):
        pass

    return None


def _brace_delimited_snippets(text: str) -> list[str]:
    """Find balanced ``{...}`` regions, respecting JSON double-quoted strings."""

    snippets: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        if text[i] != "{":
            i += 1
            continue
        start = i
        depth = 0
        in_string = False
        escape = False
        closed_at: int | None = None
        for j in range(i, length):
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    closed_at = j
                    break
        if closed_at is None:
            break
        snippets.append(text[start : closed_at + 1])
        i = closed_at + 1
    return snippets
