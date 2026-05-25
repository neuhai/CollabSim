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
        denorm = _denormalize_template_braces(stripped)
        if denorm is not None:
            _add(denorm)

    stripped = text.strip()
    _add(stripped)

    denorm_text = _denormalize_template_braces(text)
    scan_text = denorm_text if denorm_text is not None else text

    for block in _CODE_FENCE_RE.findall(scan_text):
        _add(block)

    for snippet in _brace_delimited_snippets(scan_text):
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

    peeled = _peel_double_brace_wrapper(raw)
    if peeled is not None and peeled != raw:
        inner = _loads_dict(peeled)
        if inner is not None:
            return inner

    if _looks_like_python_dict(raw):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return parsed
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            pass

    return None


def _denormalize_template_braces(raw: str) -> str | None:
    """Convert prompt-style ``{{`` / ``}}`` escapes to JSON ``{`` / ``}``."""

    if "{{" not in raw and "}}" not in raw:
        return None
    denorm = raw.replace("{{", "{").replace("}}", "}")
    if denorm == raw:
        return None
    return denorm


def _peel_double_brace_wrapper(raw: str) -> str | None:
    """If output looks like ``{ { ... } }``, return the inner ``{ ... }`` snippet."""

    stripped = raw.strip()
    if not re.match(r"^\{\s*\{", stripped):
        return None
    inner_start = stripped.find("{", 1)
    if inner_start == -1:
        return None
    inner_snips = _brace_delimited_snippets(stripped[inner_start:])
    if inner_snips:
        return inner_snips[0]
    return None


def _looks_like_python_dict(raw: str) -> bool:
    """Skip patterns that ast treats as set literals, e.g. ``{ {...} }``."""

    stripped = raw.strip()
    if not stripped.startswith("{"):
        return False
    # ``{ { ... } }`` is a set literal in Python, not a dict — literal_eval raises TypeError.
    if re.match(r"\{\s*\{", stripped):
        return False
    return ":" in stripped


def _brace_delimited_snippets(text: str) -> list[str]:
    """Find balanced ``{...}`` regions, respecting JSON double-quoted strings."""

    snippets: list[str] = []
    length = len(text)
    for i in range(length):
        if text[i] != "{":
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
        if closed_at is not None:
            snippets.append(text[start : closed_at + 1])
    return snippets
