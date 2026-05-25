"""Tests for LLM output JSON dict extraction."""

from src.agents.parse_model_json import parse_json_dict


def test_parse_plain_json_object() -> None:
    raw = '{"action": {"type": "do_nothing", "payload": {"reason": "ok"}}}'
    parsed = parse_json_dict(raw)
    assert parsed["action"]["type"] == "do_nothing"


def test_parse_json_markdown_fence() -> None:
    raw = """Here is my response:

```json
{
  "action": {"type": "message", "payload": {"channel": "direct", "content": "hi"}},
  "rationale": "start"
}
```
"""
    parsed = parse_json_dict(raw)
    assert parsed["action"]["type"] == "message"
    assert parsed["rationale"] == "start"


def test_parse_dict_embedded_in_prose() -> None:
    raw = (
        "I'll send instructions now.\n"
        '{"action": {"type": "message", "payload": {"channel": "direct", "content": "go east"}}, '
        '"rationale": "bootstrap"}'
    )
    parsed = parse_json_dict(raw)
    assert parsed["action"]["type"] == "message"


def test_parse_python_dict_literal() -> None:
    raw = "{'action': {'type': 'do_nothing', 'payload': {'reason': 'wait'}}}"
    parsed = parse_json_dict(raw)
    assert parsed["action"]["type"] == "do_nothing"


def test_parse_prefers_action_dict_when_multiple_objects() -> None:
    raw = '{"note": "ignore"} and then {"action": {"type": "do_nothing", "payload": {}}}'
    parsed = parse_json_dict(raw)
    assert "action" in parsed


def test_parse_empty_when_no_dict() -> None:
    assert parse_json_dict("no structured output here") == {}


def test_parse_nested_brace_wrapper_does_not_crash() -> None:
    raw = '{ { "action": {"type": "do_nothing", "payload": {}}, "rationale": "x" } }'
    parsed = parse_json_dict(raw)
    assert parsed["action"]["type"] == "do_nothing"
