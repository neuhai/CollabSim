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


def test_parse_template_doubled_braces_from_prompt_examples() -> None:
    raw = (
        '{{\n'
        '  "action": {{\n'
        '    "type": "message",\n'
        '    "payload": {{\n'
        '      "channel": "direct",\n'
        '      "recipients": ["B"],\n'
        '      "content": "Start at S.",\n'
        '      "content_type": "text"\n'
        '    }}\n'
        '  }},\n'
        '  "rationale": "Initial instruction."\n'
        '}}'
    )
    parsed = parse_json_dict(raw)
    assert parsed["action"]["type"] == "message"
    assert parsed["action"]["payload"]["channel"] == "direct"


def test_parse_prose_then_json_fence_with_nested_closing_braces() -> None:
    raw = (
        "Analysis with it's and 'S' at [6, 34] ...\n\n"
        "Here's my action:\n\n"
        "```json\n"
        "{\n"
        '  "action": {"type": "draw", "payload": {"cells": [[7, 34]]}},\n'
        '  "rationale": "Starting by moving down from \'S\' at [6, 34] as it\'s valid."\n'
        "}\n"
        "```"
    )
    parsed = parse_json_dict(raw)
    assert parsed["action"]["type"] == "draw"
    assert parsed["action"]["payload"]["cells"] == [[7, 34]]


def test_coerce_bare_action_object() -> None:
    raw = '{"type": "draw", "payload": {"cells": [[7, 34]]}}'
    parsed = parse_json_dict(raw)
    assert parsed["action"]["type"] == "draw"


def test_parse_batched_probe_envelope_not_last_response_row() -> None:
    raw = (
        '{"answer":{"responses":['
        '{"probe_id":"probe_2_1","answer":"first","confidence":1.0,"structured_fields":{}},'
        '{"probe_id":"probe_2_2","answer":"second","confidence":0.8,"structured_fields":{}},'
        '{"probe_id":"probe_2_3","answer":"third","confidence":0.9,"structured_fields":{}}'
        "]}}"
    )
    parsed = parse_json_dict(raw)
    assert isinstance(parsed.get("answer"), dict)
    responses = parsed["answer"]["responses"]
    assert len(responses) == 3
    assert responses[0]["probe_id"] == "probe_2_1"
