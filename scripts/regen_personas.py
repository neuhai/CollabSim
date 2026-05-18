"""Regenerate persona descriptions using Azure OpenAI (Responses API, chat fallback)."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_REPO_ROOT))
from generate_personas import resample_pool_genders
from src.utils.env import load_env_file

SYSTEM_PROMPT = (
    "You are given a structured persona defined by demographic information and "
    "psychological dimensions (Big Five). Your task is to convert this structured "
    "persona into a concrete, vivid, free-form description. The description should "
    "elaborate only on the provided attributes and traits, explaining how this specific "
    "combination may appear in everyday behavior, communication style, decision-making, "
    "and collaboration. Do not invent any information that is not explicitly listed, "
    "such as a name, nationality, occupation, background, or additional personality "
    "traits. The output should begin with \"You are\" and should be written as a single "
    "short paragraph."
)

PERSONAS_PATH = "prompts/persona_profiles.json"


def _azure_ssl_context() -> ssl.SSLContext:
    verify = os.environ.get("AZURE_OPENAI_SSL_VERIFY", "true").strip().lower()
    if verify in ("0", "false", "no"):
        return ssl._create_unverified_context()
    return ssl.create_default_context()


def _http_post(request: urllib.request.Request, *, timeout: int = 60) -> str:
    ctx = _azure_ssl_context()
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ctx) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Azure OpenAI API error ({exc.code}): {body}") from exc


def _extract_responses_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for chunk in content:
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("type") not in {"output_text", "text"}:
                    continue
                text = chunk.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return json.dumps(payload)


def _extract_chat_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content
    text = payload.get("text")
    if isinstance(text, str):
        return text
    return json.dumps(payload)


def _load_env() -> None:
    """Load repo-root and scripts/.env without overwriting existing env vars."""
    load_env_file(_REPO_ROOT / ".env")
    load_env_file(_SCRIPT_DIR / ".env")


def _resolve_api_config() -> tuple[str, str, str, str]:
    _load_env()
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing AZURE_OPENAI_API_KEY. Set it in your shell or in "
            f"{_SCRIPT_DIR / '.env'} (or {_REPO_ROOT / '.env'})."
        )

    base = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    if not base:
        raise SystemExit(
            "Missing AZURE_OPENAI_ENDPOINT. Set it in scripts/.env "
            "(e.g. https://<resource>.cognitiveservices.azure.com/)."
        )

    # Azure expects the deployment name in ``model``, not a marketing label.
    deployment = (
        os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        or os.environ.get("AZURE_OPENAI_MODEL")
        or os.environ.get("COLLABSIM_MODEL_NAME")
        or "gpt-5.5"
    )
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    return api_key, base, deployment, api_version


def _persona_messages(persona: dict) -> list[dict[str, str]]:
    persona_input = {k: v for k, v in persona.items() if k != "description"}
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(persona_input, indent=2)},
    ]


def _responses_input(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"System:\n{content}")
        elif role == "user":
            parts.append(f"User:\n{content}")
        else:
            parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def call_api(
    persona: dict,
    *,
    api_key: str,
    base: str,
    deployment: str,
    api_version: str,
) -> str:
    messages = _persona_messages(persona)
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
    }

    responses_url = f"{base}/openai/v1/responses"
    responses_request = urllib.request.Request(
        responses_url,
        data=json.dumps({
            "model": deployment,
            "input": _responses_input(messages),
        }).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        body = _http_post(responses_request)
        return _extract_responses_text(json.loads(body)).strip()
    except RuntimeError as responses_exc:
        responses_err = str(responses_exc)

    chat_url = (
        f"{base}/openai/deployments/{deployment}/chat/completions"
        f"?api-version={api_version}"
    )
    chat_request = urllib.request.Request(
        chat_url,
        data=json.dumps({"messages": messages}).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        body = _http_post(chat_request)
        return _extract_chat_text(json.loads(body)).strip()
    except RuntimeError as chat_exc:
        raise RuntimeError(
            f"Responses API failed: {responses_err}\n"
            f"Chat completions failed: {chat_exc}"
        ) from chat_exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--personas",
        default=PERSONAS_PATH,
        help=f"Persona JSON path (default: {PERSONAS_PATH})",
    )
    parser.add_argument(
        "--gender-seed",
        type=int,
        default=42,
        help="Seed for resampling genders to population-approximate ratios (default: 42)",
    )
    parser.add_argument(
        "--keep-genders",
        action="store_true",
        help="Keep existing gender labels instead of resampling",
    )
    parser.add_argument(
        "--min-non-binary",
        type=int,
        default=None,
        help="Minimum non-binary personas when resampling (default: 1 when pool size>=5)",
    )
    args = parser.parse_args()

    api_key, base, deployment, api_version = _resolve_api_config()
    print(f"Using deployment={deployment}  endpoint={base}", flush=True)

    with open(args.personas, encoding="utf-8") as f:
        personas = json.load(f)

    if not args.keep_genders:
        resample_pool_genders(
            personas,
            seed=args.gender_seed,
            min_non_binary=args.min_non_binary,
        )
        counts = Counter(p["gender"] for p in personas)
        print(
            "Resampled genders (~49.5% male / ~49.5% female; ≥1 non-binary when n≥5):",
            dict(counts),
            flush=True,
        )

    for i, persona in enumerate(personas):
        print(f"[{i + 1}/{len(personas)}] generating...", flush=True)
        for attempt in range(3):
            try:
                desc = call_api(
                    persona,
                    api_key=api_key,
                    base=base,
                    deployment=deployment,
                    api_version=api_version,
                )
                persona["description"] = desc
                print(f"  -> {desc[:90]}...", flush=True)
                break
            except Exception as exc:
                print(f"  attempt {attempt + 1} failed: {exc}", flush=True)
                if attempt < 2:
                    time.sleep(2)
        time.sleep(0.3)

    with open(args.personas, "w", encoding="utf-8") as f:
        json.dump(personas, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Wrote {len(personas)} personas to {args.personas}", flush=True)


if __name__ == "__main__":
    main()
