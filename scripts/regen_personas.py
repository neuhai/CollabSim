"""Regenerate persona descriptions using Azure GPT-5.5 Responses API."""

import json
import os
import time
import urllib.request

API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
ENDPOINT = os.environ.get(
    "AZURE_OPENAI_RESPONSES_ENDPOINT",
    "https://bosu-mljmcbpq-eastus2.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview",
)
MODEL = os.environ.get("AZURE_OPENAI_MODEL", "gpt-5.5")

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


def call_api(persona: dict) -> str:
    persona_input = {k: v for k, v in persona.items() if k != "description"}
    payload = json.dumps({
        "model": MODEL,
        "instructions": SYSTEM_PROMPT,
        "input": json.dumps(persona_input, indent=2),
    }).encode("utf-8")

    req = urllib.request.Request(
        ENDPOINT,
        data=payload,
        headers={
            "api-key": API_KEY,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    return part["text"].strip()
    raise ValueError(f"Unexpected response structure: {list(data.keys())}")


def main() -> None:
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        personas = json.load(f)

    for i, persona in enumerate(personas):
        print(f"[{i + 1}/{len(personas)}] generating...", flush=True)
        for attempt in range(3):
            try:
                desc = call_api(persona)
                persona["description"] = desc
                print(f"  -> {desc[:90]}...", flush=True)
                break
            except Exception as exc:
                print(f"  attempt {attempt + 1} failed: {exc}", flush=True)
                if attempt < 2:
                    time.sleep(2)
        time.sleep(0.3)

    with open(PERSONAS_PATH, "w", encoding="utf-8") as f:
        json.dump(personas, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Wrote {len(personas)} personas to {PERSONAS_PATH}", flush=True)


if __name__ == "__main__":
    main()
