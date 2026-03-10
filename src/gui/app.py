"""Minimal Flask GUI for CollabSim experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import zipfile

from flask import Flask, redirect, render_template, request, send_file, url_for, Response
from werkzeug.utils import secure_filename

from src.agents.factory import build_agents
from src.config.loader import ConfigError, load_experiment_config
from src.config.schema import ConfigSchemaError, validate_config_schema
from src.controller.controller import ExperimentState
from src.controller.factory import build_controller
from src.probe.loader import ProbeTemplateError, load_probe_templates
from src.tasks.registry import build_default_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "experiments" / "gui_runs"
TEMPLATES_DIR = REPO_ROOT / "src" / "gui" / "templates"
CONFIGS_DIR = REPO_ROOT / "configs"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))


def _safe_run_id(raw: str | None) -> str:
    value = (raw or "").strip()
    if value:
        return secure_filename(value)
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _list_presets() -> list[str]:
    if not CONFIGS_DIR.exists():
        return []
    presets = []
    for path in CONFIGS_DIR.glob("*.yml"):
        presets.append(path.name)
    for path in CONFIGS_DIR.glob("*.yaml"):
        presets.append(path.name)
    return sorted(set(presets))


def _resolve_preset(preset: str) -> Path | None:
    if not preset:
        return None
    allowed = _list_presets()
    if preset not in allowed:
        return None
    path = CONFIGS_DIR / preset
    if not path.exists():
        return None
    return path


def _resolve_config_payload(yaml_text: str, preset_name: str, upload) -> bytes | None:
    if yaml_text.strip():
        return yaml_text.encode("utf-8")
    if upload is not None and upload.filename:
        return upload.read()
    preset_path = _resolve_preset(preset_name)
    if preset_path is not None:
        return preset_path.read_bytes()
    return None


def _load_artifacts(output_dir: Path) -> dict[str, object]:
    events = _read_jsonl(output_dir / "events.jsonl")
    probes = _read_jsonl(output_dir / "probes.jsonl")
    metrics_path = output_dir / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = {}
    manifest_path = output_dir / "run_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    return {
        "events": events,
        "probes": probes,
        "metrics": metrics,
        "manifest": manifest,
    }


def _summarize_events(events: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    inputs = [event for event in events if event.get("event_type") == "observation_built"]
    outputs = [event for event in events if event.get("event_type") == "context_update"]
    responses = [event for event in events if event.get("event_type") == "message_delivered"]
    return {"inputs": inputs, "outputs": outputs, "responses": responses}


def _summarize_interviews(probes: list[dict[str, object]]) -> list[dict[str, object]]:
    asked = {
        record.get("probe_id"): record
        for record in probes
        if record.get("actor_id") is None
    }
    answered = [record for record in probes if record.get("actor_id") is not None]
    interviews: list[dict[str, object]] = []
    for probe_id, question in asked.items():
        responses = [record for record in answered if record.get("probe_id") == probe_id]
        interviews.append(
            {
                "probe_id": probe_id,
                "prompt": question.get("prompt"),
                "about_agent_id": question.get("about_agent_id"),
                "responses": responses,
            }
        )
    return interviews


def _write_run_meta(run_dir: Path, output_dir: Path, config_path: Path) -> None:
    payload = {
        "run_id": run_dir.name,
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_run_meta(run_dir: Path) -> dict[str, object]:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@app.get("/")
def index() -> str:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs = sorted([path.name for path in RUNS_DIR.iterdir() if path.is_dir()])
    presets = _list_presets()
    return render_template(
        "index.html",
        runs=runs,
        presets=presets,
        default_output_dir=str(RUNS_DIR),
    )


@app.get("/presets/<preset_name>")
def preset_view(preset_name: str) -> Response:
    preset_path = _resolve_preset(preset_name)
    if preset_path is None:
        return Response("Preset not found.", status=404, mimetype="text/plain")
    return Response(preset_path.read_text(encoding="utf-8"), mimetype="text/plain")


@app.post("/run")
def run_experiment() -> str:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    upload = request.files.get("config")
    preset_name = (request.form.get("preset") or "").strip()
    yaml_text = request.form.get("yaml_text") or ""
    payload = _resolve_config_payload(yaml_text, preset_name, upload)
    if payload is None:
        return render_template(
            "error.html",
            message="Config is required (editor, upload, or preset).",
        )
    run_id = _safe_run_id(request.form.get("run_id"))
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yml"
    config_path.write_bytes(payload)
    output_dir_value = (request.form.get("output_dir") or "").strip()
    output_dir = Path(output_dir_value) if output_dir_value else run_dir
    validate_only = request.form.get("validate_only") == "on"
    dry_run = request.form.get("dry_run") == "on"
    max_steps_raw = (request.form.get("max_steps") or "").strip()
    if max_steps_raw:
        try:
            max_steps = int(max_steps_raw)
        except ValueError:
            return render_template("error.html", message="Max steps must be a positive integer.")
    else:
        max_steps = None
    probe_templates = (request.form.get("probe_templates") or "").strip()
    probe_templates = probe_templates or "configs/probe_templates.yml"
    manifest_path_raw = (request.form.get("manifest_path") or "").strip()
    if manifest_path_raw and dry_run:
        return render_template(
            "error.html",
            message="Manifest path cannot be used with dry run.",
        )

    try:
        config = load_experiment_config(str(config_path))
        validate_config_schema(config)
    except (ConfigError, ConfigSchemaError) as exc:
        return render_template("error.html", message=f"Config validation failed: {exc}")

    if validate_only:
        _write_run_meta(run_dir, output_dir, config_path)
        return redirect(url_for("run_view", run_id=run_id))

    agents = build_agents(config)
    state = ExperimentState(
        agents=agents,
        task_state={},
        resources={},
        turn_state={},
        buffers={},
    )
    registry = build_default_registry()
    try:
        probe_registry = load_probe_templates(probe_templates)
    except ProbeTemplateError as exc:
        return render_template("error.html", message=f"Probe template load failed: {exc}")

    try:
        controller = build_controller(
            state,
            config=config,
            run_id=run_id,
            run_paths=None,
            output_dir=None if dry_run else str(output_dir),
            task_registry=registry,
            probe_registry=probe_registry,
        )
        controller.run(max_steps=max_steps)
    except (KeyError, ValueError) as exc:
        return render_template("error.html", message=f"Run failed: {exc}")

    if manifest_path_raw and not dry_run:
        try:
            from src.data.logging import build_run_manifest, write_json

            manifest = build_run_manifest(config, run_id=run_id, probe_registry=probe_registry)
            write_json(Path(manifest_path_raw), manifest)
        except (OSError, ValueError) as exc:
            return render_template("error.html", message=f"Manifest write failed: {exc}")

    _write_run_meta(run_dir, output_dir, config_path)
    return redirect(url_for("run_view", run_id=run_id))


@app.get("/runs/<run_id>")
def run_view(run_id: str) -> str:
    run_dir = RUNS_DIR / run_id
    meta = _load_run_meta(run_dir)
    output_dir_raw = request.args.get("output_dir") or meta.get("output_dir")
    output_dir = Path(output_dir_raw) if output_dir_raw else run_dir
    artifacts = _load_artifacts(output_dir)
    events = artifacts.get("events", [])
    probes = artifacts.get("probes", [])
    summaries = _summarize_events(events)
    interviews = _summarize_interviews(probes)
    return render_template(
        "run.html",
        run_id=run_id,
        output_dir=str(output_dir),
        summaries=summaries,
        interviews=interviews,
        metrics=artifacts.get("metrics", {}),
        manifest=artifacts.get("manifest", {}),
        config_path=meta.get("config_path"),
    )


@app.get("/runs/<run_id>/export")
def export_run(run_id: str):
    run_dir = RUNS_DIR / run_id
    meta = _load_run_meta(run_dir)
    output_dir_raw = request.args.get("output_dir") or meta.get("output_dir")
    output_dir = Path(output_dir_raw) if output_dir_raw else run_dir
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("events.jsonl", "probes.jsonl", "metrics.json", "run_manifest.json"):
            path = output_dir / name
            if path.exists():
                archive.write(path, arcname=name)
        config_path = Path(meta.get("config_path")) if meta.get("config_path") else None
        if config_path and config_path.exists():
            archive.write(config_path, arcname="config.yml")
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            archive.write(meta_path, arcname="run_meta.json")
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{run_id}_artifacts.zip",
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=True, port=port)
