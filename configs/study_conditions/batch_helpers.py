"""Helpers for run_task_batch.sh (condition skip/retry and W&B uploads)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def should_skip_condition(out_dir: str, mode: str, *, retry_failed: bool = False) -> bool:
    """Return True if the batch runner should skip this condition."""

    summary_path = os.path.join(out_dir, "run_summary.json")
    if not os.path.isfile(summary_path):
        return False
    try:
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    run_id = str(summary.get("run_id", ""))
    is_smoke_run = "smoke" in run_id
    if mode == "smoke":
        return is_smoke_run
    if is_smoke_run:
        return False
    if summary.get("complete") is True:
        return True
    if retry_failed:
        return False
    actions_path = os.path.join(out_dir, "actions.jsonl")
    return os.path.isfile(actions_path) and os.path.getsize(actions_path) > 64


def list_failed_conditions(
    root: Path,
    task: str,
    *,
    collaboration: bool = False,
) -> list[tuple[str, str]]:
    """Return (slug, out_dir) for conditions without complete run_summary."""

    cfg_dir = root / "configs" / "study_conditions" / task
    out_base = root / "experiments" / "study_conditions" / task
    failed: list[tuple[str, str]] = []
    for cfg in sorted(cfg_dir.glob("*.yml")):
        slug = cfg.stem
        out_name = f"{slug}_collab" if collaboration else slug
        out_dir = out_base / out_name
        summary_path = out_dir / "run_summary.json"
        complete = False
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                complete = summary.get("complete") is True
            except (OSError, json.JSONDecodeError):
                complete = False
        if not complete:
            failed.append((slug, str(out_dir)))
    return failed


def download_artifact_from_wandb(
    artifact_ref: str,
    *,
    artifact_type: str = "study_conditions",
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download a W&B artifact and return the local directory path."""

    try:
        import wandb  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("wandb is not installed") from exc

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_CACHE_DIR", str(cache_dir))

    api = wandb.Api()
    artifact = api.artifact(artifact_ref, type=artifact_type)
    download_root = str(output_dir) if output_dir is not None else None
    local_path = Path(artifact.download(root=download_root))
    return local_path


def upload_directory_to_wandb(
    path: Path,
    *,
    project: str,
    run_name: str,
    artifact_name: str,
    artifact_type: str = "study_conditions",
    metadata: dict[str, str] | None = None,
) -> str:
    try:
        import wandb  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("wandb is not installed") from exc

    if not path.is_dir():
        raise FileNotFoundError(f"Upload path not found: {path}")

    config = dict(metadata or {})
    config["upload_path"] = str(path)
    run = wandb.init(
        project=project,
        job_type="batch-upload",
        name=run_name,
        config=config,
    )
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=metadata or {})
    artifact.add_dir(str(path))
    wandb.log_artifact(artifact)
    url = run.url or ""
    wandb.finish()
    return url


def _cmd_should_skip(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("mode")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    sys.exit(0 if should_skip_condition(args.out_dir, args.mode, retry_failed=args.retry_failed) else 1)


def _cmd_list_failed(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--collaboration", action="store_true")
    args = parser.parse_args(argv)
    for slug, out_dir in list_failed_conditions(Path(args.root), args.task, collaboration=args.collaboration):
        print(f"{slug}\t{out_dir}")
    return 0


def _cmd_download_wandb(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        default="neu-hai/collabsim/study_conditions_20260524_000957:v0",
        help="W&B artifact ref, e.g. entity/project/name:version",
    )
    parser.add_argument("--type", default="study_conditions", help="W&B artifact type")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to download into (default: W&B cache under ./artifacts/)",
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else None
    local_path = download_artifact_from_wandb(
        args.artifact,
        artifact_type=args.type,
        output_dir=output_dir,
    )
    print(f"wandb_download_path={local_path}")
    return 0


def _cmd_upload_wandb(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "collabsim"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--metadata-json", default="{}")
    args = parser.parse_args(argv)
    metadata = json.loads(args.metadata_json)
    url = upload_directory_to_wandb(
        Path(args.path),
        project=args.project,
        run_name=args.run_name,
        artifact_name=args.artifact_name,
        metadata={k: str(v) for k, v in metadata.items()},
    )
    print(f"wandb_upload_url={url}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: batch_helpers.py <should_skip|list_failed|upload_wandb|download_wandb> ...")
    command = argv[0]
    rest = argv[1:]
    if command == "should_skip":
        return _cmd_should_skip(rest)
    if command == "list_failed":
        return _cmd_list_failed(rest)
    if command == "upload_wandb":
        return _cmd_upload_wandb(rest)
    if command == "download_wandb":
        return _cmd_download_wandb(rest)
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
