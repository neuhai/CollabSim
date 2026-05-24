#!/usr/bin/env python3
"""Download a W&B study_conditions artifact."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "configs" / "study_conditions"))

from batch_helpers import download_artifact_from_wandb  # noqa: E402


DEFAULT_ARTIFACT = "neu-hai/collabsim/study_conditions_20260524_000957:v0"
DEFAULT_TYPE = "study_conditions"
DEFAULT_OUTPUT = ROOT / "artifacts" / "study_conditions_20260524_000957"
DEFAULT_CACHE = ROOT / "artifacts" / ".wandb_cache"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a W&B artifact.")
    parser.add_argument(
        "--artifact",
        default=DEFAULT_ARTIFACT,
        help=f"Artifact ref (default: {DEFAULT_ARTIFACT})",
    )
    parser.add_argument(
        "--type",
        default=DEFAULT_TYPE,
        help=f"Artifact type (default: {DEFAULT_TYPE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Download destination (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"W&B artifact cache directory (default: {DEFAULT_CACHE})",
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_CACHE_DIR", str(args.cache_dir))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_path = download_artifact_from_wandb(
        args.artifact,
        artifact_type=args.type,
        output_dir=args.output_dir,
    )
    print(f"Downloaded to: {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
