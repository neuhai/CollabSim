"""CLI runner: analyse one or more experiment run directories and write CSVs.

Usage
-----
# Single run directory
python -m analysis.report --run experiments/hidden_profile_azure/run_001

# All runs under an experiment folder
python -m analysis.report --runs experiments/hidden_profile_azure

# Multiple roots
python -m analysis.report --runs experiments/hidden_profile_azure experiments/daytrader_v1

Output (written to --out-dir, default: analysis_output/):
  task_metrics.csv   — per-run and per-agent task outcome metrics
  probe_metrics.csv  — per-run and per-agent probe construct metrics
  combined.csv       — task + probe metrics joined on (run_id, agent_id)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from analysis.trace_parser import Trace, find_run_dirs, load_trace
from analysis.task_metrics import task_summary_rows
from analysis.probe_analysis import probe_summary_rows
from analysis.summary_stats import summary_stat_rows


# ------------------------------------------------------------------ #
# CSV helpers
# ------------------------------------------------------------------ #

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Flatten any nested dicts to JSON strings so CSV stays flat
            flat = {}
            for k, v in row.items():
                if isinstance(v, dict):
                    flat[k] = json.dumps(v, ensure_ascii=False)
                elif v is None:
                    flat[k] = ""
                else:
                    flat[k] = v
            writer.writerow(flat)
    print(f"  wrote {len(rows)} rows → {path}")


def _join_rows(
    task_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Left-join probe metrics into task rows on (run_id, agent_id)."""
    probe_index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in probe_rows:
        key = (str(row.get("run_id", "")), str(row.get("agent_id", "")))
        probe_index.setdefault(key, {}).update(row)

    combined: list[dict[str, Any]] = []
    for row in task_rows:
        key = (str(row.get("run_id", "")), str(row.get("agent_id", "")))
        merged = dict(row)
        probe_extra = probe_index.get(key, {})
        for k, v in probe_extra.items():
            if k not in merged:
                merged[k] = v
        combined.append(merged)
    return combined


# ------------------------------------------------------------------ #
# Single-trace analysis
# ------------------------------------------------------------------ #

def analyse_trace(trace: Trace) -> tuple[list[dict], list[dict], list[dict]]:
    task_rows = task_summary_rows(trace)
    probe_rows = probe_summary_rows(trace)
    stat_rows = summary_stat_rows(trace)
    return task_rows, probe_rows, stat_rows


# ------------------------------------------------------------------ #
# Multi-trace runner
# ------------------------------------------------------------------ #

def run_analysis(run_dirs: list[Path], out_dir: Path) -> None:
    all_task_rows: list[dict[str, Any]] = []
    all_probe_rows: list[dict[str, Any]] = []
    all_stat_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        print(f"  loading {run_dir} …")
        try:
            trace = load_trace(run_dir)
        except Exception as exc:
            print(f"  ⚠ skipped {run_dir}: {exc}")
            continue
        task_rows, probe_rows, stat_rows = analyse_trace(trace)
        all_task_rows.extend(task_rows)
        all_probe_rows.extend(probe_rows)
        all_stat_rows.extend(stat_rows)

    print(f"\n{len(run_dirs)} run(s) processed.")
    _write_csv(out_dir / "task_metrics.csv", all_task_rows)
    _write_csv(out_dir / "probe_metrics.csv", all_probe_rows)
    _write_csv(out_dir / "combined.csv", _join_rows(all_task_rows, all_probe_rows))
    _write_csv(out_dir / "run_summary_stats.csv", all_stat_rows)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _collect_run_dirs(paths: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for p in paths:
        root = Path(p)
        if (root / "run_manifest.json").exists():
            dirs.append(root)
        else:
            found = find_run_dirs(root)
            if found:
                dirs.extend(found)
            elif root.is_dir():
                # treat directory itself as a run dir (may lack manifest)
                dirs.append(root)
    return dirs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyse experiment run directories and write metrics CSVs.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run",
        metavar="DIR",
        help="Single run directory to analyse.",
    )
    group.add_argument(
        "--runs",
        metavar="DIR",
        nargs="+",
        help="One or more directories; each may be a run dir or an experiment root "
             "containing multiple run subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR",
        default="analysis_output",
        help="Directory to write output CSVs (default: analysis_output/).",
    )

    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    if args.run:
        paths = [args.run]
    else:
        paths = args.runs

    run_dirs = _collect_run_dirs(paths)
    if not run_dirs:
        print("No run directories found.", file=sys.stderr)
        sys.exit(1)

    print(f"Analysing {len(run_dirs)} run dir(s) → {out_dir}/")
    run_analysis(run_dirs, out_dir)


if __name__ == "__main__":
    main()
