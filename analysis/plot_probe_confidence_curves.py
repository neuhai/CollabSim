"""Probe confidence curves — eight figures (4 tasks × persona/collab).

Data: experiments/finalres_copy, all four models (Llama, Qwen, GPT, Claude).
Per model × condition: same aggregation as single-run (probe session → agents → mean).
Then mean across models for each x bucket.

Map task & hidden profile: each run rescaled so its actual end = 100% on a shared
0–100 axis (50 bins), then averaged across models. Day trader = round; shape factory = minute.

Usage:
  uv run python -m analysis.plot_probe_confidence_curves

Writes eight single-panel PNGs plus ``all_probe_confidence_curves.png`` (2×4 stitch of those files).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import yaml

from analysis.plot_persona_heatmaps import (
    MODEL_SPECS,
    condition_slug_to_label,
    condition_sort_key,
    discover_conditions,
    model_group_dir,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "study_conditions"
DEFAULT_DATA_ROOT = ROOT / "experiments" / "finalres_copy"
DEFAULT_OUT_DIR = ROOT / "analysis_output"
COMPOSITE_FILENAME = "all_probe_confidence_curves.png"
TASKS_ORDER = ["shapefactory", "maptask", "hidden_profile", "daytrader"]

PALETTE: list[str] = [
    "#8857a8",
    "#7c9fd9",
    "#ace1af",
    "#fee55d",
    "#fe8162",
]

COLOR_INDICES_BY_TASK: dict[str, list[int]] = {
    "shapefactory": [0, 1, 2, 3, 4],
    "maptask": [0, 1, 2],
    "hidden_profile": [0, 1],
    "daytrader": [0, 1, 2, 3],
}

CONSTRUCT_BY_TASK: dict[str, str] = {
    "shapefactory": "grounding",
    "maptask": "situation_awareness",
    "hidden_profile": "grounding",
    "daytrader": "grounding",
}

TASK_DEFAULT_MAX_STEPS: dict[str, int] = {
    "maptask": 120,
    "hidden_profile": 30,
}

PROGRESS_TASKS = frozenset({"maptask", "hidden_profile"})
PROGRESS_N_BINS = 50

AXIS_LABEL_SIZE = 33
TICK_LABEL_SIZE = 20
LEGEND_FONT_SIZE = 18
TITLE_FONT_SIZE = 22
LINE_WIDTH = 3.5
HANDLE_LENGTH = 4.8

TASK_DISPLAY: dict[str, str] = {
    "shapefactory": "Shape Factory",
    "maptask": "Map Task",
    "hidden_profile": "Hidden Profile",
    "daytrader": "Day Trader",
}

PANELS_BASE: list[dict[str, Any]] = [
    {
        "task": "shapefactory",
        "output_stem": "Shape_factory",
        "legend_loc": "lower right",
    },
    {
        "task": "maptask",
        "output_stem": "Map_task",
        "legend_loc": "bottom_center",
    },
    {
        "task": "hidden_profile",
        "output_stem": "Hidden_profile",
        "legend_loc": "lower right",
    },
    {
        "task": "daytrader",
        "output_stem": "Day_Trader",
        "legend_loc": "bottom_center",
        "legend_ncol": 2,
    },
]

STYLE_RC: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
}

DAYTRADER_TARGET_ROUNDS = 30
SHAPEFACTORY_MAX_MINUTES = 15
MODEL_SLUGS = [slug for _, slug in MODEL_SPECS]


def build_panels() -> list[dict[str, Any]]:
    panels: list[dict[str, Any]] = []
    for base in PANELS_BASE:
        for collab in (False, True):
            mode = "collab" if collab else "persona"
            panels.append(
                {
                    **base,
                    "collab": collab,
                    "output_name": f"{base['output_stem']}_{mode}",
                    "mode_label": "Collaboration" if collab else "Persona",
                }
            )
    return panels


def panel_for(task: str, collab: bool) -> dict[str, Any]:
    for panel in build_panels():
        if panel["task"] == task and panel["collab"] is collab:
            return panel
    raise KeyError(f"No panel for task={task!r} collab={collab}")


def xlabel_for_task(task: str) -> str:
    if task == "daytrader":
        return "Round"
    if task == "shapefactory":
        return "Time"
    if task in PROGRESS_TASKS:
        return "Completion"
    return "Step"


def legend_label(slug: str) -> str:
    label = condition_slug_to_label(slug)
    if label == "+BW↓":
        return "+BW ↓"
    return label


def iter_json_objects(path: Path) -> Iterator[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, end = dec.raw_decode(text, i)
        yield obj
        i = end


def parse_probe_timestamp(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def run_start_timestamp(records: list[dict[str, Any]]) -> datetime | None:
    times = [
        t
        for row in records
        if isinstance(row.get("timestamp"), str)
        for t in [parse_probe_timestamp(row["timestamp"])]
        if t is not None
    ]
    return min(times) if times else None


def probe_step_index(row: dict[str, Any]) -> int | None:
    probe_id = str(row.get("probe_id", ""))
    parts = probe_id.split("_")
    if len(parts) >= 2 and parts[0] == "probe" and parts[1].isdigit():
        step_index = int(parts[1])
        return step_index if step_index > 0 else None
    return None


def daytrader_step_to_round(step_index: int) -> int | None:
    game_round = ((step_index - 1) // 2) + 1
    if game_round < 1 or game_round > DAYTRADER_TARGET_ROUNDS:
        return None
    return game_round


def config_max_steps(task: str, condition: str) -> int:
    cfg_path = CONFIGS_DIR / task / f"{condition}.yml"
    if cfg_path.is_file():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        ms = (data.get("experiment") or {}).get("max_steps")
        if isinstance(ms, int) and ms > 0:
            return ms
    return TASK_DEFAULT_MAX_STEPS.get(task, 120)


def resolve_run_end_step(
    run_dir: Path,
    records: list[dict[str, Any]],
    task: str,
    condition: str,
) -> int:
    """Actual run end (100% completion): max of probe steps and steps_taken."""
    probe_steps = [
        s for row in records if (s := probe_step_index(row)) is not None
    ]
    probe_max = max(probe_steps) if probe_steps else 0
    taken = 0
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        ts = json.loads(summary_path.read_text(encoding="utf-8")).get("task_summary") or {}
        st = ts.get("steps_taken")
        if isinstance(st, int) and st > 0:
            taken = st
    end = max(probe_max, taken)
    if end <= 0:
        end = config_max_steps(task, condition)
    return end


def progress_bin(step_index: int, run_end_step: int) -> int | None:
    if run_end_step <= 0 or step_index <= 0:
        return None
    if step_index >= run_end_step:
        return PROGRESS_N_BINS - 1
    pct = 100.0 * step_index / run_end_step
    idx = int(pct * PROGRESS_N_BINS / 100.0)
    return min(PROGRESS_N_BINS - 1, max(0, idx))


def progress_bin_center(bin_idx: int) -> float:
    return (bin_idx + 0.5) * 100.0 / PROGRESS_N_BINS


def native_bucket(
    row: dict[str, Any],
    task: str,
    *,
    run_t0: datetime | None = None,
) -> int | None:
    if task == "shapefactory":
        if run_t0 is None:
            return None
        ts_raw = row.get("timestamp")
        if not isinstance(ts_raw, str):
            return None
        ts = parse_probe_timestamp(ts_raw)
        if ts is None:
            return None
        minute = int((ts - run_t0).total_seconds() // 60)
        if minute < 0 or minute >= SHAPEFACTORY_MAX_MINUTES:
            return None
        return minute

    step_index = probe_step_index(row)
    if step_index is None:
        return None
    if task == "daytrader":
        return daytrader_step_to_round(step_index)
    return step_index


def load_probe_records(path: Path) -> list[dict[str, Any]]:
    return list(iter_json_objects(path))


def _session_means_by_bucket(
    records: list[dict[str, Any]],
    construct: str,
    task: str,
    *,
    run_end_step: int | None = None,
) -> dict[int, float]:
    run_t0 = run_start_timestamp(records) if task == "shapefactory" else None
    dedup: dict[tuple[int, str, str], list[float]] = defaultdict(list)

    for row in records:
        if row.get("construct") != construct:
            continue
        conf = row.get("confidence")
        if not isinstance(conf, (int, float)):
            continue
        probe_id = str(row.get("probe_id", ""))
        actor = row.get("actor_id")
        if actor is None:
            continue

        if task in PROGRESS_TASKS:
            step_index = probe_step_index(row)
            if step_index is None or run_end_step is None:
                continue
            bucket = progress_bin(step_index, run_end_step)
        else:
            bucket = native_bucket(row, task, run_t0=run_t0)

        if bucket is None:
            continue
        dedup[(bucket, probe_id, str(actor))].append(float(conf))

    sessions: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (bucket, _probe_id, actor), confs in dedup.items():
        sessions[bucket][actor].append(statistics.mean(confs))

    out: dict[int, float] = {}
    for bucket, actor_map in sessions.items():
        actor_means = [
            statistics.mean(confs) for confs in actor_map.values() if confs
        ]
        if actor_means:
            out[bucket] = statistics.mean(actor_means)
    return out


def rescaled_progress_grid(
    records: list[dict[str, Any]],
    construct: str,
    task: str,
    run_end_step: int,
) -> dict[int, float]:
    """One run → fixed 0..100% bins; last bin always includes the true run end."""
    return _session_means_by_bucket(
        records, construct, task, run_end_step=run_end_step
    )


def bucket_series_to_xy(
    bucket_means: dict[int, float], task: str
) -> tuple[list[float], list[float]]:
    if not bucket_means:
        return [], []

    use_native_x = task in {"daytrader", "shapefactory"}
    xs: list[float] = []
    ys: list[float] = []
    for i, bucket in enumerate(sorted(bucket_means)):
        xs.append(float(bucket) if use_native_x else float(i))
        ys.append(bucket_means[bucket])
    return xs, ys


def aggregate_native_run(
    records: list[dict[str, Any]], construct: str, task: str
) -> tuple[list[float], list[float]]:
    buckets = _session_means_by_bucket(records, construct, task)
    return bucket_series_to_xy(buckets, task)


def mean_progress_grids(grids: list[dict[int, float]]) -> tuple[list[float], list[float]]:
    """Average across models after each run was rescaled to 0–100%."""
    if not grids:
        return [], []
    accum: dict[int, list[float]] = defaultdict(list)
    for grid in grids:
        for bin_idx, y in grid.items():
            accum[bin_idx].append(y)
    if not accum:
        return [], []
    bins = sorted(accum)
    return [progress_bin_center(b) for b in bins], [statistics.mean(accum[b]) for b in bins]


def mean_series_across_models(
    model_series: list[tuple[list[float], list[float]]],
) -> tuple[list[float], list[float]]:
    accum_xy: dict[float, list[float]] = defaultdict(list)
    for xs, ys in model_series:
        for x, y in zip(xs, ys, strict=True):
            accum_xy[float(x)].append(y)
    xs_out = sorted(accum_xy)
    return xs_out, [statistics.mean(accum_xy[x]) for x in xs_out]


def available_conditions(task: str, data_root: Path, *, collab: bool) -> list[str]:
    slugs = discover_conditions(task)
    found: list[str] = []
    for slug in slugs:
        for model_slug in MODEL_SLUGS:
            run_dir = model_group_dir(data_root, model_slug, collab=collab) / task / slug
            if (run_dir / "probes.jsonl").is_file():
                found.append(slug)
                break
    return sorted(set(found), key=condition_sort_key)


def apply_legend(ax: plt.Axes, panel: dict[str, Any]) -> None:
    legend_loc = panel.get("legend_loc", "lower right")
    legend_kwargs: dict[str, Any] = {
        "frameon": True,
        "fancybox": False,
        "shadow": False,
        "fontsize": LEGEND_FONT_SIZE,
        "borderpad": 0.35,
        "handlelength": HANDLE_LENGTH,
        "handletextpad": 0.8,
        "borderaxespad": 0.55,
    }
    if legend_loc == "bottom_center":
        ncol = panel.get("legend_ncol", 1)
        legend_kwargs.update(
            {
                "loc": "lower center",
                "bbox_to_anchor": (0.5, 0.08),
                "ncol": ncol if isinstance(ncol, int) and ncol > 0 else 1,
            }
        )
    elif legend_loc == "upper left":
        legend_kwargs.update({"loc": "upper left", "bbox_to_anchor": (0.02, 0.98)})
    else:
        legend_kwargs.update({"loc": "lower right", "bbox_to_anchor": (0.98, 0.02)})
    legend_ncol = panel.get("legend_ncol")
    if isinstance(legend_ncol, int) and legend_ncol > 0 and legend_loc != "bottom_center":
        legend_kwargs["ncol"] = legend_ncol

    legend = ax.legend(**legend_kwargs)
    frame = legend.get_frame()
    frame.set_linewidth(0.8)
    frame.set_edgecolor("#333333")
    frame.set_facecolor("white")
    frame.set_alpha(1.0)


def style_axes(ax: plt.Axes, task: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        color="#d0d0d0",
        alpha=0.9,
    )
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        length=4,
        width=0.8,
    )
    ymin, ymax = ax.get_ylim()
    pad = max(0.02, (ymax - ymin) * 0.06)
    ax.set_ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))

    if task in PROGRESS_TASKS:
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])


def plot_panel(
    ax: plt.Axes,
    task: str,
    data_root: Path,
    panel: dict[str, Any],
) -> None:
    construct = CONSTRUCT_BY_TASK[task]
    collab = panel["collab"]
    conditions = available_conditions(task, data_root, collab=collab)
    if not conditions:
        ax.set_visible(False)
        return

    color_indices = COLOR_INDICES_BY_TASK.get(task, list(range(len(conditions))))
    for i, slug in enumerate(conditions):
        per_model_native: list[tuple[list[float], list[float]]] = []
        per_model_progress: list[dict[int, float]] = []
        for model_slug in MODEL_SLUGS:
            run_dir = model_group_dir(data_root, model_slug, collab=collab) / task / slug
            probes_path = run_dir / "probes.jsonl"
            if not probes_path.is_file():
                continue
            records = load_probe_records(probes_path)
            if task in PROGRESS_TASKS:
                end_step = resolve_run_end_step(run_dir, records, task, slug)
                grid = rescaled_progress_grid(records, construct, task, end_step)
                if grid:
                    per_model_progress.append(grid)
            else:
                series = aggregate_native_run(records, construct, task)
                if series[0]:
                    per_model_native.append(series)

        if task in PROGRESS_TASKS:
            xs, ys = mean_progress_grids(per_model_progress)
        else:
            xs, ys = mean_series_across_models(per_model_native)
        if not xs:
            continue

        ci = color_indices[i] if i < len(color_indices) else i % len(PALETTE)
        ax.plot(
            xs,
            ys,
            color=PALETTE[ci],
            linewidth=LINE_WIDTH,
            solid_capstyle="round",
            label=legend_label(slug),
        )

    if task == "daytrader":
        ax.set_xlim(0.5, DAYTRADER_TARGET_ROUNDS + 0.5)
    elif task == "shapefactory":
        ax.set_xlim(0, SHAPEFACTORY_MAX_MINUTES)

    ax.set_xlabel(xlabel_for_task(task), fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax.set_ylabel("Probe confidence", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    style_axes(ax, task)

    title = f"{TASK_DISPLAY[task]} ({panel['mode_label']})"
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=12)

    apply_legend(ax, panel)


def build_single_figure(data_root: Path, panel: dict[str, Any]) -> plt.Figure:
    plt.rcParams.update(STYLE_RC)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), facecolor="white")
    bottom = 0.22 if panel.get("legend_loc") == "bottom_center" else 0.14
    fig.subplots_adjust(top=0.90, bottom=bottom, left=0.14, right=0.97)
    plot_panel(ax, panel["task"], data_root, panel)
    return fig


def _pad_tile(img: np.ndarray, height: int, width: int) -> np.ndarray:
    if img.ndim == 2:
        canvas = np.ones((height, width), dtype=img.dtype)
    else:
        canvas = np.ones((height, width, img.shape[2]), dtype=img.dtype)
        if img.dtype in (np.float32, np.float64):
            canvas[..., :3] = 1.0
            if img.shape[2] == 4:
                canvas[..., 3] = 1.0
    canvas[: img.shape[0], : img.shape[1]] = img
    return canvas


def stitch_composite_from_pngs(out_dir: Path, out_path: Path) -> None:
    """2×4 grid of the eight saved PNGs (pixel-identical to singles)."""
    tiles: list[list[np.ndarray]] = []
    for collab in (False, True):
        row: list[np.ndarray] = []
        for task in TASKS_ORDER:
            panel = panel_for(task, collab)
            row.append(mpimg.imread(out_dir / f"{panel['output_name']}.png"))
        tiles.append(row)

    cell_h = max(tile.shape[0] for row in tiles for tile in row)
    cell_w = max(tile.shape[1] for row in tiles for tile in row)
    row_images = [
        np.concatenate([_pad_tile(tile, cell_h, cell_w) for tile in row], axis=1)
        for row in tiles
    ]
    composite = np.concatenate(row_images, axis=0)
    mpimg.imsave(out_path, composite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = build_panels()
    for panel in panels:
        task = panel["task"]
        mode = "collab" if panel["collab"] else "persona"
        conds = available_conditions(task, data_root, collab=panel["collab"])
        labels = [legend_label(c) for c in conds]
        print(f"{task}/{mode}: {', '.join(labels)}")

        out_path = out_dir / f"{panel['output_name']}.png"
        fig = build_single_figure(data_root, panel)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Wrote {out_path}")

    composite_path = out_dir / COMPOSITE_FILENAME
    stitch_composite_from_pngs(out_dir, composite_path)
    print(f"Wrote {composite_path}")


if __name__ == "__main__":
    main()
