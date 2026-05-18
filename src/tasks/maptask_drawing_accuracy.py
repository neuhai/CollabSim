"""MapTask drawing accuracy from a fixed score_board grid (see configs/map_task_material/score_board.txt)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_SCORE_BOARD_PATH = Path("configs/map_task_material/score_board.txt")


def parse_score_board_text(raw: str) -> list[list[int]]:
    """Parse ASCII score board: '.' -> 0, '1'/'2'/'3' -> integers; skip # comment lines."""

    rows: list[list[int]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        row_vals: list[int] = []
        for ch in line.rstrip("\n"):
            if ch == ".":
                row_vals.append(0)
            elif ch.isdigit():
                row_vals.append(int(ch))
            else:
                row_vals.append(0)
        if row_vals:
            rows.append(row_vals)
    return rows


def load_score_board_file(path: Path) -> list[list[int]]:
    text = path.read_text(encoding="utf-8")
    return parse_score_board_text(text)


def _cell_score(board: list[list[int]], row: int, col: int) -> int:
    if row < 0 or col < 0 or row >= len(board):
        return 0
    line = board[row]
    if col >= len(line):
        return 0
    return int(line[col])


def _normalize_drawn(raw: Any) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            r, c = item
            if isinstance(r, int) and isinstance(c, int):
                out.append((r, c))
    return out


def _reference_route(task_state: dict[str, Any]) -> list[tuple[int, int]]:
    ref = task_state.get("reference_route_cells")
    if not isinstance(ref, list):
        return []
    pts: list[tuple[int, int]] = []
    for item in ref:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            r, c = item
            if isinstance(r, int) and isinstance(c, int):
                pts.append((r, c))
    return pts


def compute_drawing_accuracy_snapshot(task_state: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregate score_board weights over the follower's current drawn cells."""

    board = task_state.get("drawing_score_board")
    if not isinstance(board, list) or not board:
        return None

    participants = task_state.get("participants")
    if not isinstance(participants, dict):
        return None

    follower: dict[str, Any] | None = None
    for pdata in participants.values():
        if isinstance(pdata, dict) and pdata.get("role") == "follower":
            follower = pdata
            break
    if follower is None:
        return None

    drawn = _normalize_drawn(follower.get("drawn_route_points"))
    route_ref = _reference_route(task_state)

    score_sum = 0
    route_cells_hit = 0
    for row, col in drawn:
        v = _cell_score(board, row, col)
        score_sum += v
        if v >= 3:
            route_cells_hit += 1

    max_route_sum = sum(_cell_score(board, r, c) for r, c in route_ref)
    ratio = float(score_sum) / float(max_route_sum) if max_route_sum > 0 else None

    return {
        "score_board_sum_drawn_cells": score_sum,
        "max_route_score_board_sum": max_route_sum,
        "ratio_vs_ground_truth_route": ratio,
        "drawn_cell_count": len(drawn),
        "route_cells_hit_count": route_cells_hit,
    }


def render_score_board_text(board: list[list[int]]) -> str:
    """Render score board matrix as ASCII (digits for scored cells, space for 0)."""

    lines: list[str] = []
    for row in board:
        chars: list[str] = []
        for val in row:
            v = int(val)
            chars.append(str(v) if v > 0 else " ")
        lines.append("".join(chars))
    return "\n".join(lines)


def snapshot_to_route_metrics(snap: dict[str, Any]) -> dict[str, float]:
    """Map a drawing_accuracy snapshot to canonical route_score fields."""

    score = float(snap.get("score_board_sum_drawn_cells", 0))
    score_max = float(snap.get("max_route_score_board_sum", 0))
    ratio_raw = snap.get("ratio_vs_ground_truth_route")
    ratio = float(ratio_raw) if isinstance(ratio_raw, (int, float)) else (
        (score / score_max) if score_max > 0 else 0.0
    )
    out: dict[str, float] = {
        "route_score": score,
        "route_score_max": score_max,
        "route_similarity": ratio,
        "follower_accuracy": ratio,
        "drawn_cell_count": float(snap.get("drawn_cell_count", 0)),
        "route_cells_hit_count": float(snap.get("route_cells_hit_count", 0)),
    }
    return out


def compute_maptask_route_outcome(task_state: dict[str, Any]) -> dict[str, Any] | None:
    """Score-board-based run outcome from current task_state (same source as step snapshots)."""

    snap = compute_drawing_accuracy_snapshot(task_state)
    if snap is None:
        return None
    metrics = snapshot_to_route_metrics(snap)
    board = task_state.get("drawing_score_board")
    score_map_text: str | None = None
    if isinstance(board, list) and board:
        score_map_text = render_score_board_text(board)

    follower_map_text: str | None = None
    participants = task_state.get("participants")
    if isinstance(participants, dict):
        for pdata in participants.values():
            if isinstance(pdata, dict) and pdata.get("role") == "follower":
                wt = pdata.get("map_working_text")
                if isinstance(wt, str) and wt:
                    follower_map_text = wt
                break

    return {
        "maptask_route_score": metrics["route_score"],
        "maptask_route_score_max": metrics["route_score_max"],
        "maptask_route_similarity": metrics["route_similarity"],
        "maptask_drawn_cell_count": metrics["drawn_cell_count"],
        "maptask_route_cells_hit_count": metrics["route_cells_hit_count"],
        "maptask_score_map_text": score_map_text,
        "maptask_follower_map_text": follower_map_text,
        "drawing_accuracy": snap,
    }
