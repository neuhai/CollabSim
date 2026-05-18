"""MapTask drawing accuracy from a fixed score_board grid (see configs/map_task_material/score_board.txt)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
