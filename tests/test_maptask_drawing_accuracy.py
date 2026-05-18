"""Tests for MapTask score_board parsing and drawing accuracy snapshots."""

from __future__ import annotations

import unittest

from src.tasks.maptask_drawing_accuracy import (
    compute_drawing_accuracy_snapshot,
    compute_maptask_route_outcome,
    parse_score_board_text,
    snapshot_to_route_metrics,
)


class MapTaskDrawingAccuracyTests(unittest.TestCase):
    def test_parse_skips_comments_and_dots_digits(self) -> None:
        raw = "# score_board header\n.123\n"
        grid = parse_score_board_text(raw)
        self.assertEqual(grid, [[0, 1, 2, 3]])

    def test_snapshot_sums_drawn_cells(self) -> None:
        board = [
            [0, 3, 0],
            [2, 0, 1],
        ]
        task_state = {
            "drawing_score_board": board,
            "reference_route_cells": [[0, 1]],
            "participants": {
                "B": {
                    "role": "follower",
                    "drawn_route_points": [[0, 1], [1, 0]],
                },
            },
        }
        snap = compute_drawing_accuracy_snapshot(task_state)
        assert snap is not None
        self.assertEqual(snap["score_board_sum_drawn_cells"], 5)
        self.assertEqual(snap["max_route_score_board_sum"], 3)
        self.assertAlmostEqual(snap["ratio_vs_ground_truth_route"], 5.0 / 3.0)
        self.assertEqual(snap["route_cells_hit_count"], 1)

    def test_route_outcome_uses_score_board(self) -> None:
        board = [[0, 3, 0], [2, 0, 1]]
        task_state = {
            "drawing_score_board": board,
            "reference_route_cells": [[0, 1]],
            "participants": {
                "B": {
                    "role": "follower",
                    "drawn_route_points": [[0, 1]],
                    "map_working_text": ".*.\n...",
                },
            },
        }
        outcome = compute_maptask_route_outcome(task_state)
        assert outcome is not None
        self.assertEqual(outcome["maptask_route_score"], 3.0)
        self.assertEqual(outcome["maptask_route_score_max"], 3.0)
        self.assertAlmostEqual(outcome["maptask_route_similarity"], 1.0)
        metrics = snapshot_to_route_metrics(outcome["drawing_accuracy"])
        self.assertEqual(metrics["route_score"], 3.0)


if __name__ == "__main__":
    unittest.main()
