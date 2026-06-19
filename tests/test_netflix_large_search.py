from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.datasets.netflix.large_search import (
    LargeSearchConfig,
    default_large_hybrid_weight_grid,
    run_large_search,
)
from src.datasets.netflix.matrix_factorization import MatrixFactorizationConfig


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class NetflixLargeSearchTest(unittest.TestCase):
    def test_weight_grid_is_unique_and_sums_to_one(self) -> None:
        grid = default_large_hybrid_weight_grid()

        self.assertEqual(len(grid), len(set(grid)))
        self.assertTrue(grid)
        for weights in grid:
            self.assertAlmostEqual(sum(weights), 1.0, places=4)

    def test_large_search_writes_resumable_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = LargeSearchConfig(
                scale_user_counts=(7,),
                mf_user_count=7,
                hybrid_user_count=7,
                candidate_limit=20,
            )

            with (
                patch("src.datasets.netflix.large_search.compare_recommenders") as compare,
                patch("src.datasets.netflix.large_search.grid_search_matrix_factorization") as mf,
                patch("src.datasets.netflix.large_search.grid_search_hybrid_weights") as hybrid,
            ):
                compare.return_value = {
                    "metrics": [
                        {
                            "algorithm": "hybrid_mf_quality",
                            "precision_at_k": 0.5,
                            "hit_rate_at_k": 0.8,
                            "map_at_k": 0.4,
                        }
                    ],
                    "training_curve": [],
                }
                mf.return_value = {
                    "results": [
                        {
                            "trial": 1,
                            "algorithm": "biased_matrix_factorization_adam",
                            "precision_at_k": 0.4,
                            "hit_rate_at_k": 0.7,
                            "map_at_k": 0.3,
                        }
                    ],
                    "training_curves": [{"epoch": 1, "train_rmse": 1.0}],
                    "best": {},
                }
                hybrid.return_value = {
                    "results": [
                        {
                            "trial": 1,
                            "algorithm": "hybrid_mf_quality",
                            "precision_at_k": 0.6,
                            "hit_rate_at_k": 0.9,
                            "map_at_k": 0.5,
                        }
                    ],
                    "training_curve": [{"epoch": 1, "train_rmse": 1.0}],
                    "best": {},
                }

                paths = run_large_search(
                    output_dir,
                    config=config,
                    mf_configs=[MatrixFactorizationConfig(factors=2, epochs=1, backend="numpy")],
                    hybrid_weight_grid=[(0.25, 0.0, 0.75)],
                )
                paths = run_large_search(
                    output_dir,
                    config=config,
                    mf_configs=[MatrixFactorizationConfig(factors=2, epochs=1, backend="numpy")],
                    hybrid_weight_grid=[(0.25, 0.0, 0.75)],
                )

            self.assertTrue(paths["scale_comparison"].exists())
            self.assertTrue(paths["best_summary"].exists())
            self.assertEqual(len(_read_rows(paths["scale_comparison"])), 1)
            self.assertEqual(len(_read_rows(paths["best_summary"])), 3)


if __name__ == "__main__":
    unittest.main()
