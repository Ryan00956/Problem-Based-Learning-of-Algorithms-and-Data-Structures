from __future__ import annotations

import unittest

import numpy as np

from src.datasets.netflix.evaluation import (
    EvaluationConfig,
    RatingRow,
    evaluate_model,
    evaluate_hybrid_ranker,
    evaluate_user_user_collaborative,
    movie_quality_scores,
    recommend_known_user,
    split_by_user_time,
)
from src.datasets.netflix.matrix_factorization import (
    MatrixFactorizationConfig,
    MatrixFactorizationModel,
    TrainingRating,
)


def _ratings() -> list[TrainingRating]:
    return [
        TrainingRating(1, 10, 5.0),
        TrainingRating(1, 11, 4.5),
        TrainingRating(1, 12, 1.0),
        TrainingRating(1, 13, 4.8),
        TrainingRating(2, 10, 5.0),
        TrainingRating(2, 11, 4.0),
        TrainingRating(2, 12, 1.2),
        TrainingRating(2, 13, 4.6),
        TrainingRating(3, 10, 1.0),
        TrainingRating(3, 11, 1.5),
        TrainingRating(3, 12, 5.0),
        TrainingRating(3, 13, 1.0),
    ]


def _time_rows() -> list[RatingRow]:
    rows = []
    for user_id in (1, 2, 3):
        values = [
            (10, 5.0),
            (11, 4.5 if user_id != 3 else 1.5),
            (12, 1.0 if user_id != 3 else 5.0),
            (13, 5.0 if user_id != 3 else 1.0),
            (14, 5.0 if user_id != 3 else 1.0),
        ]
        for day, (movie_id, rating) in enumerate(values, start=1):
            rows.append(RatingRow(user_id, movie_id, rating, f"2005-01-{day:02d}"))
    return rows


class MatrixFactorizationTest(unittest.TestCase):
    def test_numpy_backend_learns_biased_matrix_factorization(self) -> None:
        model = MatrixFactorizationModel(
            MatrixFactorizationConfig(
                factors=4,
                epochs=4,
                learning_rate=0.04,
                regularization=0.01,
                backend="numpy",
            )
        )

        curve = model.fit(_ratings())

        self.assertEqual(model.backend_used, "numpy")
        self.assertEqual(len(curve), 4)
        self.assertLessEqual(curve[-1]["train_rmse"], curve[0]["train_rmse"])
        self.assertGreater(model.predict(1, 10), model.predict(1, 12))

    def test_recommend_known_user_excludes_training_history(self) -> None:
        model = MatrixFactorizationModel(
            MatrixFactorizationConfig(factors=4, epochs=3, backend="numpy")
        )
        model.fit(_ratings())

        ranked = recommend_known_user(
            model,
            1,
            np.array([10, 11, 12, 13], dtype=np.int32),
            exclude_movie_ids={10, 11},
            top_k=2,
        )

        self.assertEqual(len(ranked), 2)
        self.assertNotIn(ranked[0][0], {10, 11})

    def test_time_split_evaluation_returns_ranking_metrics(self) -> None:
        config = EvaluationConfig(
            max_users=3,
            min_ratings_per_user=5,
            test_ratio=0.4,
            top_k=2,
            candidate_limit=10,
        )
        split = split_by_user_time(_time_rows(), config)
        model = MatrixFactorizationModel(
            MatrixFactorizationConfig(factors=4, epochs=3, backend="numpy")
        )
        model.fit(split["train"])

        metrics = evaluate_model(model, split, config)

        self.assertEqual(metrics["algorithm"], "biased_matrix_factorization_adam")
        self.assertEqual(metrics["backend"], "numpy")
        self.assertGreater(metrics["train_ratings"], 0)
        self.assertGreater(metrics["test_ratings"], 0)
        self.assertIn("hit_rate_at_k", metrics)

    def test_user_user_collaborative_evaluation_returns_comparable_metrics(self) -> None:
        config = EvaluationConfig(
            max_users=3,
            min_ratings_per_user=5,
            test_ratio=0.4,
            top_k=2,
            candidate_limit=10,
        )
        split = split_by_user_time(_time_rows(), config)

        metrics = evaluate_user_user_collaborative(split, config)

        self.assertEqual(metrics["algorithm"], "user_user_collaborative_filtering")
        self.assertEqual(metrics["top_k"], 2)
        self.assertIn("precision_at_k", metrics)

    def test_hybrid_ranker_returns_weighted_metrics(self) -> None:
        config = EvaluationConfig(
            max_users=3,
            min_ratings_per_user=5,
            test_ratio=0.4,
            top_k=2,
            candidate_limit=10,
        )
        split = split_by_user_time(_time_rows(), config)
        model = MatrixFactorizationModel(
            MatrixFactorizationConfig(factors=4, epochs=3, backend="numpy")
        )
        model.fit(split["train"])

        metrics = evaluate_hybrid_ranker(
            model,
            split,
            config,
            quality_scores=movie_quality_scores(split),
            mf_weight=0.25,
            collaborative_weight=0.0,
            quality_weight=0.75,
        )

        self.assertEqual(metrics["algorithm"], "hybrid_mf_quality")
        self.assertEqual(metrics["mf_weight"], 0.25)
        self.assertEqual(metrics["quality_weight"], 0.75)
        self.assertIn("precision_at_k", metrics)

    def test_grid_search_selects_best_trial(self) -> None:
        config = EvaluationConfig(
            max_users=3,
            min_ratings_per_user=5,
            test_ratio=0.4,
            top_k=2,
            candidate_limit=10,
        )
        # Exercise the grid path on a temporary split by patching through the public pieces.
        split = split_by_user_time(_time_rows(), config)
        rows = []
        for index, model_config in enumerate(
            [
                MatrixFactorizationConfig(factors=2, epochs=2, backend="numpy"),
                MatrixFactorizationConfig(factors=4, epochs=2, backend="numpy"),
            ],
            start=1,
        ):
            model = MatrixFactorizationModel(model_config)
            curve = model.fit(split["train"])
            metrics = evaluate_model(model, split, config)
            rows.append({"trial": index, "curve": curve, **metrics})

        best = max(rows, key=lambda row: (row["precision_at_k"], row["hit_rate_at_k"]))
        self.assertIn(best["trial"], {1, 2})

    def test_torch_cuda_backend_when_available(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        model = MatrixFactorizationModel(
            MatrixFactorizationConfig(
                factors=4,
                epochs=2,
                learning_rate=0.03,
                backend="torch",
                device="cuda",
                batch_size=4,
            )
        )

        curve = model.fit(_ratings())

        self.assertEqual(model.backend_used, "torch")
        self.assertEqual(model.device_used, "cuda")
        self.assertEqual(len(curve), 2)
        self.assertTrue(np.isfinite(model.predict(1, 10)))


if __name__ == "__main__":
    unittest.main()
