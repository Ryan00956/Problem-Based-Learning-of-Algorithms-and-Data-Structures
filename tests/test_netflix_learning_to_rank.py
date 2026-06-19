from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.datasets.netflix.import_duckdb import DEFAULT_DB_PATH
from src.datasets.netflix.learning_to_rank import (
    FEATURE_NAMES,
    LearningToRankConfig,
    LinearRanker,
    ResidualLinearRanker,
    _blend_scores,
    _feature_stats,
    _popularity_fallback_scores,
    _rank_percentiles,
    _residual_scores_from_features,
    features_for_user_candidates,
    run_learning_to_rank,
    split_by_user_three_way,
    standard_learning_to_rank_config,
)
from src.datasets.netflix.evaluation import RatingRow, _build_collaborative_indexes
from src.datasets.netflix.matrix_factorization import MatrixFactorizationConfig, MatrixFactorizationModel
from tests.test_netflix_matrix_factorization import _ratings


class NetflixLearningToRankTest(unittest.TestCase):
    def test_three_way_split_keeps_ranker_training_between_base_and_test(self) -> None:
        rows = []
        start = date(2020, 1, 1)
        for user_id in (1, 2):
            for index in range(12):
                rows.append(
                    RatingRow(
                        user_id=user_id,
                        movie_id=user_id * 100 + index,
                        rating=5.0 if index % 2 == 0 else 3.0,
                        rating_date=start + timedelta(days=index),
                    )
                )

        split = split_by_user_three_way(rows, LearningToRankConfig(max_users=2))

        self.assertEqual(len(split["train"]), 16)
        self.assertEqual(len(split["ranker_train"]), 4)
        self.assertEqual(len(split["test"]), 4)
        self.assertTrue(split["ranker_relevant_by_user"])
        self.assertTrue(split["relevant_by_user"])

    def test_linear_ranker_learns_positive_feature_direction(self) -> None:
        base_features = np.array(
            [
                [0.9, 0.1, 0.8, 0.8, 0.7, 0.8, 0.5],
                [0.8, 0.2, 0.7, 0.7, 0.6, 0.8, 0.5],
                [0.1, 0.0, 0.2, 0.3, 0.2, 0.8, 0.5],
                [0.2, 0.0, 0.1, 0.2, 0.1, 0.8, 0.5],
            ],
            dtype=np.float32,
        )
        features = np.zeros((len(base_features), len(FEATURE_NAMES)), dtype=np.float32)
        features[:, : base_features.shape[1]] = base_features
        labels = np.array([1, 1, 0, 0], dtype=np.float32)

        ranker = LinearRanker(learning_rate=0.1, l2=0.0, epochs=60, seed=7)
        ranker.fit(features, labels)
        scores = ranker.score(features)

        self.assertGreater(scores[:2].mean(), scores[2:].mean())
        self.assertEqual(len(ranker.feature_weights()), len(FEATURE_NAMES))

    def test_residual_ranker_learns_correction_feature_direction(self) -> None:
        features = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32)
        profile_index = FEATURE_NAMES.index("profile_similarity")
        quality_index = FEATURE_NAMES.index("quality_score")
        features[:, quality_index] = 0.5
        features[[0, 2], profile_index] = 1.0
        labels = np.array([1, 0, 1, 0], dtype=np.float32)
        user_ids = np.array([1, 1, 2, 2], dtype=np.int64)

        ranker = ResidualLinearRanker(learning_rate=0.1, l2=0.0, epochs=80, seed=7)
        ranker.fit(features, labels, user_ids)
        ranker.residual_alpha = 0.5
        residual_scores = _residual_scores_from_features(ranker, features)

        self.assertGreater(ranker.score(features[[0, 2]]).mean(), ranker.score(features[[1, 3]]).mean())
        self.assertGreater(residual_scores[[0, 2]].mean(), residual_scores[[1, 3]].mean())

    def test_rank_blend_uses_percentile_order_not_score_scale(self) -> None:
        learned = np.array([100.0, 10.0, 1.0], dtype=np.float32)
        hybrid = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        ranks = _rank_percentiles(learned)
        blended = _blend_scores(learned, hybrid, 0.5, mode="rank")

        self.assertTrue(np.allclose(ranks, np.array([1.0, 0.5, 0.0], dtype=np.float32)))
        self.assertTrue(np.all(blended >= 0.0))
        self.assertTrue(np.all(blended <= 1.0))

    def test_standard_config_uses_explicit_negative_sampling(self) -> None:
        self.assertEqual(standard_learning_to_rank_config().negative_sampling, "explicit_hard")

    def test_popularity_fallback_ignores_mf_and_collaborative_scores(self) -> None:
        features = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32)
        features[0, FEATURE_NAMES.index("mf_score")] = 1.0
        features[0, FEATURE_NAMES.index("collaborative_score")] = 1.0
        features[0, FEATURE_NAMES.index("quality_score")] = 0.1
        features[0, FEATURE_NAMES.index("movie_log_count")] = 0.1
        features[1, FEATURE_NAMES.index("quality_score")] = 0.9
        features[1, FEATURE_NAMES.index("movie_log_count")] = 0.9

        scores = _popularity_fallback_scores(features)

        self.assertGreater(scores[1], scores[0])

    def test_signed_profile_and_item_item_features_emit_signal(self) -> None:
        rows = _ratings()
        model = MatrixFactorizationModel(
            MatrixFactorizationConfig(factors=4, epochs=3, backend="numpy", seed=3)
        )
        model.fit(rows)
        feature_stats = _feature_stats(rows, model=model)
        indexes = _build_collaborative_indexes(rows)
        user_profile = indexes[0][1]
        candidate_ids = np.array([10, 11, 12, 13], dtype=np.int32)

        features = features_for_user_candidates(
            model,
            1,
            candidate_ids,
            active_profile=user_profile,
            exclude_movie_ids=set(),
            quality_scores={movie_id: 0.5 for movie_id in candidate_ids},
            indexes=indexes,
            feature_stats=feature_stats,
        )

        item_item = features[:, FEATURE_NAMES.index("item_item_score")]
        negative_similarity = features[:, FEATURE_NAMES.index("negative_profile_similarity")]
        signed_similarity = features[:, FEATURE_NAMES.index("signed_profile_similarity")]
        self.assertGreater(float(item_item.max()), 0.0)
        self.assertGreater(float(negative_similarity.max()), 0.0)
        self.assertTrue(np.all((signed_similarity >= 0.0) & (signed_similarity <= 1.0)))

    def test_smoke_run_writes_learning_to_rank_outputs(self) -> None:
        if not Path(DEFAULT_DB_PATH).exists():
            self.skipTest("Netflix DuckDB database is not available")

        config = LearningToRankConfig(
            max_users=30,
            candidate_limit=180,
            epochs=5,
            mf_factors=8,
            mf_epochs=2,
            mf_batch_size=2048,
            evaluate_residual_ranker=True,
        )
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = run_learning_to_rank(Path(tmpdir), config=config)

            self.assertTrue(paths["metrics"].exists())
            self.assertTrue(paths["ranker_training"].exists())
            self.assertTrue(paths["blend_tuning"].exists())
            self.assertTrue(paths["candidate_recall"].exists())
            self.assertTrue(paths["feature_weights"].exists())
            self.assertTrue(paths["residual_training"].exists())
            self.assertTrue(paths["residual_tuning"].exists())
            self.assertIn("stacked_linear_hybrid_reranker", paths["metrics"].read_text(encoding="utf-8-sig"))
            self.assertIn("residual_hybrid_linear_reranker", paths["metrics"].read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
