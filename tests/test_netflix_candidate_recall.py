from __future__ import annotations

import unittest

import numpy as np

from src.datasets.netflix.candidate_recall import build_multi_route_candidate_pool
from src.datasets.netflix.evaluation import (
    EvaluationConfig,
    _build_collaborative_indexes,
    movie_quality_scores,
    split_by_user_time,
)
from src.datasets.netflix.learning_to_rank import _feature_stats
from src.datasets.netflix.matrix_factorization import MatrixFactorizationConfig, MatrixFactorizationModel
from tests.test_netflix_matrix_factorization import _time_rows


class NetflixCandidateRecallTest(unittest.TestCase):
    def test_multi_route_candidate_pool_keeps_relevant_movies_and_sources(self) -> None:
        result, split, model = self._build_result("legacy")

        relevant_movies = {
            movie_id
            for values in split["relevant_by_user"].values()
            for movie_id in values
            if movie_id in model.movie_to_index
        }
        self.assertTrue(relevant_movies.issubset(set(map(int, result.movie_ids))))
        sources = {row["source"] for row in result.source_rows}
        self.assertIn("popular_quality", sources)
        self.assertIn("mf_user_top", sources)
        self.assertIn("item_item_cf", sources)
        self.assertIn("all_routes", sources)
        route_rows = [row for row in result.source_rows if row["source"] not in {"all_routes", "relevant_backfill"}]
        self.assertTrue(all(row["strategy"] == "legacy" for row in route_rows))
        self.assertTrue(all("quota" in row for row in route_rows))
        self.assertTrue(all("rrf_weight" in row for row in route_rows))
        self.assertTrue(all("raw_precision" in row for row in route_rows))
        self.assertTrue(all("route_precision" in row for row in route_rows))
        self.assertTrue(all("route_marginal_recall" in row for row in route_rows))
        self.assertTrue(any(features["recall_surfaced"] == 1.0 for features in result.movie_features.values()))
        self.assertTrue(np.issubdtype(result.movie_ids.dtype, np.integer))

    def test_weighted_rrf_strategy_reports_quotas(self) -> None:
        result, _split, _model = self._build_result("weighted_rrf")

        route_rows = [row for row in result.source_rows if row["source"] not in {"all_routes", "relevant_backfill"}]
        self.assertTrue(all(row["strategy"] == "weighted_rrf" for row in route_rows))
        self.assertTrue(all(row["quota"] >= 0 for row in route_rows))
        self.assertTrue(all(row["rrf_weight"] > 0 for row in route_rows))

    def _build_result(self, strategy: str):
        config = EvaluationConfig(
            max_users=3,
            min_ratings_per_user=5,
            test_ratio=0.4,
            top_k=2,
            candidate_limit=6,
            candidate_recall_strategy=strategy,
        )
        split = split_by_user_time(_time_rows(), config)
        model = MatrixFactorizationModel(MatrixFactorizationConfig(factors=4, epochs=2, backend="numpy"))
        model.fit(split["train"])
        quality_scores = movie_quality_scores(split)
        indexes = _build_collaborative_indexes(split["train"])
        feature_stats = _feature_stats(split["train"], model=model)

        result = build_multi_route_candidate_pool(
            split,
            config,
            model,
            quality_scores=quality_scores,
            indexes=indexes,
            feature_stats=feature_stats,
        )
        return result, split, model


if __name__ == "__main__":
    unittest.main()
