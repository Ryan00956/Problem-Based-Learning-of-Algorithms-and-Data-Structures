from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.datasets.netflix.import_duckdb import DEFAULT_DB_PATH
from src.datasets.netflix.neural_reranker import (
    FEATURE_NAMES,
    NeuralReranker,
    NeuralRerankerConfig,
    RerankerExamples,
    run_neural_reranker,
    standard_neural_reranker_config,
)


class NetflixNeuralRerankerTest(unittest.TestCase):
    def test_neural_reranker_learns_simple_separable_examples(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")

        base_features = np.array(
            [
                [0.9, 0.1, 0.9, 0.8, 0.7, 0.8, 0.5],
                [0.8, 0.2, 0.8, 0.7, 0.6, 0.8, 0.5],
                [0.1, 0.0, 0.1, 0.2, 0.2, 0.8, 0.5],
                [0.2, 0.0, 0.2, 0.3, 0.1, 0.8, 0.5],
            ],
            dtype=np.float32,
        )
        features = np.zeros((len(base_features), len(FEATURE_NAMES)), dtype=np.float32)
        features[:, : base_features.shape[1]] = base_features
        examples = RerankerExamples(
            user_ids=np.array([1, 1, 1, 1], dtype=np.int64),
            movie_ids=np.array([10, 11, 12, 13], dtype=np.int64),
            features=features,
            labels=np.array([1, 1, 0, 0], dtype=np.float32),
        )
        config = NeuralRerankerConfig(
            epochs=20,
            embedding_dim=4,
            hidden_dim=16,
            batch_size=4,
            learning_rate=0.01,
        )
        ranker = NeuralReranker(
            config,
            user_to_index={1: 0},
            movie_to_index={10: 0, 11: 1, 12: 2, 13: 3},
        )

        ranker.fit(examples)
        scores = ranker.score(examples.user_ids, examples.movie_ids, examples.features)

        self.assertEqual(ranker.backend_used, "torch")
        self.assertGreater(float(scores[:2].mean()), float(scores[2:].mean()))

    def test_standard_config_uses_explicit_negative_sampling(self) -> None:
        self.assertEqual(standard_neural_reranker_config().negative_sampling, "explicit_hard")

    def test_smoke_run_writes_neural_reranker_outputs(self) -> None:
        if not Path(DEFAULT_DB_PATH).exists():
            self.skipTest("Netflix DuckDB database is not available")
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")

        config = NeuralRerankerConfig(
            max_users=30,
            candidate_limit=180,
            negatives_per_positive=4,
            epochs=2,
            mf_factors=8,
            mf_epochs=2,
            mf_batch_size=2048,
            embedding_dim=8,
            hidden_dim=16,
            batch_size=1024,
        )
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = run_neural_reranker(Path(tmpdir), config=config)

            self.assertTrue(paths["metrics"].exists())
            self.assertTrue(paths["neural_training"].exists())
            self.assertTrue(paths["blend_tuning"].exists())
            self.assertTrue(paths["candidate_recall"].exists())
            self.assertIn("stacked_neural_hybrid_reranker", paths["metrics"].read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
