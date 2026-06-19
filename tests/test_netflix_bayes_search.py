from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.datasets.netflix.bayes_search import (
    BayesSearchConfig,
    _objective_score,
    _recall_objective_score,
    load_recall_search_settings,
    run_bayes_search,
    smoke_bayes_search_config,
)


class NetflixBayesSearchTest(unittest.TestCase):
    def test_objective_score_rewards_precision_and_penalizes_low_coverage(self) -> None:
        strong = {
            "precision_at_k": 0.10,
            "map_at_k": 0.04,
            "hit_rate_at_k": 0.50,
            "catalog_coverage": 0.10,
        }
        weak_coverage = {
            "precision_at_k": 0.10,
            "map_at_k": 0.04,
            "hit_rate_at_k": 0.50,
            "catalog_coverage": 0.01,
        }

        self.assertGreater(_objective_score(strong), _objective_score(weak_coverage))

    def test_recall_objective_enforces_minimum_route_recall(self) -> None:
        feasible = {
            "route_precision": 0.65,
            "route_recall": 0.56,
            "route_marginal_recall_sum": 0.05,
        }
        infeasible = {
            "route_precision": 0.95,
            "route_recall": 0.30,
            "route_marginal_recall_sum": 0.20,
        }

        self.assertGreater(_recall_objective_score(feasible, 0.55), 0)
        self.assertLess(_recall_objective_score(infeasible, 0.55), 0)

    def test_smoke_config_sets_small_search(self) -> None:
        config = smoke_bayes_search_config("hybrid")

        self.assertEqual(config.mode, "hybrid")
        self.assertLessEqual(config.trials, 5)
        self.assertLessEqual(config.max_users, 40)

    def test_load_recall_search_settings_from_best_params(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            path = Path(tmpdir) / "best_params.json"
            params = {"rrf_k": 42.0}
            for source in ("popular_quality", "mf_user_top", "profile_centroid", "user_user_cf", "year_affinity"):
                params[f"{source}_floor"] = 0.1
                params[f"{source}_cap_extra"] = 0.2
                params[f"{source}_base_weight"] = 1.0
            path.write_text(json.dumps({"best_params": params}), encoding="utf-8")

            settings = load_recall_search_settings(path)

            self.assertEqual(settings["candidate_recall_strategy"], "weighted_rrf")
            self.assertEqual(settings["candidate_recall_rrf_k"], 42.0)
            self.assertEqual(len(settings["candidate_recall_priors"]), 5)

    def test_bayes_search_writes_outputs_and_resumes(self) -> None:
        try:
            import optuna  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("Optuna is not installed")

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            output_dir = Path(tmpdir)
            config = BayesSearchConfig(
                mode="hybrid",
                trials=2,
                max_users=20,
                candidate_limit=120,
            )

            paths = run_bayes_search(output_dir, config=config)
            paths = run_bayes_search(output_dir, config=config)

            self.assertTrue(paths["study_db"].exists())
            self.assertTrue(paths["trials"].exists())
            self.assertTrue(paths["best_params"].exists())
            payload = json.loads(paths["best_params"].read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["mode"], "hybrid")
            self.assertIn("best_metrics", payload)

    def test_recall_bayes_search_writes_route_metrics(self) -> None:
        try:
            import optuna  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("Optuna is not installed")

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            output_dir = Path(tmpdir)
            config = BayesSearchConfig(
                mode="recall",
                trials=1,
                max_users=20,
                candidate_limit=120,
                min_route_recall=0.20,
            )

            paths = run_bayes_search(output_dir, config=config)

            payload = json.loads(paths["best_params"].read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["mode"], "recall")
            self.assertIn("route_precision", payload["best_metrics"])
            self.assertIn("route_recall", payload["best_metrics"])


if __name__ == "__main__":
    unittest.main()
