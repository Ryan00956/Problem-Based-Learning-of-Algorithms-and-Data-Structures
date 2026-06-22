from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb

from src.datasets.netflix.collaborative import NetflixCollaborativeModel
from src.datasets.netflix.online_reranker import FEATURE_NAMES, OnlineNetflixReranker
from src.datasets.netflix.recommendation import recommend_for_events, recommend_similar_movies


@dataclass
class Event:
    event_type: str
    movie_id: int | None


def _scores() -> list[dict]:
    rows = [
        (1, "Seed One", 80.0),
        (2, "Seed Two", 79.0),
        (3, "Rejected", 20.0),
        (4, "Neighbor Favorite", 78.0),
        (5, "Weak Neighbor Extra", 70.0),
        (6, "Cold Start Backup", 77.0),
    ]
    return [
        {
            "movie_id": movie_id,
            "movieId": movie_id,
            "title": title,
            "release_year": 2000 + movie_id,
            "avg_rating": 4.0,
            "bayesian_rating": 4.0,
            "rating_count": 100,
            "recent_rating_count": 5,
            "rating_score": score * 0.7,
            "popularity_score": 15.0,
            "freshness_score": 2.0,
            "comprehensive_score": score,
        }
        for movie_id, title, score in rows
    ]


def _create_db(path: Path) -> None:
    conn = duckdb.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE ratings (
                movie_id INTEGER,
                user_id INTEGER,
                rating UTINYINT,
                rating_date DATE
            )
            """
        )
        conn.executemany(
            "INSERT INTO ratings VALUES (?, ?, ?, DATE '2005-01-01')",
            [
                (1, 10, 5),
                (2, 10, 5),
                (3, 10, 1),
                (4, 10, 5),
                (5, 10, 4),
                (1, 11, 5),
                (2, 11, 4),
                (3, 11, 1),
                (4, 11, 5),
                (5, 11, 4),
                (1, 12, 1),
                (2, 12, 2),
                (3, 12, 5),
                (4, 12, 2),
                (5, 12, 1),
            ],
        )
        conn.execute(
            """
            CREATE TABLE user_stats AS
            SELECT user_id, COUNT(*)::INTEGER AS rating_count, AVG(rating)::DOUBLE AS rating_avg
            FROM ratings
            GROUP BY user_id
            """
        )
        conn.execute(
            """
            CREATE TABLE user_norms AS
            SELECT
                ratings.user_id,
                SQRT(SUM(POWER(ratings.rating::DOUBLE - user_stats.rating_avg::DOUBLE, 2))) AS rating_norm
            FROM ratings
            JOIN user_stats ON user_stats.user_id = ratings.user_id
            GROUP BY ratings.user_id
            """
        )
        conn.execute(
            """
            CREATE TABLE movies (
                movie_id INTEGER,
                title VARCHAR,
                release_year INTEGER
            )
            """
        )
        conn.executemany(
            "INSERT INTO movies VALUES (?, ?, ?)",
            [
                (1, "Seed One", 2001),
                (2, "Seed Two", 2002),
                (3, "Rejected", 2003),
                (4, "Neighbor Favorite", 2004),
                (5, "Weak Neighbor Extra", 2005),
                (6, "Cold Start Backup", 2006),
            ],
        )
    finally:
        conn.close()


def _create_reranker_artifact(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "feature_weights.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("feature,weight,mean,std\n")
        for feature in FEATURE_NAMES:
            weight = 0.0
            if feature == "collaborative_score":
                weight = 1.0
            elif feature == "recall_user_user_cf":
                weight = 0.5
            elif feature == "quality_score":
                weight = 0.1
            handle.write(f"{feature},{weight},0.0,1.0\n")
    (path / "summary.json").write_text(
        """
{
  "config": {
    "candidate_limit": 60,
    "candidate_recall_rrf_k": 20.0,
    "candidate_recall_priors": []
  },
  "best_metric": {
    "algorithm": "stacked_linear_hybrid_reranker",
    "precision_at_k": 0.1
  },
  "linear_blend_weight": 1.0
}
""".strip(),
        encoding="utf-8",
    )


class NetflixCollaborativeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "netflix.duckdb"
        _create_db(self.db_path)
        self.model = NetflixCollaborativeModel(self.db_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_collaborative_model_recommends_from_similar_users(self) -> None:
        result = self.model.recommend(
            [
                Event("like", 1),
                Event("like", 2),
                Event("dislike", 3),
            ],
            exclude_movie_ids={1, 2, 3},
            limit=5,
        )

        self.assertEqual(result["eligible_movie_count"], 3)
        self.assertEqual(result["neighbor_count"], 2)
        self.assertEqual(result["candidates"][0].movie_id, 4)
        self.assertEqual(result["candidates"][0].support_count, 2)
        self.assertGreaterEqual(result["candidates"][0].avg_neighbor_rating, 5.0)

    def test_collaborative_model_falls_back_without_user_norms(self) -> None:
        with duckdb.connect(str(self.db_path)) as conn:
            conn.execute("DROP TABLE user_norms")

        result = self.model.recommend(
            [
                Event("like", 1),
                Event("like", 2),
                Event("dislike", 3),
            ],
            exclude_movie_ids={1, 2, 3},
            limit=5,
        )

        self.assertEqual(result["neighbor_count"], 2)
        self.assertEqual(result["candidates"][0].movie_id, 4)

    def test_for_you_uses_collaborative_bucket(self) -> None:
        payload = recommend_for_events(
            [
                Event("like", 1),
                Event("like", 2),
                Event("dislike", 3),
            ],
            _scores(),
            model=self.model,
            n=3,
        )

        self.assertEqual(payload["status"], "personalized")
        self.assertEqual(payload["collaborative"]["neighbor_count"], 2)
        self.assertEqual(payload["items"][0]["movieId"], 4)
        self.assertEqual(payload["items"][0]["recommendation_bucket"], "collaborative")
        self.assertEqual(payload["count"], 3)

    def test_for_you_uses_online_stacked_reranker_when_artifact_exists(self) -> None:
        artifact_dir = Path(self.tmpdir.name) / "artifact"
        _create_reranker_artifact(artifact_dir)
        reranker = OnlineNetflixReranker(self.db_path, artifact_dir)

        payload = recommend_for_events(
            [
                Event("like", 1),
                Event("like", 2),
                Event("dislike", 3),
            ],
            _scores(),
            model=self.model,
            reranker=reranker,
            n=3,
        )

        self.assertEqual(payload["engine"], "netflix_stacked_hybrid_reranker")
        self.assertEqual(payload["fallback_engine"], "netflix_user_user_collaborative")
        self.assertEqual(payload["collaborative"]["neighbor_count"], 2)
        self.assertEqual(payload["items"][0]["movieId"], 4)
        self.assertIn("user_user_cf", payload["items"][0]["recommendation_sources"])
        self.assertEqual(payload["model"]["name"], "stacked_linear_hybrid_reranker")

    def test_title_recommendation_finds_similar_movies(self) -> None:
        target, rows = recommend_similar_movies("Seed One", _scores(), model=self.model, n=3)

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target["title"], "Seed One")
        self.assertTrue(rows)
        self.assertEqual(rows[0]["movieId"], 4)

    def test_title_recommendation_promotes_same_series_before_collaborative_popularity(self) -> None:
        scores = _scores() + [
            {
                "movie_id": 10,
                "movieId": 10,
                "title": "The Matrix",
                "release_year": 1999,
                "avg_rating": 4.6,
                "bayesian_rating": 4.5,
                "rating_count": 1000,
                "recent_rating_count": 30,
                "rating_score": 63.0,
                "popularity_score": 22.0,
                "freshness_score": 5.0,
                "comprehensive_score": 90.0,
            },
            {
                "movie_id": 11,
                "movieId": 11,
                "title": "The Matrix: Reloaded",
                "release_year": 2003,
                "avg_rating": 4.0,
                "bayesian_rating": 4.0,
                "rating_count": 900,
                "recent_rating_count": 20,
                "rating_score": 56.0,
                "popularity_score": 19.0,
                "freshness_score": 4.0,
                "comprehensive_score": 79.0,
            },
            {
                "movie_id": 12,
                "movieId": 12,
                "title": "The Matrix: Revolutions",
                "release_year": 2003,
                "avg_rating": 3.8,
                "bayesian_rating": 3.8,
                "rating_count": 700,
                "recent_rating_count": 15,
                "rating_score": 53.2,
                "popularity_score": 18.0,
                "freshness_score": 3.0,
                "comprehensive_score": 74.2,
            },
            {
                "movie_id": 13,
                "movieId": 13,
                "title": "Sexual Matrix",
                "release_year": 1999,
                "avg_rating": 4.1,
                "bayesian_rating": 4.1,
                "rating_count": 500,
                "recent_rating_count": 8,
                "rating_score": 57.4,
                "popularity_score": 17.0,
                "freshness_score": 2.0,
                "comprehensive_score": 76.4,
            },
        ]

        target, rows = recommend_similar_movies("Matrix", scores, model=self.model, n=3)

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target["title"], "The Matrix")
        self.assertEqual([row["movieId"] for row in rows[:2]], [11, 12])
        self.assertTrue(rows[0]["series_match"])
        self.assertEqual(rows[0]["recommendation_bucket"], "series")
        self.assertNotIn(13, [row["movieId"] for row in rows])


if __name__ == "__main__":
    unittest.main()
