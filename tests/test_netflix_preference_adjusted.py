from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb

from src.datasets.netflix.scoring import build_movie_scores, load_movie_scores, rank_movie_scores


def _create_db(path: Path) -> None:
    conn = duckdb.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE movies (
                movie_id INTEGER PRIMARY KEY,
                release_year INTEGER,
                title VARCHAR NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO movies VALUES (?, ?, ?)",
            [
                (1, 2001, "Lenient Four"),
                (2, 2002, "Strict Four"),
                (101, 2003, "Lenient Baseline"),
                (102, 2004, "Strict Baseline"),
            ],
        )
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
                (1, 10, 4),
                (101, 10, 5),
                (2, 20, 4),
                (102, 20, 2),
            ],
        )
        conn.execute(
            """
            CREATE TABLE movie_stats AS
            SELECT movie_id, COUNT(*)::INTEGER AS rating_count, AVG(rating)::DOUBLE AS rating_avg
            FROM ratings
            GROUP BY movie_id
            """
        )
        conn.execute(
            """
            CREATE TABLE user_stats AS
            SELECT user_id, COUNT(*)::INTEGER AS rating_count, AVG(rating)::DOUBLE AS rating_avg
            FROM ratings
            GROUP BY user_id
            """
        )
    finally:
        conn.close()


class NetflixPreferenceAdjustedTest(unittest.TestCase):
    def test_preference_adjusted_score_accounts_for_user_leniency(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = Path(tmpdir) / "netflix.duckdb"
            _create_db(db_path)

            build_movie_scores(db_path, prior_rating_count=1.0, recent_days=180)
            scores = load_movie_scores(db_path)
            by_title = {item["title"]: item for item in scores}

            self.assertEqual(
                by_title["Lenient Four"]["bayesian_rating"],
                by_title["Strict Four"]["bayesian_rating"],
            )
            self.assertLess(
                by_title["Lenient Four"]["preference_adjusted_bayesian_rating"],
                by_title["Strict Four"]["preference_adjusted_bayesian_rating"],
            )

            targets = [by_title["Lenient Four"], by_title["Strict Four"]]
            top = rank_movie_scores(targets, n=1, score_mode="preference_adjusted")
            self.assertEqual(top[0]["title"], "Strict Four")


if __name__ == "__main__":
    unittest.main()
