from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb

from src.algorithms.sorting import merge_sort, top_n_heap
from src.datasets.netflix.import_duckdb import DEFAULT_DB_PATH


DEFAULT_PRIOR_RATING_COUNT = 563.0
DEFAULT_RECENT_DAYS = 180


def build_movie_scores(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    prior_rating_count: float = DEFAULT_PRIOR_RATING_COUNT,
    recent_days: int = DEFAULT_RECENT_DAYS,
) -> dict[str, Any]:
    if prior_rating_count <= 0:
        raise ValueError("prior_rating_count must be positive")
    if recent_days <= 0:
        raise ValueError("recent_days must be positive")

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            f"""
            CREATE OR REPLACE TABLE movie_recent_stats AS
            WITH dataset_window AS (
                SELECT MAX(rating_date) AS max_rating_date
                FROM ratings
            )
            SELECT
                ratings.movie_id,
                COUNT(*)::INTEGER AS recent_rating_count
            FROM ratings, dataset_window
            WHERE ratings.rating_date >= dataset_window.max_rating_date - INTERVAL {int(recent_days)} DAY
            GROUP BY ratings.movie_id
            """
        )
        conn.execute(
            """
            CREATE OR REPLACE TABLE movie_scores AS
            WITH constants AS (
                SELECT
                    (SUM(rating_avg * rating_count) / SUM(rating_count))::DOUBLE AS global_avg_rating,
                    MAX(rating_count)::DOUBLE AS max_rating_count,
                    COALESCE((SELECT MAX(recent_rating_count) FROM movie_recent_stats), 0)::DOUBLE AS max_recent_rating_count,
                    ?::DOUBLE AS prior_rating_count,
                    ?::INTEGER AS recent_days,
                    (SELECT MAX(rating_date) FROM ratings) AS max_rating_date
                FROM movie_stats
            ),
            base AS (
                SELECT
                    movies.movie_id,
                    movies.title,
                    movies.release_year,
                    movie_stats.rating_avg::DOUBLE AS avg_rating,
                    movie_stats.rating_count::INTEGER AS rating_count,
                    COALESCE(movie_recent_stats.recent_rating_count, 0)::INTEGER AS recent_rating_count,
                    constants.global_avg_rating,
                    constants.max_rating_count,
                    constants.max_recent_rating_count,
                    constants.prior_rating_count,
                    constants.recent_days,
                    constants.max_rating_date
                FROM movies
                JOIN movie_stats ON movies.movie_id = movie_stats.movie_id
                LEFT JOIN movie_recent_stats ON movies.movie_id = movie_recent_stats.movie_id
                CROSS JOIN constants
            ),
            scored AS (
                SELECT
                    *,
                    ((rating_count * avg_rating + prior_rating_count * global_avg_rating)
                        / (rating_count + prior_rating_count))::DOUBLE AS bayesian_rating
                FROM base
            ),
            components AS (
                SELECT
                    *,
                    (bayesian_rating / 5.0) * 70.0 AS rating_score,
                    CASE
                        WHEN max_rating_count > 0 THEN (LN(1 + rating_count) / LN(1 + max_rating_count)) * 25.0
                        ELSE 0.0
                    END AS popularity_score,
                    CASE
                        WHEN max_recent_rating_count > 0 THEN (LN(1 + recent_rating_count) / LN(1 + max_recent_rating_count)) * 5.0
                        ELSE 0.0
                    END AS freshness_score
                FROM scored
            )
            SELECT
                movie_id,
                title,
                release_year,
                ROUND(avg_rating, 4) AS avg_rating,
                ROUND(bayesian_rating, 4) AS bayesian_rating,
                rating_count,
                recent_rating_count,
                ROUND(rating_score, 4) AS rating_score,
                ROUND(popularity_score, 4) AS popularity_score,
                ROUND(freshness_score, 4) AS freshness_score,
                ROUND(rating_score + popularity_score + freshness_score, 4) AS comprehensive_score,
                ROUND(global_avg_rating, 4) AS global_avg_rating,
                prior_rating_count,
                recent_days,
                max_rating_date
            FROM components
            """,
            [prior_rating_count, int(recent_days)],
        )
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS movie_count,
                MAX(rating_count) AS max_rating_count,
                MAX(recent_rating_count) AS max_recent_rating_count,
                MAX(max_rating_date) AS max_rating_date,
                MAX(global_avg_rating) AS global_avg_rating,
                MAX(comprehensive_score) AS top_score
            FROM movie_scores
            """
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise RuntimeError("failed to build Netflix movie scores")
    keys = [
        "movie_count",
        "max_rating_count",
        "max_recent_rating_count",
        "max_rating_date",
        "global_avg_rating",
        "top_score",
    ]
    return dict(zip(keys, row, strict=True))


def top_movie_scores(db_path: Path = DEFAULT_DB_PATH, *, n: int = 10) -> list[dict[str, Any]]:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        if not _table_exists(conn, "movie_scores"):
            conn.close()
            build_movie_scores(db_path)
            conn = duckdb.connect(str(db_path), read_only=True)
        rows = conn.execute(
            """
            SELECT
                movie_id,
                title,
                release_year,
                avg_rating,
                bayesian_rating,
                rating_count,
                recent_rating_count,
                rating_score,
                popularity_score,
                freshness_score,
                comprehensive_score
            FROM movie_scores
            ORDER BY comprehensive_score DESC, rating_count DESC, movie_id ASC
            LIMIT ?
            """,
            [int(n)],
        ).fetchall()
    finally:
        conn.close()

    return [_movie_score_from_row(row) for row in rows]


def load_movie_scores(db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        if not _table_exists(conn, "movie_scores"):
            conn.close()
            build_movie_scores(db_path)
            conn = duckdb.connect(str(db_path), read_only=True)
        rows = conn.execute(
            """
            SELECT
                movie_id,
                title,
                release_year,
                avg_rating,
                bayesian_rating,
                rating_count,
                recent_rating_count,
                rating_score,
                popularity_score,
                freshness_score,
                comprehensive_score
            FROM movie_scores
            ORDER BY movie_id ASC
            """,
        ).fetchall()
    finally:
        conn.close()

    return [_movie_score_from_row(row) for row in rows]


def rank_movie_scores(scores: list[dict[str, Any]], *, n: int = 10, algorithm: str = "heap") -> list[dict[str, Any]]:
    if algorithm == "merge":
        return merge_sort(scores, key="comprehensive_score", reverse=True)[:n]
    if algorithm == "heap":
        return top_n_heap(scores, n=n, key="comprehensive_score", reverse=True)
    raise ValueError(f"unknown sorting algorithm: {algorithm}")


def _movie_score_from_row(row: tuple) -> dict[str, Any]:
    columns = [
        "movie_id",
        "title",
        "release_year",
        "avg_rating",
        "bayesian_rating",
        "rating_count",
        "recent_rating_count",
        "rating_score",
        "popularity_score",
        "freshness_score",
        "comprehensive_score",
    ]
    return dict(zip(columns, row, strict=True))


def _table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return bool(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and inspect Netflix Prize movie scores.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--prior-rating-count", type=float, default=DEFAULT_PRIOR_RATING_COUNT)
    parser.add_argument("--recent-days", type=int, default=DEFAULT_RECENT_DAYS)
    parser.add_argument("-n", type=int, default=10, help="Number of top movies to print after scoring.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_movie_scores(
        args.db_path,
        prior_rating_count=args.prior_rating_count,
        recent_days=args.recent_days,
    )
    print("Netflix movie_scores built:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print()
    print(f"Top {args.n} movies by comprehensive_score:")
    for index, movie in enumerate(top_movie_scores(args.db_path, n=args.n), start=1):
        year = movie["release_year"] if movie["release_year"] is not None else "unknown"
        print(
            f"{index:>2}. {movie['title']} ({year}) "
            f"score={movie['comprehensive_score']:.2f} "
            f"bayes={movie['bayesian_rating']:.2f} "
            f"ratings={movie['rating_count']} "
            f"recent={movie['recent_rating_count']}"
        )


if __name__ == "__main__":
    main()
