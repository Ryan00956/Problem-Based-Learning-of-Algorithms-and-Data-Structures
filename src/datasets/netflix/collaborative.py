from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Protocol

import duckdb

from src.datasets.netflix.import_duckdb import DEFAULT_DB_PATH


class BehaviorEvent(Protocol):
    event_type: str
    movie_id: int | None


MIN_PROFILE_MOVIES = 3
MIN_SHARED_MOVIES = 2
NEIGHBOR_LIMIT = 32
SUPPORT_LIMIT = 2
OVERLAP_SMOOTHING = 5.0
USER_NORMS_TABLE = "user_norms"


@dataclass(frozen=True)
class NetflixCollaborativeCandidate:
    movie_id: int
    score: float
    neighbor_count: int
    support_count: int
    avg_neighbor_rating: float
    max_similarity: float
    shared_movie_count: int
    preference_score: float


class NetflixCollaborativeModel:
    """DuckDB-backed user-user collaborative filtering for Netflix Prize ratings."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path

    def recommend(
        self,
        events: Iterable[BehaviorEvent],
        exclude_movie_ids: set[int],
        limit: int,
    ) -> dict:
        active_profile = _active_profile_from_events(events)
        return self.recommend_for_profile(
            active_profile,
            exclude_movie_ids=exclude_movie_ids,
            limit=limit,
            min_profile_movies=MIN_PROFILE_MOVIES,
            min_shared_movies=MIN_SHARED_MOVIES,
            support_limit=SUPPORT_LIMIT,
        )

    def recommend_for_profile(
        self,
        active_profile: dict[int, float],
        *,
        exclude_movie_ids: set[int],
        limit: int,
        min_profile_movies: int = MIN_PROFILE_MOVIES,
        min_shared_movies: int = MIN_SHARED_MOVIES,
        support_limit: int = SUPPORT_LIMIT,
    ) -> dict:
        if len(active_profile) < min_profile_movies:
            return _empty_result(active_profile)

        centered_profile, active_norm = _center_profile(active_profile)
        if not active_norm:
            return _empty_result(active_profile)

        with duckdb.connect(str(self.db_path), read_only=True) as conn:
            neighbors = self._similar_neighbors(
                conn,
                centered_profile,
                active_norm,
                min_shared_movies=min_shared_movies,
            )
            if not neighbors:
                return _empty_result(active_profile)

            candidates = self._score_candidates(
                conn,
                neighbors,
                exclude_movie_ids=exclude_movie_ids,
                limit=limit,
                support_limit=support_limit,
            )

        return {
            "candidates": candidates,
            "eligible_movie_count": len(active_profile),
            "neighbor_count": len(neighbors),
            "top_similarity": round(neighbors[0]["similarity"], 4),
            "max_shared_movies": max(item["shared_movie_count"] for item in neighbors),
        }

    def _similar_neighbors(
        self,
        conn: duckdb.DuckDBPyConnection,
        centered_profile: dict[int, float],
        active_norm: float,
        *,
        min_shared_movies: int,
    ) -> list[dict]:
        active_values = _numeric_values_clause(
            (movie_id, value)
            for movie_id, value in centered_profile.items()
            if value
        )
        if active_values is None:
            return []

        if _has_user_norms(conn):
            rows = conn.execute(
                f"""
                WITH active(movie_id, active_value) AS (
                    VALUES {active_values}
                ),
                dots AS (
                    SELECT
                        ratings.user_id,
                        SUM(active.active_value * (ratings.rating::DOUBLE - user_stats.rating_avg::DOUBLE)) AS dot_score,
                        COUNT(*)::INTEGER AS shared_movie_count
                    FROM active
                    JOIN ratings ON ratings.movie_id = active.movie_id
                    JOIN user_stats ON user_stats.user_id = ratings.user_id
                    GROUP BY ratings.user_id
                    HAVING COUNT(*) >= {int(min_shared_movies)}
                )
                SELECT
                    dots.user_id,
                    dots.dot_score,
                    dots.shared_movie_count,
                    user_norms.rating_norm AS user_norm
                FROM dots
                JOIN user_norms ON user_norms.user_id = dots.user_id
                WHERE user_norms.rating_norm > 0
                """
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                WITH active(movie_id, active_value) AS (
                    VALUES {active_values}
                ),
                dots AS (
                    SELECT
                        ratings.user_id,
                        SUM(active.active_value * (ratings.rating::DOUBLE - user_stats.rating_avg::DOUBLE)) AS dot_score,
                        COUNT(*)::INTEGER AS shared_movie_count
                    FROM active
                    JOIN ratings ON ratings.movie_id = active.movie_id
                    JOIN user_stats ON user_stats.user_id = ratings.user_id
                    GROUP BY ratings.user_id
                    HAVING COUNT(*) >= {int(min_shared_movies)}
                ),
                norms AS (
                    SELECT
                        ratings.user_id,
                        SQRT(SUM(POWER(ratings.rating::DOUBLE - user_stats.rating_avg::DOUBLE, 2))) AS user_norm
                    FROM ratings
                    JOIN user_stats ON user_stats.user_id = ratings.user_id
                    JOIN dots ON dots.user_id = ratings.user_id
                    GROUP BY ratings.user_id, user_stats.rating_avg
                )
                SELECT
                    dots.user_id,
                    dots.dot_score,
                    dots.shared_movie_count,
                    norms.user_norm
                FROM dots
                JOIN norms ON norms.user_id = dots.user_id
                WHERE norms.user_norm > 0
                """
            ).fetchall()

        neighbors = []
        for user_id, dot_score, shared_count, user_norm in rows:
            raw_similarity = float(dot_score) / (active_norm * float(user_norm))
            if raw_similarity <= 0:
                continue

            overlap_weight = int(shared_count) / (int(shared_count) + OVERLAP_SMOOTHING)
            neighbors.append(
                {
                    "user_id": int(user_id),
                    "similarity": raw_similarity * overlap_weight,
                    "shared_movie_count": int(shared_count),
                }
            )

        return sorted(neighbors, key=lambda item: item["similarity"], reverse=True)[:NEIGHBOR_LIMIT]

    def _score_candidates(
        self,
        conn: duckdb.DuckDBPyConnection,
        neighbors: list[dict],
        *,
        exclude_movie_ids: set[int],
        limit: int,
        support_limit: int,
    ) -> list[NetflixCollaborativeCandidate]:
        neighbor_values = _numeric_values_clause(
            (item["user_id"], item["similarity"], item["shared_movie_count"])
            for item in neighbors
        )
        if neighbor_values is None:
            return []

        exclude_sql = ""
        if exclude_movie_ids:
            exclude_values = ", ".join(str(int(movie_id)) for movie_id in sorted(exclude_movie_ids))
            exclude_sql = f"AND ratings.movie_id NOT IN ({exclude_values})"

        rows = conn.execute(
            f"""
            WITH neighbors(user_id, similarity, shared_movie_count) AS (
                VALUES {neighbor_values}
            )
            SELECT
                ratings.movie_id,
                SUM(neighbors.similarity * (ratings.rating::DOUBLE - user_stats.rating_avg::DOUBLE)) AS weighted_score,
                SUM(ABS(neighbors.similarity)) AS similarity_sum,
                SUM(neighbors.similarity * ratings.rating::DOUBLE) AS weighted_rating,
                COUNT(*)::INTEGER AS support_count,
                MAX(neighbors.similarity) AS max_similarity,
                MAX(neighbors.shared_movie_count)::INTEGER AS max_shared_movie_count
            FROM neighbors
            JOIN ratings ON ratings.user_id = neighbors.user_id
            JOIN user_stats ON user_stats.user_id = ratings.user_id
            WHERE ratings.rating >= 4
              AND (ratings.rating::DOUBLE - user_stats.rating_avg::DOUBLE) > 0
              {exclude_sql}
            GROUP BY ratings.movie_id
            HAVING COUNT(*) >= {int(support_limit)}
               AND SUM(ABS(neighbors.similarity)) > 0
            """
        ).fetchall()

        candidates = []
        for (
            movie_id,
            weighted_score,
            similarity_sum,
            weighted_rating,
            support_count,
            max_similarity,
            shared_movie_count,
        ) in rows:
            similarity_sum = float(similarity_sum)
            preference_score = float(weighted_score) / similarity_sum
            avg_rating = float(weighted_rating) / similarity_sum
            support_bonus = min(math.log1p(int(support_count)) / math.log1p(NEIGHBOR_LIMIT), 1.0)
            score = preference_score * 65.0 + (avg_rating / 5.0) * 20.0 + support_bonus * 15.0
            candidates.append(
                NetflixCollaborativeCandidate(
                    movie_id=int(movie_id),
                    score=round(score, 4),
                    neighbor_count=len(neighbors),
                    support_count=int(support_count),
                    avg_neighbor_rating=round(avg_rating, 4),
                    max_similarity=round(float(max_similarity), 4),
                    shared_movie_count=int(shared_movie_count),
                    preference_score=round(preference_score, 4),
                )
            )

        return sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]


def enrich_candidates(
    candidates: Iterable[NetflixCollaborativeCandidate],
    scores_by_id: dict[int, dict],
) -> list[dict]:
    items = []
    for candidate in candidates:
        movie = scores_by_id.get(candidate.movie_id)
        if not movie:
            continue

        enriched = dict(movie)
        enriched.setdefault("movieId", candidate.movie_id)
        enriched["collaborative_score"] = candidate.score
        enriched["personal_score"] = candidate.score
        enriched["similar_user_count"] = candidate.neighbor_count
        enriched["collaborative_support"] = candidate.support_count
        enriched["neighbor_avg_rating"] = candidate.avg_neighbor_rating
        enriched["max_user_similarity"] = candidate.max_similarity
        enriched["shared_movie_count"] = candidate.shared_movie_count
        enriched["preference_score"] = candidate.preference_score
        enriched["recommendation_bucket"] = "collaborative"
        enriched["recommendation_reason"] = (
            f"{candidate.support_count} similar users rated this around "
            f"{candidate.avg_neighbor_rating:.1f}/5."
        )
        items.append(enriched)
    return items


def _active_profile_from_events(events: Iterable[BehaviorEvent]) -> dict[int, float]:
    profile: dict[int, tuple[float, float, int]] = {}
    for index, event in enumerate(events):
        if event.movie_id is None:
            continue

        signal = _pseudo_rating(event.event_type)
        if signal is None:
            continue

        rating, confidence = signal
        movie_id = int(event.movie_id)
        previous = profile.get(movie_id)
        if previous is None or confidence > previous[1] or (
            confidence == previous[1] and index >= previous[2]
        ):
            profile[movie_id] = (rating, confidence, index)

    return {
        movie_id: rating
        for movie_id, (rating, confidence, _index) in profile.items()
        if confidence >= 0.3
    }


def _pseudo_rating(event_type: str) -> tuple[float, float] | None:
    if event_type == "like":
        return 5.0, 1.0
    if event_type == "dislike":
        return 1.0, 1.0
    if event_type == "view":
        return 3.6, 0.35
    return None


def _center_profile(profile: dict[int, float]) -> tuple[dict[int, float], float]:
    mean_rating = 3.0
    centered = {
        movie_id: rating - mean_rating
        for movie_id, rating in profile.items()
    }
    norm = math.sqrt(sum(value * value for value in centered.values()))
    return centered, norm


def _empty_result(active_profile: dict[int, float]) -> dict:
    return {
        "candidates": [],
        "eligible_movie_count": len(active_profile),
        "neighbor_count": 0,
        "top_similarity": 0.0,
        "max_shared_movies": 0,
    }


def _has_user_norms(conn: duckdb.DuckDBPyConnection) -> bool:
    return bool(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [USER_NORMS_TABLE],
        ).fetchone()[0]
    )


def _numeric_values_clause(rows: Iterable[tuple]) -> str | None:
    parts = []
    for row in rows:
        parts.append("(" + ", ".join(_sql_number(value) for value in row) + ")")
    if not parts:
        return None
    return ", ".join(parts)


def _sql_number(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError("boolean values are not valid numeric SQL literals")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float is not a valid SQL literal")
        return repr(float(value))
    return repr(float(value))
