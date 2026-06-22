from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np

from src.core.paths import OUTPUT_DIR
from src.datasets.netflix.collaborative import NetflixCollaborativeModel, _active_profile_from_events
from src.datasets.netflix.import_duckdb import DEFAULT_DB_PATH
from src.datasets.netflix.matrix_factorization import MatrixFactorizationModel


DEFAULT_ARTIFACT_DIR = OUTPUT_DIR / "netflix_residual_stacked_10k_rrf"
FEATURE_NAMES = (
    "mf_score",
    "collaborative_score",
    "quality_score",
    "movie_avg_rating",
    "movie_log_count",
    "user_avg_rating",
    "user_log_count",
    "profile_similarity",
    "negative_profile_similarity",
    "signed_profile_similarity",
    "item_item_score",
    "profile_strength",
    "user_positive_rate",
    "movie_year_norm",
    "movie_has_year",
    "year_affinity",
    "recall_surfaced",
    "recall_route_score",
    "recall_source_count",
    "recall_popular_quality",
    "recall_mf_user_top",
    "recall_profile_centroid",
    "recall_item_item_cf",
    "recall_user_user_cf",
    "recall_year_affinity",
)
SOURCE_NAMES = (
    "popular_quality",
    "mf_user_top",
    "profile_centroid",
    "item_item_cf",
    "user_user_cf",
    "year_affinity",
)
DEFAULT_SOURCE_PRIORS = {
    "popular_quality": (0.2589294786155412, 0.6581508457064098, 1.4416445611747633),
    "mf_user_top": (0.47721500733873834, 0.8790581058615665, 0.5578138480079862),
    "profile_centroid": (0.13494831217730507, 0.970809380510529, 0.9013537176985738),
    "item_item_cf": (0.08, 0.22, 1.05),
    "user_user_cf": (0.533013001552092, 0.6841122000911959, 0.8971168566653216),
    "year_affinity": (0.47957241078561474, 0.7449087161672673, 1.7720341118039984),
}


@dataclass(frozen=True)
class OnlineRerankArtifact:
    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    candidate_limit: int
    rrf_k: float
    source_priors: dict[str, tuple[float, float, float]]
    model_name: str
    metric: dict
    degraded_routes: tuple[str, ...]


class OnlineNetflixReranker:
    """Online adapter for the best saved Netflix stacked hybrid reranker.

    The offline run saves the linear scorer statistics but not MF embeddings.
    This adapter uses every online-available route and marks MF as a proxy route
    until a persisted factor artifact is added.
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    ) -> None:
        self.db_path = db_path
        self.artifact_dir = artifact_dir
        self._artifact: OnlineRerankArtifact | None = None
        self._mf_model: MatrixFactorizationModel | None = None
        self._mf_model_loaded = False
        self._metadata_cache: dict[int, int | None] | None = None
        self._max_user_log_count: float | None = None

    @property
    def available(self) -> bool:
        return self.artifact is not None

    @property
    def artifact(self) -> OnlineRerankArtifact | None:
        if self._artifact is None:
            self._artifact = self._load_artifact()
        return self._artifact

    def recommend_for_events(
        self,
        events: Iterable,
        scores: list[dict],
        *,
        model: NetflixCollaborativeModel,
        n: int,
    ) -> dict | None:
        artifact = self.artifact
        if artifact is None:
            return None

        events = list(events)
        profile = _active_profile_from_events(events)
        exclude_movie_ids = _exclude_movie_ids(events)
        scores_by_id = _scores_by_id(scores)
        quality_scores = _quality_scores(scores)
        movie_years = self._movie_metadata()
        profile_stats = _profile_stats(profile, scores_by_id, movie_years, self._max_user_log())

        routes, collaborative_result = self._candidate_routes(
            profile,
            scores,
            model=model,
            quality_scores=quality_scores,
            exclude_movie_ids=exclude_movie_ids,
            artifact=artifact,
            movie_years=movie_years,
            profile_stats=profile_stats,
        )
        combined, movie_sources, active_routes = _combine_routes(routes, artifact)
        if not combined:
            return None

        candidate_ids = _select_candidate_ids(
            combined,
            movie_sources,
            quality_scores,
            exclude_movie_ids,
            artifact.candidate_limit,
        )
        if not candidate_ids:
            return None

        route_features = _route_features(candidate_ids, combined, movie_sources, active_routes)
        features = _feature_matrix(
            candidate_ids,
            scores_by_id=scores_by_id,
            quality_scores=quality_scores,
            movie_years=movie_years,
            profile_stats=profile_stats,
            route_features=route_features,
            collaborative_scores=_normalize_score_dict(routes.get("user_user_cf", {})),
            mf_proxy_scores=_normalize_score_dict(routes.get("mf_user_top", {})),
        )
        raw_scores = _linear_scores(features, artifact)
        ranked = sorted(
            zip(candidate_ids, raw_scores, strict=True),
            key=lambda item: (
                float(item[1]),
                quality_scores.get(int(item[0]), 0.0),
                -int(item[0]),
            ),
            reverse=True,
        )

        items = []
        for movie_id, personal_score in ranked:
            if int(movie_id) in exclude_movie_ids:
                continue
            movie = scores_by_id.get(int(movie_id))
            if not movie:
                continue
            enriched = dict(movie)
            enriched["movieId"] = int(movie_id)
            enriched["movie_id"] = int(movie_id)
            enriched["personal_score"] = round(float(personal_score), 4)
            enriched["reranker_score"] = round(float(personal_score), 4)
            enriched["recommendation_bucket"] = _dominant_bucket(movie_sources.get(int(movie_id), set()))
            enriched["recommendation_reason"] = _recommendation_reason(enriched["recommendation_bucket"], route_features[int(movie_id)])
            enriched["recommendation_sources"] = sorted(movie_sources.get(int(movie_id), ()))
            enriched["collaborative_score"] = round(float(routes.get("user_user_cf", {}).get(int(movie_id), 0.0)), 4)
            items.append(enriched)
            if len(items) >= n:
                break

        if len(items) < n:
            existing = {int(item["movieId"]) for item in items}
            items.extend(_top_backfill(scores, n - len(items), exclude_movie_ids | existing))

        bucket_counts: dict[str, int] = {}
        for item in items:
            bucket = item.get("recommendation_bucket", "top")
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        neighbor_count = int(collaborative_result.get("neighbor_count", 0))
        return {
            "items": items,
            "count": len(items),
            "event_count": len(events),
            "profile": {
                "seed_movie_count": len(profile),
                "signal_strength": len(profile),
                "positive_movie_count": profile_stats["positive_count"],
                "negative_movie_count": profile_stats["negative_count"],
            },
            "collaborative": {
                "eligible_movie_count": collaborative_result.get("eligible_movie_count", len(profile)),
                "neighbor_count": neighbor_count,
                "top_similarity": collaborative_result.get("top_similarity", 0.0),
                "max_shared_movies": collaborative_result.get("max_shared_movies", 0),
            },
            "bucket_counts": bucket_counts,
            "candidate_count": len(candidate_ids),
            "route_counts": {source: len(values) for source, values in routes.items()},
            "engine": "netflix_stacked_hybrid_reranker",
            "fallback_engine": "netflix_user_user_collaborative",
            "status": "personalized" if neighbor_count or len(profile) >= 3 else "cold_start",
            "model": {
                "name": artifact.model_name,
                "artifact_dir": str(self.artifact_dir),
                "candidate_limit": artifact.candidate_limit,
                "degraded_routes": list(artifact.degraded_routes),
                "offline_metric": artifact.metric,
            },
        }

    def _load_artifact(self) -> OnlineRerankArtifact | None:
        weights_path = self.artifact_dir / "feature_weights.csv"
        summary_path = self.artifact_dir / "summary.json"
        if not weights_path.exists() or not summary_path.exists():
            return None

        weights = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        mean = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        std = np.ones(len(FEATURE_NAMES), dtype=np.float32)
        with weights_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                feature = row.get("feature", "")
                if feature not in FEATURE_NAMES:
                    continue
                index = FEATURE_NAMES.index(feature)
                weights[index] = float(row.get("weight") or 0.0)
                mean[index] = float(row.get("mean") or 0.0)
                std[index] = max(float(row.get("std") or 1.0), 1e-6)

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        priors = dict(DEFAULT_SOURCE_PRIORS)
        for source, floor, cap, base_weight in config.get("candidate_recall_priors") or ():
            priors[str(source)] = (float(floor), float(cap), float(base_weight))

        return OnlineRerankArtifact(
            weights=weights,
            mean=mean,
            std=std,
            candidate_limit=int(config.get("candidate_limit") or 1200),
            rrf_k=float(config.get("candidate_recall_rrf_k") or 60.0),
            source_priors=priors,
            model_name=str(payload.get("best_metric", {}).get("algorithm") or "stacked_linear_hybrid_reranker"),
            metric=dict(payload.get("best_metric") or {}),
            degraded_routes=() if (self.artifact_dir / "mf_model.npz").exists() else ("mf_user_top",),
        )

    def _candidate_routes(
        self,
        profile: dict[int, float],
        scores: list[dict],
        *,
        model: NetflixCollaborativeModel,
        quality_scores: dict[int, float],
        exclude_movie_ids: set[int],
        artifact: OnlineRerankArtifact,
        movie_years: dict[int, int | None],
        profile_stats: dict,
    ) -> tuple[dict[str, dict[int, float]], dict]:
        routes: dict[str, dict[int, float]] = {
            source: {}
            for source in SOURCE_NAMES
        }
        routes["popular_quality"] = _popular_route(quality_scores, artifact.candidate_limit)
        routes["profile_centroid"] = _profile_route(scores, profile_stats, quality_scores, movie_years, exclude_movie_ids)
        mf_model = self._load_mf_model()
        routes["mf_user_top"] = (
            _mf_model_route(mf_model, profile, quality_scores, exclude_movie_ids, artifact.candidate_limit)
            if mf_model is not None
            else _mf_proxy_route(scores, profile_stats, quality_scores, movie_years, exclude_movie_ids)
        )
        routes["year_affinity"] = _year_route(scores, profile_stats, quality_scores, movie_years, exclude_movie_ids)
        routes["item_item_cf"] = self._item_item_route(profile, exclude_movie_ids, artifact.candidate_limit)

        collaborative_result = model.recommend_for_profile(
            profile,
            exclude_movie_ids=exclude_movie_ids,
            limit=artifact.candidate_limit,
        )
        routes["user_user_cf"] = {
            int(candidate.movie_id): float(candidate.score)
            for candidate in collaborative_result.get("candidates", ())
        }
        return routes, collaborative_result

    def _load_mf_model(self) -> MatrixFactorizationModel | None:
        if self._mf_model_loaded:
            return self._mf_model
        self._mf_model_loaded = True
        path = self.artifact_dir / "mf_model.npz"
        if not path.exists():
            return None
        self._mf_model = MatrixFactorizationModel.load(path)
        return self._mf_model

    def _item_item_route(
        self,
        profile: dict[int, float],
        exclude_movie_ids: set[int],
        candidate_limit: int,
    ) -> dict[int, float]:
        seeds = [
            (movie_id, max(float(rating) - 3.0, 0.25))
            for movie_id, rating in profile.items()
            if float(rating) >= 3.6
        ]
        if not seeds or not self.db_path.exists():
            return {}
        seeds = sorted(seeds, key=lambda item: item[1], reverse=True)[:12]
        seed_values = ", ".join(f"({int(movie_id)}, {float(weight)!r})" for movie_id, weight in seeds)
        exclude_sql = ""
        if exclude_movie_ids:
            exclude_values = ", ".join(str(int(movie_id)) for movie_id in sorted(exclude_movie_ids))
            exclude_sql = f"AND candidate.movie_id NOT IN ({exclude_values})"

        with duckdb.connect(str(self.db_path), read_only=True) as conn:
            rows = conn.execute(
                f"""
                WITH seed(movie_id, seed_weight) AS (
                    VALUES {seed_values}
                ),
                seed_users AS (
                    SELECT ratings.user_id, MAX(seed.seed_weight) AS seed_weight
                    FROM seed
                    JOIN ratings ON ratings.movie_id = seed.movie_id
                    WHERE ratings.rating >= 4
                    GROUP BY ratings.user_id
                )
                SELECT
                    candidate.movie_id,
                    COUNT(*)::INTEGER AS support_count,
                    AVG(candidate.rating::DOUBLE) AS avg_rating,
                    SUM(seed_users.seed_weight * GREATEST(candidate.rating::DOUBLE - user_stats.rating_avg::DOUBLE, 0)) AS weighted_preference
                FROM seed_users
                JOIN ratings AS candidate ON candidate.user_id = seed_users.user_id
                JOIN user_stats ON user_stats.user_id = candidate.user_id
                WHERE candidate.rating >= 4
                  {exclude_sql}
                GROUP BY candidate.movie_id
                HAVING COUNT(*) >= 2
                ORDER BY weighted_preference DESC, support_count DESC, candidate.movie_id ASC
                LIMIT ?
                """,
                [int(candidate_limit)],
            ).fetchall()

        return {
            int(movie_id): float(weighted_preference or 0.0) + math.log1p(int(support_count)) + float(avg_rating or 0.0) / 5.0
            for movie_id, support_count, avg_rating, weighted_preference in rows
        }

    def _movie_metadata(self) -> dict[int, int | None]:
        if self._metadata_cache is not None:
            return self._metadata_cache
        if not self.db_path.exists():
            self._metadata_cache = {}
            return self._metadata_cache
        with duckdb.connect(str(self.db_path), read_only=True) as conn:
            table_count = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'movies'"
            ).fetchone()[0]
            if not table_count:
                self._metadata_cache = {}
                return self._metadata_cache
            rows = conn.execute("SELECT movie_id, release_year FROM movies").fetchall()
        self._metadata_cache = {
            int(movie_id): None if release_year is None else int(release_year)
            for movie_id, release_year in rows
        }
        return self._metadata_cache

    def _max_user_log(self) -> float:
        if self._max_user_log_count is not None:
            return self._max_user_log_count
        if not self.db_path.exists():
            self._max_user_log_count = 1.0
            return self._max_user_log_count
        with duckdb.connect(str(self.db_path), read_only=True) as conn:
            table_count = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'user_stats'"
            ).fetchone()[0]
            if not table_count:
                self._max_user_log_count = 1.0
                return self._max_user_log_count
            value = conn.execute("SELECT MAX(rating_count) FROM user_stats").fetchone()[0]
        self._max_user_log_count = math.log1p(int(value or 1))
        return self._max_user_log_count


def _exclude_movie_ids(events: Iterable) -> set[int]:
    return {
        int(event.movie_id)
        for event in events
        if getattr(event, "movie_id", None) is not None
        and getattr(event, "event_type", "") in {"view", "like", "dislike"}
    }


def _scores_by_id(scores: list[dict]) -> dict[int, dict]:
    result = {}
    for item in scores:
        movie_id = int(item.get("movieId", item.get("movie_id")))
        enriched = dict(item)
        enriched["movieId"] = movie_id
        enriched["movie_id"] = movie_id
        result[movie_id] = enriched
    return result


def _quality_scores(scores: list[dict]) -> dict[int, float]:
    values = [
        float(item.get("comprehensive_score", 0.0))
        for item in scores
    ]
    max_value = max(values, default=1.0)
    if max_value <= 0:
        max_value = 1.0
    return {
        int(item.get("movieId", item.get("movie_id"))): float(item.get("comprehensive_score", 0.0)) / max_value
        for item in scores
    }


def _profile_stats(
    profile: dict[int, float],
    scores_by_id: dict[int, dict],
    movie_years: dict[int, int | None],
    max_user_log_count: float,
) -> dict:
    values = list(profile.values())
    avg_rating = sum(values) / len(values) if values else 3.0
    positives = {
        movie_id: rating
        for movie_id, rating in profile.items()
        if float(rating) >= 3.6
    }
    negatives = {
        movie_id: rating
        for movie_id, rating in profile.items()
        if float(rating) <= 2.0
    }
    positive_years = [
        float(movie_years.get(movie_id) or scores_by_id.get(movie_id, {}).get("release_year") or 0)
        for movie_id in positives
        if movie_years.get(movie_id) or scores_by_id.get(movie_id, {}).get("release_year")
    ]
    negative_years = [
        float(movie_years.get(movie_id) or scores_by_id.get(movie_id, {}).get("release_year") or 0)
        for movie_id in negatives
        if movie_years.get(movie_id) or scores_by_id.get(movie_id, {}).get("release_year")
    ]
    positive_avg_rating = [
        float(scores_by_id.get(movie_id, {}).get("avg_rating", 0.0))
        for movie_id in positives
    ]
    return {
        "profile": profile,
        "avg_rating": avg_rating,
        "count": len(profile),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "positive_rate": len(positives) / len(profile) if profile else 0.0,
        "profile_strength": min(len(positives) / 20.0, 1.0),
        "user_log_count": math.log1p(len(profile)) / max(max_user_log_count, 1.0),
        "positive_year_avg": sum(positive_years) / len(positive_years) if positive_years else None,
        "negative_year_avg": sum(negative_years) / len(negative_years) if negative_years else None,
        "positive_avg_rating": sum(positive_avg_rating) / len(positive_avg_rating) if positive_avg_rating else None,
    }


def _popular_route(quality_scores: dict[int, float], candidate_limit: int) -> dict[int, float]:
    limit = max(candidate_limit // 3, 50)
    return dict(sorted(quality_scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)[:limit])


def _profile_route(
    scores: list[dict],
    profile_stats: dict,
    quality_scores: dict[int, float],
    movie_years: dict[int, int | None],
    exclude_movie_ids: set[int],
) -> dict[int, float]:
    positive_year = profile_stats.get("positive_year_avg")
    positive_avg = profile_stats.get("positive_avg_rating")
    if positive_year is None and positive_avg is None:
        return {}
    result = {}
    for item in scores:
        movie_id = int(item.get("movieId", item.get("movie_id")))
        if movie_id in exclude_movie_ids:
            continue
        year = movie_years.get(movie_id) or item.get("release_year")
        year_score = _year_score(year, positive_year)
        avg_score = 1.0
        if positive_avg is not None:
            avg_score = math.exp(-abs(float(item.get("avg_rating", 0.0)) - float(positive_avg)) / 0.8)
        result[movie_id] = year_score * 0.45 + avg_score * 0.20 + quality_scores.get(movie_id, 0.0) * 0.35
    return dict(sorted(result.items(), key=lambda item: (item[1], -item[0]), reverse=True)[:800])


def _mf_proxy_route(
    scores: list[dict],
    profile_stats: dict,
    quality_scores: dict[int, float],
    movie_years: dict[int, int | None],
    exclude_movie_ids: set[int],
) -> dict[int, float]:
    profile_scores = _profile_route(scores, profile_stats, quality_scores, movie_years, exclude_movie_ids)
    result = {}
    for movie_id, score in profile_scores.items():
        result[movie_id] = score * 0.45 + quality_scores.get(movie_id, 0.0) * 0.55
    if not result:
        result = _popular_route(quality_scores, 600)
    return dict(sorted(result.items(), key=lambda item: (item[1], -item[0]), reverse=True)[:800])


def _mf_model_route(
    model: MatrixFactorizationModel,
    profile: dict[int, float],
    quality_scores: dict[int, float],
    exclude_movie_ids: set[int],
    candidate_limit: int,
) -> dict[int, float]:
    seeds = [
        (movie_id, max(float(rating) - 3.0, 0.25))
        for movie_id, rating in profile.items()
        if float(rating) >= 3.6 and movie_id in model.movie_to_index
    ]
    if not seeds or not len(model.index_to_movie) or not model.movie_factors.size:
        return {}

    vector = np.zeros(model.movie_factors.shape[1], dtype=np.float32)
    weight_sum = 0.0
    for movie_id, weight in seeds:
        vector += model.movie_factors[model.movie_to_index[movie_id]] * float(weight)
        weight_sum += float(weight)
    if weight_sum <= 0:
        return {}
    vector /= weight_sum

    movie_ids = np.array(model.index_to_movie, dtype=np.int64)
    scores = model.movie_factors @ vector
    scores = scores + model.movie_bias
    if quality_scores:
        quality = np.array([quality_scores.get(int(movie_id), 0.0) for movie_id in movie_ids], dtype=np.float32)
        scores = scores * 0.75 + quality * 0.25

    if exclude_movie_ids:
        mask = ~np.isin(movie_ids, list(exclude_movie_ids), assume_unique=False)
        movie_ids = movie_ids[mask]
        scores = scores[mask]
    if not len(movie_ids):
        return {}

    take = min(max(candidate_limit, 100), len(movie_ids))
    positions = np.argpartition(-scores, take - 1)[:take]
    positions = positions[np.argsort(-scores[positions])]
    return {
        int(movie_ids[position]): float(scores[position])
        for position in positions
    }


def _year_route(
    scores: list[dict],
    profile_stats: dict,
    quality_scores: dict[int, float],
    movie_years: dict[int, int | None],
    exclude_movie_ids: set[int],
) -> dict[int, float]:
    positive_year = profile_stats.get("positive_year_avg")
    if positive_year is None:
        return {}
    result = {}
    for item in scores:
        movie_id = int(item.get("movieId", item.get("movie_id")))
        if movie_id in exclude_movie_ids:
            continue
        year = movie_years.get(movie_id) or item.get("release_year")
        result[movie_id] = _year_score(year, positive_year) * 0.7 + quality_scores.get(movie_id, 0.0) * 0.3
    return dict(sorted(result.items(), key=lambda item: (item[1], -item[0]), reverse=True)[:800])


def _year_score(year: object, target_year: object) -> float:
    if year is None or target_year is None:
        return 0.0
    try:
        return math.exp(-abs(float(year) - float(target_year)) / 18.0)
    except (TypeError, ValueError):
        return 0.0


def _combine_routes(
    routes: dict[str, dict[int, float]],
    artifact: OnlineRerankArtifact,
) -> tuple[dict[int, float], dict[int, set[str]], dict[str, dict[int, float]]]:
    combined: defaultdict[int, float] = defaultdict(float)
    movie_sources: defaultdict[int, set[str]] = defaultdict(set)
    active_routes: dict[str, dict[int, float]] = {}
    for source in SOURCE_NAMES:
        values = routes.get(source, {})
        if not values:
            active_routes[source] = {}
            continue
        floor, cap, base_weight = artifact.source_priors.get(source, DEFAULT_SOURCE_PRIORS[source])
        quota = max(int(artifact.candidate_limit * cap), 1)
        ranked = sorted(values.items(), key=lambda item: (item[1], -item[0]), reverse=True)[:quota]
        active_routes[source] = dict(ranked)
        weight = base_weight * (0.35 + max(floor, 0.0))
        for rank, (movie_id, _score) in enumerate(ranked, start=1):
            combined[int(movie_id)] += weight / (max(artifact.rrf_k, 1.0) + rank)
            movie_sources[int(movie_id)].add(source)
    return dict(combined), {movie_id: set(values) for movie_id, values in movie_sources.items()}, active_routes


def _select_candidate_ids(
    combined: dict[int, float],
    movie_sources: dict[int, set[str]],
    quality_scores: dict[int, float],
    exclude_movie_ids: set[int],
    candidate_limit: int,
) -> list[int]:
    ranked = [
        movie_id
        for movie_id in sorted(
            combined,
            key=lambda value: (
                combined[value],
                len(movie_sources.get(value, ())),
                quality_scores.get(value, 0.0),
                -value,
            ),
            reverse=True,
        )
        if movie_id not in exclude_movie_ids
    ]
    return ranked[:candidate_limit]


def _route_features(
    candidate_ids: list[int],
    combined: dict[int, float],
    movie_sources: dict[int, set[str]],
    active_routes: dict[str, dict[int, float]],
) -> dict[int, dict[str, float]]:
    max_combined = max((combined.get(movie_id, 0.0) for movie_id in candidate_ids), default=1.0)
    if max_combined <= 0:
        max_combined = 1.0
    normalized_routes = {
        source: _normalize_score_dict(values)
        for source, values in active_routes.items()
    }
    result = {}
    for movie_id in candidate_ids:
        result[movie_id] = {
            "recall_surfaced": 1.0 if movie_id in combined else 0.0,
            "recall_route_score": float(combined.get(movie_id, 0.0)) / max_combined,
            "recall_source_count": min(len(movie_sources.get(movie_id, ())) / float(len(SOURCE_NAMES)), 1.0),
            "recall_popular_quality": float(normalized_routes["popular_quality"].get(movie_id, 0.0)),
            "recall_mf_user_top": float(normalized_routes["mf_user_top"].get(movie_id, 0.0)),
            "recall_profile_centroid": float(normalized_routes["profile_centroid"].get(movie_id, 0.0)),
            "recall_item_item_cf": float(normalized_routes["item_item_cf"].get(movie_id, 0.0)),
            "recall_user_user_cf": float(normalized_routes["user_user_cf"].get(movie_id, 0.0)),
            "recall_year_affinity": float(normalized_routes["year_affinity"].get(movie_id, 0.0)),
        }
    return result


def _feature_matrix(
    candidate_ids: list[int],
    *,
    scores_by_id: dict[int, dict],
    quality_scores: dict[int, float],
    movie_years: dict[int, int | None],
    profile_stats: dict,
    route_features: dict[int, dict[str, float]],
    collaborative_scores: dict[int, float],
    mf_proxy_scores: dict[int, float],
) -> np.ndarray:
    rows = np.zeros((len(candidate_ids), len(FEATURE_NAMES)), dtype=np.float32)
    max_movie_log_count = max(
        (math.log1p(float(item.get("rating_count", 0.0))) for item in scores_by_id.values()),
        default=1.0,
    )
    known_years = [
        int(year)
        for year in (
            movie_years.get(movie_id) or item.get("release_year")
            for movie_id, item in scores_by_id.items()
        )
        if year
    ]
    min_year = min(known_years, default=1900)
    year_span = max(max(known_years, default=2025) - min_year, 1)
    for row_index, movie_id in enumerate(candidate_ids):
        item = scores_by_id.get(movie_id, {})
        year = movie_years.get(movie_id) or item.get("release_year")
        route = route_features.get(movie_id, {})
        values = {
            "mf_score": mf_proxy_scores.get(movie_id, quality_scores.get(movie_id, 0.0)),
            "collaborative_score": collaborative_scores.get(movie_id, 0.0),
            "quality_score": quality_scores.get(movie_id, 0.0),
            "movie_avg_rating": float(item.get("avg_rating", 0.0)) / 5.0,
            "movie_log_count": math.log1p(float(item.get("rating_count", 0.0))) / max_movie_log_count,
            "user_avg_rating": float(profile_stats["avg_rating"]) / 5.0,
            "user_log_count": float(profile_stats["user_log_count"]),
            "profile_similarity": route.get("recall_profile_centroid", 0.0),
            "negative_profile_similarity": _year_score(year, profile_stats.get("negative_year_avg")),
            "signed_profile_similarity": 0.5 + 0.5 * (route.get("recall_profile_centroid", 0.0) - _year_score(year, profile_stats.get("negative_year_avg"))),
            "item_item_score": route.get("recall_item_item_cf", 0.0),
            "profile_strength": float(profile_stats["profile_strength"]),
            "user_positive_rate": float(profile_stats["positive_rate"]),
            "movie_year_norm": ((float(year) - min_year) / year_span) if year else 0.0,
            "movie_has_year": 1.0 if year else 0.0,
            "year_affinity": _year_score(year, profile_stats.get("positive_year_avg")),
            **route,
        }
        for feature, value in values.items():
            if feature in FEATURE_NAMES:
                rows[row_index, FEATURE_NAMES.index(feature)] = float(np.clip(value, 0.0, 1.0))
    return rows


def _linear_scores(features: np.ndarray, artifact: OnlineRerankArtifact) -> np.ndarray:
    if len(features) == 0:
        return np.empty(0, dtype=np.float32)
    normalized = (features - artifact.mean) / artifact.std
    return (normalized @ artifact.weights).astype(np.float32)


def _normalize_score_dict(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_value = min(values)
    max_value = max(values)
    if max_value <= min_value:
        return {int(key): 1.0 for key in scores}
    return {
        int(key): (float(value) - min_value) / (max_value - min_value)
        for key, value in scores.items()
    }


def _dominant_bucket(sources: set[str]) -> str:
    if "user_user_cf" in sources:
        return "collaborative"
    if "item_item_cf" in sources:
        return "item_item"
    if "profile_centroid" in sources or "mf_user_top" in sources:
        return "profile"
    if "year_affinity" in sources:
        return "year"
    return "top"


def _recommendation_reason(bucket: str, route: dict[str, float]) -> str:
    if bucket == "collaborative":
        return "Stacked reranker promoted this from similar-user behavior."
    if bucket == "item_item":
        return "Stacked reranker promoted this from movies co-liked with your seeds."
    if bucket == "profile":
        return "Stacked reranker matched this to your session profile."
    if bucket == "year":
        return "Stacked reranker matched this to your release-year taste."
    if route.get("recall_source_count", 0.0) > 0.2:
        return "Stacked reranker combined several recall routes for this title."
    return "High-scoring Netflix title while your taste profile warms up."


def _top_backfill(scores: list[dict], n: int, exclude_movie_ids: set[int]) -> list[dict]:
    rows = []
    ranked = sorted(
        scores,
        key=lambda item: (
            float(item.get("comprehensive_score", 0.0)),
            float(item.get("rating_count", 0.0)),
            -int(item.get("movieId", item.get("movie_id"))),
        ),
        reverse=True,
    )
    for item in ranked:
        movie_id = int(item.get("movieId", item.get("movie_id")))
        if movie_id in exclude_movie_ids:
            continue
        enriched = dict(item)
        enriched["movieId"] = movie_id
        enriched["movie_id"] = movie_id
        enriched["personal_score"] = float(enriched.get("comprehensive_score", 0.0))
        enriched["recommendation_bucket"] = "top"
        enriched["recommendation_reason"] = "High-scoring Netflix title while your taste profile warms up."
        rows.append(enriched)
        if len(rows) >= n:
            break
    return rows
