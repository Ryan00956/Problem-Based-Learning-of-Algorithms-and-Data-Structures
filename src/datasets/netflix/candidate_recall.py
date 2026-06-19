from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from src.datasets.netflix.evaluation import (
    EvaluationConfig,
    _normalize_score_dict,
    _score_user_user_candidates,
)
from src.datasets.netflix.matrix_factorization import MatrixFactorizationModel


SOURCE_NAMES = (
    "popular_quality",
    "mf_user_top",
    "profile_centroid",
    "item_item_cf",
    "user_user_cf",
    "year_affinity",
)
DEFAULT_RRF_K = 60.0
LEGACY_STRATEGY = "legacy"
WEIGHTED_RRF_STRATEGY = "weighted_rrf"
SOURCE_PRIORS = {
    "popular_quality": {"floor": 0.25, "cap": 0.35, "base_weight": 1.25},
    "mf_user_top": {"floor": 0.10, "cap": 0.30, "base_weight": 1.00},
    "profile_centroid": {"floor": 0.08, "cap": 0.24, "base_weight": 0.90},
    "item_item_cf": {"floor": 0.08, "cap": 0.22, "base_weight": 1.05},
    "user_user_cf": {"floor": 0.05, "cap": 0.16, "base_weight": 0.85},
    "year_affinity": {"floor": 0.08, "cap": 0.25, "base_weight": 1.00},
}


@dataclass(frozen=True)
class RecallResult:
    movie_ids: np.ndarray
    source_rows: list[dict]
    movie_features: dict[int, dict[str, float]]


def build_multi_route_candidate_pool(
    split: dict,
    config: EvaluationConfig,
    model: MatrixFactorizationModel,
    *,
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
) -> RecallResult:
    train_movie_ids = np.array(model.index_to_movie, dtype=np.int32)
    known_movie_ids = set(model.movie_to_index)
    relevant_movie_ids = {
        movie_id
        for values in split["relevant_by_user"].values()
        for movie_id in values
        if movie_id in known_movie_ids
    }
    if not len(train_movie_ids):
        return RecallResult(np.array(sorted(relevant_movie_ids), dtype=np.int32), [], {})

    source_names = tuple(feature_stats.get("candidate_source_names") or SOURCE_NAMES)
    source_scores: dict[str, defaultdict[int, float]] = {source: defaultdict(float) for source in source_names}
    user_ids = sorted(split["train_history"])
    per_user_limit = _per_user_limit(config, user_ids)

    neighbor_cache = feature_stats.setdefault("user_neighbor_cache", {})
    raw_score_cache = feature_stats.setdefault("collaborative_raw_cache", {})
    if "popular_quality" in source_scores:
        _add_popular_quality(source_scores["popular_quality"], quality_scores, config)
    if "mf_user_top" in source_scores:
        _add_mf_user_top(source_scores["mf_user_top"], split, model, train_movie_ids, user_ids, per_user_limit)
    if "profile_centroid" in source_scores:
        _add_profile_centroid(source_scores["profile_centroid"], split, model, train_movie_ids, feature_stats, user_ids, per_user_limit)
    if "item_item_cf" in source_scores:
        _add_item_item_cf(source_scores["item_item_cf"], split, feature_stats, user_ids, per_user_limit)
    if "user_user_cf" in source_scores:
        _add_user_user_cf(source_scores["user_user_cf"], split, indexes, train_movie_ids, user_ids, per_user_limit, neighbor_cache=neighbor_cache, raw_score_cache=raw_score_cache)
    if "year_affinity" in source_scores:
        _add_year_affinity(source_scores["year_affinity"], split, quality_scores, train_movie_ids, feature_stats, user_ids, per_user_limit)

    raw_source_metrics = _source_relevance_metrics(source_scores, relevant_movie_ids)
    strategy = config.candidate_recall_strategy
    if strategy == WEIGHTED_RRF_STRATEGY:
        source_priors = _source_priors(config)
        quotas = _source_quotas(raw_source_metrics, config, source_priors, source_names)
        weights = _source_weights(raw_source_metrics, source_priors, source_names)
        combined_scores, movie_sources, active_source_scores = _combine_weighted_rrf(
            source_scores,
            quotas,
            weights,
            source_names,
            rrf_k=config.candidate_recall_rrf_k,
        )
    elif strategy == LEGACY_STRATEGY:
        combined_scores, movie_sources, active_source_scores = _combine_legacy_minmax(source_scores, source_names)
        quotas = {source: len(active_source_scores[source]) for source in source_names}
        weights = {source: 1.0 for source in source_names}
    else:
        raise ValueError(f"unknown candidate recall strategy: {strategy}")

    selected_before_backfill = _select_candidates(
        combined_scores,
        movie_sources,
        quality_scores,
        config,
        relevant_movie_ids,
    )
    selected = list(selected_before_backfill)
    seen = set(selected)
    for movie_id in sorted(relevant_movie_ids):
        if movie_id not in seen:
            selected.append(movie_id)
            seen.add(movie_id)

    selected_set = set(selected)
    selected_before_set = set(selected_before_backfill)
    source_rows = []
    for source, scores in source_scores.items():
        source_movies = set(scores)
        active_source_movies = set(active_source_scores[source])
        selected_source_movies = active_source_movies & selected_before_set
        relevant_hits = selected_source_movies & relevant_movie_ids
        marginal_hits = {
            movie_id
            for movie_id in relevant_hits
            if movie_sources.get(movie_id, set()) == {source}
        }
        raw_metrics = raw_source_metrics[source]
        source_rows.append(
            {
                "source": source,
                "strategy": strategy,
                "quota": quotas[source],
                "rrf_weight": round(weights[source], 6),
                "raw_movies": len(source_movies),
                "active_movies": len(active_source_movies),
                "selected_movies": len(active_source_movies & selected_set),
                "selected_before_backfill_movies": len(selected_source_movies),
                "relevant_hits_before_backfill": len(relevant_hits),
                "route_precision": round(len(relevant_hits) / len(selected_source_movies), 6)
                if selected_source_movies
                else 0.0,
                "route_recall": round(len(relevant_hits) / len(relevant_movie_ids), 6)
                if relevant_movie_ids
                else 0.0,
                "unique_relevant_hits": len(marginal_hits),
                "route_marginal_recall": round(len(marginal_hits) / len(relevant_movie_ids), 6)
                if relevant_movie_ids
                else 0.0,
                "raw_precision": raw_metrics["precision"],
                "raw_recall": raw_metrics["recall"],
                "raw_marginal_recall": raw_metrics["marginal_recall"],
            }
        )
    backfilled_relevant = relevant_movie_ids - selected_before_set
    source_rows.append(
        {
            "source": "relevant_backfill",
            "strategy": strategy,
            "raw_movies": len(relevant_movie_ids),
            "selected_movies": len(relevant_movie_ids & selected_set),
            "selected_before_backfill_movies": len(relevant_movie_ids & selected_before_set),
            "relevant_hits_before_backfill": len(relevant_movie_ids & selected_before_set),
            "route_precision": 1.0 if backfilled_relevant else 0.0,
            "route_recall": round(len(backfilled_relevant) / len(relevant_movie_ids), 6)
            if relevant_movie_ids
            else 0.0,
            "unique_relevant_hits": len(backfilled_relevant),
            "route_marginal_recall": round(len(backfilled_relevant) / len(relevant_movie_ids), 6)
            if relevant_movie_ids
            else 0.0,
        }
    )
    all_route_hits = relevant_movie_ids & selected_before_set
    source_rows.append(
        {
            "source": "all_routes",
            "strategy": strategy,
            "raw_movies": len(combined_scores),
            "selected_movies": len(selected),
            "selected_before_backfill_movies": len(selected_before_set),
            "relevant_movies": len(relevant_movie_ids),
            "relevant_hits_before_backfill": len(all_route_hits),
            "route_relevant_recall": round(
                len(all_route_hits) / len(relevant_movie_ids),
                6,
            )
            if relevant_movie_ids
            else 0.0,
            "route_precision": round(len(all_route_hits) / len(selected_before_set), 6)
            if selected_before_set
            else 0.0,
            "route_recall": round(len(all_route_hits) / len(relevant_movie_ids), 6)
            if relevant_movie_ids
            else 0.0,
        }
    )
    movie_features = _movie_route_features(selected, active_source_scores, combined_scores, movie_sources, source_names)
    return RecallResult(np.array(selected, dtype=np.int32), source_rows, movie_features)


def _per_user_limit(config: EvaluationConfig, user_ids: list[int]) -> int:
    if not user_ids:
        return config.top_k
    if not config.candidate_limit:
        return max(config.top_k * 8, 50)
    return min(max(config.top_k * 8, 30), max(config.candidate_limit // max(len(user_ids) // 10, 1), config.top_k))


def _add_popular_quality(
    scores: defaultdict[int, float],
    quality_scores: dict[int, float],
    config: EvaluationConfig,
) -> None:
    quota = max(config.candidate_limit // 3, config.top_k * 20) if config.candidate_limit else len(quality_scores)
    ranked = sorted(quality_scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)
    for rank, (movie_id, score) in enumerate(ranked[:quota], start=1):
        scores[int(movie_id)] += float(score) + _rank_bonus(rank, quota)


def _add_mf_user_top(
    scores: defaultdict[int, float],
    split: dict,
    model: MatrixFactorizationModel,
    train_movie_ids: np.ndarray,
    user_ids: list[int],
    per_user_limit: int,
) -> None:
    if not len(train_movie_ids):
        return
    for user_id in user_ids:
        exclude = split["train_history"].get(user_id, set())
        eligible_mask = ~np.isin(train_movie_ids, list(exclude), assume_unique=False)
        eligible_indices = np.flatnonzero(eligible_mask)
        if not len(eligible_indices):
            continue
        eligible_movies = train_movie_ids[eligible_indices]
        raw_scores = model.score_known_user(user_id, eligible_movies)
        for movie_id, score, rank in _top_items(eligible_movies, raw_scores, per_user_limit):
            scores[movie_id] += score + _rank_bonus(rank, per_user_limit)


def _add_profile_centroid(
    scores: defaultdict[int, float],
    split: dict,
    model: MatrixFactorizationModel,
    train_movie_ids: np.ndarray,
    feature_stats: dict,
    user_ids: list[int],
    per_user_limit: int,
) -> None:
    centroids = feature_stats.get("profile_centroids", {})
    norms = feature_stats.get("profile_norms", {})
    if not centroids or not model.movie_factors.size:
        return
    movie_indices = np.array([model.movie_to_index[int(movie_id)] for movie_id in train_movie_ids], dtype=np.int32)
    movie_vectors = model.movie_factors[movie_indices]
    movie_norms = np.linalg.norm(movie_vectors, axis=1)
    for user_id in user_ids:
        centroid = centroids.get(user_id)
        centroid_norm = float(norms.get(user_id, 0.0))
        if centroid is None or centroid_norm <= 0:
            continue
        exclude = split["train_history"].get(user_id, set())
        similarities = movie_vectors @ centroid
        denominators = np.maximum(movie_norms * centroid_norm, 1e-6)
        similarities = similarities / denominators
        eligible_mask = ~np.isin(train_movie_ids, list(exclude), assume_unique=False)
        eligible_movies = train_movie_ids[eligible_mask]
        eligible_scores = similarities[eligible_mask]
        if not len(eligible_movies):
            continue
        for movie_id, score, rank in _top_items(eligible_movies, eligible_scores, per_user_limit):
            scores[movie_id] += ((score + 1.0) / 2.0) + _rank_bonus(rank, per_user_limit)


def _add_item_item_cf(
    scores: defaultdict[int, float],
    split: dict,
    feature_stats: dict,
    user_ids: list[int],
    per_user_limit: int,
) -> None:
    neighbors = feature_stats.get("item_item_neighbors", {})
    if not neighbors:
        return

    for user_id in user_ids:
        exclude = split["train_history"].get(user_id, set())
        active_profile = feature_stats.get("user_train_ratings", {}).get(user_id, {})
        raw = _score_item_item_candidates(active_profile, neighbors, exclude)
        if not raw:
            continue
        normalized = _normalize_score_dict(raw)
        ranked = sorted(normalized.items(), key=lambda item: (item[1], -item[0]), reverse=True)[:per_user_limit]
        for rank, (movie_id, score) in enumerate(ranked, start=1):
            scores[int(movie_id)] += float(score) + _rank_bonus(rank, per_user_limit)


def _score_item_item_candidates(
    active_profile: dict[int, float],
    neighbors: dict[int, list[tuple[int, float]]],
    exclude_movie_ids: set[int],
) -> dict[int, float]:
    scores: defaultdict[int, float] = defaultdict(float)
    seed_count = 0
    for seed_movie_id, rating in active_profile.items():
        if float(rating) < 4.0:
            continue
        seed_count += 1
        seed_weight = max(float(rating) - 3.0, 0.25)
        for movie_id, similarity in neighbors.get(int(seed_movie_id), ()):
            movie_id = int(movie_id)
            if movie_id in exclude_movie_ids:
                continue
            scores[movie_id] += seed_weight * float(similarity)
    if seed_count <= 1:
        return dict(scores)
    normalizer = seed_count ** 0.5
    return {movie_id: score / normalizer for movie_id, score in scores.items()}


def _add_user_user_cf(
    scores: defaultdict[int, float],
    split: dict,
    indexes,
    train_movie_ids: np.ndarray,
    user_ids: list[int],
    per_user_limit: int,
    *,
    neighbor_cache: dict[int, list[dict]] | None = None,
    raw_score_cache: dict[int, dict[int, float]] | None = None,
) -> None:
    user_ratings = indexes[0]
    train_movie_id_set = {int(movie_id) for movie_id in train_movie_ids}
    for user_id in user_ids:
        raw = _score_user_user_candidates(
            user_id,
            user_ratings.get(user_id, {}),
            indexes=indexes,
            candidate_movie_ids=train_movie_ids,
            exclude_movie_ids=split["train_history"].get(user_id, set()),
            candidate_movie_id_set=train_movie_id_set,
            neighbor_cache=neighbor_cache,
            raw_score_cache=raw_score_cache,
        )
        if not raw:
            continue
        normalized = _normalize_score_dict(raw)
        ranked = sorted(normalized.items(), key=lambda item: (item[1], -item[0]), reverse=True)[:per_user_limit]
        for rank, (movie_id, score) in enumerate(ranked, start=1):
            scores[int(movie_id)] += float(score) + _rank_bonus(rank, per_user_limit)


def _add_year_affinity(
    scores: defaultdict[int, float],
    split: dict,
    quality_scores: dict[int, float],
    train_movie_ids: np.ndarray,
    feature_stats: dict,
    user_ids: list[int],
    per_user_limit: int,
) -> None:
    movie_years = feature_stats.get("movie_year", {})
    user_years = feature_stats.get("user_positive_year_avg", {})
    if not movie_years or not user_years:
        return
    year_values = np.array([float(movie_years.get(int(movie_id), 0.0) or 0.0) for movie_id in train_movie_ids], dtype=np.float32)
    has_year = np.array([movie_years.get(int(movie_id)) is not None for movie_id in train_movie_ids], dtype=bool)
    quality_values = np.array([float(quality_scores.get(int(movie_id), 0.0)) for movie_id in train_movie_ids], dtype=np.float32)
    for user_id in user_ids:
        user_year = user_years.get(user_id)
        if user_year is None:
            continue
        exclude = split["train_history"].get(user_id, set())
        eligible_mask = has_year & ~np.isin(train_movie_ids, list(exclude), assume_unique=False)
        eligible_movies = train_movie_ids[eligible_mask]
        if not len(eligible_movies):
            continue
        affinity = np.exp(-np.abs(year_values[eligible_mask] - float(user_year)) / 18.0)
        eligible_scores = affinity * 0.7 + quality_values[eligible_mask] * 0.3
        for movie_id, score, rank in _top_items(eligible_movies, eligible_scores, per_user_limit):
            scores[movie_id] += score + _rank_bonus(rank, per_user_limit)


def _source_relevance_metrics(
    source_scores: dict[str, defaultdict[int, float]],
    relevant_movie_ids: set[int],
) -> dict[str, dict[str, float]]:
    source_movie_sets = {source: set(scores) for source, scores in source_scores.items()}
    result = {}
    for source, source_movies in source_movie_sets.items():
        hits = source_movies & relevant_movie_ids
        unique_hits = {
            movie_id
            for movie_id in hits
            if sum(movie_id in values for values in source_movie_sets.values()) == 1
        }
        result[source] = {
            "precision": round(len(hits) / len(source_movies), 6) if source_movies else 0.0,
            "recall": round(len(hits) / len(relevant_movie_ids), 6) if relevant_movie_ids else 0.0,
            "marginal_recall": round(len(unique_hits) / len(relevant_movie_ids), 6) if relevant_movie_ids else 0.0,
        }
    return result


def _source_priors(config: EvaluationConfig) -> dict[str, dict[str, float]]:
    priors = {source: dict(values) for source, values in SOURCE_PRIORS.items()}
    for source, floor, cap, base_weight in config.candidate_recall_priors:
        if source not in priors:
            raise ValueError(f"unknown recall source prior: {source}")
        if floor < 0 or cap < 0:
            raise ValueError("recall source floor and cap must be non-negative")
        if cap < floor:
            raise ValueError("recall source cap must be greater than or equal to floor")
        if base_weight <= 0:
            raise ValueError("recall source base weight must be positive")
        priors[source] = {
            "floor": float(floor),
            "cap": float(cap),
            "base_weight": float(base_weight),
        }
    return priors


def _source_quotas(
    source_metrics: dict[str, dict[str, float]],
    config: EvaluationConfig,
    source_priors: dict[str, dict[str, float]],
    source_names: tuple[str, ...],
) -> dict[str, int]:
    candidate_limit = int(config.candidate_limit or 0)
    if candidate_limit <= 0:
        return {source: 10**9 for source in source_names}

    floors = {
        source: int(candidate_limit * source_priors[source]["floor"])
        for source in source_names
    }
    caps = {
        source: max(floors[source], int(candidate_limit * source_priors[source]["cap"]))
        for source in source_names
    }
    quota = dict(floors)
    remaining = max(candidate_limit - sum(quota.values()), 0)
    needs = {}
    for source in source_names:
        metrics = source_metrics[source]
        needs[source] = (
            0.10
            + 0.50 * metrics["precision"]
            + 0.30 * metrics["recall"]
            + 4.00 * metrics["marginal_recall"]
        )

    while remaining > 0:
        candidates = [source for source in source_names if quota[source] < caps[source]]
        if not candidates:
            break
        total_need = sum(needs[source] for source in candidates)
        if total_need <= 0:
            break
        changed = False
        for source in candidates:
            room = caps[source] - quota[source]
            addition = max(1, round(remaining * (needs[source] / total_need)))
            addition = min(addition, room, remaining)
            if addition <= 0:
                continue
            quota[source] += addition
            remaining -= addition
            changed = True
            if remaining <= 0:
                break
        if not changed:
            break
    return quota


def _source_weights(
    source_metrics: dict[str, dict[str, float]],
    source_priors: dict[str, dict[str, float]],
    source_names: tuple[str, ...],
) -> dict[str, float]:
    weights = {}
    for source in source_names:
        metrics = source_metrics[source]
        prior = source_priors[source]
        weights[source] = prior["base_weight"] * (
            0.35
            + metrics["precision"]
            + 0.50 * metrics["recall"]
            + 3.00 * metrics["marginal_recall"]
        )
    return weights


def _combine_weighted_rrf(
    source_scores: dict[str, defaultdict[int, float]],
    quotas: dict[str, int],
    weights: dict[str, float],
    source_names: tuple[str, ...],
    *,
    rrf_k: float = DEFAULT_RRF_K,
) -> tuple[defaultdict[int, float], defaultdict[int, set[str]], dict[str, dict[int, float]]]:
    combined_scores: defaultdict[int, float] = defaultdict(float)
    movie_sources: defaultdict[int, set[str]] = defaultdict(set)
    active_source_scores: dict[str, dict[int, float]] = {}
    for source in source_names:
        ranked = sorted(source_scores[source].items(), key=lambda item: (item[1], -item[0]), reverse=True)
        ranked = ranked[: quotas[source]]
        active_source_scores[source] = {}
        for rank, (movie_id, raw_score) in enumerate(ranked, start=1):
            movie_id = int(movie_id)
            active_source_scores[source][movie_id] = float(raw_score)
            combined_scores[movie_id] += weights[source] / (max(float(rrf_k), 1.0) + rank)
            movie_sources[movie_id].add(source)
    return combined_scores, movie_sources, active_source_scores


def _combine_legacy_minmax(
    source_scores: dict[str, defaultdict[int, float]],
    source_names: tuple[str, ...],
) -> tuple[defaultdict[int, float], defaultdict[int, set[str]], dict[str, dict[int, float]]]:
    combined_scores: defaultdict[int, float] = defaultdict(float)
    movie_sources: defaultdict[int, set[str]] = defaultdict(set)
    active_source_scores: dict[str, dict[int, float]] = {}
    for source in source_names:
        normalized = _normalize_score_dict(dict(source_scores[source]))
        active_source_scores[source] = normalized
        for movie_id, score in normalized.items():
            combined_scores[int(movie_id)] += float(score)
            movie_sources[int(movie_id)].add(source)
    return combined_scores, movie_sources, active_source_scores


def _select_candidates(
    combined_scores: dict[int, float],
    movie_sources: dict[int, set[str]],
    quality_scores: dict[int, float],
    config: EvaluationConfig,
    relevant_movie_ids: set[int],
) -> list[int]:
    ranked = sorted(
        combined_scores,
        key=lambda movie_id: (
            combined_scores[movie_id],
            len(movie_sources[movie_id]),
            quality_scores.get(movie_id, 0.0),
            -movie_id,
        ),
        reverse=True,
    )
    if not config.candidate_limit:
        return ranked
    return ranked[: config.candidate_limit]


def _top_items(movie_ids: np.ndarray, scores: np.ndarray, limit: int) -> list[tuple[int, float, int]]:
    if not len(movie_ids) or limit <= 0:
        return []
    take = min(limit, len(movie_ids))
    top_positions = np.argpartition(-scores, take - 1)[:take]
    ranked_positions = top_positions[np.argsort(-scores[top_positions])]
    return [
        (int(movie_ids[position]), float(scores[position]), rank)
        for rank, position in enumerate(ranked_positions, start=1)
    ]


def _rank_bonus(rank: int, limit: int) -> float:
    if limit <= 1:
        return 0.0
    return 1.0 - ((rank - 1) / (limit - 1))


def _movie_route_features(
    selected_movie_ids: list[int],
    source_scores: dict[str, dict[int, float]],
    combined_scores: dict[int, float],
    movie_sources: dict[int, set[str]],
    source_names: tuple[str, ...],
) -> dict[int, dict[str, float]]:
    max_combined = max(combined_scores.values(), default=1.0)
    if max_combined <= 0:
        max_combined = 1.0
    normalized_sources = {
        source: _normalize_score_dict(dict(scores))
        for source, scores in source_scores.items()
    }
    result = {}
    for movie_id in selected_movie_ids:
        result[int(movie_id)] = {
            "recall_surfaced": 1.0 if movie_id in combined_scores else 0.0,
            "recall_route_score": float(combined_scores.get(movie_id, 0.0)) / max_combined,
            "recall_source_count": min(len(movie_sources.get(movie_id, ())) / float(len(source_names)), 1.0),
            "recall_popular_quality": float(normalized_sources["popular_quality"].get(movie_id, 0.0)),
            "recall_mf_user_top": float(normalized_sources["mf_user_top"].get(movie_id, 0.0)),
            "recall_profile_centroid": float(normalized_sources["profile_centroid"].get(movie_id, 0.0)),
            "recall_item_item_cf": float(normalized_sources.get("item_item_cf", {}).get(movie_id, 0.0)),
            "recall_user_user_cf": float(normalized_sources["user_user_cf"].get(movie_id, 0.0)),
            "recall_year_affinity": float(normalized_sources["year_affinity"].get(movie_id, 0.0)),
        }
    return result
