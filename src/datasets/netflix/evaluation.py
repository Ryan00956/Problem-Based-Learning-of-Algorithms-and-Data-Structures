from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np

from src.datasets.netflix.import_duckdb import DEFAULT_DB_PATH
from src.datasets.netflix.matrix_factorization import (
    MatrixFactorizationConfig,
    MatrixFactorizationModel,
    TrainingRating,
)


NEIGHBOR_LIMIT = 32
MIN_SHARED_MOVIES = 2
SUPPORT_LIMIT = 2
OVERLAP_SMOOTHING = 5.0
HYBRID_MF_WEIGHT = 0.35518699276413085
HYBRID_COLLABORATIVE_WEIGHT = 0.08412705524518764
HYBRID_QUALITY_WEIGHT = 0.5606859519906815
MOVIE_RATING_PRIOR = 25.0


@dataclass(frozen=True)
class EvaluationConfig:
    max_users: int = 160
    min_ratings_per_user: int = 12
    test_ratio: float = 0.2
    relevant_threshold: float = 4.0
    top_k: int = 10
    candidate_limit: int = 1200
    candidate_recall_strategy: str = "legacy"
    candidate_recall_rrf_k: float = 60.0
    candidate_recall_priors: tuple[tuple[str, float, float, float], ...] = ()


@dataclass(frozen=True)
class RatingRow:
    user_id: int
    movie_id: int
    rating: float
    rating_date: object


def run_matrix_factorization_evaluation(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    model_config: MatrixFactorizationConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
) -> dict:
    model_config = model_config or MatrixFactorizationConfig()
    evaluation_config = evaluation_config or EvaluationConfig()
    rows = load_time_split_source(db_path, evaluation_config)
    split = split_by_user_time(rows, evaluation_config)

    model = MatrixFactorizationModel(model_config)
    training_curve = model.fit(split["train"])
    metrics = evaluate_model(model, split, evaluation_config)
    return {
        "metrics": metrics,
        "training_curve": training_curve,
        "config": {
            **model_config.__dict__,
            **evaluation_config.__dict__,
        },
    }


def compare_recommenders(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    model_config: MatrixFactorizationConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
) -> dict:
    model_config = model_config or MatrixFactorizationConfig()
    evaluation_config = evaluation_config or EvaluationConfig()
    rows = load_time_split_source(db_path, evaluation_config)
    split = split_by_user_time(rows, evaluation_config)

    mf_model = MatrixFactorizationModel(model_config)
    training_curve = mf_model.fit(split["train"])
    candidate_movie_ids = candidate_movie_ids_from_split(
        split,
        evaluation_config,
        known_movie_ids=set(mf_model.movie_to_index),
    )
    quality_scores = movie_quality_scores(split)
    collaborative_indexes = _build_collaborative_indexes(split["train"])
    popular_metrics = evaluate_popular_baseline(split, evaluation_config, candidate_movie_ids, quality_scores)
    bias_metrics = evaluate_bias_baseline(split, evaluation_config, candidate_movie_ids)
    collaborative_metrics = evaluate_user_user_collaborative(
        split,
        evaluation_config,
        candidate_movie_ids=candidate_movie_ids,
        indexes=collaborative_indexes,
    )
    mf_metrics = evaluate_model(mf_model, split, evaluation_config, candidate_movie_ids=candidate_movie_ids)
    hybrid_metrics = evaluate_hybrid_ranker(
        mf_model,
        split,
        evaluation_config,
        candidate_movie_ids=candidate_movie_ids,
        quality_scores=quality_scores,
        indexes=collaborative_indexes,
    )
    return {
        "metrics": [popular_metrics, bias_metrics, collaborative_metrics, mf_metrics, hybrid_metrics],
        "training_curve": training_curve,
        "config": {
            **model_config.__dict__,
            **evaluation_config.__dict__,
        },
    }


def grid_search_matrix_factorization(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    model_configs: list[MatrixFactorizationConfig],
    evaluation_config: EvaluationConfig | None = None,
) -> dict:
    if not model_configs:
        raise ValueError("model_configs must not be empty")
    evaluation_config = evaluation_config or EvaluationConfig()
    rows = load_time_split_source(db_path, evaluation_config)
    split = split_by_user_time(rows, evaluation_config)

    results = []
    curves = []
    for index, model_config in enumerate(model_configs, start=1):
        model = MatrixFactorizationModel(model_config)
        training_curve = model.fit(split["train"])
        metrics = evaluate_model(model, split, evaluation_config)
        config_values = {
            "trial": index,
            "factors": model_config.factors,
            "epochs": model_config.epochs,
            "learning_rate": model_config.learning_rate,
            "regularization": model_config.regularization,
            "backend_config": model_config.backend,
            "device_config": model_config.device,
            "batch_size": model_config.batch_size,
            "optimizer": model_config.optimizer,
        }
        results.append({**config_values, **metrics})
        for row in training_curve:
            curves.append({"trial": index, **row})

    best = max(
        results,
        key=lambda row: (
            row["precision_at_k"],
            row["hit_rate_at_k"],
            row["map_at_k"],
            -row["rmse"],
        ),
    )
    return {
        "results": results,
        "training_curves": curves,
        "best": best,
    }


def grid_search_hybrid_weights(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    model_config: MatrixFactorizationConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
    weight_grid: list[tuple[float, float, float]] | None = None,
) -> dict:
    model_config = model_config or MatrixFactorizationConfig()
    evaluation_config = evaluation_config or EvaluationConfig()
    weight_grid = weight_grid or default_hybrid_weight_grid()

    rows = load_time_split_source(db_path, evaluation_config)
    split = split_by_user_time(rows, evaluation_config)
    model = MatrixFactorizationModel(model_config)
    training_curve = model.fit(split["train"])
    candidate_movie_ids = candidate_movie_ids_from_split(
        split,
        evaluation_config,
        known_movie_ids=set(model.movie_to_index),
    )
    quality_scores = movie_quality_scores(split)
    indexes = _build_collaborative_indexes(split["train"])

    results = []
    for index, (mf_weight, collaborative_weight, quality_weight) in enumerate(weight_grid, start=1):
        metrics = evaluate_hybrid_ranker(
            model,
            split,
            evaluation_config,
            candidate_movie_ids=candidate_movie_ids,
            quality_scores=quality_scores,
            indexes=indexes,
            mf_weight=mf_weight,
            collaborative_weight=collaborative_weight,
            quality_weight=quality_weight,
        )
        results.append({"trial": index, **metrics})

    best = max(
        results,
        key=lambda row: (
            row["precision_at_k"],
            row["hit_rate_at_k"],
            row["map_at_k"],
        ),
    )
    return {
        "results": results,
        "best": best,
        "training_curve": training_curve,
    }


def default_hybrid_weight_grid() -> list[tuple[float, float, float]]:
    return [
        (0.10, 0.00, 0.90),
        (0.15, 0.00, 0.85),
        (0.20, 0.00, 0.80),
        (0.25, 0.00, 0.75),
        (0.30, 0.00, 0.70),
        (0.20, 0.05, 0.75),
        (0.25, 0.05, 0.70),
        (0.30, 0.05, 0.65),
        (0.35, 0.10, 0.55),
        (0.50, 0.30, 0.20),
    ]


def load_time_split_source(db_path: Path, config: EvaluationConfig) -> list[RatingRow]:
    if config.max_users <= 0:
        raise ValueError("max_users must be positive")
    if config.min_ratings_per_user < 2:
        raise ValueError("min_ratings_per_user must be at least 2")
    if not 0 < config.test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = conn.execute(
            """
            WITH eligible_users AS (
                SELECT user_id
                FROM user_stats
                WHERE rating_count >= ?
                ORDER BY user_id
                LIMIT ?
            )
            SELECT
                ratings.user_id,
                ratings.movie_id,
                ratings.rating::DOUBLE,
                ratings.rating_date
            FROM ratings
            JOIN eligible_users ON eligible_users.user_id = ratings.user_id
            ORDER BY ratings.user_id, ratings.rating_date, ratings.movie_id
            """,
            [int(config.min_ratings_per_user), int(config.max_users)],
        ).fetchall()
    finally:
        conn.close()

    return [
        RatingRow(
            user_id=int(user_id),
            movie_id=int(movie_id),
            rating=float(rating),
            rating_date=rating_date,
        )
        for user_id, movie_id, rating, rating_date in rows
    ]


def split_by_user_time(rows: Iterable[RatingRow], config: EvaluationConfig) -> dict:
    by_user: dict[int, list[RatingRow]] = defaultdict(list)
    for row in rows:
        by_user[row.user_id].append(row)

    train: list[TrainingRating] = []
    test: list[TrainingRating] = []
    train_history: dict[int, set[int]] = defaultdict(set)
    relevant_by_user: dict[int, set[int]] = defaultdict(set)
    test_by_user: dict[int, list[TrainingRating]] = defaultdict(list)

    for user_id, user_rows in by_user.items():
        if len(user_rows) < config.min_ratings_per_user:
            continue
        ordered = sorted(user_rows, key=lambda row: (row.rating_date, row.movie_id))
        test_count = max(1, round(len(ordered) * config.test_ratio))
        cut = max(1, len(ordered) - test_count)
        user_train = ordered[:cut]
        user_test = ordered[cut:]
        if not user_train or not user_test:
            continue

        for row in user_train:
            train.append(TrainingRating(row.user_id, row.movie_id, row.rating))
            train_history[user_id].add(row.movie_id)
        for row in user_test:
            item = TrainingRating(row.user_id, row.movie_id, row.rating)
            test.append(item)
            test_by_user[user_id].append(item)
            if row.rating >= config.relevant_threshold:
                relevant_by_user[user_id].add(row.movie_id)

    if not train or not test:
        raise ValueError("not enough ratings to build a train/test split")

    return {
        "train": train,
        "test": test,
        "train_history": dict(train_history),
        "relevant_by_user": dict(relevant_by_user),
        "test_by_user": dict(test_by_user),
    }


def evaluate_model(
    model: MatrixFactorizationModel,
    split: dict,
    config: EvaluationConfig,
    *,
    candidate_movie_ids: np.ndarray | None = None,
) -> dict:
    test_rows: list[TrainingRating] = split["test"]
    squared_error = 0.0
    for row in test_rows:
        error = row.rating - model.predict(row.user_id, row.movie_id)
        squared_error += error * error
    rmse = math.sqrt(squared_error / len(test_rows))

    if candidate_movie_ids is None:
        candidate_movie_ids = candidate_movie_ids_from_split(
            split,
            config,
            known_movie_ids=set(model.movie_to_index),
        )

    metrics = _evaluate_ranker(
        "biased_matrix_factorization_" + model.config.optimizer.lower(),
        split,
        config,
        candidate_movie_ids,
        lambda user_id, candidates, exclude: recommend_known_user(
            model,
            user_id,
            candidates,
            exclude_movie_ids=exclude,
            top_k=config.top_k,
        ),
        backend=model.backend_used,
        device=model.device_used,
        rmse=round(rmse, 6),
        users=len(model.user_to_index),
        movies=len(model.movie_to_index),
    )
    return metrics


def evaluate_popular_baseline(
    split: dict,
    config: EvaluationConfig,
    candidate_movie_ids: np.ndarray | None = None,
    quality_scores: dict[int, float] | None = None,
) -> dict:
    if candidate_movie_ids is None:
        candidate_movie_ids = candidate_movie_ids_from_split(split, config)
    quality_scores = quality_scores or movie_quality_scores(split)
    return _evaluate_ranker(
        "popular_quality_baseline",
        split,
        config,
        candidate_movie_ids,
        _static_score_ranker(candidate_movie_ids, quality_scores, config.top_k),
        backend="python",
        device="cpu",
        users=len(split["train_history"]),
        movies=len({row.movie_id for row in split["train"]}),
    )


def evaluate_bias_baseline(
    split: dict,
    config: EvaluationConfig,
    candidate_movie_ids: np.ndarray | None = None,
) -> dict:
    if candidate_movie_ids is None:
        candidate_movie_ids = candidate_movie_ids_from_split(split, config)
    bias_scores = movie_bias_scores(split)
    return _evaluate_ranker(
        "movie_bias_baseline",
        split,
        config,
        candidate_movie_ids,
        _static_score_ranker(candidate_movie_ids, bias_scores, config.top_k),
        backend="python",
        device="cpu",
        users=len(split["train_history"]),
        movies=len({row.movie_id for row in split["train"]}),
    )


def evaluate_user_user_collaborative(
    split: dict,
    config: EvaluationConfig,
    *,
    candidate_movie_ids: np.ndarray | None = None,
    indexes: tuple[
        dict[int, dict[int, float]],
        dict[int, dict[int, float]],
        dict[int, float],
        dict[int, list[tuple[int, float]]],
    ] | None = None,
    score_cache: dict[tuple, dict[int, float]] | None = None,
    neighbor_cache: dict[int, list[dict]] | None = None,
    raw_score_cache: dict[int, dict[int, float]] | None = None,
) -> dict:
    if candidate_movie_ids is None:
        candidate_movie_ids = candidate_movie_ids_from_split(split, config)
    indexes = indexes or _build_collaborative_indexes(split["train"])
    user_ratings, _user_centered, _user_norms, _movie_users = indexes
    return _evaluate_ranker(
        "user_user_collaborative_filtering",
        split,
        config,
        candidate_movie_ids,
        lambda user_id, candidates, exclude: _recommend_user_user(
            user_id,
            user_ratings.get(user_id, {}),
            candidate_movie_ids=candidates,
            exclude_movie_ids=exclude,
            indexes=indexes,
            top_k=config.top_k,
            score_cache=score_cache,
            neighbor_cache=neighbor_cache,
            raw_score_cache=raw_score_cache,
        ),
        backend="python",
        device="cpu",
        users=len(user_ratings),
        movies=len({row.movie_id for row in split["train"]}),
    )


def evaluate_hybrid_ranker(
    model: MatrixFactorizationModel,
    split: dict,
    config: EvaluationConfig,
    *,
    candidate_movie_ids: np.ndarray | None = None,
    quality_scores: dict[int, float] | None = None,
    indexes: tuple[
        dict[int, dict[int, float]],
        dict[int, dict[int, float]],
        dict[int, float],
        dict[int, list[tuple[int, float]]],
    ] | None = None,
    mf_weight: float = HYBRID_MF_WEIGHT,
    collaborative_weight: float = HYBRID_COLLABORATIVE_WEIGHT,
    quality_weight: float = HYBRID_QUALITY_WEIGHT,
) -> dict:
    if candidate_movie_ids is None:
        candidate_movie_ids = candidate_movie_ids_from_split(
            split,
            config,
            known_movie_ids=set(model.movie_to_index),
        )
    quality_scores = quality_scores or movie_quality_scores(split)
    indexes = indexes or _build_collaborative_indexes(split["train"])
    user_ratings = indexes[0]
    algorithm = "hybrid_mf_quality" if collaborative_weight == 0 else "hybrid_mf_user_user_quality"
    metrics = _evaluate_ranker(
        algorithm,
        split,
        config,
        candidate_movie_ids,
        lambda user_id, candidates, exclude: _recommend_hybrid(
            model,
            user_id,
            candidates,
            exclude_movie_ids=exclude,
            active_profile=user_ratings.get(user_id, {}),
            quality_scores=quality_scores,
            indexes=indexes,
            mf_weight=mf_weight,
            collaborative_weight=collaborative_weight,
            quality_weight=quality_weight,
            top_k=config.top_k,
        ),
        backend=model.backend_used,
        device=model.device_used,
        rmse="",
        users=len(model.user_to_index),
        movies=len(model.movie_to_index),
    )
    metrics["mf_weight"] = mf_weight
    metrics["collaborative_weight"] = collaborative_weight
    metrics["quality_weight"] = quality_weight
    return metrics


def _build_collaborative_indexes(
    train_rows: list[TrainingRating],
) -> tuple[
    dict[int, dict[int, float]],
    dict[int, dict[int, float]],
    dict[int, float],
    dict[int, list[tuple[int, float]]],
]:
    grouped: dict[int, dict[int, float]] = defaultdict(dict)
    for row in train_rows:
        grouped[row.user_id][row.movie_id] = row.rating

    user_ratings: dict[int, dict[int, float]] = {}
    user_centered: dict[int, dict[int, float]] = {}
    user_norms: dict[int, float] = {}
    movie_users: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for user_id, values in grouped.items():
        mean_rating = sum(values.values()) / len(values)
        centered = {
            movie_id: rating - mean_rating
            for movie_id, rating in values.items()
        }
        norm = math.sqrt(sum(value * value for value in centered.values()))
        if not norm:
            continue
        user_ratings[user_id] = dict(values)
        user_centered[user_id] = centered
        user_norms[user_id] = norm
        for movie_id, centered_rating in centered.items():
            movie_users[movie_id].append((user_id, centered_rating))
    return user_ratings, user_centered, user_norms, dict(movie_users)


def candidate_movie_ids_from_split(
    split: dict,
    config: EvaluationConfig,
    *,
    known_movie_ids: set[int] | None = None,
) -> np.ndarray:
    movie_counts: Counter[int] = Counter(row.movie_id for row in split["train"])
    movie_rating_sums: defaultdict[int, float] = defaultdict(float)
    for row in split["train"]:
        movie_rating_sums[row.movie_id] += row.rating

    relevant_movie_ids = {
        movie_id
        for values in split["relevant_by_user"].values()
        for movie_id in values
        if known_movie_ids is None or movie_id in known_movie_ids
    }
    ranked_movies = sorted(
        (
            movie_id
            for movie_id in movie_counts
            if known_movie_ids is None or movie_id in known_movie_ids
        ),
        key=lambda movie_id: (
            movie_counts[movie_id],
            movie_rating_sums[movie_id] / movie_counts[movie_id],
            -movie_id,
        ),
        reverse=True,
    )
    if not config.candidate_limit or len(ranked_movies) <= config.candidate_limit:
        selected = list(ranked_movies)
    else:
        headroom = max(config.candidate_limit - len(relevant_movie_ids), 0)
        selected = list(ranked_movies[:headroom])

    seen = set(selected)
    for movie_id in sorted(relevant_movie_ids):
        if movie_id not in seen:
            selected.append(movie_id)
            seen.add(movie_id)
    return np.array(selected, dtype=np.int32)


def movie_quality_scores(split: dict) -> dict[int, float]:
    counts: Counter[int] = Counter()
    sums: defaultdict[int, float] = defaultdict(float)
    for row in split["train"]:
        counts[row.movie_id] += 1
        sums[row.movie_id] += row.rating

    global_avg = sum(sums.values()) / sum(counts.values()) if counts else 0.0
    max_log_count = math.log1p(max(counts.values(), default=1))
    scores = {}
    for movie_id, count in counts.items():
        avg_rating = sums[movie_id] / count
        bayesian = (count * avg_rating + MOVIE_RATING_PRIOR * global_avg) / (count + MOVIE_RATING_PRIOR)
        rating_component = bayesian / 5.0
        popularity_component = math.log1p(count) / max_log_count if max_log_count else 0.0
        scores[movie_id] = rating_component * 0.75 + popularity_component * 0.25
    return scores


def movie_bias_scores(split: dict) -> dict[int, float]:
    counts: Counter[int] = Counter()
    sums: defaultdict[int, float] = defaultdict(float)
    for row in split["train"]:
        counts[row.movie_id] += 1
        sums[row.movie_id] += row.rating
    global_avg = sum(sums.values()) / sum(counts.values()) if counts else 0.0
    raw_scores = {
        movie_id: (sums[movie_id] / count) - global_avg
        for movie_id, count in counts.items()
    }
    return _normalize_score_dict(raw_scores)


def _evaluate_ranker(
    algorithm: str,
    split: dict,
    config: EvaluationConfig,
    candidate_movie_ids: np.ndarray,
    ranker,
    *,
    backend: str,
    device: str,
    rmse: float | str = "",
    users: int,
    movies: int,
) -> dict:
    precision_sum = 0.0
    recall_sum = 0.0
    hit_count = 0
    average_precision_sum = 0.0
    evaluated_users = 0
    empty_users = 0
    recommended_movies: set[int] = set()

    for user_id, relevant_movies in split["relevant_by_user"].items():
        if not relevant_movies:
            continue
        exclude = split["train_history"].get(user_id, set())
        ranked = ranker(user_id, candidate_movie_ids, exclude)
        if not ranked:
            empty_users += 1
            continue
        evaluated_users += 1
        movie_ids = [movie_id for movie_id, _score in ranked[: config.top_k]]
        recommended_movies.update(movie_ids)
        hits = [movie_id for movie_id in movie_ids if movie_id in relevant_movies]
        precision_sum += len(hits) / config.top_k
        recall_sum += len(hits) / len(relevant_movies)
        hit_count += 1 if hits else 0
        average_precision_sum += _average_precision_at_k(movie_ids, relevant_movies, config.top_k)

    return {
        "algorithm": algorithm,
        "backend": backend,
        "device": device,
        "rmse": rmse,
        "precision_at_k": round(precision_sum / evaluated_users, 6) if evaluated_users else 0.0,
        "recall_at_k": round(recall_sum / evaluated_users, 6) if evaluated_users else 0.0,
        "hit_rate_at_k": round(hit_count / evaluated_users, 6) if evaluated_users else 0.0,
        "map_at_k": round(average_precision_sum / evaluated_users, 6) if evaluated_users else 0.0,
        "catalog_coverage": round(len(recommended_movies) / len(candidate_movie_ids), 6)
        if len(candidate_movie_ids)
        else 0.0,
        "top_k": config.top_k,
        "users": users,
        "movies": movies,
        "train_ratings": len(split["train"]),
        "test_ratings": len(split["test"]),
        "evaluated_users": evaluated_users,
        "candidate_movies": len(candidate_movie_ids),
        "relevant_threshold": config.relevant_threshold,
        "empty_users": empty_users,
    }


def _rank_static_scores(
    candidate_movie_ids: np.ndarray,
    exclude_movie_ids: set[int],
    scores: dict[int, float],
    top_k: int,
) -> list[tuple[int, float]]:
    ranked = [
        (int(movie_id), float(scores.get(int(movie_id), 0.0)))
        for movie_id in candidate_movie_ids
        if int(movie_id) not in exclude_movie_ids
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def _static_score_ranker(candidate_movie_ids: np.ndarray, scores: dict[int, float], top_k: int):
    """Build a per-user ranker for a fixed score table.

    The candidate score vector is materialized once (instead of once per user),
    and ranking uses a stable descending sort so ties keep candidate order,
    matching the original ``_rank_static_scores`` output exactly.
    """
    movie_ids = candidate_movie_ids.astype(np.int64, copy=False)
    score_values = np.array([float(scores.get(int(movie_id), 0.0)) for movie_id in movie_ids], dtype=np.float64)

    def ranker(_user_id, _candidates, exclude_movie_ids):
        return _rank_static_scores_array(movie_ids, score_values, exclude_movie_ids, top_k)

    return ranker


def _rank_static_scores_array(
    movie_ids: np.ndarray,
    score_values: np.ndarray,
    exclude_movie_ids: set[int],
    top_k: int,
) -> list[tuple[int, float]]:
    if exclude_movie_ids:
        keep_mask = ~np.isin(
            movie_ids,
            np.fromiter(exclude_movie_ids, dtype=np.int64, count=len(exclude_movie_ids)),
        )
        ids = movie_ids[keep_mask]
        values = score_values[keep_mask]
    else:
        ids = movie_ids
        values = score_values
    if len(ids) == 0:
        return []
    take = min(top_k, len(ids))
    order = np.argsort(-values, kind="stable")[:take]
    return [(int(ids[position]), float(values[position])) for position in order]


def _recommend_user_user(
    active_user_id: int,
    active_profile: dict[int, float],
    *,
    candidate_movie_ids: np.ndarray,
    exclude_movie_ids: set[int],
    indexes: tuple[
        dict[int, dict[int, float]],
        dict[int, dict[int, float]],
        dict[int, float],
        dict[int, list[tuple[int, float]]],
    ],
    top_k: int,
    score_cache: dict[tuple, dict[int, float]] | None = None,
    neighbor_cache: dict[int, list[dict]] | None = None,
    raw_score_cache: dict[int, dict[int, float]] | None = None,
) -> list[tuple[int, float]]:
    scores = _score_user_user_candidates(
        active_user_id,
        active_profile,
        indexes=indexes,
        candidate_movie_ids=candidate_movie_ids,
        exclude_movie_ids=exclude_movie_ids,
        score_cache=score_cache,
        neighbor_cache=neighbor_cache,
        raw_score_cache=raw_score_cache,
    )
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def _compute_user_neighbors(
    active_user_id: int,
    active_profile: dict[int, float],
    indexes: tuple[
        dict[int, dict[int, float]],
        dict[int, dict[int, float]],
        dict[int, float],
        dict[int, list[tuple[int, float]]],
    ],
) -> list[dict]:
    """Find the top similar users for ``active_user_id``.

    This is the most expensive part of user-user CF and depends only on the
    active profile and the training indexes (not on the candidate set or the
    exclusion list). Splitting it out lets callers cache the result and reuse
    it across the recall, evaluation, and feature-building stages.
    """
    _user_ratings, _user_centered, user_norms, movie_users = indexes
    if len(active_profile) < 3:
        return []

    active_mean = sum(active_profile.values()) / len(active_profile)
    active_centered = {
        movie_id: rating - active_mean
        for movie_id, rating in active_profile.items()
    }
    active_norm = math.sqrt(sum(value * value for value in active_centered.values()))
    if not active_norm:
        return []

    dot_scores: defaultdict[int, float] = defaultdict(float)
    shared_counts: Counter[int] = Counter()
    for movie_id, active_value in active_centered.items():
        for user_id, user_value in movie_users.get(movie_id, ()):
            if user_id == active_user_id:
                continue
            dot_scores[user_id] += active_value * user_value
            shared_counts[user_id] += 1

    neighbors = []
    for user_id, dot_score in dot_scores.items():
        shared_count = shared_counts[user_id]
        if shared_count < MIN_SHARED_MOVIES:
            continue
        raw_similarity = dot_score / (active_norm * user_norms[user_id])
        if raw_similarity <= 0:
            continue
        overlap_weight = shared_count / (shared_count + OVERLAP_SMOOTHING)
        neighbors.append(
            {
                "user_id": user_id,
                "similarity": raw_similarity * overlap_weight,
            }
        )
    neighbors.sort(key=lambda item: item["similarity"], reverse=True)
    return neighbors[:NEIGHBOR_LIMIT]


def _score_user_user_candidates(
    active_user_id: int,
    active_profile: dict[int, float],
    *,
    indexes: tuple[
        dict[int, dict[int, float]],
        dict[int, dict[int, float]],
        dict[int, float],
        dict[int, list[tuple[int, float]]],
    ],
    candidate_movie_ids: np.ndarray,
    exclude_movie_ids: set[int],
    candidate_movie_id_set: set[int] | None = None,
    score_cache: dict[tuple, dict[int, float]] | None = None,
    neighbor_cache: dict[int, list[dict]] | None = None,
    raw_score_cache: dict[int, dict[int, float]] | None = None,
) -> dict[int, float]:
    cache_key = None
    if score_cache is not None:
        cache_key = _score_cache_key(active_user_id, candidate_movie_ids, exclude_movie_ids)
        cached = score_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

    user_ratings, user_centered, user_norms, movie_users = indexes

    if neighbor_cache is not None:
        neighbors = neighbor_cache.get(active_user_id)
        if neighbors is None:
            neighbors = _compute_user_neighbors(active_user_id, active_profile, indexes)
            neighbor_cache[active_user_id] = neighbors
    else:
        neighbors = _compute_user_neighbors(active_user_id, active_profile, indexes)

    if not neighbors:
        if score_cache is not None and cache_key is not None:
            score_cache[cache_key] = {}
        return {}

    if raw_score_cache is not None:
        raw_scores = raw_score_cache.get(active_user_id)
        if raw_scores is None:
            raw_scores = _score_neighbors_full_catalog(neighbors, indexes)
            raw_score_cache[active_user_id] = raw_scores
    else:
        raw_scores = _score_neighbors_full_catalog(neighbors, indexes)

    candidate_set = candidate_movie_id_set or {int(movie_id) for movie_id in candidate_movie_ids}
    scores = {
        movie_id: score
        for movie_id, score in raw_scores.items()
        if movie_id in candidate_set and movie_id not in exclude_movie_ids
    }
    if score_cache is not None and cache_key is not None:
        score_cache[cache_key] = dict(scores)
    return scores


def _score_neighbors_full_catalog(
    neighbors: list[dict],
    indexes: tuple[
        dict[int, dict[int, float]],
        dict[int, dict[int, float]],
        dict[int, float],
        dict[int, list[tuple[int, float]]],
    ],
) -> dict[int, float]:
    """Score every catalog movie reachable from ``neighbors``.

    This is the candidate-scoring core with the candidate/exclude filters
    removed: each movie's raw score depends only on the neighbours (not on the
    candidate set), so it can be computed once per user and cached, then the
    caller cheaply intersects with the candidate set and exclusion list. The
    per-movie accumulation order is identical to the original implementation,
    so the result is bit-for-bit lossless after filtering.
    """
    user_ratings, user_centered, _user_norms, _movie_users = indexes
    weighted_scores: defaultdict[int, float] = defaultdict(float)
    similarity_sums: defaultdict[int, float] = defaultdict(float)
    support_counts: Counter[int] = Counter()
    for neighbor in neighbors:
        user_id = neighbor["user_id"]
        similarity = float(neighbor["similarity"])
        centered_lookup = user_centered[user_id]
        for movie_id, rating in user_ratings[user_id].items():
            if rating < 4.0:
                continue
            centered_rating = centered_lookup.get(movie_id, 0.0)
            if centered_rating <= 0:
                continue
            weighted_scores[movie_id] += similarity * centered_rating
            similarity_sums[movie_id] += abs(similarity)
            support_counts[movie_id] += 1

    scores = {}
    for movie_id, weighted_score in weighted_scores.items():
        if support_counts[movie_id] < SUPPORT_LIMIT or similarity_sums[movie_id] <= 0:
            continue
        support_bonus = min(math.log1p(support_counts[movie_id]) / math.log1p(NEIGHBOR_LIMIT), 1.0)
        score = (weighted_score / similarity_sums[movie_id]) * 80.0 + support_bonus * 20.0
        scores[movie_id] = score
    return scores


def _score_cache_key(user_id: int, candidate_movie_ids: np.ndarray, exclude_movie_ids: set[int]) -> tuple:
    if len(candidate_movie_ids) == 0:
        candidate_fingerprint = (0, 0, 0, 0)
    else:
        values = candidate_movie_ids.astype(np.int64, copy=False)
        if exclude_movie_ids:
            values = values[~np.isin(values, list(exclude_movie_ids), assume_unique=False)]
        if len(values) == 0:
            candidate_fingerprint = (0, 0, 0, 0)
        else:
            candidate_fingerprint = (
                int(len(values)),
                int(values[0]),
                int(values[-1]),
                int(values.sum()),
            )
    exclude_fingerprint = (
        len(exclude_movie_ids),
        int(sum(exclude_movie_ids)) if exclude_movie_ids else 0,
    )
    return (int(user_id), candidate_fingerprint, exclude_fingerprint)

def _recommend_hybrid(
    model: MatrixFactorizationModel,
    user_id: int,
    candidate_movie_ids: np.ndarray,
    *,
    exclude_movie_ids: set[int],
    active_profile: dict[int, float],
    quality_scores: dict[int, float],
    indexes: tuple[
        dict[int, dict[int, float]],
        dict[int, dict[int, float]],
        dict[int, float],
        dict[int, list[tuple[int, float]]],
    ],
    mf_weight: float,
    collaborative_weight: float,
    quality_weight: float,
    top_k: int,
) -> list[tuple[int, float]]:
    mf_raw = model.score_known_user(user_id, candidate_movie_ids)
    mf_scores = {
        int(movie_id): (float(score) - model.config.min_rating) / (model.config.max_rating - model.config.min_rating)
        for movie_id, score in zip(candidate_movie_ids, mf_raw, strict=True)
    }
    collaborative_scores = _normalize_score_dict(
        _score_user_user_candidates(
            user_id,
            active_profile,
            indexes=indexes,
            candidate_movie_ids=candidate_movie_ids,
            exclude_movie_ids=exclude_movie_ids,
        )
    )
    ranked = []
    for movie_id_value in candidate_movie_ids:
        movie_id = int(movie_id_value)
        if movie_id in exclude_movie_ids:
            continue
        score = (
            mf_weight * mf_scores.get(movie_id, 0.0)
            + collaborative_weight * collaborative_scores.get(movie_id, 0.0)
            + quality_weight * quality_scores.get(movie_id, 0.0)
        )
        ranked.append((movie_id, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def _normalize_score_dict(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_value = min(values)
    max_value = max(values)
    if max_value <= min_value:
        return {key: 1.0 for key in scores}
    return {
        key: (value - min_value) / (max_value - min_value)
        for key, value in scores.items()
    }


def _candidate_movie_ids(
    model: MatrixFactorizationModel,
    split: dict,
    config: EvaluationConfig,
) -> list[int]:
    movie_ids = list(model.index_to_movie)
    if not config.candidate_limit or len(movie_ids) <= config.candidate_limit:
        return movie_ids

    relevant_movie_ids = {
        movie_id
        for values in split["relevant_by_user"].values()
        for movie_id in values
        if movie_id in model.movie_to_index
    }
    headroom = max(config.candidate_limit - len(relevant_movie_ids), 0)
    selected = list(movie_ids[:headroom])
    seen = set(selected)
    for movie_id in sorted(relevant_movie_ids):
        if movie_id not in seen:
            selected.append(movie_id)
            seen.add(movie_id)
    return selected[: max(config.candidate_limit, len(relevant_movie_ids))]


def recommend_known_user(
    model: MatrixFactorizationModel,
    user_id: int,
    candidate_movie_ids: np.ndarray,
    *,
    exclude_movie_ids: set[int],
    top_k: int,
) -> list[tuple[int, float]]:
    if top_k <= 0:
        return []
    scores = model.score_known_user(user_id, candidate_movie_ids)
    eligible_mask = np.array(
        [int(movie_id) not in exclude_movie_ids for movie_id in candidate_movie_ids],
        dtype=bool,
    )
    eligible_indices = np.flatnonzero(eligible_mask)
    if not len(eligible_indices):
        return []

    take = min(top_k, len(eligible_indices))
    eligible_scores = scores[eligible_indices]
    top_positions = np.argpartition(-eligible_scores, take - 1)[:take]
    ranked_positions = top_positions[np.argsort(-eligible_scores[top_positions])]
    result = []
    for position in ranked_positions:
        source_index = int(eligible_indices[position])
        result.append((int(candidate_movie_ids[source_index]), float(scores[source_index])))
    return result


def _average_precision_at_k(movie_ids: list[int], relevant_movies: set[int], top_k: int) -> float:
    hits = 0
    total = 0.0
    for rank, movie_id in enumerate(movie_ids[:top_k], start=1):
        if movie_id in relevant_movies:
            hits += 1
            total += hits / rank
    return total / min(len(relevant_movies), top_k)
