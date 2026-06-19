from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

import duckdb
import numpy as np

from src.datasets.netflix.evaluation import (
    EvaluationConfig,
    HYBRID_COLLABORATIVE_WEIGHT,
    HYBRID_MF_WEIGHT,
    HYBRID_QUALITY_WEIGHT,
    RatingRow,
    _average_precision_at_k,
    _build_collaborative_indexes,
    _evaluate_ranker,
    _normalize_score_dict,
    _score_user_user_candidates,
    candidate_movie_ids_from_split,
    evaluate_model,
    evaluate_popular_baseline,
    evaluate_user_user_collaborative,
    load_time_split_source,
    movie_quality_scores,
)
from src.datasets.netflix.import_duckdb import DEFAULT_DB_PATH
from src.datasets.netflix.candidate_recall import build_multi_route_candidate_pool
from src.datasets.netflix.collaborative_vectorized import prime_collaborative_caches
from src.datasets.netflix.matrix_factorization import (
    MatrixFactorizationConfig,
    MatrixFactorizationModel,
    TrainingRating,
    mf_batch_enabled,
)


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

ITEM_ITEM_MAX_PROFILE_MOVIES = 40
ITEM_ITEM_MAX_NEIGHBORS = 64
ITEM_ITEM_MIN_PAIR_COUNT = 2
ITEM_ITEM_SHRINKAGE = 10.0


@dataclass(frozen=True)
class LearningToRankConfig:
    max_users: int = 160
    min_ratings_per_user: int = 12
    ranker_ratio: float = 0.2
    test_ratio: float = 0.2
    relevant_threshold: float = 4.0
    top_k: int = 10
    candidate_limit: int = 1200
    candidate_recall_strategy: str = "legacy"
    candidate_recall_rrf_k: float = 60.0
    candidate_recall_priors: tuple[tuple[str, float, float, float], ...] = ()
    negatives_per_positive: int = 8
    negative_sampling: str = "hybrid_hard"
    epochs: int = 160
    learning_rate: float = 0.08
    l2: float = 0.001
    min_blend_precision_gain: float = 0.0125
    blend_mode: str = "value"
    seed: int = 42
    mf_factors: int = 48
    mf_epochs: int = 20
    mf_learning_rate: float = 0.02
    mf_regularization: float = 0.04
    mf_batch_size: int = 8192
    experimental_algorithm_features: bool = False
    evaluate_pairwise_ranker: bool = False
    evaluate_residual_ranker: bool = False
    evaluate_mmr_ranker: bool = False


class LinearRanker:
    def __init__(self, *, learning_rate: float, l2: float, epochs: int, seed: int) -> None:
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.seed = seed
        self.weights = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        self.bias = 0.0
        self.blend_weight = 1.0
        self.blend_mode = "value"
        self.mean = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        self.std = np.ones(len(FEATURE_NAMES), dtype=np.float32)
        self.training_curve: list[dict] = []

    def fit(self, features: np.ndarray, labels: np.ndarray) -> list[dict]:
        if len(features) == 0:
            raise ValueError("learning-to-rank needs at least one training example")
        if not labels.any():
            raise ValueError("learning-to-rank needs at least one positive example")

        self.mean = features.mean(axis=0).astype(np.float32)
        self.std = features.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        x = (features - self.mean) / self.std
        y = labels.astype(np.float32)

        positive_rate = float(y.mean())
        positive_rate = min(max(positive_rate, 1e-4), 1.0 - 1e-4)
        self.bias = math.log(positive_rate / (1.0 - positive_rate))
        self.weights = np.zeros(x.shape[1], dtype=np.float32)

        # Full-batch gradient descent: the gradient sums over every example, so a
        # per-epoch row shuffle is mathematically a no-op (it only changed float
        # summation order). Dropping it removes a full N x F copy per epoch.
        x_t = x.T
        for epoch in range(1, self.epochs + 1):
            logits = np.clip(x @ self.weights + self.bias, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            errors = probabilities - y
            grad_w = (x_t @ errors) / len(x) + self.l2 * self.weights
            grad_b = float(errors.mean())
            self.weights -= self.learning_rate * grad_w.astype(np.float32)
            self.bias -= self.learning_rate * grad_b

            loss = -np.mean(
                y * np.log(probabilities + 1e-8)
                + (1.0 - y) * np.log(1.0 - probabilities + 1e-8)
            )
            loss += 0.5 * self.l2 * float(self.weights @ self.weights)
            self.training_curve.append(
                {
                    "epoch": epoch,
                    "loss": round(float(loss), 6),
                    "positive_rate": round(float(positive_rate), 6),
                    "examples": int(len(features)),
                    "positives": int(labels.sum()),
                    "negatives": int(len(labels) - labels.sum()),
                }
            )
        return list(self.training_curve)

    def score(self, features: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.empty(0, dtype=np.float32)
        x = (features - self.mean) / self.std
        return x @ self.weights + self.bias

    def feature_weights(self) -> list[dict]:
        return [
            {
                "feature": name,
                "weight": round(float(weight), 6),
                "mean": round(float(mean), 6),
                "std": round(float(std), 6),
            }
            for name, weight, mean, std in zip(FEATURE_NAMES, self.weights, self.mean, self.std, strict=True)
        ]


class PairwiseLinearRanker(LinearRanker):
    def __init__(
        self,
        *,
        learning_rate: float,
        l2: float,
        epochs: int,
        seed: int,
        pairs_per_positive: int = 4,
        batch_size: int = 65536,
    ) -> None:
        super().__init__(learning_rate=learning_rate, l2=l2, epochs=epochs, seed=seed)
        self.pairs_per_positive = pairs_per_positive
        self.batch_size = batch_size

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        user_ids: np.ndarray,
    ) -> list[dict]:
        if len(features) == 0:
            raise ValueError("pairwise learning-to-rank needs at least one training example")
        if not labels.any():
            raise ValueError("pairwise learning-to-rank needs at least one positive example")

        left, right = _build_pairwise_indices(
            user_ids,
            labels,
            features,
            pairs_per_positive=self.pairs_per_positive,
        )
        if len(left) == 0:
            raise ValueError("pairwise learning-to-rank needs at least one positive-negative pair")

        self.mean = features.mean(axis=0).astype(np.float32)
        self.std = features.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        x = ((features - self.mean) / self.std).astype(np.float32)
        self.weights = np.zeros(x.shape[1], dtype=np.float32)
        self.bias = 0.0

        rng = np.random.default_rng(self.seed)
        self.training_curve = []
        batch_size = max(1, int(self.batch_size))
        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(len(left))
            total_loss = 0.0
            total_pairs = 0
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                diffs = x[left[batch]] - x[right[batch]]
                margins = np.clip(diffs @ self.weights, -30.0, 30.0)
                coefficients = -1.0 / (1.0 + np.exp(margins))
                grad_w = (coefficients.astype(np.float32) @ diffs) / len(batch)
                grad_w += self.l2 * self.weights
                self.weights -= self.learning_rate * grad_w.astype(np.float32)
                total_loss += float(np.logaddexp(0.0, -margins).sum())
                total_pairs += len(batch)

            self.training_curve.append(
                {
                    "epoch": epoch,
                    "loss": round(total_loss / max(total_pairs, 1), 6),
                    "positive_rate": round(float(labels.mean()), 6),
                    "examples": int(len(features)),
                    "positives": int(labels.sum()),
                    "negatives": int(len(labels) - labels.sum()),
                    "pairs": int(len(left)),
                    "pairs_per_positive": int(self.pairs_per_positive),
                }
            )
        return list(self.training_curve)


class ResidualLinearRanker(LinearRanker):
    def __init__(self, *, learning_rate: float, l2: float, epochs: int, seed: int) -> None:
        super().__init__(learning_rate=learning_rate, l2=l2, epochs=epochs, seed=seed)
        self.residual_alpha = 0.0

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        user_ids: np.ndarray,
        *,
        base_ranker: LinearRanker | None = None,
    ) -> list[dict]:
        if len(features) == 0:
            raise ValueError("residual learning-to-rank needs at least one training example")
        if not labels.any():
            raise ValueError("residual learning-to-rank needs at least one positive example")

        self.mean = features.mean(axis=0).astype(np.float32)
        self.std = features.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        x = (features - self.mean) / self.std
        base_scores = _residual_base_scores_by_user(features, user_ids, base_ranker=base_ranker)
        targets = labels.astype(np.float32) - base_scores

        self.bias = float(targets.mean())
        self.weights = np.zeros(x.shape[1], dtype=np.float32)
        self.training_curve = []
        x_t = x.T
        base_rmse = float(np.sqrt(np.mean(np.square(targets))))
        for epoch in range(1, self.epochs + 1):
            predictions = x @ self.weights + self.bias
            errors = predictions - targets
            grad_w = (x_t @ errors) / len(x) + self.l2 * self.weights
            grad_b = float(errors.mean())
            self.weights -= self.learning_rate * grad_w.astype(np.float32)
            self.bias -= self.learning_rate * grad_b

            mse = float(np.mean(np.square(errors)))
            loss = mse + 0.5 * self.l2 * float(self.weights @ self.weights)
            self.training_curve.append(
                {
                    "epoch": epoch,
                    "loss": round(loss, 6),
                    "residual_rmse": round(math.sqrt(mse), 6),
                    "base_residual_rmse": round(base_rmse, 6),
                    "examples": int(len(features)),
                    "positives": int(labels.sum()),
                    "negatives": int(len(labels) - labels.sum()),
                }
            )
        return list(self.training_curve)


def _build_pairwise_indices(
    user_ids: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray,
    *,
    pairs_per_positive: int,
) -> tuple[np.ndarray, np.ndarray]:
    left: list[int] = []
    right: list[int] = []
    hard_scores = _hard_negative_scores(features)
    for user_id in np.unique(user_ids):
        user_positions = np.flatnonzero(user_ids == user_id)
        positive_positions = user_positions[labels[user_positions] > 0]
        negative_positions = user_positions[labels[user_positions] <= 0]
        if len(positive_positions) == 0 or len(negative_positions) == 0:
            continue
        negative_order = negative_positions[np.argsort(-hard_scores[negative_positions])]
        chosen_negatives = negative_order[: max(1, min(len(negative_order), pairs_per_positive))]
        for positive_position in positive_positions:
            for negative_position in chosen_negatives:
                left.append(int(positive_position))
                right.append(int(negative_position))
    return np.array(left, dtype=np.int64), np.array(right, dtype=np.int64)


def smoke_learning_to_rank_config() -> LearningToRankConfig:
    return LearningToRankConfig(
        max_users=40,
        candidate_limit=300,
        epochs=40,
        mf_factors=16,
        mf_epochs=5,
        mf_batch_size=4096,
    )


def standard_learning_to_rank_config() -> LearningToRankConfig:
    return LearningToRankConfig(negative_sampling="explicit_hard")


def _base_candidate_source_names() -> tuple[str, ...]:
    return (
        "popular_quality",
        "mf_user_top",
        "profile_centroid",
        "user_user_cf",
        "year_affinity",
    )


def run_learning_to_rank(
    output_dir: Path,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    config: LearningToRankConfig | None = None,
) -> dict[str, Path]:
    config = config or LearningToRankConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_time_split_source(db_path, _evaluation_config(config))
    split = split_by_user_three_way(rows, config)
    model = MatrixFactorizationModel(_mf_config(config))
    mf_training_curve = model.fit(split["train"])
    quality_scores = movie_quality_scores(split)
    indexes = _build_collaborative_indexes(split["train"])
    feature_stats = _feature_stats(
        split["train"],
        db_path=db_path,
        model=model,
        enable_item_item=config.experimental_algorithm_features,
    )
    feature_stats["enable_experimental_algorithm_features"] = config.experimental_algorithm_features
    feature_stats["candidate_source_names"] = (
        None if config.experimental_algorithm_features else _base_candidate_source_names()
    )
    feature_stats["cache_user_candidate_features"] = True
    prime_collaborative_caches(indexes, sorted(split["train_history"]), feature_stats)
    recall_result = build_multi_route_candidate_pool(
        _candidate_split(split, include_test_relevant=False),
        _evaluation_config(config),
        model,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    feature_stats["candidate_recall_features"] = recall_result.movie_features
    candidate_movie_ids = _ensure_training_candidates(recall_result.movie_ids, split, model)
    if mf_batch_enabled():
        model.prime_candidate_score_cache(sorted(split["train_history"]), candidate_movie_ids)

    train_features, train_labels, train_user_ids = build_ranker_training_examples(
        model,
        split,
        config,
        candidate_movie_ids=candidate_movie_ids,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    ranker = LinearRanker(
        learning_rate=config.learning_rate,
        l2=config.l2,
        epochs=config.epochs,
        seed=config.seed,
    )
    ranker_training_curve = ranker.fit(train_features, train_labels)
    pairwise_ranker = None
    pairwise_training_curve: list[dict] = []
    if config.evaluate_pairwise_ranker:
        pairwise_ranker = PairwiseLinearRanker(
            learning_rate=config.learning_rate,
            l2=config.l2,
            epochs=config.epochs,
            seed=config.seed,
        )
        pairwise_training_curve = pairwise_ranker.fit(train_features, train_labels, train_user_ids)
        pairwise_ranker.blend_weight = 1.0
        pairwise_ranker.blend_mode = "value"
    blend_weight, blend_tuning_rows = tune_blend_weight(
        ranker,
        model,
        split,
        config,
        candidate_movie_ids=candidate_movie_ids,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    ranker.blend_weight = blend_weight
    residual_ranker = None
    residual_training_curve: list[dict] = []
    residual_tuning_rows: list[dict] = []
    residual_alpha = 0.0
    if config.evaluate_residual_ranker:
        residual_ranker = ResidualLinearRanker(
            learning_rate=config.learning_rate,
            l2=config.l2,
            epochs=config.epochs,
            seed=config.seed,
        )
        residual_training_curve = residual_ranker.fit(
            train_features,
            train_labels,
            train_user_ids,
            base_ranker=ranker,
        )
        residual_alpha, residual_tuning_rows = tune_residual_alpha(
            residual_ranker,
            ranker,
            model,
            split,
            config,
            candidate_movie_ids=candidate_movie_ids,
            quality_scores=quality_scores,
            indexes=indexes,
            feature_stats=feature_stats,
        )
        residual_ranker.residual_alpha = residual_alpha
    feature_stats.get("user_candidate_feature_cache", {}).clear()

    baseline_split = _evaluation_split(split)
    popular_metrics = evaluate_popular_baseline(
        baseline_split,
        _evaluation_config(config),
        candidate_movie_ids=candidate_movie_ids,
        quality_scores=quality_scores,
    )
    collaborative_metrics = evaluate_user_user_collaborative(
        baseline_split,
        _evaluation_config(config),
        candidate_movie_ids=candidate_movie_ids,
        indexes=indexes,
        score_cache=feature_stats.setdefault("collaborative_score_cache", {}),
        neighbor_cache=feature_stats.setdefault("user_neighbor_cache", {}),
        raw_score_cache=feature_stats.setdefault("collaborative_raw_cache", {}),
    )
    mf_metrics = evaluate_model(
        model,
        baseline_split,
        _evaluation_config(config),
        candidate_movie_ids=candidate_movie_ids,
    )
    hybrid_metrics = evaluate_hybrid_ranker_from_features(
        model,
        baseline_split,
        _evaluation_config(config),
        candidate_movie_ids=candidate_movie_ids,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    learned_metrics = evaluate_learned_ranker(
        ranker,
        model,
        baseline_split,
        _evaluation_config(config),
        candidate_movie_ids=candidate_movie_ids,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    extra_metrics = []
    if pairwise_ranker is not None:
        extra_metrics.append(
            evaluate_learned_ranker(
                pairwise_ranker,
                model,
                baseline_split,
                _evaluation_config(config),
                candidate_movie_ids=candidate_movie_ids,
                quality_scores=quality_scores,
                indexes=indexes,
                feature_stats=feature_stats,
                algorithm="pairwise_linear_hybrid_reranker",
            )
        )
    if residual_ranker is not None:
        extra_metrics.append(
            evaluate_residual_ranker(
                residual_ranker,
                ranker,
                model,
                baseline_split,
                _evaluation_config(config),
                candidate_movie_ids=candidate_movie_ids,
                quality_scores=quality_scores,
                indexes=indexes,
                feature_stats=feature_stats,
            )
        )
    if config.evaluate_mmr_ranker:
        extra_metrics.append(
            evaluate_mmr_learned_ranker(
                ranker,
                model,
                baseline_split,
                _evaluation_config(config),
                candidate_movie_ids=candidate_movie_ids,
                quality_scores=quality_scores,
                indexes=indexes,
                feature_stats=feature_stats,
            )
        )
    feature_stats.get("user_candidate_feature_cache", {}).clear()
    feature_stats.get("collaborative_score_cache", {}).clear()

    metrics_path = output_dir / "metrics.csv"
    ranker_training_path = output_dir / "ranker_training.csv"
    pairwise_training_path = output_dir / "pairwise_training.csv"
    residual_training_path = output_dir / "residual_training.csv"
    blend_tuning_path = output_dir / "blend_tuning.csv"
    residual_tuning_path = output_dir / "residual_tuning.csv"
    candidate_recall_path = output_dir / "candidate_recall.csv"
    mf_training_path = output_dir / "mf_training.csv"
    weights_path = output_dir / "feature_weights.csv"
    summary_path = output_dir / "summary.json"
    metrics_rows = [
        popular_metrics,
        collaborative_metrics,
        mf_metrics,
        hybrid_metrics,
        learned_metrics,
        *extra_metrics,
    ]
    _write_csv(metrics_path, metrics_rows)
    _write_csv(ranker_training_path, ranker_training_curve)
    _write_csv(pairwise_training_path, pairwise_training_curve)
    _write_csv(residual_training_path, residual_training_curve)
    _write_csv(blend_tuning_path, blend_tuning_rows)
    _write_csv(residual_tuning_path, residual_tuning_rows)
    _write_csv(candidate_recall_path, recall_result.source_rows)
    _write_csv(mf_training_path, mf_training_curve)
    _write_csv(weights_path, ranker.feature_weights())
    summary_path.write_text(
        json.dumps(
            {
                "config": config.__dict__,
                "training_examples": int(len(train_labels)),
                "training_positives": int(train_labels.sum()),
                "training_negatives": int(len(train_labels) - train_labels.sum()),
                "linear_blend_weight": float(blend_weight),
                "linear_blend_mode": ranker.blend_mode,
                "residual_alpha": float(residual_alpha),
                "pairwise_pairs": int(pairwise_training_curve[-1]["pairs"]) if pairwise_training_curve else 0,
                "best_metric": max(
                    metrics_rows,
                    key=lambda row: (row["precision_at_k"], row["hit_rate_at_k"], row["map_at_k"]),
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "metrics": metrics_path,
        "ranker_training": ranker_training_path,
        "pairwise_training": pairwise_training_path,
        "residual_training": residual_training_path,
        "blend_tuning": blend_tuning_path,
        "residual_tuning": residual_tuning_path,
        "candidate_recall": candidate_recall_path,
        "mf_training": mf_training_path,
        "feature_weights": weights_path,
        "summary": summary_path,
    }


def split_by_user_three_way(rows: list[RatingRow], config: LearningToRankConfig) -> dict:
    if not 0 < config.ranker_ratio < 1:
        raise ValueError("ranker_ratio must be between 0 and 1")
    if not 0 < config.test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    by_user: dict[int, list[RatingRow]] = defaultdict(list)
    for row in rows:
        by_user[row.user_id].append(row)

    train: list[TrainingRating] = []
    ranker_train: list[TrainingRating] = []
    test: list[TrainingRating] = []
    base_history: dict[int, set[int]] = defaultdict(set)
    evaluation_history: dict[int, set[int]] = defaultdict(set)
    ranker_relevant_by_user: dict[int, set[int]] = defaultdict(set)
    relevant_by_user: dict[int, set[int]] = defaultdict(set)
    test_by_user: dict[int, list[TrainingRating]] = defaultdict(list)

    for user_id, user_rows in by_user.items():
        ordered = sorted(user_rows, key=lambda row: (row.rating_date, row.movie_id))
        if len(ordered) < config.min_ratings_per_user:
            continue
        test_count = max(1, round(len(ordered) * config.test_ratio))
        ranker_count = max(1, round(len(ordered) * config.ranker_ratio))
        if len(ordered) - test_count - ranker_count < 1:
            continue

        test_start = len(ordered) - test_count
        ranker_start = test_start - ranker_count
        user_train = ordered[:ranker_start]
        user_ranker = ordered[ranker_start:test_start]
        user_test = ordered[test_start:]

        for row in user_train:
            item = TrainingRating(row.user_id, row.movie_id, row.rating)
            train.append(item)
            base_history[user_id].add(row.movie_id)
            evaluation_history[user_id].add(row.movie_id)
        for row in user_ranker:
            item = TrainingRating(row.user_id, row.movie_id, row.rating)
            ranker_train.append(item)
            evaluation_history[user_id].add(row.movie_id)
            if row.rating >= config.relevant_threshold:
                ranker_relevant_by_user[user_id].add(row.movie_id)
        for row in user_test:
            item = TrainingRating(row.user_id, row.movie_id, row.rating)
            test.append(item)
            test_by_user[user_id].append(item)
            if row.rating >= config.relevant_threshold:
                relevant_by_user[user_id].add(row.movie_id)

    if not train or not ranker_train or not test:
        raise ValueError("not enough ratings to build a three-way ranking split")

    return {
        "train": train,
        "ranker_train": ranker_train,
        "test": test,
        "base_history": dict(base_history),
        "train_history": dict(evaluation_history),
        "ranker_relevant_by_user": dict(ranker_relevant_by_user),
        "relevant_by_user": dict(relevant_by_user),
        "test_by_user": dict(test_by_user),
    }


def build_ranker_training_examples(
    model: MatrixFactorizationModel,
    split: dict,
    config: LearningToRankConfig,
    *,
    candidate_movie_ids: np.ndarray,
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    user_ids: list[int] = []
    rng = np.random.default_rng(config.seed)
    ranker_ratings = _ranker_ratings_by_user(split)

    for user_id, positives in split["ranker_relevant_by_user"].items():
        if not positives:
            continue
        exclude = split["base_history"].get(user_id, set())
        candidate_set = [int(movie_id) for movie_id in candidate_movie_ids if int(movie_id) not in exclude]
        if not candidate_set:
            continue

        user_features = features_for_user_candidates(
            model,
            user_id,
            np.array(candidate_set, dtype=np.int32),
            active_profile=indexes[0].get(user_id, {}),
            exclude_movie_ids=exclude,
            quality_scores=quality_scores,
            indexes=indexes,
            feature_stats=feature_stats,
        )
        candidate_index = {movie_id: index for index, movie_id in enumerate(candidate_set)}
        positive_ids = [movie_id for movie_id in positives if movie_id in candidate_index]
        if not positive_ids:
            continue
        for movie_id in positive_ids:
            features.append(user_features[candidate_index[movie_id]])
            labels.append(1)
            user_ids.append(int(user_id))

        negative_ids = _sample_ranker_negative_ids(
            candidate_set,
            candidate_index,
            set(positive_ids),
            ranker_ratings.get(user_id, {}),
            user_features,
            relevant_threshold=config.relevant_threshold,
            negatives_per_positive=config.negatives_per_positive,
            positive_count=len(positive_ids),
            strategy=config.negative_sampling,
            rng=rng,
        )
        for movie_id in negative_ids:
            features.append(user_features[candidate_index[movie_id]])
            labels.append(0)
            user_ids.append(int(user_id))

    if not features:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.vstack(features).astype(np.float32),
        np.array(labels, dtype=np.float32),
        np.array(user_ids, dtype=np.int64),
    )


def build_ranker_training_data(
    model: MatrixFactorizationModel,
    split: dict,
    config: LearningToRankConfig,
    *,
    candidate_movie_ids: np.ndarray,
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
) -> tuple[np.ndarray, np.ndarray]:
    features, labels, _user_ids = build_ranker_training_examples(
        model,
        split,
        config,
        candidate_movie_ids=candidate_movie_ids,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    return features, labels


def _ranker_ratings_by_user(split: dict) -> dict[int, dict[int, float]]:
    ratings: dict[int, dict[int, float]] = defaultdict(dict)
    for row in split.get("ranker_train", []):
        ratings[int(row.user_id)][int(row.movie_id)] = float(row.rating)
    return ratings


def _sample_ranker_negative_ids(
    candidate_set: list[int],
    candidate_index: dict[int, int],
    positive_ids: set[int],
    ranker_ratings: dict[int, float],
    user_features: np.ndarray,
    *,
    relevant_threshold: float,
    negatives_per_positive: int,
    positive_count: int,
    strategy: str,
    rng: np.random.Generator,
) -> list[int]:
    negative_ids = [movie_id for movie_id in candidate_set if movie_id not in positive_ids]
    if not negative_ids:
        return []

    if strategy == "hybrid_hard":
        negative_scores = user_features[[candidate_index[movie_id] for movie_id in negative_ids], :3].sum(axis=1)
        hard_order = np.argsort(-negative_scores)
        negative_limit = min(len(negative_ids), max(1, positive_count * negatives_per_positive))
        hard_negative_ids = [negative_ids[int(index)] for index in hard_order[:negative_limit]]
        random_limit = min(len(negative_ids), max(1, positive_count))
        random_negative_ids = list(rng.choice(negative_ids, size=random_limit, replace=False))
        return list(dict.fromkeys([*hard_negative_ids, *random_negative_ids]))

    if strategy != "explicit_hard":
        raise ValueError("negative_sampling must be 'hybrid_hard' or 'explicit_hard'")

    target = min(len(negative_ids), max(1, positive_count * negatives_per_positive))
    explicit_negative_ids = [
        movie_id
        for movie_id, rating in ranker_ratings.items()
        if rating < relevant_threshold and movie_id in candidate_index and movie_id not in positive_ids
    ]
    explicit_fallback_scores = _popularity_fallback_scores(user_features)
    explicit_negative_ids.sort(
        key=lambda movie_id: (
            ranker_ratings[movie_id],
            -explicit_fallback_scores[candidate_index[movie_id]],
        )
    )
    explicit_negative_ids = explicit_negative_ids[:target]

    explicit_set = set(explicit_negative_ids)
    fallback_pool = [movie_id for movie_id in negative_ids if movie_id not in explicit_set]
    fallback_limit = min(len(fallback_pool), max(target - len(explicit_negative_ids), 0))
    fallback_negative_ids = _sample_popularity_weighted_ids(
        fallback_pool,
        candidate_index,
        user_features,
        fallback_limit,
        rng,
    )
    return list(dict.fromkeys([*explicit_negative_ids, *fallback_negative_ids]))


def _hard_negative_scores(features: np.ndarray) -> np.ndarray:
    mf_index = FEATURE_NAMES.index("mf_score")
    collaborative_index = FEATURE_NAMES.index("collaborative_score")
    quality_index = FEATURE_NAMES.index("quality_score")
    recall_surfaced_index = FEATURE_NAMES.index("recall_surfaced")
    route_score_index = FEATURE_NAMES.index("recall_route_score")
    source_count_index = FEATURE_NAMES.index("recall_source_count")
    return (
        0.35 * features[:, mf_index]
        + 0.10 * features[:, collaborative_index]
        + 0.35 * features[:, quality_index]
        + 0.10 * features[:, route_score_index]
        + 0.05 * features[:, source_count_index]
        + 0.05 * features[:, recall_surfaced_index]
    )


def _sample_popularity_weighted_ids(
    movie_ids: list[int],
    candidate_index: dict[int, int],
    user_features: np.ndarray,
    limit: int,
    rng: np.random.Generator,
) -> list[int]:
    if limit <= 0 or not movie_ids:
        return []
    take = min(limit, len(movie_ids))
    fallback_scores = _popularity_fallback_scores(user_features)
    positions = np.fromiter(
        (candidate_index[movie_id] for movie_id in movie_ids),
        dtype=np.int64,
        count=len(movie_ids),
    )
    weights = np.maximum(fallback_scores[positions].astype(np.float64), 1e-6)
    total = float(weights.sum())
    probabilities = None if total <= 0 else weights / total
    return [
        int(movie_id)
        for movie_id in rng.choice(movie_ids, size=take, replace=False, p=probabilities)
    ]


def _popularity_fallback_scores(features: np.ndarray) -> np.ndarray:
    quality_index = FEATURE_NAMES.index("quality_score")
    movie_log_count_index = FEATURE_NAMES.index("movie_log_count")
    return 0.20 + 0.55 * features[:, quality_index] + 0.25 * features[:, movie_log_count_index]


def evaluate_learned_ranker(
    ranker: LinearRanker,
    model: MatrixFactorizationModel,
    split: dict,
    config: EvaluationConfig,
    *,
    candidate_movie_ids: np.ndarray,
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
    algorithm: str = "stacked_linear_hybrid_reranker",
) -> dict:
    return _evaluate_ranker(
        algorithm,
        split,
        config,
        candidate_movie_ids,
        lambda user_id, candidates, exclude: recommend_learned(
            ranker,
            model,
            user_id,
            candidates,
            exclude_movie_ids=exclude,
            active_profile=indexes[0].get(user_id, {}),
            quality_scores=quality_scores,
            indexes=indexes,
            feature_stats=feature_stats,
            top_k=config.top_k,
        ),
        backend=model.backend_used,
        device=model.device_used,
        rmse="",
        users=len(model.user_to_index),
        movies=len(model.movie_to_index),
    )


def evaluate_residual_ranker(
    ranker: ResidualLinearRanker,
    base_ranker: LinearRanker,
    model: MatrixFactorizationModel,
    split: dict,
    config: EvaluationConfig,
    *,
    candidate_movie_ids: np.ndarray,
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
) -> dict:
    metrics = _evaluate_ranker(
        "residual_hybrid_linear_reranker",
        split,
        config,
        candidate_movie_ids,
        lambda user_id, candidates, exclude: recommend_residual(
            ranker,
            base_ranker,
            model,
            user_id,
            candidates,
            exclude_movie_ids=exclude,
            active_profile=indexes[0].get(user_id, {}),
            quality_scores=quality_scores,
            indexes=indexes,
            feature_stats=feature_stats,
            top_k=config.top_k,
        ),
        backend=model.backend_used,
        device=model.device_used,
        rmse="",
        users=len(model.user_to_index),
        movies=len(model.movie_to_index),
    )
    metrics["residual_alpha"] = round(float(ranker.residual_alpha), 6)
    return metrics


def evaluate_mmr_learned_ranker(
    ranker: LinearRanker,
    model: MatrixFactorizationModel,
    split: dict,
    config: EvaluationConfig,
    *,
    candidate_movie_ids: np.ndarray,
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
    diversity_weight: float = 0.15,
    pool_multiplier: int = 20,
) -> dict:
    return _evaluate_ranker(
        "mmr_stacked_linear_hybrid_reranker",
        split,
        config,
        candidate_movie_ids,
        lambda user_id, candidates, exclude: recommend_learned_mmr(
            ranker,
            model,
            user_id,
            candidates,
            exclude_movie_ids=exclude,
            active_profile=indexes[0].get(user_id, {}),
            quality_scores=quality_scores,
            indexes=indexes,
            feature_stats=feature_stats,
            top_k=config.top_k,
            diversity_weight=diversity_weight,
            pool_multiplier=pool_multiplier,
        ),
        backend=model.backend_used,
        device=model.device_used,
        rmse="",
        users=len(model.user_to_index),
        movies=len(model.movie_to_index),
    )


def evaluate_hybrid_ranker_from_features(
    model: MatrixFactorizationModel,
    split: dict,
    config: EvaluationConfig,
    *,
    candidate_movie_ids: np.ndarray,
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
) -> dict:
    metrics = _evaluate_ranker(
        "hybrid_mf_user_user_quality",
        split,
        config,
        candidate_movie_ids,
        lambda user_id, candidates, exclude: recommend_feature_hybrid(
            model,
            user_id,
            candidates,
            exclude_movie_ids=exclude,
            active_profile=indexes[0].get(user_id, {}),
            quality_scores=quality_scores,
            indexes=indexes,
            feature_stats=feature_stats,
            top_k=config.top_k,
        ),
        backend=model.backend_used,
        device=model.device_used,
        rmse="",
        users=len(model.user_to_index),
        movies=len(model.movie_to_index),
    )
    metrics["mf_weight"] = HYBRID_MF_WEIGHT
    metrics["collaborative_weight"] = HYBRID_COLLABORATIVE_WEIGHT
    metrics["quality_weight"] = HYBRID_QUALITY_WEIGHT
    return metrics


def recommend_feature_hybrid(
    model: MatrixFactorizationModel,
    user_id: int,
    candidate_movie_ids: np.ndarray,
    *,
    exclude_movie_ids: set[int],
    active_profile: dict[int, float],
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
    top_k: int,
) -> list[tuple[int, float]]:
    eligible = _eligible_candidate_ids(candidate_movie_ids, exclude_movie_ids)
    if len(eligible) == 0:
        return []
    features = features_for_user_candidates(
        model,
        user_id,
        eligible,
        active_profile=active_profile,
        exclude_movie_ids=exclude_movie_ids,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    scores = _hybrid_scores_from_features(features)
    take = min(top_k, len(eligible))
    top_positions = np.argpartition(-scores, take - 1)[:take]
    ranked_positions = top_positions[np.argsort(-scores[top_positions])]
    return [(int(eligible[position]), float(scores[position])) for position in ranked_positions]


def recommend_learned(
    ranker: LinearRanker,
    model: MatrixFactorizationModel,
    user_id: int,
    candidate_movie_ids: np.ndarray,
    *,
    exclude_movie_ids: set[int],
    active_profile: dict[int, float],
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
    top_k: int,
) -> list[tuple[int, float]]:
    eligible = _eligible_candidate_ids(candidate_movie_ids, exclude_movie_ids)
    if len(eligible) == 0:
        return []
    features = features_for_user_candidates(
        model,
        user_id,
        eligible,
        active_profile=active_profile,
        exclude_movie_ids=exclude_movie_ids,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    scores = _blended_scores_from_features(ranker, features)
    take = min(top_k, len(eligible))
    top_positions = np.argpartition(-scores, take - 1)[:take]
    ranked_positions = top_positions[np.argsort(-scores[top_positions])]
    return [(int(eligible[position]), float(scores[position])) for position in ranked_positions]


def recommend_residual(
    ranker: ResidualLinearRanker,
    base_ranker: LinearRanker,
    model: MatrixFactorizationModel,
    user_id: int,
    candidate_movie_ids: np.ndarray,
    *,
    exclude_movie_ids: set[int],
    active_profile: dict[int, float],
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
    top_k: int,
) -> list[tuple[int, float]]:
    eligible = _eligible_candidate_ids(candidate_movie_ids, exclude_movie_ids)
    if len(eligible) == 0:
        return []
    features = features_for_user_candidates(
        model,
        user_id,
        eligible,
        active_profile=active_profile,
        exclude_movie_ids=exclude_movie_ids,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    scores = _residual_scores_from_features(ranker, features, base_ranker=base_ranker)
    take = min(top_k, len(eligible))
    top_positions = np.argpartition(-scores, take - 1)[:take]
    ranked_positions = top_positions[np.argsort(-scores[top_positions])]
    return [(int(eligible[position]), float(scores[position])) for position in ranked_positions]


def recommend_learned_mmr(
    ranker: LinearRanker,
    model: MatrixFactorizationModel,
    user_id: int,
    candidate_movie_ids: np.ndarray,
    *,
    exclude_movie_ids: set[int],
    active_profile: dict[int, float],
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
    top_k: int,
    diversity_weight: float,
    pool_multiplier: int,
) -> list[tuple[int, float]]:
    eligible = _eligible_candidate_ids(candidate_movie_ids, exclude_movie_ids)
    if len(eligible) == 0:
        return []
    features = features_for_user_candidates(
        model,
        user_id,
        eligible,
        active_profile=active_profile,
        exclude_movie_ids=exclude_movie_ids,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    scores = _blended_scores_from_features(ranker, features)
    selected = _mmr_select(
        model,
        eligible,
        scores,
        top_k=top_k,
        diversity_weight=diversity_weight,
        pool_size=max(top_k, top_k * max(pool_multiplier, 1)),
    )
    return [(int(eligible[position]), float(scores[position])) for position in selected]


def _mmr_select(
    model: MatrixFactorizationModel,
    movie_ids: np.ndarray,
    scores: np.ndarray,
    *,
    top_k: int,
    diversity_weight: float,
    pool_size: int,
) -> list[int]:
    if top_k <= 0 or len(movie_ids) == 0:
        return []
    take = min(len(movie_ids), max(top_k, pool_size))
    top_positions = np.argpartition(-scores, take - 1)[:take]
    pool_positions = top_positions[np.argsort(-scores[top_positions])]
    if len(pool_positions) <= top_k or diversity_weight <= 0:
        return [int(position) for position in pool_positions[:top_k]]

    vectors = _candidate_vectors(model, movie_ids[pool_positions])
    if vectors is None:
        return [int(position) for position in pool_positions[:top_k]]

    selected_pool_positions: list[int] = []
    remaining = list(range(len(pool_positions)))
    relevance = _normalize_array(scores[pool_positions])
    while remaining and len(selected_pool_positions) < top_k:
        if not selected_pool_positions:
            best_remaining = max(remaining, key=lambda position: (relevance[position], -position))
        else:
            selected_vectors = vectors[selected_pool_positions]
            similarities = vectors[remaining] @ selected_vectors.T
            max_similarity = similarities.max(axis=1) if similarities.size else np.zeros(len(remaining))
            mmr_scores = (1.0 - diversity_weight) * relevance[remaining] - diversity_weight * max_similarity
            best_offset = int(np.argmax(mmr_scores))
            best_remaining = remaining[best_offset]
        selected_pool_positions.append(best_remaining)
        remaining.remove(best_remaining)
    return [int(pool_positions[position]) for position in selected_pool_positions]


def _candidate_vectors(model: MatrixFactorizationModel, movie_ids: np.ndarray) -> np.ndarray | None:
    if not model.movie_factors.size:
        return None
    movie_indices = [model.movie_to_index.get(int(movie_id)) for movie_id in movie_ids]
    if any(index is None for index in movie_indices):
        return None
    vectors = model.movie_factors[np.array(movie_indices, dtype=np.int32)].astype(np.float32, copy=True)
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 0
    if not valid.all():
        return None
    vectors /= norms[:, None]
    return vectors


def _eligible_candidate_ids(candidate_movie_ids: np.ndarray, exclude_movie_ids: set[int]) -> np.ndarray:
    candidate_movie_ids = candidate_movie_ids.astype(np.int32, copy=False)
    if not exclude_movie_ids:
        return candidate_movie_ids
    exclude_array = np.fromiter(exclude_movie_ids, dtype=np.int64, count=len(exclude_movie_ids))
    keep_mask = ~np.isin(candidate_movie_ids.astype(np.int64, copy=False), exclude_array)
    return candidate_movie_ids[keep_mask]


def tune_blend_weight(
    ranker: LinearRanker,
    model: MatrixFactorizationModel,
    split: dict,
    config: LearningToRankConfig,
    *,
    candidate_movie_ids: np.ndarray,
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
) -> tuple[float, list[dict]]:
    tuning_split = {
        "train": split["train"],
        "test": split["ranker_train"],
        "train_history": split["base_history"],
        "relevant_by_user": split["ranker_relevant_by_user"],
        "test_by_user": {},
    }
    rows = []
    original_blend = ranker.blend_weight
    original_mode = ranker.blend_mode
    blend_modes = _blend_modes(config)
    score_records = _precompute_linear_blend_scores(
        ranker,
        model,
        tuning_split,
        candidate_movie_ids=candidate_movie_ids,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
        include_rank_scores="rank" in blend_modes,
    )
    for blend_mode in blend_modes:
        for blend_weight in _blend_weight_grid():
            metrics = _evaluate_precomputed_blend_scores(
                "stacked_linear_hybrid_reranker",
                score_records,
                tuning_split,
                _evaluation_config(config),
                blend_weight=blend_weight,
                blend_mode=blend_mode,
                backend=model.backend_used,
                device=model.device_used,
                users=len(model.user_to_index),
                movies=len(model.movie_to_index),
                candidate_count=len(candidate_movie_ids),
            )
            rows.append({"linear_blend_weight": blend_weight, "blend_mode": blend_mode, **metrics})
    ranker.blend_weight = original_blend
    ranker.blend_mode = original_mode
    best = max(
        rows,
        key=lambda row: (
            row["precision_at_k"],
            row["hit_rate_at_k"],
            row["map_at_k"],
            row["catalog_coverage"],
        ),
    )
    baseline = rows[0]
    if (
        float(best["precision_at_k"])
        < float(baseline["precision_at_k"]) + config.min_blend_precision_gain
        or float(best["hit_rate_at_k"]) < float(baseline["hit_rate_at_k"])
    ):
        return 0.0, rows
    ranker.blend_mode = str(best["blend_mode"])
    return float(best["linear_blend_weight"]), rows


def tune_residual_alpha(
    ranker: ResidualLinearRanker,
    base_ranker: LinearRanker,
    model: MatrixFactorizationModel,
    split: dict,
    config: LearningToRankConfig,
    *,
    candidate_movie_ids: np.ndarray,
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
) -> tuple[float, list[dict]]:
    tuning_split = {
        "train": split["train"],
        "test": split["ranker_train"],
        "train_history": split["base_history"],
        "relevant_by_user": split["ranker_relevant_by_user"],
        "test_by_user": {},
    }
    records = _precompute_residual_scores(
        ranker,
        base_ranker,
        model,
        tuning_split,
        candidate_movie_ids=candidate_movie_ids,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    rows = []
    for residual_alpha in _residual_alpha_grid():
        metrics = _evaluate_precomputed_residual_scores(
            records,
            tuning_split,
            _evaluation_config(config),
            residual_alpha=residual_alpha,
            backend=model.backend_used,
            device=model.device_used,
            users=len(model.user_to_index),
            movies=len(model.movie_to_index),
            candidate_count=len(candidate_movie_ids),
        )
        rows.append({"residual_alpha": residual_alpha, **metrics})
    best = max(
        rows,
        key=lambda row: (
            row["precision_at_k"],
            row["hit_rate_at_k"],
            row["map_at_k"],
            row["catalog_coverage"],
        ),
    )
    baseline = rows[0]
    if (
        float(best["precision_at_k"])
        < float(baseline["precision_at_k"]) + config.min_blend_precision_gain
        or float(best["hit_rate_at_k"]) < float(baseline["hit_rate_at_k"])
    ):
        return 0.0, rows
    return float(best["residual_alpha"]), rows


def _precompute_linear_blend_scores(
    ranker: LinearRanker,
    model: MatrixFactorizationModel,
    split: dict,
    *,
    candidate_movie_ids: np.ndarray,
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
    include_rank_scores: bool = False,
) -> list[dict]:
    records = []
    for user_id, relevant_movies in split["relevant_by_user"].items():
        if not relevant_movies:
            continue
        exclude = split["train_history"].get(user_id, set())
        eligible = np.array(
            [int(movie_id) for movie_id in candidate_movie_ids if int(movie_id) not in exclude],
            dtype=np.int32,
        )
        if len(eligible) == 0:
            records.append({"relevant_movies": relevant_movies, "movie_ids": np.empty(0, dtype=np.int32)})
            continue
        features = features_for_user_candidates(
            model,
            user_id,
            eligible,
            active_profile=indexes[0].get(user_id, {}),
            exclude_movie_ids=exclude,
            quality_scores=quality_scores,
            indexes=indexes,
            feature_stats=feature_stats,
        )
        learned_scores = _normalize_array(ranker.score(features))
        hybrid_scores = _hybrid_scores_from_features(features)
        record = {
            "relevant_movies": relevant_movies,
            "movie_ids": eligible,
            "learned_scores": learned_scores,
            "hybrid_scores": hybrid_scores,
        }
        if include_rank_scores:
            record["learned_rank_scores"] = _rank_percentiles(learned_scores)
            record["hybrid_rank_scores"] = _rank_percentiles(hybrid_scores)
        records.append(record)
    return records


def _precompute_residual_scores(
    ranker: ResidualLinearRanker,
    base_ranker: LinearRanker,
    model: MatrixFactorizationModel,
    split: dict,
    *,
    candidate_movie_ids: np.ndarray,
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
) -> list[dict]:
    records = []
    for user_id, relevant_movies in split["relevant_by_user"].items():
        if not relevant_movies:
            continue
        exclude = split["train_history"].get(user_id, set())
        eligible = np.array(
            [int(movie_id) for movie_id in candidate_movie_ids if int(movie_id) not in exclude],
            dtype=np.int32,
        )
        if len(eligible) == 0:
            records.append({"relevant_movies": relevant_movies, "movie_ids": np.empty(0, dtype=np.int32)})
            continue
        features = features_for_user_candidates(
            model,
            user_id,
            eligible,
            active_profile=indexes[0].get(user_id, {}),
            exclude_movie_ids=exclude,
            quality_scores=quality_scores,
            indexes=indexes,
            feature_stats=feature_stats,
        )
        records.append(
            {
                "relevant_movies": relevant_movies,
                "movie_ids": eligible,
                "base_scores": _blended_scores_from_features(base_ranker, features),
                "residual_scores": ranker.score(features).astype(np.float32),
            }
        )
    return records


def _evaluate_precomputed_blend_scores(
    algorithm: str,
    records: list[dict],
    split: dict,
    config: EvaluationConfig,
    *,
    blend_weight: float,
    blend_mode: str,
    backend: str,
    device: str,
    users: int,
    movies: int,
    candidate_count: int,
) -> dict:
    precision_sum = 0.0
    recall_sum = 0.0
    hit_count = 0
    average_precision_sum = 0.0
    evaluated_users = 0
    empty_users = 0
    recommended_movies: set[int] = set()

    for record in records:
        movie_ids = record["movie_ids"]
        relevant_movies = record["relevant_movies"]
        if len(movie_ids) == 0:
            empty_users += 1
            continue
        if blend_mode == "rank" and "learned_rank_scores" in record and "hybrid_rank_scores" in record:
            scores = (
                blend_weight * record["learned_rank_scores"]
                + (1.0 - blend_weight) * record["hybrid_rank_scores"]
            )
        else:
            scores = _blend_scores(
                record["learned_scores"],
                record["hybrid_scores"],
                blend_weight,
                mode=blend_mode,
            )
        take = min(config.top_k, len(movie_ids))
        top_positions = np.argpartition(-scores, take - 1)[:take]
        ranked_positions = top_positions[np.argsort(-scores[top_positions])]
        recommended = [int(movie_ids[position]) for position in ranked_positions]
        evaluated_users += 1
        recommended_movies.update(recommended)
        hits = [movie_id for movie_id in recommended if movie_id in relevant_movies]
        precision_sum += len(hits) / config.top_k
        recall_sum += len(hits) / len(relevant_movies)
        hit_count += 1 if hits else 0
        average_precision_sum += _average_precision_at_k(recommended, relevant_movies, config.top_k)

    return {
        "algorithm": algorithm,
        "backend": backend,
        "device": device,
        "rmse": "",
        "precision_at_k": round(precision_sum / evaluated_users, 6) if evaluated_users else 0.0,
        "recall_at_k": round(recall_sum / evaluated_users, 6) if evaluated_users else 0.0,
        "hit_rate_at_k": round(hit_count / evaluated_users, 6) if evaluated_users else 0.0,
        "map_at_k": round(average_precision_sum / evaluated_users, 6) if evaluated_users else 0.0,
        "catalog_coverage": round(len(recommended_movies) / candidate_count, 6) if candidate_count else 0.0,
        "top_k": config.top_k,
        "users": users,
        "movies": movies,
        "train_ratings": len(split["train"]),
        "test_ratings": len(split["test"]),
        "evaluated_users": evaluated_users,
        "candidate_movies": candidate_count,
        "relevant_threshold": config.relevant_threshold,
        "empty_users": empty_users,
    }


def _evaluate_precomputed_residual_scores(
    records: list[dict],
    split: dict,
    config: EvaluationConfig,
    *,
    residual_alpha: float,
    backend: str,
    device: str,
    users: int,
    movies: int,
    candidate_count: int,
) -> dict:
    precision_sum = 0.0
    recall_sum = 0.0
    hit_count = 0
    average_precision_sum = 0.0
    evaluated_users = 0
    empty_users = 0
    recommended_movies: set[int] = set()

    for record in records:
        movie_ids = record["movie_ids"]
        relevant_movies = record["relevant_movies"]
        if len(movie_ids) == 0:
            empty_users += 1
            continue
        scores = np.clip(
            record["base_scores"] + residual_alpha * record["residual_scores"],
            0.0,
            1.0,
        )
        take = min(config.top_k, len(movie_ids))
        top_positions = np.argpartition(-scores, take - 1)[:take]
        ranked_positions = top_positions[np.argsort(-scores[top_positions])]
        recommended = [int(movie_ids[position]) for position in ranked_positions]
        evaluated_users += 1
        recommended_movies.update(recommended)
        hits = [movie_id for movie_id in recommended if movie_id in relevant_movies]
        precision_sum += len(hits) / config.top_k
        recall_sum += len(hits) / len(relevant_movies)
        hit_count += 1 if hits else 0
        average_precision_sum += _average_precision_at_k(recommended, relevant_movies, config.top_k)

    return {
        "algorithm": "residual_hybrid_linear_reranker",
        "backend": backend,
        "device": device,
        "rmse": "",
        "precision_at_k": round(precision_sum / evaluated_users, 6) if evaluated_users else 0.0,
        "recall_at_k": round(recall_sum / evaluated_users, 6) if evaluated_users else 0.0,
        "hit_rate_at_k": round(hit_count / evaluated_users, 6) if evaluated_users else 0.0,
        "map_at_k": round(average_precision_sum / evaluated_users, 6) if evaluated_users else 0.0,
        "catalog_coverage": round(len(recommended_movies) / candidate_count, 6) if candidate_count else 0.0,
        "top_k": config.top_k,
        "users": users,
        "movies": movies,
        "train_ratings": len(split["train"]),
        "test_ratings": len(split["test"]),
        "evaluated_users": evaluated_users,
        "candidate_movies": candidate_count,
        "relevant_threshold": config.relevant_threshold,
        "empty_users": empty_users,
    }


def _blend_weight_grid() -> tuple[float, ...]:
    return tuple(round(index * 0.05, 2) for index in range(21))


def _residual_alpha_grid() -> tuple[float, ...]:
    return tuple(round(index * 0.05, 2) for index in range(7))


def _blend_modes(config: LearningToRankConfig) -> tuple[str, ...]:
    if config.blend_mode == "auto":
        return ("value", "rank")
    if config.blend_mode not in {"value", "rank"}:
        raise ValueError("blend_mode must be 'value', 'rank', or 'auto'")
    return (config.blend_mode,)


def _blend_scores(learned_scores: np.ndarray, hybrid_scores: np.ndarray, blend_weight: float, *, mode: str) -> np.ndarray:
    if mode == "value":
        calibrated_learned = learned_scores
        calibrated_hybrid = hybrid_scores
    elif mode == "rank":
        calibrated_learned = _rank_percentiles(learned_scores)
        calibrated_hybrid = _rank_percentiles(hybrid_scores)
    else:
        raise ValueError("blend mode must be 'value' or 'rank'")
    return blend_weight * calibrated_learned + (1.0 - blend_weight) * calibrated_hybrid


def _rank_percentiles(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores.astype(np.float32)
    if len(scores) == 1:
        return np.ones(1, dtype=np.float32)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(len(scores), dtype=np.float32)
    return ranks / float(len(scores) - 1)


def _normalize_cf_from_arrays(
    entry: tuple[np.ndarray, np.ndarray],
    movie_ids: np.ndarray,
    exclude_movie_ids: set[int],
) -> np.ndarray:
    """Vectorized equivalent of ``_normalize_score_dict(_score_user_user_candidates(...))``.

    ``entry`` is the per-user ``(ascending movie_ids, raw scores)`` from the CF
    engine. The candidate column is gathered with ``searchsorted`` (no per-movie
    ``dict.get``) and min-max normalized over the present candidates, matching the
    legacy dict path bit-for-bit.
    """
    column = np.zeros(len(movie_ids), dtype=np.float32)
    user_movie_ids, user_values = entry
    if len(user_movie_ids) == 0 or len(movie_ids) == 0:
        return column

    mids = movie_ids.astype(np.int64, copy=False)
    positions = np.minimum(np.searchsorted(user_movie_ids, mids), len(user_movie_ids) - 1)
    present = user_movie_ids[positions] == mids
    if exclude_movie_ids:
        exclude_array = np.fromiter(exclude_movie_ids, dtype=np.int64, count=len(exclude_movie_ids))
        present &= ~np.isin(mids, exclude_array)
    if not present.any():
        return column

    raw_values = user_values[positions]
    present_values = raw_values[present]
    min_value = float(present_values.min())
    max_value = float(present_values.max())
    if max_value <= min_value:
        column[present] = 1.0
    else:
        column[present] = ((raw_values[present] - min_value) / (max_value - min_value)).astype(np.float32)
    return column


def _collaborative_score_column(
    user_id: int,
    candidate_movie_ids: np.ndarray,
    movie_ids: np.ndarray,
    exclude_movie_ids: set[int],
    *,
    active_profile: dict[int, float],
    indexes,
    feature_stats: dict,
) -> np.ndarray:
    """Collaborative-score feature column, vectorized when the array cache is primed.

    Falls back to the legacy dict path for any user not covered by the array cache
    so coverage/values are unchanged.
    """
    array_cache = feature_stats.get("collaborative_raw_array_cache")
    if array_cache is not None and user_id in array_cache:
        return _normalize_cf_from_arrays(array_cache[user_id], movie_ids, exclude_movie_ids)

    collaborative_scores = _normalize_score_dict(
        _score_user_user_candidates(
            user_id,
            active_profile,
            indexes=indexes,
            candidate_movie_ids=candidate_movie_ids,
            exclude_movie_ids=exclude_movie_ids,
            score_cache=feature_stats.setdefault("collaborative_score_cache", {})
            if feature_stats.get("cache_user_candidate_features")
            else None,
            neighbor_cache=feature_stats.setdefault("user_neighbor_cache", {}),
            raw_score_cache=feature_stats.setdefault("collaborative_raw_cache", {}),
        )
    )
    return np.array(
        [float(collaborative_scores.get(int(movie_id), 0.0)) for movie_id in movie_ids],
        dtype=np.float32,
    )


def _profile_similarity_column(
    centroid: np.ndarray | None,
    centroid_norm: float,
    model: MatrixFactorizationModel,
    lookup: dict[str, np.ndarray],
    lookup_indices: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(lookup_indices), dtype=np.float32)
    if centroid is None or centroid_norm <= 0 or not model.movie_factors.size:
        return result

    movie_indices = lookup["movie_factor_index"][lookup_indices]
    valid = movie_indices >= 0
    if not valid.any():
        return result

    vectors = model.movie_factors[movie_indices[valid]]
    norms = np.linalg.norm(vectors, axis=1)
    valid_norms = norms > 0
    if valid_norms.any():
        similarities = np.zeros(len(vectors), dtype=np.float32)
        similarities[valid_norms] = (
            vectors[valid_norms] @ centroid
        ) / np.maximum(norms[valid_norms] * float(centroid_norm), 1e-6)
        result[valid] = (similarities + 1.0) / 2.0
    return result


def _item_item_score_column(
    active_profile: dict[int, float],
    movie_ids: np.ndarray,
    exclude_movie_ids: set[int],
    feature_stats: dict,
) -> np.ndarray:
    neighbors = feature_stats.get("item_item_neighbors", {})
    if not neighbors or len(movie_ids) == 0:
        return np.zeros(len(movie_ids), dtype=np.float32)

    positions = {int(movie_id): index for index, movie_id in enumerate(movie_ids)}
    values = np.zeros(len(movie_ids), dtype=np.float32)
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
            position = positions.get(movie_id)
            if position is not None:
                values[position] += seed_weight * float(similarity)

    if seed_count > 1:
        values /= seed_count ** 0.5
    return _normalize_array(values.astype(np.float32))


def features_for_user_candidates(
    model: MatrixFactorizationModel,
    user_id: int,
    candidate_movie_ids: np.ndarray,
    *,
    active_profile: dict[int, float],
    exclude_movie_ids: set[int],
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
) -> np.ndarray:
    cache_key = _feature_cache_key(user_id, candidate_movie_ids, exclude_movie_ids)
    if feature_stats.get("cache_user_candidate_features"):
        cache = feature_stats.setdefault("user_candidate_feature_cache", {})
        cached = cache.get(cache_key)
        if cached is not None:
            return cached.copy()

    mf_raw = model.score_known_user(user_id, candidate_movie_ids)
    mf_scores = (mf_raw - model.config.min_rating) / (model.config.max_rating - model.config.min_rating)
    movie_stats = feature_stats["movie"]
    user_stats = feature_stats["user"]
    user_positive_counts = feature_stats.get("user_positive_counts", {})
    movie_years = feature_stats.get("movie_year", {})
    user_positive_year_avg = feature_stats.get("user_positive_year_avg", {})
    profile_centroids = feature_stats.get("profile_centroids", {})
    profile_norms = feature_stats.get("profile_norms", {})
    negative_profile_centroids = feature_stats.get("negative_profile_centroids", {})
    negative_profile_norms = feature_stats.get("negative_profile_norms", {})
    max_movie_log_count = feature_stats["max_movie_log_count"]
    max_user_log_count = feature_stats["max_user_log_count"]
    min_year = feature_stats.get("min_year", 1900)
    year_span = feature_stats.get("year_span", 125)
    user_avg, user_count = user_stats.get(user_id, (model.global_mean, 0))
    user_log_count = math.log1p(user_count) / max_user_log_count if max_user_log_count else 0.0
    user_positive_rate = user_positive_counts.get(user_id, 0) / user_count if user_count else 0.0
    profile_centroid = profile_centroids.get(user_id)
    profile_norm = profile_norms.get(user_id, 0.0)
    negative_profile_centroid = negative_profile_centroids.get(user_id)
    negative_profile_norm = negative_profile_norms.get(user_id, 0.0)
    user_year_avg = user_positive_year_avg.get(user_id)

    rows = np.zeros((len(candidate_movie_ids), len(FEATURE_NAMES)), dtype=np.float32)
    lookup = _feature_lookup_arrays(feature_stats, model, quality_scores)
    movie_ids = candidate_movie_ids.astype(np.int64, copy=False)
    lookup_indices = np.where((movie_ids >= 0) & (movie_ids < len(lookup["movie_avg_norm"])), movie_ids, 0)
    movie_avgs = lookup["movie_avg_norm"][lookup_indices]
    movie_log_counts = lookup["movie_log_count"][lookup_indices]
    movie_year_values = lookup["movie_year_value"][lookup_indices]
    movie_has_year = lookup["movie_has_year"][lookup_indices]
    movie_year_norm = np.where(movie_has_year > 0, (movie_year_values - min_year) / year_span, 0.0).astype(np.float32)
    if user_year_avg is not None:
        year_affinity = np.where(
            movie_has_year > 0,
            np.exp(-np.abs(movie_year_values - float(user_year_avg)) / 18.0),
            0.0,
        ).astype(np.float32)
    else:
        year_affinity = np.zeros(len(candidate_movie_ids), dtype=np.float32)

    experimental_features = bool(feature_stats.get("enable_experimental_algorithm_features", True))
    profile_similarity = _profile_similarity_column(
        profile_centroid,
        float(profile_norm),
        model,
        lookup,
        lookup_indices,
    )
    if experimental_features:
        negative_profile_similarity = _profile_similarity_column(
            negative_profile_centroid,
            float(negative_profile_norm),
            model,
            lookup,
            lookup_indices,
        )
        signed_profile_similarity = np.clip(
            0.5 + 0.5 * (profile_similarity - negative_profile_similarity),
            0.0,
            1.0,
        ).astype(np.float32)
        item_item_scores = _item_item_score_column(
            active_profile,
            movie_ids,
            exclude_movie_ids,
            feature_stats,
        )
    else:
        negative_profile_similarity = np.zeros(len(candidate_movie_ids), dtype=np.float32)
        signed_profile_similarity = np.zeros(len(candidate_movie_ids), dtype=np.float32)
        item_item_scores = np.zeros(len(candidate_movie_ids), dtype=np.float32)

    profile_strength = min(user_positive_counts.get(user_id, 0) / 20.0, 1.0)
    recall_arrays = {
        name: np.array(
            lookup[name][lookup_indices],
            dtype=np.float32,
        )
        for name in (
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
    }

    rows[:, FEATURE_NAMES.index("mf_score")] = mf_scores.astype(np.float32)
    rows[:, FEATURE_NAMES.index("collaborative_score")] = _collaborative_score_column(
        user_id,
        candidate_movie_ids,
        movie_ids,
        exclude_movie_ids,
        active_profile=active_profile,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    rows[:, FEATURE_NAMES.index("quality_score")] = lookup["quality_score"][lookup_indices]
    rows[:, FEATURE_NAMES.index("movie_avg_rating")] = movie_avgs
    rows[:, FEATURE_NAMES.index("movie_log_count")] = movie_log_counts
    rows[:, FEATURE_NAMES.index("user_avg_rating")] = float(user_avg / 5.0)
    rows[:, FEATURE_NAMES.index("user_log_count")] = float(user_log_count)
    rows[:, FEATURE_NAMES.index("profile_similarity")] = profile_similarity
    rows[:, FEATURE_NAMES.index("negative_profile_similarity")] = negative_profile_similarity
    rows[:, FEATURE_NAMES.index("signed_profile_similarity")] = signed_profile_similarity
    rows[:, FEATURE_NAMES.index("item_item_score")] = item_item_scores
    rows[:, FEATURE_NAMES.index("profile_strength")] = float(profile_strength)
    rows[:, FEATURE_NAMES.index("user_positive_rate")] = float(user_positive_rate)
    rows[:, FEATURE_NAMES.index("movie_year_norm")] = movie_year_norm
    rows[:, FEATURE_NAMES.index("movie_has_year")] = movie_has_year
    rows[:, FEATURE_NAMES.index("year_affinity")] = year_affinity
    rows[:, FEATURE_NAMES.index("recall_surfaced")] = recall_arrays["recall_surfaced"]
    rows[:, FEATURE_NAMES.index("recall_route_score")] = recall_arrays["recall_route_score"]
    rows[:, FEATURE_NAMES.index("recall_source_count")] = recall_arrays["recall_source_count"]
    rows[:, FEATURE_NAMES.index("recall_popular_quality")] = recall_arrays["recall_popular_quality"]
    rows[:, FEATURE_NAMES.index("recall_mf_user_top")] = recall_arrays["recall_mf_user_top"]
    rows[:, FEATURE_NAMES.index("recall_profile_centroid")] = recall_arrays["recall_profile_centroid"]
    rows[:, FEATURE_NAMES.index("recall_item_item_cf")] = recall_arrays["recall_item_item_cf"]
    rows[:, FEATURE_NAMES.index("recall_user_user_cf")] = recall_arrays["recall_user_user_cf"]
    rows[:, FEATURE_NAMES.index("recall_year_affinity")] = recall_arrays["recall_year_affinity"]
    if feature_stats.get("cache_user_candidate_features"):
        feature_stats.setdefault("user_candidate_feature_cache", {})[cache_key] = rows.copy()
    return rows


def _feature_cache_key(user_id: int, candidate_movie_ids: np.ndarray, exclude_movie_ids: set[int]) -> tuple:
    if len(candidate_movie_ids) == 0:
        candidate_fingerprint = (0, 0, 0, 0)
    else:
        candidate_fingerprint = (
            int(len(candidate_movie_ids)),
            int(candidate_movie_ids[0]),
            int(candidate_movie_ids[-1]),
            int(candidate_movie_ids.astype(np.int64, copy=False).sum()),
        )
    exclude_fingerprint = (
        len(exclude_movie_ids),
        int(sum(exclude_movie_ids)) if exclude_movie_ids else 0,
    )
    return (int(user_id), candidate_fingerprint, exclude_fingerprint)


def _feature_lookup_arrays(
    feature_stats: dict,
    model: MatrixFactorizationModel,
    quality_scores: dict[int, float],
) -> dict[str, np.ndarray]:
    cached = feature_stats.get("movie_feature_lookup_arrays")
    if cached is not None:
        return cached

    movie_stats = feature_stats["movie"]
    movie_years = feature_stats.get("movie_year", {})
    recall_features = feature_stats.get("candidate_recall_features", {})
    max_movie_id = max(
        [
            0,
            *movie_stats.keys(),
            *movie_years.keys(),
            *quality_scores.keys(),
            *model.movie_to_index.keys(),
            *recall_features.keys(),
        ]
    )
    size = int(max_movie_id) + 1
    max_movie_log_count = feature_stats["max_movie_log_count"]
    arrays: dict[str, np.ndarray] = {
        "movie_avg_norm": np.full(size, float(model.global_mean / 5.0), dtype=np.float32),
        "movie_log_count": np.zeros(size, dtype=np.float32),
        "movie_year_value": np.zeros(size, dtype=np.float32),
        "movie_has_year": np.zeros(size, dtype=np.float32),
        "movie_factor_index": np.full(size, -1, dtype=np.int32),
        "quality_score": np.zeros(size, dtype=np.float32),
    }
    for movie_id, (movie_avg, movie_count) in movie_stats.items():
        arrays["movie_avg_norm"][movie_id] = float(movie_avg / 5.0)
        arrays["movie_log_count"][movie_id] = (
            float(math.log1p(movie_count) / max_movie_log_count)
            if max_movie_log_count
            else 0.0
        )
    for movie_id, year in movie_years.items():
        if year is None:
            continue
        arrays["movie_year_value"][movie_id] = float(year)
        arrays["movie_has_year"][movie_id] = 1.0
    for movie_id, movie_index in model.movie_to_index.items():
        arrays["movie_factor_index"][movie_id] = int(movie_index)
    for movie_id, score in quality_scores.items():
        arrays["quality_score"][movie_id] = float(score)
    for name in (
        "recall_surfaced",
        "recall_route_score",
        "recall_source_count",
        "recall_popular_quality",
        "recall_mf_user_top",
        "recall_profile_centroid",
        "recall_item_item_cf",
        "recall_user_user_cf",
        "recall_year_affinity",
    ):
        values = np.zeros(size, dtype=np.float32)
        for movie_id, feature_values in recall_features.items():
            values[movie_id] = float(feature_values.get(name, 0.0))
        arrays[name] = values

    feature_stats["movie_feature_lookup_arrays"] = arrays
    return arrays


def _hybrid_scores_from_features(features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.empty(0, dtype=np.float32)
    return _normalize_array(_raw_hybrid_scores_from_features(features))


def _raw_hybrid_scores_from_features(features: np.ndarray) -> np.ndarray:
    mf_index = FEATURE_NAMES.index("mf_score")
    collaborative_index = FEATURE_NAMES.index("collaborative_score")
    quality_index = FEATURE_NAMES.index("quality_score")
    return (
        HYBRID_MF_WEIGHT * features[:, mf_index]
        + HYBRID_COLLABORATIVE_WEIGHT * features[:, collaborative_index]
        + HYBRID_QUALITY_WEIGHT * features[:, quality_index]
    ).astype(np.float32)


def _hybrid_scores_by_user(features: np.ndarray, user_ids: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.empty(0, dtype=np.float32)
    raw_scores = _raw_hybrid_scores_from_features(features)
    scores = np.empty(len(features), dtype=np.float32)
    for user_id in np.unique(user_ids):
        positions = np.flatnonzero(user_ids == user_id)
        scores[positions] = _normalize_array(raw_scores[positions])
    return scores


def _blended_scores_from_features(ranker: LinearRanker, features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.empty(0, dtype=np.float32)
    linear_scores = _normalize_array(ranker.score(features))
    hybrid_scores = _hybrid_scores_from_features(features)
    return _blend_scores(linear_scores, hybrid_scores, ranker.blend_weight, mode=ranker.blend_mode)


def _blended_scores_by_user(ranker: LinearRanker, features: np.ndarray, user_ids: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.empty(0, dtype=np.float32)
    scores = np.empty(len(features), dtype=np.float32)
    for user_id in np.unique(user_ids):
        positions = np.flatnonzero(user_ids == user_id)
        scores[positions] = _blended_scores_from_features(ranker, features[positions])
    return scores


def _residual_base_scores_by_user(
    features: np.ndarray,
    user_ids: np.ndarray,
    *,
    base_ranker: LinearRanker | None,
) -> np.ndarray:
    if base_ranker is None:
        return _hybrid_scores_by_user(features, user_ids)
    return _blended_scores_by_user(base_ranker, features, user_ids)


def _residual_scores_from_features(
    ranker: ResidualLinearRanker,
    features: np.ndarray,
    *,
    base_ranker: LinearRanker | None = None,
) -> np.ndarray:
    if len(features) == 0:
        return np.empty(0, dtype=np.float32)
    base_scores = (
        _hybrid_scores_from_features(features)
        if base_ranker is None
        else _blended_scores_from_features(base_ranker, features)
    )
    return np.clip(
        base_scores + ranker.residual_alpha * ranker.score(features),
        0.0,
        1.0,
    ).astype(np.float32)


def _hybrid_scores_for_user(
    model: MatrixFactorizationModel,
    user_id: int,
    candidate_movie_ids: np.ndarray,
    *,
    active_profile: dict[int, float],
    exclude_movie_ids: set[int],
    quality_scores: dict[int, float],
    indexes,
) -> np.ndarray:
    mf_raw = model.score_known_user(user_id, candidate_movie_ids)
    mf_scores = (mf_raw - model.config.min_rating) / (model.config.max_rating - model.config.min_rating)
    collaborative_scores = _normalize_score_dict(
        _score_user_user_candidates(
            user_id,
            active_profile,
            indexes=indexes,
            candidate_movie_ids=candidate_movie_ids,
            exclude_movie_ids=exclude_movie_ids,
        )
    )
    scores = np.zeros(len(candidate_movie_ids), dtype=np.float32)
    for index, movie_id_value in enumerate(candidate_movie_ids):
        movie_id = int(movie_id_value)
        scores[index] = (
            HYBRID_MF_WEIGHT * float(mf_scores[index])
            + HYBRID_COLLABORATIVE_WEIGHT * float(collaborative_scores.get(movie_id, 0.0))
            + HYBRID_QUALITY_WEIGHT * float(quality_scores.get(movie_id, 0.0))
        )
    return _normalize_array(scores)


def _normalize_array(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        return np.ones(len(values), dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def _feature_stats(
    train_rows: list[TrainingRating],
    *,
    db_path: Path | None = None,
    model: MatrixFactorizationModel | None = None,
    enable_item_item: bool = True,
) -> dict:
    movie_counts: Counter[int] = Counter()
    movie_sums: defaultdict[int, float] = defaultdict(float)
    user_counts: Counter[int] = Counter()
    user_sums: defaultdict[int, float] = defaultdict(float)
    user_positive_counts: Counter[int] = Counter()
    user_train_ratings: dict[int, dict[int, float]] = defaultdict(dict)
    user_positive_movies: dict[int, list[tuple[float, int]]] = defaultdict(list)
    user_negative_movies: dict[int, list[tuple[float, int]]] = defaultdict(list)
    user_positive_year_sums: defaultdict[int, float] = defaultdict(float)
    user_positive_year_counts: Counter[int] = Counter()
    for row in train_rows:
        movie_counts[row.movie_id] += 1
        movie_sums[row.movie_id] += row.rating
        user_counts[row.user_id] += 1
        user_sums[row.user_id] += row.rating
        user_train_ratings[row.user_id][row.movie_id] = row.rating
        if row.rating >= 4.0:
            user_positive_counts[row.user_id] += 1
            user_positive_movies[row.user_id].append((row.rating, row.movie_id))
        elif row.rating <= 2.0:
            user_negative_movies[row.user_id].append((row.rating, row.movie_id))

    movie = {
        movie_id: (movie_sums[movie_id] / count, count)
        for movie_id, count in movie_counts.items()
    }
    user = {
        user_id: (user_sums[user_id] / count, count)
        for user_id, count in user_counts.items()
    }
    movie_metadata = _load_movie_metadata(db_path) if db_path is not None else {}
    year_values = [year for year in movie_metadata.values() if year is not None]
    min_year = min(year_values, default=1900)
    max_year = max(year_values, default=2025)
    year_span = max(max_year - min_year, 1)

    for row in train_rows:
        if row.rating < 4.0:
            continue
        year = movie_metadata.get(row.movie_id)
        if year is None:
            continue
        user_positive_year_sums[row.user_id] += float(year)
        user_positive_year_counts[row.user_id] += 1

    profile_centroids: dict[int, np.ndarray] = {}
    profile_norms: dict[int, float] = {}
    negative_profile_centroids: dict[int, np.ndarray] = {}
    negative_profile_norms: dict[int, float] = {}
    if model is not None and model.movie_factors.size:
        profile_centroids, profile_norms = _build_profile_centroids(
            user_positive_movies,
            model,
            positive=True,
        )
        negative_profile_centroids, negative_profile_norms = _build_profile_centroids(
            user_negative_movies,
            model,
            positive=False,
        )

    item_item_neighbors = (
        _build_item_item_neighbors(user_positive_movies, movie_counts)
        if enable_item_item
        else {}
    )

    return {
        "movie": movie,
        "user": user,
        "user_train_ratings": {user_id: dict(values) for user_id, values in user_train_ratings.items()},
        "user_positive_counts": dict(user_positive_counts),
        "user_positive_year_avg": {
            user_id: user_positive_year_sums[user_id] / count
            for user_id, count in user_positive_year_counts.items()
        },
        "movie_year": movie_metadata,
        "min_year": min_year,
        "year_span": year_span,
        "profile_centroids": profile_centroids,
        "profile_norms": profile_norms,
        "negative_profile_centroids": negative_profile_centroids,
        "negative_profile_norms": negative_profile_norms,
        "item_item_neighbors": item_item_neighbors,
        "max_movie_log_count": math.log1p(max(movie_counts.values(), default=1)),
        "max_user_log_count": math.log1p(max(user_counts.values(), default=1)),
    }


def _build_profile_centroids(
    user_movies: dict[int, list[tuple[float, int]]],
    model: MatrixFactorizationModel,
    *,
    positive: bool,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    centroids: dict[int, np.ndarray] = {}
    norms: dict[int, float] = {}
    for user_id, values in user_movies.items():
        vector_sum = np.zeros(model.movie_factors.shape[1], dtype=np.float32)
        weight_sum = 0.0
        for rating, movie_id in values:
            movie_index = model.movie_to_index.get(movie_id)
            if movie_index is None:
                continue
            if positive:
                weight = max(float(rating) - 3.0, 0.25)
            else:
                weight = max(3.0 - float(rating), 0.25)
            vector_sum += model.movie_factors[movie_index] * weight
            weight_sum += weight
        if weight_sum <= 0:
            continue
        centroid = vector_sum / weight_sum
        norm = float(np.linalg.norm(centroid))
        if norm <= 0:
            continue
        centroids[user_id] = centroid.astype(np.float32)
        norms[user_id] = norm
    return centroids, norms


def _build_item_item_neighbors(
    user_positive_movies: dict[int, list[tuple[float, int]]],
    movie_counts: Counter[int],
) -> dict[int, list[tuple[int, float]]]:
    pair_counts: Counter[tuple[int, int]] = Counter()
    for values in user_positive_movies.values():
        movie_ids = _bounded_positive_movie_ids(values)
        for left_index, left_movie_id in enumerate(movie_ids):
            for right_movie_id in movie_ids[left_index + 1 :]:
                if left_movie_id == right_movie_id:
                    continue
                a, b = sorted((int(left_movie_id), int(right_movie_id)))
                pair_counts[(a, b)] += 1

    neighbors: defaultdict[int, list[tuple[int, float]]] = defaultdict(list)
    for (left_movie_id, right_movie_id), count in pair_counts.items():
        if count < ITEM_ITEM_MIN_PAIR_COUNT:
            continue
        left_count = max(movie_counts[left_movie_id], 1)
        right_count = max(movie_counts[right_movie_id], 1)
        cosine = count / math.sqrt(left_count * right_count)
        shrink = count / (count + ITEM_ITEM_SHRINKAGE)
        score = float(cosine * shrink)
        if score <= 0:
            continue
        neighbors[left_movie_id].append((right_movie_id, score))
        neighbors[right_movie_id].append((left_movie_id, score))

    return {
        movie_id: sorted(values, key=lambda item: (item[1], -item[0]), reverse=True)[:ITEM_ITEM_MAX_NEIGHBORS]
        for movie_id, values in neighbors.items()
    }


def _bounded_positive_movie_ids(values: list[tuple[float, int]]) -> list[int]:
    ranked = sorted(values, key=lambda item: (-float(item[0]), int(item[1])))
    return [int(movie_id) for _rating, movie_id in ranked[:ITEM_ITEM_MAX_PROFILE_MOVIES]]


def _load_movie_metadata(db_path: Path) -> dict[int, int | None]:
    if not db_path.exists():
        return {}
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = conn.execute("SELECT movie_id, release_year FROM movies").fetchall()
    finally:
        conn.close()
    return {
        int(movie_id): None if release_year is None else int(release_year)
        for movie_id, release_year in rows
    }


def _evaluation_config(config: LearningToRankConfig) -> EvaluationConfig:
    return EvaluationConfig(
        max_users=config.max_users,
        min_ratings_per_user=config.min_ratings_per_user,
        test_ratio=config.test_ratio,
        relevant_threshold=config.relevant_threshold,
        top_k=config.top_k,
        candidate_limit=config.candidate_limit,
        candidate_recall_strategy=config.candidate_recall_strategy,
        candidate_recall_rrf_k=config.candidate_recall_rrf_k,
        candidate_recall_priors=config.candidate_recall_priors,
    )


def _mf_config(config: LearningToRankConfig) -> MatrixFactorizationConfig:
    return MatrixFactorizationConfig(
        factors=config.mf_factors,
        epochs=config.mf_epochs,
        learning_rate=config.mf_learning_rate,
        regularization=config.mf_regularization,
        batch_size=config.mf_batch_size,
        backend="auto",
        device="auto",
        optimizer="adam",
    )


def _candidate_split(split: dict, *, include_test_relevant: bool = True) -> dict:
    relevant = {}
    if include_test_relevant:
        relevant.update(
            {
                user_id: set(values)
                for user_id, values in split["relevant_by_user"].items()
            }
        )
    for user_id, values in split["ranker_relevant_by_user"].items():
        relevant.setdefault(user_id, set()).update(values)
    return {
        "train": split["train"],
        "test": [*split["ranker_train"], *split["test"]],
        "train_history": split["base_history"],
        "relevant_by_user": relevant,
        "test_by_user": split["test_by_user"],
    }


def _evaluation_split(split: dict) -> dict:
    return {
        "train": split["train"],
        "test": split["test"],
        "train_history": split["train_history"],
        "relevant_by_user": split["relevant_by_user"],
        "test_by_user": split["test_by_user"],
    }


def _ensure_training_candidates(candidate_movie_ids: np.ndarray, split: dict, model: MatrixFactorizationModel) -> np.ndarray:
    seen = {int(movie_id) for movie_id in candidate_movie_ids}
    values = [int(movie_id) for movie_id in candidate_movie_ids]
    for movie_id in sorted(
        {
            movie_id
            for movie_ids in split["ranker_relevant_by_user"].values()
            for movie_id in movie_ids
            if movie_id in model.movie_to_index
        }
    ):
        if movie_id not in seen:
            values.append(movie_id)
            seen.add(movie_id)
    return np.array(values, dtype=np.int32)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
