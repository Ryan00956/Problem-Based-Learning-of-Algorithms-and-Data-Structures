from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src.datasets.netflix.evaluation import (
    _build_collaborative_indexes,
    candidate_movie_ids_from_split,
    evaluate_model,
    evaluate_popular_baseline,
    evaluate_user_user_collaborative,
    load_time_split_source,
    movie_quality_scores,
)
from src.datasets.netflix.candidate_recall import build_multi_route_candidate_pool
from src.datasets.netflix.collaborative_vectorized import prime_collaborative_caches
from src.datasets.netflix.import_duckdb import DEFAULT_DB_PATH
from src.datasets.netflix.learning_to_rank import (
    FEATURE_NAMES,
    LearningToRankConfig,
    _candidate_split,
    _blend_modes,
    _blend_scores,
    _blend_weight_grid,
    _evaluate_precomputed_blend_scores,
    _ensure_training_candidates,
    _evaluation_config,
    _evaluation_split,
    _feature_stats,
    _hard_negative_scores,
    _hybrid_scores_from_features,
    _mf_config,
    _normalize_array,
    _rank_percentiles,
    _ranker_ratings_by_user,
    _sample_ranker_negative_ids,
    evaluate_hybrid_ranker_from_features,
    features_for_user_candidates,
    split_by_user_three_way,
)
from src.datasets.netflix.matrix_factorization import MatrixFactorizationModel, mf_batch_enabled


@dataclass(frozen=True)
class NeuralRerankerConfig(LearningToRankConfig):
    embedding_dim: int = 24
    hidden_dim: int = 64
    dropout: float = 0.10
    batch_size: int = 2048
    epochs: int = 25
    learning_rate: float = 0.001
    l2: float = 0.0001
    min_blend_precision_gain: float = 0.015
    loss: str = "pairwise"
    pairs_per_positive: int = 6
    pair_negative_sampling: str = "random"


@dataclass(frozen=True)
class RerankerExamples:
    user_ids: np.ndarray
    movie_ids: np.ndarray
    features: np.ndarray
    labels: np.ndarray


class NeuralReranker:
    def __init__(
        self,
        config: NeuralRerankerConfig,
        *,
        user_to_index: dict[int, int],
        movie_to_index: dict[int, int],
    ) -> None:
        self.config = config
        self.user_to_index = dict(user_to_index)
        self.movie_to_index = dict(movie_to_index)
        self.unknown_user_index = len(self.user_to_index)
        self.unknown_movie_index = len(self.movie_to_index)
        self.mean = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        self.std = np.ones(len(FEATURE_NAMES), dtype=np.float32)
        self.blend_weight = 1.0
        self.blend_mode = "value"
        self.backend_used = "unfitted"
        self.device_used = "none"
        self.model = None
        self.training_curve: list[dict] = []

    def fit(self, examples: RerankerExamples) -> list[dict]:
        if len(examples.labels) == 0:
            raise ValueError("neural reranker needs at least one training example")
        if not examples.labels.any():
            raise ValueError("neural reranker needs at least one positive example")

        try:
            import torch
            from torch import nn
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyTorch is not installed. Install GPU dependencies with "
                "`python -m pip install -r requirements-gpu-cu128.txt`."
            ) from exc

        torch.manual_seed(self.config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backend_used = "torch"
        self.device_used = str(device)

        self.mean = examples.features.mean(axis=0).astype(np.float32)
        self.std = examples.features.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        features = ((examples.features - self.mean) / self.std).astype(np.float32)
        user_indices = self._user_indices(examples.user_ids)
        movie_indices = self._movie_indices(examples.movie_ids)
        labels = examples.labels.astype(np.float32)

        model = _TorchNeuralReranker(
            user_count=len(self.user_to_index) + 1,
            movie_count=len(self.movie_to_index) + 1,
            feature_count=len(FEATURE_NAMES),
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        ).to(device)
        positive_count = max(float(labels.sum()), 1.0)
        negative_count = max(float(len(labels) - labels.sum()), 1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.l2)

        user_tensor = torch.as_tensor(user_indices, dtype=torch.long)
        movie_tensor = torch.as_tensor(movie_indices, dtype=torch.long)
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        label_tensor = torch.as_tensor(labels, dtype=torch.float32)
        if self.config.loss == "pairwise":
            pair_indices = _build_pair_indices(
                user_ids=examples.user_ids,
                labels=examples.labels,
                features=examples.features if self.config.pair_negative_sampling == "hard" else None,
                pairs_per_positive=self.config.pairs_per_positive,
                seed=self.config.seed,
            )
            pair_left = torch.as_tensor(pair_indices[0], dtype=torch.long)
            pair_right = torch.as_tensor(pair_indices[1], dtype=torch.long)
        elif self.config.loss == "pointwise":
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative_count / positive_count], device=device))
            pair_left = torch.empty(0, dtype=torch.long)
            pair_right = torch.empty(0, dtype=torch.long)
        else:
            raise ValueError("loss must be 'pairwise' or 'pointwise'")

        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        self.training_curve = []
        batch_size = max(1, int(self.config.batch_size))
        for epoch in range(1, self.config.epochs + 1):
            total_loss = 0.0
            total_examples = 0
            model.train()
            if self.config.loss == "pairwise":
                order = torch.randperm(len(pair_left), generator=generator)
                for start in range(0, len(pair_left), batch_size):
                    batch = order[start : start + batch_size]
                    positive_batch = pair_left[batch]
                    negative_batch = pair_right[batch]
                    positive_logits = model(
                        user_tensor[positive_batch].to(device),
                        movie_tensor[positive_batch].to(device),
                        feature_tensor[positive_batch].to(device),
                    )
                    negative_logits = model(
                        user_tensor[negative_batch].to(device),
                        movie_tensor[negative_batch].to(device),
                        feature_tensor[negative_batch].to(device),
                    )
                    loss = torch.nn.functional.softplus(-(positive_logits - negative_logits)).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach().cpu()) * len(batch)
                    total_examples += len(batch)
            else:
                order = torch.randperm(len(labels), generator=generator)
                for start in range(0, len(labels), batch_size):
                    batch = order[start : start + batch_size]
                    user_batch = user_tensor[batch].to(device)
                    movie_batch = movie_tensor[batch].to(device)
                    feature_batch = feature_tensor[batch].to(device)
                    label_batch = label_tensor[batch].to(device)

                    logits = model(user_batch, movie_batch, feature_batch)
                    loss = criterion(logits, label_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach().cpu()) * len(batch)
                    total_examples += len(batch)

            self.training_curve.append(
                {
                    "epoch": epoch,
                    "loss": round(total_loss / max(total_examples, 1), 6),
                    "examples": int(len(labels)),
                    "positives": int(labels.sum()),
                    "negatives": int(len(labels) - labels.sum()),
                    "backend": self.backend_used,
                    "device": self.device_used,
                    "loss_type": self.config.loss,
                    "pairs": int(len(pair_left)),
                    "embedding_dim": self.config.embedding_dim,
                    "hidden_dim": self.config.hidden_dim,
                    "dropout": self.config.dropout,
                }
            )

        self.model = model
        return list(self.training_curve)

    def score(self, user_ids: np.ndarray, movie_ids: np.ndarray, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("neural reranker has not been fitted")
        if len(user_ids) == 0:
            return np.empty(0, dtype=np.float32)

        import torch

        device = torch.device(self.device_used)
        normalized = ((features - self.mean) / self.std).astype(np.float32)
        user_tensor = torch.as_tensor(self._user_indices(user_ids), dtype=torch.long, device=device)
        movie_tensor = torch.as_tensor(self._movie_indices(movie_ids), dtype=torch.long, device=device)
        feature_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(user_tensor, movie_tensor, feature_tensor)
        return logits.detach().cpu().numpy().astype(np.float32)

    def _user_indices(self, user_ids: np.ndarray) -> np.ndarray:
        return np.array(
            [self.user_to_index.get(int(user_id), self.unknown_user_index) for user_id in user_ids],
            dtype=np.int64,
        )

    def _movie_indices(self, movie_ids: np.ndarray) -> np.ndarray:
        return np.array(
            [self.movie_to_index.get(int(movie_id), self.unknown_movie_index) for movie_id in movie_ids],
            dtype=np.int64,
        )


def smoke_neural_reranker_config() -> NeuralRerankerConfig:
    return NeuralRerankerConfig(
        max_users=40,
        candidate_limit=300,
        negatives_per_positive=6,
        epochs=5,
        mf_factors=16,
        mf_epochs=5,
        mf_batch_size=4096,
        embedding_dim=12,
        hidden_dim=32,
        batch_size=1024,
    )


def standard_neural_reranker_config() -> NeuralRerankerConfig:
    return NeuralRerankerConfig(negative_sampling="explicit_hard")


def run_neural_reranker(
    output_dir: Path,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    config: NeuralRerankerConfig | None = None,
) -> dict[str, Path]:
    config = config or NeuralRerankerConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_time_split_source(db_path, _evaluation_config(config))
    split = split_by_user_three_way(rows, config)
    model = MatrixFactorizationModel(_mf_config(config))
    mf_training_curve = model.fit(split["train"])
    quality_scores = movie_quality_scores(split)
    indexes = _build_collaborative_indexes(split["train"])
    feature_stats = _feature_stats(split["train"], db_path=db_path, model=model)
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
    fit_split, validation_split = split_ranker_fit_validation(split, config)

    train_examples = build_neural_training_examples(
        model,
        fit_split,
        config,
        candidate_movie_ids=candidate_movie_ids,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    ranker = NeuralReranker(
        config,
        user_to_index=model.user_to_index,
        movie_to_index=model.movie_to_index,
    )
    ranker_training_curve = ranker.fit(train_examples)
    blend_weight, blend_tuning_rows = tune_neural_blend_weight(
        ranker,
        model,
        split,
        config,
        tuning_split=validation_split,
        candidate_movie_ids=candidate_movie_ids,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    ranker.blend_weight = blend_weight
    feature_stats.get("user_candidate_feature_cache", {}).clear()

    baseline_split = _evaluation_split(split)
    evaluation_config = _evaluation_config(config)
    popular_metrics = evaluate_popular_baseline(
        baseline_split,
        evaluation_config,
        candidate_movie_ids=candidate_movie_ids,
        quality_scores=quality_scores,
    )
    collaborative_metrics = evaluate_user_user_collaborative(
        baseline_split,
        evaluation_config,
        candidate_movie_ids=candidate_movie_ids,
        indexes=indexes,
        score_cache=feature_stats.setdefault("collaborative_score_cache", {}),
        neighbor_cache=feature_stats.setdefault("user_neighbor_cache", {}),
        raw_score_cache=feature_stats.setdefault("collaborative_raw_cache", {}),
    )
    mf_metrics = evaluate_model(
        model,
        baseline_split,
        evaluation_config,
        candidate_movie_ids=candidate_movie_ids,
    )
    hybrid_metrics = evaluate_hybrid_ranker_from_features(
        model,
        baseline_split,
        evaluation_config,
        candidate_movie_ids=candidate_movie_ids,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    neural_metrics = evaluate_neural_reranker(
        ranker,
        model,
        baseline_split,
        evaluation_config,
        candidate_movie_ids=candidate_movie_ids,
        quality_scores=quality_scores,
        indexes=indexes,
        feature_stats=feature_stats,
    )
    feature_stats.get("user_candidate_feature_cache", {}).clear()
    feature_stats.get("collaborative_score_cache", {}).clear()

    metrics_path = output_dir / "metrics.csv"
    ranker_training_path = output_dir / "neural_training.csv"
    blend_tuning_path = output_dir / "blend_tuning.csv"
    candidate_recall_path = output_dir / "candidate_recall.csv"
    mf_training_path = output_dir / "mf_training.csv"
    summary_path = output_dir / "summary.json"
    _write_csv(metrics_path, [popular_metrics, collaborative_metrics, mf_metrics, hybrid_metrics, neural_metrics])
    _write_csv(ranker_training_path, ranker_training_curve)
    _write_csv(blend_tuning_path, blend_tuning_rows)
    _write_csv(candidate_recall_path, recall_result.source_rows)
    _write_csv(mf_training_path, mf_training_curve)
    summary_path.write_text(
        json.dumps(
            {
                "config": config.__dict__,
                "training_examples": int(len(train_examples.labels)),
                "training_positives": int(train_examples.labels.sum()),
                "training_negatives": int(len(train_examples.labels) - train_examples.labels.sum()),
                "neural_blend_weight": float(blend_weight),
                "neural_blend_mode": ranker.blend_mode,
                "backend": ranker.backend_used,
                "device": ranker.device_used,
                "best_metric": max(
                    [popular_metrics, collaborative_metrics, mf_metrics, hybrid_metrics, neural_metrics],
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
        "neural_training": ranker_training_path,
        "blend_tuning": blend_tuning_path,
        "candidate_recall": candidate_recall_path,
        "mf_training": mf_training_path,
        "summary": summary_path,
    }


def build_neural_training_examples(
    model: MatrixFactorizationModel,
    split: dict,
    config: NeuralRerankerConfig,
    *,
    candidate_movie_ids: np.ndarray,
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
) -> RerankerExamples:
    user_ids: list[int] = []
    movie_ids: list[int] = []
    features: list[np.ndarray] = []
    labels: list[int] = []
    rng = np.random.default_rng(config.seed)
    ranker_ratings = _ranker_ratings_by_user(split)

    for user_id, positives in split["ranker_relevant_by_user"].items():
        if not positives:
            continue
        exclude = split["base_history"].get(user_id, set())
        candidate_set = [int(movie_id) for movie_id in candidate_movie_ids if int(movie_id) not in exclude]
        if not candidate_set:
            continue

        candidate_array = np.array(candidate_set, dtype=np.int32)
        user_features = features_for_user_candidates(
            model,
            user_id,
            candidate_array,
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
            user_ids.append(user_id)
            movie_ids.append(movie_id)
            features.append(user_features[candidate_index[movie_id]])
            labels.append(1)

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
            user_ids.append(user_id)
            movie_ids.append(movie_id)
            features.append(user_features[candidate_index[movie_id]])
            labels.append(0)

    if not features:
        return RerankerExamples(
            user_ids=np.empty(0, dtype=np.int64),
            movie_ids=np.empty(0, dtype=np.int64),
            features=np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
            labels=np.empty(0, dtype=np.float32),
        )
    return RerankerExamples(
        user_ids=np.array(user_ids, dtype=np.int64),
        movie_ids=np.array(movie_ids, dtype=np.int64),
        features=np.vstack(features).astype(np.float32),
        labels=np.array(labels, dtype=np.float32),
    )


def _build_pair_indices(
    *,
    user_ids: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray | None = None,
    pairs_per_positive: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    by_user: dict[int, dict[str, list[int]]] = {}
    for index, (user_id, label) in enumerate(zip(user_ids, labels, strict=True)):
        bucket = by_user.setdefault(int(user_id), {"positive": [], "negative": []})
        if label >= 0.5:
            bucket["positive"].append(index)
        else:
            bucket["negative"].append(index)

    rng = np.random.default_rng(seed)
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    sample_count = max(1, int(pairs_per_positive))
    for values in by_user.values():
        positives = values["positive"]
        negatives = values["negative"]
        if not positives or not negatives:
            continue
        ordered_negatives = list(negatives)
        if features is not None and len(features):
            hard_scores = _hard_negative_scores(features)
            ordered_negatives.sort(key=lambda index: hard_scores[index], reverse=True)
        for positive_index in positives:
            take = min(sample_count, len(negatives))
            hard_take = min(max(1, round(take * 0.75)), len(ordered_negatives))
            hard_selected = ordered_negatives[:hard_take]
            remaining = [index for index in negatives if index not in set(hard_selected)]
            random_take = min(take - len(hard_selected), len(remaining))
            random_selected = list(rng.choice(remaining, size=random_take, replace=False)) if random_take else []
            selected = list(dict.fromkeys([*hard_selected, *random_selected]))
            for negative_index in selected:
                positive_indices.append(positive_index)
                negative_indices.append(int(negative_index))

    if not positive_indices:
        raise ValueError("pairwise neural reranker needs at least one positive-negative pair")
    return np.array(positive_indices, dtype=np.int64), np.array(negative_indices, dtype=np.int64)


def split_ranker_fit_validation(source_split: dict, config: NeuralRerankerConfig) -> tuple[dict, dict]:
    by_user: dict[int, list] = {}
    for row in source_split["ranker_train"]:
        by_user.setdefault(row.user_id, []).append(row)

    fit_rows = []
    validation_rows = []
    fit_history = {user_id: set(movie_ids) for user_id, movie_ids in source_split["base_history"].items()}
    fit_relevant: dict[int, set[int]] = {}
    validation_relevant: dict[int, set[int]] = {}

    for user_id, rows in by_user.items():
        if len(rows) < 2:
            user_fit_rows = rows
            user_validation_rows = []
        else:
            validation_count = max(1, round(len(rows) * 0.5))
            user_fit_rows = rows[:-validation_count]
            user_validation_rows = rows[-validation_count:]
        fit_rows.extend(user_fit_rows)
        validation_rows.extend(user_validation_rows)

        for row in user_fit_rows:
            fit_history.setdefault(user_id, set()).add(row.movie_id)
            if row.rating >= config.relevant_threshold:
                fit_relevant.setdefault(user_id, set()).add(row.movie_id)
        for row in user_validation_rows:
            if row.rating >= config.relevant_threshold:
                validation_relevant.setdefault(user_id, set()).add(row.movie_id)

    if not validation_relevant:
        validation_rows = list(source_split["ranker_train"])
        validation_relevant = {
            user_id: set(values)
            for user_id, values in source_split["ranker_relevant_by_user"].items()
        }
        fit_history = source_split["base_history"]

    fit_split = {
        **source_split,
        "ranker_train": fit_rows,
        "ranker_relevant_by_user": fit_relevant,
    }
    validation_split = {
        "train": source_split["train"],
        "test": validation_rows,
        "train_history": fit_history,
        "relevant_by_user": validation_relevant,
        "test_by_user": {},
    }
    return fit_split, validation_split


def evaluate_neural_reranker(
    ranker: NeuralReranker,
    model: MatrixFactorizationModel,
    split: dict,
    config,
    *,
    candidate_movie_ids: np.ndarray,
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
) -> dict:
    from src.datasets.netflix.evaluation import _evaluate_ranker

    return _evaluate_ranker(
        "stacked_neural_hybrid_reranker",
        split,
        config,
        candidate_movie_ids,
        lambda user_id, candidates, exclude: recommend_neural(
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
        backend=ranker.backend_used,
        device=ranker.device_used,
        rmse="",
        users=len(model.user_to_index),
        movies=len(model.movie_to_index),
    )


def recommend_neural(
    ranker: NeuralReranker,
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
    eligible = np.array(
        [int(movie_id) for movie_id in candidate_movie_ids if int(movie_id) not in exclude_movie_ids],
        dtype=np.int32,
    )
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
    user_ids = np.full(len(eligible), int(user_id), dtype=np.int64)
    neural_scores = _normalize_array(ranker.score(user_ids, eligible.astype(np.int64), features))
    hybrid_scores = _hybrid_scores_from_features(features)
    scores = _blend_scores(neural_scores, hybrid_scores, ranker.blend_weight, mode=ranker.blend_mode)
    take = min(top_k, len(eligible))
    top_positions = np.argpartition(-scores, take - 1)[:take]
    ranked_positions = top_positions[np.argsort(-scores[top_positions])]
    return [(int(eligible[position]), float(scores[position])) for position in ranked_positions]


def tune_neural_blend_weight(
    ranker: NeuralReranker,
    model: MatrixFactorizationModel,
    split: dict,
    config: NeuralRerankerConfig,
    *,
    tuning_split: dict | None = None,
    candidate_movie_ids: np.ndarray,
    quality_scores: dict[int, float],
    indexes,
    feature_stats: dict,
) -> tuple[float, list[dict]]:
    if tuning_split is None:
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
    score_records = _precompute_neural_blend_scores(
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
        ranker.blend_mode = blend_mode
        for blend_weight in _blend_weight_grid():
            metrics = _evaluate_precomputed_blend_scores(
                "stacked_neural_hybrid_reranker",
                score_records,
                tuning_split,
                _evaluation_config(config),
                blend_weight=blend_weight,
                blend_mode=blend_mode,
                backend=ranker.backend_used,
                device=ranker.device_used,
                users=len(model.user_to_index),
                movies=len(model.movie_to_index),
                candidate_count=len(candidate_movie_ids),
            )
            rows.append({"neural_blend_weight": blend_weight, "blend_mode": blend_mode, **metrics})
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
        or float(best["map_at_k"]) < float(baseline["map_at_k"])
    ):
        return 0.0, rows
    ranker.blend_mode = str(best["blend_mode"])
    return float(best["neural_blend_weight"]), rows


def _precompute_neural_blend_scores(
    ranker: NeuralReranker,
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
        user_ids = np.full(len(eligible), int(user_id), dtype=np.int64)
        learned_scores = _normalize_array(ranker.score(user_ids, eligible.astype(np.int64), features))
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


class _TorchNeuralReranker:
    def __new__(cls, *args, **kwargs):
        import torch
        from torch import nn

        class Module(nn.Module):
            def __init__(
                self,
                *,
                user_count: int,
                movie_count: int,
                feature_count: int,
                embedding_dim: int,
                hidden_dim: int,
                dropout: float,
            ) -> None:
                super().__init__()
                self.user_embedding = nn.Embedding(user_count, embedding_dim)
                self.movie_embedding = nn.Embedding(movie_count, embedding_dim)
                input_dim = embedding_dim * 3 + feature_count
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
                    nn.ReLU(),
                    nn.Linear(max(8, hidden_dim // 2), 1),
                )

            def forward(self, user_ids, movie_ids, features):
                user_vector = self.user_embedding(user_ids)
                movie_vector = self.movie_embedding(movie_ids)
                interaction = user_vector * movie_vector
                x = torch.cat([user_vector, movie_vector, interaction, features], dim=1)
                return self.network(x).squeeze(1)

        return Module(*args, **kwargs)
