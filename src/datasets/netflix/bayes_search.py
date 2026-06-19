from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.datasets.netflix.candidate_recall import SOURCE_NAMES, build_multi_route_candidate_pool
from src.datasets.netflix.evaluation import (
    EvaluationConfig,
    _build_collaborative_indexes,
    candidate_movie_ids_from_split,
    evaluate_hybrid_ranker,
    evaluate_model,
    load_time_split_source,
    movie_quality_scores,
    split_by_user_time,
)
from src.datasets.netflix.import_duckdb import DEFAULT_DB_PATH
from src.datasets.netflix.learning_to_rank import (
    LearningToRankConfig,
    _candidate_split,
    _feature_stats,
    split_by_user_three_way,
)
from src.datasets.netflix.matrix_factorization import MatrixFactorizationConfig, MatrixFactorizationModel


SearchMode = Literal["hybrid", "mf", "recall"]


@dataclass(frozen=True)
class BayesSearchConfig:
    mode: SearchMode = "hybrid"
    trials: int = 30
    max_users: int = 160
    min_ratings_per_user: int = 12
    candidate_limit: int = 1200
    top_k: int = 10
    seed: int = 42
    min_route_recall: float = 0.55


def smoke_bayes_search_config(mode: SearchMode = "hybrid") -> BayesSearchConfig:
    return BayesSearchConfig(
        mode=mode,
        trials=5,
        max_users=40,
        candidate_limit=300,
    )


def standard_bayes_search_config(mode: SearchMode = "hybrid", trials: int = 30) -> BayesSearchConfig:
    return BayesSearchConfig(mode=mode, trials=trials)


def run_bayes_search(
    output_dir: Path,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    config: BayesSearchConfig | None = None,
) -> dict[str, Path]:
    config = config or BayesSearchConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = output_dir / "optuna_study.db"
    trials_csv = output_dir / "trials.csv"
    best_params_json = output_dir / "best_params.json"
    best_summary_csv = output_dir / "best_summary.csv"

    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Optuna is not installed. Install tuning dependencies with "
            "`python -m pip install -r requirements-tuning.txt`."
        ) from exc

    evaluation_config = EvaluationConfig(
        max_users=config.max_users,
        min_ratings_per_user=config.min_ratings_per_user,
        top_k=config.top_k,
        candidate_limit=config.candidate_limit,
    )
    rows = load_time_split_source(db_path, evaluation_config)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        storage=f"sqlite:///{storage_path.as_posix()}",
        study_name=f"netflix_{config.mode}_bayes_search",
        load_if_exists=True,
    )

    if config.mode == "hybrid":
        split = split_by_user_time(rows, evaluation_config)
        objective = _hybrid_objective(split, evaluation_config)
    elif config.mode == "mf":
        split = split_by_user_time(rows, evaluation_config)
        objective = _mf_objective(split, evaluation_config)
    elif config.mode == "recall":
        objective = _recall_objective(rows, config, db_path)
    else:
        raise ValueError("mode must be 'hybrid', 'mf', or 'recall'")

    remaining_trials = max(config.trials - len(study.trials), 0)
    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False)

    trial_rows = _trial_rows(study)
    _write_csv(trials_csv, trial_rows)
    best_payload = _best_payload(study, config)
    best_params_json.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(best_summary_csv, [best_payload["best_metrics"]])
    if hasattr(study._storage, "remove_session"):
        study._storage.remove_session()

    return {
        "study_db": storage_path,
        "trials": trials_csv,
        "best_params": best_params_json,
        "best_summary": best_summary_csv,
    }


def _hybrid_objective(split: dict, evaluation_config: EvaluationConfig):
    base_model = MatrixFactorizationModel(MatrixFactorizationConfig())
    base_model.fit(split["train"])
    candidate_movie_ids = candidate_movie_ids_from_split(
        split,
        evaluation_config,
        known_movie_ids=set(base_model.movie_to_index),
    )
    quality_scores = movie_quality_scores(split)
    indexes = _build_collaborative_indexes(split["train"])

    def objective(trial) -> float:
        mf_weight = trial.suggest_float("mf_weight", 0.05, 0.45)
        collaborative_weight = trial.suggest_float("collaborative_weight", 0.0, 0.15)
        if mf_weight + collaborative_weight >= 0.98:
            return -1.0
        quality_weight = 1.0 - mf_weight - collaborative_weight
        metrics = evaluate_hybrid_ranker(
            base_model,
            split,
            evaluation_config,
            candidate_movie_ids=candidate_movie_ids,
            quality_scores=quality_scores,
            indexes=indexes,
            mf_weight=mf_weight,
            collaborative_weight=collaborative_weight,
            quality_weight=quality_weight,
        )
        _set_metric_attrs(trial, metrics)
        trial.set_user_attr("quality_weight", quality_weight)
        return _objective_score(metrics)

    return objective


def _mf_objective(split: dict, evaluation_config: EvaluationConfig):
    def objective(trial) -> float:
        model_config = MatrixFactorizationConfig(
            factors=trial.suggest_int("factors", 24, 128, step=8),
            epochs=trial.suggest_int("epochs", 10, 40, step=5),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
            regularization=trial.suggest_float("regularization", 1e-3, 1e-1, log=True),
            batch_size=trial.suggest_categorical("batch_size", [4096, 8192, 16384]),
            optimizer="adam",
            backend="auto",
            device="auto",
        )
        model = MatrixFactorizationModel(model_config)
        model.fit(split["train"])
        candidate_movie_ids = candidate_movie_ids_from_split(
            split,
            evaluation_config,
            known_movie_ids=set(model.movie_to_index),
        )
        metrics = evaluate_model(model, split, evaluation_config, candidate_movie_ids=candidate_movie_ids)
        _set_metric_attrs(trial, metrics)
        return _objective_score(metrics)

    return objective


def _recall_objective(rows: list, config: BayesSearchConfig, db_path: Path):
    ranker_config = LearningToRankConfig(
        max_users=config.max_users,
        min_ratings_per_user=config.min_ratings_per_user,
        top_k=config.top_k,
        candidate_limit=config.candidate_limit,
    )
    split = split_by_user_three_way(rows, ranker_config)
    model = MatrixFactorizationModel(MatrixFactorizationConfig())
    model.fit(split["train"])
    quality_scores = movie_quality_scores(split)
    indexes = _build_collaborative_indexes(split["train"])
    feature_stats = _feature_stats(split["train"], db_path=db_path, model=model)
    recall_split = _candidate_split(split, include_test_relevant=False)

    def objective(trial) -> float:
        priors = _suggest_recall_priors(trial)
        rrf_k = trial.suggest_float("rrf_k", 20.0, 140.0)
        evaluation_config = EvaluationConfig(
            max_users=config.max_users,
            min_ratings_per_user=config.min_ratings_per_user,
            top_k=config.top_k,
            candidate_limit=config.candidate_limit,
            candidate_recall_strategy="weighted_rrf",
            candidate_recall_rrf_k=rrf_k,
            candidate_recall_priors=priors,
        )
        result = build_multi_route_candidate_pool(
            recall_split,
            evaluation_config,
            model,
            quality_scores=quality_scores,
            indexes=indexes,
            feature_stats=feature_stats,
        )
        metrics = _recall_metrics(result.source_rows)
        effective_min_route_recall = _effective_min_route_recall(metrics, config)
        metrics["min_route_recall"] = config.min_route_recall
        metrics["effective_min_route_recall"] = effective_min_route_recall
        _set_metric_attrs(trial, metrics)
        trial.set_user_attr("feasible", metrics["route_recall"] >= effective_min_route_recall)
        for source, floor, cap, base_weight in priors:
            trial.set_user_attr(f"{source}_floor", floor)
            trial.set_user_attr(f"{source}_cap", cap)
            trial.set_user_attr(f"{source}_base_weight", base_weight)
        return _recall_objective_score(metrics, effective_min_route_recall)

    return objective


def _suggest_recall_priors(trial) -> tuple[tuple[str, float, float, float], ...]:
    priors = []
    floor_ranges = {
        "popular_quality": (0.12, 0.45),
        "mf_user_top": (0.12, 0.85),
        "profile_centroid": (0.12, 0.95),
        "item_item_cf": (0.08, 0.75),
        "user_user_cf": (0.05, 0.55),
        "year_affinity": (0.10, 0.85),
    }
    cap_extra_ranges = {
        "popular_quality": (0.04, 0.40),
        "mf_user_top": (0.10, 0.75),
        "profile_centroid": (0.10, 0.85),
        "item_item_cf": (0.08, 0.65),
        "user_user_cf": (0.05, 0.45),
        "year_affinity": (0.10, 0.75),
    }
    weight_ranges = {
        "popular_quality": (0.75, 1.80),
        "mf_user_top": (0.55, 1.70),
        "profile_centroid": (0.45, 1.80),
        "item_item_cf": (0.45, 1.80),
        "user_user_cf": (0.35, 1.40),
        "year_affinity": (0.45, 1.80),
    }
    for source in SOURCE_NAMES:
        floor = trial.suggest_float(f"{source}_floor", *floor_ranges[source])
        cap = min(floor + trial.suggest_float(f"{source}_cap_extra", *cap_extra_ranges[source]), 1.50)
        base_weight = trial.suggest_float(f"{source}_base_weight", *weight_ranges[source])
        priors.append((source, floor, cap, base_weight))
    return tuple(priors)


def _recall_metrics(source_rows: list[dict]) -> dict:
    all_routes = next(row for row in source_rows if row["source"] == "all_routes")
    backfill = next(row for row in source_rows if row["source"] == "relevant_backfill")
    route_rows = [row for row in source_rows if row["source"] not in {"all_routes", "relevant_backfill"}]
    marginal_recall = sum(float(row.get("route_marginal_recall") or 0.0) for row in route_rows)
    return {
        "algorithm": "candidate_recall_weighted_rrf",
        "route_precision": float(all_routes.get("route_precision") or 0.0),
        "route_recall": float(all_routes.get("route_recall") or 0.0),
        "route_marginal_recall_sum": round(marginal_recall, 6),
        "backfill_recall": float(backfill.get("route_recall") or 0.0),
        "selected_before_backfill_movies": int(all_routes.get("selected_before_backfill_movies") or 0),
        "selected_movies": int(all_routes.get("selected_movies") or 0),
        "relevant_hits_before_backfill": int(all_routes.get("relevant_hits_before_backfill") or 0),
        "relevant_movies": int(all_routes.get("relevant_movies") or 0),
    }


def _recall_objective_score(metrics: dict, min_route_recall: float) -> float:
    route_recall = float(metrics["route_recall"])
    route_precision = float(metrics["route_precision"])
    marginal_recall = float(metrics["route_marginal_recall_sum"])
    if route_recall < min_route_recall:
        return (route_recall - min_route_recall) - 0.05 * max(min_route_recall - route_recall, 0.0)
    return route_precision + 0.25 * marginal_recall + 0.10 * route_recall


def _effective_min_route_recall(metrics: dict, config: BayesSearchConfig) -> float:
    relevant_movies = max(int(metrics.get("relevant_movies") or 0), 1)
    capacity_limit = min(float(config.candidate_limit) / relevant_movies, 1.0)
    return min(config.min_route_recall, capacity_limit * 0.95)


def _objective_score(metrics: dict) -> float:
    coverage = float(metrics.get("catalog_coverage") or 0.0)
    coverage_penalty = max(0.08 - coverage, 0.0) * 0.25
    return (
        float(metrics["precision_at_k"])
        + 0.25 * float(metrics["map_at_k"])
        + 0.10 * float(metrics["hit_rate_at_k"])
        - coverage_penalty
    )


def _set_metric_attrs(trial, metrics: dict) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float, str)):
            trial.set_user_attr(key, value)


def _trial_rows(study) -> list[dict]:
    rows = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": str(trial.state).split(".")[-1],
            "value": trial.value,
        }
        row.update(trial.params)
        row.update(trial.user_attrs)
        rows.append(row)
    return rows


def _best_payload(study, config: BayesSearchConfig) -> dict:
    best = study.best_trial
    best_metrics = {
        "mode": config.mode,
        "number": best.number,
        "objective_value": best.value,
        **best.params,
        **best.user_attrs,
    }
    return {
        "config": config.__dict__,
        "best_params": best.params,
        "best_metrics": best_metrics,
    }


def load_recall_search_settings(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    priors = []
    for source in SOURCE_NAMES:
        if f"{source}_floor" not in params:
            continue
        floor = float(params[f"{source}_floor"])
        if f"{source}_cap" in params:
            cap = float(params[f"{source}_cap"])
        else:
            cap = min(floor + float(params[f"{source}_cap_extra"]), 1.50)
        base_weight = float(params[f"{source}_base_weight"])
        priors.append((source, floor, cap, base_weight))
    return {
        "candidate_recall_strategy": "weighted_rrf",
        "candidate_recall_rrf_k": float(params["rrf_k"]),
        "candidate_recall_priors": tuple(priors),
    }


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
