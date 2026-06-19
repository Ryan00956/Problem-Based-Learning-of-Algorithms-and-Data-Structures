from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.datasets.netflix.evaluation import (
    EvaluationConfig,
    compare_recommenders,
    default_hybrid_weight_grid,
    grid_search_hybrid_weights,
    grid_search_matrix_factorization,
)
from src.datasets.netflix.import_duckdb import DEFAULT_DB_PATH
from src.datasets.netflix.matrix_factorization import MatrixFactorizationConfig


@dataclass(frozen=True)
class LargeSearchConfig:
    scale_user_counts: tuple[int, ...]
    mf_user_count: int
    hybrid_user_count: int
    min_ratings_per_user: int = 12
    candidate_limit: int = 1200
    top_k: int = 10


def smoke_large_search_config() -> LargeSearchConfig:
    return LargeSearchConfig(
        scale_user_counts=(40,),
        mf_user_count=40,
        hybrid_user_count=40,
        candidate_limit=300,
    )


def standard_large_search_config() -> LargeSearchConfig:
    return LargeSearchConfig(
        scale_user_counts=(160, 500, 1000),
        mf_user_count=500,
        hybrid_user_count=500,
        candidate_limit=1200,
    )


def run_large_search(
    output_dir: Path,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    config: LargeSearchConfig | None = None,
    mf_configs: list[MatrixFactorizationConfig] | None = None,
    hybrid_weight_grid: list[tuple[float, float, float]] | None = None,
) -> dict[str, Path]:
    config = config or standard_large_search_config()
    mf_configs = default_mf_search_grid() if mf_configs is None else mf_configs
    hybrid_weight_grid = default_large_hybrid_weight_grid() if hybrid_weight_grid is None else hybrid_weight_grid
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_csv = output_dir / "scale_comparison.csv"
    _run_scale_search(scale_csv, db_path, config)

    mf_csv = output_dir / "mf_search.csv"
    mf_training_csv = output_dir / "mf_search_training.csv"
    _run_mf_search(mf_csv, mf_training_csv, db_path, config, mf_configs)

    hybrid_csv = output_dir / "hybrid_search.csv"
    hybrid_training_csv = output_dir / "hybrid_search_training.csv"
    _run_hybrid_search(hybrid_csv, hybrid_training_csv, db_path, config, hybrid_weight_grid)

    summary_csv = output_dir / "best_summary.csv"
    _write_best_summary(summary_csv, [scale_csv, mf_csv, hybrid_csv])

    return {
        "scale_comparison": scale_csv,
        "mf_search": mf_csv,
        "mf_search_training": mf_training_csv,
        "hybrid_search": hybrid_csv,
        "hybrid_search_training": hybrid_training_csv,
        "best_summary": summary_csv,
    }


def default_mf_search_grid() -> list[MatrixFactorizationConfig]:
    values = []
    for factors in (32, 48, 64, 96):
        for epochs in (15, 20, 30):
            values.append(
                MatrixFactorizationConfig(
                    factors=factors,
                    epochs=epochs,
                    learning_rate=0.02 if factors <= 64 else 0.015,
                    regularization=0.04 if epochs <= 20 else 0.05,
                    backend="auto",
                    device="auto",
                    batch_size=8192,
                    optimizer="adam",
                )
            )
    return values


def default_large_hybrid_weight_grid() -> list[tuple[float, float, float]]:
    values = list(default_hybrid_weight_grid())
    for mf_weight in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        for collaborative_weight in (0.00, 0.03, 0.05, 0.08, 0.10):
            quality_weight = round(1.0 - mf_weight - collaborative_weight, 4)
            if quality_weight < 0.50:
                continue
            values.append((mf_weight, collaborative_weight, quality_weight))
    return _unique_weights(values)


def _run_scale_search(path: Path, db_path: Path, config: LargeSearchConfig) -> None:
    completed = _completed_trials(path)
    for user_count in config.scale_user_counts:
        trial_id = f"scale_users_{user_count}"
        if trial_id in completed:
            continue
        result = compare_recommenders(
            db_path,
            evaluation_config=_evaluation_config(config, user_count),
        )
        rows = []
        for row in result["metrics"]:
            rows.append(
                {
                    "trial_id": trial_id,
                    "search_stage": "scale_comparison",
                    "max_users": user_count,
                    **row,
                }
            )
        _append_rows(path, rows)


def _run_mf_search(
    path: Path,
    training_path: Path,
    db_path: Path,
    config: LargeSearchConfig,
    mf_configs: list[MatrixFactorizationConfig],
) -> None:
    completed = _completed_trials(path)
    for index, model_config in enumerate(mf_configs, start=1):
        trial_id = _mf_trial_id(index, model_config, config.mf_user_count)
        if trial_id in completed:
            continue
        result = grid_search_matrix_factorization(
            db_path,
            model_configs=[model_config],
            evaluation_config=_evaluation_config(config, config.mf_user_count),
        )
        rows = []
        for row in result["results"]:
            rows.append(
                {
                    "trial_id": trial_id,
                    "search_stage": "mf_search",
                    "max_users": config.mf_user_count,
                    **row,
                }
            )
        curve_rows = []
        for row in result["training_curves"]:
            curve_rows.append(
                {
                    "trial_id": trial_id,
                    "search_stage": "mf_search",
                    "max_users": config.mf_user_count,
                    **row,
                }
            )
        _append_rows(path, rows)
        _append_rows(training_path, curve_rows)


def _run_hybrid_search(
    path: Path,
    training_path: Path,
    db_path: Path,
    config: LargeSearchConfig,
    hybrid_weight_grid: list[tuple[float, float, float]],
) -> None:
    completed = _completed_trials(path)
    model_config = MatrixFactorizationConfig()
    for index, weights in enumerate(hybrid_weight_grid, start=1):
        trial_id = _hybrid_trial_id(index, weights, config.hybrid_user_count)
        if trial_id in completed:
            continue
        result = grid_search_hybrid_weights(
            db_path,
            model_config=model_config,
            evaluation_config=_evaluation_config(config, config.hybrid_user_count),
            weight_grid=[weights],
        )
        rows = []
        for row in result["results"]:
            rows.append(
                {
                    "trial_id": trial_id,
                    "search_stage": "hybrid_search",
                    "max_users": config.hybrid_user_count,
                    **row,
                }
            )
        curve_rows = []
        for row in result["training_curve"]:
            curve_rows.append(
                {
                    "trial_id": trial_id,
                    "search_stage": "hybrid_search",
                    "max_users": config.hybrid_user_count,
                    **row,
                }
            )
        _append_rows(path, rows)
        _append_rows(training_path, curve_rows)


def _write_best_summary(path: Path, source_paths: list[Path]) -> None:
    rows = []
    for source_path in source_paths:
        records = _read_rows(source_path)
        scored = [row for row in records if row.get("precision_at_k")]
        if not scored:
            continue
        best = max(
            scored,
            key=lambda row: (
                _float_value(row.get("precision_at_k")),
                _float_value(row.get("hit_rate_at_k")),
                _float_value(row.get("map_at_k")),
            ),
        )
        rows.append({"source": source_path.name, **best})
    _write_rows(path, rows)


def _evaluation_config(config: LargeSearchConfig, max_users: int) -> EvaluationConfig:
    return EvaluationConfig(
        max_users=max_users,
        min_ratings_per_user=config.min_ratings_per_user,
        top_k=config.top_k,
        candidate_limit=config.candidate_limit,
    )


def _completed_trials(path: Path) -> set[str]:
    return {
        row["trial_id"]
        for row in _read_rows(path)
        if row.get("trial_id")
    }


def _append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    existing_rows = _read_rows(path)
    _write_rows(path, existing_rows + rows)


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _fieldnames(rows: Iterable[dict]) -> list[str]:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _mf_trial_id(index: int, config: MatrixFactorizationConfig, max_users: int) -> str:
    return (
        f"mf_{index:03d}_users_{max_users}_f{config.factors}_e{config.epochs}_"
        f"lr{config.learning_rate}_reg{config.regularization}_{config.optimizer}"
    )


def _hybrid_trial_id(index: int, weights: tuple[float, float, float], max_users: int) -> str:
    mf_weight, collaborative_weight, quality_weight = weights
    return (
        f"hybrid_{index:03d}_users_{max_users}_"
        f"mf{mf_weight}_cf{collaborative_weight}_q{quality_weight}"
    )


def _unique_weights(values: Iterable[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    seen = set()
    result = []
    for mf_weight, collaborative_weight, quality_weight in values:
        item = (round(mf_weight, 4), round(collaborative_weight, 4), round(quality_weight, 4))
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _float_value(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
