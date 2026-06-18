from __future__ import annotations

from src.core.pipeline import DatasetPipeline
from src.datasets.movielens.pipeline import PIPELINE as MOVIELENS_PIPELINE
from src.datasets.netflix.pipeline import PIPELINE as NETFLIX_PIPELINE


DATASETS: dict[str, DatasetPipeline] = {
    MOVIELENS_PIPELINE.name: MOVIELENS_PIPELINE,
    NETFLIX_PIPELINE.name: NETFLIX_PIPELINE,
}


def get_dataset(name: str) -> DatasetPipeline:
    try:
        return DATASETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASETS))
        raise ValueError(f"unknown dataset: {name}. Available datasets: {available}") from exc
