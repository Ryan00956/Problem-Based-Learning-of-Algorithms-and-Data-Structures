from __future__ import annotations

from pathlib import Path

from src.core.pipeline import DatasetPipeline


def _not_implemented(*args, **kwargs):
    raise NotImplementedError(
        "Netflix Prize support is intentionally isolated and has not been implemented yet. "
        "Use --dataset movielens until the Netflix-specific loader and algorithms are added."
    )


def _not_implemented_recommend(title: str, n: int = 10) -> tuple[dict | None, list[dict]]:
    _not_implemented(title, n)


def _not_implemented_experiments(output_dir: Path) -> dict[str, Path]:
    _not_implemented(output_dir)


def _not_implemented_export() -> dict[str, Path]:
    _not_implemented()


PIPELINE = DatasetPipeline(
    name="netflix",
    display_name="Netflix Prize",
    load_profiles=_not_implemented,
    print_movies=_not_implemented,
    run_demo=_not_implemented,
    show_top=_not_implemented,
    search=_not_implemented,
    recommend=_not_implemented_recommend,
    run_experiments=_not_implemented_experiments,
    export_frontend_data=_not_implemented_export,
)
