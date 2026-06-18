from __future__ import annotations

from pathlib import Path

from src.core.pipeline import DatasetPipeline
from src.datasets.netflix.experiment import run_experiments
from src.datasets.netflix.scoring import build_movie_scores, load_movie_scores, rank_movie_scores, top_movie_scores


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


def _print_movies(rows: list[dict], n: int) -> None:
    for index, movie in enumerate(rows[:n], start=1):
        year = movie["release_year"] if movie["release_year"] is not None else "unknown"
        print(
            f"{index:>2}. {movie['title']} ({year}) "
            f"score={movie['comprehensive_score']:.2f} "
            f"bayes={movie['bayesian_rating']:.2f} "
            f"avg={movie['avg_rating']:.2f} "
            f"ratings={movie['rating_count']} "
            f"recent={movie['recent_rating_count']}"
        )


def _show_top(n: int, algorithm: str = "heap") -> list[dict]:
    if algorithm == "heap":
        return top_movie_scores(n=n)
    return rank_movie_scores(load_movie_scores(), n=n, algorithm=algorithm)


def _run_demo(n: int = 10) -> None:
    summary = build_movie_scores()
    print("Netflix Prize scoring summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print()
    print(f"Top {n} movies:")
    _print_movies(top_movie_scores(n=n), n)


def _run_experiments_for_pipeline(output_dir: Path) -> dict[str, Path]:
    return run_experiments(output_dir)


PIPELINE = DatasetPipeline(
    name="netflix",
    display_name="Netflix Prize",
    load_profiles=_not_implemented,
    print_movies=_print_movies,
    run_demo=_run_demo,
    show_top=_show_top,
    search=_not_implemented,
    recommend=_not_implemented_recommend,
    run_experiments=_run_experiments_for_pipeline,
    export_frontend_data=_not_implemented_export,
)
