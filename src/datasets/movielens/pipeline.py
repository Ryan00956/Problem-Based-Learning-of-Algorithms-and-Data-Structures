from __future__ import annotations

from pathlib import Path

from src.core.pipeline import DatasetPipeline
from src.datasets.movielens.experiment import OUTPUT_SUBDIR, run_experiments
from src.datasets.movielens.export_frontend import export_frontend_data
from src.datasets.movielens.loader import load_movielens
from src.datasets.movielens.profiles import build_movie_profiles, save_profiles_csv
from src.datasets.movielens.recommendation import recommend_similar_movies, top_n_movies
from src.datasets.movielens.search import MovieLensSearchEngine


def load_profiles() -> list[dict]:
    movies, ratings, tags = load_movielens()
    return build_movie_profiles(movies, ratings, tags)


def print_movies(rows: list[dict], limit: int = 10) -> None:
    for index, item in enumerate(rows[:limit], 1):
        genres = "|".join(item["genres"][:4])
        print(
            f"{index:>2}. {item['title']} | score={item.get('comprehensive_score', 0):.2f} "
            f"| rating={item['avg_rating']:.2f} | count={item['rating_count']} | genres={genres}"
        )


def run_demo(n: int = 10) -> None:
    OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles()
    save_profiles_csv(profiles, OUTPUT_SUBDIR / "movie_profiles.csv")
    engine = MovieLensSearchEngine(profiles)

    print("\nTop movies by Top-N heap:")
    print_movies(top_n_movies(profiles, n=n, algorithm="heap"), n)

    print("\nTop movies by merge sort:")
    print_movies(top_n_movies(profiles, n=n, algorithm="merge"), n)

    print("\nIndexed title search: Toy Story")
    print_movies(engine.index_title_search("Toy Story"), 5)

    print("\nIndexed genre search: Comedy")
    print_movies(engine.index_genre_search("Comedy"), 5)

    print("\nIndexed tag search: funny")
    print_movies(engine.index_tag_search("funny"), 5)

    print("\nSimilar movie recommendation for Toy Story:")
    target, recommendations = recommend_similar_movies("Toy Story", profiles, engine, n=n)
    if target is None:
        print("No target movie found.")
    else:
        print(f"Target: {target['title']}")
        print_movies(recommendations, n)

    print(f"\nSaved profiles to: {OUTPUT_SUBDIR / 'movie_profiles.csv'}")


def show_top(n: int = 10, algorithm: str = "heap") -> list[dict]:
    return top_n_movies(load_profiles(), n=n, algorithm=algorithm)


def search(kind: str, query: str, n: int = 10) -> list[dict]:
    profiles = load_profiles()
    engine = MovieLensSearchEngine(profiles)
    if kind == "title":
        rows = engine.index_title_search(query)
    elif kind == "genre":
        rows = engine.index_genre_search(query)
    elif kind == "tag":
        rows = engine.index_tag_search(query)
    else:
        raise ValueError(kind)
    return rows[:n]


def recommend(title: str, n: int = 10) -> tuple[dict | None, list[dict]]:
    profiles = load_profiles()
    engine = MovieLensSearchEngine(profiles)
    return recommend_similar_movies(title, profiles, engine, n=n)


def run_experiments_for_pipeline(output_dir: Path) -> dict[str, Path]:
    return run_experiments(output_dir)


PIPELINE = DatasetPipeline(
    name="movielens",
    display_name="MovieLens",
    load_profiles=load_profiles,
    print_movies=print_movies,
    run_demo=run_demo,
    show_top=show_top,
    search=search,
    recommend=recommend,
    run_experiments=run_experiments_for_pipeline,
    export_frontend_data=export_frontend_data,
)
