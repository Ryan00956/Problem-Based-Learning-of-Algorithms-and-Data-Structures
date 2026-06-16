from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import OUTPUT_DIR, build_movie_profiles, load_movielens, save_profiles_csv
from src.experiment import run_experiments
from src.recommendation import recommend_similar_movies, top_n_movies
from src.search import MovieSearchEngine


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


def command_demo(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles()
    save_profiles_csv(profiles, OUTPUT_DIR / "movie_profiles.csv")
    engine = MovieSearchEngine(profiles)

    print("\nTop movies by heap sort:")
    top_heap = top_n_movies(profiles, n=args.n, algorithm="heap")
    print_movies(top_heap, args.n)

    print("\nTop movies by merge sort:")
    top_merge = top_n_movies(profiles, n=args.n, algorithm="merge")
    print_movies(top_merge, args.n)

    print("\nIndexed title search: Toy Story")
    print_movies(engine.index_title_search("Toy Story"), 5)

    print("\nIndexed genre search: Comedy")
    print_movies(engine.index_genre_search("Comedy"), 5)

    print("\nIndexed tag search: funny")
    print_movies(engine.index_tag_search("funny"), 5)

    print("\nSimilar movie recommendation for Toy Story:")
    target, recommendations = recommend_similar_movies("Toy Story", profiles, engine, n=args.n)
    if target is None:
        print("No target movie found.")
    else:
        print(f"Target: {target['title']}")
        print_movies(recommendations, args.n)

    print(f"\nSaved profiles to: {OUTPUT_DIR / 'movie_profiles.csv'}")


def command_top(args: argparse.Namespace) -> None:
    profiles = load_profiles()
    print_movies(top_n_movies(profiles, n=args.n, algorithm=args.algorithm), args.n)


def command_search(args: argparse.Namespace) -> None:
    profiles = load_profiles()
    engine = MovieSearchEngine(profiles)
    if args.kind == "title":
        rows = engine.index_title_search(args.query)
    elif args.kind == "genre":
        rows = engine.index_genre_search(args.query)
    elif args.kind == "tag":
        rows = engine.index_tag_search(args.query)
    else:
        raise ValueError(args.kind)
    print_movies(rows, args.n)


def command_recommend(args: argparse.Namespace) -> None:
    profiles = load_profiles()
    engine = MovieSearchEngine(profiles)
    target, rows = recommend_similar_movies(args.title, profiles, engine, n=args.n)
    if target is None:
        print("No target movie found.")
        return
    print(f"Target: {target['title']}")
    print_movies(rows, args.n)


def command_experiment(args: argparse.Namespace) -> None:
    paths = run_experiments(Path(args.output_dir))
    print("Experiment outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Movie streaming recommendation course project")
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="Run a full feature demonstration")
    demo.add_argument("-n", type=int, default=10)
    demo.set_defaults(func=command_demo)

    top = sub.add_parser("top", help="Show Top-N movies")
    top.add_argument("-n", type=int, default=10)
    top.add_argument("--algorithm", choices=["heap", "merge"], default="heap")
    top.set_defaults(func=command_top)

    search = sub.add_parser("search", help="Search by title, genre, or tag")
    search.add_argument("kind", choices=["title", "genre", "tag"])
    search.add_argument("query")
    search.add_argument("-n", type=int, default=10)
    search.set_defaults(func=command_search)

    recommend = sub.add_parser("recommend", help="Recommend similar movies")
    recommend.add_argument("title")
    recommend.add_argument("-n", type=int, default=10)
    recommend.set_defaults(func=command_recommend)

    experiment = sub.add_parser("experiment", help="Run sorting and search runtime experiments")
    experiment.add_argument("--output-dir", default=str(OUTPUT_DIR))
    experiment.set_defaults(func=command_experiment)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        args = parser.parse_args(["demo"])
    args.func(args)


if __name__ == "__main__":
    main()
