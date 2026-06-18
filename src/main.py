from __future__ import annotations

import argparse
from pathlib import Path

from src.core.registry import DATASETS, get_dataset
from src.datasets.movielens.experiment import OUTPUT_SUBDIR as MOVIELENS_OUTPUT_DIR


def command_demo(args: argparse.Namespace) -> None:
    pipeline = get_dataset(args.dataset)
    pipeline.run_demo(args.n)


def command_top(args: argparse.Namespace) -> None:
    pipeline = get_dataset(args.dataset)
    rows = pipeline.show_top(args.n, args.algorithm)
    pipeline.print_movies(rows, args.n)


def command_search(args: argparse.Namespace) -> None:
    pipeline = get_dataset(args.dataset)
    rows = pipeline.search(args.kind, args.query, args.n)
    pipeline.print_movies(rows, args.n)


def command_recommend(args: argparse.Namespace) -> None:
    pipeline = get_dataset(args.dataset)
    target, rows = pipeline.recommend(args.title, args.n)
    if target is None:
        print("No target movie found.")
        return
    print(f"Target: {target['title']}")
    pipeline.print_movies(rows, args.n)


def command_experiment(args: argparse.Namespace) -> None:
    pipeline = get_dataset(args.dataset)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.dataset)
    paths = pipeline.run_experiments(output_dir)
    print(f"{pipeline.display_name} experiment outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


def default_output_dir(dataset: str) -> Path:
    if dataset == "movielens":
        return MOVIELENS_OUTPUT_DIR
    return Path("output") / dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Movie streaming recommendation course project")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="movielens")
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

    experiment = sub.add_parser("experiment", help="Run runtime experiments")
    experiment.add_argument("--output-dir", default=None)
    experiment.set_defaults(func=command_experiment)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        args.func = command_demo
        args.n = 10
    try:
        args.func(args)
    except NotImplementedError as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
