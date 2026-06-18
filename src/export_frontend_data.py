from __future__ import annotations

import argparse

from src.core.registry import DATASETS, get_dataset


def export_frontend_data(dataset: str = "movielens"):
    return get_dataset(dataset).export_frontend_data()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export frontend dashboard data")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="movielens")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name, path in export_frontend_data(args.dataset).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
