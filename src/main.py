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


def command_large_search(args: argparse.Namespace) -> None:
    if args.dataset != "netflix":
        raise NotImplementedError("large-search is currently implemented for --dataset netflix only.")
    from src.datasets.netflix.large_search import (
        run_large_search,
        smoke_large_search_config,
        standard_large_search_config,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path("output") / "netflix_large_search"
    config = smoke_large_search_config() if args.profile == "smoke" else standard_large_search_config()
    paths = run_large_search(output_dir, config=config)
    print("Netflix large search outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


def command_bayes_search(args: argparse.Namespace) -> None:
    if args.dataset != "netflix":
        raise NotImplementedError("bayes-search is currently implemented for --dataset netflix only.")
    from src.datasets.netflix.bayes_search import (
        run_bayes_search,
        smoke_bayes_search_config,
        standard_bayes_search_config,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path("output") / "netflix_bayes_search"
    if args.profile == "smoke":
        config = smoke_bayes_search_config(args.mode)
    else:
        config = standard_bayes_search_config(args.mode, args.trials)
    paths = run_bayes_search(output_dir, config=config)
    print("Netflix Optuna Bayesian search outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


def command_rerank(args: argparse.Namespace) -> None:
    if args.dataset != "netflix":
        raise NotImplementedError("rerank is currently implemented for --dataset netflix only.")
    from dataclasses import replace

    from src.datasets.netflix.bayes_search import load_recall_search_settings
    from src.datasets.netflix.learning_to_rank import (
        run_learning_to_rank,
        smoke_learning_to_rank_config,
        standard_learning_to_rank_config,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path("output") / "netflix_learning_to_rank"
    config = smoke_learning_to_rank_config() if args.profile == "smoke" else standard_learning_to_rank_config()
    if args.recall_params:
        config = replace(config, **load_recall_search_settings(Path(args.recall_params)))
    elif args.recall_strategy:
        config = replace(config, candidate_recall_strategy=args.recall_strategy)
    if args.negative_sampling:
        config = replace(config, negative_sampling=args.negative_sampling)
    if args.blend_mode:
        config = replace(config, blend_mode=args.blend_mode)
    if args.min_blend_precision_gain is not None:
        config = replace(config, min_blend_precision_gain=args.min_blend_precision_gain)
    if args.max_users is not None:
        config = replace(config, max_users=args.max_users)
    if args.experimental_algorithm_features:
        config = replace(config, experimental_algorithm_features=True)
    if args.evaluate_pairwise:
        config = replace(config, evaluate_pairwise_ranker=True)
    if args.evaluate_residual:
        config = replace(config, evaluate_residual_ranker=True)
    if args.evaluate_mmr:
        config = replace(config, evaluate_mmr_ranker=True)
    paths = run_learning_to_rank(output_dir, config=config)
    print("Netflix learning-to-rank outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


def command_neural_rerank(args: argparse.Namespace) -> None:
    if args.dataset != "netflix":
        raise NotImplementedError("neural-rerank is currently implemented for --dataset netflix only.")
    from dataclasses import replace

    from src.datasets.netflix.bayes_search import load_recall_search_settings
    from src.datasets.netflix.neural_reranker import (
        run_neural_reranker,
        smoke_neural_reranker_config,
        standard_neural_reranker_config,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path("output") / "netflix_neural_reranker"
    config = smoke_neural_reranker_config() if args.profile == "smoke" else standard_neural_reranker_config()
    if args.recall_params:
        config = replace(config, **load_recall_search_settings(Path(args.recall_params)))
    elif args.recall_strategy:
        config = replace(config, candidate_recall_strategy=args.recall_strategy)
    updates = {}
    if args.negative_sampling:
        updates["negative_sampling"] = args.negative_sampling
    if args.pair_negative_sampling:
        updates["pair_negative_sampling"] = args.pair_negative_sampling
    if args.blend_mode:
        updates["blend_mode"] = args.blend_mode
    if args.min_blend_precision_gain is not None:
        updates["min_blend_precision_gain"] = args.min_blend_precision_gain
    if args.max_users is not None:
        updates["max_users"] = args.max_users
    if updates:
        config = replace(config, **updates)
    paths = run_neural_reranker(output_dir, config=config)
    print("Netflix neural reranker outputs:")
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

    large_search = sub.add_parser("large-search", help="Run resumable Netflix model-search experiments")
    large_search.add_argument("--output-dir", default=None)
    large_search.add_argument("--profile", choices=["smoke", "standard"], default="standard")
    large_search.set_defaults(func=command_large_search)

    bayes_search = sub.add_parser("bayes-search", help="Run resumable Optuna Bayesian search")
    bayes_search.add_argument("--output-dir", default=None)
    bayes_search.add_argument("--profile", choices=["smoke", "standard"], default="standard")
    bayes_search.add_argument("--mode", choices=["hybrid", "mf", "recall"], default="hybrid")
    bayes_search.add_argument("--trials", type=int, default=30)
    bayes_search.set_defaults(func=command_bayes_search)

    rerank = sub.add_parser("rerank", help="Train and evaluate a Netflix learning-to-rank reranker")
    rerank.add_argument("--output-dir", default=None)
    rerank.add_argument("--profile", choices=["smoke", "standard"], default="standard")
    rerank.add_argument("--recall-strategy", choices=["legacy", "weighted_rrf"], default=None)
    rerank.add_argument("--recall-params", default=None, help="Path to recall bayes-search best_params.json")
    rerank.add_argument("--negative-sampling", choices=["hybrid_hard", "explicit_hard"], default=None)
    rerank.add_argument("--blend-mode", choices=["value", "rank", "auto"], default=None)
    rerank.add_argument("--min-blend-precision-gain", type=float, default=None)
    rerank.add_argument("--max-users", type=int, default=None)
    rerank.add_argument("--experimental-algorithm-features", action="store_true")
    rerank.add_argument("--evaluate-pairwise", action="store_true")
    rerank.add_argument("--evaluate-residual", action="store_true")
    rerank.add_argument("--evaluate-mmr", action="store_true")
    rerank.set_defaults(func=command_rerank)

    neural_rerank = sub.add_parser("neural-rerank", help="Train and evaluate a Netflix PyTorch neural reranker")
    neural_rerank.add_argument("--output-dir", default=None)
    neural_rerank.add_argument("--profile", choices=["smoke", "standard"], default="standard")
    neural_rerank.add_argument("--recall-strategy", choices=["legacy", "weighted_rrf"], default=None)
    neural_rerank.add_argument("--recall-params", default=None, help="Path to recall bayes-search best_params.json")
    neural_rerank.add_argument("--negative-sampling", choices=["hybrid_hard", "explicit_hard"], default=None)
    neural_rerank.add_argument("--pair-negative-sampling", choices=["random", "hard"], default=None)
    neural_rerank.add_argument("--blend-mode", choices=["value", "rank", "auto"], default=None)
    neural_rerank.add_argument("--min-blend-precision-gain", type=float, default=None)
    neural_rerank.add_argument("--max-users", type=int, default=None)
    neural_rerank.set_defaults(func=command_neural_rerank)

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
