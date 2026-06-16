from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

from src.data_loader import (
    OUTPUT_DIR,
    build_movie_profiles,
    build_netflix_sample_profiles,
    load_movielens,
    save_profiles_csv,
)
from src.search import MovieSearchEngine
from src.sorting import heap_sort, merge_sort


def timed_call(func: Callable, *args, repeat: int = 3, **kwargs) -> tuple[object, float]:
    best = None
    best_seconds = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if elapsed < best_seconds:
            best = result
            best_seconds = elapsed
    return best, best_seconds


def run_experiments(output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    movies, ratings, tags = load_movielens()
    profiles = build_movie_profiles(movies, ratings, tags)
    save_profiles_csv(profiles, output_dir / "movie_profiles.csv")

    sort_results = []
    sizes = [100, 500, 1000, 5000, len(profiles)]
    for size in sizes:
        subset = profiles[:size]
        _, merge_seconds = timed_call(merge_sort, subset, key="comprehensive_score", reverse=True)
        _, heap_seconds = timed_call(heap_sort, subset, key="comprehensive_score", reverse=True)
        sort_results.append(
            {
                "task": "sort_by_comprehensive_score",
                "data_size": size,
                "merge_sort_seconds": round(merge_seconds, 8),
                "heap_sort_seconds": round(heap_seconds, 8),
            }
        )

    sort_csv = output_dir / "sorting_runtime.csv"
    _write_csv(sort_csv, sort_results)

    netflix_profiles = build_netflix_sample_profiles(limit_movies=200)
    save_profiles_csv(netflix_profiles, output_dir / "netflix_sample_profiles.csv")
    netflix_sort_results = []
    for size in [50, 100, 200]:
        subset = netflix_profiles[:size]
        _, merge_seconds = timed_call(merge_sort, subset, key="comprehensive_score", reverse=True)
        _, heap_seconds = timed_call(heap_sort, subset, key="comprehensive_score", reverse=True)
        netflix_sort_results.append(
            {
                "task": "netflix_sample_sort_by_score",
                "data_size": size,
                "merge_sort_seconds": round(merge_seconds, 8),
                "heap_sort_seconds": round(heap_seconds, 8),
            }
        )
    _write_csv(output_dir / "netflix_sorting_runtime.csv", netflix_sort_results)

    engine = MovieSearchEngine(profiles)
    search_cases = [
        ("title", "Toy Story", engine.linear_title_search, engine.index_title_search),
        ("genre", "Comedy", engine.linear_genre_search, engine.index_genre_search),
        ("tag", "funny", engine.linear_tag_search, engine.index_tag_search),
    ]
    search_results = []
    for kind, query, linear_func, index_func in search_cases:
        linear_matches, linear_seconds = timed_call(linear_func, query, repeat=10)
        index_matches, index_seconds = timed_call(index_func, query, repeat=10)
        search_results.append(
            {
                "query_type": kind,
                "query": query,
                "linear_seconds": round(linear_seconds, 8),
                "index_seconds": round(index_seconds, 8),
                "linear_result_count": len(linear_matches),
                "index_result_count": len(index_matches),
            }
        )

    search_csv = output_dir / "search_runtime.csv"
    _write_csv(search_csv, search_results)

    chart_svg = output_dir / "runtime_chart.svg"
    _write_sort_svg(sort_results, chart_svg)

    return {
        "profiles": output_dir / "movie_profiles.csv",
        "netflix_sample_profiles": output_dir / "netflix_sample_profiles.csv",
        "sorting_runtime": sort_csv,
        "netflix_sorting_runtime": output_dir / "netflix_sorting_runtime.csv",
        "search_runtime": search_csv,
        "runtime_chart": chart_svg,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_sort_svg(rows: list[dict], path: Path) -> None:
    width = 860
    height = 460
    margin_left = 80
    margin_bottom = 70
    plot_width = width - 130
    plot_height = height - 130
    max_time = max(max(row["merge_sort_seconds"], row["heap_sort_seconds"]) for row in rows) or 1.0
    max_size = max(row["data_size"] for row in rows) or 1

    def x(size: int) -> float:
        return margin_left + (size / max_size) * plot_width

    def y(seconds: float) -> float:
        return height - margin_bottom - (seconds / max_time) * plot_height

    merge_points = " ".join(f"{x(row['data_size']):.2f},{y(row['merge_sort_seconds']):.2f}" for row in rows)
    heap_points = " ".join(f"{x(row['data_size']):.2f},{y(row['heap_sort_seconds']):.2f}" for row in rows)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="80" y="35" font-size="22" font-family="Arial" fill="#222">Sorting Runtime Comparison</text>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-50}" y2="{height-margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="60" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#333"/>',
        f'<polyline points="{merge_points}" fill="none" stroke="#1f77b4" stroke-width="3"/>',
        f'<polyline points="{heap_points}" fill="none" stroke="#d62728" stroke-width="3"/>',
        '<text x="650" y="80" font-size="14" font-family="Arial" fill="#1f77b4">Merge sort</text>',
        '<text x="650" y="105" font-size="14" font-family="Arial" fill="#d62728">Heap sort</text>',
        '<text x="360" y="430" font-size="14" font-family="Arial" fill="#333">Data size</text>',
        '<text x="15" y="230" font-size="14" font-family="Arial" fill="#333" transform="rotate(-90 15,230)">Seconds</text>',
    ]
    for row in rows:
        label = str(row["data_size"])
        lines.append(f'<text x="{x(row["data_size"])-15:.2f}" y="410" font-size="11" font-family="Arial">{label}</text>')
        lines.append(f'<circle cx="{x(row["data_size"]):.2f}" cy="{y(row["merge_sort_seconds"]):.2f}" r="4" fill="#1f77b4"/>')
        lines.append(f'<circle cx="{x(row["data_size"]):.2f}" cy="{y(row["heap_sort_seconds"]):.2f}" r="4" fill="#d62728"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")
