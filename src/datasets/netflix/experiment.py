from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

from src.algorithms.sorting import heap_sort, merge_sort, top_n_heap
from src.core.paths import OUTPUT_DIR
from src.datasets.netflix.scoring import build_movie_scores, load_movie_scores


OUTPUT_SUBDIR = OUTPUT_DIR / "netflix"


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


def run_experiments(output_dir: Path = OUTPUT_SUBDIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    build_movie_scores()
    scores = load_movie_scores()

    score_csv = output_dir / "movie_scores.csv"
    _write_csv(score_csv, scores)

    sort_results = []
    sizes = _experiment_sizes(len(scores))
    for size in sizes:
        subset = scores[:size]
        _, merge_seconds = timed_call(merge_sort, subset, key="comprehensive_score", reverse=True)
        _, heap_seconds = timed_call(heap_sort, subset, key="comprehensive_score", reverse=True)
        _, top_n_heap_seconds = timed_call(top_n_heap, subset, 10, key="comprehensive_score", reverse=True)
        sort_results.append(
            {
                "task": "sort_by_comprehensive_score",
                "data_size": size,
                "top_n": 10,
                "merge_sort_seconds": round(merge_seconds, 8),
                "heap_sort_seconds": round(heap_seconds, 8),
                "top_n_heap_seconds": round(top_n_heap_seconds, 8),
            }
        )

    sort_csv = output_dir / "sorting_runtime.csv"
    _write_csv(sort_csv, sort_results)

    chart_svg = output_dir / "runtime_chart.svg"
    _write_sort_svg(sort_results, chart_svg)

    return {
        "scores": score_csv,
        "sorting_runtime": sort_csv,
        "runtime_chart": chart_svg,
    }


def _experiment_sizes(total: int) -> list[int]:
    candidates = [100, 500, 1000, 5000, 10000, total]
    return sorted({size for size in candidates if 0 < size <= total})


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
    max_time = max(
        max(row["merge_sort_seconds"], row["heap_sort_seconds"], row["top_n_heap_seconds"])
        for row in rows
    ) or 1.0
    max_size = max(row["data_size"] for row in rows) or 1

    def x(size: int) -> float:
        return margin_left + (size / max_size) * plot_width

    def y(seconds: float) -> float:
        return height - margin_bottom - (seconds / max_time) * plot_height

    merge_points = " ".join(f"{x(row['data_size']):.2f},{y(row['merge_sort_seconds']):.2f}" for row in rows)
    heap_points = " ".join(f"{x(row['data_size']):.2f},{y(row['heap_sort_seconds']):.2f}" for row in rows)
    top_n_points = " ".join(f"{x(row['data_size']):.2f},{y(row['top_n_heap_seconds']):.2f}" for row in rows)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="80" y="35" font-size="22" font-family="Arial" fill="#222">Netflix Sorting Runtime Comparison</text>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-50}" y2="{height-margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="60" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#333"/>',
        f'<polyline points="{merge_points}" fill="none" stroke="#1f77b4" stroke-width="3"/>',
        f'<polyline points="{heap_points}" fill="none" stroke="#d62728" stroke-width="3"/>',
        f'<polyline points="{top_n_points}" fill="none" stroke="#2ca02c" stroke-width="3"/>',
        '<text x="650" y="80" font-size="14" font-family="Arial" fill="#1f77b4">Merge sort</text>',
        '<text x="650" y="105" font-size="14" font-family="Arial" fill="#d62728">Heap sort</text>',
        '<text x="650" y="130" font-size="14" font-family="Arial" fill="#2ca02c">Top-N heap</text>',
        '<text x="360" y="430" font-size="14" font-family="Arial" fill="#333">Data size</text>',
        '<text x="15" y="230" font-size="14" font-family="Arial" fill="#333" transform="rotate(-90 15,230)">Seconds</text>',
    ]
    for row in rows:
        label = str(row["data_size"])
        lines.append(f'<text x="{x(row["data_size"])-15:.2f}" y="410" font-size="11" font-family="Arial">{label}</text>')
        lines.append(f'<circle cx="{x(row["data_size"]):.2f}" cy="{y(row["merge_sort_seconds"]):.2f}" r="4" fill="#1f77b4"/>')
        lines.append(f'<circle cx="{x(row["data_size"]):.2f}" cy="{y(row["heap_sort_seconds"]):.2f}" r="4" fill="#d62728"/>')
        lines.append(f'<circle cx="{x(row["data_size"]):.2f}" cy="{y(row["top_n_heap_seconds"]):.2f}" r="4" fill="#2ca02c"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")
