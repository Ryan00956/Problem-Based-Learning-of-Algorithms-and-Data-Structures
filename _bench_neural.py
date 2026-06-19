from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from src.datasets.netflix.neural_reranker import (
    run_neural_reranker,
    standard_neural_reranker_config,
)

METRIC_KEYS = (
    "algorithm",
    "precision_at_k",
    "recall_at_k",
    "hit_rate_at_k",
    "map_at_k",
    "catalog_coverage",
)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    tag = sys.argv[2] if len(sys.argv) > 2 else "neural"
    config = replace(standard_neural_reranker_config(), max_users=n)
    out = Path("output") / f"_bench_neural_{n}_{tag}"
    start = time.perf_counter()
    run_neural_reranker(out, config=config)
    elapsed = time.perf_counter() - start
    with (out / "metrics.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    summary = {
        "max_users": n,
        "tag": tag,
        "elapsed_s": round(elapsed, 2),
        "metrics": [{k: r.get(k) for k in METRIC_KEYS} for r in rows],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
