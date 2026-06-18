from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from src.core.paths import WEB_DATA_DIR
from src.datasets.movielens.experiment import OUTPUT_SUBDIR
from src.datasets.movielens.loader import load_movielens
from src.datasets.movielens.profiles import build_movie_profiles
from src.datasets.movielens.recommendation import top_n_movies
from src.datasets.movielens.tag_aliases import load_accepted_tag_aliases


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def export_frontend_data() -> dict[str, Path]:
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    movies, ratings, tags = load_movielens()
    profiles = build_movie_profiles(movies, ratings, tags, tag_aliases=load_accepted_tag_aliases())
    ranked = top_n_movies(profiles, n=len(profiles), algorithm="heap")

    searchable = sorted(profiles, key=lambda item: item["rating_count"], reverse=True)[:1200]
    payload = {
        "dataset": {
            "name": "movielens",
            "display_name": "MovieLens",
        },
        "summary": {
            "movie_count": len(profiles),
            "rating_count": int(len(ratings)),
            "tag_count": int(sum(len(item["tags"]) for item in profiles)),
            "user_count": int(ratings["userId"].nunique()),
            "top_algorithm": "top_n_heap",
        },
        "topMovies": ranked[:100],
        "movies": searchable,
        "sortRuntime": _read_csv(OUTPUT_SUBDIR / "sorting_runtime.csv"),
        "searchRuntime": _read_csv(OUTPUT_SUBDIR / "search_runtime.csv"),
    }

    dataset_path = WEB_DATA_DIR / "movielens-dashboard-data.json"
    dataset_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    default_path = WEB_DATA_DIR / "dashboard-data.json"
    shutil.copyfile(dataset_path, default_path)
    return {"dashboard_data": default_path, "movielens_dashboard_data": dataset_path}
