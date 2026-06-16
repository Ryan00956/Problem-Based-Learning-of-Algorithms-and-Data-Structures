from __future__ import annotations

import csv
import json
from pathlib import Path

from src.data_loader import OUTPUT_DIR, PROJECT_ROOT, build_movie_profiles, load_movielens
from src.recommendation import top_n_movies


WEB_DATA_DIR = PROJECT_ROOT / "web" / "data"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def export_frontend_data() -> dict[str, Path]:
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    movies, ratings, tags = load_movielens()
    profiles = build_movie_profiles(movies, ratings, tags)
    ranked = top_n_movies(profiles, n=len(profiles), algorithm="heap")

    # Keep the frontend payload compact but still large enough for visible search.
    searchable = sorted(profiles, key=lambda item: item["rating_count"], reverse=True)[:1200]
    payload = {
        "summary": {
            "movie_count": len(profiles),
            "rating_count": int(len(ratings)),
            "tag_count": int(len(tags)),
            "user_count": int(ratings["userId"].nunique()),
            "top_algorithm": "heap_sort",
        },
        "topMovies": ranked[:100],
        "movies": searchable,
        "sortRuntime": _read_csv(OUTPUT_DIR / "sorting_runtime.csv"),
        "searchRuntime": _read_csv(OUTPUT_DIR / "search_runtime.csv"),
        "netflixRuntime": _read_csv(OUTPUT_DIR / "netflix_sorting_runtime.csv"),
    }

    path = WEB_DATA_DIR / "dashboard-data.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"dashboard_data": path}


def main() -> None:
    for name, path in export_frontend_data().items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
