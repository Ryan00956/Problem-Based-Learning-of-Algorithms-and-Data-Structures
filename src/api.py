from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from src.core.paths import OUTPUT_DIR, PROJECT_ROOT
from src.core.registry import DATASETS
from src.datasets.movielens.loader import load_movielens
from src.datasets.movielens.profiles import build_movie_profiles
from src.datasets.movielens.recommendation import recommend_similar_movies, top_n_movies
from src.datasets.movielens.search import MovieLensSearchEngine


WEB_DIR = PROJECT_ROOT / "web"


class MovieLensApiService:
    def __init__(self) -> None:
        self._profiles: list[dict] | None = None
        self._engine: MovieLensSearchEngine | None = None
        self._summary: dict | None = None

    def _ensure_loaded(self) -> None:
        if self._profiles is not None and self._engine is not None and self._summary is not None:
            return

        movies, ratings, tags = load_movielens()
        profiles = build_movie_profiles(movies, ratings, tags)
        self._profiles = profiles
        self._engine = MovieLensSearchEngine(profiles)
        self._summary = {
            "movie_count": len(profiles),
            "rating_count": int(len(ratings)),
            "tag_count": int(sum(len(item["tags"]) for item in profiles)),
            "user_count": int(ratings["userId"].nunique()),
            "top_algorithm": "heap_sort",
        }

    @property
    def profiles(self) -> list[dict]:
        self._ensure_loaded()
        assert self._profiles is not None
        return self._profiles

    @property
    def engine(self) -> MovieLensSearchEngine:
        self._ensure_loaded()
        assert self._engine is not None
        return self._engine

    def dashboard(self) -> dict:
        self._ensure_loaded()
        assert self._summary is not None
        return {
            "dataset": {
                "name": "movielens",
                "display_name": "MovieLens",
            },
            "summary": self._summary,
            "sortRuntime": _read_csv(OUTPUT_DIR / "movielens" / "sorting_runtime.csv"),
            "searchRuntime": _read_csv(OUTPUT_DIR / "movielens" / "search_runtime.csv"),
        }

    def top(self, n: int, algorithm: Literal["heap", "merge"]) -> dict:
        started = time.perf_counter()
        rows = top_n_movies(self.profiles, n=n, algorithm=algorithm)
        return {
            "items": rows,
            "count": len(rows),
            "algorithm": algorithm,
            "elapsed_ms": _elapsed_ms(started),
        }

    def search(self, kind: Literal["title", "genre", "tag"], query: str, n: int) -> dict:
        started = time.perf_counter()
        if kind == "title":
            rows = self.engine.index_title_search(query)
        elif kind == "genre":
            rows = self.engine.index_genre_search(query)
        else:
            rows = self.engine.index_tag_search(query)

        return {
            "items": rows[:n],
            "count": len(rows),
            "kind": kind,
            "query": query,
            "elapsed_ms": _elapsed_ms(started),
            "engine": "indexed_search",
        }

    def recommend(self, title: str, n: int) -> dict:
        started = time.perf_counter()
        target, rows = recommend_similar_movies(title, self.profiles, self.engine, n=n)
        return {
            "target": target,
            "items": rows,
            "count": len(rows),
            "elapsed_ms": _elapsed_ms(started),
            "engine": "similarity_recommendation",
        }


class ApiState:
    def __init__(self, dataset: str) -> None:
        if dataset != "movielens":
            available = ", ".join(sorted(DATASETS))
            raise ValueError(f"API server currently supports movielens. Available datasets: {available}")
        self.dataset = dataset
        self.movielens = MovieLensApiService()


def create_app(dataset: str = "movielens") -> FastAPI:
    try:
        state = ApiState(dataset)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    app = FastAPI(
        title="Movie Recommendation Lab API",
        description="Backend API for dataset summary, Top-N ranking, indexed search, and recommendation.",
        version="0.1.0",
    )

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "dataset": state.dataset}

    @app.get("/api/dashboard")
    def dashboard() -> dict:
        return _call_service(state.movielens.dashboard)

    @app.get("/api/top")
    def top(
        n: int = Query(default=10, ge=1, le=100),
        algorithm: Literal["heap", "merge"] = "heap",
    ) -> dict:
        return _call_service(state.movielens.top, n=n, algorithm=algorithm)

    @app.get("/api/search")
    def search(
        kind: Literal["title", "genre", "tag"] = "title",
        query: str = Query(default="", min_length=0),
        n: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        return _call_service(state.movielens.search, kind=kind, query=query.strip(), n=n)

    @app.get("/api/recommend")
    def recommend(
        title: str = Query(default="", min_length=0),
        n: int = Query(default=10, ge=1, le=50),
    ) -> dict:
        return _call_service(state.movielens.recommend, title=title.strip(), n=n)

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def _call_service(func, *args, **kwargs) -> dict:
    try:
        return func(*args, **kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"missing data file: {exc.filename}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Movie recommendation FastAPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="movielens")
    parser.add_argument("--reload", action="store_true")
    return parser


app = create_app()


def main() -> None:
    import uvicorn

    args = build_parser().parse_args()
    if args.reload and args.dataset == "movielens":
        target = "src.api:app"
    else:
        target = create_app(args.dataset)
    uvicorn.run(target, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
