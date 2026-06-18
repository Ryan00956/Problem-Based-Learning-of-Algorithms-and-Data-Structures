from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.core.paths import OUTPUT_DIR, PROJECT_ROOT
from src.core.registry import DATASETS
from src.datasets.movielens.collaborative import UserCollaborativeModel
from src.datasets.movielens.loader import load_movielens
from src.datasets.movielens.personalization import (
    InteractionEvent,
    MovieVectorModel,
    PersonalizationStore,
    recommend_for_you,
)
from src.datasets.movielens.profiles import build_movie_profiles
from src.datasets.movielens.recommendation import recommend_similar_movies, top_n_movies
from src.datasets.movielens.search import MovieLensSearchEngine
from src.datasets.movielens.tag_semantics import TagSemanticModel
from src.datasets.netflix.import_duckdb import DEFAULT_DB_PATH as NETFLIX_DB_PATH
from src.datasets.netflix.scoring import load_movie_scores, rank_movie_scores


WEB_DIR = PROJECT_ROOT / "web"
MOVIELENS_OUTPUT_DIR = OUTPUT_DIR / "movielens"
NETFLIX_OUTPUT_DIR = OUTPUT_DIR / "netflix"


class InteractionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=120)
    event_type: Literal["search", "similar", "view", "like", "dislike", "reset"]
    kind: Literal["title", "genre", "tag"] | None = None
    query: str = Field(default="", max_length=160)
    movie_id: int | None = None
    source: Literal["interest", "collaborative", "explore", "top", "search", "similar", "detail"] | None = None


class MovieLensApiService:
    def __init__(self) -> None:
        self._profiles: list[dict] | None = None
        self._engine: MovieLensSearchEngine | None = None
        self._vector_model: MovieVectorModel | None = None
        self._collaborative_model: UserCollaborativeModel | None = None
        self._tag_semantics: TagSemanticModel | None = None
        self._summary: dict | None = None
        self._events = PersonalizationStore(MOVIELENS_OUTPUT_DIR / "user_events.jsonl")

    def _ensure_loaded(self) -> None:
        if (
            self._profiles is not None
            and self._engine is not None
            and self._vector_model is not None
            and self._collaborative_model is not None
            and self._tag_semantics is not None
            and self._summary is not None
        ):
            return

        movies, ratings, tags = load_movielens()
        tag_aliases: dict[str, str] = {}
        profiles = build_movie_profiles(movies, ratings, tags, tag_aliases=tag_aliases)
        self._profiles = profiles
        self._engine = MovieLensSearchEngine(profiles, tag_aliases=tag_aliases)
        self._vector_model = MovieVectorModel(profiles)
        self._collaborative_model = UserCollaborativeModel(ratings)
        self._tag_semantics = TagSemanticModel.from_profiles(
            profiles,
            cache_path=MOVIELENS_OUTPUT_DIR / "tag_semantics.json",
        )
        self._summary = {
            "movie_count": len(profiles),
            "rating_count": int(len(ratings)),
            "tag_count": int(sum(len(item["tags"]) for item in profiles)),
            "user_count": int(ratings["userId"].nunique()),
            "top_algorithm": "top_n_heap",
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

    @property
    def vector_model(self) -> MovieVectorModel:
        self._ensure_loaded()
        assert self._vector_model is not None
        return self._vector_model

    @property
    def tag_semantics(self) -> TagSemanticModel:
        self._ensure_loaded()
        assert self._tag_semantics is not None
        return self._tag_semantics

    @property
    def collaborative_model(self) -> UserCollaborativeModel:
        self._ensure_loaded()
        assert self._collaborative_model is not None
        return self._collaborative_model

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

    def record_event(self, request: InteractionRequest) -> dict:
        event = InteractionEvent(
            session_id=request.session_id.strip(),
            event_type=request.event_type,
            timestamp=time.time(),
            kind=request.kind,
            query=request.query.strip(),
            movie_id=request.movie_id,
            source=request.source,
        )
        event_count = self._events.add(event)
        return {
            "ok": True,
            "event_count": event_count,
            "event_type": event.event_type,
        }

    def for_you(self, session_id: str, n: int) -> dict:
        started = time.perf_counter()
        payload = recommend_for_you(
            self.profiles,
            self.engine,
            self._events.get(session_id),
            n=n,
            vector_model=self.vector_model,
            tag_semantics=self.tag_semantics,
            collaborative_model=self.collaborative_model,
        )
        payload["elapsed_ms"] = _elapsed_ms(started)
        return payload

    def tag_semantic_neighbors(self, tag: str, n: int) -> dict:
        started = time.perf_counter()
        canonical = self.engine.canonicalize_tag(tag) or tag.strip().lower()
        neighbors = self.tag_semantics.neighbors(canonical, limit=n)
        return {
            "tag": canonical,
            "query": tag,
            "neighbors": neighbors,
            "count": len(neighbors),
            "summary": self.tag_semantics.summary,
            "elapsed_ms": _elapsed_ms(started),
            "engine": "tag_movie_lsa",
        }

class NetflixApiService:
    def __init__(self) -> None:
        self._scores: list[dict] | None = None
        self._summary: dict | None = None

    def _ensure_loaded(self) -> None:
        if self._scores is not None and self._summary is not None:
            return

        scores = load_movie_scores()
        self._scores = scores
        self._summary = {
            "movie_count": len(scores),
            "rating_count": int(sum(item["rating_count"] for item in scores)),
            "tag_count": 0,
            "user_count": _netflix_user_count(),
            "top_algorithm": "top_n_heap",
        }

    @property
    def scores(self) -> list[dict]:
        self._ensure_loaded()
        assert self._scores is not None
        return self._scores

    def dashboard(self) -> dict:
        self._ensure_loaded()
        assert self._summary is not None
        return {
            "dataset": {
                "name": "netflix",
                "display_name": "Netflix Prize",
            },
            "summary": self._summary,
            "sortRuntime": _read_csv(NETFLIX_OUTPUT_DIR / "sorting_runtime.csv"),
            "searchRuntime": [],
        }

    def top(self, n: int, algorithm: Literal["heap", "merge"]) -> dict:
        started = time.perf_counter()
        rows = rank_movie_scores(self.scores, n=n, algorithm=algorithm)
        return {
            "items": rows,
            "count": len(rows),
            "algorithm": algorithm,
            "elapsed_ms": _elapsed_ms(started),
        }

    def search(self, *args, **kwargs) -> dict:
        raise NotImplementedError("Netflix search is not implemented yet.")

    def recommend(self, *args, **kwargs) -> dict:
        raise NotImplementedError("Netflix similar recommendation is not implemented yet.")

    def record_event(self, *args, **kwargs) -> dict:
        raise NotImplementedError("Netflix behavior events are not implemented yet.")

    def for_you(self, *args, **kwargs) -> dict:
        raise NotImplementedError("Netflix personalized recommendation is not implemented yet.")

    def tag_semantic_neighbors(self, *args, **kwargs) -> dict:
        raise NotImplementedError("Netflix tag semantics are not implemented because this dataset has no tags.")


class ApiState:
    def __init__(self, dataset: str) -> None:
        if dataset == "movielens":
            self.service = MovieLensApiService()
        elif dataset == "netflix":
            self.service = NetflixApiService()
        else:
            available = ", ".join(sorted(DATASETS))
            raise ValueError(f"unknown API dataset: {dataset}. Available datasets: {available}")
        self.dataset = dataset


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
        return _call_service(state.service.dashboard)

    @app.get("/api/top")
    def top(
        n: int = Query(default=10, ge=1, le=100),
        algorithm: Literal["heap", "merge"] = "heap",
    ) -> dict:
        return _call_service(state.service.top, n=n, algorithm=algorithm)

    @app.get("/api/search")
    def search(
        kind: Literal["title", "genre", "tag"] = "title",
        query: str = Query(default="", min_length=0),
        n: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        return _call_service(state.service.search, kind=kind, query=query.strip(), n=n)

    @app.get("/api/recommend")
    def recommend(
        title: str = Query(default="", min_length=0),
        n: int = Query(default=10, ge=1, le=50),
    ) -> dict:
        return _call_service(state.service.recommend, title=title.strip(), n=n)

    @app.post("/api/events")
    def record_event(request: InteractionRequest) -> dict:
        return _call_service(state.service.record_event, request=request)

    @app.get("/api/for-you")
    def for_you(
        session_id: str = Query(min_length=1, max_length=120),
        n: int = Query(default=10, ge=1, le=50),
    ) -> dict:
        return _call_service(state.service.for_you, session_id=session_id.strip(), n=n)

    @app.get("/api/tag-semantics")
    def tag_semantics(
        tag: str = Query(min_length=1, max_length=120),
        n: int = Query(default=8, ge=1, le=30),
    ) -> dict:
        return _call_service(state.service.tag_semantic_neighbors, tag=tag.strip(), n=n)

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def _call_service(func, *args, **kwargs) -> dict:
    try:
        return func(*args, **kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"missing data file: {exc.filename}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


def _netflix_user_count() -> int:
    import duckdb

    conn = duckdb.connect(str(NETFLIX_DB_PATH), read_only=True)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM user_stats").fetchone()[0])
    finally:
        conn.close()


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
