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
from src.datasets.movielens.tag_aliases import (
    DEFAULT_TAG_ALIAS_DECISIONS_PATH,
    TagAliasDecisionStore,
    build_tag_alias_report,
)
from src.datasets.movielens.tag_semantics import TagSemanticModel


WEB_DIR = PROJECT_ROOT / "web"
MOVIELENS_OUTPUT_DIR = OUTPUT_DIR / "movielens"


class InteractionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=120)
    event_type: Literal["search", "similar", "view", "like", "dislike", "reset"]
    kind: Literal["title", "genre", "tag"] | None = None
    query: str = Field(default="", max_length=160)
    movie_id: int | None = None
    source: Literal["interest", "explore", "top", "search", "similar", "detail"] | None = None


class TagAliasDecisionRequest(BaseModel):
    source: str = Field(min_length=1, max_length=120)
    target: str = Field(min_length=1, max_length=120)
    decision: Literal["accept", "reject", "ignore"]


class MovieLensApiService:
    def __init__(self) -> None:
        self._profiles: list[dict] | None = None
        self._engine: MovieLensSearchEngine | None = None
        self._vector_model: MovieVectorModel | None = None
        self._tag_semantics: TagSemanticModel | None = None
        self._summary: dict | None = None
        self._tag_alias_report: dict | None = None
        self._events = PersonalizationStore(MOVIELENS_OUTPUT_DIR / "user_events.jsonl")
        self._tag_decisions = TagAliasDecisionStore(DEFAULT_TAG_ALIAS_DECISIONS_PATH)

    def _ensure_loaded(self) -> None:
        if (
            self._profiles is not None
            and self._engine is not None
            and self._vector_model is not None
            and self._tag_semantics is not None
            and self._summary is not None
            and self._tag_alias_report is not None
        ):
            return

        movies, ratings, tags = load_movielens()
        tag_aliases = self._tag_decisions.accepted_aliases()
        profiles = build_movie_profiles(movies, ratings, tags, tag_aliases=tag_aliases)
        self._profiles = profiles
        self._engine = MovieLensSearchEngine(profiles, tag_aliases=tag_aliases)
        self._vector_model = MovieVectorModel(profiles)
        self._tag_semantics = TagSemanticModel.from_profiles(
            profiles,
            cache_path=MOVIELENS_OUTPUT_DIR / "tag_semantics.json",
        )
        self._tag_alias_report = build_tag_alias_report(
            tags,
            aliases=tag_aliases,
            decisions=self._tag_decisions.decisions,
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

    def tag_alias_candidates(self, n: int) -> dict:
        started = time.perf_counter()
        self._ensure_loaded()
        assert self._tag_alias_report is not None
        candidates = self._tag_alias_report["candidates"][:n]
        return {
            **self._tag_alias_report,
            "candidates": candidates,
            "count": len(candidates),
            "elapsed_ms": _elapsed_ms(started),
            "engine": "tag_alias_candidate_miner",
        }

    def record_tag_alias_decision(self, request: TagAliasDecisionRequest) -> dict:
        self._ensure_loaded()
        assert self._tag_alias_report is not None
        candidate = _find_alias_candidate(
            self._tag_alias_report["candidates"],
            request.source,
            request.target,
        )
        decision = self._tag_decisions.record(
            source=request.source,
            target=request.target,
            decision=request.decision,
            candidate=candidate,
        )
        self._clear_loaded()
        if request.decision == "accept":
            self._ensure_loaded()
        return {
            "ok": True,
            "decision": decision,
            "summary": self._tag_decisions.summary(),
            "accepted_aliases": self._tag_decisions.accepted_aliases(),
        }

    def _clear_loaded(self) -> None:
        self._profiles = None
        self._engine = None
        self._vector_model = None
        self._tag_semantics = None
        self._summary = None
        self._tag_alias_report = None


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

    @app.post("/api/events")
    def record_event(request: InteractionRequest) -> dict:
        return _call_service(state.movielens.record_event, request=request)

    @app.get("/api/for-you")
    def for_you(
        session_id: str = Query(min_length=1, max_length=120),
        n: int = Query(default=10, ge=1, le=50),
    ) -> dict:
        return _call_service(state.movielens.for_you, session_id=session_id.strip(), n=n)

    @app.get("/api/tag-alias-candidates")
    def tag_alias_candidates(n: int = Query(default=12, ge=1, le=50)) -> dict:
        return _call_service(state.movielens.tag_alias_candidates, n=n)

    @app.post("/api/tag-alias-decisions")
    def tag_alias_decision(request: TagAliasDecisionRequest) -> dict:
        return _call_service(state.movielens.record_tag_alias_decision, request=request)

    @app.get("/api/tag-semantics")
    def tag_semantics(
        tag: str = Query(min_length=1, max_length=120),
        n: int = Query(default=8, ge=1, le=30),
    ) -> dict:
        return _call_service(state.movielens.tag_semantic_neighbors, tag=tag.strip(), n=n)

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


def _find_alias_candidate(candidates: list[dict], source: str, target: str) -> dict | None:
    source_key = source.strip().lower()
    target_key = target.strip().lower()
    for item in candidates:
        if str(item.get("source", "")).lower() == source_key and str(item.get("target", "")).lower() == target_key:
            return item
    return None


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
