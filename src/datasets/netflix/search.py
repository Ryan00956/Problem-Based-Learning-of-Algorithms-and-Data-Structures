from __future__ import annotations

from typing import Any, Iterable, Literal

from src.datasets.movielens.search import MovieLensSearchEngine


def build_search_profiles(scores: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for item in scores:
        movie_id = int(item["movie_id"])
        profiles.append(
            {
                **item,
                "movieId": movie_id,
                "movie_id": movie_id,
                "title": str(item["title"]),
                "genres": [],
                "genres_text": "",
                "tags": [],
                "tag_details": [],
                "tag_count": 0,
                "tag_evidence": 0.0,
                "tag_score": 0.0,
                "comprehensive_score": float(item.get("comprehensive_score", 0.0)),
            }
        )
    return profiles


def build_search_engine(scores: Iterable[dict[str, Any]]) -> MovieLensSearchEngine:
    return MovieLensSearchEngine(build_search_profiles(scores))


def search_titles(
    scores: Iterable[dict[str, Any]],
    kind: Literal["title", "genre", "tag"],
    query: str,
    n: int = 10,
) -> list[dict[str, Any]]:
    if kind != "title":
        raise ValueError("Netflix search only supports title because this dataset has no genres or tags.")
    engine = build_search_engine(scores)
    return engine.index_title_search(query)[:n]
