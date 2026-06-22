from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.algorithms.sorting import top_n_heap
from src.core.title_series import series_match
from src.datasets.netflix.collaborative import (
    NetflixCollaborativeModel,
    enrich_candidates,
)
from src.datasets.netflix.import_duckdb import DEFAULT_DB_PATH
from src.datasets.netflix.online_reranker import OnlineNetflixReranker
from src.datasets.netflix.scoring import load_movie_scores
from src.datasets.netflix.search import build_search_engine


SERIES_MATCH_BONUS = 180.0
SUPPLEMENTAL_SERIES_PENALTY = 45.0
TITLE_SIMILARITY_WEIGHT = 10.0
COLLABORATIVE_BLEND_WEIGHT = 0.25


def recommend_similar_movies(
    title_query: str,
    scores: list[dict],
    model: NetflixCollaborativeModel | None = None,
    n: int = 10,
) -> tuple[dict | None, list[dict]]:
    engine = build_search_engine(scores)
    matches = engine.index_title_search(title_query)
    if not matches:
        return None, []

    target = matches[0]
    movie_id = int(target["movieId"])
    model = model or NetflixCollaborativeModel()
    result = model.recommend_for_profile(
        {movie_id: 5.0},
        exclude_movie_ids={movie_id},
        limit=max(n * 8, 40),
        min_profile_movies=1,
        min_shared_movies=1,
        support_limit=2,
    )
    rows = enrich_candidates(result["candidates"], _scores_by_id(scores))
    for item in rows:
        item["recommendation_reason"] = (
            f"Users who strongly align with {target['title']} also rated this highly."
        )
        item["similarity_score"] = float(item.get("collaborative_score", 0.0))

    rows = _merge_series_and_collaborative_candidates(target, scores, rows)
    return target, rows[:n]


def recommend_for_events(
    events: Iterable,
    scores: list[dict],
    *,
    model: NetflixCollaborativeModel | None = None,
    reranker: OnlineNetflixReranker | None = None,
    n: int = 10,
) -> dict:
    events = list(events)
    model = model or NetflixCollaborativeModel()
    if reranker is not None and reranker.available:
        payload = reranker.recommend_for_events(events, scores, model=model, n=n)
        if payload is not None:
            return payload

    exclude_movie_ids = {
        int(event.movie_id)
        for event in events
        if getattr(event, "movie_id", None) is not None
        and getattr(event, "event_type", "") in {"view", "like", "dislike"}
    }
    result = model.recommend(events, exclude_movie_ids=exclude_movie_ids, limit=max(n * 8, 40))
    items = enrich_candidates(result["candidates"], _scores_by_id(scores))[:n]

    if len(items) < n:
        existing_ids = {int(item["movieId"]) for item in items}
        backfill = _cold_start_backfill(
            scores,
            n - len(items),
            exclude_movie_ids=exclude_movie_ids | existing_ids,
        )
        items.extend(backfill)

    status = "personalized" if result["neighbor_count"] else "cold_start"
    bucket_counts = {}
    for item in items:
        bucket = item.get("recommendation_bucket", "top")
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    return {
        "items": items,
        "count": len(items),
        "event_count": len(events),
        "profile": {
            "seed_movie_count": result["eligible_movie_count"],
            "signal_strength": result["eligible_movie_count"],
        },
        "collaborative": {
            "eligible_movie_count": result["eligible_movie_count"],
            "neighbor_count": result["neighbor_count"],
            "top_similarity": result["top_similarity"],
            "max_shared_movies": result["max_shared_movies"],
        },
        "bucket_counts": bucket_counts,
        "engine": "netflix_user_user_collaborative",
        "status": status,
    }


def recommend_for_user_id(
    user_id: int,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    scores: list[dict] | None = None,
    n: int = 10,
) -> dict:
    import duckdb

    scores = scores or load_movie_scores(db_path)
    with duckdb.connect(str(db_path), read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT movie_id, rating::DOUBLE
            FROM ratings
            WHERE user_id = ?
            ORDER BY rating_date DESC, movie_id ASC
            LIMIT 120
            """,
            [int(user_id)],
        ).fetchall()

    profile = {int(movie_id): float(rating) for movie_id, rating in rows}
    model = NetflixCollaborativeModel(db_path)
    result = model.recommend_for_profile(
        profile,
        exclude_movie_ids=set(profile),
        limit=max(n * 8, 40),
    )
    items = enrich_candidates(result["candidates"], _scores_by_id(scores))[:n]
    return {
        "user_id": int(user_id),
        "items": items,
        "count": len(items),
        "profile": {
            "seed_movie_count": len(profile),
            "sampled_recent_ratings": len(rows),
        },
        "collaborative": {
            "eligible_movie_count": result["eligible_movie_count"],
            "neighbor_count": result["neighbor_count"],
            "top_similarity": result["top_similarity"],
            "max_shared_movies": result["max_shared_movies"],
        },
        "engine": "netflix_existing_user_collaborative",
        "status": "personalized" if result["neighbor_count"] else "cold_start",
    }


def _scores_by_id(scores: list[dict]) -> dict[int, dict]:
    values = {}
    for item in scores:
        movie_id = int(item.get("movieId", item.get("movie_id")))
        enriched = dict(item)
        enriched["movieId"] = movie_id
        enriched["movie_id"] = movie_id
        values[movie_id] = enriched
    return values


def _merge_series_and_collaborative_candidates(
    target: dict,
    scores: list[dict],
    collaborative_rows: list[dict],
) -> list[dict]:
    merged: dict[int, dict] = {}
    for item in collaborative_rows:
        movie_id = int(item.get("movieId", item.get("movie_id")))
        merged[movie_id] = dict(item)

    for series_item in _series_candidates(target, scores):
        movie_id = int(series_item.get("movieId", series_item.get("movie_id")))
        existing = merged.get(movie_id)
        if existing:
            enriched = {**existing, **series_item}
            collaborative_score = float(existing.get("collaborative_score", 0.0))
            enriched["similarity_score"] = round(
                float(series_item["similarity_score"]) + collaborative_score * COLLABORATIVE_BLEND_WEIGHT,
                4,
            )
            if collaborative_score:
                enriched["recommendation_reason"] = (
                    f"Same series as {target['title']}; similar users also rated it highly."
                )
            merged[movie_id] = enriched
        else:
            merged[movie_id] = series_item

    return sorted(
        merged.values(),
        key=lambda item: (
            1 if item.get("series_match") else 0,
            float(item.get("similarity_score", item.get("personal_score", 0.0))),
            float(item.get("comprehensive_score", 0.0)),
        ),
        reverse=True,
    )


def _series_candidates(target: dict, scores: list[dict]) -> list[dict]:
    target_id = int(target.get("movieId", target.get("movie_id")))
    rows = []
    for item in scores:
        movie_id = int(item.get("movieId", item.get("movie_id")))
        if movie_id == target_id:
            continue

        series = series_match(str(target["title"]), str(item["title"]))
        if not series.is_match:
            continue

        series_score = SERIES_MATCH_BONUS
        if series.supplemental:
            series_score -= SUPPLEMENTAL_SERIES_PENALTY
        similarity_score = (
            series_score
            + series.title_similarity * TITLE_SIMILARITY_WEIGHT
            + float(item.get("comprehensive_score", 0.0)) * 0.1
        )
        enriched = dict(item)
        enriched["movieId"] = movie_id
        enriched["movie_id"] = movie_id
        enriched["series_match"] = True
        enriched["series_key"] = series.key
        enriched["series_score"] = round(series_score, 4)
        enriched["title_similarity"] = round(series.title_similarity, 4)
        enriched["similarity_score"] = round(similarity_score, 4)
        enriched["personal_score"] = round(similarity_score, 4)
        enriched["recommendation_bucket"] = "series"
        enriched["recommendation_reason"] = f"Same series as {target['title']}."
        rows.append(enriched)

    return sorted(
        rows,
        key=lambda item: (
            float(item["similarity_score"]),
            float(item.get("comprehensive_score", 0.0)),
        ),
        reverse=True,
    )


def _cold_start_backfill(scores: list[dict], n: int, *, exclude_movie_ids: set[int]) -> list[dict]:
    rows = []
    for item in top_n_heap(scores, n=max(n * 6, 30), key="comprehensive_score", reverse=True):
        movie_id = int(item.get("movieId", item.get("movie_id")))
        if movie_id in exclude_movie_ids:
            continue
        enriched = dict(item)
        enriched["movieId"] = movie_id
        enriched["movie_id"] = movie_id
        enriched["personal_score"] = float(enriched.get("comprehensive_score", 0.0))
        enriched["recommendation_bucket"] = "top"
        enriched["recommendation_reason"] = "High-scoring Netflix title while your taste profile warms up."
        rows.append(enriched)
        if len(rows) >= n:
            break
    return rows
