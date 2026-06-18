from __future__ import annotations

from src.algorithms.sorting import heap_sort, merge_sort
from src.datasets.movielens.search import MovieLensSearchEngine, normalize


def top_n_movies(profiles: list[dict], n: int = 10, algorithm: str = "heap") -> list[dict]:
    if algorithm == "merge":
        ranked = merge_sort(profiles, key="comprehensive_score", reverse=True)
    elif algorithm == "heap":
        ranked = heap_sort(profiles, key="comprehensive_score", reverse=True)
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    return ranked[:n]


def recommend_similar_movies(
    title_query: str,
    profiles: list[dict],
    search_engine: MovieLensSearchEngine,
    n: int = 10,
) -> tuple[dict | None, list[dict]]:
    candidates = search_engine.index_title_search(title_query)
    if not candidates:
        return None, []

    target = candidates[0]
    target_genres = {normalize(value) for value in target["genres"]}
    target_tags = {normalize(value) for value in target["tags"]}

    scored = []
    for item in profiles:
        if item["movieId"] == target["movieId"]:
            continue
        genre_overlap = len(target_genres & {normalize(value) for value in item["genres"]})
        tag_overlap = len(target_tags & {normalize(value) for value in item["tags"]})
        similarity_score = genre_overlap * 10.0 + tag_overlap * 15.0 + item["comprehensive_score"] * 0.1
        if genre_overlap or tag_overlap:
            enriched = dict(item)
            enriched["similarity_score"] = round(similarity_score, 4)
            enriched["shared_genres"] = genre_overlap
            enriched["shared_tags"] = tag_overlap
            scored.append(enriched)

    return target, heap_sort(scored, key="similarity_score", reverse=True)[:n]
