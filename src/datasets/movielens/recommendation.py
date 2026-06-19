from __future__ import annotations

from src.algorithms.sorting import heap_sort, merge_sort, top_n_heap
from src.core.title_series import series_match
from src.datasets.movielens.search import MovieLensSearchEngine, normalize


SERIES_MATCH_BONUS = 120.0
SUPPLEMENTAL_SERIES_PENALTY = 35.0
TITLE_SIMILARITY_WEIGHT = 8.0

TOP_N_SCORE_KEYS = {
    "default": "comprehensive_score",
    "preference_adjusted": "preference_adjusted_comprehensive_score",
}


def top_n_movies(
    profiles: list[dict],
    n: int = 10,
    algorithm: str = "heap",
    score_mode: str = "default",
) -> list[dict]:
    try:
        score_key = TOP_N_SCORE_KEYS[score_mode]
    except KeyError as exc:
        raise ValueError(f"unknown score mode: {score_mode}") from exc

    if algorithm == "merge":
        ranked = merge_sort(profiles, key=score_key, reverse=True)
        return ranked[:n]
    elif algorithm == "heap":
        return top_n_heap(profiles, n=n, key=score_key, reverse=True)
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")


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
        series = series_match(target["title"], item["title"])
        series_score = 0.0
        if series.is_match:
            series_score = SERIES_MATCH_BONUS
            if series.supplemental:
                series_score -= SUPPLEMENTAL_SERIES_PENALTY
        title_score = series.title_similarity * TITLE_SIMILARITY_WEIGHT
        similarity_score = (
            series_score
            + genre_overlap * 10.0
            + tag_overlap * 15.0
            + title_score
            + item["comprehensive_score"] * 0.1
        )
        if series.is_match or genre_overlap or tag_overlap:
            enriched = dict(item)
            enriched["similarity_score"] = round(similarity_score, 4)
            enriched["shared_genres"] = genre_overlap
            enriched["shared_tags"] = tag_overlap
            enriched["title_similarity"] = round(series.title_similarity, 4)
            if series.is_match:
                enriched["series_match"] = True
                enriched["series_key"] = series.key
                enriched["series_score"] = round(series_score, 4)
                enriched["recommendation_reason"] = f"Same series as {target['title']}."
            scored.append(enriched)

    return target, top_n_heap(scored, n=n, key="similarity_score", reverse=True)
