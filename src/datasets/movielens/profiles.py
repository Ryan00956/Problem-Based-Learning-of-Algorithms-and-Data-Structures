from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd


def _split_pipe(value: str) -> list[str]:
    if not isinstance(value, str) or not value or value == "(no genres listed)":
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def _clean_movie_tags(values: Iterable[object]) -> list[str]:
    tags = []
    for value in values:
        tag = str(value).strip()
        if tag and "netflix" not in tag.lower():
            tags.append(tag)
    return sorted(set(tags))


def build_movie_profiles(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    tags: pd.DataFrame,
) -> list[dict]:
    dataset_now = int(ratings["timestamp"].max()) if "timestamp" in ratings and not ratings.empty else 0
    recent_window_seconds = 180 * 24 * 60 * 60
    recent_cutoff = dataset_now - recent_window_seconds
    global_avg_rating = float(ratings["rating"].mean()) if not ratings.empty else 0.0

    rating_stats = (
        ratings.groupby("movieId")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "size"),
        )
        .reset_index()
    )
    recent_stats = (
        ratings[ratings["timestamp"] >= recent_cutoff]
        .groupby("movieId")
        .agg(recent_rating_count=("rating", "size"))
        .reset_index()
        if dataset_now
        else pd.DataFrame(columns=["movieId", "recent_rating_count"])
    )
    tag_stats = tags.groupby("movieId").agg(tags=("tag", _clean_movie_tags)).reset_index()

    merged = (
        movies.merge(rating_stats, on="movieId", how="left")
        .merge(recent_stats, on="movieId", how="left")
        .merge(tag_stats, on="movieId", how="left")
    )
    merged["avg_rating"] = merged["avg_rating"].fillna(0.0)
    merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)
    merged["recent_rating_count"] = merged["recent_rating_count"].fillna(0).astype(int)
    merged["tags"] = merged["tags"].apply(lambda value: value if isinstance(value, list) else [])
    merged["tag_count"] = merged["tags"].apply(len)
    merged["genre_list"] = merged["genres"].apply(_split_pipe)

    max_log_count = math.log1p(max(int(merged["rating_count"].max()), 1))
    max_log_recent_count = math.log1p(max(int(merged["recent_rating_count"].max()), 1))
    max_log_tags = math.log1p(max(int(merged["tag_count"].max()), 1))
    prior_rating_count = 25.0

    profiles: list[dict] = []
    for row in merged.itertuples(index=False):
        avg_rating = float(row.avg_rating)
        rating_count = int(row.rating_count)
        recent_rating_count = int(row.recent_rating_count)
        tag_count = int(row.tag_count)

        if rating_count:
            bayesian_rating = (
                rating_count * avg_rating + prior_rating_count * global_avg_rating
            ) / (rating_count + prior_rating_count)
        else:
            bayesian_rating = 0.0

        rating_score = (bayesian_rating / 5.0) * 70.0
        popularity_score = (math.log1p(rating_count) / max_log_count) * 20.0 if max_log_count else 0.0
        tag_score = (math.log1p(tag_count) / max_log_tags) * 5.0 if max_log_tags else 0.0
        freshness_score = (
            (math.log1p(recent_rating_count) / max_log_recent_count) * 5.0
            if max_log_recent_count
            else 0.0
        )
        comprehensive_score = rating_score + popularity_score + tag_score + freshness_score

        profiles.append(
            {
                "movieId": int(row.movieId),
                "title": str(row.title),
                "genres": list(row.genre_list),
                "genres_text": str(row.genres),
                "avg_rating": round(avg_rating, 4),
                "bayesian_rating": round(bayesian_rating, 4),
                "rating_count": rating_count,
                "recent_rating_count": recent_rating_count,
                "tag_count": tag_count,
                "tags": list(row.tags),
                "rating_score": round(rating_score, 4),
                "popularity_score": round(popularity_score, 4),
                "tag_score": round(tag_score, 4),
                "freshness_score": round(freshness_score, 4),
                "comprehensive_score": round(comprehensive_score, 4),
            }
        )
    return profiles


def save_profiles_csv(profiles: Iterable[dict], output_path: Path) -> None:
    rows = []
    for item in profiles:
        rows.append(
            {
                "movieId": item["movieId"],
                "title": item["title"],
                "genres": "|".join(item["genres"]),
                "avg_rating": item["avg_rating"],
                "bayesian_rating": item["bayesian_rating"],
                "rating_count": item["rating_count"],
                "recent_rating_count": item["recent_rating_count"],
                "tag_count": item["tag_count"],
                "rating_score": item["rating_score"],
                "popularity_score": item["popularity_score"],
                "tag_score": item["tag_score"],
                "freshness_score": item["freshness_score"],
                "comprehensive_score": item["comprehensive_score"],
                "tags": "|".join(item["tags"]),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
