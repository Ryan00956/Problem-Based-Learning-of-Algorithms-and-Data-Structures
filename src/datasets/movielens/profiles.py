from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from src.datasets.movielens.tags import build_tag_details, top_tag_names


def _split_pipe(value: str) -> list[str]:
    if not isinstance(value, str) or not value or value == "(no genres listed)":
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def build_movie_profiles(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    tags: pd.DataFrame,
    tag_aliases: Mapping[str, str] | None = None,
) -> list[dict]:
    dataset_now = int(ratings["timestamp"].max()) if "timestamp" in ratings and not ratings.empty else 0
    recent_window_seconds = 180 * 24 * 60 * 60
    recent_cutoff = dataset_now - recent_window_seconds
    global_avg_rating = float(ratings["rating"].mean()) if not ratings.empty else 0.0
    min_rating = float(ratings["rating"].min()) if not ratings.empty else 0.0
    max_rating = float(ratings["rating"].max()) if not ratings.empty else 5.0

    rating_stats = (
        ratings.groupby("movieId")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "size"),
        )
        .reset_index()
    )
    if ratings.empty:
        preference_rating_stats = pd.DataFrame(columns=["movieId", "preference_adjusted_rating"])
    else:
        adjusted_ratings = ratings[["movieId", "userId", "rating"]].copy()
        user_avg_rating = adjusted_ratings.groupby("userId")["rating"].transform("mean")
        adjusted_ratings["preference_adjusted_rating"] = (
            adjusted_ratings["rating"] - user_avg_rating + global_avg_rating
        ).clip(lower=min_rating, upper=max_rating)
        preference_rating_stats = (
            adjusted_ratings.groupby("movieId")
            .agg(preference_adjusted_rating=("preference_adjusted_rating", "mean"))
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
    tag_stats = (
        tags.groupby("movieId")
        .agg(tag_details=("tag", lambda values: build_tag_details(values, tag_aliases)))
        .reset_index()
    )

    merged = (
        movies.merge(rating_stats, on="movieId", how="left")
        .merge(preference_rating_stats, on="movieId", how="left")
        .merge(recent_stats, on="movieId", how="left")
        .merge(tag_stats, on="movieId", how="left")
    )
    merged["avg_rating"] = merged["avg_rating"].fillna(0.0)
    merged["preference_adjusted_rating"] = merged["preference_adjusted_rating"].fillna(0.0)
    merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)
    merged["recent_rating_count"] = merged["recent_rating_count"].fillna(0).astype(int)
    merged["tag_details"] = merged["tag_details"].apply(lambda value: value if isinstance(value, list) else [])
    merged["tags"] = merged["tag_details"].apply(top_tag_names)
    merged["tag_count"] = merged["tag_details"].apply(len)
    merged["tag_evidence"] = merged["tag_details"].apply(lambda values: sum(float(item["weight"]) for item in values))
    merged["genre_list"] = merged["genres"].apply(_split_pipe)

    max_log_count = math.log1p(max(int(merged["rating_count"].max()), 1))
    max_log_recent_count = math.log1p(max(int(merged["recent_rating_count"].max()), 1))
    max_log_tag_evidence = math.log1p(max(float(merged["tag_evidence"].max()), 1.0))
    prior_rating_count = 25.0

    profiles: list[dict] = []
    for row in merged.itertuples(index=False):
        avg_rating = float(row.avg_rating)
        preference_adjusted_rating = float(row.preference_adjusted_rating)
        rating_count = int(row.rating_count)
        recent_rating_count = int(row.recent_rating_count)
        tag_count = int(row.tag_count)
        tag_evidence = float(row.tag_evidence)

        if rating_count:
            bayesian_rating = (
                rating_count * avg_rating + prior_rating_count * global_avg_rating
            ) / (rating_count + prior_rating_count)
            preference_adjusted_bayesian_rating = (
                rating_count * preference_adjusted_rating + prior_rating_count * global_avg_rating
            ) / (rating_count + prior_rating_count)
        else:
            bayesian_rating = 0.0
            preference_adjusted_bayesian_rating = 0.0

        rating_score = (bayesian_rating / 5.0) * 70.0
        preference_adjusted_rating_score = (preference_adjusted_bayesian_rating / 5.0) * 70.0
        popularity_score = (math.log1p(rating_count) / max_log_count) * 20.0 if max_log_count else 0.0
        tag_score = (math.log1p(tag_evidence) / max_log_tag_evidence) * 5.0 if max_log_tag_evidence else 0.0
        freshness_score = (
            (math.log1p(recent_rating_count) / max_log_recent_count) * 5.0
            if max_log_recent_count
            else 0.0
        )
        comprehensive_score = rating_score + popularity_score + tag_score + freshness_score
        preference_adjusted_comprehensive_score = (
            preference_adjusted_rating_score + popularity_score + tag_score + freshness_score
        )

        profiles.append(
            {
                "movieId": int(row.movieId),
                "title": str(row.title),
                "genres": list(row.genre_list),
                "genres_text": str(row.genres),
                "avg_rating": round(avg_rating, 4),
                "bayesian_rating": round(bayesian_rating, 4),
                "preference_adjusted_rating": round(preference_adjusted_rating, 4),
                "preference_adjusted_bayesian_rating": round(preference_adjusted_bayesian_rating, 4),
                "preference_adjustment": round(preference_adjusted_bayesian_rating - bayesian_rating, 4),
                "rating_count": rating_count,
                "recent_rating_count": recent_rating_count,
                "tag_count": tag_count,
                "tag_evidence": round(tag_evidence, 4),
                "tags": list(row.tags),
                "tag_details": list(row.tag_details),
                "rating_score": round(rating_score, 4),
                "preference_adjusted_rating_score": round(preference_adjusted_rating_score, 4),
                "popularity_score": round(popularity_score, 4),
                "tag_score": round(tag_score, 4),
                "freshness_score": round(freshness_score, 4),
                "comprehensive_score": round(comprehensive_score, 4),
                "preference_adjusted_comprehensive_score": round(preference_adjusted_comprehensive_score, 4),
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
                "preference_adjusted_rating": item.get("preference_adjusted_rating", 0.0),
                "preference_adjusted_bayesian_rating": item.get("preference_adjusted_bayesian_rating", 0.0),
                "preference_adjustment": item.get("preference_adjustment", 0.0),
                "rating_count": item["rating_count"],
                "recent_rating_count": item["recent_rating_count"],
                "tag_count": item["tag_count"],
                "tag_evidence": item.get("tag_evidence", 0.0),
                "rating_score": item["rating_score"],
                "preference_adjusted_rating_score": item.get("preference_adjusted_rating_score", 0.0),
                "popularity_score": item["popularity_score"],
                "tag_score": item["tag_score"],
                "freshness_score": item["freshness_score"],
                "comprehensive_score": item["comprehensive_score"],
                "preference_adjusted_comprehensive_score": item.get(
                    "preference_adjusted_comprehensive_score",
                    item["comprehensive_score"],
                ),
                "tags": "|".join(item["tags"]),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
