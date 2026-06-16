from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MOVIELENS_DIR = DATA_DIR / "ml-latest-small"
NETFLIX_DIR = DATA_DIR / "netflix-prize" / "download"
OUTPUT_DIR = PROJECT_ROOT / "output"


def load_movielens(data_dir: Path = MOVIELENS_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(data_dir / "movies.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    tags = pd.read_csv(data_dir / "tags.csv")
    return movies, ratings, tags


def _split_pipe(value: str) -> list[str]:
    if not isinstance(value, str) or not value or value == "(no genres listed)":
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def build_movie_profiles(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    tags: pd.DataFrame,
) -> list[dict]:
    rating_stats = (
        ratings.groupby("movieId")
        .agg(avg_rating=("rating", "mean"), rating_count=("rating", "size"))
        .reset_index()
    )
    tag_stats = (
        tags.groupby("movieId")
        .agg(tag_count=("tag", "size"), tags=("tag", lambda values: sorted({str(v).strip() for v in values})))
        .reset_index()
    )

    merged = movies.merge(rating_stats, on="movieId", how="left").merge(tag_stats, on="movieId", how="left")
    merged["avg_rating"] = merged["avg_rating"].fillna(0.0)
    merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)
    merged["tag_count"] = merged["tag_count"].fillna(0).astype(int)
    merged["tags"] = merged["tags"].apply(lambda value: value if isinstance(value, list) else [])
    merged["genre_list"] = merged["genres"].apply(_split_pipe)

    max_log_count = math.log1p(max(int(merged["rating_count"].max()), 1))
    max_log_tags = math.log1p(max(int(merged["tag_count"].max()), 1))

    profiles: list[dict] = []
    for row in merged.itertuples(index=False):
        avg_rating = float(row.avg_rating)
        rating_count = int(row.rating_count)
        tag_count = int(row.tag_count)

        rating_score = (avg_rating / 5.0) * 70.0
        popularity_score = (math.log1p(rating_count) / max_log_count) * 25.0 if max_log_count else 0.0
        tag_score = (math.log1p(tag_count) / max_log_tags) * 5.0 if max_log_tags else 0.0
        comprehensive_score = rating_score + popularity_score + tag_score

        profiles.append(
            {
                "movieId": int(row.movieId),
                "title": str(row.title),
                "genres": list(row.genre_list),
                "genres_text": str(row.genres),
                "avg_rating": round(avg_rating, 4),
                "rating_count": rating_count,
                "tag_count": tag_count,
                "tags": list(row.tags),
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
                "rating_count": item["rating_count"],
                "tag_count": item["tag_count"],
                "comprehensive_score": item["comprehensive_score"],
                "tags": "|".join(item["tags"]),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def load_netflix_titles(path: Path = NETFLIX_DIR / "movie_titles.txt") -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="latin-1") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            movie_id, year, title = line.split(",", 2)
            parsed_year = int(year) if year and year != "NULL" else None
            rows.append(
                {
                    "movieId": int(movie_id),
                    "year": parsed_year,
                    "title": title,
                }
            )
    return rows


def load_netflix_rating_sample(limit_movies: int = 200) -> list[dict]:
    training_dir = NETFLIX_DIR / "training_set"
    files = sorted(training_dir.glob("mv_*.txt"))[:limit_movies]
    rows: list[dict] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            movie_id_line = handle.readline().strip().rstrip(":")
            movie_id = int(movie_id_line)
            for line in handle:
                customer_id, rating, date = line.strip().split(",")
                rows.append(
                    {
                        "movieId": movie_id,
                        "customerId": int(customer_id),
                        "rating": int(rating),
                        "date": date,
                    }
                )
    return rows


def build_netflix_sample_profiles(limit_movies: int = 200) -> list[dict]:
    titles = {item["movieId"]: item for item in load_netflix_titles()}
    rows = load_netflix_rating_sample(limit_movies=limit_movies)
    stats: dict[int, dict] = {}
    for row in rows:
        movie_id = row["movieId"]
        bucket = stats.setdefault(movie_id, {"rating_sum": 0, "rating_count": 0})
        bucket["rating_sum"] += row["rating"]
        bucket["rating_count"] += 1

    max_log_count = math.log1p(max((item["rating_count"] for item in stats.values()), default=1))
    profiles: list[dict] = []
    for movie_id, stat in stats.items():
        title_row = titles.get(movie_id, {"title": f"Movie {movie_id}", "year": None})
        avg_rating = stat["rating_sum"] / stat["rating_count"]
        rating_score = (avg_rating / 5.0) * 75.0
        popularity_score = (math.log1p(stat["rating_count"]) / max_log_count) * 25.0 if max_log_count else 0.0
        profiles.append(
            {
                "movieId": movie_id,
                "title": title_row["title"],
                "year": title_row["year"],
                "genres": [],
                "genres_text": "",
                "avg_rating": round(avg_rating, 4),
                "rating_count": stat["rating_count"],
                "tag_count": 0,
                "tags": [],
                "comprehensive_score": round(rating_score + popularity_score, 4),
            }
        )
    return profiles
