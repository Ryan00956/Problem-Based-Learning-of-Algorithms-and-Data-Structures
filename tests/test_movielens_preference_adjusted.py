from __future__ import annotations

import pandas as pd

from src.datasets.movielens.profiles import build_movie_profiles
from src.datasets.movielens.recommendation import top_n_movies


def test_preference_adjusted_rating_accounts_for_user_leniency() -> None:
    movies = pd.DataFrame(
        [
            {"movieId": 1, "title": "Lenient Four", "genres": "Drama"},
            {"movieId": 2, "title": "Strict Four", "genres": "Drama"},
        ]
    )
    ratings = pd.DataFrame(
        [
            {"userId": 10, "movieId": 1, "rating": 4.0, "timestamp": 100},
            {"userId": 10, "movieId": 101, "rating": 5.0, "timestamp": 100},
            {"userId": 20, "movieId": 2, "rating": 4.0, "timestamp": 100},
            {"userId": 20, "movieId": 102, "rating": 2.0, "timestamp": 100},
        ]
    )
    tags = pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])

    profiles = build_movie_profiles(movies, ratings, tags)
    by_title = {item["title"]: item for item in profiles}

    assert by_title["Lenient Four"]["bayesian_rating"] == by_title["Strict Four"]["bayesian_rating"]
    assert (
        by_title["Lenient Four"]["preference_adjusted_bayesian_rating"]
        < by_title["Strict Four"]["preference_adjusted_bayesian_rating"]
    )
    assert top_n_movies(profiles, n=1, score_mode="preference_adjusted")[0]["title"] == "Strict Four"
