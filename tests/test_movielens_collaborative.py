from __future__ import annotations

import pandas as pd

from src.datasets.movielens.collaborative import UserCollaborativeModel
from src.datasets.movielens.personalization import (
    InteractionEvent,
    MovieVectorModel,
    recommend_for_you,
)
from src.datasets.movielens.search import MovieLensSearchEngine


def _ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"userId": 10, "movieId": 1, "rating": 5.0},
            {"userId": 10, "movieId": 2, "rating": 4.5},
            {"userId": 10, "movieId": 3, "rating": 1.0},
            {"userId": 10, "movieId": 4, "rating": 5.0},
            {"userId": 10, "movieId": 5, "rating": 4.5},
            {"userId": 11, "movieId": 1, "rating": 4.5},
            {"userId": 11, "movieId": 2, "rating": 5.0},
            {"userId": 11, "movieId": 3, "rating": 1.5},
            {"userId": 11, "movieId": 4, "rating": 4.5},
            {"userId": 11, "movieId": 5, "rating": 4.0},
            {"userId": 12, "movieId": 1, "rating": 1.0},
            {"userId": 12, "movieId": 2, "rating": 2.0},
            {"userId": 12, "movieId": 3, "rating": 5.0},
            {"userId": 12, "movieId": 4, "rating": 2.0},
            {"userId": 12, "movieId": 5, "rating": 1.5},
        ]
    )


def _events() -> list[InteractionEvent]:
    return [
        InteractionEvent("test", "like", 1.0, movie_id=1),
        InteractionEvent("test", "like", 2.0, movie_id=2),
        InteractionEvent("test", "dislike", 3.0, movie_id=3),
    ]


def _profiles() -> list[dict]:
    rows = [
        (1, "Liked One", ["Action"], 80.0, ["heroic"]),
        (2, "Liked Two", ["Adventure"], 78.0, ["quest"]),
        (3, "Rejected", ["Horror"], 30.0, ["bleak"]),
        (4, "Neighbor Favorite", ["Adventure"], 76.0, ["quest"]),
        (5, "Neighbor Also Likes", ["Action"], 70.0, ["heroic"]),
        (6, "Explore Backup", ["Comedy"], 74.0, ["funny"]),
    ]
    profiles = []
    for movie_id, title, genres, score, tags in rows:
        profiles.append(
            {
                "movieId": movie_id,
                "title": title,
                "genres": genres,
                "genres_text": "|".join(genres),
                "avg_rating": 4.0,
                "bayesian_rating": 4.0,
                "rating_count": 30,
                "recent_rating_count": 5,
                "tag_count": len(tags),
                "tag_evidence": float(len(tags)),
                "tags": tags,
                "tag_details": [
                    {
                        "tag": tag,
                        "display": tag,
                        "count": 2,
                        "facet": "theme",
                        "confidence": 1.0,
                        "weight": 1.0,
                    }
                    for tag in tags
                ],
                "rating_score": score * 0.7,
                "popularity_score": 10.0,
                "tag_score": 2.0,
                "freshness_score": 1.0,
                "comprehensive_score": score,
            }
        )
    return profiles


def test_collaborative_model_recommends_movies_from_similar_users() -> None:
    model = UserCollaborativeModel(_ratings())

    result = model.recommend(_events(), exclude_movie_ids={1, 2, 3}, limit=10)

    assert result["eligible_movie_count"] == 3
    assert result["neighbor_count"] == 2
    assert result["candidates"][0].movie_id == 4
    assert result["candidates"][0].support_count == 2
    assert result["candidates"][0].avg_neighbor_rating >= 4.5


def test_for_you_blends_collaborative_bucket() -> None:
    profiles = _profiles()
    vector_model = MovieVectorModel(profiles)
    search_engine = MovieLensSearchEngine(profiles)
    collaborative_model = UserCollaborativeModel(_ratings())

    payload = recommend_for_you(
        profiles,
        search_engine,
        _events(),
        n=4,
        vector_model=vector_model,
        collaborative_model=collaborative_model,
    )

    assert payload["status"] == "personalized"
    assert payload["collaborative"]["neighbor_count"] == 2
    assert payload["bucket_counts"]["collaborative"] >= 1
    assert any(item["recommendation_bucket"] == "collaborative" for item in payload["items"])
