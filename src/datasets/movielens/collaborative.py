from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Protocol

import pandas as pd


class BehaviorEvent(Protocol):
    event_type: str
    movie_id: int | None


MIN_PROFILE_MOVIES = 3
MIN_SHARED_MOVIES = 2
NEIGHBOR_LIMIT = 24
SUPPORT_LIMIT = 2
OVERLAP_SMOOTHING = 5.0


@dataclass(frozen=True)
class CollaborativeCandidate:
    movie_id: int
    score: float
    neighbor_count: int
    support_count: int
    avg_neighbor_rating: float
    max_similarity: float
    shared_movie_count: int


class UserCollaborativeModel:
    def __init__(self, ratings: pd.DataFrame) -> None:
        self.user_ratings: dict[int, dict[int, float]] = {}
        self.user_centered: dict[int, dict[int, float]] = {}
        self.user_norms: dict[int, float] = {}
        self.movie_users: dict[int, list[tuple[int, float]]] = defaultdict(list)

        if ratings.empty:
            return

        for user_id, group in ratings.groupby("userId"):
            values = {
                int(row.movieId): float(row.rating)
                for row in group.itertuples(index=False)
            }
            if not values:
                continue

            mean_rating = sum(values.values()) / len(values)
            centered = {
                movie_id: rating - mean_rating
                for movie_id, rating in values.items()
            }
            norm = math.sqrt(sum(value * value for value in centered.values()))
            if not norm:
                continue

            uid = int(user_id)
            self.user_ratings[uid] = values
            self.user_centered[uid] = centered
            self.user_norms[uid] = norm
            for movie_id, centered_rating in centered.items():
                self.movie_users[movie_id].append((uid, centered_rating))

    def recommend(
        self,
        events: Iterable[BehaviorEvent],
        exclude_movie_ids: set[int],
        limit: int,
    ) -> dict:
        active_profile = _active_profile_from_events(events)
        if len(active_profile) < MIN_PROFILE_MOVIES:
            return _empty_result(active_profile)

        centered_profile, active_norm = _center_profile(active_profile)
        if not active_norm:
            return _empty_result(active_profile)

        neighbors = self._similar_neighbors(centered_profile, active_norm)
        if not neighbors:
            return {
                **_empty_result(active_profile),
                "eligible_movie_count": len(active_profile),
            }

        candidates = self._score_candidates(neighbors, exclude_movie_ids, limit)
        return {
            "candidates": candidates,
            "eligible_movie_count": len(active_profile),
            "neighbor_count": len(neighbors),
            "top_similarity": round(neighbors[0]["similarity"], 4),
            "max_shared_movies": max(item["shared_movie_count"] for item in neighbors),
        }

    def _similar_neighbors(
        self,
        centered_profile: dict[int, float],
        active_norm: float,
    ) -> list[dict]:
        dot_scores: defaultdict[int, float] = defaultdict(float)
        shared_counts: Counter[int] = Counter()

        for movie_id, active_value in centered_profile.items():
            for user_id, user_value in self.movie_users.get(movie_id, ()):
                dot_scores[user_id] += active_value * user_value
                shared_counts[user_id] += 1

        neighbors = []
        for user_id, dot_score in dot_scores.items():
            shared_count = shared_counts[user_id]
            if shared_count < MIN_SHARED_MOVIES:
                continue

            raw_similarity = dot_score / (active_norm * self.user_norms[user_id])
            if raw_similarity <= 0:
                continue

            overlap_weight = shared_count / (shared_count + OVERLAP_SMOOTHING)
            similarity = raw_similarity * overlap_weight
            neighbors.append(
                {
                    "user_id": user_id,
                    "similarity": similarity,
                    "shared_movie_count": shared_count,
                }
            )

        return sorted(neighbors, key=lambda item: item["similarity"], reverse=True)[:NEIGHBOR_LIMIT]

    def _score_candidates(
        self,
        neighbors: list[dict],
        exclude_movie_ids: set[int],
        limit: int,
    ) -> list[CollaborativeCandidate]:
        weighted_scores: defaultdict[int, float] = defaultdict(float)
        similarity_sums: defaultdict[int, float] = defaultdict(float)
        weighted_ratings: defaultdict[int, float] = defaultdict(float)
        support_counts: Counter[int] = Counter()
        max_similarity: defaultdict[int, float] = defaultdict(float)
        max_shared: defaultdict[int, int] = defaultdict(int)

        for neighbor in neighbors:
            user_id = neighbor["user_id"]
            similarity = float(neighbor["similarity"])
            shared_count = int(neighbor["shared_movie_count"])
            for movie_id, rating in self.user_ratings[user_id].items():
                if movie_id in exclude_movie_ids or rating < 4.0:
                    continue

                centered_rating = self.user_centered[user_id].get(movie_id, 0.0)
                if centered_rating <= 0:
                    continue

                weighted_scores[movie_id] += similarity * centered_rating
                similarity_sums[movie_id] += abs(similarity)
                weighted_ratings[movie_id] += similarity * rating
                support_counts[movie_id] += 1
                max_similarity[movie_id] = max(max_similarity[movie_id], similarity)
                max_shared[movie_id] = max(max_shared[movie_id], shared_count)

        candidates = []
        for movie_id, weighted_score in weighted_scores.items():
            support_count = support_counts[movie_id]
            if support_count < SUPPORT_LIMIT:
                continue

            similarity_sum = similarity_sums[movie_id]
            if similarity_sum <= 0:
                continue

            preference_score = weighted_score / similarity_sum
            avg_rating = weighted_ratings[movie_id] / similarity_sum
            support_bonus = min(math.log1p(support_count) / math.log1p(NEIGHBOR_LIMIT), 1.0)
            score = preference_score * 70.0 + support_bonus * 18.0 + (avg_rating / 5.0) * 12.0
            candidates.append(
                CollaborativeCandidate(
                    movie_id=movie_id,
                    score=round(score, 4),
                    neighbor_count=len(neighbors),
                    support_count=support_count,
                    avg_neighbor_rating=round(avg_rating, 4),
                    max_similarity=round(max_similarity[movie_id], 4),
                    shared_movie_count=max_shared[movie_id],
                )
            )

        return sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]


def _active_profile_from_events(events: Iterable[BehaviorEvent]) -> dict[int, float]:
    profile: dict[int, tuple[float, float, int]] = {}
    for index, event in enumerate(events):
        if event.movie_id is None:
            continue

        signal = _pseudo_rating(event.event_type)
        if signal is None:
            continue

        rating, confidence = signal
        movie_id = int(event.movie_id)
        previous = profile.get(movie_id)
        if previous is None or confidence > previous[1] or (
            confidence == previous[1] and index >= previous[2]
        ):
            profile[movie_id] = (rating, confidence, index)

    return {
        movie_id: rating
        for movie_id, (rating, confidence, _index) in profile.items()
        if confidence >= 0.3
    }


def _pseudo_rating(event_type: str) -> tuple[float, float] | None:
    if event_type == "like":
        return 5.0, 1.0
    if event_type == "dislike":
        return 1.0, 1.0
    if event_type == "view":
        return 3.6, 0.35
    return None


def _center_profile(profile: dict[int, float]) -> tuple[dict[int, float], float]:
    mean_rating = 3.0
    centered = {
        movie_id: rating - mean_rating
        for movie_id, rating in profile.items()
    }
    norm = math.sqrt(sum(value * value for value in centered.values()))
    return centered, norm


def _empty_result(active_profile: dict[int, float]) -> dict:
    return {
        "candidates": [],
        "eligible_movie_count": len(active_profile),
        "neighbor_count": 0,
        "top_similarity": 0.0,
        "max_shared_movies": 0,
    }
