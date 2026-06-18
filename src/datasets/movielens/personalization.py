from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import heapq
import json
import math
import time
from pathlib import Path
from typing import Literal

from src.algorithms.sorting import heap_sort
from src.datasets.movielens.search import MovieLensSearchEngine, normalize
from src.datasets.movielens.tag_semantics import TagSemanticModel
from src.datasets.movielens.tags import canonicalize_tag


InteractionType = Literal["search", "similar", "view", "like", "dislike", "reset"]
SearchKind = Literal["title", "genre", "tag"]
RecommendationSource = Literal["interest", "explore", "top", "search", "similar", "detail"]

LONG_EVENT_LIMIT = 240
SHORT_EVENT_LIMIT = 25
LONG_VECTOR_WEIGHT = 0.55
SHORT_VECTOR_WEIGHT = 0.32
EXPLICIT_LIKE_WEIGHT = 0.28
DISLIKE_VECTOR_WEIGHT = 0.45
INTEREST_SHARE = 0.75
EXPLORE_SHARE = 0.25
MAX_GENRE_SHARE = 0.5
DISLIKED_SIMILARITY_LIMIT = 0.72
SEMANTIC_EXPANSION_WEIGHT = 0.3
SEMANTIC_EXPANSION_LIMIT = 5


@dataclass
class InteractionEvent:
    session_id: str
    event_type: InteractionType
    timestamp: float
    kind: SearchKind | None = None
    query: str = ""
    movie_id: int | None = None
    source: RecommendationSource | None = None


@dataclass
class UserPreferenceModel:
    genres: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    tags: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    vector: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    long_genres: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    long_tags: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    long_vector: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    short_genres: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    short_tags: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    short_vector: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    explicit_genres: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    explicit_tags: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    explicit_like_vector: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    disliked_vector: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    semantic_tags: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    semantic_expansions: list[dict] = field(default_factory=list)
    seed_movie_ids: set[int] = field(default_factory=set)
    liked_movie_ids: set[int] = field(default_factory=set)
    disliked_movie_ids: set[int] = field(default_factory=set)
    event_count: int = 0
    signal_strength: float = 0.0
    long_signal_strength: float = 0.0
    short_signal_strength: float = 0.0
    explicit_signal_strength: float = 0.0
    short_weight: float = 0.0
    short_alignment: float = 0.0


class MovieVectorModel:
    def __init__(self, profiles: list[dict]) -> None:
        self.profiles = profiles
        self.by_id = {item["movieId"]: item for item in profiles}
        self.max_values = _numeric_feature_maxima(profiles)
        self.tag_idf = _tag_idf(profiles)
        self.vectors = {item["movieId"]: self._movie_vector(item) for item in profiles}
        self.norms = {movie_id: _norm(vector) for movie_id, vector in self.vectors.items()}
        self.unit_vectors = {
            movie_id: _unit_vector(vector, self.norms[movie_id])
            for movie_id, vector in self.vectors.items()
        }
        self.inverted_unit_index = self._build_inverted_unit_index()

    def movie_vector(self, movie_id: int) -> dict[str, float]:
        return self.vectors.get(movie_id, {})

    def cosine(self, user_vector: dict[str, float], movie_id: int) -> float:
        user_norm = _norm(user_vector)
        return self.cosine_with_norm(user_vector, user_norm, movie_id)

    def cosine_with_norm(self, user_vector: dict[str, float], user_norm: float, movie_id: int) -> float:
        movie_unit_vector = self.unit_vectors.get(movie_id)
        if not movie_unit_vector or not user_norm:
            return 0.0

        dot = 0.0
        smaller, larger = (
            (user_vector, movie_unit_vector)
            if len(user_vector) <= len(movie_unit_vector)
            else (movie_unit_vector, user_vector)
        )
        for key, value in smaller.items():
            dot += value * larger.get(key, 0.0)
        return dot / user_norm

    def cosine_scores(
        self,
        user_vector: dict[str, float],
        exclude_movie_ids: set[int],
    ) -> tuple[dict[int, float], int]:
        user_norm = _norm(user_vector)
        if not user_norm:
            return {}, 0

        dot_scores: defaultdict[int, float] = defaultdict(float)
        posting_count = 0
        for feature, user_weight in user_vector.items():
            postings = self.inverted_unit_index.get(feature, ())
            posting_count += len(postings)
            for movie_id, movie_weight in postings:
                if movie_id in exclude_movie_ids:
                    continue
                dot_scores[movie_id] += user_weight * movie_weight

        cosine_scores = {}
        for movie_id, dot_score in dot_scores.items():
            if dot_score <= 0:
                continue
            cosine_scores[movie_id] = dot_score / user_norm
        return cosine_scores, posting_count

    def _movie_vector(self, item: dict) -> dict[str, float]:
        vector: defaultdict[str, float] = defaultdict(float)

        genres = [normalize(genre) for genre in item.get("genres", []) if genre]
        genre_weight = 2.6 / math.sqrt(max(len(genres), 1))
        for genre in genres:
            vector[f"genre:{genre}"] += genre_weight

        tag_details = _item_tag_details(item)[:40]
        tag_scale = 1.4 / math.sqrt(max(sum(float(tag.get("weight", 1.0)) for tag in tag_details), 1.0))
        for tag in tag_details:
            name = normalize(tag["tag"])
            confidence_weight = min(math.sqrt(float(tag.get("weight", 1.0))), 2.2)
            vector[f"tag:{name}"] += self.tag_idf.get(name, 1.0) * tag_scale * confidence_weight
            vector[f"tag_facet:{tag.get('facet', 'theme')}"] += tag_scale * 0.18 * confidence_weight

        for name, field, weight in (
            ("avg_rating", "avg_rating", 0.7),
            ("trusted_rating", "bayesian_rating", 0.9),
            ("rating_count", "rating_count", 0.55),
            ("recent_rating_count", "recent_rating_count", 0.4),
            ("quality", "comprehensive_score", 0.85),
        ):
            vector[f"num:{name}"] = _normalized_number(item.get(field, 0.0), self.max_values[field]) * weight

        return dict(vector)

    def _build_inverted_unit_index(self) -> dict[str, list[tuple[int, float]]]:
        index: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for movie_id, vector in self.unit_vectors.items():
            for feature, value in vector.items():
                if value:
                    index[feature].append((movie_id, value))
        return dict(index)


class PersonalizationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.events_by_session: dict[str, list[InteractionEvent]] = defaultdict(list)
        self._load()

    def add(self, event: InteractionEvent) -> int:
        if event.event_type == "reset":
            self.events_by_session[event.session_id] = []
        else:
            self.events_by_session[event.session_id].append(event)
        self._append(event)
        return len(self.events_by_session[event.session_id])

    def get(self, session_id: str) -> list[InteractionEvent]:
        return list(self.events_by_session.get(session_id, []))

    def _load(self) -> None:
        if not self.path.exists():
            return

        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = _event_from_dict(json.loads(line))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if event.event_type == "reset":
                    self.events_by_session[event.session_id] = []
                else:
                    self.events_by_session[event.session_id].append(event)

    def _append(self, event: InteractionEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_event_to_dict(event), ensure_ascii=False) + "\n")


def recommend_for_you(
    profiles: list[dict],
    search_engine: MovieLensSearchEngine,
    events: list[InteractionEvent],
    n: int = 10,
    vector_model: MovieVectorModel | None = None,
    tag_semantics: TagSemanticModel | None = None,
) -> dict:
    vector_model = vector_model or MovieVectorModel(profiles)
    memory_events = sorted(events, key=lambda event: event.timestamp)[-LONG_EVENT_LIMIT:]
    model = _build_preference_model(search_engine, vector_model, memory_events, tag_semantics)

    if model.signal_strength < 0.3:
        items = _diverse_cold_start(profiles, n, exclude_movie_ids=model.seed_movie_ids)
        for item in items:
            item["recommendation_bucket"] = "explore"
            item["recommendation_reason"] = "High-scoring movie to start learning your taste."
        return {
            "items": items,
            "count": len(items),
            "event_count": model.event_count,
            "profile": _profile_summary(model),
            "engine": "cold_start_diverse_high_score",
            "status": "cold_start",
            "bucket_counts": {"explore": len(items)},
        }

    interest_candidates = []
    cosine_scores, posting_count = vector_model.cosine_scores(model.vector, model.seed_movie_ids)
    for movie_id, vector_similarity in cosine_scores.items():
        item = vector_model.by_id.get(movie_id)
        if not item:
            continue

        quality_boost = _normalized_number(item.get("comprehensive_score", 0.0), vector_model.max_values["comprehensive_score"]) * 18.0
        personal_score = vector_similarity * 100.0 + quality_boost
        if personal_score <= 0:
            continue

        enriched = dict(item)
        enriched["personal_score"] = round(personal_score, 4)
        enriched["vector_similarity"] = round(vector_similarity, 4)
        enriched["vector_score"] = round(vector_similarity * 100.0, 4)
        enriched["quality_boost"] = round(quality_boost, 4)
        enriched["recommendation_bucket"] = "interest"
        enriched["recommendation_reason"] = _interest_reason(model, item)
        interest_candidates.append(enriched)

    ranked_interest = heapq.nlargest(
        max(n * 18, 160),
        interest_candidates,
        key=lambda item: item["personal_score"],
    )
    exploration_candidates = _exploration_candidates(
        profiles,
        model,
        vector_model,
        exclude_movie_ids=model.seed_movie_ids,
        limit=max(n * 18, 160),
    )
    items = _blend_recommendations(ranked_interest, exploration_candidates, n, model)
    if len(items) < n:
        existing_ids = {item["movieId"] for item in items}
        backfill_excludes = model.seed_movie_ids | existing_ids
        backfill = _diverse_cold_start(profiles, n - len(items), exclude_movie_ids=backfill_excludes)
        for item in backfill:
            item.setdefault("personal_score", round(float(item.get("comprehensive_score", 0.0)), 4))
            item.setdefault("vector_similarity", 0.0)
            item["recommendation_bucket"] = "explore"
            item["recommendation_reason"] = "High-scoring backfill to keep the list fresh."
        items.extend(backfill)

    bucket_counts = Counter(item.get("recommendation_bucket", "interest") for item in items)
    return {
        "items": items,
        "count": len(items),
        "event_count": model.event_count,
        "candidate_count": len(cosine_scores),
        "scored_count": len(interest_candidates),
        "posting_count": posting_count,
        "bucket_counts": dict(bucket_counts),
        "interest_target": _interest_target(n),
        "explore_target": _explore_target(n),
        "profile": _profile_summary(model),
        "engine": "inverted_vector_cosine_blend",
        "status": "personalized",
    }


def _build_preference_model(
    search_engine: MovieLensSearchEngine,
    vector_model: MovieVectorModel,
    events: list[InteractionEvent],
    tag_semantics: TagSemanticModel | None = None,
) -> UserPreferenceModel:
    model = UserPreferenceModel(event_count=len(events))
    newest_first = list(reversed(events))
    explicit_feedback_seen: set[int] = set()
    for index, event in enumerate(newest_first):
        event_weight = _event_weight(event.event_type)
        long_weight = event_weight * (0.985**index) * _long_memory_multiplier(event)
        short_weight = (
            event_weight * (0.82**index) * _short_memory_multiplier(event)
            if index < SHORT_EVENT_LIMIT
            else 0.0
        )
        model.long_signal_strength += abs(long_weight)
        model.short_signal_strength += abs(short_weight)

        if event.movie_id is not None:
            item = search_engine.by_id.get(event.movie_id)
            if item:
                model.seed_movie_ids.add(event.movie_id)
                if event.event_type in {"like", "dislike"} and event.movie_id not in explicit_feedback_seen:
                    explicit_feedback_seen.add(event.movie_id)
                    if event.event_type == "like":
                        model.liked_movie_ids.add(event.movie_id)
                        explicit_weight = abs(event_weight) * (0.992**index) * 1.6
                        model.explicit_signal_strength += explicit_weight
                        _add_movie_signal_to(
                            vector_model,
                            item,
                            explicit_weight,
                            model.explicit_like_vector,
                            model.explicit_genres,
                            model.explicit_tags,
                        )
                    else:
                        model.disliked_movie_ids.add(event.movie_id)
                        dislike_weight = abs(event_weight) * (0.992**index) * 1.5
                        model.explicit_signal_strength += dislike_weight
                        _add_sparse_vector(
                            model.disliked_vector,
                            vector_model.movie_vector(event.movie_id),
                            dislike_weight,
                        )
                _add_movie_signal_to(
                    vector_model,
                    item,
                    long_weight * 1.5,
                    model.long_vector,
                    model.long_genres,
                    model.long_tags,
                )
                _add_movie_signal_to(
                    vector_model,
                    item,
                    short_weight * 1.5,
                    model.short_vector,
                    model.short_genres,
                    model.short_tags,
                )

        if event.event_type == "search" and event.query:
            _add_query_signal_to(
                search_engine,
                vector_model,
                event.kind,
                event.query,
                long_weight,
                model.long_vector,
                model.long_genres,
                model.long_tags,
            )
            _add_query_signal_to(
                search_engine,
                vector_model,
                event.kind,
                event.query,
                short_weight,
                model.short_vector,
                model.short_genres,
                model.short_tags,
            )
        elif event.event_type == "similar" and event.query:
            matches = search_engine.index_title_search(event.query)
            if matches:
                model.seed_movie_ids.add(matches[0]["movieId"])
                _add_movie_signal_to(
                    vector_model,
                    matches[0],
                    long_weight * 1.5,
                    model.long_vector,
                    model.long_genres,
                    model.long_tags,
                )
                _add_movie_signal_to(
                    vector_model,
                    matches[0],
                    short_weight * 1.5,
                    model.short_vector,
                    model.short_genres,
                    model.short_tags,
                )

    _finalize_memory_model(model, tag_semantics)
    return model


def _add_query_signal_to(
    search_engine: MovieLensSearchEngine,
    vector_model: MovieVectorModel,
    kind: SearchKind | None,
    query: str,
    weight: float,
    vector: defaultdict[str, float],
    genres: defaultdict[str, float],
    tags: defaultdict[str, float],
) -> None:
    if not weight:
        return

    normalized = normalize(query)
    if kind == "genre":
        if normalized in search_engine.genre_index:
            genres[normalized] += weight * 2.8
            vector[f"genre:{normalized}"] += weight * 2.8
        matches = search_engine.index_genre_search(query)[:8]
    elif kind == "tag":
        normalized_tag = search_engine.canonicalize_tag(query) or normalized
        if normalized_tag in search_engine.tag_index:
            tags[normalized_tag] += weight * 2.8
            vector[f"tag:{normalized_tag}"] += weight * 2.8
        matches = search_engine.index_tag_search(query)[:8]
    else:
        matches = search_engine.index_title_search(query)[:6]

    for rank, item in enumerate(matches):
        _add_movie_signal_to(vector_model, item, weight * (1.0 / (rank + 1)), vector, genres, tags)


def _add_movie_signal_to(
    vector_model: MovieVectorModel,
    item: dict,
    weight: float,
    vector: defaultdict[str, float],
    genres: defaultdict[str, float],
    tags: defaultdict[str, float],
) -> None:
    if not weight:
        return

    _add_sparse_vector(vector, vector_model.movie_vector(item["movieId"]), weight)
    for genre in item.get("genres", []):
        genres[normalize(genre)] += weight * 1.8
    for tag in _item_tag_details(item)[:12]:
        tags[normalize(tag["tag"])] += weight * 0.9 * float(tag.get("confidence", 1.0))


def _finalize_memory_model(model: UserPreferenceModel, tag_semantics: TagSemanticModel | None = None) -> None:
    model.short_alignment = round(_cosine_vectors(model.long_vector, model.short_vector), 4)
    model.short_weight = round(_dynamic_short_weight(model), 4)

    _add_weighted_sparse(model.vector, model.long_vector, LONG_VECTOR_WEIGHT)
    _add_weighted_sparse(model.vector, model.short_vector, model.short_weight)
    _add_weighted_sparse(model.vector, model.explicit_like_vector, EXPLICIT_LIKE_WEIGHT)
    _add_weighted_sparse(model.vector, model.disliked_vector, -DISLIKE_VECTOR_WEIGHT)

    _merge_weighted_counter(model.genres, model.long_genres, LONG_VECTOR_WEIGHT)
    _merge_weighted_counter(model.genres, model.short_genres, model.short_weight)
    _merge_weighted_counter(model.genres, model.explicit_genres, EXPLICIT_LIKE_WEIGHT)
    _merge_weighted_counter(model.tags, model.long_tags, LONG_VECTOR_WEIGHT)
    _merge_weighted_counter(model.tags, model.short_tags, model.short_weight)
    _merge_weighted_counter(model.tags, model.explicit_tags, EXPLICIT_LIKE_WEIGHT)
    _add_semantic_expansion(model, tag_semantics)

    model.signal_strength = (
        model.long_signal_strength * LONG_VECTOR_WEIGHT
        + model.short_signal_strength * model.short_weight
        + model.explicit_signal_strength * EXPLICIT_LIKE_WEIGHT
    )


def _add_semantic_expansion(model: UserPreferenceModel, tag_semantics: TagSemanticModel | None) -> None:
    if tag_semantics is None:
        return

    positive_tags = {tag: weight for tag, weight in model.tags.items() if weight > 0}
    if not positive_tags:
        return

    expanded = tag_semantics.expand(
        positive_tags,
        limit_per_tag=SEMANTIC_EXPANSION_LIMIT,
        scale=SEMANTIC_EXPANSION_WEIGHT,
    )
    for tag, weight in expanded.items():
        model.semantic_tags[tag] += weight
        model.tags[tag] += weight
        model.vector[f"tag:{tag}"] += weight

    model.semantic_expansions = tag_semantics.explain_expansion(
        positive_tags,
        limit_per_tag=SEMANTIC_EXPANSION_LIMIT,
        scale=SEMANTIC_EXPANSION_WEIGHT,
        max_items=10,
    )


def _dynamic_short_weight(model: UserPreferenceModel) -> float:
    if model.short_signal_strength <= 0:
        return 0.0

    evidence = min(model.short_signal_strength / 10.0, 1.0)
    alignment = max(model.short_alignment, 0.0)
    if model.long_signal_strength < 1.0:
        return min(0.42, 0.22 + evidence * 0.2)

    if alignment >= 0.42:
        return min(0.46, 0.24 + evidence * 0.2)
    if evidence >= 0.75:
        return 0.24
    return 0.1 + evidence * 0.1


def _long_memory_multiplier(event: InteractionEvent) -> float:
    if event.event_type == "search":
        return 0.38
    if event.event_type == "similar":
        return 0.58
    if event.event_type == "view":
        return 0.12 if event.source == "explore" else 0.66
    if event.event_type in {"like", "dislike"}:
        return 1.0
    return 0.0


def _short_memory_multiplier(event: InteractionEvent) -> float:
    if event.event_type == "view" and event.source == "explore":
        return 0.45
    if event.event_type in {"search", "similar", "view"}:
        return 1.0
    if event.event_type in {"like", "dislike"}:
        return 0.85
    return 0.0


def _diverse_cold_start(
    profiles: list[dict],
    n: int,
    exclude_movie_ids: set[int] | None = None,
) -> list[dict]:
    excluded = exclude_movie_ids or set()
    ranked = heap_sort(profiles, key="comprehensive_score", reverse=True)
    return _diversify([dict(item) for item in ranked if item["movieId"] not in excluded], n)


def _exploration_candidates(
    profiles: list[dict],
    model: UserPreferenceModel,
    vector_model: MovieVectorModel,
    exclude_movie_ids: set[int],
    limit: int,
) -> list[dict]:
    ranked = heap_sort(profiles, key="comprehensive_score", reverse=True)
    candidates = []
    user_norm = _norm(model.vector)
    disliked_norm = _norm(model.disliked_vector)
    for item in ranked[: max(limit * 6, 600)]:
        movie_id = item["movieId"]
        if movie_id in exclude_movie_ids:
            continue

        vector_similarity = vector_model.cosine_with_norm(model.vector, user_norm, movie_id)
        disliked_similarity = _cosine_to_vector(vector_model, model.disliked_vector, disliked_norm, movie_id)
        if disliked_similarity >= DISLIKED_SIMILARITY_LIMIT:
            continue

        quality_score = _normalized_number(
            item.get("comprehensive_score", 0.0),
            vector_model.max_values["comprehensive_score"],
        ) * 100.0
        explore_score = quality_score + max(vector_similarity, 0.0) * 12.0 - disliked_similarity * 35.0
        if explore_score <= 0:
            continue

        enriched = dict(item)
        enriched["personal_score"] = round(explore_score, 4)
        enriched["explore_score"] = round(explore_score, 4)
        enriched["vector_similarity"] = round(max(vector_similarity, 0.0), 4)
        enriched["quality_boost"] = round(quality_score, 4)
        enriched["disliked_similarity"] = round(disliked_similarity, 4)
        enriched["recommendation_bucket"] = "explore"
        enriched["recommendation_reason"] = _exploration_reason(model, item, vector_similarity)
        candidates.append(enriched)
        if len(candidates) >= limit:
            break

    return candidates


def _blend_recommendations(
    interest_candidates: list[dict],
    exploration_candidates: list[dict],
    n: int,
    model: UserPreferenceModel,
) -> list[dict]:
    interest_target = _interest_target(n)
    explore_target = _explore_target(n)
    genre_cap = max(2, math.ceil(n * MAX_GENRE_SHARE))
    top_profile_genres = [name for name, value in _positive_ranked(model.genres, 6)]

    selected: list[dict] = []
    selected_ids: set[int] = set()
    genre_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    state = (selected, selected_ids, genre_counts, tag_counts, bucket_counts)

    _take_diverse(
        interest_candidates,
        interest_target,
        state,
        interest_target,
        explore_target,
        genre_cap,
        top_profile_genres,
    )
    _take_diverse(
        exploration_candidates,
        explore_target,
        state,
        interest_target,
        explore_target,
        genre_cap,
        top_profile_genres,
    )
    if len(selected) < n:
        _take_diverse(
            interest_candidates + exploration_candidates,
            n - len(selected),
            state,
            interest_target,
            explore_target,
            genre_cap,
            top_profile_genres,
        )

    return _interleave_for_display(selected)


def _take_diverse(
    candidates: list[dict],
    target_count: int,
    state: tuple[list[dict], set[int], Counter[str], Counter[str], Counter[str]],
    interest_target: int,
    explore_target: int,
    genre_cap: int,
    top_profile_genres: list[str],
) -> None:
    if target_count <= 0:
        return

    selected, selected_ids, genre_counts, tag_counts, bucket_counts = state
    remaining = [dict(item) for item in candidates if item["movieId"] not in selected_ids]
    added = 0
    while remaining and added < target_count:
        best_index = 0
        best_score = float("-inf")
        window = min(len(remaining), 260)
        for index, item in enumerate(remaining[:window]):
            score = _selection_score(
                item,
                genre_counts,
                tag_counts,
                bucket_counts,
                interest_target,
                explore_target,
                genre_cap,
                top_profile_genres,
            )
            if score > best_score:
                best_score = score
                best_index = index

        item = remaining.pop(best_index)
        if item["movieId"] in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item["movieId"])
        bucket_counts[item.get("recommendation_bucket", "interest")] += 1
        for genre in _normalized_genres(item):
            genre_counts[genre] += 1
        for tag in _normalized_tags(item)[:12]:
            tag_counts[tag] += 1
        added += 1


def _interleave_for_display(items: list[dict]) -> list[dict]:
    interest = [item for item in items if item.get("recommendation_bucket") != "explore"]
    explore = [item for item in items if item.get("recommendation_bucket") == "explore"]
    ordered: list[dict] = []
    while interest or explore:
        for _ in range(3):
            if interest:
                ordered.append(interest.pop(0))
        if explore:
            ordered.append(explore.pop(0))
    return ordered


def _selection_score(
    item: dict,
    genre_counts: Counter[str],
    tag_counts: Counter[str],
    bucket_counts: Counter[str],
    interest_target: int,
    explore_target: int,
    genre_cap: int,
    top_profile_genres: list[str],
) -> float:
    bucket = item.get("recommendation_bucket", "interest")
    score = float(item.get("personal_score", item.get("comprehensive_score", 0.0)))

    if bucket == "explore":
        score += 28.0 if bucket_counts["explore"] < explore_target else -32.0
    else:
        score += 10.0 if bucket_counts["interest"] < interest_target else -12.0

    genres = _normalized_genres(item)
    tags = _normalized_tags(item)
    repeated_genre_penalty = sum(genre_counts[genre] for genre in genres[:3]) * 8.0
    over_cap_penalty = sum(1 for genre in genres[:3] if genre_counts[genre] >= genre_cap) * 42.0
    repeated_tag_penalty = sum(tag_counts[tag] for tag in tags[:10]) * 1.4
    new_genre_bonus = 8.0 if any(genre_counts[genre] == 0 for genre in genres[:3]) else 0.0
    secondary_interest_bonus = 0.0
    for rank, genre in enumerate(top_profile_genres[1:], start=1):
        if genre in genres and genre_counts[genre] == 0:
            secondary_interest_bonus = max(secondary_interest_bonus, 10.0 - rank)

    return (
        score
        + new_genre_bonus
        + secondary_interest_bonus
        - repeated_genre_penalty
        - over_cap_penalty
        - repeated_tag_penalty
    )


def _interest_target(n: int) -> int:
    if n <= 1:
        return n
    return max(1, min(n, round(n * INTEREST_SHARE)))


def _explore_target(n: int) -> int:
    return max(0, n - _interest_target(n))


def _interest_reason(model: UserPreferenceModel, item: dict) -> str:
    top_genres = {name for name, _ in _positive_ranked(model.genres, 8)}
    top_tags = {name for name, _ in _positive_ranked(model.tags, 8)}
    matching_genres = [genre for genre in item.get("genres", []) if normalize(genre) in top_genres][:2]
    matching_tags = [tag.get("display", tag["tag"]) for tag in _item_tag_details(item) if normalize(tag["tag"]) in top_tags][:2]
    if matching_genres and matching_tags:
        return f"Matches your {', '.join(matching_genres)} taste and tags like {', '.join(matching_tags)}."
    if matching_genres:
        return f"Matches your {', '.join(matching_genres)} taste."
    if matching_tags:
        return f"Matches tags you reacted to: {', '.join(matching_tags)}."
    return "Close to your recent movie behavior."


def _exploration_reason(model: UserPreferenceModel, item: dict, vector_similarity: float) -> str:
    if vector_similarity > 0.25:
        return "High-scoring exploration with a light connection to your taste."
    top_genres = {name for name, _ in _positive_ranked(model.genres, 4)}
    new_genres = [genre for genre in item.get("genres", []) if normalize(genre) not in top_genres][:2]
    if new_genres:
        return f"High-scoring exploration outside your usual {', '.join(new_genres)} mix."
    return "High-scoring exploration to test a new direction."


def _cosine_to_vector(
    vector_model: MovieVectorModel,
    target_vector: dict[str, float],
    target_norm: float,
    movie_id: int,
) -> float:
    if not target_norm:
        return 0.0
    return max(vector_model.cosine_with_norm(target_vector, target_norm, movie_id), 0.0)


def _normalized_genres(item: dict) -> list[str]:
    return [normalize(genre) for genre in item.get("genres", []) if genre]


def _normalized_tags(item: dict) -> list[str]:
    return [normalize(tag["tag"]) for tag in _item_tag_details(item)]


def _item_tag_details(item: dict) -> list[dict]:
    details = item.get("tag_details") or []
    if details:
        return [tag for tag in details if tag.get("tag")]
    return [
        {
            "tag": normalize(tag),
            "display": str(tag),
            "count": 1,
            "facet": "theme",
            "confidence": 0.65,
            "weight": 1.0,
        }
        for tag in item.get("tags", [])
        if tag
    ]


def _positive_ranked(values: dict[str, float], limit: int) -> list[tuple[str, float]]:
    return [
        (name, value)
        for name, value in sorted(values.items(), key=lambda item: item[1], reverse=True)
        if value > 0
    ][:limit]


def _diversify(items: list[dict], n: int) -> list[dict]:
    selected: list[dict] = []
    genre_counts: Counter[str] = Counter()
    remaining = list(items)

    while remaining and len(selected) < n:
        best_index = 0
        best_score = float("-inf")
        for index, item in enumerate(remaining[: min(len(remaining), 120)]):
            genres = [normalize(genre) for genre in item.get("genres", [])]
            crowding_penalty = sum(genre_counts[genre] for genre in genres[:3]) * 4.0
            base = float(item.get("personal_score", item.get("comprehensive_score", 0.0)))
            diversity_bonus = 6.0 if any(genre_counts[genre] == 0 for genre in genres[:3]) else 0.0
            score = base + diversity_bonus - crowding_penalty
            if score > best_score:
                best_score = score
                best_index = index

        item = remaining.pop(best_index)
        selected.append(item)
        for genre in item.get("genres", [])[:3]:
            genre_counts[normalize(genre)] += 1

    return selected


def _profile_summary(model: UserPreferenceModel) -> dict:
    return {
        "top_genres": _top_weighted(model.genres, 5),
        "top_tags": _top_weighted(model.tags, 5),
        "semantic_tags": _top_weighted(model.semantic_tags, 5),
        "semantic_expansions": model.semantic_expansions[:5],
        "long_term_genres": _top_weighted(model.long_genres, 5),
        "short_term_genres": _top_weighted(model.short_genres, 5),
        "explicit_genres": _top_weighted(model.explicit_genres, 5),
        "seed_movie_count": len(model.seed_movie_ids),
        "liked_movie_count": len(model.liked_movie_ids),
        "disliked_movie_count": len(model.disliked_movie_ids),
        "signal_strength": round(model.signal_strength, 4),
        "long_signal_strength": round(model.long_signal_strength, 4),
        "short_signal_strength": round(model.short_signal_strength, 4),
        "explicit_signal_strength": round(model.explicit_signal_strength, 4),
        "short_weight": model.short_weight,
        "short_alignment": model.short_alignment,
        "vector_feature_count": len(model.vector),
    }


def _top_weighted(values: dict[str, float], limit: int) -> list[dict]:
    return [{"name": key, "weight": round(value, 3)} for key, value in _positive_ranked(values, limit)]


def _event_weight(event_type: InteractionType) -> float:
    return {
        "search": 1.0,
        "similar": 2.2,
        "view": 2.8,
        "like": 7.0,
        "dislike": -7.5,
        "reset": 0.0,
    }.get(event_type, 0.0)


def _numeric_feature_maxima(profiles: list[dict]) -> dict[str, float]:
    fields = (
        "avg_rating",
        "bayesian_rating",
        "rating_count",
        "recent_rating_count",
        "comprehensive_score",
    )
    maxima = {}
    for field in fields:
        maxima[field] = max(float(item.get(field, 0.0) or 0.0) for item in profiles) or 1.0
    return maxima


def _tag_idf(profiles: list[dict]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for item in profiles:
        document_frequency.update({normalize(tag["tag"]) for tag in _item_tag_details(item)})

    total = len(profiles) or 1
    return {
        tag: min(math.log((total + 1) / (frequency + 1)) + 1.0, 4.2)
        for tag, frequency in document_frequency.items()
    }


def _normalized_number(value: object, maximum: float) -> float:
    number = float(value or 0.0)
    if maximum <= 0:
        return 0.0
    return min(max(number / maximum, 0.0), 1.0)


def _add_sparse_vector(target: defaultdict[str, float], source: dict[str, float], weight: float) -> None:
    if not weight:
        return
    for key, value in source.items():
        target[key] += value * weight


def _add_weighted_sparse(target: defaultdict[str, float], source: dict[str, float], scale: float) -> None:
    _add_sparse_vector(target, source, scale)


def _merge_weighted_counter(target: defaultdict[str, float], source: dict[str, float], scale: float) -> None:
    if not scale:
        return
    for key, value in source.items():
        target[key] += value * scale


def _norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _cosine_vectors(left: dict[str, float], right: dict[str, float]) -> float:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if not left_norm or not right_norm:
        return 0.0

    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    dot = sum(value * larger.get(key, 0.0) for key, value in smaller.items())
    return dot / (left_norm * right_norm)


def _unit_vector(vector: dict[str, float], norm: float) -> dict[str, float]:
    if not norm:
        return {}
    return {key: value / norm for key, value in vector.items()}


def _event_from_dict(raw: dict) -> InteractionEvent:
    return InteractionEvent(
        session_id=str(raw["session_id"])[:120],
        event_type=raw["event_type"],
        timestamp=float(raw.get("timestamp") or time.time()),
        kind=raw.get("kind"),
        query=str(raw.get("query") or "")[:160],
        movie_id=int(raw["movie_id"]) if raw.get("movie_id") is not None else None,
        source=raw.get("source"),
    )


def _event_to_dict(event: InteractionEvent) -> dict:
    return {
        "session_id": event.session_id,
        "event_type": event.event_type,
        "timestamp": event.timestamp,
        "kind": event.kind,
        "query": event.query,
        "movie_id": event.movie_id,
        "source": event.source,
    }
