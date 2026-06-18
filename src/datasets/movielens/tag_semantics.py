from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from src.datasets.movielens.search import normalize


DEFAULT_DIMENSIONS = 48
DEFAULT_NEIGHBOR_LIMIT = 10
DEFAULT_MIN_MOVIE_COUNT = 3
DEFAULT_MIN_SIMILARITY = 0.24
DEFAULT_MIN_SHARED_MOVIES = 1


@dataclass(frozen=True)
class TagNeighbor:
    tag: str
    similarity: float
    movie_count: int
    shared_movies: int

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "similarity": self.similarity,
            "movie_count": self.movie_count,
            "shared_movies": self.shared_movies,
        }


class TagSemanticModel:
    def __init__(self, neighbors_by_tag: dict[str, list[TagNeighbor]], summary: dict) -> None:
        self.neighbors_by_tag = neighbors_by_tag
        self.summary = summary

    @classmethod
    def from_profiles(
        cls,
        profiles: Iterable[dict],
        cache_path: Path | None = None,
        dimensions: int = DEFAULT_DIMENSIONS,
        neighbor_limit: int = DEFAULT_NEIGHBOR_LIMIT,
        min_movie_count: int = DEFAULT_MIN_MOVIE_COUNT,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        min_shared_movies: int = DEFAULT_MIN_SHARED_MOVIES,
    ) -> "TagSemanticModel":
        profile_list = list(profiles)
        signature = _profile_signature(
            profile_list,
            dimensions,
            neighbor_limit,
            min_movie_count,
            min_similarity,
            min_shared_movies,
        )
        cached = _load_cached(cache_path, signature)
        if cached is not None:
            return cached

        started = time.perf_counter()
        neighbors_by_tag, stats = _train_neighbors(
            profile_list,
            dimensions=dimensions,
            neighbor_limit=neighbor_limit,
            min_movie_count=min_movie_count,
            min_similarity=min_similarity,
            min_shared_movies=min_shared_movies,
        )
        summary = {
            **stats,
            "engine": "tag_movie_lsa",
            "cache_status": "miss",
            "signature": signature,
            "dimensions": dimensions,
            "neighbor_limit": neighbor_limit,
            "min_movie_count": min_movie_count,
            "min_similarity": min_similarity,
            "min_shared_movies": min_shared_movies,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 4),
        }
        model = cls(neighbors_by_tag, summary)
        _save_cached(cache_path, model)
        return model

    def neighbors(self, tag: str, limit: int = DEFAULT_NEIGHBOR_LIMIT) -> list[dict]:
        key = normalize(tag)
        return [neighbor.to_dict() for neighbor in self.neighbors_by_tag.get(key, [])[:limit]]

    def expand(self, tag_weights: dict[str, float], limit_per_tag: int = 5, scale: float = 0.3) -> dict[str, float]:
        expanded: defaultdict[str, float] = defaultdict(float)
        for tag, weight in sorted(tag_weights.items(), key=lambda item: item[1], reverse=True):
            if weight <= 0:
                continue
            for neighbor in self.neighbors_by_tag.get(normalize(tag), [])[:limit_per_tag]:
                if neighbor.tag == tag:
                    continue
                expanded[neighbor.tag] += weight * scale * max(neighbor.similarity, 0.0)
        return dict(expanded)

    def explain_expansion(
        self,
        tag_weights: dict[str, float],
        limit_per_tag: int = 5,
        scale: float = 0.3,
        max_items: int = 10,
    ) -> list[dict]:
        rows = []
        for source, weight in sorted(tag_weights.items(), key=lambda item: item[1], reverse=True):
            if weight <= 0:
                continue
            for neighbor in self.neighbors_by_tag.get(normalize(source), [])[:limit_per_tag]:
                added_weight = weight * scale * max(neighbor.similarity, 0.0)
                if added_weight <= 0:
                    continue
                rows.append(
                    {
                        "source": source,
                        "tag": neighbor.tag,
                        "weight": round(added_weight, 3),
                        "similarity": neighbor.similarity,
                    }
                )
        rows.sort(key=lambda item: item["weight"], reverse=True)
        return rows[:max_items]


def _train_neighbors(
    profiles: list[dict],
    dimensions: int,
    neighbor_limit: int,
    min_movie_count: int,
    min_similarity: float,
    min_shared_movies: int,
) -> tuple[dict[str, list[TagNeighbor]], dict]:
    tag_movie_weights: dict[str, dict[int, float]] = defaultdict(dict)
    tag_movies: dict[str, set[int]] = defaultdict(set)

    for movie_index, item in enumerate(profiles):
        for tag in _item_tag_details(item):
            key = normalize(tag["tag"])
            if not key:
                continue
            weight = max(float(tag.get("weight", tag.get("count", 1.0)) or 1.0), 0.0)
            if weight <= 0:
                continue
            tag_movie_weights[key][movie_index] = max(tag_movie_weights[key].get(movie_index, 0.0), weight)
            tag_movies[key].add(movie_index)

    tags = sorted(tag for tag, movies in tag_movies.items() if len(movies) >= min_movie_count)
    if len(tags) < 2:
        return {}, {
            "tag_count": len(tags),
            "movie_count": len(profiles),
            "neighbor_count": 0,
            "status": "not_enough_tags",
        }

    matrix = _build_tfidf_matrix(tags, tag_movie_weights, movie_count=len(profiles))
    embeddings = _lsa_embeddings(matrix, dimensions)
    if embeddings.size == 0:
        return {}, {
            "tag_count": len(tags),
            "movie_count": len(profiles),
            "neighbor_count": 0,
            "status": "empty_embeddings",
        }

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0)
    similarities = unit @ unit.T

    neighbors_by_tag: dict[str, list[TagNeighbor]] = {}
    total_neighbors = 0
    tag_movie_counts = {tag: len(tag_movies[tag]) for tag in tags}
    for row_index, tag in enumerate(tags):
        scored = []
        for col_index, other in enumerate(tags):
            if row_index == col_index:
                continue
            similarity = float(similarities[row_index, col_index])
            if similarity < min_similarity:
                continue
            shared_movies = len(tag_movies[tag] & tag_movies[other])
            if shared_movies < min_shared_movies:
                continue
            scored.append(
                TagNeighbor(
                    tag=other,
                    similarity=round(similarity, 4),
                    movie_count=tag_movie_counts[other],
                    shared_movies=shared_movies,
                )
            )
        scored.sort(key=lambda item: (item.similarity, item.shared_movies, item.movie_count, item.tag), reverse=True)
        neighbors_by_tag[tag] = scored[:neighbor_limit]
        total_neighbors += len(neighbors_by_tag[tag])

    return neighbors_by_tag, {
        "tag_count": len(tags),
        "movie_count": len(profiles),
        "neighbor_count": total_neighbors,
        "status": "ready",
    }


def _build_tfidf_matrix(
    tags: list[str],
    tag_movie_weights: dict[str, dict[int, float]],
    movie_count: int,
) -> np.ndarray:
    matrix = np.zeros((len(tags), movie_count), dtype=np.float32)
    for row_index, tag in enumerate(tags):
        movies = tag_movie_weights[tag]
        idf = math.log((movie_count + 1) / (len(movies) + 1)) + 1.0
        for movie_index, weight in movies.items():
            matrix[row_index, movie_index] = math.log1p(weight) * idf

    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, row_norms, out=np.zeros_like(matrix), where=row_norms > 0)


def _lsa_embeddings(matrix: np.ndarray, dimensions: int) -> np.ndarray:
    gram = matrix @ matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    positive = [index for index in order if eigenvalues[index] > 1e-8]
    if not positive:
        return np.empty((matrix.shape[0], 0), dtype=np.float32)
    selected = positive[: min(dimensions, len(positive))]
    values = np.sqrt(np.maximum(eigenvalues[selected], 0.0))
    return (eigenvectors[:, selected] * values).astype(np.float32)


def _profile_signature(
    profiles: list[dict],
    dimensions: int,
    neighbor_limit: int,
    min_movie_count: int,
    min_similarity: float,
    min_shared_movies: int,
) -> str:
    tag_movie_weights: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
    for item in profiles:
        movie_id = int(item.get("movieId", 0))
        for tag in _item_tag_details(item):
            key = normalize(tag["tag"])
            if key:
                weight = round(float(tag.get("weight", tag.get("count", 1.0)) or 1.0), 6)
                tag_movie_weights[key].append((movie_id, weight))
    payload = {
        "signature_version": 2,
        "movie_count": len(profiles),
        "tag_movie_weights": [
            (tag, sorted(values))
            for tag, values in sorted(tag_movie_weights.items())
        ],
        "dimensions": dimensions,
        "neighbor_limit": neighbor_limit,
        "min_movie_count": min_movie_count,
        "min_similarity": min_similarity,
        "min_shared_movies": min_shared_movies,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_cached(cache_path: Path | None, signature: str) -> TagSemanticModel | None:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if raw.get("summary", {}).get("signature") != signature:
        return None

    neighbors_by_tag = {
        tag: [TagNeighbor(**neighbor) for neighbor in neighbors]
        for tag, neighbors in raw.get("neighbors_by_tag", {}).items()
    }
    summary = dict(raw.get("summary") or {})
    summary["cache_status"] = "hit"
    return TagSemanticModel(neighbors_by_tag, summary)


def _save_cached(cache_path: Path | None, model: TagSemanticModel) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": model.summary,
        "neighbors_by_tag": {
            tag: [neighbor.to_dict() for neighbor in neighbors]
            for tag, neighbors in model.neighbors_by_tag.items()
        },
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _item_tag_details(item: dict) -> list[dict]:
    details = item.get("tag_details") or []
    if details:
        return [tag for tag in details if tag.get("tag")]
    return [{"tag": tag, "weight": 1.0} for tag in item.get("tags", []) if tag]
