"""Vectorized / batched user-user collaborative filtering.

This module replaces the per-user Python double loops in
:func:`evaluation._compute_user_neighbors` and
:func:`evaluation._score_neighbors_full_catalog` with sparse matrix algebra.

Neighbor discovery is the O(N^2) bottleneck of the original implementation:
for every active user it scans every other user that shares a rated movie. Here
the same computation is expressed as two sparse matrix products,

    dot     = C[A] @ C.T        (centered-rating dot products)
    shared  = M[A] @ M.T        (shared-movie counts)

evaluated in blocks of active users so memory stays bounded. Full-catalog
candidate scoring is then a small sparse product over each user's neighbours.

The arithmetic is performed in float64, matching the Python reference, so the
results are numerically equivalent (differences are at floating-point rounding
level, ~1e-12). Ranking ties may break differently, which is acceptable for the
recommendation metrics.
"""

from __future__ import annotations

import math
import os

import numpy as np
from scipy import sparse

from src.datasets.netflix.evaluation import (
    MIN_SHARED_MOVIES,
    NEIGHBOR_LIMIT,
    OVERLAP_SMOOTHING,
    SUPPORT_LIMIT,
)

MIN_PROFILE_MOVIES = 3

# --- ANN (approximate neighbour discovery) defaults -------------------------
# The exact engine computes ``C[A] @ C.T`` over every user, which is the
# O(N^2) wall at 480k users. The ANN path projects the sparse centred vectors
# to a dense low-dimensional space (Johnson-Lindenstrauss), retrieves a small
# over-fetched candidate set with hnswlib (cosine), and then reranks that set
# with the *exact* similarity formula. Only the candidate set is approximate;
# the similarity values kept for the final neighbours are exact.
ANN_PROJECTION_DIM = 512
ANN_OVER_FETCH = 20
ANN_EF_CONSTRUCTION = 200
ANN_M = 24
ANN_SEED = 20240517
# ``auto`` switches to ANN once the candidate-user pool is large enough that the
# dense O(N^2) product stops being cheap. The exact sparse engine is both faster
# and numerically equivalent in the tens-of-thousands range, so ANN only takes
# over for genuinely large pools where the exact product blows up.
ANN_AUTO_MIN_USERS = 50000


def _resolve_cf_method(n_users: int, method: str | None) -> str:
    """Return ``"ann"`` or ``"exact"`` from an explicit arg / env / auto rule."""
    chosen = (method or os.environ.get("NETFLIX_CF_ENGINE") or "auto").lower()
    if chosen == "auto":
        threshold = int(os.environ.get("NETFLIX_CF_ANN_MIN", ANN_AUTO_MIN_USERS))
        return "ann" if n_users >= threshold else "exact"
    if chosen not in {"ann", "exact"}:
        raise ValueError("NETFLIX_CF_ENGINE must be one of: auto, ann, exact")
    return chosen


class VectorizedCollaborativeEngine:
    def __init__(self, indexes) -> None:
        user_ratings, user_centered, user_norms, _movie_users = indexes

        self.user_ids = sorted(user_centered.keys())
        self.user_row = {user_id: row for row, user_id in enumerate(self.user_ids)}
        n_users = len(self.user_ids)

        movie_set: set[int] = set()
        for ratings in user_ratings.values():
            movie_set.update(ratings.keys())
        self.movie_ids = np.array(sorted(movie_set), dtype=np.int64)
        movie_col = {int(movie_id): col for col, movie_id in enumerate(self.movie_ids)}
        n_movies = len(self.movie_ids)

        self.n_users = n_users
        self.n_movies = n_movies
        self.user_norms = np.array(
            [float(user_norms[user_id]) for user_id in self.user_ids], dtype=np.float64
        )

        centered_rows: list[int] = []
        centered_cols: list[int] = []
        centered_vals: list[float] = []
        mask_rows: list[int] = []
        mask_cols: list[int] = []
        positive_rows: list[int] = []
        positive_cols: list[int] = []
        positive_vals: list[float] = []

        for user_id in self.user_ids:
            row = self.user_row[user_id]
            ratings = user_ratings[user_id]
            centered = user_centered[user_id]
            for movie_id, rating in ratings.items():
                col = movie_col[int(movie_id)]
                centered_value = float(centered.get(movie_id, 0.0))
                mask_rows.append(row)
                mask_cols.append(col)
                if centered_value != 0.0:
                    centered_rows.append(row)
                    centered_cols.append(col)
                    centered_vals.append(centered_value)
                if rating >= 4.0 and centered_value > 0.0:
                    positive_rows.append(row)
                    positive_cols.append(col)
                    positive_vals.append(centered_value)

        shape = (n_users, n_movies)
        self.centered = sparse.csr_matrix(
            (np.array(centered_vals, dtype=np.float64), (centered_rows, centered_cols)),
            shape=shape,
        )
        self.mask = sparse.csr_matrix(
            (np.ones(len(mask_rows), dtype=np.float64), (mask_rows, mask_cols)),
            shape=shape,
        )
        self.positive = sparse.csr_matrix(
            (np.array(positive_vals, dtype=np.float64), (positive_rows, positive_cols)),
            shape=shape,
        )
        self.positive_mask = sparse.csr_matrix(
            (np.ones(len(positive_rows), dtype=np.float64), (positive_rows, positive_cols)),
            shape=shape,
        )
        self.profile_sizes = np.asarray(self.mask.sum(axis=1)).ravel()

        self._centered_t = self.centered.transpose().tocsr()
        self._mask_t = self.mask.transpose().tocsr()
        self._positive_t = self.positive.transpose().tocsr()
        self._positive_mask_t = self.positive_mask.transpose().tocsr()
        self._log_neighbor_limit = math.log1p(NEIGHBOR_LIMIT)

        # Lazily-built ANN structures (only used by the approximate path).
        self._ann_index = None
        self._ann_vectors: np.ndarray | None = None

        self._empty_movie_ids = np.empty(0, dtype=np.int64)
        self._empty_scores = np.empty(0, dtype=np.float64)

    def precompute(
        self,
        active_user_ids,
        *,
        block_size: int = 512,
    ) -> tuple[dict[int, list[dict]], dict[int, dict[int, float]], dict[int, tuple[np.ndarray, np.ndarray]]]:
        """Return ``(neighbor_cache, raw_score_cache, raw_array_cache)``.

        ``neighbor_cache[user_id]`` mirrors ``_compute_user_neighbors`` output,
        ``raw_score_cache[user_id]`` mirrors ``_score_neighbors_full_catalog``
        (dict, for legacy consumers), and ``raw_array_cache[user_id]`` is the same
        scores as ``(ascending movie_ids, values)`` arrays for vectorized lookup.
        """
        neighbor_cache: dict[int, list[dict]] = {}
        raw_score_cache: dict[int, dict[int, float]] = {}
        raw_array_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        eligible_rows: list[int] = []
        eligible_user_ids: list[int] = []
        for user_id in active_user_ids:
            row = self.user_row.get(user_id)
            if row is None or self.profile_sizes[row] < MIN_PROFILE_MOVIES:
                neighbor_cache[user_id] = []
                continue
            eligible_rows.append(row)
            eligible_user_ids.append(user_id)

        if not eligible_rows:
            return neighbor_cache, raw_score_cache, raw_array_cache

        eligible_rows_array = np.array(eligible_rows, dtype=np.int64)
        for start in range(0, len(eligible_rows_array), block_size):
            block_rows = eligible_rows_array[start : start + block_size]
            dot_block = (self.centered[block_rows] @ self._centered_t).toarray()
            shared_block = (self.mask[block_rows] @ self._mask_t).toarray()

            for offset, row in enumerate(block_rows):
                user_id = eligible_user_ids[start + offset]
                neighbors, neighbor_rows, neighbor_sims = self._neighbors_for_row(
                    int(row), dot_block[offset], shared_block[offset]
                )
                neighbor_cache[user_id] = neighbors
                if neighbor_rows is None:
                    continue
                self._store_raw_scores(
                    user_id, neighbor_rows, neighbor_sims, raw_score_cache, raw_array_cache
                )
        return neighbor_cache, raw_score_cache, raw_array_cache

    def _store_raw_scores(
        self,
        user_id: int,
        neighbor_rows: np.ndarray,
        neighbor_sims: np.ndarray,
        raw_score_cache: dict[int, dict[int, float]],
        raw_array_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        movie_ids, scores = self._raw_scores_for_neighbors(neighbor_rows, neighbor_sims)
        raw_array_cache[user_id] = (movie_ids, scores)
        raw_score_cache[user_id] = {
            int(movie_id): float(score) for movie_id, score in zip(movie_ids, scores)
        }

    def _neighbors_for_row(self, row: int, dot: np.ndarray, shared: np.ndarray):
        dot = dot.copy()
        shared = shared.copy()
        dot[row] = 0.0
        shared[row] = 0.0

        candidate_mask = (shared >= MIN_SHARED_MOVIES) & (dot > 0.0)
        candidate_indices = np.flatnonzero(candidate_mask)
        if len(candidate_indices) == 0:
            return [], None, None

        norm_active = self.user_norms[row]
        raw_similarity = dot[candidate_indices] / (norm_active * self.user_norms[candidate_indices])
        shared_counts = shared[candidate_indices]
        overlap_weight = shared_counts / (shared_counts + OVERLAP_SMOOTHING)
        similarity = raw_similarity * overlap_weight

        if len(candidate_indices) > NEIGHBOR_LIMIT:
            top = np.argpartition(-similarity, NEIGHBOR_LIMIT - 1)[:NEIGHBOR_LIMIT]
        else:
            top = np.arange(len(candidate_indices))
        order = top[np.argsort(-similarity[top], kind="stable")]

        neighbor_rows = candidate_indices[order]
        neighbor_sims = similarity[order]
        neighbors = [
            {"user_id": int(self.user_ids[int(neighbor_row)]), "similarity": float(sim)}
            for neighbor_row, sim in zip(neighbor_rows, neighbor_sims)
        ]
        return neighbors, neighbor_rows, neighbor_sims

    def _raw_scores_for_neighbors(
        self, neighbor_rows: np.ndarray, neighbor_sims: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(movie_ids, scores)`` for movies reachable from the neighbours.

        ``movie_ids`` is ascending (``self.movie_ids`` is sorted and ``columns``
        comes from ``flatnonzero``), so callers can ``searchsorted`` into it. The
        values match :func:`evaluation._score_neighbors_full_catalog`.
        """
        positive_block = self.positive[neighbor_rows]
        positive_mask_block = self.positive_mask[neighbor_rows]

        weighted = positive_block.transpose().tocsr() @ neighbor_sims
        similarity_sums = positive_mask_block.transpose().tocsr() @ np.abs(neighbor_sims)
        support = np.asarray(positive_mask_block.sum(axis=0)).ravel()

        valid = (support >= SUPPORT_LIMIT) & (similarity_sums > 0.0)
        columns = np.flatnonzero(valid)
        if len(columns) == 0:
            return self._empty_movie_ids, self._empty_scores

        support_bonus = np.minimum(np.log1p(support[columns]) / self._log_neighbor_limit, 1.0)
        scores = (weighted[columns] / similarity_sums[columns]) * 80.0 + support_bonus * 20.0
        movie_ids = self.movie_ids[columns].astype(np.int64, copy=False)
        return movie_ids, scores

    # --- Approximate (ANN) neighbour discovery ------------------------------
    def build_ann_index(
        self,
        *,
        projection_dim: int = ANN_PROJECTION_DIM,
        ef_construction: int = ANN_EF_CONSTRUCTION,
        m: int = ANN_M,
        seed: int = ANN_SEED,
    ) -> None:
        """Build a hnswlib cosine index over random-projected centred vectors.

        The centred sparse matrix ``C`` (n_users x n_movies) is projected with a
        Gaussian random matrix ``R`` (n_movies x d) into a dense ``P = C @ R``
        (n_users x d). Cosine distances in ``P`` approximate cosine distances in
        the original space (Johnson-Lindenstrauss), and hnswlib gives sub-linear
        approximate retrieval. The exact rerank later restores precise scores.
        """
        import hnswlib

        dim = min(projection_dim, self.n_movies) if self.n_movies else projection_dim
        rng = np.random.default_rng(seed)
        projection = rng.standard_normal((self.n_movies, dim)).astype(np.float32)
        projection /= math.sqrt(dim)
        vectors = np.asarray(self.centered.dot(projection), dtype=np.float32)

        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(
            max_elements=self.n_users,
            ef_construction=ef_construction,
            M=m,
            random_seed=seed,
        )
        index.add_items(vectors, np.arange(self.n_users, dtype=np.int64))
        self._ann_index = index
        self._ann_vectors = vectors

    def _neighbors_from_candidate_rows(self, row: int, candidate_rows: np.ndarray):
        """Exact-rerank an ANN candidate set into the final neighbour list.

        Mirrors :meth:`_neighbors_for_row` but works on a small candidate subset
        instead of the full dense row, so the similarity values are computed with
        the exact formula (only the candidate *set* is approximate).
        """
        candidate_rows = candidate_rows[candidate_rows != row]
        if len(candidate_rows) == 0:
            return [], None, None

        active_centered = self.centered[row]
        active_mask = self.mask[row]
        dot = np.asarray(self.centered[candidate_rows].dot(active_centered.T).todense()).ravel()
        shared = np.asarray(self.mask[candidate_rows].dot(active_mask.T).todense()).ravel()

        keep = (shared >= MIN_SHARED_MOVIES) & (dot > 0.0)
        if not keep.any():
            return [], None, None
        candidate_rows = candidate_rows[keep]
        dot = dot[keep]
        shared = shared[keep]

        norm_active = self.user_norms[row]
        raw_similarity = dot / (norm_active * self.user_norms[candidate_rows])
        overlap_weight = shared / (shared + OVERLAP_SMOOTHING)
        similarity = raw_similarity * overlap_weight

        if len(candidate_rows) > NEIGHBOR_LIMIT:
            top = np.argpartition(-similarity, NEIGHBOR_LIMIT - 1)[:NEIGHBOR_LIMIT]
        else:
            top = np.arange(len(candidate_rows))
        order = top[np.argsort(-similarity[top], kind="stable")]

        neighbor_rows = candidate_rows[order]
        neighbor_sims = similarity[order]
        neighbors = [
            {"user_id": int(self.user_ids[int(neighbor_row)]), "similarity": float(sim)}
            for neighbor_row, sim in zip(neighbor_rows, neighbor_sims)
        ]
        return neighbors, neighbor_rows, neighbor_sims

    def precompute_ann(
        self,
        active_user_ids,
        *,
        over_fetch: int = ANN_OVER_FETCH,
        block_size: int = 1024,
    ) -> tuple[dict[int, list[dict]], dict[int, dict[int, float]], dict[int, tuple[np.ndarray, np.ndarray]]]:
        """ANN counterpart of :meth:`precompute` (approximate candidate set).

        The expensive ``C[A] @ C.T`` over every user is replaced by an
        approximate cosine search (hnswlib, batched/multithreaded per block).
        Each returned candidate set is then exact-reranked, so neighbour
        similarity values are exact and only the candidate *set* is approximate.
        """
        if self._ann_index is None:
            self.build_ann_index()

        neighbor_cache: dict[int, list[dict]] = {}
        raw_score_cache: dict[int, dict[int, float]] = {}
        raw_array_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        eligible_rows: list[int] = []
        eligible_user_ids: list[int] = []
        for user_id in active_user_ids:
            row = self.user_row.get(user_id)
            if row is None or self.profile_sizes[row] < MIN_PROFILE_MOVIES:
                neighbor_cache[user_id] = []
                continue
            eligible_rows.append(row)
            eligible_user_ids.append(user_id)

        if not eligible_rows:
            return neighbor_cache, raw_score_cache, raw_array_cache

        query_k = min(self.n_users, NEIGHBOR_LIMIT * over_fetch + 1)
        self._ann_index.set_ef(max(query_k + 16, ANN_EF_CONSTRUCTION))

        eligible_rows_array = np.array(eligible_rows, dtype=np.int64)
        vectors = self._ann_vectors
        for start in range(0, len(eligible_rows_array), block_size):
            block_rows = eligible_rows_array[start : start + block_size]
            labels, _distances = self._ann_index.knn_query(vectors[block_rows], k=query_k)
            for offset, row in enumerate(block_rows):
                user_id = eligible_user_ids[start + offset]
                candidate_rows = np.asarray(labels[offset], dtype=np.int64)
                neighbors, neighbor_rows, neighbor_sims = self._neighbors_from_candidate_rows(
                    int(row), candidate_rows
                )
                neighbor_cache[user_id] = neighbors
                if neighbor_rows is None:
                    continue
                self._store_raw_scores(
                    user_id, neighbor_rows, neighbor_sims, raw_score_cache, raw_array_cache
                )
        return neighbor_cache, raw_score_cache, raw_array_cache


def prime_collaborative_caches(
    indexes,
    active_user_ids,
    feature_stats: dict,
    *,
    block_size: int = 512,
    method: str | None = None,
) -> VectorizedCollaborativeEngine:
    """Batch-precompute user-user CF neighbours and raw scores into ``feature_stats``.

    Populates ``feature_stats['user_neighbor_cache']`` and
    ``feature_stats['collaborative_raw_cache']`` so the recall, evaluation, and
    feature-building stages all reuse the vectorized results instead of running
    the per-user Python loops.

    ``method`` selects neighbour discovery: ``"exact"`` (sparse matrix product,
    numerically equivalent) or ``"ann"`` (approximate, for very large user
    pools). ``None``/``"auto"`` reads ``NETFLIX_CF_ENGINE`` and otherwise switches
    to ANN once the user pool crosses ``NETFLIX_CF_ANN_MIN``.
    """
    engine = VectorizedCollaborativeEngine(indexes)
    resolved = _resolve_cf_method(engine.n_users, method)
    if resolved == "ann":
        neighbor_cache, raw_score_cache, raw_array_cache = engine.precompute_ann(active_user_ids)
    else:
        neighbor_cache, raw_score_cache, raw_array_cache = engine.precompute(
            active_user_ids, block_size=block_size
        )
    feature_stats["user_neighbor_cache"] = neighbor_cache
    feature_stats["collaborative_raw_cache"] = raw_score_cache
    feature_stats["collaborative_raw_array_cache"] = raw_array_cache
    feature_stats["collaborative_engine_method"] = resolved
    return engine
