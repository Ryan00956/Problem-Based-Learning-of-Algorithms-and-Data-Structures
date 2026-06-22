from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MatrixFactorizationConfig:
    factors: int = 48
    epochs: int = 20
    learning_rate: float = 0.02
    regularization: float = 0.04
    seed: int = 42
    min_rating: float = 1.0
    max_rating: float = 5.0
    backend: str = "auto"
    device: str = "auto"
    batch_size: int = 8192
    optimizer: str = "adam"


@dataclass(frozen=True)
class TrainingRating:
    user_id: int
    movie_id: int
    rating: float


class MatrixFactorizationModel:
    """Biased matrix factorization trained with SGD.

    The model follows the classic Netflix Prize shape:
    rating ~= global_mean + user_bias + movie_bias + user_vector dot movie_vector.
    """

    def __init__(self, config: MatrixFactorizationConfig | None = None) -> None:
        self.config = config or MatrixFactorizationConfig()
        self.global_mean = 0.0
        self.user_to_index: dict[int, int] = {}
        self.movie_to_index: dict[int, int] = {}
        self.index_to_user: list[int] = []
        self.index_to_movie: list[int] = []
        self.user_bias = np.empty(0, dtype=np.float32)
        self.movie_bias = np.empty(0, dtype=np.float32)
        self.user_factors = np.empty((0, self.config.factors), dtype=np.float32)
        self.movie_factors = np.empty((0, self.config.factors), dtype=np.float32)
        self.training_curve: list[dict] = []
        self.backend_used = "unfitted"
        self.device_used = "none"
        self._movie_index_lookup: np.ndarray | None = None
        # Optional batched candidate-score cache (see prime_candidate_score_cache).
        self._batch_score_user_row: dict[int, int] | None = None
        self._batch_score_col_lookup: np.ndarray | None = None
        self._batch_score_matrix: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return bool(self.user_to_index and self.movie_to_index)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            config=np.array(json.dumps(asdict(self.config), ensure_ascii=False)),
            global_mean=np.array(self.global_mean, dtype=np.float32),
            index_to_user=np.array(self.index_to_user, dtype=np.int64),
            index_to_movie=np.array(self.index_to_movie, dtype=np.int64),
            user_bias=self.user_bias.astype(np.float32, copy=False),
            movie_bias=self.movie_bias.astype(np.float32, copy=False),
            user_factors=self.user_factors.astype(np.float32, copy=False),
            movie_factors=self.movie_factors.astype(np.float32, copy=False),
            backend_used=np.array(self.backend_used),
            device_used=np.array(self.device_used),
            training_curve=np.array(json.dumps(self.training_curve, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: Path) -> "MatrixFactorizationModel":
        with np.load(path, allow_pickle=False) as payload:
            config = MatrixFactorizationConfig(**json.loads(str(payload["config"].item())))
            model = cls(config)
            model.global_mean = float(payload["global_mean"].item())
            model.index_to_user = [int(value) for value in payload["index_to_user"].tolist()]
            model.index_to_movie = [int(value) for value in payload["index_to_movie"].tolist()]
            model.user_to_index = {user_id: index for index, user_id in enumerate(model.index_to_user)}
            model.movie_to_index = {movie_id: index for index, movie_id in enumerate(model.index_to_movie)}
            model.user_bias = payload["user_bias"].astype(np.float32)
            model.movie_bias = payload["movie_bias"].astype(np.float32)
            model.user_factors = payload["user_factors"].astype(np.float32)
            model.movie_factors = payload["movie_factors"].astype(np.float32)
            model.backend_used = str(payload["backend_used"].item())
            model.device_used = str(payload["device_used"].item())
            model.training_curve = json.loads(str(payload["training_curve"].item()))
        model._movie_index_lookup = None
        model._batch_score_user_row = None
        model._batch_score_col_lookup = None
        model._batch_score_matrix = None
        return model

    def fit(self, ratings: Iterable[TrainingRating]) -> list[dict]:
        rows = list(ratings)
        if not rows:
            raise ValueError("matrix factorization needs at least one training rating")

        user_ids = sorted({row.user_id for row in rows})
        movie_ids = sorted({row.movie_id for row in rows})
        self.user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
        self.movie_to_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}
        self.index_to_user = user_ids
        self.index_to_movie = movie_ids
        self._movie_index_lookup = None
        self._batch_score_user_row = None
        self._batch_score_col_lookup = None
        self._batch_score_matrix = None

        user_indices = np.array([self.user_to_index[row.user_id] for row in rows], dtype=np.int32)
        movie_indices = np.array([self.movie_to_index[row.movie_id] for row in rows], dtype=np.int32)
        values = np.array([row.rating for row in rows], dtype=np.float32)
        self.global_mean = float(values.mean())

        torch_device = _resolve_torch_device(self.config)
        if torch_device is not None:
            return self._fit_torch(user_indices, movie_indices, values, len(user_ids), len(movie_ids), torch_device)
        return self._fit_numpy(user_indices, movie_indices, values, len(user_ids), len(movie_ids))

    def _fit_numpy(
        self,
        user_indices: np.ndarray,
        movie_indices: np.ndarray,
        values: np.ndarray,
        user_count: int,
        movie_count: int,
    ) -> list[dict]:
        self.backend_used = "numpy"
        self.device_used = "cpu"
        rng = np.random.default_rng(self.config.seed)
        scale = 0.08 / math.sqrt(max(self.config.factors, 1))
        self.user_bias = np.zeros(user_count, dtype=np.float32)
        self.movie_bias = np.zeros(movie_count, dtype=np.float32)
        self.user_factors = rng.normal(0.0, scale, size=(user_count, self.config.factors)).astype(np.float32)
        self.movie_factors = rng.normal(0.0, scale, size=(movie_count, self.config.factors)).astype(np.float32)

        self.training_curve = []
        order = np.arange(len(values))
        for epoch in range(1, self.config.epochs + 1):
            rng.shuffle(order)
            squared_error = 0.0
            for row_index in order:
                user_index = int(user_indices[row_index])
                movie_index = int(movie_indices[row_index])
                rating = float(values[row_index])

                prediction = self._predict_indices(user_index, movie_index, clamp=False)
                error = rating - prediction
                squared_error += error * error

                user_vector = self.user_factors[user_index].copy()
                movie_vector = self.movie_factors[movie_index].copy()
                self.user_bias[user_index] += self.config.learning_rate * (
                    error - self.config.regularization * self.user_bias[user_index]
                )
                self.movie_bias[movie_index] += self.config.learning_rate * (
                    error - self.config.regularization * self.movie_bias[movie_index]
                )
                self.user_factors[user_index] += self.config.learning_rate * (
                    error * movie_vector - self.config.regularization * user_vector
                )
                self.movie_factors[movie_index] += self.config.learning_rate * (
                    error * user_vector - self.config.regularization * movie_vector
                )

            self.training_curve.append(
                {
                    "epoch": epoch,
                    "train_rmse": round(math.sqrt(squared_error / len(values)), 6),
                    "training_ratings": int(len(values)),
                    "users": user_count,
                    "movies": movie_count,
                    "factors": self.config.factors,
                    "learning_rate": self.config.learning_rate,
                    "regularization": self.config.regularization,
                    "backend": self.backend_used,
                    "device": self.device_used,
                    "optimizer": "sgd",
                }
            )
        return list(self.training_curve)

    def _fit_torch(
        self,
        user_indices: np.ndarray,
        movie_indices: np.ndarray,
        values: np.ndarray,
        user_count: int,
        movie_count: int,
        device_name: str,
    ) -> list[dict]:
        import torch

        self.backend_used = "torch"
        self.device_used = device_name
        torch.manual_seed(self.config.seed)
        device = torch.device(device_name)
        scale = 0.08 / math.sqrt(max(self.config.factors, 1))

        user_tensor = torch.as_tensor(user_indices, dtype=torch.long, device=device)
        movie_tensor = torch.as_tensor(movie_indices, dtype=torch.long, device=device)
        rating_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
        user_bias = torch.zeros(user_count, dtype=torch.float32, device=device, requires_grad=True)
        movie_bias = torch.zeros(movie_count, dtype=torch.float32, device=device, requires_grad=True)
        user_factors = (torch.randn(user_count, self.config.factors, dtype=torch.float32, device=device) * scale).requires_grad_()
        movie_factors = (torch.randn(movie_count, self.config.factors, dtype=torch.float32, device=device) * scale).requires_grad_()
        parameters = [user_bias, movie_bias, user_factors, movie_factors]
        optimizer_name = self.config.optimizer.lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(parameters, lr=self.config.learning_rate)
        else:
            raise ValueError("optimizer must be 'adam' or 'sgd'")

        batch_size = max(1, int(self.config.batch_size))
        self.training_curve = []
        for epoch in range(1, self.config.epochs + 1):
            order = torch.randperm(len(values), device=device)
            squared_error = 0.0
            for start in range(0, len(values), batch_size):
                batch = order[start : start + batch_size]
                user_batch = user_tensor[batch]
                movie_batch = movie_tensor[batch]
                rating_batch = rating_tensor[batch]

                prediction = (
                    self.global_mean
                    + user_bias[user_batch]
                    + movie_bias[movie_batch]
                    + (user_factors[user_batch] * movie_factors[movie_batch]).sum(dim=1)
                )
                error = rating_batch - prediction
                mse = (error * error).mean()
                penalty = (
                    user_bias[user_batch].pow(2).mean()
                    + movie_bias[movie_batch].pow(2).mean()
                    + user_factors[user_batch].pow(2).mean()
                    + movie_factors[movie_batch].pow(2).mean()
                )
                loss = mse + self.config.regularization * penalty

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                squared_error += float((error.detach() * error.detach()).sum().cpu())

            self.training_curve.append(
                {
                    "epoch": epoch,
                    "train_rmse": round(math.sqrt(squared_error / len(values)), 6),
                    "training_ratings": int(len(values)),
                    "users": user_count,
                    "movies": movie_count,
                    "factors": self.config.factors,
                    "learning_rate": self.config.learning_rate,
                    "regularization": self.config.regularization,
                    "backend": self.backend_used,
                    "device": self.device_used,
                    "optimizer": optimizer_name,
                }
            )

        self.user_bias = user_bias.detach().cpu().numpy().astype(np.float32)
        self.movie_bias = movie_bias.detach().cpu().numpy().astype(np.float32)
        self.user_factors = user_factors.detach().cpu().numpy().astype(np.float32)
        self.movie_factors = movie_factors.detach().cpu().numpy().astype(np.float32)
        return list(self.training_curve)

    def predict(self, user_id: int, movie_id: int) -> float:
        user_index = self.user_to_index.get(int(user_id))
        movie_index = self.movie_to_index.get(int(movie_id))
        prediction = self.global_mean
        if user_index is not None:
            prediction += float(self.user_bias[user_index])
        if movie_index is not None:
            prediction += float(self.movie_bias[movie_index])
        if user_index is not None and movie_index is not None:
            prediction += float(self.user_factors[user_index] @ self.movie_factors[movie_index])
        return self._clamp(prediction)

    def _movie_index_array(self) -> np.ndarray:
        """Return a dense ``movie_id -> factor index`` lookup table (-1 if unknown).

        Replaces a per-call Python ``dict.get`` comprehension with a single
        vectorized fancy-index, which is numerically identical but far faster
        when scoring many candidates per user.
        """
        cached = self._movie_index_lookup
        if cached is not None:
            return cached
        max_movie_id = max(self.index_to_movie) if self.index_to_movie else -1
        lookup = np.full(max_movie_id + 1, -1, dtype=np.int32)
        for movie_id, index in self.movie_to_index.items():
            lookup[movie_id] = index
        self._movie_index_lookup = lookup
        return lookup

    def score_known_user(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        user_index = self.user_to_index.get(int(user_id))
        if user_index is None:
            return np.full(len(movie_ids), self.global_mean, dtype=np.float32)

        movie_id_array = np.asarray(movie_ids).astype(np.int64, copy=False)

        cached = self._score_from_batch_cache(int(user_id), movie_id_array)
        if cached is not None:
            return cached

        lookup = self._movie_index_array()
        movie_indices = np.full(len(movie_id_array), -1, dtype=np.int32)
        if len(lookup):
            in_range = (movie_id_array >= 0) & (movie_id_array < len(lookup))
            movie_indices[in_range] = lookup[movie_id_array[in_range]]
        scores = np.full(len(movie_id_array), self.global_mean + self.user_bias[user_index], dtype=np.float32)
        known_mask = movie_indices >= 0
        if known_mask.any():
            known_indices = movie_indices[known_mask]
            scores[known_mask] += self.movie_bias[known_indices]
            scores[known_mask] += self.movie_factors[known_indices] @ self.user_factors[user_index]
        return np.clip(scores, self.config.min_rating, self.config.max_rating)

    def _score_from_batch_cache(self, user_id: int, movie_id_array: np.ndarray) -> np.ndarray | None:
        """Serve candidate scores from the precomputed batch matrix when possible.

        Returns ``None`` (caller falls back to the direct computation) unless a
        cache is primed, the user is present, and *every* requested movie is in
        the cached candidate pool — guaranteeing identical coverage semantics.
        """
        matrix = self._batch_score_matrix
        if matrix is None or self._batch_score_user_row is None:
            return None
        row = self._batch_score_user_row.get(user_id)
        if row is None:
            return None
        lookup = self._batch_score_col_lookup
        cols = np.full(len(movie_id_array), -1, dtype=np.int64)
        in_range = (movie_id_array >= 0) & (movie_id_array < len(lookup))
        cols[in_range] = lookup[movie_id_array[in_range]]
        if np.any(cols < 0):
            return None
        return matrix[row, cols]

    def candidate_score_matrix(self, user_ids, movie_ids) -> np.ndarray:
        """Score ``len(user_ids) x len(movie_ids)`` candidates in one batched pass.

        Uses a single (optionally GPU) GEMM instead of one matmul per user. The
        bias/clip arithmetic matches :meth:`score_known_user`; floating-point
        results can differ at GEMM-reduction level (~1e-6, float32).
        """
        user_rows = np.array([self.user_to_index.get(int(u), -1) for u in user_ids], dtype=np.int64)
        movie_cols = np.array([self.movie_to_index.get(int(m), -1) for m in movie_ids], dtype=np.int64)

        backend = (os.environ.get("NETFLIX_MF_BATCH") or "auto").lower()
        use_gpu = backend == "gpu" or (backend == "auto" and self.device_used == "cuda")

        known_users = user_rows >= 0
        known_movies = movie_cols >= 0
        u_idx = np.where(known_users, user_rows, 0)
        m_idx = np.where(known_movies, movie_cols, 0)

        if use_gpu:
            try:
                import torch

                with torch.no_grad():
                    device = torch.device("cuda")
                    uf = torch.from_numpy(self.user_factors[u_idx]).to(device)
                    mf = torch.from_numpy(self.movie_factors[m_idx]).to(device)
                    ub = torch.from_numpy(self.user_bias[u_idx]).to(device)
                    mb = torch.from_numpy(self.movie_bias[m_idx]).to(device)
                    scores = self.global_mean + ub[:, None] + mb[None, :] + uf @ mf.t()
                    scores = torch.clamp(scores, self.config.min_rating, self.config.max_rating)
                    matrix = scores.float().cpu().numpy()
            except ModuleNotFoundError:
                use_gpu = False

        if not use_gpu:
            uf = self.user_factors[u_idx]
            mf = self.movie_factors[m_idx]
            matrix = (
                self.global_mean
                + self.user_bias[u_idx][:, None]
                + self.movie_bias[m_idx][None, :]
                + uf @ mf.T
            ).astype(np.float32)
            np.clip(matrix, self.config.min_rating, self.config.max_rating, out=matrix)

        if not known_users.all():
            matrix[~known_users, :] = self.global_mean
        if not known_movies.all():
            unknown_cols = ~known_movies
            base = (self.global_mean + self.user_bias[u_idx]).astype(np.float32)
            base = np.clip(base, self.config.min_rating, self.config.max_rating)
            matrix[:, unknown_cols] = base[:, None]
            matrix[~known_users, :] = self.global_mean
        return matrix

    def prime_candidate_score_cache(self, user_ids, movie_ids) -> None:
        """Precompute and cache scores for ``user_ids x movie_ids``.

        Subsequent :meth:`score_known_user` calls whose movies all fall inside
        ``movie_ids`` are served from this matrix (cheap column gather) instead
        of recomputing a matmul per user.
        """
        unique_users = list(dict.fromkeys(int(u) for u in user_ids))
        movie_id_array = np.asarray(movie_ids).astype(np.int64, copy=False)
        matrix = self.candidate_score_matrix(unique_users, movie_id_array)

        max_movie_id = int(movie_id_array.max()) if len(movie_id_array) else -1
        lookup = np.full(max_movie_id + 1, -1, dtype=np.int64)
        lookup[movie_id_array] = np.arange(len(movie_id_array), dtype=np.int64)

        self._batch_score_user_row = {user_id: row for row, user_id in enumerate(unique_users)}
        self._batch_score_col_lookup = lookup
        self._batch_score_matrix = matrix

    def _predict_indices(self, user_index: int, movie_index: int, *, clamp: bool) -> float:
        prediction = (
            self.global_mean
            + float(self.user_bias[user_index])
            + float(self.movie_bias[movie_index])
            + float(self.user_factors[user_index] @ self.movie_factors[movie_index])
        )
        if clamp:
            return self._clamp(prediction)
        return prediction

    def _clamp(self, value: float) -> float:
        return min(max(float(value), self.config.min_rating), self.config.max_rating)


def mf_batch_enabled() -> bool:
    """Whether batched MF candidate scoring is enabled via ``NETFLIX_MF_BATCH``.

    Values: ``auto`` (default; GPU when the model trained on CUDA, else batched
    CPU GEMM), ``gpu`` (CUDA GEMM), ``numpy`` (batched CPU GEMM), ``off``
    (per-user numpy, bit-identical to the original). The batched paths keep the
    final reranker output identical; only the standalone MF baseline can shift at
    GEMM-reduction level (~1e-4).
    """
    return (os.environ.get("NETFLIX_MF_BATCH") or "auto").lower() in {"auto", "gpu", "numpy"}


def _resolve_torch_device(config: MatrixFactorizationConfig) -> str | None:
    backend = config.backend.lower()
    if backend == "numpy":
        return None
    if backend not in {"auto", "torch"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")

    try:
        import torch
    except ModuleNotFoundError:
        if backend == "torch":
            raise
        return None

    requested_device = config.device.lower()
    if requested_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu" if backend == "torch" else None
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            if backend == "torch":
                raise RuntimeError("PyTorch is installed, but CUDA is not available")
            return None
        return "cuda"
    if requested_device == "cpu":
        return "cpu" if backend == "torch" else None
    raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
