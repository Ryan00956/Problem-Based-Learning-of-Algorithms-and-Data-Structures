from __future__ import annotations

from src.core.paths import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT
from src.datasets.movielens.loader import MOVIELENS_DIR, load_movielens
from src.datasets.movielens.profiles import build_movie_profiles, save_profiles_csv


__all__ = [
    "DATA_DIR",
    "MOVIELENS_DIR",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "build_movie_profiles",
    "load_movielens",
    "save_profiles_csv",
]
