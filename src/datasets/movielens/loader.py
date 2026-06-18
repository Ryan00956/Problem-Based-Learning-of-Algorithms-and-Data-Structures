from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.paths import DATA_DIR


MOVIELENS_DIR = DATA_DIR / "ml-latest-small"


def load_movielens(data_dir: Path = MOVIELENS_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(data_dir / "movies.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    tags = pd.read_csv(data_dir / "tags.csv")
    return movies, ratings, tags
