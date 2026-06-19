from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "netflix-prize" / "download"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "netflix-prize" / "netflix.duckdb"


def import_netflix_to_duckdb(
    raw_dir: Path = DEFAULT_RAW_DIR,
    db_path: Path = DEFAULT_DB_PATH,
    *,
    force: bool = False,
    threads: int | None = None,
    limit_movies: int | None = None,
) -> dict[str, int | str]:
    raw_dir = raw_dir.resolve()
    db_path = db_path.resolve()
    training_dir = raw_dir / "training_set"
    movie_titles_path = raw_dir / "movie_titles.txt"

    if not training_dir.exists():
        raise FileNotFoundError(f"missing Netflix training directory: {training_dir}")
    if not movie_titles_path.exists():
        raise FileNotFoundError(f"missing Netflix movie titles file: {movie_titles_path}")
    if db_path.exists() and not force:
        raise FileExistsError(f"DuckDB database already exists: {db_path}. Use --force to rebuild it.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(db_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    started = time.perf_counter()
    movie_rows = list(read_movie_titles(movie_titles_path))
    training_files = sorted(training_dir.glob("mv_*.txt"))
    if limit_movies is None:
        rating_source: str | list[str] = (training_dir / "mv_*.txt").as_posix()
    else:
        rating_source = [path.as_posix() for path in training_files[:limit_movies]]

    conn = duckdb.connect(str(tmp_path))
    try:
        if threads is not None:
            conn.execute("SET threads = ?", [threads])
        create_schema(conn)
        insert_movies(conn, movie_rows)
        import_ratings(conn, rating_source)
        create_stats(conn)
        record_metadata(
            conn,
            {
                "raw_dir": str(raw_dir),
                "rating_source": str(rating_source),
                "training_files": str(len(training_files)),
                "limit_movies": "" if limit_movies is None else str(limit_movies),
            },
        )
        summary = read_summary(conn)
    finally:
        conn.close()

    if db_path.exists():
        db_path.unlink()
    tmp_path.replace(db_path)
    elapsed_seconds = time.perf_counter() - started
    summary["db_path"] = str(db_path)
    summary["elapsed_seconds"] = f"{elapsed_seconds:.1f}"
    return summary


def read_movie_titles(path: Path) -> Iterator[tuple[int, int | None, str]]:
    with path.open("r", encoding="latin-1", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = line.rstrip("\n").split(",", 2)
            if len(row) != 3:
                raise ValueError(f"invalid movie_titles row at line {line_number}: {line!r}")
            movie_id_text, year_text, title = row
            release_year = None if year_text in {"", "NULL"} else int(year_text)
            yield int(movie_id_text), release_year, title


def create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE movies (
            movie_id INTEGER PRIMARY KEY,
            release_year INTEGER,
            title VARCHAR NOT NULL
        )
        """
    )


def insert_movies(conn: duckdb.DuckDBPyConnection, movie_rows: list[tuple[int, int | None, str]]) -> None:
    movie_frame = pd.DataFrame(movie_rows, columns=["movie_id", "release_year", "title"])
    conn.register("movie_rows", movie_frame)
    try:
        conn.execute(
            """
            INSERT INTO movies(movie_id, release_year, title)
            SELECT movie_id::INTEGER, release_year::INTEGER, title::VARCHAR
            FROM movie_rows
            """
        )
    finally:
        conn.unregister("movie_rows")


def import_ratings(conn: duckdb.DuckDBPyConnection, rating_source: str | list[str]) -> None:
    conn.execute(
        """
        CREATE TABLE ratings AS
        SELECT
            CAST(regexp_extract(filename, 'mv_(\\d+)\\.txt$', 1) AS INTEGER) AS movie_id,
            user_id::INTEGER AS user_id,
            rating::UTINYINT AS rating,
            rating_date::DATE AS rating_date
        FROM read_csv(
            ?,
            auto_detect = false,
            header = false,
            skip = 1,
            filename = true,
            parallel = true,
            dateformat = '%Y-%m-%d',
            columns = {
                'user_id': 'INTEGER',
                'rating': 'UTINYINT',
                'rating_date': 'DATE'
            }
        )
        """,
        [rating_source],
    )


def create_stats(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE movie_stats AS
        SELECT movie_id, COUNT(*)::INTEGER AS rating_count, AVG(rating)::DOUBLE AS rating_avg
        FROM ratings
        GROUP BY movie_id
        """
    )
    conn.execute(
        """
        CREATE TABLE user_stats AS
        SELECT user_id, COUNT(*)::INTEGER AS rating_count, AVG(rating)::DOUBLE AS rating_avg
        FROM ratings
        GROUP BY user_id
        """
    )
    conn.execute(
        """
        CREATE TABLE user_norms AS
        SELECT
            ratings.user_id,
            SQRT(SUM(POWER(ratings.rating::DOUBLE - user_stats.rating_avg::DOUBLE, 2))) AS rating_norm
        FROM ratings
        JOIN user_stats ON user_stats.user_id = ratings.user_id
        GROUP BY ratings.user_id
        """
    )


def record_metadata(conn: duckdb.DuckDBPyConnection, values: dict[str, str]) -> None:
    conn.execute("CREATE TABLE metadata (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL)")
    rows = [(key, value) for key, value in values.items()]
    conn.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", rows)


def read_summary(conn: duckdb.DuckDBPyConnection) -> dict[str, int | str]:
    row = conn.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM movies) AS movies,
            (SELECT COUNT(DISTINCT movie_id) FROM ratings) AS rated_movies,
            (SELECT COUNT(*) FROM user_stats) AS users,
            (SELECT COUNT(*) FROM ratings) AS ratings
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("failed to read DuckDB import summary")
    movies, rated_movies, users, ratings = row
    return {
        "movies": movies,
        "rated_movies": rated_movies,
        "users": users,
        "ratings": ratings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Netflix Prize raw files into DuckDB.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--force", action="store_true", help="Rebuild the database if it already exists.")
    parser.add_argument("--threads", type=int, default=None, help="Override DuckDB worker thread count.")
    parser.add_argument("--limit-movies", type=int, default=None, help="Import only the first N movie files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = import_netflix_to_duckdb(
        args.raw_dir,
        args.db_path,
        force=args.force,
        threads=args.threads,
        limit_movies=args.limit_movies,
    )
    print("Netflix DuckDB import complete:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
