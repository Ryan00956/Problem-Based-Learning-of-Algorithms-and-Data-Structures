from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from src.core.paths import DATA_DIR


MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_DIR = DATA_DIR / "ml-latest-small"
MOVIELENS_ZIP = DATA_DIR / "ml-latest-small.zip"
MOVIELENS_REQUIRED_FILES = ("movies.csv", "ratings.csv", "tags.csv")
NETFLIX_PRIZE_URL = os.environ.get(
    "NETFLIX_PRIZE_URL",
    "https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz",
)
NETFLIX_ARCHIVE = DATA_DIR / "nf_prize_dataset.tar.gz"
NETFLIX_CONFIRM_TIMEOUT_SECONDS = 8


def ensure_demo_data(dataset: str = "movielens", *, netflix_download: str = "ask") -> list[str]:
    messages: list[str] = []
    if dataset in {"movielens", "all"}:
        messages.extend(ensure_movielens_small())
    if dataset in {"netflix", "all"}:
        messages.extend(ensure_netflix_database(download_mode=netflix_download))
    return messages


def ensure_movielens_small() -> list[str]:
    if _movielens_ready():
        return [f"MovieLens small data ready: {MOVIELENS_DIR}"]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not MOVIELENS_ZIP.exists():
        print(f"Downloading MovieLens small data from {MOVIELENS_SMALL_URL}")
        _download_file(MOVIELENS_SMALL_URL, MOVIELENS_ZIP)
    else:
        print(f"Using cached MovieLens archive: {MOVIELENS_ZIP}")

    _extract_movielens_archive(MOVIELENS_ZIP, MOVIELENS_DIR)
    if not _movielens_ready():
        missing = ", ".join(name for name in MOVIELENS_REQUIRED_FILES if not (MOVIELENS_DIR / name).exists())
        raise RuntimeError(f"MovieLens download/extract did not create required files: {missing}")
    return [f"MovieLens small data downloaded and ready: {MOVIELENS_DIR}"]


def ensure_netflix_database(*, download_mode: str = "ask") -> list[str]:
    from src.datasets.netflix.import_duckdb import DEFAULT_DB_PATH, DEFAULT_RAW_DIR, import_netflix_to_duckdb

    if DEFAULT_DB_PATH.exists():
        return [f"Netflix DuckDB ready: {DEFAULT_DB_PATH}"]

    if not _netflix_raw_ready(DEFAULT_RAW_DIR):
        _prepare_netflix_raw_data(DEFAULT_RAW_DIR, download_mode=download_mode)

    print("Netflix raw data found. Building local DuckDB database; this can take a while...")
    summary = import_netflix_to_duckdb(force=False)
    return [
        "Netflix DuckDB built: "
        + ", ".join(f"{key}={value}" for key, value in summary.items())
    ]


def _movielens_ready() -> bool:
    return all((MOVIELENS_DIR / name).exists() for name in MOVIELENS_REQUIRED_FILES)


def _netflix_raw_ready(raw_dir: Path) -> bool:
    training_dir = raw_dir / "training_set"
    movie_titles_path = raw_dir / "movie_titles.txt"
    return movie_titles_path.exists() and any(training_dir.glob("mv_*.txt"))


def _prepare_netflix_raw_data(raw_dir: Path, *, download_mode: str) -> None:
    if download_mode not in {"ask", "never", "yes"}:
        raise ValueError(f"unknown Netflix download mode: {download_mode}")

    if download_mode == "never":
        _raise_netflix_missing(raw_dir)

    if download_mode == "ask" and not _confirm_netflix_download():
        _raise_netflix_missing(raw_dir)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if NETFLIX_ARCHIVE.exists():
        print(f"Using cached Netflix Prize archive: {NETFLIX_ARCHIVE}")
    else:
        print(f"Downloading Netflix Prize archive from {NETFLIX_PRIZE_URL}")
        print("This file is large and may take several minutes.")
        _download_file(NETFLIX_PRIZE_URL, NETFLIX_ARCHIVE)

    print("Extracting Netflix Prize archive; this can take a while...")
    _extract_netflix_archive(NETFLIX_ARCHIVE)
    if not _netflix_raw_ready(raw_dir):
        raise RuntimeError(
            "Netflix archive extraction finished, but required raw files were not found under "
            f"{raw_dir}."
        )


def _confirm_netflix_download() -> bool:
    print()
    print("Netflix Prize raw data is missing.")
    print(f"Download/extract the large Netflix Prize archive now? [y/N] ({NETFLIX_CONFIRM_TIMEOUT_SECONDS}s)")
    print("Press y then Enter to continue; anything else, or no answer, skips it.")

    answer = _read_answer_with_timeout(NETFLIX_CONFIRM_TIMEOUT_SECONDS)
    return answer.strip().lower() in {"y", "yes"}


def _read_answer_with_timeout(timeout_seconds: int) -> str:
    if not sys.stdin.isatty():
        print("No interactive terminal detected. Skipping Netflix download by default.")
        return ""

    if os.name == "nt":
        import msvcrt

        started = time.monotonic()
        chars: list[str] = []
        while time.monotonic() - started < timeout_seconds:
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                if char in {"\r", "\n"}:
                    print()
                    return "".join(chars)
                if char == "\003":
                    raise KeyboardInterrupt
                if char == "\b":
                    if chars:
                        chars.pop()
                        print("\b \b", end="", flush=True)
                    continue
                chars.append(char)
                print(char, end="", flush=True)
            time.sleep(0.05)
        print()
        return "".join(chars)

    import select

    ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    if not ready:
        print()
        return ""
    return sys.stdin.readline()


def _raise_netflix_missing(raw_dir: Path) -> None:
    raise FileNotFoundError(
        "Netflix Prize raw data was not found. To run the Netflix demo, either rerun and answer y "
        f"to download/extract it, or manually place movie_titles.txt and training_set/mv_*.txt under {raw_dir}."
    )


def _download_file(url: str, destination: Path) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with tmp_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        tmp_path.replace(destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _extract_movielens_archive(archive_path: Path, target_dir: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="movielens_", dir=DATA_DIR) as temp_name:
        temp_dir = Path(temp_name)
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract_zip(archive, temp_dir)

        extracted_dir = temp_dir / "ml-latest-small"
        if not extracted_dir.exists():
            raise RuntimeError(f"MovieLens archive did not contain ml-latest-small/: {archive_path}")

        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(extracted_dir), str(target_dir))


def _extract_netflix_archive(archive_path: Path) -> None:
    netflix_root = DATA_DIR / "netflix-prize"
    netflix_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as archive:
        _safe_extract_tar(archive, netflix_root, strip_prefix="download")

    raw_dir = netflix_root / "download"
    training_archive = raw_dir / "training_set.tar"
    training_dir = raw_dir / "training_set"
    if training_archive.exists() and not any(training_dir.glob("mv_*.txt")):
        with tarfile.open(training_archive, "r:") as archive:
            _safe_extract_tar(archive, raw_dir)


def _safe_extract_zip(archive: zipfile.ZipFile, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    for member in archive.infolist():
        resolved = (target_dir / member.filename).resolve()
        if target_root not in (resolved, *resolved.parents):
            raise RuntimeError(f"Refusing to extract unsafe zip member: {member.filename}")
    archive.extractall(target_dir)


def _safe_extract_tar(archive: tarfile.TarFile, target_dir: Path, *, strip_prefix: str | None = None) -> None:
    target_root = target_dir.resolve()
    members: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        target_name = member.name
        if strip_prefix and (member.name == strip_prefix or member.name.startswith(f"{strip_prefix}/")):
            target_name = str(Path("download") / Path(member.name).relative_to(strip_prefix))
        member.name = target_name
        resolved = (target_dir / member.name).resolve()
        if target_root not in (resolved, *resolved.parents):
            raise RuntimeError(f"Refusing to extract unsafe tar member: {member.name}")
        members.append(member)
    archive.extractall(target_dir, members=members)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download or build local demo data when needed.")
    parser.add_argument("--dataset", choices=["movielens", "netflix", "all"], default="movielens")
    parser.add_argument(
        "--netflix-download",
        choices=["ask", "never", "yes"],
        default="ask",
        help="What to do if Netflix raw data is missing. Default: ask and skip after a short timeout.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for message in ensure_demo_data(args.dataset, netflix_download=args.netflix_download):
        print(message)


if __name__ == "__main__":
    main()
