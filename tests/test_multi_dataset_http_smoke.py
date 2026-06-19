from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from urllib import error, parse, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOVIELENS_DATA_DIR = PROJECT_ROOT / "data" / "ml-latest-small"
NETFLIX_DB_PATH = PROJECT_ROOT / "data" / "netflix-prize" / "netflix.duckdb"
NETFLIX_SCORES_PATH = PROJECT_ROOT / "output" / "netflix" / "movie_scores.csv"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _json_get(port: int, path: str, params: dict | None = None) -> tuple[int, dict]:
    query = f"?{parse.urlencode(params)}" if params else ""
    try:
        with request.urlopen(f"http://127.0.0.1:{port}{path}{query}", timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


@unittest.skipUnless(
    all((MOVIELENS_DATA_DIR / name).exists() for name in ("movies.csv", "ratings.csv", "tags.csv"))
    and NETFLIX_DB_PATH.exists()
    and NETFLIX_SCORES_PATH.exists(),
    "MovieLens and Netflix local datasets are required for the multi-dataset HTTP smoke test",
)
class MultiDatasetHttpSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.port = _free_port()
        cls.process = subprocess.Popen(
            [sys.executable, "-m", "src.api", "--port", str(cls.port)],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        cls._wait_until_ready()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.process.terminate()
        try:
            cls.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait(timeout=10)

    @classmethod
    def _wait_until_ready(cls) -> None:
        last_error: Exception | None = None
        for _ in range(40):
            if cls.process.poll() is not None:
                stderr = cls.process.stderr.read() if cls.process.stderr else ""
                raise RuntimeError(f"API process exited early: {stderr}")
            try:
                status, payload = _json_get(cls.port, "/api/health", {"dataset": "movielens"})
                if status == 200 and payload.get("ok"):
                    return
            except Exception as exc:  # pragma: no cover - only useful on startup failure.
                last_error = exc
                time.sleep(0.25)
        raise RuntimeError(f"API did not become ready: {last_error}")

    def test_one_port_selects_movielens_and_netflix_by_query_param(self) -> None:
        status, movielens = _json_get(self.port, "/api/dashboard", {"dataset": "movielens"})
        self.assertEqual(status, 200)
        self.assertEqual(movielens["dataset"]["name"], "movielens")
        self.assertGreater(movielens["summary"]["tag_count"], 0)

        status, netflix = _json_get(self.port, "/api/dashboard", {"dataset": "netflix"})
        self.assertEqual(status, 200)
        self.assertEqual(netflix["dataset"]["name"], "netflix")
        self.assertEqual(netflix["summary"]["tag_count"], 0)

        status, search = _json_get(
            self.port,
            "/api/search",
            {"dataset": "netflix", "kind": "title", "query": "Matrix", "n": 5},
        )
        self.assertEqual(status, 200)
        self.assertEqual(search["items"][0]["title"], "The Matrix")

        status, tag_search = _json_get(
            self.port,
            "/api/search",
            {"dataset": "netflix", "kind": "tag", "query": "funny", "n": 5},
        )
        self.assertEqual(status, 400)
        self.assertIn("only supports title", tag_search["detail"])


if __name__ == "__main__":
    unittest.main()
