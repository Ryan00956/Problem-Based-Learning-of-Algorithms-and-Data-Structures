from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from urllib import request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "ml-latest-small"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _json_get(port: int, path: str) -> dict:
    with request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _json_post(port: int, path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


@unittest.skipUnless(
    all((DATA_DIR / name).exists() for name in ("movies.csv", "ratings.csv", "tags.csv")),
    "MovieLens small dataset is not available under data/ml-latest-small",
)
class MovieLensHttpSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.port = _free_port()
        cls.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "src.api",
                "--port",
                str(cls.port),
                "--dataset",
                "movielens",
            ],
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
                payload = _json_get(cls.port, "/api/health")
                if payload.get("ok"):
                    return
            except Exception as exc:  # pragma: no cover - only useful on startup failure.
                last_error = exc
                time.sleep(0.25)
        cls.process.terminate()
        try:
            _, stderr = cls.process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            _, stderr = cls.process.communicate(timeout=5)
        raise RuntimeError(f"API did not become ready: {last_error}\n{stderr}")

    def test_core_movielens_endpoints(self) -> None:
        dashboard = _json_get(self.port, "/api/dashboard")
        self.assertEqual(dashboard["summary"]["top_algorithm"], "top_n_heap")
        self.assertGreater(dashboard["summary"]["movie_count"], 0)

        top = _json_get(self.port, "/api/top?n=5&algorithm=heap")
        self.assertEqual(top["count"], 5)
        self.assertEqual(top["items"][0]["title"], "Shawshank Redemption, The (1994)")

        search = _json_get(self.port, "/api/search?kind=tag&query=mind-bending&n=5")
        self.assertGreaterEqual(search["count"], 1)
        self.assertLessEqual(len(search["items"]), 5)

        similar = _json_get(self.port, "/api/recommend?title=Toy%20Story&n=5")
        self.assertEqual(similar["target"]["title"], "Toy Story (1995)")
        self.assertGreaterEqual(similar["count"], 1)

        semantics = _json_get(self.port, "/api/tag-semantics?tag=mind-bending&n=5")
        self.assertEqual(semantics["summary"]["status"], "ready")
        self.assertGreaterEqual(semantics["count"], 1)

        event = _json_post(
            self.port,
            "/api/events",
            {
                "session_id": "unittest-http-smoke",
                "event_type": "search",
                "kind": "tag",
                "query": "mind-bending",
            },
        )
        self.assertTrue(event["ok"])

        for_you = _json_get(self.port, "/api/for-you?session_id=unittest-http-smoke&n=5")
        self.assertEqual(for_you["count"], 5)
        self.assertIn(for_you["status"], {"cold_start", "personalized"})


if __name__ == "__main__":
    unittest.main()
