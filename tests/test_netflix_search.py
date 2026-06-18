from __future__ import annotations

import unittest

from src.datasets.netflix.search import build_search_profiles, search_titles


def _scores() -> list[dict]:
    return [
        {
            "movie_id": 1,
            "title": "The Matrix",
            "release_year": 1999,
            "avg_rating": 4.5,
            "bayesian_rating": 4.4,
            "rating_count": 1000,
            "recent_rating_count": 10,
            "rating_score": 61.6,
            "popularity_score": 20.0,
            "freshness_score": 3.0,
            "comprehensive_score": 84.6,
        },
        {
            "movie_id": 2,
            "title": "Toy Story",
            "release_year": 1995,
            "avg_rating": 4.0,
            "bayesian_rating": 4.0,
            "rating_count": 900,
            "recent_rating_count": 8,
            "rating_score": 56.0,
            "popularity_score": 19.0,
            "freshness_score": 2.5,
            "comprehensive_score": 77.5,
        },
    ]


class NetflixSearchTest(unittest.TestCase):
    def test_search_profiles_are_compatible_with_index_engine(self) -> None:
        profiles = build_search_profiles(_scores())

        self.assertEqual(profiles[0]["movieId"], 1)
        self.assertEqual(profiles[0]["genres"], [])
        self.assertEqual(profiles[0]["tags"], [])
        self.assertEqual(profiles[0]["tag_count"], 0)

    def test_title_search_reuses_indexed_title_matching(self) -> None:
        rows = search_titles(_scores(), "title", "matrx", n=5)

        self.assertEqual([row["movieId"] for row in rows], [1])
        self.assertEqual(rows[0]["title"], "The Matrix")

    def test_rejects_genre_and_tag_search(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supports title"):
            search_titles(_scores(), "genre", "Comedy", n=5)


if __name__ == "__main__":
    unittest.main()
