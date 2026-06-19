from __future__ import annotations

import unittest

from src.core.title_series import is_supplemental_title, series_key, series_match
from src.datasets.movielens.recommendation import recommend_similar_movies
from src.datasets.movielens.search import MovieLensSearchEngine


class TitleSeriesTest(unittest.TestCase):
    def test_matrix_titles_share_series_key_across_dataset_formats(self) -> None:
        self.assertEqual(series_key("Matrix, The (1999)"), "matrix")
        self.assertEqual(series_key("The Matrix: Reloaded"), "matrix")
        self.assertEqual(series_key("Matrix Revolutions, The (2003)"), "matrix")

        match = series_match("Matrix, The (1999)", "Matrix Reloaded, The (2003)")

        self.assertTrue(match.is_match)
        self.assertEqual(match.key, "matrix")

    def test_nearby_non_series_title_is_not_a_match(self) -> None:
        self.assertFalse(series_match("The Matrix", "Sexual Matrix").is_match)
        self.assertFalse(
            series_match(
                "Lord of the Rings: The Return of the King: Extended Edition",
                "Lord of the Flies",
            ).is_match
        )

    def test_supplemental_titles_are_detected(self) -> None:
        self.assertTrue(is_supplemental_title("The Matrix: Reloaded: Bonus Material"))

    def test_long_colon_titles_match_their_series(self) -> None:
        match = series_match(
            "Lord of the Rings: The Return of the King: Extended Edition",
            "Lord of the Rings: The Two Towers",
        )

        self.assertTrue(match.is_match)
        self.assertEqual(match.key, "lord of the rings")

    def test_movielens_similar_recommendation_promotes_same_series_without_tags(self) -> None:
        profiles = [
            {
                "movieId": 1,
                "title": "Matrix, The (1999)",
                "genres": ["Action", "Sci-Fi", "Thriller"],
                "tags": ["philosophy", "sci-fi"],
                "comprehensive_score": 84.0,
            },
            {
                "movieId": 2,
                "title": "Matrix Reloaded, The (2003)",
                "genres": ["Action", "Adventure", "Sci-Fi", "Thriller"],
                "tags": [],
                "comprehensive_score": 66.0,
            },
            {
                "movieId": 3,
                "title": "Matrix Revolutions, The (2003)",
                "genres": ["Action", "Adventure", "Sci-Fi", "Thriller"],
                "tags": [],
                "comprehensive_score": 63.0,
            },
            {
                "movieId": 4,
                "title": "Inception (2010)",
                "genres": ["Action", "Sci-Fi", "Thriller"],
                "tags": ["philosophy", "sci-fi"],
                "comprehensive_score": 90.0,
            },
        ]
        engine = MovieLensSearchEngine(profiles)

        target, rows = recommend_similar_movies("Matrix", profiles, engine, n=3)

        self.assertIsNotNone(target)
        self.assertEqual([row["movieId"] for row in rows[:2]], [2, 3])
        self.assertTrue(rows[0]["series_match"])
        self.assertEqual(rows[0]["series_key"], "matrix")


if __name__ == "__main__":
    unittest.main()
