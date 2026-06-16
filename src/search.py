from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def normalize(text: str) -> str:
    return text.strip().lower()


class MovieSearchEngine:
    def __init__(self, profiles: Iterable[dict]):
        self.profiles = list(profiles)
        self.by_id = {item["movieId"]: item for item in self.profiles}
        self.title_index: dict[str, list[dict]] = defaultdict(list)
        self.genre_index: dict[str, list[dict]] = defaultdict(list)
        self.tag_index: dict[str, list[dict]] = defaultdict(list)
        self._build_indexes()

    def _build_indexes(self) -> None:
        for item in self.profiles:
            title = normalize(item["title"])
            self.title_index[title].append(item)
            for token in title.replace("(", " ").replace(")", " ").replace(",", " ").split():
                self.title_index[normalize(token)].append(item)
            for genre in item["genres"]:
                self.genre_index[normalize(genre)].append(item)
            for tag in item["tags"]:
                self.tag_index[normalize(tag)].append(item)

    def linear_title_search(self, query: str) -> list[dict]:
        q = normalize(query)
        return [item for item in self.profiles if q in normalize(item["title"])]

    def index_title_search(self, query: str) -> list[dict]:
        q = normalize(query)
        exact = list(self.title_index.get(q, []))
        if exact:
            return _unique_by_movie_id(exact)
        matches = []
        for title, items in self.title_index.items():
            if q in title:
                matches.extend(items)
        return _unique_by_movie_id(matches)

    def linear_genre_search(self, genre: str) -> list[dict]:
        q = normalize(genre)
        return [item for item in self.profiles if q in {normalize(g) for g in item["genres"]}]

    def index_genre_search(self, genre: str) -> list[dict]:
        return list(self.genre_index.get(normalize(genre), []))

    def linear_tag_search(self, tag: str) -> list[dict]:
        q = normalize(tag)
        return [item for item in self.profiles if any(q in normalize(t) for t in item["tags"])]

    def index_tag_search(self, tag: str) -> list[dict]:
        q = normalize(tag)
        matches = []
        for indexed_tag, items in self.tag_index.items():
            if q in indexed_tag:
                matches.extend(items)
        return _unique_by_movie_id(matches)


def _unique_by_movie_id(items: Iterable[dict]) -> list[dict]:
    seen = set()
    result = []
    for item in items:
        movie_id = item["movieId"]
        if movie_id in seen:
            continue
        seen.add(movie_id)
        result.append(item)
    return result
