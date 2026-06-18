from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
import re
from typing import Iterable


def normalize(text: str) -> str:
    return text.strip().lower()


def tokenize(text: str) -> list[str]:
    normalized = normalize(text)
    return [part for part in re.sub(r"[^a-z0-9]+", " ", normalized).split() if part]


class MovieLensSearchEngine:
    def __init__(self, profiles: Iterable[dict]):
        self.profiles = list(profiles)
        self.by_id = {item["movieId"]: item for item in self.profiles}
        self.profile_order = {item["movieId"]: index for index, item in enumerate(self.profiles)}
        self.title_index: dict[str, list[dict]] = defaultdict(list)
        self.title_phrase_index: dict[str, list[dict]] = defaultdict(list)
        self.title_token_index: dict[str, list[dict]] = defaultdict(list)
        self.genre_index: dict[str, list[dict]] = defaultdict(list)
        self.tag_index: dict[str, list[dict]] = defaultdict(list)
        self.tag_token_index: dict[str, list[dict]] = defaultdict(list)
        self.title_ngram_index: dict[str, set[str]] = defaultdict(set)
        self.genre_ngram_index: dict[str, set[str]] = defaultdict(set)
        self.tag_ngram_index: dict[str, set[str]] = defaultdict(set)
        self._build_indexes()

    def _build_indexes(self) -> None:
        for item in self.profiles:
            title = normalize(item["title"])
            self.title_index[title].append(item)
            self.title_phrase_index[title].append(item)
            self._add_ngrams(self.title_ngram_index, title)
            for token in tokenize(title):
                self.title_index[token].append(item)
                self.title_token_index[token].append(item)
                self._add_ngrams(self.title_ngram_index, token)
            for genre in item["genres"]:
                key = normalize(genre)
                self.genre_index[key].append(item)
                self._add_ngrams(self.genre_ngram_index, key)
            for tag in item["tags"]:
                key = normalize(tag)
                self.tag_index[key].append(item)
                self._add_ngrams(self.tag_ngram_index, key)
                for token in tokenize(tag):
                    self.tag_token_index[token].append(item)
                    self._add_ngrams(self.tag_ngram_index, token)

    @staticmethod
    def _add_ngrams(index: dict[str, set[str]], key: str) -> None:
        for ngram in _ngrams(key):
            index[ngram].add(key)

    def linear_title_search(self, query: str) -> list[dict]:
        q = normalize(query)
        return [item for item in self.profiles if q in normalize(item["title"])]

    def index_title_search(self, query: str) -> list[dict]:
        q = normalize(query)
        if not q:
            return []

        exact = list(self.title_index.get(q, []))
        if exact:
            return _unique_by_movie_id(exact)

        token_matches = self._token_search(tokenize(q), self.title_token_index, self.title_ngram_index)
        if token_matches:
            return token_matches

        substring_matches = self._key_search(q, self.title_phrase_index, self.title_ngram_index)
        if substring_matches:
            return substring_matches

        return self._fuzzy_key_search(q, self.title_phrase_index, self.title_ngram_index)

    def linear_genre_search(self, genre: str) -> list[dict]:
        q = normalize(genre)
        return [item for item in self.profiles if q in {normalize(g) for g in item["genres"]}]

    def index_genre_search(self, genre: str) -> list[dict]:
        q = normalize(genre)
        exact = list(self.genre_index.get(q, []))
        if exact or not q:
            return exact
        return self._fuzzy_key_search(q, self.genre_index, self.genre_ngram_index)

    def linear_tag_search(self, tag: str) -> list[dict]:
        q = normalize(tag)
        return [item for item in self.profiles if any(q in normalize(t) for t in item["tags"])]

    def index_tag_search(self, tag: str) -> list[dict]:
        q = normalize(tag)
        if not q:
            return []

        substring_matches = self._key_search(q, self.tag_index, self.tag_ngram_index)
        if substring_matches:
            return substring_matches

        token_matches = self._token_search(tokenize(q), self.tag_token_index, self.tag_ngram_index)
        if token_matches:
            return token_matches

        return self._fuzzy_key_search(q, self.tag_index, self.tag_ngram_index)

    def _token_search(
        self,
        tokens: list[str],
        token_index: dict[str, list[dict]],
        ngram_index: dict[str, set[str]],
    ) -> list[dict]:
        if not tokens:
            return []

        postings = []
        for token in tokens:
            direct = list(token_index.get(token, []))
            if direct:
                postings.append(direct)
                continue

            fuzzy_keys = _fuzzy_keys(
                token,
                token_index,
                ngram_index,
                min_score=0.78,
                best_only=len(tokens) == 1,
            )
            fuzzy_items = []
            for key in fuzzy_keys:
                fuzzy_items.extend(token_index[key])
            if not fuzzy_items:
                return []
            postings.append(_unique_by_movie_id(fuzzy_items))

        return self._order_by_profile(_intersect_by_movie_id(postings))

    def _key_search(
        self,
        query: str,
        key_index: dict[str, list[dict]],
        ngram_index: dict[str, set[str]],
    ) -> list[dict]:
        matches = []
        for key in _candidate_keys(query, key_index, ngram_index):
            if query in key:
                matches.extend(key_index[key])
        return self._order_by_profile(_unique_by_movie_id(matches))

    def _fuzzy_key_search(
        self,
        query: str,
        key_index: dict[str, list[dict]],
        ngram_index: dict[str, set[str]],
    ) -> list[dict]:
        matches = []
        for key in _fuzzy_keys(query, key_index, ngram_index):
            matches.extend(key_index[key])
        return self._order_by_profile(_unique_by_movie_id(matches))

    def _order_by_profile(self, items: Iterable[dict]) -> list[dict]:
        return sorted(items, key=lambda item: self.profile_order[item["movieId"]])


def _ngrams(text: str, size: int = 2) -> set[str]:
    compact = normalize(text)
    if not compact:
        return set()
    if len(compact) <= size:
        return {compact}
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _candidate_keys(
    query: str,
    key_index: dict[str, list[dict]],
    ngram_index: dict[str, set[str]],
) -> set[str]:
    query_ngrams = _ngrams(query)
    if not query_ngrams:
        return set()
    if len(query) < 3:
        return set(key_index)

    candidates = set()
    for ngram in query_ngrams:
        candidates.update(ngram_index.get(ngram, set()))
    return {key for key in candidates if key in key_index}


def _fuzzy_keys(
    query: str,
    key_index: dict[str, list[dict]],
    ngram_index: dict[str, set[str]],
    min_score: float = 0.72,
    best_only: bool = True,
) -> list[str]:
    query_ngrams = _ngrams(query)
    scored_matches = []
    for key in _candidate_keys(query, key_index, ngram_index):
        key_ngrams = _ngrams(key)
        overlap = len(query_ngrams & key_ngrams)
        union = len(query_ngrams | key_ngrams) or 1
        ngram_score = overlap / union
        edit_score = SequenceMatcher(None, query, key).ratio()
        score = (ngram_score + edit_score) / 2
        if score >= min_score or edit_score >= min_score:
            scored_matches.append((score, key))
    if not scored_matches:
        return []
    if not best_only:
        return [key for _, key in scored_matches]

    best_score = max(score for score, _ in scored_matches)
    return [key for score, key in scored_matches if score >= best_score - 0.04]


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


def _intersect_by_movie_id(postings: list[list[dict]]) -> list[dict]:
    if not postings:
        return []

    common_ids = {item["movieId"] for item in postings[0]}
    for items in postings[1:]:
        common_ids &= {item["movieId"] for item in items}

    return [item for item in postings[0] if item["movieId"] in common_ids]
