from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Iterable


def normalize(text: str) -> str:
    return text.strip().lower()


def tokenize(text: str) -> list[str]:
    normalized = normalize(text)
    return [part for part in re.sub(r"[^a-z0-9]+", " ", normalized).split() if part]


@dataclass(frozen=True)
class SearchMatch:
    item: dict
    field: str
    match_type: str
    similarity: float
    matched_token_count: int = 1
    query_token_count: int = 1


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
        self._sort_indexes()

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

    def _sort_indexes(self) -> None:
        for index in (
            self.title_index,
            self.title_phrase_index,
            self.title_token_index,
            self.genre_index,
            self.tag_index,
            self.tag_token_index,
        ):
            for items in index.values():
                items.sort(key=self._quality_order)

    def _quality_order(self, item: dict) -> tuple[float, int]:
        return (-float(item.get("comprehensive_score", 0.0)), self.profile_order[item["movieId"]])

    def linear_title_search(self, query: str) -> list[dict]:
        q = normalize(query)
        return [item for item in self.profiles if q in normalize(item["title"])]

    def index_title_search(self, query: str) -> list[dict]:
        q = normalize(query)
        if not q:
            return []

        exact_phrase = list(self.title_phrase_index.get(q, []))
        if exact_phrase:
            return self._rank_items(exact_phrase, "title", "exact_phrase")

        token_matches = self._token_search(tokenize(q), self.title_token_index, self.title_ngram_index, "title")
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
        return self._fuzzy_key_search(q, self.genre_index, self.genre_ngram_index, "genre")

    def linear_tag_search(self, tag: str) -> list[dict]:
        q = normalize(tag)
        return [item for item in self.profiles if any(q in normalize(t) for t in item["tags"])]

    def index_tag_search(self, tag: str) -> list[dict]:
        q = normalize(tag)
        if not q:
            return []

        substring_matches = self._key_search(q, self.tag_index, self.tag_ngram_index, "tag")
        if substring_matches:
            return substring_matches

        token_matches = self._token_search(tokenize(q), self.tag_token_index, self.tag_ngram_index, "tag")
        if token_matches:
            return token_matches

        return self._fuzzy_key_search(q, self.tag_index, self.tag_ngram_index, "tag")

    def _token_search(
        self,
        tokens: list[str],
        token_index: dict[str, list[dict]],
        ngram_index: dict[str, set[str]],
        field: str,
    ) -> list[dict]:
        if not tokens:
            return []

        postings = []
        for token in tokens:
            direct = list(token_index.get(token, []))
            if direct:
                postings.append({item["movieId"]: (item, 1.0) for item in direct})
                continue

            fuzzy_keys = _fuzzy_key_scores(
                token,
                token_index,
                ngram_index,
                min_score=0.78,
                best_only=len(tokens) == 1,
            )
            fuzzy_items = {}
            for score, key in fuzzy_keys:
                for item in token_index[key]:
                    movie_id = item["movieId"]
                    current = fuzzy_items.get(movie_id)
                    if current is None or score > current[1]:
                        fuzzy_items[movie_id] = (item, score)
            if not fuzzy_items:
                return []
            postings.append(fuzzy_items)

        common_ids = set(postings[0])
        for posting in postings[1:]:
            common_ids &= set(posting)

        matches = []
        for movie_id in common_ids:
            item = postings[0][movie_id][0]
            scores = [posting[movie_id][1] for posting in postings]
            match_type = "exact_token" if all(score == 1.0 for score in scores) else "fuzzy_token"
            matches.append(
                SearchMatch(
                    item=item,
                    field=field,
                    match_type=match_type,
                    similarity=sum(scores) / len(scores),
                    matched_token_count=len(tokens),
                    query_token_count=len(tokens),
                )
            )
        return self._rank_matches(matches)

    def _key_search(
        self,
        query: str,
        key_index: dict[str, list[dict]],
        ngram_index: dict[str, set[str]],
        field: str = "title",
    ) -> list[dict]:
        matches = []
        for key in _candidate_keys(query, key_index, ngram_index):
            if query in key:
                match_type = "exact_key" if query == key else "substring"
                similarity = 1.0 if query == key else SequenceMatcher(None, query, key).ratio()
                for item in key_index[key]:
                    matches.append(SearchMatch(item, field, match_type, similarity))
        return self._rank_matches(matches)

    def _fuzzy_key_search(
        self,
        query: str,
        key_index: dict[str, list[dict]],
        ngram_index: dict[str, set[str]],
        field: str = "title",
    ) -> list[dict]:
        matches = []
        for score, key in _fuzzy_key_scores(query, key_index, ngram_index):
            for item in key_index[key]:
                matches.append(SearchMatch(item, field, "fuzzy_key", score))
        return self._rank_matches(matches)

    def _rank_items(
        self,
        items: Iterable[dict],
        field: str,
        match_type: str,
        similarity: float = 1.0,
    ) -> list[dict]:
        return self._rank_matches([SearchMatch(item, field, match_type, similarity) for item in items])

    def _rank_matches(self, matches: Iterable[SearchMatch]) -> list[dict]:
        best_by_id: dict[int, SearchMatch] = {}
        for match in matches:
            movie_id = match.item["movieId"]
            current = best_by_id.get(movie_id)
            if current is None or self._rank_tuple(match) > self._rank_tuple(current):
                best_by_id[movie_id] = match

        ranked = sorted(best_by_id.values(), key=self._rank_tuple, reverse=True)
        return [match.item for match in ranked]

    def _rank_tuple(self, match: SearchMatch) -> tuple[float, float, float, float, float, int]:
        type_priority = {
            "exact_phrase": 700.0,
            "exact_key": 620.0,
            "exact_token": 560.0,
            "substring": 440.0,
            "fuzzy_token": 360.0,
            "fuzzy_key": 300.0,
        }.get(match.match_type, 0.0)
        field_priority = {"title": 30.0, "genre": 20.0, "tag": 10.0}.get(match.field, 0.0)
        coverage = match.matched_token_count / max(match.query_token_count, 1)
        quality = float(match.item.get("comprehensive_score", 0.0))
        return (
            type_priority,
            field_priority,
            match.similarity,
            coverage,
            quality,
            -self.profile_order[match.item["movieId"]],
        )


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


def _fuzzy_key_scores(
    query: str,
    key_index: dict[str, list[dict]],
    ngram_index: dict[str, set[str]],
    min_score: float = 0.72,
    best_only: bool = True,
) -> list[tuple[float, str]]:
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
        return scored_matches

    best_score = max(score for score, _ in scored_matches)
    return [(score, key) for score, key in scored_matches if score >= best_score - 0.04]

