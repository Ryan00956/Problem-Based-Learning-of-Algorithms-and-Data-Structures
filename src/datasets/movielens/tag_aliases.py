from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import combinations
import json
import math
from pathlib import Path
import re
import time
from typing import Iterable, Literal, Mapping

import pandas as pd

from src.datasets.movielens.tags import TAG_ALIASES, canonicalize_tag, normalize_tag_key


AliasDecision = Literal["accept", "reject", "ignore"]


class TagAliasDecisionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.decisions: list[dict] = self._load()

    def record(
        self,
        source: str,
        target: str,
        decision: AliasDecision,
        candidate: dict | None = None,
    ) -> dict:
        normalized_source = normalize_tag_key(source)
        normalized_target = normalize_tag_key(target)
        if not normalized_source or not normalized_target:
            raise ValueError("source and target are required")
        if normalized_source == normalized_target:
            raise ValueError("source and target must differ")

        item = {
            "source": normalized_source,
            "target": normalized_target,
            "decision": decision,
            "timestamp": time.time(),
        }
        if candidate:
            item["candidate"] = {
                key: candidate[key]
                for key in (
                    "confidence",
                    "confidence_band",
                    "source_count",
                    "target_count",
                    "movie_overlap",
                    "reasons",
                )
                if key in candidate
            }

        self.decisions = [
            existing
            for existing in self.decisions
            if _decision_key(existing.get("source"), existing.get("target")) != _decision_key(normalized_source, normalized_target)
        ]
        self.decisions.append(item)
        self._save()
        return item

    def accepted_aliases(self) -> dict[str, str]:
        return {
            item["source"]: item["target"]
            for item in self.decisions
            if item.get("decision") == "accept" and item.get("source") and item.get("target")
        }

    def summary(self) -> dict:
        counts = Counter(item.get("decision", "unknown") for item in self.decisions)
        return {
            "accepted_count": int(counts.get("accept", 0)),
            "rejected_count": int(counts.get("reject", 0)),
            "ignored_count": int(counts.get("ignore", 0)),
            "decision_count": len(self.decisions),
        }

    def latest_by_pair(self) -> dict[tuple[str, str], dict]:
        return {
            _decision_key(item.get("source"), item.get("target")): item
            for item in self.decisions
            if item.get("source") and item.get("target")
        }

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            raw_items = payload.get("decisions", [])
        elif isinstance(payload, list):
            raw_items = payload
        else:
            return []
        return [item for item in raw_items if isinstance(item, dict)]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"decisions": self.decisions}
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")


def build_tag_alias_report(
    tags: pd.DataFrame,
    limit: int = 24,
    min_confidence: float = 0.68,
    aliases: Mapping[str, str] | None = None,
    decisions: Iterable[dict] | None = None,
) -> dict:
    stats = _collect_tag_stats(tags, aliases=aliases)
    latest_decisions = {
        _decision_key(item.get("source"), item.get("target")): item
        for item in decisions or []
        if isinstance(item, dict)
    }
    candidates = _candidate_aliases(stats, min_confidence=min_confidence, decisions=latest_decisions)
    candidates = candidates[:limit]
    decision_counts = Counter(item.get("decision", "unknown") for item in latest_decisions.values())

    return {
        "summary": {
            "raw_tag_count": len(stats["raw_counts"]),
            "canonical_tag_count": len(stats["tag_counts"]),
            "configured_alias_count": len(TAG_ALIASES) + len(aliases or {}),
            "candidate_count": len(candidates),
            "high_confidence_count": sum(1 for item in candidates if item["confidence_band"] == "high"),
            "medium_confidence_count": sum(1 for item in candidates if item["confidence_band"] == "medium"),
            "accepted_count": int(decision_counts.get("accept", 0)),
            "rejected_count": int(decision_counts.get("reject", 0)),
            "ignored_count": int(decision_counts.get("ignore", 0)),
        },
        "configured_aliases": _configured_alias_samples(stats, aliases=aliases),
        "candidates": candidates,
    }


def _collect_tag_stats(tags: pd.DataFrame, aliases: Mapping[str, str] | None = None) -> dict:
    raw_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    tag_movies: dict[str, set[int]] = defaultdict(set)
    tag_raw_values: dict[str, Counter[str]] = defaultdict(Counter)
    movie_tags: dict[int, set[str]] = defaultdict(set)

    if tags.empty or not {"movieId", "tag"}.issubset(tags.columns):
        return {
            "raw_counts": raw_counts,
            "tag_counts": tag_counts,
            "tag_movies": tag_movies,
            "tag_raw_values": tag_raw_values,
            "movie_tags": movie_tags,
        }

    for row in tags[["movieId", "tag"]].itertuples(index=False):
        canonical = canonicalize_tag(row.tag, aliases)
        if not canonical:
            continue
        movie_id = int(row.movieId)
        raw = str(row.tag).strip()
        raw_counts[_normalize_alias_key(raw)] += 1
        tag_counts[canonical] += 1
        tag_movies[canonical].add(movie_id)
        tag_raw_values[canonical][raw] += 1
        movie_tags[movie_id].add(canonical)

    return {
        "raw_counts": raw_counts,
        "tag_counts": tag_counts,
        "tag_movies": tag_movies,
        "tag_raw_values": tag_raw_values,
        "movie_tags": movie_tags,
    }


def _configured_alias_samples(stats: dict, aliases: Mapping[str, str] | None = None) -> list[dict]:
    samples = []
    merged_aliases = {**TAG_ALIASES, **(aliases or {})}
    for source, target in sorted(merged_aliases.items()):
        source_key = normalize_tag_key(source)
        source_count = int(stats["raw_counts"].get(source_key, 0))
        target_count = int(stats["tag_counts"].get(target, 0))
        samples.append(
            {
                "source": source,
                "target": target,
                "source_count": source_count,
                "target_count": target_count,
                "active_in_dataset": source_count > 0,
                "source_type": "accepted" if aliases and normalize_tag_key(source) in aliases else "configured",
            }
        )
    return sorted(
        samples,
        key=lambda item: (
            -int(item["source_type"] == "accepted"),
            -int(item["active_in_dataset"]),
            -item["source_count"],
            item["source"],
        ),
    )[:18]


def _candidate_aliases(
    stats: dict,
    min_confidence: float,
    decisions: Mapping[tuple[str, str], dict],
) -> list[dict]:
    tag_counts: Counter[str] = stats["tag_counts"]
    tag_movies: dict[str, set[int]] = stats["tag_movies"]
    candidate_pairs = _text_candidate_pairs(tag_counts)
    candidate_pairs.update(_cooccurrence_candidate_pairs(stats["movie_tags"], tag_counts))

    candidates = []
    for left, right in candidate_pairs:
        if left == right:
            continue
        if _is_noisy_candidate_pair(left, right):
            continue
        metrics = _pair_metrics(left, right, tag_counts, tag_movies)
        if not _has_candidate_signal(metrics):
            continue
        source, target = _orient_pair(left, right, tag_counts, metrics)
        if _decision_key(source, target) in decisions:
            continue
        confidence = _confidence_score(metrics)
        if confidence < min_confidence:
            continue
        candidates.append(
            {
                "source": source,
                "target": target,
                "confidence": round(confidence, 4),
                "confidence_band": _confidence_band(confidence, metrics),
                "source_count": int(tag_counts[source]),
                "target_count": int(tag_counts[target]),
                "text_similarity": round(metrics["text_similarity"], 4),
                "token_similarity": round(metrics["token_similarity"], 4),
                "movie_overlap": metrics["movie_overlap"],
                "movie_jaccard": round(metrics["movie_jaccard"], 4),
                "movie_containment": round(metrics["movie_containment"], 4),
                "reasons": _candidate_reasons(metrics, source, target, tag_counts),
            }
        )

    return sorted(
        candidates,
        key=lambda item: (
            -item["confidence"],
            -item["movie_overlap"],
            -max(item["source_count"], item["target_count"]),
            item["target"],
            item["source"],
        ),
    )


def _text_candidate_pairs(tag_counts: Counter[str]) -> set[tuple[str, str]]:
    tags = list(tag_counts)
    pairs: set[tuple[str, str]] = set()
    compact_index: dict[str, list[str]] = defaultdict(list)
    token_index: dict[str, list[str]] = defaultdict(list)

    for tag in tags:
        compact_index[_compact_signature(tag)].append(tag)
        for token in _tokens(tag):
            if len(token) >= 4:
                token_index[_simple_singular(token)].append(tag)

    for bucket in compact_index.values():
        _add_bucket_pairs(pairs, bucket, max_bucket_size=60)
    for bucket in token_index.values():
        _add_bucket_pairs(pairs, bucket, max_bucket_size=120)

    return pairs


def _cooccurrence_candidate_pairs(movie_tags: dict[int, set[str]], tag_counts: Counter[str]) -> set[tuple[str, str]]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    for tags in movie_tags.values():
        useful_tags = sorted(tags, key=lambda tag: (-tag_counts[tag], tag))[:70]
        for left, right in combinations(useful_tags, 2):
            pair_counts[_ordered_pair(left, right)] += 1

    pairs = set()
    for pair, overlap in pair_counts.items():
        if overlap >= 2:
            pairs.add(pair)
    return pairs


def _pair_metrics(
    left: str,
    right: str,
    tag_counts: Counter[str],
    tag_movies: dict[str, set[int]],
) -> dict:
    left_movies = tag_movies.get(left, set())
    right_movies = tag_movies.get(right, set())
    overlap = len(left_movies & right_movies)
    union = len(left_movies | right_movies)
    min_movies = max(min(len(left_movies), len(right_movies)), 1)
    text_similarity = SequenceMatcher(None, left, right).ratio()
    compact_similarity = SequenceMatcher(None, _compact_signature(left), _compact_signature(right)).ratio()
    token_similarity = _token_similarity(left, right)

    return {
        "text_similarity": max(text_similarity, compact_similarity),
        "token_similarity": token_similarity,
        "movie_overlap": overlap,
        "movie_jaccard": overlap / union if union else 0.0,
        "movie_containment": overlap / min_movies,
        "support": min(tag_counts[left], tag_counts[right]),
    }


def _has_candidate_signal(metrics: dict) -> bool:
    text_signal = max(metrics["text_similarity"], metrics["token_similarity"]) >= 0.82
    cooccurrence_signal = metrics["movie_overlap"] >= 2 and (
        metrics["movie_jaccard"] >= 0.55 or metrics["movie_containment"] >= 0.8
    )
    return text_signal or cooccurrence_signal


def _confidence_score(metrics: dict) -> float:
    text_score = max(metrics["text_similarity"], metrics["token_similarity"] * 0.96)
    overlap_score = max(metrics["movie_jaccard"], metrics["movie_containment"] * 0.82)
    support_bonus = min(math.log1p(metrics["support"]) / math.log1p(12), 1.0) * 0.08
    overlap_bonus = min(metrics["movie_overlap"] / 8.0, 1.0) * 0.05
    return min(0.98, text_score * 0.58 + overlap_score * 0.32 + support_bonus + overlap_bonus)


def _orient_pair(left: str, right: str, tag_counts: Counter[str], metrics: dict) -> tuple[str, str]:
    if tag_counts[left] != tag_counts[right]:
        return (left, right) if tag_counts[left] < tag_counts[right] else (right, left)

    if len(left) != len(right):
        return (right, left) if len(left) < len(right) else (left, right)

    return (right, left) if right < left else (left, right)


def _candidate_reasons(metrics: dict, source: str, target: str, tag_counts: Counter[str]) -> list[str]:
    reasons = []
    if metrics["text_similarity"] >= 0.88:
        reasons.append("very similar spelling")
    elif metrics["text_similarity"] >= 0.8 or metrics["token_similarity"] >= 0.82:
        reasons.append("similar words")
    if metrics["movie_overlap"] >= 2:
        reasons.append(
            f"appears together on {metrics['movie_overlap']} movies"
        )
    if metrics["movie_containment"] >= 0.8 and metrics["movie_overlap"] >= 2:
        reasons.append("smaller tag is mostly contained in the larger one")
    if tag_counts[target] > tag_counts[source]:
        reasons.append("target has stronger dataset support")
    return reasons[:4]


def _confidence_band(confidence: float, metrics: dict) -> str:
    has_dataset_support = metrics["support"] >= 2 or metrics["movie_overlap"] >= 2
    if confidence >= 0.86 and has_dataset_support:
        return "high"
    if confidence >= 0.74:
        return "medium"
    return "low"


def _add_bucket_pairs(pairs: set[tuple[str, str]], bucket: list[str], max_bucket_size: int) -> None:
    if len(bucket) < 2 or len(bucket) > max_bucket_size:
        return
    for left, right in combinations(sorted(set(bucket)), 2):
        pairs.add(_ordered_pair(left, right))


def _ordered_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _decision_key(source: object, target: object) -> tuple[str, str]:
    return normalize_tag_key(source), normalize_tag_key(target)


def _token_similarity(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    jaccard = len(intersection) / len(union)
    if len(intersection) >= 2 or len(left_tokens) == len(right_tokens) == 1:
        containment = len(intersection) / min(len(left_tokens), len(right_tokens))
    else:
        containment = 0.0
    return max(jaccard, containment * 0.86)


def _is_noisy_candidate_pair(left: str, right: str) -> bool:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    rating_tokens = {"g", "pg", "pg13", "r", "nc17"}
    if left_tokens & rating_tokens or right_tokens & rating_tokens:
        return True

    for source, target in ((left_tokens, right_tokens), (right_tokens, left_tokens)):
        extra = source - target
        if target < source and extra <= {"ass", "s"}:
            return True
        if target < source and extra <= {"good", "great", "multiple", "some", "strong", "sustained", "stylized", "bloody"}:
            return True

    return False


def _tokens(value: str) -> list[str]:
    return [_simple_singular(token) for token in re.findall(r"[a-z0-9]+", value.lower())]


def _compact_signature(value: str) -> str:
    return "".join(_simple_singular(token) for token in _tokens(value))


def _simple_singular(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _normalize_alias_key(value: object) -> str:
    return normalize_tag_key(value)
