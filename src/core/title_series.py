from __future__ import annotations

from dataclasses import dataclass
import re
from difflib import SequenceMatcher


ARTICLES = {"a", "an", "the"}
SEQUENCE_WORDS = {
    "part",
    "pt",
    "episode",
    "ep",
    "volume",
    "vol",
    "chapter",
}
ROMAN_NUMERALS = {
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
}
KNOWN_SUBTITLE_SUFFIXES = {
    "reloaded",
    "revolutions",
    "resurrection",
    "resurrections",
    "revisited",
}
SUPPLEMENTAL_WORDS = {
    "bonus",
    "material",
    "extras",
    "extra",
    "behind",
    "scenes",
    "making",
    "documentary",
    "revisited",
}
WEAK_PREFIX_SUFFIXES = {
    "a",
    "an",
    "and",
    "for",
    "of",
    "or",
    "the",
    "to",
}


@dataclass(frozen=True)
class SeriesMatch:
    is_match: bool
    key: str
    title_similarity: float
    supplemental: bool = False


def series_key(title: str) -> str:
    tokens = _series_tokens(title)
    return " ".join(tokens)


def title_similarity(left: str, right: str) -> float:
    left_key = " ".join(_canonical_tokens(left))
    right_key = " ".join(_canonical_tokens(right))
    if not left_key or not right_key:
        return 0.0
    return SequenceMatcher(None, left_key, right_key).ratio()


def series_match(left_title: str, right_title: str) -> SeriesMatch:
    left_key = series_key(left_title)
    right_key = series_key(right_title)
    similarity = title_similarity(left_title, right_title)
    if left_key and left_key == right_key and _series_key_is_specific(left_key):
        return SeriesMatch(
            is_match=True,
            key=left_key,
            title_similarity=similarity,
            supplemental=is_supplemental_title(right_title),
        )

    left_tokens = _canonical_tokens(left_title)
    right_tokens = _canonical_tokens(right_title)
    prefix = _shared_prefix_key(left_tokens, right_tokens)
    if prefix:
        return SeriesMatch(
            is_match=True,
            key=prefix,
            title_similarity=similarity,
            supplemental=is_supplemental_title(right_title),
        )

    return SeriesMatch(is_match=False, key="", title_similarity=similarity)


def is_supplemental_title(title: str) -> bool:
    tokens = set(_canonical_tokens(title))
    return bool(tokens & SUPPLEMENTAL_WORDS)


def _series_tokens(title: str) -> list[str]:
    core = _move_trailing_article(_strip_year(title))
    first_segment = re.split(r"\s*[:;/]\s*", core, maxsplit=1)[0]
    tokens = _tokens(first_segment)
    tokens = _strip_leading_articles(tokens)
    tokens = _strip_trailing_sequence(tokens)
    tokens = _strip_known_subtitle_suffix(tokens)
    return tokens


def _canonical_tokens(title: str) -> list[str]:
    core = _move_trailing_article(_strip_year(title))
    return _strip_leading_articles(_tokens(core))


def _strip_year(title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title.strip())


def _move_trailing_article(title: str) -> str:
    match = re.match(r"^(.*),\s*(the|a|an)$", title.strip(), flags=re.IGNORECASE)
    if not match:
        return title.strip()
    return f"{match.group(2)} {match.group(1)}"


def _tokens(title: str) -> list[str]:
    normalized = title.lower().replace("&", " and ")
    return [token for token in re.sub(r"[^a-z0-9]+", " ", normalized).split() if token]


def _strip_leading_articles(tokens: list[str]) -> list[str]:
    while tokens and tokens[0] in ARTICLES:
        tokens = tokens[1:]
    return tokens


def _strip_trailing_sequence(tokens: list[str]) -> list[str]:
    values = list(tokens)
    while values:
        last = values[-1]
        if last.isdigit() or last in ROMAN_NUMERALS:
            values.pop()
            if values and values[-1] in SEQUENCE_WORDS:
                values.pop()
            continue
        if last in SEQUENCE_WORDS:
            values.pop()
            continue
        break
    return values


def _strip_known_subtitle_suffix(tokens: list[str]) -> list[str]:
    if len(tokens) > 1 and tokens[-1] in KNOWN_SUBTITLE_SUFFIXES:
        return tokens[:-1]
    return tokens


def _shared_prefix_key(left_tokens: list[str], right_tokens: list[str]) -> str:
    if not left_tokens or not right_tokens:
        return ""

    shared: list[str] = []
    for left, right in zip(left_tokens, right_tokens, strict=False):
        if left != right:
            break
        shared.append(left)

    while shared and shared[-1] in WEAK_PREFIX_SUFFIXES:
        shared.pop()

    if len(shared) >= 2:
        return " ".join(shared)
    if len(shared) == 1 and _single_token_prefix_is_specific(shared[0], left_tokens, right_tokens):
        return shared[0]
    return ""


def _single_token_prefix_is_specific(token: str, left_tokens: list[str], right_tokens: list[str]) -> bool:
    if len(token) < 5:
        return False
    shorter, longer = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    if len(shorter) != 1 or shorter[0] != token:
        return False
    remaining = longer[1:]
    return bool(remaining and (remaining[0] in KNOWN_SUBTITLE_SUFFIXES or remaining[0].isdigit() or remaining[0] in ROMAN_NUMERALS))


def _series_key_is_specific(key: str) -> bool:
    tokens = key.split()
    if len(tokens) >= 2:
        return True
    return bool(tokens and len(tokens[0]) >= 5)
