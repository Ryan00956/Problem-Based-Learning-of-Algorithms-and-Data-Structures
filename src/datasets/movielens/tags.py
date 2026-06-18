from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Iterable, Mapping


TAG_ALIASES = {
    "sci fi": "sci-fi",
    "scifi": "sci-fi",
    "science fiction": "sci-fi",
    "science-fiction": "sci-fi",
    "sf": "sci-fi",
    "black comedy": "dark comedy",
    "black humor": "dark comedy",
    "black humour": "dark comedy",
    "dark humor": "dark comedy",
    "dark humour": "dark comedy",
    "time-travel": "time travel",
    "twist": "twist ending",
    "twists": "twist ending",
    "plot twist": "twist ending",
    "super hero": "superhero",
    "super-hero": "superhero",
    "comic book": "superhero",
    "comic-book": "superhero",
    "coming of age": "coming-of-age",
    "coming-of-age": "coming-of-age",
    "world war ii": "world war ii",
    "wwii": "world war ii",
    "ww2": "world war ii",
    "serial killers": "serial killer",
    "mindfuck": "mind-bending",
    "mind fuck": "mind-bending",
    "mind bending": "mind-bending",
    "feel good": "feel-good",
    "feelgood": "feel-good",
    "hitman": "hit men",
    "hitmen": "hit men",
    "nonlinear": "non-linear",
    "nonlinear narrative": "non-linear",
    "nonlinear storyline": "non-linear",
    "nonlinear timeline": "non-linear",
    "non-linear timeline": "non-linear",
    "disjointed timeline": "non-linear",
    "achronological": "non-linear",
    "out of order": "non-linear",
    "great dialogue": "good dialogue",
    "amazing dialogues": "good dialogue",
    "dialogue": "good dialogue",
    "smart writing": "good dialogue",
    "good music": "soundtrack",
    "great soundtrack": "soundtrack",
    "notable soundtrack": "soundtrack",
    "cult": "cult film",
    "cult classic": "cult film",
}

TAG_FACET_MULTIPLIERS = {
    "theme": 1.18,
    "style": 1.12,
    "franchise": 1.0,
    "person": 0.86,
    "meta": 0.66,
}

THEME_TERMS = {
    "aliens",
    "animation",
    "apocalypse",
    "assassin",
    "coming-of-age",
    "crime",
    "cyberpunk",
    "dystopia",
    "friendship",
    "future",
    "gangster",
    "heist",
    "hit men",
    "horror",
    "magic",
    "murder",
    "post-apocalyptic",
    "revenge",
    "robot",
    "serial killer",
    "space",
    "space action",
    "space opera",
    "superhero",
    "survival",
    "tarantino",
    "time travel",
    "vampire",
    "war",
    "world war ii",
    "zombie",
}

STYLE_TERMS = {
    "atmospheric",
    "beautiful",
    "bittersweet",
    "bleak",
    "clever",
    "dark",
    "dark comedy",
    "disturbing",
    "dreamlike",
    "emotional",
    "epic",
    "feel-good",
    "fun",
    "funny",
    "good dialogue",
    "gritty",
    "inspirational",
    "mind-bending",
    "quirky",
    "satire",
    "slow",
    "stylized",
    "suspense",
    "tense",
    "thought-provoking",
    "twist ending",
    "visually appealing",
}

META_TERMS = {
    "based on a book",
    "classic",
    "criterion",
    "cult film",
    "dvd",
    "imdb top 250",
    "oscar",
    "overrated",
    "seen more than once",
    "soundtrack",
    "underrated",
}

FRANCHISE_TERMS = {
    "batman",
    "disney",
    "harry potter",
    "lord of the rings",
    "marvel",
    "pixar",
    "star trek",
    "star wars",
    "tolkien",
}

PERSON_TERMS = {
    "alfred hitchcock",
    "brad pitt",
    "christopher nolan",
    "christina ricci",
    "christopher lloyd",
    "coen brothers",
    "bruce willis",
    "harvey keitel",
    "hayao miyazaki",
    "jim carrey",
    "john travolta",
    "leonardo dicaprio",
    "martin scorsese",
    "meryl streep",
    "quentin tarantino",
    "samuel l. jackson",
    "stanley kubrick",
    "steven spielberg",
    "steve buscemi",
    "tom hanks",
    "uma thurman",
}


def canonicalize_tag(value: object, aliases: Mapping[str, str] | None = None) -> str:
    text = str(value or "").strip().lower()
    if not text or "netflix" in text:
        return ""

    text = normalize_tag_key(text)
    alias_map = _merged_aliases(aliases)
    seen = set()
    while text in alias_map and text not in seen:
        seen.add(text)
        text = normalize_tag_key(alias_map[text])
    return text


def tag_facet(tag: str) -> str:
    if tag in THEME_TERMS:
        return "theme"
    if tag in STYLE_TERMS:
        return "style"
    if tag in FRANCHISE_TERMS:
        return "franchise"
    if tag in PERSON_TERMS:
        return "person"
    if tag in META_TERMS or _looks_like_meta(tag):
        return "meta"
    return "theme"


def tag_confidence(count: int) -> float:
    if count <= 0:
        return 0.0
    return round(0.52 + 0.48 * min(math.log1p(count) / math.log1p(5), 1.0), 4)


def tag_weight(tag: str, count: int) -> float:
    confidence = tag_confidence(count)
    facet = tag_facet(tag)
    multiplier = TAG_FACET_MULTIPLIERS.get(facet, 1.0)
    return round(math.log1p(max(count, 1)) * confidence * multiplier, 4)


def build_tag_details(values: Iterable[object], aliases: Mapping[str, str] | None = None) -> list[dict]:
    raw_by_canonical: dict[str, Counter[str]] = defaultdict(Counter)
    for value in values:
        canonical = canonicalize_tag(value, aliases)
        if not canonical:
            continue
        raw_by_canonical[canonical][str(value).strip()] += 1

    details = []
    for tag, raw_counts in raw_by_canonical.items():
        count = sum(raw_counts.values())
        facet = tag_facet(tag)
        details.append(
            {
                "tag": tag,
                "display": _display_tag(tag),
                "count": count,
                "facet": facet,
                "confidence": tag_confidence(count),
                "weight": tag_weight(tag, count),
                "aliases": sorted(raw_counts),
            }
        )

    return sorted(details, key=lambda item: (-item["weight"], item["tag"]))


def top_tag_names(tag_details: Iterable[dict], limit: int | None = None) -> list[str]:
    names = [str(item["tag"]) for item in tag_details]
    return names if limit is None else names[:limit]


def normalize_tag_key(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _merged_aliases(aliases: Mapping[str, str] | None) -> dict[str, str]:
    merged = {normalize_tag_key(source): normalize_tag_key(target) for source, target in TAG_ALIASES.items()}
    for source, target in (aliases or {}).items():
        normalized_source = normalize_tag_key(source)
        normalized_target = normalize_tag_key(target)
        if normalized_source and normalized_target and normalized_source != normalized_target:
            merged[normalized_source] = normalized_target
    return merged


def _display_tag(tag: str) -> str:
    if tag == "sci-fi":
        return "Sci-Fi"
    if tag == "world war ii":
        return "World War II"
    return tag.title()


def _looks_like_meta(tag: str) -> bool:
    return any(
        marker in tag
        for marker in (
            "adapted from",
            "based on",
            "criterion",
            "imdb",
            "oscar",
            "overrated",
            "seen",
            "soundtrack",
            "underrated",
        )
    )
