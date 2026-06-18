from __future__ import annotations

import heapq
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")


def _key_function(key: str | Callable[[T], object]) -> Callable[[T], object]:
    if callable(key):
        return key
    return lambda item: item[key]  # type: ignore[index]


def merge_sort(items: Iterable[T], key: str | Callable[[T], object], reverse: bool = True) -> list[T]:
    values = list(items)
    key_func = _key_function(key)

    def comes_before(left: T, right: T) -> bool:
        if reverse:
            return key_func(left) >= key_func(right)
        return key_func(left) <= key_func(right)

    def sort_part(part: list[T]) -> list[T]:
        if len(part) <= 1:
            return part
        mid = len(part) // 2
        left = sort_part(part[:mid])
        right = sort_part(part[mid:])
        merged: list[T] = []
        i = j = 0
        while i < len(left) and j < len(right):
            if comes_before(left[i], right[j]):
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged

    return sort_part(values)


def heap_sort(items: Iterable[T], key: str | Callable[[T], object], reverse: bool = True) -> list[T]:
    values = list(items)
    key_func = _key_function(key)

    def heap_priority(left: T, right: T) -> bool:
        if reverse:
            return key_func(left) < key_func(right)
        return key_func(left) > key_func(right)

    def sift_down(index: int, size: int) -> None:
        while True:
            left = index * 2 + 1
            right = left + 1
            best = index
            if left < size and heap_priority(values[left], values[best]):
                best = left
            if right < size and heap_priority(values[right], values[best]):
                best = right
            if best == index:
                break
            values[index], values[best] = values[best], values[index]
            index = best

    size = len(values)
    for index in range(size // 2 - 1, -1, -1):
        sift_down(index, size)

    for end in range(size - 1, 0, -1):
        values[0], values[end] = values[end], values[0]
        sift_down(0, end)

    return values


def top_n_heap(items: Iterable[T], n: int, key: str | Callable[[T], object], reverse: bool = True) -> list[T]:
    if n <= 0:
        return []

    key_func = _key_function(key)
    if reverse:
        return heapq.nlargest(n, items, key=key_func)
    return heapq.nsmallest(n, items, key=key_func)
