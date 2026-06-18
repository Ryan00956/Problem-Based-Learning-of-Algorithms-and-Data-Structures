from __future__ import annotations

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
    heap: list[T] = []
    key_func = _key_function(key)

    def higher_priority(left: T, right: T) -> bool:
        if reverse:
            return key_func(left) > key_func(right)
        return key_func(left) < key_func(right)

    def sift_up(index: int) -> None:
        while index > 0:
            parent = (index - 1) // 2
            if not higher_priority(heap[index], heap[parent]):
                break
            heap[index], heap[parent] = heap[parent], heap[index]
            index = parent

    def sift_down(index: int) -> None:
        size = len(heap)
        while True:
            left = index * 2 + 1
            right = left + 1
            best = index
            if left < size and higher_priority(heap[left], heap[best]):
                best = left
            if right < size and higher_priority(heap[right], heap[best]):
                best = right
            if best == index:
                break
            heap[index], heap[best] = heap[best], heap[index]
            index = best

    def push(item: T) -> None:
        heap.append(item)
        sift_up(len(heap) - 1)

    def pop() -> T:
        root = heap[0]
        last = heap.pop()
        if heap:
            heap[0] = last
            sift_down(0)
        return root

    for item in items:
        push(item)
    result: list[T] = []
    while heap:
        result.append(pop())
    return result
