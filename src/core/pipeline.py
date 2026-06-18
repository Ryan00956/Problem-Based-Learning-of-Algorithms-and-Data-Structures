from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class DatasetPipeline:
    name: str
    display_name: str
    load_profiles: Callable[[], list[dict]]
    print_movies: Callable[[list[dict], int], None]
    run_demo: Callable[[int], None]
    show_top: Callable[[int, str], list[dict]]
    search: Callable[[str, str, int], list[dict]]
    recommend: Callable[[str, int], tuple[dict | None, list[dict]]]
    run_experiments: Callable[[Path], dict[str, Path]]
    export_frontend_data: Callable[[], dict[str, Path]]
