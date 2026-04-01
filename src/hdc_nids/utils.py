"""General utilities."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


def chunked(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def batched_iterable(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def stable_seed(*parts: Any) -> int:
    payload = "||".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0


def json_dump(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    if is_dataclass(payload):
        payload = asdict(payload)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def is_floatlike(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def nan_to_num(value: float, replacement: float) -> float:
    if math.isnan(value):
        return replacement
    return value
