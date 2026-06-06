from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KVCacheConfig:
    """Static KV-cache shape for one decoder-only model."""

    name: str = "default-32l-32kvh-128d-fp16"
    layers: int = 32
    kv_heads: int = 32
    head_dim: int = 128
    bytes_per_element: int = 2

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.bytes_per_element

    def request_bytes(self, prompt_tokens: int, output_tokens: int) -> int:
        return (prompt_tokens + output_tokens) * self.bytes_per_token


@dataclass(frozen=True)
class KVCachePoint:
    time_ms: float
    active_bytes: int
    delta_bytes: int
    events: int

    @property
    def active_mib(self) -> float:
        return bytes_to_mib(self.active_bytes)


def load_kv_cache_config(path: str | Path) -> KVCacheConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model config must contain a JSON object")

    name = payload.get("name", Path(path).stem)
    if not isinstance(name, str) or not name:
        raise ValueError("model config name must be a non-empty string")

    return KVCacheConfig(
        name=name,
        layers=_positive_int(payload, "layers"),
        kv_heads=_positive_int(payload, "kv_heads"),
        head_dim=_positive_int(payload, "head_dim"),
        bytes_per_element=_positive_int(payload, "bytes_per_element"),
    )


def summarize_kv_cache_timeline(
    points: list[KVCachePoint],
    pressure_fraction_of_peak: float = 0.80,
) -> dict[str, float]:
    if not points:
        return {
            "peak_active_kv_cache_mib": 0.0,
            "p95_active_kv_cache_mib": 0.0,
            "memory_pressure_duration_ms": 0.0,
        }

    peak_bytes = max(point.active_bytes for point in points)
    pressure_threshold = peak_bytes * pressure_fraction_of_peak
    return {
        "peak_active_kv_cache_mib": bytes_to_mib(peak_bytes),
        "p95_active_kv_cache_mib": bytes_to_mib(_weighted_percentile_bytes(points, 95)),
        "memory_pressure_duration_ms": _duration_at_or_above(points, pressure_threshold),
    }


def bytes_to_mib(value: int | float) -> float:
    return value / (1024 * 1024)


def _positive_int(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"model config {field} must be a positive integer")
    return value


def _duration_at_or_above(points: list[KVCachePoint], threshold_bytes: float) -> float:
    duration = 0.0
    for current, next_point in zip(points, points[1:]):
        interval_ms = next_point.time_ms - current.time_ms
        if interval_ms > 0 and current.active_bytes >= threshold_bytes:
            duration += interval_ms
    return duration


def _weighted_percentile_bytes(points: list[KVCachePoint], percentile: int) -> int:
    intervals: list[tuple[float, int]] = []
    for current, next_point in zip(points, points[1:]):
        duration_ms = next_point.time_ms - current.time_ms
        if duration_ms > 0:
            intervals.append((duration_ms, current.active_bytes))

    if not intervals:
        return points[-1].active_bytes

    total_duration = sum(duration for duration, _ in intervals)
    target = (percentile / 100) * total_duration
    seen = 0.0
    for duration, active_bytes in sorted(intervals, key=lambda item: item[1]):
        seen += duration
        if seen >= target:
            return active_bytes

    return max(active_bytes for _, active_bytes in intervals)
