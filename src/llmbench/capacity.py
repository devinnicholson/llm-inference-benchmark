from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmbench.kv_cache import KVCachePoint


@dataclass(frozen=True)
class ServingCapacityConfig:
    name: str
    total_memory_mib: float
    model_weights_mib: float
    runtime_reserved_mib: float
    kv_cache_budget_mib: float | None = None

    @property
    def effective_kv_cache_budget_mib(self) -> float:
        if self.kv_cache_budget_mib is not None:
            return self.kv_cache_budget_mib
        return self.total_memory_mib - self.model_weights_mib - self.runtime_reserved_mib

    @property
    def kv_cache_budget_bytes(self) -> int:
        return int(self.effective_kv_cache_budget_mib * 1024 * 1024)


def load_capacity_config(path: str | Path) -> ServingCapacityConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("capacity config must contain a JSON object")

    name = payload.get("name", Path(path).stem)
    if not isinstance(name, str) or not name:
        raise ValueError("capacity config name must be a non-empty string")

    config = ServingCapacityConfig(
        name=name,
        total_memory_mib=_positive_number(payload, "total_memory_mib"),
        model_weights_mib=_non_negative_number(payload, "model_weights_mib"),
        runtime_reserved_mib=_non_negative_number(payload, "runtime_reserved_mib"),
        kv_cache_budget_mib=_optional_positive_number(payload, "kv_cache_budget_mib"),
    )
    if config.effective_kv_cache_budget_mib <= 0:
        raise ValueError("effective KV-cache budget must be positive")
    return config


def summarize_capacity_timeline(
    points: list[KVCachePoint],
    capacity: ServingCapacityConfig,
    pressure_fraction_of_budget: float = 0.80,
) -> dict[str, float]:
    budget_bytes = capacity.kv_cache_budget_bytes
    if budget_bytes <= 0:
        raise ValueError("KV-cache budget must be positive")

    if not points:
        return {
            "kv_cache_budget_mib": capacity.effective_kv_cache_budget_mib,
            "peak_kv_budget_utilization": 0.0,
            "p95_kv_budget_utilization": 0.0,
            "kv_budget_pressure_duration_ms": 0.0,
            "kv_budget_exceeded_duration_ms": 0.0,
        }

    peak_bytes = max(point.active_bytes for point in points)
    p95_bytes = _weighted_percentile_bytes(points, 95)
    return {
        "kv_cache_budget_mib": capacity.effective_kv_cache_budget_mib,
        "peak_kv_budget_utilization": peak_bytes / budget_bytes,
        "p95_kv_budget_utilization": p95_bytes / budget_bytes,
        "kv_budget_pressure_duration_ms": _duration_at_or_above(
            points,
            budget_bytes * pressure_fraction_of_budget,
        ),
        "kv_budget_exceeded_duration_ms": _duration_above(points, budget_bytes),
    }


def _positive_number(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"capacity config {field} must be a positive number")
    return float(value)


def _non_negative_number(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if not isinstance(value, int | float) or value < 0:
        raise ValueError(f"capacity config {field} must be a non-negative number")
    return float(value)


def _optional_positive_number(payload: dict[str, Any], field: str) -> float | None:
    if field not in payload or payload[field] is None:
        return None
    return _positive_number(payload, field)


def _duration_at_or_above(points: list[KVCachePoint], threshold_bytes: float) -> float:
    duration = 0.0
    for current, next_point in zip(points, points[1:]):
        interval_ms = next_point.time_ms - current.time_ms
        if interval_ms > 0 and current.active_bytes >= threshold_bytes:
            duration += interval_ms
    return duration


def _duration_above(points: list[KVCachePoint], threshold_bytes: float) -> float:
    duration = 0.0
    for current, next_point in zip(points, points[1:]):
        interval_ms = next_point.time_ms - current.time_ms
        if interval_ms > 0 and current.active_bytes > threshold_bytes:
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
