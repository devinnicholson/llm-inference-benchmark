from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from llmbench.capacity import ServingCapacityConfig, load_capacity_config
from llmbench.kv_cache import KVCacheConfig, load_kv_cache_config
from llmbench.simulate import SCHEDULING_POLICIES, simulate_scheduler, summarize_traces
from llmbench.workload import Workload, load_workload


SWEEP_METADATA_FIELDS = (
    "workload_name",
    "workload_path",
    "model_config",
    "model_config_path",
    "capacity_config",
    "capacity_config_path",
    "max_concurrent_requests",
    "scheduler_policy",
)

PREFERRED_SUMMARY_FIELDS = (
    "requests",
    "output_tokens",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "p95_queue_wait_ms",
    "max_request_kv_cache_mib",
    "total_request_kv_cache_mib",
    "peak_active_kv_cache_mib",
    "p95_active_kv_cache_mib",
    "memory_pressure_duration_ms",
    "kv_cache_budget_mib",
    "peak_kv_budget_utilization",
    "p95_kv_budget_utilization",
    "kv_budget_pressure_duration_ms",
    "kv_budget_exceeded_duration_ms",
    "deadline_tracked_requests",
    "deadline_miss_rate",
    "p95_deadline_lateness_ms",
    "requests_per_second",
    "output_tokens_per_second",
)


@dataclass(frozen=True)
class SweepResult:
    workload_name: str
    workload_path: str
    model_config: str
    model_config_path: str
    capacity_config: str
    capacity_config_path: str
    max_concurrent_requests: int
    scheduler_policy: str
    summary: dict[str, float]

    def to_record(self) -> dict[str, Any]:
        return {
            "workload_name": self.workload_name,
            "workload_path": self.workload_path,
            "model_config": self.model_config,
            "model_config_path": self.model_config_path,
            "capacity_config": self.capacity_config,
            "capacity_config_path": self.capacity_config_path,
            "max_concurrent_requests": self.max_concurrent_requests,
            "scheduler_policy": self.scheduler_policy,
            **self.summary,
        }


def run_sweep(
    workload_paths: Sequence[str | Path],
    model_config_paths: Sequence[str | Path | None],
    capacity_config_paths: Sequence[str | Path | None],
    max_concurrent_requests_values: Sequence[int],
    scheduler_policies: Sequence[str],
) -> list[SweepResult]:
    if not workload_paths:
        raise ValueError("at least one workload path is required")
    if not model_config_paths:
        raise ValueError("at least one model config path is required")
    if not capacity_config_paths:
        raise ValueError("at least one capacity config entry is required")
    if not max_concurrent_requests_values:
        raise ValueError("at least one concurrency value is required")
    if not scheduler_policies:
        raise ValueError("at least one scheduler policy is required")

    unknown_policies = sorted(set(scheduler_policies) - set(SCHEDULING_POLICIES))
    if unknown_policies:
        raise ValueError(f"unknown scheduler policies: {', '.join(unknown_policies)}")

    results: list[SweepResult] = []
    for workload_path in workload_paths:
        workload_path = Path(workload_path)
        workload = load_workload(workload_path)
        for model_config_path in model_config_paths:
            kv_cache = _load_kv_config(model_config_path)
            for capacity_config_path in capacity_config_paths:
                capacity = _load_capacity(capacity_config_path)
                for max_concurrent_requests in max_concurrent_requests_values:
                    for scheduler_policy in scheduler_policies:
                        if scheduler_policy == "memory-aware-deadline" and capacity is None:
                            continue
                        results.append(
                            _run_case(
                                workload,
                                workload_path,
                                kv_cache,
                                model_config_path,
                                capacity,
                                capacity_config_path,
                                max_concurrent_requests,
                                scheduler_policy,
                            )
                        )

    return results


def fieldnames_for_results(results: Sequence[SweepResult]) -> list[str]:
    seen_summary_fields = set()
    for result in results:
        seen_summary_fields.update(result.summary)

    summary_fields = [
        field for field in PREFERRED_SUMMARY_FIELDS if field in seen_summary_fields
    ]
    summary_fields.extend(
        sorted(seen_summary_fields - set(PREFERRED_SUMMARY_FIELDS))
    )
    return [*SWEEP_METADATA_FIELDS, *summary_fields]


def write_sweep_csv(path: str | Path, results: Sequence[SweepResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames_for_results(results)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_record())


def write_sweep_json(path: str | Path, results: Sequence[SweepResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "results": [result.to_record() for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_case(
    workload: Workload,
    workload_path: Path,
    kv_cache: KVCacheConfig,
    model_config_path: str | Path | None,
    capacity: ServingCapacityConfig | None,
    capacity_config_path: str | Path | None,
    max_concurrent_requests: int,
    scheduler_policy: str,
) -> SweepResult:
    traces = simulate_scheduler(
        workload,
        kv_cache=kv_cache,
        max_concurrent_requests=max_concurrent_requests,
        scheduling_policy=scheduler_policy,
        capacity=capacity,
    )
    return SweepResult(
        workload_name=workload.name,
        workload_path=_path_label(workload_path),
        model_config=kv_cache.name,
        model_config_path=_path_label(model_config_path),
        capacity_config=capacity.name if capacity else "unbounded",
        capacity_config_path=_path_label(capacity_config_path),
        max_concurrent_requests=max_concurrent_requests,
        scheduler_policy=scheduler_policy,
        summary=summarize_traces(traces, capacity=capacity),
    )


def _load_kv_config(path: str | Path | None) -> KVCacheConfig:
    return load_kv_cache_config(path) if path else KVCacheConfig()


def _load_capacity(path: str | Path | None) -> ServingCapacityConfig | None:
    return load_capacity_config(path) if path else None


def _path_label(path: str | Path | None) -> str:
    if not path:
        return ""
    path = Path(path)
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()
