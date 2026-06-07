#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llmbench import (
    KVCacheConfig,
    SCHEDULING_POLICIES,
    load_kv_cache_config,
    load_capacity_config,
    load_workload,
    simulate_scheduler,
    summarize_traces,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay an LLM inference workload.")
    parser.add_argument("workload", type=Path, help="Path to workload JSON")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Path to model KV-cache config JSON",
    )
    parser.add_argument(
        "--capacity-config",
        type=Path,
        default=None,
        help="Path to serving capacity config JSON",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=1,
        help="Maximum concurrent requests in the simulated serving baseline",
    )
    parser.add_argument(
        "--scheduler-policy",
        choices=SCHEDULING_POLICIES,
        default="fifo",
        help="Request selection policy when requests are waiting",
    )
    args = parser.parse_args()

    kv_cache = load_kv_cache_config(args.model_config) if args.model_config else KVCacheConfig()
    capacity = load_capacity_config(args.capacity_config) if args.capacity_config else None
    workload = load_workload(args.workload)
    traces = simulate_scheduler(
        workload,
        kv_cache=kv_cache,
        max_concurrent_requests=args.max_concurrent_requests,
        scheduling_policy=args.scheduler_policy,
        capacity=capacity,
    )
    summary = summarize_traces(traces, capacity=capacity)

    print(f"workload: {workload.name}")
    if workload.description:
        print(f"description: {workload.description}")
    print(f"model_config: {kv_cache.name}")
    print(f"kv_bytes_per_token: {kv_cache.bytes_per_token}")
    if capacity:
        print(f"capacity_config: {capacity.name}")
        print(f"kv_cache_budget_mib: {capacity.effective_kv_cache_budget_mib}")
    print(f"max_concurrent_requests: {args.max_concurrent_requests}")
    print(f"scheduler_policy: {args.scheduler_policy}")
    print()
    print(
        "request_id                 arrival   queue    ttft     total    "
        "prompt   output     kv_mib"
    )
    print("-" * 90)
    for trace in traces:
        print(
            f"{trace.request_id:<25} "
            f"{trace.arrival_ms:>7.1f} "
            f"{trace.queue_wait_ms:>7.1f} "
            f"{trace.ttft_ms:>7.1f} "
            f"{trace.latency_ms:>8.1f} "
            f"{trace.prompt_tokens:>7} "
            f"{trace.output_tokens:>8} "
            f"{trace.kv_cache_mib:>10.1f}"
        )

    print()
    print("summary")
    print("-" * 90)
    for key, value in summary.items():
        print(f"{key:<25} {value:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
