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
    SCHEDULING_POLICIES,
    run_sweep,
    write_sweep_csv,
    write_sweep_json,
)

DEFAULT_WORKLOAD = ROOT / "workloads/generated/mixed_bursty_32_seed568.json"
DEFAULT_MODEL_CONFIGS = (ROOT / "configs/models/llama-7b-gqa-fp16.json",)
DEFAULT_CAPACITY_CONFIGS = (
    ROOT / "configs/capacity/tight-1gb-kv.json",
    ROOT / "configs/capacity/tight-1-2gb-kv.json",
    ROOT / "configs/capacity/a10g-24gb-7b-gqa.json",
)
DEFAULT_CONCURRENCY = (1, 2, 4, 8)
DEFAULT_POLICIES = (
    "fifo",
    "shortest-cache",
    "deadline",
    "memory-aware-deadline",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a deterministic local sweep across workloads, models, capacities, and schedulers."
    )
    parser.add_argument(
        "workload",
        type=Path,
        nargs="*",
        help="Workload JSON path. Defaults to the generated mixed_bursty workload.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        action="append",
        default=None,
        help="Model KV-cache config JSON. Repeat to compare models.",
    )
    parser.add_argument(
        "--capacity-config",
        type=Path,
        action="append",
        default=None,
        help="Capacity config JSON. Repeat to compare GPU/KV budgets.",
    )
    parser.add_argument(
        "--include-unbounded-capacity",
        action="store_true",
        help="Also run policies that do not require a KV-cache budget without a capacity config.",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        action="append",
        default=None,
        help="Concurrent request slots. Repeat to compare concurrency levels.",
    )
    parser.add_argument(
        "--scheduler-policy",
        choices=SCHEDULING_POLICIES,
        action="append",
        default=None,
        help="Scheduler policy. Repeat to compare policies.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/experiment-001-capacity-sweep",
        help="Directory for sweep-results.json and sweep-results.csv.",
    )
    args = parser.parse_args()

    workloads = args.workload or [DEFAULT_WORKLOAD]
    model_configs = args.model_config or list(DEFAULT_MODEL_CONFIGS)
    capacity_configs: list[Path | None] = args.capacity_config or list(DEFAULT_CAPACITY_CONFIGS)
    if args.include_unbounded_capacity:
        capacity_configs = [None, *capacity_configs]

    results = run_sweep(
        workload_paths=workloads,
        model_config_paths=model_configs,
        capacity_config_paths=capacity_configs,
        max_concurrent_requests_values=args.max_concurrent_requests or list(DEFAULT_CONCURRENCY),
        scheduler_policies=args.scheduler_policy or list(DEFAULT_POLICIES),
    )

    json_path = args.output_dir / "sweep-results.json"
    csv_path = args.output_dir / "sweep-results.csv"
    write_sweep_json(json_path, results)
    write_sweep_csv(csv_path, results)

    print(f"cases: {len(results)}")
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
