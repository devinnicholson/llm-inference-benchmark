#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llmbench import load_workload, simulate_fifo, summarize_traces


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay an LLM inference workload.")
    parser.add_argument("workload", type=Path, help="Path to workload JSON")
    args = parser.parse_args()

    workload = load_workload(args.workload)
    traces = simulate_fifo(workload)
    summary = summarize_traces(traces)

    print(f"workload: {workload.name}")
    if workload.description:
        print(f"description: {workload.description}")
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
