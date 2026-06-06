#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llmbench.generate import PROFILES, generate_workload, write_workload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic LLM workloads.")
    parser.add_argument("profile", choices=sorted(PROFILES), help="Workload profile")
    parser.add_argument("--requests", type=int, default=32, help="Number of requests")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    workload = generate_workload(args.profile, args.requests, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_workload(workload, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

