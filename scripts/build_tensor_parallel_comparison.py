#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SINGLE_JSON = (
    ROOT
    / "results/modal-vllm-qwen7b-l4-tp1-batched8192-r3-seed4409"
    / "paired-server-async.json"
)
DEFAULT_TENSOR_JSON = (
    ROOT
    / "results/modal-vllm-qwen7b-l4x2-tp2-batched8192-r3-seed4409"
    / "paired-server-async.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/tensor-parallel-qwen7b-l4-comparison"

MATCHED_FIELDS = (
    "model_id",
    "request_counts",
    "prompt_profiles",
    "output_tokens",
    "repeats",
    "scenario_seed",
    "warmup_runs",
    "phase_order",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "gpu_memory_utilization",
    "async_enable_prefix_caching_configured",
    "server_prefix_caching_configured",
    "vllm_version",
)

METRICS = {
    "server": {
        "throughput": ("server_output_tokens_per_second", "output tokens/s", "higher"),
        "p95_ttft": ("server_p95_first_content_ms", "ms", "lower"),
        "p95_latency": ("server_p95_latency_ms", "ms", "lower"),
        "p95_tpot": ("server_p95_stream_tpot_ms", "ms/token", "lower"),
    },
    "async_llm": {
        "throughput": ("async_output_tokens_per_second", "output tokens/s", "higher"),
        "p95_ttft": ("async_p95_first_event_ms", "ms", "lower"),
        "p95_latency": ("async_p95_latency_ms", "ms", "lower"),
        "p95_tpot": ("async_p95_stream_tpot_ms", "ms/token", "lower"),
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _gpu_lines(payload: dict[str, Any]) -> list[str]:
    value = payload.get("nvidia_smi_before")
    if not isinstance(value, str):
        raise ValueError("nvidia_smi_before must be a string")
    return [line.strip() for line in value.splitlines() if line.strip()]


def _server_command_tp_size(payload: dict[str, Any]) -> int:
    command = payload.get("server_command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ValueError("server_command must be a list of strings")
    if "--tensor-parallel-size" not in command:
        return 1
    index = command.index("--tensor-parallel-size")
    try:
        return int(command[index + 1])
    except (IndexError, ValueError) as error:
        raise ValueError("invalid --tensor-parallel-size in server_command") from error


def _validate_inputs(single: dict[str, Any], tensor: dict[str, Any]) -> None:
    for label, payload in (("single", single), ("tensor", tensor)):
        if payload.get("mode") != "vllm-server-async-paired":
            raise ValueError(f"{label} artifact has unsupported mode")
        if not isinstance(payload.get("paired_runs"), list) or not payload["paired_runs"]:
            raise ValueError(f"{label} artifact has no paired_runs")

    mismatches = [
        field
        for field in MATCHED_FIELDS
        if single.get(field) != tensor.get(field)
    ]
    if mismatches:
        raise ValueError(
            "artifacts do not have matched configurations: " + ", ".join(mismatches)
        )

    if single.get("tensor_parallel_size") != 1:
        raise ValueError("single artifact must use tensor_parallel_size=1")
    tensor_parallel_size = tensor.get("tensor_parallel_size")
    if not isinstance(tensor_parallel_size, int) or tensor_parallel_size <= 1:
        raise ValueError("tensor artifact must use tensor_parallel_size > 1")

    single_gpu_count = len(_gpu_lines(single))
    tensor_gpu_count = len(_gpu_lines(tensor))
    if single_gpu_count != 1:
        raise ValueError("single artifact must report exactly one GPU")
    if tensor_gpu_count != tensor_parallel_size:
        raise ValueError("tensor artifact GPU count must match tensor_parallel_size")
    if _server_command_tp_size(single) != 1:
        raise ValueError("single server command has an unexpected TP size")
    if _server_command_tp_size(tensor) != tensor_parallel_size:
        raise ValueError("tensor server command does not match artifact TP size")


def _parse_capacity(payload: dict[str, Any]) -> dict[str, Any]:
    logs = payload.get("server_logs_head", []) + payload.get("server_logs_tail", [])
    if not isinstance(logs, list):
        raise ValueError("server logs must be lists")
    text = "\n".join(str(line) for line in logs)

    def last_float(pattern: str) -> float | None:
        matches = re.findall(pattern, text)
        return float(matches[-1].replace(",", "")) if matches else None

    return {
        "gpu_count": len(_gpu_lines(payload)),
        "gpu_names": [line.split(",", 1)[0] for line in _gpu_lines(payload)],
        "available_kv_cache_memory_gib": last_float(
            r"Available KV cache memory:\s*([0-9.]+)\s*GiB"
        ),
        "gpu_kv_cache_size_tokens": last_float(
            r"GPU KV cache size:\s*([0-9,]+)\s*tokens"
        ),
        "max_concurrency_for_request": last_float(
            r"Maximum concurrency for [0-9,]+ tokens per request:\s*([0-9.]+)x"
        ),
        "custom_all_reduce_disabled": "Custom allreduce is disabled" in text,
        "p2p_unavailable": "lacks GPU P2P capability" in text,
    }


def _paired_run_map(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in payload["paired_runs"]:
        if not isinstance(row, dict):
            raise ValueError("paired run rows must be objects")
        key = (str(row.get("scenario_id")), int(row.get("repeat_index")))
        if key in rows:
            raise ValueError(f"duplicate paired run key: {key}")
        rows[key] = row
    return rows


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _bootstrap_median_interval(
    values: list[float],
    *,
    seed: int = 20260807,
    samples: int = 10_000,
    confidence: float = 0.90,
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    estimates = [
        median(rng.choice(values) for _ in values)
        for _ in range(samples)
    ]
    tail = (1.0 - confidence) / 2.0
    return _percentile(estimates, tail), _percentile(estimates, 1.0 - tail)


def _comparison_rows(
    single: dict[str, Any],
    tensor: dict[str, Any],
) -> list[dict[str, Any]]:
    single_runs = _paired_run_map(single)
    tensor_runs = _paired_run_map(tensor)
    if set(single_runs) != set(tensor_runs):
        raise ValueError("artifacts do not contain the same scenario/repeat keys")

    rows: list[dict[str, Any]] = []
    scenario_ids = sorted({key[0] for key in single_runs})
    for scenario_id in scenario_ids:
        keys = sorted(key for key in single_runs if key[0] == scenario_id)
        for backend, metrics in METRICS.items():
            for metric, (field, unit, favorable_direction) in metrics.items():
                single_values = [float(single_runs[key][field]) for key in keys]
                tensor_values = [float(tensor_runs[key][field]) for key in keys]
                ratios = [
                    tensor_value / single_value
                    for single_value, tensor_value in zip(single_values, tensor_values)
                ]
                ratio_median = median(ratios)
                ci_low, ci_high = _bootstrap_median_interval(ratios)
                favorable = (
                    ratio_median > 1.0
                    if favorable_direction == "higher"
                    else ratio_median < 1.0
                )
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "backend": backend,
                        "metric": metric,
                        "unit": unit,
                        "favorable_direction": favorable_direction,
                        "paired_observations": len(ratios),
                        "single_median": median(single_values),
                        "tensor_median": median(tensor_values),
                        "tensor_div_single_median": ratio_median,
                        "tensor_div_single_bootstrap_90_low": ci_low,
                        "tensor_div_single_bootstrap_90_high": ci_high,
                        "percent_change": (ratio_median - 1.0) * 100.0,
                        "favorable": favorable,
                    }
                )
    return rows


def _format_markdown(payload: dict[str, Any]) -> str:
    matched = payload["matched_configuration"]
    single_capacity = payload["capacity"]["single"]
    tensor_capacity = payload["capacity"]["tensor"]
    lines = [
        "# Tensor-Parallel Serving Comparison",
        "",
        (
            f"Model: `{matched['model_id']}` on one NVIDIA L4 versus "
            f"{payload['topology']['tensor_gpu_count']} tensor-parallel L4s."
        ),
        "",
        (
            f"Matched workload: `{matched['prompt_profiles'][0]}`, "
            f"n={matched['request_counts'][0]}, output tokens={matched['output_tokens'][0]}, "
            f"scheduler budget={matched['max_num_batched_tokens']}, "
            f"repeats={matched['repeats']}."
        ),
        "",
        "## Capacity",
        "",
        "| Topology | KV cache tokens | Max concurrency | P2P/custom all-reduce |",
        "| --- | ---: | ---: | --- |",
        (
            f"| 1x L4 / TP1 | {single_capacity['gpu_kv_cache_size_tokens']:.0f} | "
            f"{single_capacity['max_concurrency_for_request']:.2f}x | n/a |"
        ),
        (
            f"| {payload['topology']['tensor_gpu_count']}x L4 / TP{payload['topology']['tensor_parallel_size']} | "
            f"{tensor_capacity['gpu_kv_cache_size_tokens']:.0f} | "
            f"{tensor_capacity['max_concurrency_for_request']:.2f}x | "
            f"{'unavailable; NCCL fallback' if tensor_capacity['p2p_unavailable'] else 'available'} |"
        ),
        "",
        "## Request-path results",
        "",
        "| Backend | Metric | TP1 median | TP2 median | TP2 / TP1 | 90% bootstrap interval |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['backend']}` | `{row['metric']}` | "
            f"{row['single_median']:.3f} | {row['tensor_median']:.3f} | "
            f"{row['tensor_div_single_median']:.3f}x | "
            f"[{row['tensor_div_single_bootstrap_90_low']:.3f}, "
            f"{row['tensor_div_single_bootstrap_90_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "For throughput, values above 1.0 favor TP2. For TTFT, latency, and TPOT, values below 1.0 favor TP2.",
            "",
            "## Interpretation boundary",
            "",
            (
                "This is a controlled topology experiment, not a production-scale claim. "
                "The bootstrap interval resamples matched repeat indices and does not "
                "capture variation across hosts, regions, or GPU interconnects."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_comparison(single_json: Path, tensor_json: Path) -> dict[str, Any]:
    single = _read_json(single_json)
    tensor = _read_json(tensor_json)
    _validate_inputs(single, tensor)
    single_capacity = _parse_capacity(single)
    tensor_capacity = _parse_capacity(tensor)
    if single_capacity["gpu_kv_cache_size_tokens"] is None:
        raise ValueError("single artifact is missing server KV-cache capacity")
    if tensor_capacity["gpu_kv_cache_size_tokens"] is None:
        raise ValueError("tensor artifact is missing server KV-cache capacity")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "mode": "tensor-parallel-comparison",
        "sources": {
            "single": _display_path(single_json),
            "tensor": _display_path(tensor_json),
        },
        "matched_configuration": {
            field: single.get(field) for field in MATCHED_FIELDS
        },
        "topology": {
            "single_gpu_count": single_capacity["gpu_count"],
            "single_tensor_parallel_size": single["tensor_parallel_size"],
            "tensor_gpu_count": tensor_capacity["gpu_count"],
            "tensor_parallel_size": tensor["tensor_parallel_size"],
        },
        "capacity": {
            "single": single_capacity,
            "tensor": tensor_capacity,
            "tensor_div_single_kv_cache_tokens": (
                tensor_capacity["gpu_kv_cache_size_tokens"]
                / single_capacity["gpu_kv_cache_size_tokens"]
            ),
        },
        "rows": _comparison_rows(single, tensor),
    }
    payload["markdown"] = _format_markdown(payload)
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-json", type=Path, default=DEFAULT_SINGLE_JSON)
    parser.add_argument("--tensor-json", type=Path, default=DEFAULT_TENSOR_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    payload = build_comparison(args.single_json, args.tensor_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "tensor-parallel-comparison.json"
    csv_path = args.output_dir / "tensor-parallel-comparison.csv"
    markdown_path = args.output_dir / "tensor-parallel-comparison.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, payload["rows"])
    markdown_path.write_text(payload["markdown"], encoding="utf-8")
    print(f"rows: {len(payload['rows'])}")
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")
    print(f"markdown: {markdown_path}")


if __name__ == "__main__":
    main()
