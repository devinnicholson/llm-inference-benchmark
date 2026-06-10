#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_N16_PRESSURE_CURVE_JSON = (
    ROOT
    / "results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu0325-n16-batched-tokens60640-seed4107-pressure-curve-r1"
    / "server-cache-pressure-curve.json"
)
DEFAULT_N32_PRESSURE_CURVE_JSON = (
    ROOT
    / "results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu045-gpu040-gpu035-gpu0325-n32-batched-tokens60640-seed3805-seed3906-pressure-curve-r1"
    / "server-cache-pressure-curve.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu0325-n16-vs-n32-request-count-r1"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON root to be an object")
    return payload


def _relative_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _require_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"expected numeric field {key!r}")
    return float(value)


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"expected integer field {key!r}")
    return value


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"expected string field {key!r}")
    return value


def _profile_label(prompt_profile: str) -> str:
    if prompt_profile.startswith("matched_unique"):
        return "matched_unique"
    if prompt_profile.startswith("shared_prefix"):
        return "shared_prefix"
    return prompt_profile


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _load_rows(
    pressure_curve_json: Path,
    label: str,
    gpu_memory_utilization: float,
) -> list[dict[str, Any]]:
    payload = _read_json(pressure_curve_json)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("expected non-empty pressure-curve 'rows' list")

    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("expected pressure-curve rows to be objects")
        row_gpu = _require_float(row, "gpu_memory_utilization_mean")
        if row_gpu != gpu_memory_utilization:
            continue
        prompt_profile = _require_string(row, "prompt_profile")
        normalized.append(
            {
                "source_label": label,
                "source_json": _relative_display_path(pressure_curve_json),
                "phase_order": _require_string(row, "phase_order"),
                "prompt_profile": prompt_profile,
                "profile_label": _profile_label(prompt_profile),
                "gpu_memory_utilization": row_gpu,
                "request_count": _require_int(row, "request_count"),
                "max_new_tokens": _require_int(row, "max_new_tokens"),
                "trial_count": _require_int(row, "trial_count"),
                "prompt_tokens_mean": _require_float(row, "prompt_tokens_mean"),
                "estimated_prompt_tokens": _require_float(
                    row, "estimated_prompt_tokens"
                ),
                "kv_cache_size_tokens": _require_float(
                    row, "on_server_gpu_kv_cache_size_tokens_mean"
                ),
                "max_concurrency_for_request": _require_float(
                    row, "on_server_max_concurrency_for_request_mean"
                ),
                "estimated_prompt_token_pressure_ratio": _require_float(
                    row, "estimated_prompt_token_pressure_ratio"
                ),
                "prefix_cache_hit_rate_pct": _require_float(
                    row, "on_prefix_cache_hit_rate_pct_mean"
                ),
                "throughput_ratio": _require_float(row, "throughput_ratio_mean"),
                "p95_latency_ratio": _require_float(row, "p95_latency_ratio_mean"),
                "p95_first_content_ratio": _require_float(
                    row, "p95_first_content_ratio_mean"
                ),
                "p95_stream_tpot_ratio": _require_float(
                    row, "p95_stream_tpot_ratio_mean"
                ),
            }
        )
    if not normalized:
        raise ValueError(
            f"no rows found for gpu_memory_utilization={gpu_memory_utilization}"
        )
    return normalized


def _delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_profile.setdefault(row["profile_label"], []).append(row)

    deltas = []
    for profile_label, profile_rows in sorted(by_profile.items()):
        if len(profile_rows) != 2:
            raise ValueError(
                f"expected two rows for profile {profile_label!r}, got {len(profile_rows)}"
            )
        low, high = sorted(profile_rows, key=lambda row: row["request_count"])
        deltas.append(
            {
                "profile_label": profile_label,
                "low_request_count": low["request_count"],
                "high_request_count": high["request_count"],
                "low_source_label": low["source_label"],
                "high_source_label": high["source_label"],
                "low_trial_count": low["trial_count"],
                "high_trial_count": high["trial_count"],
                "prompt_pressure_delta": (
                    high["estimated_prompt_token_pressure_ratio"]
                    - low["estimated_prompt_token_pressure_ratio"]
                ),
                "prompt_pressure_ratio": _ratio(
                    high["estimated_prompt_token_pressure_ratio"],
                    low["estimated_prompt_token_pressure_ratio"],
                ),
                "throughput_ratio_delta": (
                    high["throughput_ratio"] - low["throughput_ratio"]
                ),
                "throughput_ratio_ratio": _ratio(
                    high["throughput_ratio"],
                    low["throughput_ratio"],
                ),
                "p95_latency_ratio_delta": (
                    high["p95_latency_ratio"] - low["p95_latency_ratio"]
                ),
                "hit_rate_delta_pct": (
                    high["prefix_cache_hit_rate_pct"]
                    - low["prefix_cache_hit_rate_pct"]
                ),
            }
        )
    return deltas


def build_comparison(
    low_pressure_curve_json: Path,
    high_pressure_curve_json: Path,
    low_label: str,
    high_label: str,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    rows = _load_rows(low_pressure_curve_json, low_label, gpu_memory_utilization)
    rows.extend(
        _load_rows(high_pressure_curve_json, high_label, gpu_memory_utilization)
    )
    rows = sorted(
        rows,
        key=lambda row: (
            row["profile_label"],
            row["request_count"],
            row["source_label"],
        ),
    )
    deltas = _delta_rows(rows)
    return {
        "schema_version": 1,
        "mode": "kv-budget-request-count-comparison",
        "gpu_memory_utilization": gpu_memory_utilization,
        "source_jsons": [
            _relative_display_path(low_pressure_curve_json),
            _relative_display_path(high_pressure_curve_json),
        ],
        "row_count": len(rows),
        "delta_count": len(deltas),
        "rows": rows,
        "deltas": deltas,
        "markdown": _format_markdown(
            rows,
            deltas,
            low_pressure_curve_json,
            high_pressure_curve_json,
            gpu_memory_utilization,
        ),
    }


def _format_markdown(
    rows: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    low_pressure_curve_json: Path,
    high_pressure_curve_json: Path,
    gpu_memory_utilization: float,
) -> str:
    shared_rows = [
        row for row in rows if row["profile_label"] == "shared_prefix"
    ]
    matched_rows = [
        row for row in rows if row["profile_label"] == "matched_unique"
    ]
    shared_low, shared_high = sorted(
        shared_rows, key=lambda row: row["request_count"]
    )
    matched_low, matched_high = sorted(
        matched_rows, key=lambda row: row["request_count"]
    )
    lines = [
        "# KV-Budget Request-Count Comparison",
        "",
        f"GPU memory utilization: `{_fmt(gpu_memory_utilization)}`",
        f"Low-n source: `{_relative_display_path(low_pressure_curve_json)}`",
        f"High-n source: `{_relative_display_path(high_pressure_curve_json)}`",
        "",
        "## Headline",
        "",
        (
            f"At the same KV-budget floor, shared-prefix throughput rises from "
            f"{_fmt(shared_low['throughput_ratio'], 'x')} at n={shared_low['request_count']} "
            f"to {_fmt(shared_high['throughput_ratio'], 'x')} at n={shared_high['request_count']}."
        ),
        "",
        (
            "The matched-unique control remains near neutral over the same request-count change: "
            f"{_fmt(matched_low['throughput_ratio'], 'x')} to "
            f"{_fmt(matched_high['throughput_ratio'], 'x')}."
        ),
        "",
        "## Rows",
        "",
        (
            "| Source | Profile | n | Trials | Prompt Pressure | Hit Rate | "
            "Throughput Ratio | p95 Latency Ratio |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['source_label']}` | "
            f"`{row['profile_label']}` | "
            f"{row['request_count']} | "
            f"{row['trial_count']} | "
            f"{_fmt(row['estimated_prompt_token_pressure_ratio'])} | "
            f"{_fmt(row['prefix_cache_hit_rate_pct'], '%')} | "
            f"{_fmt(row['throughput_ratio'], 'x')} | "
            f"{_fmt(row['p95_latency_ratio'], 'x')} |"
        )
    lines.extend(
        [
            "",
            "## Request-Count Delta",
            "",
            (
                "| Profile | n Change | Pressure Ratio | Throughput Delta | "
                "Throughput Ratio Change | p95 Latency Delta | Hit Rate Delta |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in deltas:
        lines.append(
            "| "
            f"`{row['profile_label']}` | "
            f"{row['low_request_count']} -> {row['high_request_count']} | "
            f"{_fmt(row['prompt_pressure_ratio'], 'x')} | "
            f"{_fmt(row['throughput_ratio_delta'], 'x')} | "
            f"{_fmt(row['throughput_ratio_ratio'], 'x')} | "
            f"{_fmt(row['p95_latency_ratio_delta'], 'x')} | "
            f"{_fmt(row['hit_rate_delta_pct'], ' pp')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            (
                "This comparison isolates request count at the lowest successful "
                "KV-budget point. The n=16 row is a one-seed probe; the n=32 row "
                "is the existing two-seed replicated floor point."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--low-pressure-curve-json",
        type=Path,
        default=DEFAULT_N16_PRESSURE_CURVE_JSON,
    )
    parser.add_argument(
        "--high-pressure-curve-json",
        type=Path,
        default=DEFAULT_N32_PRESSURE_CURVE_JSON,
    )
    parser.add_argument("--low-label", default="n16_seed4107")
    parser.add_argument("--high-label", default="n32_seed3805_seed3906")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.325)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    payload = build_comparison(
        low_pressure_curve_json=args.low_pressure_curve_json,
        high_pressure_curve_json=args.high_pressure_curve_json,
        low_label=args.low_label,
        high_label=args.high_label,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "kv-budget-request-count-comparison.json"
    rows_csv_path = args.output_dir / "kv-budget-request-count-rows.csv"
    deltas_csv_path = args.output_dir / "kv-budget-request-count-deltas.csv"
    markdown_path = args.output_dir / "kv-budget-request-count-comparison.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(rows_csv_path, payload["rows"])
    _write_csv(deltas_csv_path, payload["deltas"])
    markdown_path.write_text(payload["markdown"], encoding="utf-8")
    print(f"rows: {payload['row_count']}")
    print(f"deltas: {payload['delta_count']}")
    print(f"markdown: {_relative_display_path(markdown_path)}")


if __name__ == "__main__":
    main()
