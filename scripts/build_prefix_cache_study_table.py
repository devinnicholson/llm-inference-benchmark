#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_JSON = (
    ROOT
    / "results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary-ttft"
    / "prefix-cache-isolated-stability-summary.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/prefix-cache-study"

Row = dict[str, str]


def _relative_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"expected {key!r} to be an object")
    return value


def _require_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"expected numeric field {key!r}")
    return float(value)


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"expected string field {key!r}")
    return value


def _single_profile_control_row(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("profile_control_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("expected non-empty 'profile_control_rows' list")
    if len(rows) != 1:
        raise ValueError(
            "expected exactly one profile-control row; use a fixed-shape summary for this table"
        )
    row = rows[0]
    if not isinstance(row, dict):
        raise ValueError("expected profile-control row to be an object")
    return row


def _scenario_by_profile(payload: dict[str, Any], prompt_profile: str) -> dict[str, Any]:
    rows = payload.get("scenario_rows")
    if not isinstance(rows, list):
        raise ValueError("expected 'scenario_rows' list")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("prompt_profile") == prompt_profile
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one scenario row for prompt profile {prompt_profile!r}"
        )
    return matches[0]


def _percent(value: float) -> str:
    return f"{value:.3f}%"


def _percentage_points(value: float) -> str:
    return f"{value:.3f} pp"


def _ratio(value: float) -> str:
    return f"{value:.3f}"


def _interval(low: float, high: float, suffix: str = "") -> str:
    return f"{low:.3f}{suffix} to {high:.3f}{suffix}"


def _row(metric: str, value: str, interval: str, interpretation: str) -> Row:
    return {
        "metric": metric,
        "value": value,
        "bootstrap_interval": interval,
        "interpretation": interpretation,
    }


def build_rows(payload: dict[str, Any]) -> list[Row]:
    _require_mapping(payload, "summary")
    control_profile = _require_string(payload, "control_profile")
    shared_profile = _require_string(payload, "shared_profile")
    control = _scenario_by_profile(payload, control_profile)
    shared = _scenario_by_profile(payload, shared_profile)
    profile_control = _single_profile_control_row(payload)

    direct_delta = _require_float(
        profile_control, "shared_minus_control_cache_counter_hit_rate_pct_mean"
    )
    direct_low = _require_float(
        profile_control, "shared_minus_control_cache_counter_hit_rate_pct_bootstrap_mean_p05"
    )
    direct_high = _require_float(
        profile_control, "shared_minus_control_cache_counter_hit_rate_pct_bootstrap_mean_p95"
    )

    first_event_delta = _require_float(
        profile_control, "shared_minus_control_cache_to_cold_p95_first_event_ratio_mean"
    )
    first_event_low = _require_float(
        profile_control,
        "shared_minus_control_cache_to_cold_p95_first_event_ratio_bootstrap_mean_p05",
    )
    first_event_high = _require_float(
        profile_control,
        "shared_minus_control_cache_to_cold_p95_first_event_ratio_bootstrap_mean_p95",
    )

    throughput_delta = _require_float(
        profile_control, "shared_minus_control_cache_to_cold_throughput_ratio_mean"
    )
    throughput_low = _require_float(
        profile_control,
        "shared_minus_control_cache_to_cold_throughput_ratio_bootstrap_mean_p05",
    )
    throughput_high = _require_float(
        profile_control,
        "shared_minus_control_cache_to_cold_throughput_ratio_bootstrap_mean_p95",
    )

    latency_delta = _require_float(
        profile_control, "shared_minus_control_cache_to_cold_p95_latency_ratio_mean"
    )
    latency_low = _require_float(
        profile_control,
        "shared_minus_control_cache_to_cold_p95_latency_ratio_bootstrap_mean_p05",
    )
    latency_high = _require_float(
        profile_control,
        "shared_minus_control_cache_to_cold_p95_latency_ratio_bootstrap_mean_p95",
    )

    stream_tpot_delta = _require_float(
        profile_control, "shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_mean"
    )
    stream_tpot_low = _require_float(
        profile_control,
        "shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_bootstrap_mean_p05",
    )
    stream_tpot_high = _require_float(
        profile_control,
        "shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_bootstrap_mean_p95",
    )

    return [
        _row(
            "Control direct counter hit rate",
            _percent(_require_float(control, "cache_prefix_cache_counter_hit_rate_pct_mean")),
            "n/a",
            "Matched unique-prefix baseline stays low after the no-repeat prompt audit.",
        ),
        _row(
            "Shared direct counter hit rate",
            _percent(_require_float(shared, "cache_prefix_cache_counter_hit_rate_pct_mean")),
            "n/a",
            "Shared-prefix workload triggers measured-window KV reuse.",
        ),
        _row(
            "Shared-minus-control direct counter delta",
            _percentage_points(direct_delta),
            _interval(direct_low, direct_high, " pp"),
            "Stable positive reuse effect; interval is far above zero.",
        ),
        _row(
            "p95 first-event/TTFT ratio delta",
            _ratio(first_event_delta),
            _interval(first_event_low, first_event_high),
            "Stable favorable first-token effect; negative latency ratio is better.",
        ),
        _row(
            "Throughput-ratio delta",
            _ratio(throughput_delta),
            _interval(throughput_low, throughput_high),
            "Interval crosses zero; no stable throughput claim.",
        ),
        _row(
            "p95 latency-ratio delta",
            _ratio(latency_delta),
            _interval(latency_low, latency_high),
            "Interval crosses zero; no stable end-to-end latency claim.",
        ),
        _row(
            "p95 stream TPOT-ratio delta",
            _ratio(stream_tpot_delta),
            _interval(stream_tpot_low, stream_tpot_high),
            "Interval crosses zero; no stable decode TPOT claim.",
        ),
    ]


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["metric", "value", "bootstrap_interval", "interpretation"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[Row], source_json: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Prefix-Cache Study Key Results",
        "",
        f"Source: `{_relative_display_path(source_json)}`",
        "",
        "| Metric | Value | 90% bootstrap interval | Interpretation |",
        "| --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {metric} | {value} | {bootstrap_interval} | {interpretation} |".format(
                **row
            )
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the GitHub-facing key result table for the prefix-cache study."
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="TTFT-aware prefix-cache stability summary JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for key-results.csv and key-results.md.",
    )
    args = parser.parse_args()

    payload = json.loads(args.summary_json.read_text())
    if not isinstance(payload, dict):
        raise ValueError("expected summary JSON root to be an object")
    rows = build_rows(payload)

    csv_path = args.output_dir / "key-results.csv"
    markdown_path = args.output_dir / "key-results.md"
    write_csv(csv_path, rows)
    write_markdown(markdown_path, rows, args.summary_json)

    print(f"rows: {len(rows)}")
    print(f"csv: {csv_path}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
