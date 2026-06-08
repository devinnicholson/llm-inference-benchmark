#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class EffectInterval:
    metric: str
    value: float
    low: float
    high: float
    unit: str
    axis_low: float
    axis_high: float
    interpretation: str


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


def _crosses_zero(low: float, high: float) -> bool:
    return low <= 0.0 <= high


def _direct_counter_interpretation(low: float, high: float) -> str:
    if low > 0.0:
        return "Stable positive reuse effect; interval is far above zero."
    if high < 0.0:
        return "Stable negative reuse effect; investigate the control shape."
    return "Interval crosses zero; no stable direct-counter claim."


def _throughput_interpretation(low: float, high: float) -> str:
    if _crosses_zero(low, high):
        return "Interval crosses zero; no stable throughput claim."
    if low > 0.0:
        return "Stable favorable throughput effect; positive ratio is better."
    return "Stable unfavorable throughput effect; positive ratio is better."


def _latency_interpretation(low: float, high: float, metric: str) -> str:
    if _crosses_zero(low, high):
        return f"Interval crosses zero; no stable {metric} claim."
    if high < 0.0:
        return f"Stable favorable {metric} effect; negative latency ratio is better."
    return f"Stable unfavorable {metric} effect; negative latency ratio is better."


def _row(metric: str, value: str, interval: str, interpretation: str) -> Row:
    return {
        "metric": metric,
        "value": value,
        "bootstrap_interval": interval,
        "interpretation": interpretation,
    }


def _effect(
    payload: dict[str, Any],
    mean_key: str,
    low_key: str,
    high_key: str,
) -> tuple[float, float, float]:
    return (
        _require_float(payload, mean_key),
        _require_float(payload, low_key),
        _require_float(payload, high_key),
    )


def build_rows(payload: dict[str, Any]) -> list[Row]:
    _require_mapping(payload, "summary")
    control_profile = _require_string(payload, "control_profile")
    shared_profile = _require_string(payload, "shared_profile")
    control = _scenario_by_profile(payload, control_profile)
    shared = _scenario_by_profile(payload, shared_profile)
    profile_control = _single_profile_control_row(payload)

    direct_delta, direct_low, direct_high = _effect(
        profile_control,
        "shared_minus_control_cache_counter_hit_rate_pct_mean",
        "shared_minus_control_cache_counter_hit_rate_pct_bootstrap_mean_p05",
        "shared_minus_control_cache_counter_hit_rate_pct_bootstrap_mean_p95",
    )
    first_event_delta, first_event_low, first_event_high = _effect(
        profile_control,
        "shared_minus_control_cache_to_cold_p95_first_event_ratio_mean",
        "shared_minus_control_cache_to_cold_p95_first_event_ratio_bootstrap_mean_p05",
        "shared_minus_control_cache_to_cold_p95_first_event_ratio_bootstrap_mean_p95",
    )
    throughput_delta, throughput_low, throughput_high = _effect(
        profile_control,
        "shared_minus_control_cache_to_cold_throughput_ratio_mean",
        "shared_minus_control_cache_to_cold_throughput_ratio_bootstrap_mean_p05",
        "shared_minus_control_cache_to_cold_throughput_ratio_bootstrap_mean_p95",
    )
    latency_delta, latency_low, latency_high = _effect(
        profile_control,
        "shared_minus_control_cache_to_cold_p95_latency_ratio_mean",
        "shared_minus_control_cache_to_cold_p95_latency_ratio_bootstrap_mean_p05",
        "shared_minus_control_cache_to_cold_p95_latency_ratio_bootstrap_mean_p95",
    )
    stream_tpot_delta, stream_tpot_low, stream_tpot_high = _effect(
        profile_control,
        "shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_mean",
        "shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_bootstrap_mean_p05",
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
            _direct_counter_interpretation(direct_low, direct_high),
        ),
        _row(
            "p95 first-event/TTFT ratio delta",
            _ratio(first_event_delta),
            _interval(first_event_low, first_event_high),
            _latency_interpretation(
                first_event_low,
                first_event_high,
                "first-token",
            ),
        ),
        _row(
            "Throughput-ratio delta",
            _ratio(throughput_delta),
            _interval(throughput_low, throughput_high),
            _throughput_interpretation(throughput_low, throughput_high),
        ),
        _row(
            "p95 latency-ratio delta",
            _ratio(latency_delta),
            _interval(latency_low, latency_high),
            _latency_interpretation(
                latency_low,
                latency_high,
                "end-to-end latency",
            ),
        ),
        _row(
            "p95 stream TPOT-ratio delta",
            _ratio(stream_tpot_delta),
            _interval(stream_tpot_low, stream_tpot_high),
            _latency_interpretation(
                stream_tpot_low,
                stream_tpot_high,
                "decode TPOT",
            ),
        ),
    ]


def build_intervals(payload: dict[str, Any]) -> list[EffectInterval]:
    profile_control = _single_profile_control_row(payload)
    direct_delta, direct_low, direct_high = _effect(
        profile_control,
        "shared_minus_control_cache_counter_hit_rate_pct_mean",
        "shared_minus_control_cache_counter_hit_rate_pct_bootstrap_mean_p05",
        "shared_minus_control_cache_counter_hit_rate_pct_bootstrap_mean_p95",
    )
    first_event_delta, first_event_low, first_event_high = _effect(
        profile_control,
        "shared_minus_control_cache_to_cold_p95_first_event_ratio_mean",
        "shared_minus_control_cache_to_cold_p95_first_event_ratio_bootstrap_mean_p05",
        "shared_minus_control_cache_to_cold_p95_first_event_ratio_bootstrap_mean_p95",
    )
    throughput_delta, throughput_low, throughput_high = _effect(
        profile_control,
        "shared_minus_control_cache_to_cold_throughput_ratio_mean",
        "shared_minus_control_cache_to_cold_throughput_ratio_bootstrap_mean_p05",
        "shared_minus_control_cache_to_cold_throughput_ratio_bootstrap_mean_p95",
    )
    latency_delta, latency_low, latency_high = _effect(
        profile_control,
        "shared_minus_control_cache_to_cold_p95_latency_ratio_mean",
        "shared_minus_control_cache_to_cold_p95_latency_ratio_bootstrap_mean_p05",
        "shared_minus_control_cache_to_cold_p95_latency_ratio_bootstrap_mean_p95",
    )
    stream_tpot_delta, stream_tpot_low, stream_tpot_high = _effect(
        profile_control,
        "shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_mean",
        "shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_bootstrap_mean_p05",
        "shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_bootstrap_mean_p95",
    )

    counter_axis_low = 0.0
    counter_axis_high = max(direct_high * 1.05, 1.0)
    ratio_lows = [first_event_low, throughput_low, latency_low, stream_tpot_low, 0.0]
    ratio_highs = [
        first_event_high,
        throughput_high,
        latency_high,
        stream_tpot_high,
        0.0,
    ]
    ratio_axis_low = min(ratio_lows)
    ratio_axis_high = max(ratio_highs)
    ratio_span = max(ratio_axis_high - ratio_axis_low, 1e-9)
    ratio_axis_low -= ratio_span * 0.05
    ratio_axis_high += ratio_span * 0.05

    return [
        EffectInterval(
            metric="Direct counter delta",
            value=direct_delta,
            low=direct_low,
            high=direct_high,
            unit="pp",
            axis_low=counter_axis_low,
            axis_high=counter_axis_high,
            interpretation=_direct_counter_interpretation(
                direct_low,
                direct_high,
            ),
        ),
        EffectInterval(
            metric="p95 first-event/TTFT ratio delta",
            value=first_event_delta,
            low=first_event_low,
            high=first_event_high,
            unit="ratio",
            axis_low=ratio_axis_low,
            axis_high=ratio_axis_high,
            interpretation=_latency_interpretation(
                first_event_low,
                first_event_high,
                "first-token",
            ),
        ),
        EffectInterval(
            metric="Throughput-ratio delta",
            value=throughput_delta,
            low=throughput_low,
            high=throughput_high,
            unit="ratio",
            axis_low=ratio_axis_low,
            axis_high=ratio_axis_high,
            interpretation=_throughput_interpretation(
                throughput_low,
                throughput_high,
            ),
        ),
        EffectInterval(
            metric="p95 latency-ratio delta",
            value=latency_delta,
            low=latency_low,
            high=latency_high,
            unit="ratio",
            axis_low=ratio_axis_low,
            axis_high=ratio_axis_high,
            interpretation=_latency_interpretation(
                latency_low,
                latency_high,
                "end-to-end latency",
            ),
        ),
        EffectInterval(
            metric="p95 stream TPOT-ratio delta",
            value=stream_tpot_delta,
            low=stream_tpot_low,
            high=stream_tpot_high,
            unit="ratio",
            axis_low=ratio_axis_low,
            axis_high=ratio_axis_high,
            interpretation=_latency_interpretation(
                stream_tpot_low,
                stream_tpot_high,
                "decode TPOT",
            ),
        ),
    ]


def _format_interval_value(value: float, unit: str) -> str:
    if unit == "pp":
        return f"{value:.3f} pp"
    return f"{value:.3f}"


def _interval_bar(interval: EffectInterval, width: int = 48) -> str:
    def position(value: float) -> int:
        span = interval.axis_high - interval.axis_low
        scaled = (value - interval.axis_low) / span
        return max(0, min(width - 1, round(scaled * (width - 1))))

    low_pos = position(interval.low)
    high_pos = position(interval.high)
    value_pos = position(interval.value)
    zero_pos = position(0.0)

    chars = ["-"] * width
    for index in range(min(low_pos, high_pos), max(low_pos, high_pos) + 1):
        chars[index] = "="
    if interval.axis_low <= 0.0 <= interval.axis_high:
        chars[zero_pos] = "|"
    chars[value_pos] = "*"
    return "[" + "".join(chars) + "]"


def _render_interval_row(interval: EffectInterval) -> str:
    interval_text = (
        f"{_format_interval_value(interval.low, interval.unit)} to "
        f"{_format_interval_value(interval.high, interval.unit)}"
    )
    value_text = _format_interval_value(interval.value, interval.unit)
    return (
        f"{interval.metric:<34} {_interval_bar(interval)} "
        f"{value_text:>10}  {interval_text:<24}  {interval.interpretation}"
    )


def render_interval_chart(intervals: list[EffectInterval], source_json: Path) -> str:
    counter_intervals = [interval for interval in intervals if interval.unit == "pp"]
    ratio_intervals = [interval for interval in intervals if interval.unit == "ratio"]
    counter_axis = counter_intervals[0] if counter_intervals else None
    ratio_axis = ratio_intervals[0] if ratio_intervals else None
    lines = [
        "# Prefix-Cache Study Interval Chart",
        "",
        f"Source: `{_relative_display_path(source_json)}`",
        "",
        "`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.",
        "",
        "## Direct Cache Counter Delta",
        "",
        "Positive percentage points mean the shared-prefix profile reused more KV cache.",
        "",
        "```text",
        (
            "scale: "
            f"{counter_axis.axis_low:.3f} pp to {counter_axis.axis_high:.3f} pp"
            if counter_axis
            else "scale: n/a"
        ),
    ]
    lines.extend(_render_interval_row(interval) for interval in counter_intervals)
    lines.extend(
        [
            "```",
            "",
            "## Cache-To-Cold Ratio Deltas",
            "",
            "For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.",
            "",
            "```text",
            (
                "scale: "
                f"{ratio_axis.axis_low:.3f} to {ratio_axis.axis_high:.3f}"
                if ratio_axis
                else "scale: n/a"
            ),
        ]
    )
    lines.extend(_render_interval_row(interval) for interval in ratio_intervals)
    lines.extend(["```", ""])
    return "\n".join(lines)


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


def write_interval_chart(
    path: Path,
    intervals: list[EffectInterval],
    source_json: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_interval_chart(intervals, source_json))


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
    intervals = build_intervals(payload)

    csv_path = args.output_dir / "key-results.csv"
    markdown_path = args.output_dir / "key-results.md"
    intervals_path = args.output_dir / "intervals.md"
    write_csv(csv_path, rows)
    write_markdown(markdown_path, rows, args.summary_json)
    write_interval_chart(intervals_path, intervals, args.summary_json)

    print(f"rows: {len(rows)}")
    print(f"csv: {csv_path}")
    print(f"markdown: {markdown_path}")
    print(f"intervals: {intervals_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
