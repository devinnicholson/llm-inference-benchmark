#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_SUMMARY_JSON = (
    ROOT
    / "results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n16-merged-r8-summary"
    / "prefix-cache-isolated-stability-summary.json"
)
DEFAULT_CANDIDATE_SUMMARY_JSON = (
    ROOT
    / "results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-merged-r8-summary"
    / "prefix-cache-isolated-stability-summary.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/prefix-cache-study-mega-long-qwen15b-l4-n16-vs-n32-r8"
)

Row = dict[str, str]


@dataclass(frozen=True)
class BatchPressurePoint:
    label: str
    summary_json: Path
    source_json: str
    source_dir: str
    request_count: int
    max_new_tokens: int
    paired_observations: int
    control_direct_hit_rate_pct: float
    shared_direct_hit_rate_pct: float
    direct_counter_delta_pp: float
    direct_counter_delta_low_pp: float
    direct_counter_delta_high_pp: float
    first_event_delta: float
    first_event_low: float
    first_event_high: float
    throughput_delta: float
    throughput_low: float
    throughput_high: float
    latency_delta: float
    latency_low: float
    latency_high: float
    stream_tpot_delta: float
    stream_tpot_low: float
    stream_tpot_high: float


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


def _single_profile_control_row(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("profile_control_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("expected non-empty 'profile_control_rows' list")
    if len(rows) != 1:
        raise ValueError("expected exactly one profile-control row")
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


def _point_payload(point: BatchPressurePoint) -> dict[str, Any]:
    return {
        "label": point.label,
        "summary_json": _relative_display_path(point.summary_json),
        "source_json": point.source_json,
        "source_dir": point.source_dir,
        "request_count": point.request_count,
        "max_new_tokens": point.max_new_tokens,
        "paired_observations": point.paired_observations,
        "control_direct_hit_rate_pct": point.control_direct_hit_rate_pct,
        "shared_direct_hit_rate_pct": point.shared_direct_hit_rate_pct,
        "direct_counter_delta_pp": point.direct_counter_delta_pp,
        "direct_counter_delta_low_pp": point.direct_counter_delta_low_pp,
        "direct_counter_delta_high_pp": point.direct_counter_delta_high_pp,
        "first_event_delta": point.first_event_delta,
        "first_event_low": point.first_event_low,
        "first_event_high": point.first_event_high,
        "throughput_delta": point.throughput_delta,
        "throughput_low": point.throughput_low,
        "throughput_high": point.throughput_high,
        "latency_delta": point.latency_delta,
        "latency_low": point.latency_low,
        "latency_high": point.latency_high,
        "stream_tpot_delta": point.stream_tpot_delta,
        "stream_tpot_low": point.stream_tpot_low,
        "stream_tpot_high": point.stream_tpot_high,
    }


def _percent(value: float) -> str:
    return f"{value:.3f}%"


def _pp(value: float) -> str:
    return f"{value:.3f} pp"


def _signed_pp(value: float) -> str:
    return f"{value:+.3f} pp"


def _ratio(value: float) -> str:
    return f"{value:.3f}"


def _signed_ratio(value: float) -> str:
    return f"{value:+.3f}"


def _signed_int(value: int) -> str:
    return f"{value:+d}"


def _row(
    metric: str,
    baseline_value: str,
    candidate_value: str,
    candidate_minus_baseline: str,
    interpretation: str,
) -> Row:
    return {
        "metric": metric,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "candidate_minus_baseline": candidate_minus_baseline,
        "interpretation": interpretation,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON root to be an object")
    return payload


def load_point(summary_json: Path, label: str) -> BatchPressurePoint:
    payload = _read_json(summary_json)
    _require_mapping(payload, "summary")
    control_profile = _require_string(payload, "control_profile")
    shared_profile = _require_string(payload, "shared_profile")
    control = _scenario_by_profile(payload, control_profile)
    shared = _scenario_by_profile(payload, shared_profile)
    profile_control = _single_profile_control_row(payload)

    return BatchPressurePoint(
        label=label,
        summary_json=summary_json,
        source_json=_require_string(payload, "source_json"),
        source_dir=_require_string(payload, "source_dir"),
        request_count=_require_int(profile_control, "request_count"),
        max_new_tokens=_require_int(profile_control, "max_new_tokens"),
        paired_observations=int(
            _require_float(
                profile_control,
                "shared_minus_control_cache_counter_hit_rate_pct_paired_observation_count",
            )
        ),
        control_direct_hit_rate_pct=_require_float(
            control, "cache_prefix_cache_counter_hit_rate_pct_mean"
        ),
        shared_direct_hit_rate_pct=_require_float(
            shared, "cache_prefix_cache_counter_hit_rate_pct_mean"
        ),
        direct_counter_delta_pp=_require_float(
            profile_control, "shared_minus_control_cache_counter_hit_rate_pct_mean"
        ),
        direct_counter_delta_low_pp=_require_float(
            profile_control,
            "shared_minus_control_cache_counter_hit_rate_pct_bootstrap_mean_p05",
        ),
        direct_counter_delta_high_pp=_require_float(
            profile_control,
            "shared_minus_control_cache_counter_hit_rate_pct_bootstrap_mean_p95",
        ),
        first_event_delta=_require_float(
            profile_control,
            "shared_minus_control_cache_to_cold_p95_first_event_ratio_mean",
        ),
        first_event_low=_require_float(
            profile_control,
            "shared_minus_control_cache_to_cold_p95_first_event_ratio_bootstrap_mean_p05",
        ),
        first_event_high=_require_float(
            profile_control,
            "shared_minus_control_cache_to_cold_p95_first_event_ratio_bootstrap_mean_p95",
        ),
        throughput_delta=_require_float(
            profile_control, "shared_minus_control_cache_to_cold_throughput_ratio_mean"
        ),
        throughput_low=_require_float(
            profile_control,
            "shared_minus_control_cache_to_cold_throughput_ratio_bootstrap_mean_p05",
        ),
        throughput_high=_require_float(
            profile_control,
            "shared_minus_control_cache_to_cold_throughput_ratio_bootstrap_mean_p95",
        ),
        latency_delta=_require_float(
            profile_control, "shared_minus_control_cache_to_cold_p95_latency_ratio_mean"
        ),
        latency_low=_require_float(
            profile_control,
            "shared_minus_control_cache_to_cold_p95_latency_ratio_bootstrap_mean_p05",
        ),
        latency_high=_require_float(
            profile_control,
            "shared_minus_control_cache_to_cold_p95_latency_ratio_bootstrap_mean_p95",
        ),
        stream_tpot_delta=_require_float(
            profile_control,
            "shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_mean",
        ),
        stream_tpot_low=_require_float(
            profile_control,
            "shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_bootstrap_mean_p05",
        ),
        stream_tpot_high=_require_float(
            profile_control,
            "shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_bootstrap_mean_p95",
        ),
    )


def build_rows(
    baseline: BatchPressurePoint,
    candidate: BatchPressurePoint,
) -> list[Row]:
    return [
        _row(
            "Request count",
            str(baseline.request_count),
            str(candidate.request_count),
            _signed_int(candidate.request_count - baseline.request_count),
            "Batch pressure is higher while model, GPU, prompt family, repeats, and output tokens stay matched.",
        ),
        _row(
            "Shared-minus-control direct counter delta",
            _pp(baseline.direct_counter_delta_pp),
            _pp(candidate.direct_counter_delta_pp),
            _signed_pp(candidate.direct_counter_delta_pp - baseline.direct_counter_delta_pp),
            "Direct measured-window KV reuse remains strong at the higher request count.",
        ),
        _row(
            "Shared direct counter hit rate",
            _percent(baseline.shared_direct_hit_rate_pct),
            _percent(candidate.shared_direct_hit_rate_pct),
            _signed_pp(
                candidate.shared_direct_hit_rate_pct - baseline.shared_direct_hit_rate_pct
            ),
            "The shared-prefix workload still drives high direct cache reuse under n32.",
        ),
        _row(
            "Control direct counter hit rate",
            _percent(baseline.control_direct_hit_rate_pct),
            _percent(candidate.control_direct_hit_rate_pct),
            _signed_pp(
                candidate.control_direct_hit_rate_pct
                - baseline.control_direct_hit_rate_pct
            ),
            "The matched unique-prefix control remains near zero in both artifacts.",
        ),
        _row(
            "Throughput-ratio delta",
            _ratio(baseline.throughput_delta),
            _ratio(candidate.throughput_delta),
            _signed_ratio(candidate.throughput_delta - baseline.throughput_delta),
            "The n32 artifact shows a larger favorable throughput effect.",
        ),
        _row(
            "p95 first-event/TTFT ratio delta",
            _ratio(baseline.first_event_delta),
            _ratio(candidate.first_event_delta),
            _signed_ratio(candidate.first_event_delta - baseline.first_event_delta),
            "First-token latency remains favorable and is slightly stronger under n32.",
        ),
        _row(
            "p95 latency-ratio delta",
            _ratio(baseline.latency_delta),
            _ratio(candidate.latency_delta),
            _signed_ratio(candidate.latency_delta - baseline.latency_delta),
            "End-to-end p95 latency remains favorable and slightly stronger under n32.",
        ),
        _row(
            "p95 stream TPOT-ratio delta",
            _ratio(baseline.stream_tpot_delta),
            _ratio(candidate.stream_tpot_delta),
            _signed_ratio(candidate.stream_tpot_delta - baseline.stream_tpot_delta),
            "Decode TPOT remains favorable, but the n32 improvement is weaker than n16.",
        ),
    ]


def build_payload(
    baseline: BatchPressurePoint,
    candidate: BatchPressurePoint,
    rows: list[Row],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "prefix-cache-batch-pressure-comparison",
        "baseline": _point_payload(baseline),
        "candidate": _point_payload(candidate),
        "comparison_rows": rows,
    }


def render_markdown(
    baseline: BatchPressurePoint,
    candidate: BatchPressurePoint,
    rows: list[Row],
) -> str:
    lines = [
        "# Prefix-Cache Batch-Pressure Comparison",
        "",
        f"Baseline: `{baseline.label}` from `{_relative_display_path(baseline.summary_json)}`",
        f"Candidate: `{candidate.label}` from `{_relative_display_path(candidate.summary_json)}`",
        "",
        "This compares repeat-count-matched Qwen 1.5B L4 mega-long artifacts while changing request count from n16 to n32.",
        "",
        "## Key Differences",
        "",
        f"| Metric | {baseline.label} | {candidate.label} | {candidate.label} - {baseline.label} | Interpretation |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    lines.extend(
        "| {metric} | {baseline_value} | {candidate_value} | {candidate_minus_baseline} | {interpretation} |".format(
            **row
        )
        for row in rows
    )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "The higher-request-count artifact preserves the direct KV-cache reuse claim and strengthens the throughput and first-token effects. The weaker stream TPOT delta means the next systems question is not whether prefix reuse exists, but why the n32 capacity profile changes decode behavior.",
            "",
            "The observed n32 vLLM capacity drop came from Modal console logs, not from the persisted JSON schema, so this comparison should be paired with a follow-up scheduler/capacity diagnostic before making backend-level claims.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "baseline_value",
                "candidate_value",
                "candidate_minus_baseline",
                "interpretation",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed-model n16-vs-n32 prefix-cache batch-pressure comparison."
    )
    parser.add_argument(
        "--baseline-summary-json",
        type=Path,
        default=DEFAULT_BASELINE_SUMMARY_JSON,
        help="Baseline fixed-shape stability summary JSON.",
    )
    parser.add_argument(
        "--candidate-summary-json",
        type=Path,
        default=DEFAULT_CANDIDATE_SUMMARY_JSON,
        help="Higher-pressure fixed-shape stability summary JSON.",
    )
    parser.add_argument(
        "--baseline-label",
        default="n16 r8",
        help="Display label for the baseline artifact.",
    )
    parser.add_argument(
        "--candidate-label",
        default="n32 r8",
        help="Display label for the candidate artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for comparison JSON, CSV, and Markdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_point(args.baseline_summary_json, args.baseline_label)
    candidate = load_point(args.candidate_summary_json, args.candidate_label)
    rows = build_rows(baseline, candidate)
    payload = build_payload(baseline, candidate, rows)
    markdown = render_markdown(baseline, candidate, rows)

    json_path = args.output_dir / "batch-pressure-comparison.json"
    csv_path = args.output_dir / "batch-pressure-comparison.csv"
    markdown_path = args.output_dir / "batch-pressure-comparison.md"
    write_json(json_path, payload)
    write_csv(csv_path, rows)
    write_markdown(markdown_path, markdown)

    print(f"rows: {len(rows)}")
    print(f"json: {_relative_display_path(json_path)}")
    print(f"csv: {_relative_display_path(csv_path)}")
    print(f"markdown: {_relative_display_path(markdown_path)}")


if __name__ == "__main__":
    main()
