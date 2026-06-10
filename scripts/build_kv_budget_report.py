#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESSURE_CURVE_JSON = (
    ROOT
    / "results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu045-gpu040-gpu035-gpu0325-n32-batched-tokens60640-seed3805-seed3906-pressure-curve-r1"
    / "server-cache-pressure-curve.json"
)
DEFAULT_FAILURE_JSON = (
    ROOT
    / "results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu030-n32-batched-tokens60640-feasibility-failure-r1"
    / "failure.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/modal-vllm-server-async-qwen15b-l4-kvbudget-report-r1"
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


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"expected optional numeric field {key!r}")
    return float(value)


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


def _fmt_int(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.0f}"


def _load_curve_rows(pressure_curve_json: Path) -> list[dict[str, Any]]:
    payload = _read_json(pressure_curve_json)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("expected non-empty pressure-curve 'rows' list")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("expected pressure-curve rows to be objects")
        prompt_profile = _require_string(row, "prompt_profile")
        normalized.append(
            {
                "status": "measured",
                "phase_order": _require_string(row, "phase_order"),
                "prompt_profile": prompt_profile,
                "profile_label": _profile_label(prompt_profile),
                "gpu_memory_utilization": _require_float(
                    row, "gpu_memory_utilization_mean"
                ),
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
                "max_num_batched_tokens": _require_float(
                    row, "max_num_batched_tokens_mean"
                ),
                "estimated_prompt_token_pressure_ratio": _require_float(
                    row, "estimated_prompt_token_pressure_ratio"
                ),
                "prefix_cache_hit_rate_pct": _require_float(
                    row, "on_prefix_cache_hit_rate_pct_mean"
                ),
                "prefix_cache_queries": _require_float(
                    row, "on_prefix_cache_queries_mean"
                ),
                "prefix_cache_hits": _require_float(
                    row, "on_prefix_cache_hits_mean"
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
    return sorted(
        normalized,
        key=lambda row: (
            row["profile_label"],
            row["gpu_memory_utilization"],
            row["request_count"],
        ),
    )


def _load_failure(failure_json: Path | None) -> dict[str, Any] | None:
    if failure_json is None:
        return None
    payload = _read_json(failure_json)
    return {
        "status": _require_string(payload, "status"),
        "failure_phase": _require_string(payload, "failure_phase"),
        "gpu_memory_utilization": _require_float(
            payload, "gpu_memory_utilization"
        ),
        "prompt_profile": _require_string(payload, "prompt_profile"),
        "profile_label": _profile_label(_require_string(payload, "prompt_profile")),
        "request_count": _require_int(payload, "request_count"),
        "max_new_tokens": _require_int(payload, "output_tokens"),
        "phase_order": _require_string(payload, "phase_order"),
        "max_num_batched_tokens": _require_float(
            payload, "server_async_max_num_batched_tokens"
        ),
        "available_kv_cache_memory_gib": _optional_float(
            payload, "available_kv_cache_memory_gib"
        ),
        "error_type": _require_string(payload, "error_type"),
        "error_message": _require_string(payload, "error_message"),
        "interpretation": _require_string(payload, "interpretation"),
    }


def _startup_floor(
    rows: list[dict[str, Any]], failure: dict[str, Any] | None
) -> dict[str, Any]:
    min_success_gpu = min(row["gpu_memory_utilization"] for row in rows)
    min_success_rows = [
        row for row in rows if row["gpu_memory_utilization"] == min_success_gpu
    ]
    return {
        "failed_gpu_memory_utilization": (
            failure["gpu_memory_utilization"] if failure is not None else None
        ),
        "min_successful_gpu_memory_utilization": min_success_gpu,
        "min_successful_trial_count": sum(
            row["trial_count"] for row in min_success_rows
        ),
        "min_successful_profiles": sorted(
            {row["profile_label"] for row in min_success_rows}
        ),
        "failure": failure,
    }


def _headline(rows: list[dict[str, Any]], startup: dict[str, Any]) -> dict[str, Any]:
    shared_rows = [row for row in rows if row["profile_label"] == "shared_prefix"]
    matched_rows = [row for row in rows if row["profile_label"] == "matched_unique"]
    if not shared_rows or not matched_rows:
        raise ValueError("expected both shared_prefix and matched_unique rows")
    floor_gpu = startup["min_successful_gpu_memory_utilization"]
    shared_floor = next(
        row
        for row in shared_rows
        if row["gpu_memory_utilization"] == floor_gpu
    )
    matched_floor = next(
        row
        for row in matched_rows
        if row["gpu_memory_utilization"] == floor_gpu
    )
    best_shared = max(shared_rows, key=lambda row: row["throughput_ratio"])
    return {
        "floor_gpu_memory_utilization": floor_gpu,
        "failed_gpu_memory_utilization": startup["failed_gpu_memory_utilization"],
        "shared_floor_prompt_pressure": shared_floor[
            "estimated_prompt_token_pressure_ratio"
        ],
        "shared_floor_throughput_ratio": shared_floor["throughput_ratio"],
        "shared_floor_p95_latency_ratio": shared_floor["p95_latency_ratio"],
        "shared_floor_hit_rate_pct": shared_floor["prefix_cache_hit_rate_pct"],
        "matched_floor_throughput_ratio": matched_floor["throughput_ratio"],
        "best_shared_gpu_memory_utilization": best_shared["gpu_memory_utilization"],
        "best_shared_throughput_ratio": best_shared["throughput_ratio"],
        "matched_throughput_ratio_min": min(
            row["throughput_ratio"] for row in matched_rows
        ),
        "matched_throughput_ratio_max": max(
            row["throughput_ratio"] for row in matched_rows
        ),
        "shared_throughput_ratio_min": min(
            row["throughput_ratio"] for row in shared_rows
        ),
        "shared_throughput_ratio_max": max(
            row["throughput_ratio"] for row in shared_rows
        ),
        "shared_p95_latency_ratio_min": min(
            row["p95_latency_ratio"] for row in shared_rows
        ),
        "shared_p95_latency_ratio_max": max(
            row["p95_latency_ratio"] for row in shared_rows
        ),
    }


def _claims(
    rows: list[dict[str, Any]],
    startup: dict[str, Any],
    headline: dict[str, Any],
) -> list[dict[str, str]]:
    min_success_trials = startup["min_successful_trial_count"]
    successful_budgets = sorted(
        {row["gpu_memory_utilization"] for row in rows}
    )
    return [
        {
            "claim": "The current workload has a measured startup floor between 0.30 and 0.325 GPU memory utilization.",
            "evidence": (
                f"0.30 fails before artifact write; "
                f"0.325 succeeds with {min_success_trials} measured profile trials."
            ),
            "support_level": "direct measurement",
            "caveat": "The interval is bounded by tested points, not by a full binary search.",
        },
        {
            "claim": "Prefix caching is not a generic throughput boost for every prompt shape.",
            "evidence": (
                "Matched-unique cache-on/cache-off throughput stays between "
                f"{_fmt(headline['matched_throughput_ratio_min'], 'x')} and "
                f"{_fmt(headline['matched_throughput_ratio_max'], 'x')}."
            ),
            "support_level": "negative control",
            "caveat": "The control result applies to this no-repeat prompt generator and server configuration.",
        },
        {
            "claim": "Shared-prefix reuse remains valuable at the lowest successful KV budget.",
            "evidence": (
                f"At 0.325 GPU memory utilization, shared-prefix pressure is "
                f"{_fmt(headline['shared_floor_prompt_pressure'])}x, hit rate is "
                f"{_fmt(headline['shared_floor_hit_rate_pct'], '%')}, throughput is "
                f"{_fmt(headline['shared_floor_throughput_ratio'], 'x')}, and p95 latency is "
                f"{_fmt(headline['shared_floor_p95_latency_ratio'], 'x')}."
            ),
            "support_level": "two-seed replicated",
            "caveat": "This is strongest for the synthetic high-overlap workload; broader traffic mixes still need testing.",
        },
        {
            "claim": "The shared-prefix effect is stable across the successful budget curve.",
            "evidence": (
                "Across GPU memory utilization "
                f"{_fmt(min(successful_budgets))} to {_fmt(max(successful_budgets))}, "
                "shared-prefix throughput ranges from "
                f"{_fmt(headline['shared_throughput_ratio_min'], 'x')} to "
                f"{_fmt(headline['shared_throughput_ratio_max'], 'x')}, and p95 latency ranges from "
                f"{_fmt(headline['shared_p95_latency_ratio_min'], 'x')} to "
                f"{_fmt(headline['shared_p95_latency_ratio_max'], 'x')}."
            ),
            "support_level": "replicated sweep",
            "caveat": "All points use one model, one GPU class, one request count, and one output-token setting.",
        },
    ]


def build_report(
    pressure_curve_json: Path,
    failure_json: Path | None = None,
) -> dict[str, Any]:
    rows = _load_curve_rows(pressure_curve_json)
    failure = _load_failure(failure_json)
    startup = _startup_floor(rows, failure)
    headline = _headline(rows, startup)
    claims = _claims(rows, startup, headline)
    return {
        "schema_version": 1,
        "mode": "kv-budget-report",
        "pressure_curve_json": _relative_display_path(pressure_curve_json),
        "failure_json": (
            _relative_display_path(failure_json) if failure_json is not None else None
        ),
        "row_count": len(rows),
        "startup_floor": startup,
        "headline": headline,
        "claims": claims,
        "rows": rows,
        "markdown": _format_markdown(
            rows,
            startup,
            headline,
            claims,
            pressure_curve_json,
            failure_json,
        ),
    }


def _format_markdown(
    rows: list[dict[str, Any]],
    startup: dict[str, Any],
    headline: dict[str, Any],
    claims: list[dict[str, str]],
    pressure_curve_json: Path,
    failure_json: Path | None,
) -> str:
    lines = [
        "# KV-Budget Report",
        "",
        f"Pressure curve source: `{_relative_display_path(pressure_curve_json)}`",
    ]
    if failure_json is not None:
        lines.append(f"Startup floor source: `{_relative_display_path(failure_json)}`")
    lines.extend(
        [
            "",
            "## Headline",
            "",
            (
                f"`gpu_memory_utilization={_fmt(headline['failed_gpu_memory_utilization'])}` "
                "fails during vLLM KV-cache initialization, while "
                f"`gpu_memory_utilization={_fmt(headline['floor_gpu_memory_utilization'])}` "
                "runs successfully."
            ),
            "",
            (
                "At the lowest successful budget, the shared-prefix workload reaches "
                f"{_fmt(headline['shared_floor_prompt_pressure'])}x estimated prompt "
                "pressure with "
                f"{_fmt(headline['shared_floor_hit_rate_pct'], '%')} cache hit rate, "
                f"{_fmt(headline['shared_floor_throughput_ratio'], 'x')} throughput, "
                "and "
                f"{_fmt(headline['shared_floor_p95_latency_ratio'], 'x')} p95 latency "
                "relative to cache-off."
            ),
            "",
            (
                "Matched-unique control throughput stays near neutral across the "
                "successful curve: "
                f"{_fmt(headline['matched_throughput_ratio_min'], 'x')} to "
                f"{_fmt(headline['matched_throughput_ratio_max'], 'x')}."
            ),
            "",
            "## Claim/Evidence Matrix",
            "",
            "| Claim | Evidence | Support | Caveat |",
            "| --- | --- | --- | --- |",
        ]
    )
    for claim in claims:
        lines.append(
            "| "
            f"{claim['claim']} | "
            f"{claim['evidence']} | "
            f"{claim['support_level']} | "
            f"{claim['caveat']} |"
        )
    lines.extend(
        [
            "",
            "## Startup Floor",
            "",
            "| GPU Mem | Status | Phase | Available KV Memory | Interpretation |",
            "| ---: | --- | --- | ---: | --- |",
        ]
    )
    failure = startup["failure"]
    if failure is not None:
        lines.append(
            "| "
            f"{_fmt(failure['gpu_memory_utilization'])} | "
            f"`{failure['status']}` | "
            f"{failure['failure_phase']} | "
            f"{_fmt(failure['available_kv_cache_memory_gib'], ' GiB')} | "
            f"{failure['interpretation']} |"
        )
    lines.extend(
        [
            "| "
            f"{_fmt(startup['min_successful_gpu_memory_utilization'])} | "
            "`measured` | "
            "paired server/Async cache-control benchmark | "
            "n/a | "
            "Lowest successful budget point in the replicated curve. |",
            "",
            "## Successful Budget Curve",
            "",
            (
                "| Profile | GPU Mem | KV Tokens | Max Concurrency | Prompt Pressure | "
                "Hit Rate | Throughput Ratio | p95 Latency Ratio | Trials |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"`{row['profile_label']}` | "
            f"{_fmt(row['gpu_memory_utilization'])} | "
            f"{_fmt_int(row['kv_cache_size_tokens'])} | "
            f"{_fmt(row['max_concurrency_for_request'])} | "
            f"{_fmt(row['estimated_prompt_token_pressure_ratio'])} | "
            f"{_fmt(row['prefix_cache_hit_rate_pct'], '%')} | "
            f"{_fmt(row['throughput_ratio'], 'x')} | "
            f"{_fmt(row['p95_latency_ratio'], 'x')} | "
            f"{row['trial_count']} |"
        )
    lines.extend(
        [
            "",
            "## Reading The Table",
            "",
            (
                "Prompt pressure is estimated prompt tokens divided by the parsed "
                "server GPU KV-cache token capacity. Throughput and latency ratios "
                "are cache-on divided by cache-off for the same prompt profile, "
                "request count, and budget."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "status",
        "phase_order",
        "profile_label",
        "prompt_profile",
        "gpu_memory_utilization",
        "request_count",
        "max_new_tokens",
        "trial_count",
        "prompt_tokens_mean",
        "estimated_prompt_tokens",
        "kv_cache_size_tokens",
        "max_concurrency_for_request",
        "max_num_batched_tokens",
        "estimated_prompt_token_pressure_ratio",
        "prefix_cache_hit_rate_pct",
        "prefix_cache_queries",
        "prefix_cache_hits",
        "throughput_ratio",
        "p95_latency_ratio",
        "p95_first_content_ratio",
        "p95_stream_tpot_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pressure-curve-json",
        type=Path,
        default=DEFAULT_PRESSURE_CURVE_JSON,
    )
    parser.add_argument(
        "--failure-json",
        type=Path,
        default=DEFAULT_FAILURE_JSON,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report = build_report(args.pressure_curve_json, args.failure_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "kv-budget-report.json"
    csv_path = args.output_dir / "kv-budget-report.csv"
    markdown_path = args.output_dir / "kv-budget-report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, report["rows"])
    markdown_path.write_text(report["markdown"], encoding="utf-8")
    print(f"rows: {report['row_count']}")
    print(f"markdown: {_relative_display_path(markdown_path)}")


if __name__ == "__main__":
    main()
