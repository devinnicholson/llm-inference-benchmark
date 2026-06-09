#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_ABSOLUTE_JSON = (
    ROOT
    / "results/modal-vllm-server-async-qwen15b-l4-n32-batched-tokens60640-nowarmup-server-cache-control-r1-r2"
    / "server-cache-control-absolute.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/server-cache-pressure-curve"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON root to be an object")
    return payload


def _mean_present(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return mean(numeric) if numeric else None


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def _relative_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _curve_row(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = group_rows[0]
    prompt_tokens_mean = _mean_present(
        [row.get("prompt_tokens_mean") for row in group_rows]
    )
    request_count = int(first["request_count"])
    estimated_prompt_tokens = (
        prompt_tokens_mean * request_count if prompt_tokens_mean is not None else None
    )
    kv_cache_size_tokens = _mean_present(
        [row.get("on_server_gpu_kv_cache_size_tokens") for row in group_rows]
    )
    return {
        "phase_order": first["phase_order"],
        "prompt_profile": first["prompt_profile"],
        "request_count": request_count,
        "max_new_tokens": int(first["max_new_tokens"]),
        "trial_count": len(group_rows),
        "prompt_tokens_mean": prompt_tokens_mean,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "on_server_gpu_kv_cache_size_tokens_mean": kv_cache_size_tokens,
        "estimated_prompt_token_pressure_ratio": (
            estimated_prompt_tokens / kv_cache_size_tokens
            if estimated_prompt_tokens is not None and kv_cache_size_tokens
            else None
        ),
        "on_server_max_concurrency_for_request_mean": _mean_present(
            [row.get("on_server_max_concurrency_for_request") for row in group_rows]
        ),
        "max_num_batched_tokens_mean": _mean_present(
            [row.get("max_num_batched_tokens") for row in group_rows]
        ),
        "throughput_ratio_mean": _mean_present(
            [
                row.get("cache_on_div_off_server_output_tokens_per_second")
                for row in group_rows
            ]
        ),
        "p95_latency_ratio_mean": _mean_present(
            [row.get("cache_on_div_off_server_p95_latency_ms") for row in group_rows]
        ),
        "p95_first_content_ratio_mean": _mean_present(
            [
                row.get("cache_on_div_off_server_p95_first_content_ms")
                for row in group_rows
            ]
        ),
        "p95_stream_tpot_ratio_mean": _mean_present(
            [
                row.get("cache_on_div_off_server_p95_stream_tpot_ms")
                for row in group_rows
            ]
        ),
        "on_prefix_cache_hit_rate_pct_mean": _mean_present(
            [
                row.get("on_server_prefix_cache_counter_hit_rate_pct")
                for row in group_rows
            ]
        ),
        "on_prefix_cache_queries_mean": _mean_present(
            [row.get("on_server_prefix_cache_counter_queries") for row in group_rows]
        ),
        "on_prefix_cache_hits_mean": _mean_present(
            [row.get("on_server_prefix_cache_counter_hits") for row in group_rows]
        ),
    }


def build_pressure_curve(server_absolute_json: Path) -> dict[str, Any]:
    payload = _read_json(server_absolute_json)
    trial_rows = payload.get("trial_rows")
    if not isinstance(trial_rows, list) or not trial_rows:
        raise ValueError("expected non-empty 'trial_rows' list")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in trial_rows:
        if not isinstance(row, dict):
            raise ValueError("expected trial rows to be objects")
        key = (
            row.get("phase_order"),
            row.get("prompt_profile"),
            row.get("request_count"),
            row.get("max_new_tokens"),
        )
        groups[key].append(row)

    rows = [
        _curve_row(group_rows)
        for _, group_rows in sorted(
            groups.items(),
            key=lambda item: (
                str(item[0][0]),
                str(item[0][1]),
                int(item[0][2]),
                int(item[0][3]),
            ),
        )
    ]
    return {
        "schema_version": 1,
        "mode": "server-cache-pressure-curve",
        "source_json": _relative_display_path(server_absolute_json),
        "row_count": len(rows),
        "rows": rows,
        "markdown": _format_markdown(rows, server_absolute_json),
    }


def _format_markdown(rows: list[dict[str, Any]], source_json: Path) -> str:
    lines = [
        "# Server Cache Pressure Curve",
        "",
        f"Source: `{_relative_display_path(source_json)}`",
        "",
        (
            "| Phase | Profile | n | Prompt Pressure | Hit Rate | "
            "Throughput Ratio | p95 Latency Ratio | TTFT Ratio | TPOT Ratio |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['phase_order']}` | "
            f"`{row['prompt_profile']}` | "
            f"{row['request_count']} | "
            f"{_fmt(row['estimated_prompt_token_pressure_ratio'])} | "
            f"{_fmt(row['on_prefix_cache_hit_rate_pct_mean'], '%')} | "
            f"{_fmt(row['throughput_ratio_mean'], 'x')} | "
            f"{_fmt(row['p95_latency_ratio_mean'], 'x')} | "
            f"{_fmt(row['p95_first_content_ratio_mean'], 'x')} | "
            f"{_fmt(row['p95_stream_tpot_ratio_mean'], 'x')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            (
                "Prompt pressure is estimated prompt tokens divided by the parsed "
                "server GPU KV-cache token capacity. Values near 1.0 are close to "
                "the reported KV capacity for the configured model length."
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
        "--server-absolute-json",
        type=Path,
        default=DEFAULT_SERVER_ABSOLUTE_JSON,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    payload = build_pressure_curve(args.server_absolute_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "server-cache-pressure-curve.json"
    csv_path = args.output_dir / "server-cache-pressure-curve.csv"
    markdown_path = args.output_dir / "server-cache-pressure-curve.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, payload["rows"])
    markdown_path.write_text(payload["markdown"], encoding="utf-8")

    print(f"rows: {payload['row_count']}")
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")
    print(f"markdown: {markdown_path}")


if __name__ == "__main__":
    main()
