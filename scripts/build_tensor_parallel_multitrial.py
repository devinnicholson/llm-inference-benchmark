#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import median
from typing import Any

from build_tensor_parallel_comparison import (
    MATCHED_FIELDS,
    METRICS,
    ROOT,
    _display_path,
    _paired_run_map,
    _parse_capacity,
    _percentile,
    _read_json,
    _validate_inputs,
)


DEFAULT_OUTPUT_DIR = ROOT / "results/tensor-parallel-qwen7b-l4-multitrial"
CROSS_TRIAL_FIELDS = tuple(
    field for field in MATCHED_FIELDS if field not in {"scenario_seed", "max_model_len"}
)


def _hierarchical_median_interval(
    trial_values: list[list[float]],
    *,
    seed: int = 20260807,
    samples: int = 20_000,
    confidence: float = 0.90,
) -> tuple[float, float]:
    if not trial_values or any(not values for values in trial_values):
        raise ValueError("hierarchical bootstrap requires non-empty trials")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        selected_trials = [rng.choice(trial_values) for _ in trial_values]
        observations: list[float] = []
        for values in selected_trials:
            observations.extend(rng.choice(values) for _ in values)
        estimates.append(median(observations))
    tail = (1.0 - confidence) / 2.0
    return _percentile(estimates, tail), _percentile(estimates, 1.0 - tail)


def _validate_cross_trial_configuration(
    artifacts: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    baseline_single = artifacts[0][0]
    baseline_tensor = artifacts[0][1]
    for trial_index, (single, tensor) in enumerate(artifacts):
        _validate_inputs(single, tensor)
        mismatches = [
            field
            for field in CROSS_TRIAL_FIELDS
            if single.get(field) != baseline_single.get(field)
            or tensor.get(field) != baseline_tensor.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"trial {trial_index} differs from the aggregate protocol: "
                + ", ".join(mismatches)
            )


def _aggregate_rows(
    artifacts: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    trial_maps = [
        (_paired_run_map(single), _paired_run_map(tensor))
        for single, tensor in artifacts
    ]
    scenario_ids = sorted(
        {key[0] for single_runs, _ in trial_maps for key in single_runs}
    )
    rows: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        for backend, metrics in METRICS.items():
            for metric, (field, unit, favorable_direction) in metrics.items():
                trial_ratios: list[list[float]] = []
                all_single: list[float] = []
                all_tensor: list[float] = []
                for single_runs, tensor_runs in trial_maps:
                    single_keys = sorted(
                        key for key in single_runs if key[0] == scenario_id
                    )
                    tensor_keys = sorted(
                        key for key in tensor_runs if key[0] == scenario_id
                    )
                    if single_keys != tensor_keys or not single_keys:
                        raise ValueError(
                            f"trial does not contain matched keys for {scenario_id}"
                        )
                    single_values = [
                        float(single_runs[key][field]) for key in single_keys
                    ]
                    tensor_values = [
                        float(tensor_runs[key][field]) for key in tensor_keys
                    ]
                    ratios = [
                        tensor_value / single_value
                        for single_value, tensor_value in zip(
                            single_values, tensor_values
                        )
                    ]
                    trial_ratios.append(ratios)
                    all_single.extend(single_values)
                    all_tensor.extend(tensor_values)

                flattened_ratios = [
                    ratio for ratios in trial_ratios for ratio in ratios
                ]
                trial_medians = [median(ratios) for ratios in trial_ratios]
                ratio_median = median(flattened_ratios)
                ci_low, ci_high = _hierarchical_median_interval(trial_ratios)
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
                        "independent_trials": len(trial_ratios),
                        "paired_observations": len(flattened_ratios),
                        "single_median": median(all_single),
                        "tensor_median": median(all_tensor),
                        "tensor_div_single_median": ratio_median,
                        "tensor_div_single_hierarchical_bootstrap_90_low": ci_low,
                        "tensor_div_single_hierarchical_bootstrap_90_high": ci_high,
                        "trial_median_ratio_min": min(trial_medians),
                        "trial_median_ratio_max": max(trial_medians),
                        "percent_change": (ratio_median - 1.0) * 100.0,
                        "favorable": favorable,
                    }
                )
    return rows


def _format_markdown(payload: dict[str, Any]) -> str:
    protocol = payload["protocol"]
    lines = [
        "# Replicated Tensor-Parallel Serving Study",
        "",
        (
            f"`{protocol['model_id']}` on one NVIDIA L4 versus two "
            "tensor-parallel L4s, with cache disabled on both execution paths."
        ),
        "",
        (
            f"Protocol: `{protocol['prompt_profiles'][0]}`, "
            f"n={protocol['request_counts'][0]}, output tokens={protocol['output_tokens'][0]}, "
            f"scheduler budget={protocol['max_num_batched_tokens']}. "
            f"The aggregate contains {payload['independent_trial_count']} independent "
            f"run/seed trials and {payload['paired_observation_count']} paired repeats."
        ),
        "",
        "## Capacity by independent trial",
        "",
        "| Seed | TP1 KV tokens | TP2 KV tokens | TP2 / TP1 | TP2 interconnect |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["capacity_trials"]:
        lines.append(
            f"| {row['scenario_seed']} | {row['single_kv_cache_tokens']:.0f} | "
            f"{row['tensor_kv_cache_tokens']:.0f} | "
            f"{row['tensor_div_single_kv_cache_tokens']:.3f}x | "
            f"{'NCCL fallback; no P2P/custom all-reduce' if row['tensor_p2p_unavailable'] else 'P2P available'} |"
        )
    lines.extend(
        [
            "",
            "## Replicated request-path results",
            "",
            "| Backend | Metric | TP1 median | TP2 median | TP2 / TP1 | Hierarchical 90% interval | Trial-median range |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| `{row['backend']}` | `{row['metric']}` | "
            f"{row['single_median']:.3f} | {row['tensor_median']:.3f} | "
            f"{row['tensor_div_single_median']:.3f}x | "
            f"[{row['tensor_div_single_hierarchical_bootstrap_90_low']:.3f}, "
            f"{row['tensor_div_single_hierarchical_bootstrap_90_high']:.3f}] | "
            f"[{row['trial_median_ratio_min']:.3f}, {row['trial_median_ratio_max']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "Throughput ratios above 1.0 favor TP2; latency-style ratios below 1.0 favor TP2.",
            "",
            "## Interpretation boundary",
            "",
            (
                "The hierarchical bootstrap resamples independent Modal run/seed trials "
                "and then repeats within each trial. The artifacts do not expose physical "
                "host identity. Two trials check whether the effect replicates, but they "
                "do not characterize the full Modal L4 fleet or other interconnect "
                "topologies. This remains a controlled serving study, not a "
                "production-traffic benchmark."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_multitrial_comparison(
    pairs: list[tuple[Path, Path]],
) -> dict[str, Any]:
    if len(pairs) < 2:
        raise ValueError("multitrial comparison requires at least two artifact pairs")
    artifacts = [(_read_json(single), _read_json(tensor)) for single, tensor in pairs]
    _validate_cross_trial_configuration(artifacts)
    scenario_seeds = [single["scenario_seed"] for single, _ in artifacts]
    if len(set(scenario_seeds)) != len(scenario_seeds):
        raise ValueError("multitrial comparison requires distinct scenario seeds")

    capacity_trials: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for (single_path, tensor_path), (single, tensor) in zip(pairs, artifacts):
        single_capacity = _parse_capacity(single)
        tensor_capacity = _parse_capacity(tensor)
        single_tokens = single_capacity["gpu_kv_cache_size_tokens"]
        tensor_tokens = tensor_capacity["gpu_kv_cache_size_tokens"]
        if single_tokens is None or tensor_tokens is None:
            raise ValueError("all artifacts must contain server KV-cache capacity")
        capacity_trials.append(
            {
                "scenario_seed": single["scenario_seed"],
                "single_max_model_len": single["max_model_len"],
                "tensor_max_model_len": tensor["max_model_len"],
                "single_kv_cache_tokens": single_tokens,
                "tensor_kv_cache_tokens": tensor_tokens,
                "tensor_div_single_kv_cache_tokens": tensor_tokens / single_tokens,
                "tensor_p2p_unavailable": tensor_capacity["p2p_unavailable"],
                "tensor_custom_all_reduce_disabled": tensor_capacity[
                    "custom_all_reduce_disabled"
                ],
            }
        )
        sources.append(
            {
                "scenario_seed": single["scenario_seed"],
                "single": _display_path(single_path),
                "tensor": _display_path(tensor_path),
            }
        )

    rows = _aggregate_rows(artifacts)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "mode": "tensor-parallel-multitrial-comparison",
        "sources": sources,
        "protocol": {
            field: artifacts[0][0].get(field) for field in CROSS_TRIAL_FIELDS
        },
        "independent_trial_count": len(artifacts),
        "paired_observation_count": sum(
            len(single["paired_runs"]) for single, _ in artifacts
        ),
        "scenario_seeds": scenario_seeds,
        "capacity_trials": capacity_trials,
        "rows": rows,
    }
    payload["markdown"] = _format_markdown(payload)
    return payload


def _parse_pair(value: str) -> tuple[Path, Path]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("pair must be SINGLE_JSON,TENSOR_JSON")
    return Path(parts[0]), Path(parts[1])


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
    parser.add_argument("--pair", action="append", type=_parse_pair, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    payload = build_multitrial_comparison(args.pair)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "tensor-parallel-multitrial.json"
    csv_path = args.output_dir / "tensor-parallel-multitrial.csv"
    markdown_path = args.output_dir / "tensor-parallel-multitrial.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, payload["rows"])
    markdown_path.write_text(payload["markdown"], encoding="utf-8")
    print(f"independent_trials: {payload['independent_trial_count']}")
    print(f"paired_observations: {payload['paired_observation_count']}")
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")
    print(f"markdown: {markdown_path}")


if __name__ == "__main__":
    main()
