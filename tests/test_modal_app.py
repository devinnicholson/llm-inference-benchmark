from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import modal_app


class ModalAppTests(unittest.TestCase):
    def test_isolated_window_summary_estimates_measured_cache_hit_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics_dir = Path(directory)
            (metrics_dir / "prefix-cache-isolated-metrics.json").write_text(
                json.dumps(_window_source_payload()),
                encoding="utf-8",
            )

            payload = modal_app._summarize_vllm_prefix_cache_isolated_window(
                metrics_dir,
            )

        self.assertEqual(payload["mode"], "vllm-prefix-cache-isolated-window-summary")
        self.assertEqual(payload["source_mode"], "vllm-prefix-cache-isolated-neutral-warmup")
        self.assertEqual(payload["scenario_count"], 2)
        self.assertEqual(payload["profile_control_row_count"], 1)

        comparison = payload["profile_control_rows"][0]
        self.assertAlmostEqual(
            comparison["shared_estimated_measured_cache_hit_rate_pct"],
            50.0,
        )
        self.assertAlmostEqual(
            comparison["control_estimated_measured_cache_hit_rate_pct"],
            8.0,
        )
        self.assertAlmostEqual(
            comparison["shared_minus_control_estimated_measured_cache_hit_rate_pct"],
            42.0,
        )
        self.assertIn("Mean shared-minus-control estimated", payload["markdown"])

    def test_estimated_window_rate_clips_to_valid_percentage(self) -> None:
        self.assertEqual(
            modal_app._estimated_window_rate_pct(
                warmup_rate_pct=90.0,
                after_rate_pct=10.0,
                warmup_tokens=100.0,
                measured_tokens=100.0,
            ),
            0.0,
        )
        self.assertEqual(
            modal_app._estimated_window_rate_pct(
                warmup_rate_pct=0.0,
                after_rate_pct=90.0,
                warmup_tokens=100.0,
                measured_tokens=100.0,
            ),
            100.0,
        )

    def test_prefix_cache_counter_delta_reports_window_rate(self) -> None:
        delta = modal_app._prefix_cache_counter_delta(
            {
                "total_requests": 3,
                "total_queries": 100,
                "total_hits": 10,
            },
            {
                "total_requests": 7,
                "total_queries": 300,
                "total_hits": 160,
            },
        )

        self.assertEqual(delta["requests"], 4)
        self.assertEqual(delta["queries"], 200)
        self.assertEqual(delta["hits"], 150)
        self.assertEqual(delta["hit_rate_pct"], 75.0)

    def test_prefix_cache_counter_hit_rate_handles_empty_baseline(self) -> None:
        self.assertEqual(
            modal_app._prefix_cache_counter_hit_rate_pct(hits=0, queries=0),
            0.0,
        )
        self.assertIsNone(
            modal_app._prefix_cache_counter_hit_rate_pct(hits=1, queries=0),
        )

    def test_isolated_stability_summary_reports_paired_counter_ci(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics_dir = Path(directory)
            (metrics_dir / "prefix-cache-isolated-metrics.json").write_text(
                json.dumps(
                    {
                        "mode": "vllm-prefix-cache-isolated-neutral-warmup",
                        "warmup_runs": 1,
                        "warmup_prompt_profile": "neutral_long",
                        "repeats": 2,
                        "phase_order": "cold_first",
                        "scenario_count": 2,
                        "paired_run_count": 4,
                        "remote_call_count": 4,
                    }
                ),
                encoding="utf-8",
            )
            modal_app._write_records_csv(
                metrics_dir / "prefix-cache-isolated-metrics-runs.csv",
                _stability_run_rows(),
            )

            payload = modal_app._summarize_vllm_prefix_cache_isolated_stability(
                metrics_dir,
            )

        comparison = payload["profile_control_rows"][0]
        self.assertEqual(
            comparison[
                "shared_minus_control_cache_counter_hit_rate_pct_paired_observation_count"
            ],
            2,
        )
        self.assertAlmostEqual(
            comparison["shared_minus_control_cache_counter_hit_rate_pct_mean"],
            52.5,
        )
        self.assertAlmostEqual(
            comparison["shared_minus_control_cache_counter_hit_rate_pct_min"],
            45.0,
        )
        self.assertAlmostEqual(
            comparison["shared_minus_control_cache_counter_hit_rate_pct_max"],
            60.0,
        )
        self.assertIsNotNone(
            payload["summary"][
                "mean_shared_minus_control_cache_counter_hit_rate_pct_bootstrap_mean_p05"
            ]
        )
        self.assertIn("90% bootstrap interval", payload["markdown"])


def _window_source_payload() -> dict:
    return {
        "mode": "vllm-prefix-cache-isolated-neutral-warmup",
        "warmup_runs": 1,
        "warmup_prompt_profile": "neutral_long",
        "scenario_count": 2,
        "paired_run_count": 2,
        "remote_call_count": 2,
        "paired_runs": [
            {
                "scenario_id": "matched_unique_prefix_out8_n2",
                "prompt_profile": "matched_unique_prefix",
                "request_count": 2,
                "max_new_tokens": 8,
                "repeat_index": 0,
                "isolation_call_index": 0,
                "prompt_tokens_mean": 50.0,
                "cache_prefix_cache_hit_rate_pct": 5.0,
                "cold_prefix_cache_hit_rate_pct": 0.0,
                "cache_to_cold_output_tokens_per_second_ratio": 1.2,
                "cache_to_cold_p95_latency_ms_ratio": 0.9,
            },
            {
                "scenario_id": "shared_prefix_long_out8_n2",
                "prompt_profile": "shared_prefix_long",
                "request_count": 2,
                "max_new_tokens": 8,
                "repeat_index": 0,
                "isolation_call_index": 1,
                "prompt_tokens_mean": 50.0,
                "cache_prefix_cache_hit_rate_pct": 30.0,
                "cold_prefix_cache_hit_rate_pct": 0.0,
                "cache_to_cold_output_tokens_per_second_ratio": 2.5,
                "cache_to_cold_p95_latency_ms_ratio": 0.7,
            },
        ],
        "remote_call_summaries": [
            {
                "isolation_call_index": 0,
                "warmup_prompt_profile": "neutral_long",
                "cache_warmup": {
                    "summaries": [{"total_prompt_tokens": 100.0}],
                },
                "cache_warmup_cache_metrics": [
                    {"prefix_cache_hit_rate_pct": 2.0},
                ],
                "cold_warmup_cache_metrics": [
                    {"prefix_cache_hit_rate_pct": 0.0},
                ],
            },
            {
                "isolation_call_index": 1,
                "warmup_prompt_profile": "neutral_long",
                "cache_warmup": {
                    "summaries": [{"total_prompt_tokens": 100.0}],
                },
                "cache_warmup_cache_metrics": [
                    {"prefix_cache_hit_rate_pct": 10.0},
                ],
                "cold_warmup_cache_metrics": [
                    {"prefix_cache_hit_rate_pct": 0.0},
                ],
            },
        ],
    }


def _stability_run_rows() -> list[dict]:
    rows = []
    values = [
        ("shared_prefix_long", 0, 50.0, 1.4, 0.8),
        ("matched_unique_prefix", 0, 5.0, 1.0, 1.0),
        ("shared_prefix_long", 1, 70.0, 1.6, 0.7),
        ("matched_unique_prefix", 1, 10.0, 1.1, 0.95),
    ]
    for profile, repeat_index, counter_hit, throughput_ratio, latency_ratio in values:
        rows.append(
            {
                "scenario_id": f"{profile}_out8_n2",
                "prompt_profile": profile,
                "request_count": 2,
                "max_new_tokens": 8,
                "repeat_index": repeat_index,
                "prompt_tokens_mean": 1200.0,
                "cold_prefix_cache_hit_rate_pct": 0.0,
                "cache_prefix_cache_hit_rate_pct": counter_hit / 2.0,
                "cache_to_cold_prefix_cache_hit_rate_pct_delta": counter_hit / 2.0,
                "cold_prefix_cache_counter_queries": 1000,
                "cache_prefix_cache_counter_queries": 1000,
                "cold_prefix_cache_counter_hits": 0,
                "cache_prefix_cache_counter_hits": int(counter_hit * 10),
                "cold_prefix_cache_counter_hit_rate_pct": 0.0,
                "cache_prefix_cache_counter_hit_rate_pct": counter_hit,
                "cache_to_cold_prefix_cache_counter_hit_rate_pct_delta": counter_hit,
                "cache_to_cold_output_tokens_per_second_ratio": throughput_ratio,
                "cache_to_cold_p95_latency_ms_ratio": latency_ratio,
                "cache_to_cold_p95_stream_tpot_ms_ratio": latency_ratio,
            }
        )
    return rows


if __name__ == "__main__":
    unittest.main()
