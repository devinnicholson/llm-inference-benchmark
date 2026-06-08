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


if __name__ == "__main__":
    unittest.main()
