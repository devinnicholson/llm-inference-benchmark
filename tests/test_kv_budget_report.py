from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_kv_budget_report import build_report


class KvBudgetReportTests(unittest.TestCase):
    def test_builds_report_from_pressure_curve_and_failure_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pressure_curve_json = root / "pressure.json"
            failure_json = root / "failure.json"
            pressure_curve_json.write_text(
                json.dumps(
                    {
                        "mode": "server-cache-pressure-curve",
                        "rows": [
                            {
                                "phase_order": "async_first",
                                "prompt_profile": "matched_unique_prefix_profile",
                                "gpu_memory_utilization_mean": 0.325,
                                "request_count": 32,
                                "max_new_tokens": 8,
                                "trial_count": 2,
                                "prompt_tokens_mean": 1000.0,
                                "estimated_prompt_tokens": 32000.0,
                                "on_server_gpu_kv_cache_size_tokens_mean": 4000.0,
                                "on_server_max_concurrency_for_request_mean": 4.0,
                                "max_num_batched_tokens_mean": 60640.0,
                                "estimated_prompt_token_pressure_ratio": 8.0,
                                "on_prefix_cache_hit_rate_pct_mean": 0.4,
                                "on_prefix_cache_queries_mean": 1000.0,
                                "on_prefix_cache_hits_mean": 4.0,
                                "throughput_ratio_mean": 0.98,
                                "p95_latency_ratio_mean": 1.02,
                                "p95_first_content_ratio_mean": 1.01,
                                "p95_stream_tpot_ratio_mean": 1.10,
                            },
                            {
                                "phase_order": "async_first",
                                "prompt_profile": "shared_prefix_profile",
                                "gpu_memory_utilization_mean": 0.325,
                                "request_count": 32,
                                "max_new_tokens": 8,
                                "trial_count": 2,
                                "prompt_tokens_mean": 1000.0,
                                "estimated_prompt_tokens": 32000.0,
                                "on_server_gpu_kv_cache_size_tokens_mean": 4000.0,
                                "on_server_max_concurrency_for_request_mean": 4.0,
                                "max_num_batched_tokens_mean": 60640.0,
                                "estimated_prompt_token_pressure_ratio": 8.0,
                                "on_prefix_cache_hit_rate_pct_mean": 96.0,
                                "on_prefix_cache_queries_mean": 1000.0,
                                "on_prefix_cache_hits_mean": 960.0,
                                "throughput_ratio_mean": 10.0,
                                "p95_latency_ratio_mean": 0.1,
                                "p95_first_content_ratio_mean": 0.09,
                                "p95_stream_tpot_ratio_mean": 0.3,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            failure_json.write_text(
                json.dumps(
                    {
                        "status": "failed_before_artifact_write",
                        "failure_phase": "AsyncLLM engine KV-cache initialization",
                        "gpu_memory_utilization": 0.3,
                        "prompt_profile": "matched_unique_prefix_profile",
                        "request_count": 32,
                        "output_tokens": 8,
                        "phase_order": "async_first",
                        "server_async_max_num_batched_tokens": 60640,
                        "available_kv_cache_memory_gib": -0.17,
                        "error_type": "ValueError",
                        "error_message": "No available memory for the cache blocks.",
                        "interpretation": "0.30 is below the startup floor.",
                    }
                ),
                encoding="utf-8",
            )

            report = build_report(pressure_curve_json, failure_json)

        self.assertEqual(report["mode"], "kv-budget-report")
        self.assertEqual(report["row_count"], 2)
        self.assertEqual(
            report["startup_floor"]["failed_gpu_memory_utilization"], 0.3
        )
        self.assertEqual(
            report["startup_floor"]["min_successful_gpu_memory_utilization"],
            0.325,
        )
        self.assertEqual(report["headline"]["shared_floor_throughput_ratio"], 10.0)
        self.assertEqual(report["rows"][0]["profile_label"], "matched_unique")
        self.assertIn("# KV-Budget Report", report["markdown"])
        self.assertIn("`gpu_memory_utilization=0.300` fails", report["markdown"])


if __name__ == "__main__":
    unittest.main()
