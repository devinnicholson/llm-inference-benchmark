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

from build_kv_budget_request_count_comparison import build_comparison


def _row(
    prompt_profile: str,
    request_count: int,
    pressure: float,
    hit_rate: float,
    throughput: float,
    latency: float,
    trial_count: int,
) -> dict[str, object]:
    return {
        "phase_order": "async_first",
        "prompt_profile": prompt_profile,
        "gpu_memory_utilization_mean": 0.325,
        "request_count": request_count,
        "max_new_tokens": 8,
        "trial_count": trial_count,
        "prompt_tokens_mean": 1000.0,
        "estimated_prompt_tokens": 1000.0 * request_count,
        "on_server_gpu_kv_cache_size_tokens_mean": 4000.0,
        "on_server_max_concurrency_for_request_mean": 4.0,
        "estimated_prompt_token_pressure_ratio": pressure,
        "on_prefix_cache_hit_rate_pct_mean": hit_rate,
        "throughput_ratio_mean": throughput,
        "p95_latency_ratio_mean": latency,
        "p95_first_content_ratio_mean": latency,
        "p95_stream_tpot_ratio_mean": 0.5,
    }


class KvBudgetRequestCountComparisonTests(unittest.TestCase):
    def test_compares_low_and_high_request_counts_by_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low_json = root / "low.json"
            high_json = root / "high.json"
            low_json.write_text(
                json.dumps(
                    {
                        "rows": [
                            _row(
                                "matched_unique_prefix_profile",
                                16,
                                4.0,
                                0.4,
                                0.95,
                                1.05,
                                1,
                            ),
                            _row(
                                "shared_prefix_profile",
                                16,
                                4.0,
                                93.0,
                                4.5,
                                0.2,
                                1,
                            ),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            high_json.write_text(
                json.dumps(
                    {
                        "rows": [
                            _row(
                                "matched_unique_prefix_profile",
                                32,
                                8.0,
                                0.4,
                                0.98,
                                1.02,
                                2,
                            ),
                            _row(
                                "shared_prefix_profile",
                                32,
                                8.0,
                                96.0,
                                10.0,
                                0.1,
                                2,
                            ),
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = build_comparison(
                low_pressure_curve_json=low_json,
                high_pressure_curve_json=high_json,
                low_label="n16",
                high_label="n32",
                gpu_memory_utilization=0.325,
            )

        self.assertEqual(payload["mode"], "kv-budget-request-count-comparison")
        self.assertEqual(payload["row_count"], 4)
        self.assertEqual(payload["delta_count"], 2)
        by_profile = {row["profile_label"]: row for row in payload["deltas"]}
        self.assertEqual(by_profile["shared_prefix"]["prompt_pressure_ratio"], 2.0)
        self.assertEqual(
            by_profile["shared_prefix"]["throughput_ratio_delta"], 5.5
        )
        self.assertIn("# KV-Budget Request-Count Comparison", payload["markdown"])
        self.assertIn("n=16", payload["markdown"])
        self.assertIn("n=32", payload["markdown"])


if __name__ == "__main__":
    unittest.main()
