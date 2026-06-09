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

from build_server_cache_pressure_curve import build_pressure_curve


class ServerCachePressureCurveTests(unittest.TestCase):
    def test_builds_pressure_rows_by_profile_and_request_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "server-cache-control-absolute.json"
            source.write_text(
                json.dumps(
                    {
                        "mode": "vllm-server-async-cache-control-server-absolute",
                        "trial_rows": [
                            {
                                "phase_order": "async_first",
                                "prompt_profile": "shared_prefix",
                                "request_count": 16,
                                "max_new_tokens": 8,
                                "prompt_tokens_mean": 1000.0,
                                "on_server_gpu_kv_cache_size_tokens": 40000,
                                "on_server_max_concurrency_for_request": 40.0,
                                "max_num_batched_tokens": 60640,
                                "cache_on_div_off_server_output_tokens_per_second": 4.0,
                                "cache_on_div_off_server_p95_latency_ms": 0.25,
                                "cache_on_div_off_server_p95_first_content_ms": 0.2,
                                "cache_on_div_off_server_p95_stream_tpot_ms": 0.1,
                                "on_server_prefix_cache_counter_hit_rate_pct": 95.0,
                                "on_server_prefix_cache_counter_queries": 1000,
                                "on_server_prefix_cache_counter_hits": 950,
                            },
                            {
                                "phase_order": "async_first",
                                "prompt_profile": "shared_prefix",
                                "request_count": 40,
                                "max_new_tokens": 8,
                                "prompt_tokens_mean": 1000.0,
                                "on_server_gpu_kv_cache_size_tokens": 40000,
                                "on_server_max_concurrency_for_request": 40.0,
                                "max_num_batched_tokens": 60640,
                                "cache_on_div_off_server_output_tokens_per_second": 2.0,
                                "cache_on_div_off_server_p95_latency_ms": 0.5,
                                "cache_on_div_off_server_p95_first_content_ms": 0.4,
                                "cache_on_div_off_server_p95_stream_tpot_ms": 0.2,
                                "on_server_prefix_cache_counter_hit_rate_pct": 80.0,
                                "on_server_prefix_cache_counter_queries": 2000,
                                "on_server_prefix_cache_counter_hits": 1600,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = build_pressure_curve(source)

        self.assertEqual(payload["mode"], "server-cache-pressure-curve")
        self.assertEqual(payload["row_count"], 2)
        by_request_count = {row["request_count"]: row for row in payload["rows"]}
        self.assertEqual(
            by_request_count[16]["estimated_prompt_token_pressure_ratio"],
            0.4,
        )
        self.assertEqual(by_request_count[40]["throughput_ratio_mean"], 2.0)
        self.assertIn("| `async_first` | `shared_prefix` | 40 |", payload["markdown"])


if __name__ == "__main__":
    unittest.main()
