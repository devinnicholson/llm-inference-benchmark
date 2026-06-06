from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llmbench import (
    KVCacheConfig,
    LatencyModel,
    build_kv_cache_timeline,
    load_kv_cache_config,
    load_workload,
    simulate_fifo,
    summarize_traces,
)


class WorkloadTests(unittest.TestCase):
    def test_loads_and_sorts_requests_by_arrival(self) -> None:
        path = _write_workload(
            {
                "name": "unit",
                "requests": [
                    {
                        "id": "second",
                        "arrival_ms": 10,
                        "prompt_tokens": 8,
                        "output_tokens": 2,
                    },
                    {
                        "id": "first",
                        "arrival_ms": 0,
                        "prompt_tokens": 4,
                        "output_tokens": 1,
                    },
                ],
            }
        )

        workload = load_workload(path)

        self.assertEqual([request.id for request in workload.requests], ["first", "second"])

    def test_rejects_non_positive_token_counts(self) -> None:
        path = _write_workload(
            {
                "requests": [
                    {
                        "id": "bad",
                        "arrival_ms": 0,
                        "prompt_tokens": 0,
                        "output_tokens": 1,
                    }
                ]
            }
        )

        with self.assertRaisesRegex(ValueError, "prompt_tokens"):
            load_workload(path)

    def test_fifo_simulation_accumulates_queue_wait(self) -> None:
        path = _write_workload(
            {
                "requests": [
                    {
                        "id": "a",
                        "arrival_ms": 0,
                        "prompt_tokens": 10,
                        "output_tokens": 10,
                    },
                    {
                        "id": "b",
                        "arrival_ms": 1,
                        "prompt_tokens": 10,
                        "output_tokens": 10,
                    },
                ]
            }
        )
        workload = load_workload(path)
        model = LatencyModel(
            scheduler_overhead_ms=1,
            tokenize_ms_per_prompt_token=0,
            prefill_ms_per_prompt_token=0,
            decode_ms_per_output_token=1,
            stream_ms_per_output_token=0,
        )

        traces = simulate_fifo(workload, model)

        self.assertEqual(traces[0].queue_wait_ms, 0)
        self.assertGreater(traces[1].queue_wait_ms, 0)
        self.assertEqual(traces[0].latency_ms, 11)

    def test_summary_reports_tail_latency_and_throughput(self) -> None:
        workload = load_workload(
            _write_workload(
                {
                    "requests": [
                        {
                            "id": "a",
                            "arrival_ms": 0,
                            "prompt_tokens": 10,
                            "output_tokens": 5,
                        }
                    ]
                }
            )
        )

        summary = summarize_traces(simulate_fifo(workload))

        self.assertEqual(summary["requests"], 1.0)
        self.assertEqual(summary["output_tokens"], 5.0)
        self.assertGreater(summary["p95_latency_ms"], 0)
        self.assertGreater(summary["peak_active_kv_cache_mib"], 0)
        self.assertGreater(summary["output_tokens_per_second"], 0)

    def test_kv_cache_estimate_is_attached_to_traces(self) -> None:
        workload = load_workload(
            _write_workload(
                {
                    "requests": [
                        {
                            "id": "a",
                            "arrival_ms": 0,
                            "prompt_tokens": 10,
                            "output_tokens": 5,
                        }
                    ]
                }
            )
        )
        kv_cache = KVCacheConfig(layers=2, kv_heads=4, head_dim=8, bytes_per_element=2)

        traces = simulate_fifo(workload, kv_cache=kv_cache)

        self.assertEqual(traces[0].kv_cache_bytes, 15 * 2 * 2 * 4 * 8 * 2)
        self.assertGreater(traces[0].kv_cache_mib, 0)

    def test_loads_kv_cache_config(self) -> None:
        path = _write_json(
            {
                "name": "tiny",
                "layers": 2,
                "kv_heads": 4,
                "head_dim": 8,
                "bytes_per_element": 2,
            }
        )

        config = load_kv_cache_config(path)

        self.assertEqual(config.name, "tiny")
        self.assertEqual(config.bytes_per_token, 2 * 2 * 4 * 8 * 2)

    def test_active_kv_cache_timeline_tracks_growth_and_release(self) -> None:
        workload = load_workload(
            _write_workload(
                {
                    "requests": [
                        {
                            "id": "a",
                            "arrival_ms": 0,
                            "prompt_tokens": 2,
                            "output_tokens": 2,
                        }
                    ]
                }
            )
        )
        model = LatencyModel(
            scheduler_overhead_ms=0,
            tokenize_ms_per_prompt_token=0,
            prefill_ms_per_prompt_token=0.5,
            decode_ms_per_output_token=1,
            stream_ms_per_output_token=0.5,
        )
        kv_cache = KVCacheConfig(layers=1, kv_heads=1, head_dim=1, bytes_per_element=1)

        timeline = build_kv_cache_timeline(simulate_fifo(workload, model, kv_cache))

        self.assertEqual([point.active_bytes for point in timeline], [4, 6, 8, 0])
        self.assertEqual(max(point.active_bytes for point in timeline), 8)


def _write_workload(payload: dict) -> Path:
    return _write_json(payload)


def _write_json(payload: dict) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    with handle:
        json.dump(payload, handle)
    return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
