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
    generate_workload,
    load_kv_cache_config,
    load_workload,
    simulate_fifo,
    simulate_scheduler,
    summarize_traces,
    workload_to_dict,
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

    def test_concurrent_fifo_reduces_queue_wait_and_overlaps_kv_cache(self) -> None:
        workload = load_workload(
            _write_workload(
                {
                    "requests": [
                        {
                            "id": "a",
                            "arrival_ms": 0,
                            "prompt_tokens": 2,
                            "output_tokens": 1,
                        },
                        {
                            "id": "b",
                            "arrival_ms": 0,
                            "prompt_tokens": 2,
                            "output_tokens": 1,
                        },
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

        serial = simulate_fifo(workload, model, kv_cache, max_concurrent_requests=1)
        concurrent = simulate_fifo(workload, model, kv_cache, max_concurrent_requests=2)
        concurrent_timeline = build_kv_cache_timeline(concurrent)

        self.assertGreater(serial[1].queue_wait_ms, 0)
        self.assertEqual(concurrent[1].queue_wait_ms, 0)
        self.assertEqual(max(point.active_bytes for point in concurrent_timeline), 12)

    def test_rejects_invalid_concurrency(self) -> None:
        workload = load_workload(
            _write_workload(
                {
                    "requests": [
                        {
                            "id": "a",
                            "arrival_ms": 0,
                            "prompt_tokens": 2,
                            "output_tokens": 1,
                        }
                    ]
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "max_concurrent_requests"):
            simulate_fifo(workload, max_concurrent_requests=0)

    def test_workload_generation_is_deterministic(self) -> None:
        first = generate_workload("mixed_bursty", requests=5, seed=11)
        second = generate_workload("mixed_bursty", requests=5, seed=11)

        self.assertEqual(workload_to_dict(first), workload_to_dict(second))
        self.assertEqual(len(first.requests), 5)

    def test_workload_generation_rejects_unknown_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown profile"):
            generate_workload("unknown", requests=1, seed=0)

    def test_shortest_cache_policy_selects_smaller_waiting_request(self) -> None:
        workload = load_workload(
            _write_workload(
                {
                    "requests": [
                        {
                            "id": "running",
                            "arrival_ms": 0,
                            "prompt_tokens": 100,
                            "output_tokens": 100,
                        },
                        {
                            "id": "large-waiting",
                            "arrival_ms": 1,
                            "prompt_tokens": 100,
                            "output_tokens": 1,
                        },
                        {
                            "id": "small-waiting",
                            "arrival_ms": 2,
                            "prompt_tokens": 1,
                            "output_tokens": 1,
                        },
                    ]
                }
            )
        )
        model = LatencyModel(
            scheduler_overhead_ms=0,
            tokenize_ms_per_prompt_token=0,
            prefill_ms_per_prompt_token=0,
            decode_ms_per_output_token=1,
            stream_ms_per_output_token=0,
        )

        fifo = simulate_scheduler(
            workload,
            model=model,
            max_concurrent_requests=1,
            scheduling_policy="fifo",
        )
        shortest_cache = simulate_scheduler(
            workload,
            model=model,
            max_concurrent_requests=1,
            scheduling_policy="shortest-cache",
        )

        self.assertEqual([trace.request_id for trace in fifo], ["running", "large-waiting", "small-waiting"])
        self.assertEqual(
            [trace.request_id for trace in shortest_cache],
            ["running", "small-waiting", "large-waiting"],
        )

    def test_deadline_policy_selects_earliest_deadline(self) -> None:
        workload = load_workload(
            _write_workload(
                {
                    "requests": [
                        {
                            "id": "running",
                            "arrival_ms": 0,
                            "prompt_tokens": 100,
                            "output_tokens": 100,
                        },
                        {
                            "id": "loose-deadline",
                            "arrival_ms": 1,
                            "prompt_tokens": 1,
                            "output_tokens": 1,
                            "deadline_ms": 1000,
                        },
                        {
                            "id": "tight-deadline",
                            "arrival_ms": 2,
                            "prompt_tokens": 1,
                            "output_tokens": 1,
                            "deadline_ms": 50,
                        },
                    ]
                }
            )
        )
        model = LatencyModel(
            scheduler_overhead_ms=0,
            tokenize_ms_per_prompt_token=0,
            prefill_ms_per_prompt_token=0,
            decode_ms_per_output_token=1,
            stream_ms_per_output_token=0,
        )

        traces = simulate_scheduler(
            workload,
            model=model,
            max_concurrent_requests=1,
            scheduling_policy="deadline",
        )

        self.assertEqual(
            [trace.request_id for trace in traces],
            ["running", "tight-deadline", "loose-deadline"],
        )

    def test_rejects_unknown_scheduler_policy(self) -> None:
        workload = load_workload(
            _write_workload(
                {
                    "requests": [
                        {
                            "id": "a",
                            "arrival_ms": 0,
                            "prompt_tokens": 1,
                            "output_tokens": 1,
                        }
                    ]
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "unknown scheduling_policy"):
            simulate_scheduler(workload, scheduling_policy="bad-policy")


def _write_workload(payload: dict) -> Path:
    return _write_json(payload)


def _write_json(payload: dict) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    with handle:
        json.dump(payload, handle)
    return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
