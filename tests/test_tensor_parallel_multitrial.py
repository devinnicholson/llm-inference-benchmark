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

from build_tensor_parallel_multitrial import build_multitrial_comparison


def _artifact(tp_size: int, seed: int, ratios: list[float]) -> dict:
    gpu_lines = "\n".join("NVIDIA L4, 23034, 22000" for _ in range(tp_size))
    command = ["vllm", "serve", "model"]
    if tp_size > 1:
        command.extend(["--tensor-parallel-size", str(tp_size)])
    paired_runs = []
    for index, ratio in enumerate(ratios):
        scale = ratio if tp_size > 1 else 1.0
        paired_runs.append(
            {
                "scenario_id": "shared_out8_n16",
                "repeat_index": index,
                "server_output_tokens_per_second": 10.0 * scale,
                "server_p95_first_content_ms": 100.0 / scale,
                "server_p95_latency_ms": 110.0 / scale,
                "server_p95_stream_tpot_ms": 12.0 / scale,
                "async_output_tokens_per_second": 9.0 * scale,
                "async_p95_first_event_ms": 105.0 / scale,
                "async_p95_latency_ms": 115.0 / scale,
                "async_p95_stream_tpot_ms": 13.0 / scale,
            }
        )
    return {
        "mode": "vllm-server-async-paired",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "tensor_parallel_size": tp_size,
        "request_counts": [16],
        "prompt_profiles": ["shared"],
        "output_tokens": [8],
        "repeats": len(ratios),
        "scenario_seed": seed,
        "warmup_runs": 1,
        "phase_order": "async_first",
        "max_model_len": 3600 + seed % 10,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.9,
        "async_enable_prefix_caching_configured": False,
        "server_prefix_caching_configured": "off",
        "vllm_version": "0.21.0",
        "nvidia_smi_before": gpu_lines,
        "server_command": command,
        "server_logs_head": [],
        "server_logs_tail": [
            f"GPU KV cache size: {80000 * tp_size:,} tokens",
            f"Maximum concurrency for 3,600 tokens per request: {22 * tp_size:.2f}x",
            *( ["Custom allreduce is disabled because your platform lacks GPU P2P capability"] if tp_size > 1 else [] ),
        ],
        "paired_runs": paired_runs,
    }


class TensorParallelMultitrialTests(unittest.TestCase):
    def test_aggregates_independent_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = []
            for seed, ratios in ((101, [1.2, 1.3]), (202, [1.4, 1.5])):
                single = root / f"single-{seed}.json"
                tensor = root / f"tensor-{seed}.json"
                single.write_text(json.dumps(_artifact(1, seed, ratios)))
                tensor.write_text(json.dumps(_artifact(2, seed, ratios)))
                pairs.append((single, tensor))

            payload = build_multitrial_comparison(pairs)

        self.assertEqual(payload["independent_trial_count"], 2)
        self.assertEqual(payload["paired_observation_count"], 4)
        throughput = next(
            row
            for row in payload["rows"]
            if row["backend"] == "server" and row["metric"] == "throughput"
        )
        self.assertAlmostEqual(throughput["tensor_div_single_median"], 1.35)
        self.assertEqual(throughput["trial_median_ratio_min"], 1.25)
        self.assertEqual(throughput["trial_median_ratio_max"], 1.45)
        self.assertIn("hierarchical bootstrap", payload["markdown"])
        self.assertIn("run/seed trials", payload["markdown"])
        self.assertIn("do not expose physical host identity", payload["markdown"])

    def test_requires_two_trials(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_multitrial_comparison([])

    def test_rejects_duplicate_trial_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = []
            for suffix in ("a", "b"):
                single = root / f"single-{suffix}.json"
                tensor = root / f"tensor-{suffix}.json"
                single.write_text(json.dumps(_artifact(1, 101, [1.2, 1.3])))
                tensor.write_text(json.dumps(_artifact(2, 101, [1.2, 1.3])))
                pairs.append((single, tensor))

            with self.assertRaisesRegex(ValueError, "distinct scenario seeds"):
                build_multitrial_comparison(pairs)


if __name__ == "__main__":
    unittest.main()
