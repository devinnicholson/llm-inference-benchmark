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

from build_tensor_parallel_comparison import build_comparison


def _artifact(tp_size: int, throughput: float, latency: float) -> dict:
    gpu_lines = "\n".join(
        "NVIDIA L4, 23034, 22000, 580.95.05" for _ in range(tp_size)
    )
    server_command = ["vllm", "serve", "model"]
    if tp_size > 1:
        server_command.extend(["--tensor-parallel-size", str(tp_size)])
    row = {
        "scenario_id": "shared_out8_n16",
        "repeat_index": 0,
        "server_output_tokens_per_second": throughput,
        "server_p95_first_content_ms": latency * 0.9,
        "server_p95_latency_ms": latency,
        "server_p95_stream_tpot_ms": latency / 8,
        "async_output_tokens_per_second": throughput * 0.9,
        "async_p95_first_event_ms": latency,
        "async_p95_latency_ms": latency * 1.1,
        "async_p95_stream_tpot_ms": latency / 7,
    }
    return {
        "mode": "vllm-server-async-paired",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "modal_gpu": "L4" if tp_size == 1 else "L4:2",
        "tensor_parallel_size": tp_size,
        "request_counts": [16],
        "prompt_profiles": ["shared"],
        "output_tokens": [8],
        "repeats": 1,
        "scenario_seed": 42,
        "warmup_runs": 1,
        "phase_order": "async_first",
        "max_model_len": 3628,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.9,
        "async_enable_prefix_caching_configured": False,
        "server_prefix_caching_configured": "off",
        "vllm_version": "0.21.0",
        "nvidia_smi_before": gpu_lines,
        "server_command": server_command,
        "server_logs_head": [],
        "server_logs_tail": [
            f"Available KV cache memory: {4.0 * tp_size:.2f} GiB",
            f"GPU KV cache size: {80000 * tp_size:,} tokens",
            f"Maximum concurrency for 3,628 tokens per request: {22.0 * tp_size:.2f}x",
            *( ["Custom allreduce is disabled because your platform lacks GPU P2P capability"] if tp_size > 1 else [] ),
        ],
        "paired_runs": [row],
    }


class TensorParallelComparisonTests(unittest.TestCase):
    def test_builds_matched_topology_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single_path = root / "single.json"
            tensor_path = root / "tensor.json"
            single_path.write_text(json.dumps(_artifact(1, 10.0, 100.0)))
            tensor_path.write_text(json.dumps(_artifact(2, 12.0, 80.0)))

            payload = build_comparison(single_path, tensor_path)

        self.assertEqual(payload["mode"], "tensor-parallel-comparison")
        self.assertEqual(payload["topology"]["tensor_parallel_size"], 2)
        self.assertEqual(payload["capacity"]["tensor_div_single_kv_cache_tokens"], 2.0)
        server_throughput = next(
            row
            for row in payload["rows"]
            if row["backend"] == "server" and row["metric"] == "throughput"
        )
        self.assertAlmostEqual(server_throughput["tensor_div_single_median"], 1.2)
        self.assertTrue(server_throughput["favorable"])
        self.assertIn("NCCL fallback", payload["markdown"])

    def test_rejects_mismatched_configuration(self) -> None:
        single = _artifact(1, 10.0, 100.0)
        tensor = _artifact(2, 12.0, 80.0)
        tensor["scenario_seed"] = 99
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single_path = root / "single.json"
            tensor_path = root / "tensor.json"
            single_path.write_text(json.dumps(single))
            tensor_path.write_text(json.dumps(tensor))

            with self.assertRaisesRegex(ValueError, "scenario_seed"):
                build_comparison(single_path, tensor_path)


if __name__ == "__main__":
    unittest.main()
