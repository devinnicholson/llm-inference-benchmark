from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import modal_app


class ModalAppTests(unittest.TestCase):
    def test_prefix_cache_paired_gpu_selector_accepts_supported_gpus(self) -> None:
        self.assertIs(
            modal_app._select_vllm_prefix_cache_paired_remote("T4"),
            modal_app.run_vllm_prefix_cache_paired_remote,
        )
        self.assertIs(
            modal_app._select_vllm_prefix_cache_paired_remote("l4"),
            modal_app.run_vllm_prefix_cache_paired_l4_remote,
        )
        with self.assertRaisesRegex(ValueError, "T4, L4"):
            modal_app._select_vllm_prefix_cache_paired_remote("A100")

    def test_capacity_diagnostic_gpu_selector_accepts_supported_gpus(self) -> None:
        self.assertIs(
            modal_app._select_vllm_capacity_diagnostic_remote("T4"),
            modal_app.run_vllm_capacity_diagnostic_remote,
        )
        self.assertIs(
            modal_app._select_vllm_capacity_diagnostic_remote("l4"),
            modal_app.run_vllm_capacity_diagnostic_l4_remote,
        )
        with self.assertRaisesRegex(ValueError, "T4, L4"):
            modal_app._select_vllm_capacity_diagnostic_remote("A100")

    def test_server_async_paired_gpu_selector_accepts_supported_gpus(self) -> None:
        self.assertIs(
            modal_app._select_vllm_server_async_paired_remote("T4"),
            modal_app.run_vllm_server_async_paired_remote,
        )
        self.assertIs(
            modal_app._select_vllm_server_async_paired_remote("l4"),
            modal_app.run_vllm_server_async_paired_l4_remote,
        )
        with self.assertRaisesRegex(ValueError, "T4, L4"):
            modal_app._select_vllm_server_async_paired_remote("A100")

    def test_resolves_max_num_batched_tokens_override(self) -> None:
        self.assertEqual(
            modal_app._resolve_max_num_batched_tokens(
                default_value=121280,
                override_value=0,
                label="test_override",
            ),
            (121280, "default"),
        )
        self.assertEqual(
            modal_app._resolve_max_num_batched_tokens(
                default_value=121280,
                override_value=60640,
                label="test_override",
            ),
            (60640, "override"),
        )
        with self.assertRaisesRegex(ValueError, "test_override"):
            modal_app._resolve_max_num_batched_tokens(
                default_value=121280,
                override_value=-1,
                label="test_override",
            )

    def test_builds_vllm_server_prefix_caching_args(self) -> None:
        self.assertEqual(
            modal_app._vllm_server_prefix_caching_args("default"),
            [],
        )
        self.assertEqual(
            modal_app._vllm_server_prefix_caching_args("on"),
            ["--enable-prefix-caching"],
        )
        self.assertEqual(
            modal_app._vllm_server_prefix_caching_args("off"),
            ["--no-enable-prefix-caching"],
        )
        self.assertEqual(
            modal_app._normalize_server_prefix_caching_choice("disabled"),
            "off",
        )
        with self.assertRaisesRegex(ValueError, "default, on, or off"):
            modal_app._vllm_server_prefix_caching_args("maybe")

    def test_parses_vllm_engine_capacity_log_metrics(self) -> None:
        metrics = modal_app._parse_vllm_engine_capacity_log_metrics(
            {
                "captured_logs": [
                    "INFO Using max model len 3790",
                    "INFO Available KV cache memory: 0.5 GiB",
                    "INFO GPU KV cache size: 18,854 tokens",
                    "INFO Maximum concurrency for 3,790 tokens per request: 4.97x",
                ],
                "stdout": "",
                "stderr": "",
            }
        )

        self.assertEqual(metrics["max_model_len"], 3790)
        self.assertEqual(metrics["gpu_kv_cache_size_tokens"], 18854)
        self.assertEqual(metrics["max_concurrency_request_tokens"], 3790)
        self.assertEqual(metrics["max_concurrency_for_request"], 4.97)
        self.assertEqual(metrics["available_kv_cache_memory_gib"], 0.5)

    def test_parses_vllm_server_log_metrics(self) -> None:
        metrics = modal_app._parse_vllm_server_log_metrics(
            [
                "INFO non-default args: {'max_num_batched_tokens': 60640}",
                "INFO GPU KV cache size: 158,624 tokens",
                "INFO Maximum concurrency for 3,648 tokens per request: 43.48x",
                "INFO Available KV cache memory: 4.24 GiB",
                "INFO config: enable_prefix_caching=True, enable_chunked_prefill=True",
                (
                    "INFO Engine 000: Avg prompt throughput: 10464.5 tokens/s, "
                    "Avg generation throughput: 28.1 tokens/s, Running: 0 reqs, "
                    "Waiting: 0 reqs, GPU KV cache usage: 0.0%, "
                    "Prefix cache hit rate: 65.5%"
                ),
            ]
        )

        self.assertTrue(metrics["enable_prefix_caching"])
        self.assertEqual(metrics["max_num_batched_tokens"], 60640)
        self.assertEqual(metrics["gpu_kv_cache_size_tokens"], 158624)
        self.assertEqual(metrics["max_concurrency_request_tokens"], 3648)
        self.assertEqual(metrics["max_concurrency_for_request"], 43.48)
        self.assertEqual(metrics["available_kv_cache_memory_gib"], 4.24)
        self.assertEqual(metrics["latest_prefix_cache_hit_rate_pct"], 65.5)
        self.assertEqual(metrics["runtime_metric_count"], 1)

    def test_summarizes_server_async_cache_control_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            off_dir = root / "off"
            on_dir = root / "on"
            off_dir.mkdir()
            on_dir.mkdir()
            _write_paired_server_async_payload(
                off_dir,
                configured_mode="off",
                command_flag="--no-enable-prefix-caching",
                observed_cache=False,
                hit_rate_pct=0.0,
            )
            _write_paired_server_async_payload(
                on_dir,
                configured_mode="on",
                command_flag="--enable-prefix-caching",
                observed_cache=True,
                hit_rate_pct=74.1,
            )

            summary = modal_app._summarize_vllm_server_async_log_metrics(
                [off_dir, on_dir]
            )

        self.assertEqual(
            summary["server_enable_prefix_caching_observed_false_count"],
            1,
        )
        self.assertEqual(
            summary["server_enable_prefix_caching_observed_true_count"],
            1,
        )
        self.assertEqual(
            summary["server_prefix_caching_configured_modes"],
            ["off", "on"],
        )
        self.assertTrue(summary["any_server_command_has_explicit_prefix_cache_flag"])
        self.assertFalse(summary["all_server_enable_prefix_caching_observed"])
        self.assertFalse(summary["all_server_disable_prefix_caching_observed"])

    def test_compares_server_async_cache_control_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dirs = []
            for name, phase_order, mode, flag, observed, hit, throughput, latency in (
                (
                    "off-async",
                    "async_first",
                    "off",
                    "--no-enable-prefix-caching",
                    False,
                    0.0,
                    1.0,
                    1.1,
                ),
                (
                    "off-server",
                    "server_first",
                    "off",
                    "--no-enable-prefix-caching",
                    False,
                    0.0,
                    0.9,
                    1.2,
                ),
                (
                    "on-async",
                    "async_first",
                    "on",
                    "--enable-prefix-caching",
                    True,
                    70.0,
                    10.0,
                    0.1,
                ),
                (
                    "on-server",
                    "server_first",
                    "on",
                    "--enable-prefix-caching",
                    True,
                    80.0,
                    9.0,
                    0.2,
                ),
            ):
                path = root / name
                path.mkdir()
                _write_paired_server_async_payload(
                    path,
                    configured_mode=mode,
                    command_flag=flag,
                    observed_cache=observed,
                    hit_rate_pct=hit,
                    phase_order=phase_order,
                    throughput_ratio=throughput,
                    latency_ratio=latency,
                    tpot_ratio=latency / 2,
                )
                dirs.append(path)

            payload = modal_app._compare_vllm_server_async_cache_control_matrix(
                dirs
            )

        self.assertEqual(payload["row_count"], 4)
        self.assertEqual(payload["cache_modes"], ["off", "on"])
        self.assertEqual(payload["phase_orders"], ["async_first", "server_first"])
        self.assertEqual(len(payload["phase_order_cache_contrasts"]), 2)
        async_first = {
            row["phase_order"]: row
            for row in payload["phase_order_cache_contrasts"]
        }["async_first"]
        self.assertEqual(async_first["cache_on_minus_off_throughput_ratio"], 9.0)
        self.assertAlmostEqual(
            async_first["cache_on_div_off_latency_ratio"],
            0.1 / 1.1,
        )
        mode_summary = {
            row["server_prefix_caching_configured"]: row
            for row in payload["cache_mode_summary"]
        }
        self.assertAlmostEqual(
            mode_summary["off"]["server_first_minus_async_first_throughput_ratio"],
            -0.1,
        )
        self.assertAlmostEqual(
            mode_summary["on"]["server_first_minus_async_first_latency_ratio"],
            0.1,
        )

    def test_aggregates_server_async_cache_control_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compare_dirs = []
            for trial_index, async_delta, server_delta in (
                (1, 9.0, 10.0),
                (2, 11.0, 12.0),
            ):
                compare_dir = root / f"trial-{trial_index}"
                compare_dir.mkdir()
                _write_cache_control_compare_payload(
                    compare_dir,
                    async_delta=async_delta,
                    server_delta=server_delta,
                )
                compare_dirs.append(compare_dir)

            payload = modal_app._aggregate_vllm_server_async_cache_control_trials(
                compare_dirs
            )

        self.assertEqual(payload["trial_count"], 2)
        self.assertEqual(len(payload["trial_rows"]), 4)
        self.assertTrue(payload["all_off_server_disable_observed"])
        self.assertTrue(payload["all_on_server_enable_observed"])
        summary = {
            (row["phase_order"], row["metric"]): row
            for row in payload["summary"]
        }
        self.assertEqual(
            summary[
                ("async_first", "cache_on_minus_off_throughput_ratio")
            ]["mean"],
            10.0,
        )
        self.assertEqual(
            summary[
                ("server_first", "cache_on_minus_off_throughput_ratio")
            ]["mean"],
            11.0,
        )
        self.assertEqual(
            summary[
                ("async_first", "cache_on_minus_off_latency_ratio")
            ]["trial_count"],
            2,
        )

    def test_parses_vllm_server_cli_help_prefix_flags(self) -> None:
        parsed = modal_app._parse_vllm_server_cli_help(
            "  --enable-prefix-caching\n"
            "  --no-enable-prefix-caching\n"
            "  --prefix-caching-hash-algo {builtin,sha256}\n"
            "  --disable-log-requests\n"
        )

        self.assertEqual(
            parsed["prefix_related_flags"],
            [
                "--enable-prefix-caching",
                "--no-enable-prefix-caching",
                "--prefix-caching-hash-algo",
            ],
        )
        self.assertTrue(parsed["has_enable_prefix_caching_flag"])
        self.assertTrue(parsed["has_no_enable_prefix_caching_flag"])
        self.assertFalse(parsed["has_disable_prefix_caching_flag"])
        self.assertEqual(len(parsed["prefix_related_lines"]), 3)

    def test_formats_vllm_capacity_diagnostic_markdown(self) -> None:
        markdown = modal_app._format_vllm_capacity_diagnostic_markdown(
            {
                "model_id": "fake-model",
                "modal_gpu": "L4",
                "prompt_profiles": ["shared_prefix"],
                "warmup_prompt_profile": "neutral",
                "summary": [
                    {
                        "request_count": 16,
                        "enable_prefix_caching": False,
                        "max_model_len": 3790,
                        "max_num_batched_tokens": 60640,
                        "max_num_batched_tokens_source": "default",
                        "max_num_seqs": 16,
                        "gpu_kv_cache_size_tokens": 158540,
                        "available_kv_cache_memory_gib": 4.24,
                        "max_concurrency_for_request": 41.83,
                    },
                    {
                        "request_count": 32,
                        "enable_prefix_caching": False,
                        "max_model_len": 3790,
                        "max_num_batched_tokens": 121280,
                        "max_num_batched_tokens_source": "override",
                        "max_num_seqs": 32,
                        "gpu_kv_cache_size_tokens": 18854,
                        "available_kv_cache_memory_gib": 0.5,
                        "max_concurrency_for_request": 4.97,
                    },
                ],
            }
        )

        self.assertIn(
            "| 16 | False | 3790 | 60640 | default | 16 | 158540 | 4.24 | 41.83 |",
            markdown,
        )
        self.assertIn("2.000x", markdown)
        self.assertIn("0.119x", markdown)

    def test_snapshots_vllm_engine_capacity_from_config_fallback(self) -> None:
        engine_args = _Object(
            max_model_len=1024,
            max_num_batched_tokens=8192,
            max_num_seqs=8,
            gpu_memory_utilization=0.5,
        )
        engine = _Object(
            engine_core=_Object(
                engine_core_config=_Object(
                    model_config=_Object(max_model_len=1024),
                    scheduler_config=_Object(
                        max_num_batched_tokens=8192,
                        max_num_seqs=8,
                    ),
                    cache_config=_Object(
                        block_size=16,
                        num_gpu_blocks=512,
                        gpu_memory_utilization=0.5,
                    ),
                )
            )
        )

        snapshot = modal_app._snapshot_vllm_engine_capacity(
            engine=engine,
            engine_args=engine_args,
            phase_label="cache",
            engine_init_log_stats={"captured_logs": [], "stdout": "", "stderr": ""},
            requested_max_model_len=1024,
            requested_max_num_batched_tokens=8192,
            requested_max_num_seqs=8,
            requested_gpu_memory_utilization=0.5,
        )

        self.assertEqual(snapshot["gpu_kv_cache_size_tokens"], 8192)
        self.assertEqual(snapshot["derived_gpu_kv_cache_size_tokens"], 8192)
        self.assertEqual(snapshot["max_concurrency_for_request"], 8.0)
        self.assertEqual(snapshot["max_num_batched_tokens"], 8192)
        self.assertEqual(snapshot["max_num_seqs"], 8)

    def test_paired_rows_carry_engine_capacity_fields(self) -> None:
        cold_run = _paired_capacity_run("cold", gpu_tokens=18854)
        cache_run = _paired_capacity_run("cache", gpu_tokens=18854)

        paired = modal_app._make_vllm_prefix_cache_paired_rows(
            [cold_run],
            [cache_run],
        )

        row = paired[0]
        self.assertEqual(row["cold_engine_gpu_kv_cache_size_tokens"], 18854)
        self.assertEqual(row["cache_engine_gpu_kv_cache_size_tokens"], 18854)
        self.assertEqual(row["cold_engine_max_num_batched_tokens"], 121280)
        self.assertEqual(row["cache_engine_max_concurrency_for_request"], 4.97)

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

    def test_isolated_summary_source_paths_fall_back_to_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics_dir = Path(directory)
            (metrics_dir / "prefix-cache-isolated-metrics.partial.json").write_text(
                "{}",
                encoding="utf-8",
            )
            partial_json, partial_runs = (
                modal_app._prefix_cache_isolated_metrics_source_paths(metrics_dir)
            )

            self.assertEqual(
                partial_json.name,
                "prefix-cache-isolated-metrics.partial.json",
            )
            self.assertEqual(
                partial_runs.name,
                "prefix-cache-isolated-metrics-runs.partial.csv",
            )

            (metrics_dir / "prefix-cache-isolated-metrics.json").write_text(
                "{}",
                encoding="utf-8",
            )
            final_json, final_runs = (
                modal_app._prefix_cache_isolated_metrics_source_paths(metrics_dir)
            )

            self.assertEqual(final_json.name, "prefix-cache-isolated-metrics.json")
            self.assertEqual(final_runs.name, "prefix-cache-isolated-metrics-runs.csv")

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
        self.assertEqual(
            comparison[
                "shared_minus_control_cache_to_cold_p95_first_event_ratio_paired_observation_count"
            ],
            2,
        )
        self.assertAlmostEqual(
            comparison["shared_minus_control_cache_to_cold_p95_first_event_ratio_mean"],
            -0.4,
        )
        self.assertEqual(
            comparison[
                "shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_paired_observation_count"
            ],
            2,
        )
        self.assertAlmostEqual(
            comparison["shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_mean"],
            -0.175,
        )
        self.assertIsNotNone(
            payload["summary"][
                "mean_shared_minus_control_cache_to_cold_p95_first_event_ratio_bootstrap_mean_p05"
            ]
        )
        self.assertIn("90% bootstrap interval", payload["markdown"])
        self.assertIn("first-event/TTFT", payload["markdown"])
        self.assertIn("Stream TPOT", payload["markdown"])

    def test_isolated_stability_summary_accepts_custom_profile_pair(self) -> None:
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
                _stability_run_rows(
                    shared_profile="shared_prefix_long_variant",
                    control_profile="matched_unique_prefix_variant",
                ),
            )

            payload = modal_app._summarize_vllm_prefix_cache_isolated_stability(
                metrics_dir,
                shared_profile="shared_prefix_long_variant",
                control_profile="matched_unique_prefix_variant",
            )

        self.assertEqual(payload["profile_control_row_count"], 1)
        self.assertEqual(payload["shared_profile"], "shared_prefix_long_variant")
        self.assertEqual(payload["control_profile"], "matched_unique_prefix_variant")

    def test_isolated_metrics_merge_reindexes_chunk_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dirs = []
            for chunk_index, partial in enumerate((False, True)):
                metrics_dir = root / f"chunk-{chunk_index}"
                metrics_dir.mkdir()
                suffix = ".partial" if partial else ""
                (metrics_dir / f"prefix-cache-isolated-metrics{suffix}.json").write_text(
                    json.dumps(
                        {
                            "mode": "vllm-prefix-cache-isolated-neutral-warmup",
                            "model_id": "fake-model",
                            "warmup_runs": 1,
                            "warmup_prompt_profile": "neutral_long",
                            "repeats": 2,
                            "scenario_seed": 577 + chunk_index,
                            "phase_order": "cold_first",
                            "checkpoint_complete": not partial,
                            "scenario_count": 2,
                            "paired_run_count": 4,
                            "remote_call_count": 4,
                            "planned_remote_call_count": 4,
                        }
                    ),
                    encoding="utf-8",
                )
                modal_app._write_records_csv(
                    metrics_dir / f"prefix-cache-isolated-metrics-runs{suffix}.csv",
                    _stability_run_rows(),
                )
                source_dirs.append(metrics_dir)

            merged = modal_app._merge_vllm_prefix_cache_isolated_metrics(source_dirs)
            merged_dir = root / "merged"
            merged_dir.mkdir()
            (merged_dir / "prefix-cache-isolated-metrics.json").write_text(
                json.dumps(merged),
                encoding="utf-8",
            )
            modal_app._write_records_csv(
                merged_dir / "prefix-cache-isolated-metrics-runs.csv",
                merged["paired_runs"],
            )

            summary = modal_app._summarize_vllm_prefix_cache_isolated_stability(
                merged_dir,
            )

        self.assertEqual(merged["mode"], "vllm-prefix-cache-isolated-merge")
        self.assertEqual(merged["source_chunk_count"], 2)
        self.assertEqual(merged["paired_run_count"], 8)
        self.assertEqual(merged["repeats"], 4)
        self.assertFalse(merged["checkpoint_complete"])
        self.assertEqual(
            sorted({row["repeat_index"] for row in merged["paired_runs"]}),
            [0, 1, 2, 3],
        )
        comparison = summary["profile_control_rows"][0]
        self.assertEqual(
            comparison[
                "shared_minus_control_cache_counter_hit_rate_pct_paired_observation_count"
            ],
            4,
        )

    def test_variant_prompt_profiles_change_family_by_seed(self) -> None:
        first_family = modal_app._select_sweep_prompts(
            2,
            "shared_prefix_long_variant",
            variant_index=0,
        )
        second_family = modal_app._select_sweep_prompts(
            2,
            "shared_prefix_long_variant",
            variant_index=1,
        )
        first_prefix = first_family[0].split("\n\nTask", 1)[0]
        second_prompt_prefix = first_family[1].split("\n\nTask", 1)[0]
        self.assertEqual(first_prefix, second_prompt_prefix)
        self.assertNotEqual(first_family[0], second_family[0])

        controls = modal_app._select_sweep_prompts(
            2,
            "matched_unique_prefix_variant",
            variant_index=0,
        )
        self.assertNotEqual(
            controls[0].split(" ", 1)[0],
            controls[1].split(" ", 1)[0],
        )

    def test_no_repeat_variant_profiles_keep_n16_unique(self) -> None:
        shared = modal_app._select_sweep_prompts(
            16,
            "shared_prefix_long_no_repeat_variant",
            variant_index=577,
        )
        controls = modal_app._select_sweep_prompts(
            16,
            "matched_unique_prefix_no_repeat_variant",
            variant_index=577,
        )

        self.assertEqual(len(set(shared)), 16)
        self.assertEqual(len(set(controls)), 16)
        self.assertEqual(
            shared[0].split("\n\nTask", 1)[0],
            shared[15].split("\n\nTask", 1)[0],
        )
        self.assertNotEqual(
            controls[0].split(" ", 1)[0],
            controls[15].split(" ", 1)[0],
        )

    def test_extra_long_no_repeat_profiles_keep_n16_unique(self) -> None:
        base_shared = modal_app._select_sweep_prompts(
            16,
            "shared_prefix_long_no_repeat_variant",
            variant_index=577,
        )
        shared = modal_app._select_sweep_prompts(
            16,
            "shared_prefix_extra_long_no_repeat_variant",
            variant_index=577,
        )
        controls = modal_app._select_sweep_prompts(
            16,
            "matched_unique_prefix_extra_long_no_repeat_variant",
            variant_index=577,
        )
        neutral = modal_app._select_sweep_prompts(
            16,
            "neutral_extra_long",
            variant_index=577,
        )

        self.assertEqual(len(set(shared)), 16)
        self.assertEqual(len(set(controls)), 16)
        self.assertEqual(len(set(neutral)), 16)
        self.assertGreater(len(shared[0].split()), len(base_shared[0].split()) * 2)
        self.assertGreater(len(neutral[0].split()), len(base_shared[0].split()) * 2)
        self.assertEqual(
            shared[0].split("\n\nTask", 1)[0],
            shared[15].split("\n\nTask", 1)[0],
        )
        self.assertNotEqual(
            controls[0].split(" ", 1)[0],
            controls[15].split(" ", 1)[0],
        )

    def test_ultra_long_no_repeat_profiles_keep_n16_unique(self) -> None:
        extra_shared = modal_app._select_sweep_prompts(
            16,
            "shared_prefix_extra_long_no_repeat_variant",
            variant_index=577,
        )
        shared = modal_app._select_sweep_prompts(
            16,
            "shared_prefix_ultra_long_no_repeat_variant",
            variant_index=577,
        )
        controls = modal_app._select_sweep_prompts(
            16,
            "matched_unique_prefix_ultra_long_no_repeat_variant",
            variant_index=577,
        )
        neutral = modal_app._select_sweep_prompts(
            16,
            "neutral_ultra_long",
            variant_index=577,
        )

        self.assertEqual(len(set(shared)), 16)
        self.assertEqual(len(set(controls)), 16)
        self.assertEqual(len(set(neutral)), 16)
        self.assertGreater(len(shared[0].split()), len(extra_shared[0].split()))
        self.assertGreater(len(neutral[0].split()), len(extra_shared[0].split()))
        self.assertEqual(
            shared[0].split("\n\nTask", 1)[0],
            shared[15].split("\n\nTask", 1)[0],
        )
        self.assertNotEqual(
            controls[0].split(" ", 1)[0],
            controls[15].split(" ", 1)[0],
        )

    def test_mega_long_no_repeat_profiles_keep_n16_unique(self) -> None:
        ultra_shared = modal_app._select_sweep_prompts(
            16,
            "shared_prefix_ultra_long_no_repeat_variant",
            variant_index=577,
        )
        shared = modal_app._select_sweep_prompts(
            16,
            "shared_prefix_mega_long_no_repeat_variant",
            variant_index=577,
        )
        controls = modal_app._select_sweep_prompts(
            16,
            "matched_unique_prefix_mega_long_no_repeat_variant",
            variant_index=577,
        )
        neutral = modal_app._select_sweep_prompts(
            16,
            "neutral_mega_long",
            variant_index=577,
        )

        self.assertEqual(len(set(shared)), 16)
        self.assertEqual(len(set(controls)), 16)
        self.assertEqual(len(set(neutral)), 16)
        self.assertGreater(len(shared[0].split()), len(ultra_shared[0].split()))
        self.assertGreater(len(neutral[0].split()), len(ultra_shared[0].split()))
        self.assertEqual(
            shared[0].split("\n\nTask", 1)[0],
            shared[15].split("\n\nTask", 1)[0],
        )
        self.assertNotEqual(
            controls[0].split(" ", 1)[0],
            controls[15].split(" ", 1)[0],
        )

    def test_common_prefix_token_count_stops_at_first_difference(self) -> None:
        self.assertEqual(
            modal_app._common_prefix_token_count(
                [
                    [1, 2, 3, 4],
                    [1, 2, 5, 4],
                    [1, 2, 3, 4],
                ]
            ),
            2,
        )
        self.assertEqual(modal_app._common_prefix_token_count([]), 0)

    def test_prompt_audit_reports_more_shared_reusable_blocks(self) -> None:
        payload = modal_app._build_vllm_prefix_cache_prompt_audit_payload(
            tokenizer=_WhitespaceTokenizer(),
            hf_model="fake-model",
            request_count_values=[4],
            prompt_profile_values=[
                "shared_prefix_long_variant",
                "matched_unique_prefix_variant",
            ],
            output_token_values=[8],
            repeats=2,
            scenario_seed=577,
            kv_cache_block_size=8,
        )

        self.assertEqual(payload["scenario_count"], 4)
        self.assertEqual(payload["profile_control_row_count"], 2)
        self.assertGreater(
            payload["summary"]["shared_minus_control_common_prefix_blocks_mean"],
            0,
        )
        for row in payload["profile_control_rows"]:
            self.assertGreater(
                row["shared_common_prefix_full_blocks"],
                row["control_common_prefix_full_blocks"],
            )
            self.assertGreater(
                row["shared_minus_control_estimated_reusable_block_tokens"],
                0,
            )
        self.assertIn("Prefix-Cache Prompt Token Audit", payload["markdown"])

    def test_prompt_audit_reports_exact_duplicate_reuse(self) -> None:
        payload = modal_app._build_vllm_prefix_cache_prompt_audit_payload(
            tokenizer=_WhitespaceTokenizer(),
            hf_model="fake-model",
            request_count_values=[16],
            prompt_profile_values=[
                "shared_prefix_long_variant",
                "matched_unique_prefix_variant",
            ],
            output_token_values=[8],
            repeats=1,
            scenario_seed=577,
            kv_cache_block_size=8,
        )

        control = next(
            row
            for row in payload["scenario_rows"]
            if row["prompt_profile"] == "matched_unique_prefix_variant"
        )
        self.assertEqual(control["unique_prompt_count"], 8)
        self.assertEqual(control["exact_duplicate_prompt_repeated_count"], 8)
        self.assertGreater(
            control["estimated_exact_duplicate_reusable_block_tokens"],
            control["estimated_reusable_block_tokens"],
        )

        comparison = payload["profile_control_rows"][0]
        self.assertEqual(comparison["control_unique_prompt_count"], 8)
        self.assertGreater(
            comparison[
                "control_estimated_exact_duplicate_reusable_block_tokens"
            ],
            0,
        )

        duplicate_prompt_rows = [
            row
            for row in payload["prompt_rows"]
            if row["prompt_profile"] == "matched_unique_prefix_variant"
        ]
        self.assertEqual(
            duplicate_prompt_rows[0]["formatted_prompt_duplicate_group_size"],
            2,
        )
        self.assertEqual(
            duplicate_prompt_rows[8]["formatted_prompt_duplicate_ordinal"],
            1,
        )
        self.assertIn("Mean control exact-duplicate", payload["markdown"])

    def test_prompt_audit_reports_no_repeat_profiles(self) -> None:
        payload = modal_app._build_vllm_prefix_cache_prompt_audit_payload(
            tokenizer=_WhitespaceTokenizer(),
            hf_model="fake-model",
            request_count_values=[16],
            prompt_profile_values=[
                "shared_prefix_long_no_repeat_variant",
                "matched_unique_prefix_no_repeat_variant",
            ],
            output_token_values=[8],
            repeats=1,
            scenario_seed=577,
            kv_cache_block_size=8,
        )

        self.assertEqual(payload["profile_control_row_count"], 1)
        comparison = payload["profile_control_rows"][0]
        self.assertEqual(
            comparison["shared_profile"],
            "shared_prefix_long_no_repeat_variant",
        )
        self.assertEqual(
            comparison["control_profile"],
            "matched_unique_prefix_no_repeat_variant",
        )
        for row in payload["scenario_rows"]:
            self.assertEqual(row["unique_prompt_count"], 16)
            self.assertEqual(row["exact_duplicate_prompt_repeated_count"], 0)
            self.assertEqual(
                row["estimated_exact_duplicate_reusable_block_tokens"],
                0,
            )

    def test_prompt_audit_reports_extra_long_no_repeat_profiles(self) -> None:
        base_payload = modal_app._build_vllm_prefix_cache_prompt_audit_payload(
            tokenizer=_WhitespaceTokenizer(),
            hf_model="fake-model",
            request_count_values=[16],
            prompt_profile_values=[
                "shared_prefix_long_no_repeat_variant",
                "matched_unique_prefix_no_repeat_variant",
            ],
            output_token_values=[8],
            repeats=1,
            scenario_seed=577,
            kv_cache_block_size=8,
        )
        extra_payload = modal_app._build_vllm_prefix_cache_prompt_audit_payload(
            tokenizer=_WhitespaceTokenizer(),
            hf_model="fake-model",
            request_count_values=[16],
            prompt_profile_values=[
                "shared_prefix_extra_long_no_repeat_variant",
                "matched_unique_prefix_extra_long_no_repeat_variant",
            ],
            output_token_values=[8],
            repeats=1,
            scenario_seed=577,
            kv_cache_block_size=8,
        )

        self.assertEqual(extra_payload["profile_control_row_count"], 1)
        comparison = extra_payload["profile_control_rows"][0]
        self.assertEqual(
            comparison["shared_profile"],
            "shared_prefix_extra_long_no_repeat_variant",
        )
        self.assertEqual(
            comparison["control_profile"],
            "matched_unique_prefix_extra_long_no_repeat_variant",
        )
        self.assertGreater(
            extra_payload["summary"]["shared_common_prefix_full_blocks_mean"],
            base_payload["summary"]["shared_common_prefix_full_blocks_mean"],
        )
        self.assertGreater(
            comparison["shared_common_prefix_full_blocks"],
            comparison["control_common_prefix_full_blocks"],
        )
        for row in extra_payload["scenario_rows"]:
            self.assertEqual(row["unique_prompt_count"], 16)
            self.assertEqual(row["exact_duplicate_prompt_repeated_count"], 0)
            self.assertEqual(
                row["estimated_exact_duplicate_reusable_block_tokens"],
                0,
            )

    def test_prompt_audit_reports_ultra_long_no_repeat_profiles(self) -> None:
        extra_payload = modal_app._build_vllm_prefix_cache_prompt_audit_payload(
            tokenizer=_WhitespaceTokenizer(),
            hf_model="fake-model",
            request_count_values=[16],
            prompt_profile_values=[
                "shared_prefix_extra_long_no_repeat_variant",
                "matched_unique_prefix_extra_long_no_repeat_variant",
            ],
            output_token_values=[8],
            repeats=1,
            scenario_seed=577,
            kv_cache_block_size=8,
        )
        ultra_payload = modal_app._build_vllm_prefix_cache_prompt_audit_payload(
            tokenizer=_WhitespaceTokenizer(),
            hf_model="fake-model",
            request_count_values=[16],
            prompt_profile_values=[
                "shared_prefix_ultra_long_no_repeat_variant",
                "matched_unique_prefix_ultra_long_no_repeat_variant",
            ],
            output_token_values=[8],
            repeats=1,
            scenario_seed=577,
            kv_cache_block_size=8,
        )

        self.assertEqual(ultra_payload["profile_control_row_count"], 1)
        comparison = ultra_payload["profile_control_rows"][0]
        self.assertEqual(
            comparison["shared_profile"],
            "shared_prefix_ultra_long_no_repeat_variant",
        )
        self.assertEqual(
            comparison["control_profile"],
            "matched_unique_prefix_ultra_long_no_repeat_variant",
        )
        self.assertGreater(
            ultra_payload["summary"]["shared_common_prefix_full_blocks_mean"],
            extra_payload["summary"]["shared_common_prefix_full_blocks_mean"],
        )
        self.assertGreater(
            comparison["shared_common_prefix_full_blocks"],
            comparison["control_common_prefix_full_blocks"],
        )
        for row in ultra_payload["scenario_rows"]:
            self.assertEqual(row["unique_prompt_count"], 16)
            self.assertEqual(row["exact_duplicate_prompt_repeated_count"], 0)
            self.assertEqual(
                row["estimated_exact_duplicate_reusable_block_tokens"],
                0,
            )

    def test_prompt_audit_reports_mega_long_no_repeat_profiles(self) -> None:
        ultra_payload = modal_app._build_vllm_prefix_cache_prompt_audit_payload(
            tokenizer=_WhitespaceTokenizer(),
            hf_model="fake-model",
            request_count_values=[16],
            prompt_profile_values=[
                "shared_prefix_ultra_long_no_repeat_variant",
                "matched_unique_prefix_ultra_long_no_repeat_variant",
            ],
            output_token_values=[8],
            repeats=1,
            scenario_seed=577,
            kv_cache_block_size=8,
        )
        mega_payload = modal_app._build_vllm_prefix_cache_prompt_audit_payload(
            tokenizer=_WhitespaceTokenizer(),
            hf_model="fake-model",
            request_count_values=[16],
            prompt_profile_values=[
                "shared_prefix_mega_long_no_repeat_variant",
                "matched_unique_prefix_mega_long_no_repeat_variant",
            ],
            output_token_values=[8],
            repeats=1,
            scenario_seed=577,
            kv_cache_block_size=8,
        )

        self.assertEqual(mega_payload["profile_control_row_count"], 1)
        comparison = mega_payload["profile_control_rows"][0]
        self.assertEqual(
            comparison["shared_profile"],
            "shared_prefix_mega_long_no_repeat_variant",
        )
        self.assertEqual(
            comparison["control_profile"],
            "matched_unique_prefix_mega_long_no_repeat_variant",
        )
        self.assertGreater(
            mega_payload["summary"]["shared_common_prefix_full_blocks_mean"],
            ultra_payload["summary"]["shared_common_prefix_full_blocks_mean"],
        )
        self.assertGreater(
            comparison["shared_common_prefix_full_blocks"],
            comparison["control_common_prefix_full_blocks"],
        )
        for row in mega_payload["scenario_rows"]:
            self.assertEqual(row["unique_prompt_count"], 16)
            self.assertEqual(row["exact_duplicate_prompt_repeated_count"], 0)
            self.assertEqual(
                row["estimated_exact_duplicate_reusable_block_tokens"],
                0,
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


def _stability_run_rows(
    shared_profile: str = "shared_prefix_long",
    control_profile: str = "matched_unique_prefix",
) -> list[dict]:
    rows = []
    values = [
        (shared_profile, 0, 50.0, 1.4, 0.6, 0.8, 0.9),
        (control_profile, 0, 5.0, 1.0, 1.0, 1.0, 1.1),
        (shared_profile, 1, 70.0, 1.6, 0.55, 0.7, 0.85),
        (control_profile, 1, 10.0, 1.1, 0.95, 0.95, 1.0),
    ]
    for (
        profile,
        repeat_index,
        counter_hit,
        throughput_ratio,
        first_event_ratio,
        latency_ratio,
        tpot_ratio,
    ) in values:
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
                "cache_to_cold_p95_first_event_ms_ratio": first_event_ratio,
                "cache_to_cold_p95_latency_ms_ratio": latency_ratio,
                "cache_to_cold_p95_stream_tpot_ms_ratio": tpot_ratio,
            }
        )
    return rows


def _write_paired_server_async_payload(
    directory: Path,
    configured_mode: str,
    command_flag: str,
    observed_cache: bool,
    hit_rate_pct: float,
    phase_order: str = "async_first",
    throughput_ratio: float = 1.0,
    latency_ratio: float = 1.0,
    tpot_ratio: float = 1.0,
) -> None:
    directory.joinpath("paired-server-async.json").write_text(
        json.dumps(
            {
                "phase_order": phase_order,
                "model_id": "fake-model",
                "modal_gpu": "L4",
                "request_counts": [32],
                "prompt_profiles": ["shared_prefix"],
                "output_tokens": [8],
                "repeats": 1,
                "max_model_len": 1024,
                "max_num_batched_tokens": 2048,
                "default_max_num_batched_tokens": 2048,
                "max_num_batched_tokens_source": "default",
                "async_enable_prefix_caching_configured": False,
                "server_prefix_caching_configured": configured_mode,
                "server_prefix_caching_flag": [command_flag],
                "server_command": ["vllm", "serve", "fake-model", command_flag],
                "server_logs_head": [
                    f"INFO config: enable_prefix_caching={observed_cache}",
                    "INFO non-default args: {'max_num_batched_tokens': 2048}",
                    "INFO GPU KV cache size: 4,096 tokens",
                    "INFO Maximum concurrency for 1,024 tokens per request: 4.00x",
                    "INFO Available KV cache memory: 1.0 GiB",
                ],
                "server_logs_tail": [
                    (
                        "INFO Engine 000: Avg prompt throughput: 10.0 tokens/s, "
                        "Avg generation throughput: 2.0 tokens/s, Running: 0 reqs, "
                        "Waiting: 0 reqs, GPU KV cache usage: 0.0%, "
                        f"Prefix cache hit rate: {hit_rate_pct}%"
                    )
                ],
                "mean_server_to_async_throughput_ratio": throughput_ratio,
                "mean_server_to_async_latency_ratio": latency_ratio,
                "mean_server_to_async_tpot_ratio": tpot_ratio,
            }
        ),
        encoding="utf-8",
    )


def _write_cache_control_compare_payload(
    directory: Path,
    async_delta: float,
    server_delta: float,
) -> None:
    directory.joinpath("cache-control-phase-order-compare.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "vllm-server-async-cache-control-compare",
                "row_count": 4,
                "cache_modes": ["off", "on"],
                "phase_orders": ["async_first", "server_first"],
                "phase_order_cache_contrasts": [
                    _cache_control_contrast_row(
                        "async_first",
                        throughput_delta=async_delta,
                        latency_delta=-0.9,
                    ),
                    _cache_control_contrast_row(
                        "server_first",
                        throughput_delta=server_delta,
                        latency_delta=-0.8,
                    ),
                ],
                "matrix_rows": [],
                "cache_mode_summary": [],
            }
        ),
        encoding="utf-8",
    )


def _cache_control_contrast_row(
    phase_order: str,
    throughput_delta: float,
    latency_delta: float,
) -> dict:
    off_throughput = 1.0
    on_throughput = off_throughput + throughput_delta
    off_latency = 1.0
    on_latency = off_latency + latency_delta
    return {
        "phase_order": phase_order,
        "off_source_dir": f"off-{phase_order}",
        "on_source_dir": f"on-{phase_order}",
        "off_server_enable_prefix_caching_observed": False,
        "on_server_enable_prefix_caching_observed": True,
        "off_latest_prefix_cache_hit_rate_pct": 0.0,
        "on_latest_prefix_cache_hit_rate_pct": 50.0,
        "cache_on_minus_off_prefix_cache_hit_rate_pct": 50.0,
        "off_throughput_ratio": off_throughput,
        "on_throughput_ratio": on_throughput,
        "cache_on_minus_off_throughput_ratio": throughput_delta,
        "cache_on_div_off_throughput_ratio": on_throughput / off_throughput,
        "off_latency_ratio": off_latency,
        "on_latency_ratio": on_latency,
        "cache_on_minus_off_latency_ratio": latency_delta,
        "cache_on_div_off_latency_ratio": on_latency / off_latency,
        "off_tpot_ratio": 0.5,
        "on_tpot_ratio": 0.05,
        "cache_on_minus_off_tpot_ratio": -0.45,
    }


class _Object:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def _paired_capacity_run(phase: str, gpu_tokens: int) -> dict:
    return {
        "scenario_id": "shared_prefix_mega_long_no_repeat_variant_out8_n32",
        "prompt_profile": "shared_prefix_mega_long_no_repeat_variant",
        "request_count": 32,
        "max_new_tokens": 8,
        "repeat_index": 0,
        "run_order": 0 if phase == "cold" else 1,
        "prompt_tokens_mean": 3790.0,
        "estimated_peak_sequence_tokens": 121280,
        "aggregate_output_tokens_per_second": 100.0,
        "p95_first_chunk_ms": 10.0,
        "p95_latency_ms": 20.0,
        "p95_stream_tpot_ms": 2.0,
        "batch_wall_ms": 50.0,
        "prefix_cache_hit_rate_pct": 0.0,
        "prefix_cache_counter_requests": 1,
        "prefix_cache_counter_queries": 100,
        "prefix_cache_counter_hits": 0,
        "prefix_cache_counter_hit_rate_pct": 0.0,
        "gpu_kv_cache_usage_pct": 0.0,
        "engine_max_model_len": 3790,
        "engine_max_num_batched_tokens": 121280,
        "engine_max_num_seqs": 32,
        "engine_gpu_memory_utilization": 0.5,
        "engine_gpu_kv_cache_size_tokens": gpu_tokens,
        "engine_available_kv_cache_memory_gib": 0.5,
        "engine_max_concurrency_for_request": 4.97,
        "engine_max_concurrency_request_tokens": 3790,
        "engine_derived_gpu_kv_cache_size_tokens": gpu_tokens,
    }


class _WhitespaceTokenizer:
    chat_template = None

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        ids = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            ids.append(self._token_to_id[token])
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._id_to_token.get(token_id, "?") for token_id in ids)


if __name__ == "__main__":
    unittest.main()
