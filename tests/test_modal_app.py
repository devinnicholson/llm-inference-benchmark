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
