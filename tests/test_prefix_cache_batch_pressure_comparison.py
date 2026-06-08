from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_prefix_cache_batch_pressure_comparison import (
    build_payload,
    build_rows,
    load_point,
    render_markdown,
)


N16_SUMMARY_JSON = (
    ROOT
    / "results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n16-merged-r8-summary"
    / "prefix-cache-isolated-stability-summary.json"
)
N32_SUMMARY_JSON = (
    ROOT
    / "results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-merged-r8-summary"
    / "prefix-cache-isolated-stability-summary.json"
)


class PrefixCacheBatchPressureComparisonTests(unittest.TestCase):
    def test_builds_expected_n16_vs_n32_rows(self) -> None:
        baseline = load_point(N16_SUMMARY_JSON, "n16 r8")
        candidate = load_point(N32_SUMMARY_JSON, "n32 r8")

        rows = build_rows(baseline, candidate)

        self.assertEqual(len(rows), 8)
        by_metric = {row["metric"]: row for row in rows}
        self.assertEqual(by_metric["Request count"]["candidate_minus_baseline"], "+16")
        self.assertEqual(
            by_metric["Shared-minus-control direct counter delta"]["baseline_value"],
            "92.628 pp",
        )
        self.assertEqual(
            by_metric["Shared-minus-control direct counter delta"]["candidate_value"],
            "95.797 pp",
        )
        self.assertEqual(
            by_metric["Shared-minus-control direct counter delta"][
                "candidate_minus_baseline"
            ],
            "+3.169 pp",
        )
        self.assertEqual(
            by_metric["p95 stream TPOT-ratio delta"]["candidate_minus_baseline"],
            "+0.264",
        )
        self.assertIn(
            "weaker than n16",
            by_metric["p95 stream TPOT-ratio delta"]["interpretation"],
        )

    def test_renders_markdown_and_payload(self) -> None:
        baseline = load_point(N16_SUMMARY_JSON, "n16 r8")
        candidate = load_point(N32_SUMMARY_JSON, "n32 r8")
        rows = build_rows(baseline, candidate)

        markdown = render_markdown(baseline, candidate, rows)
        payload = build_payload(baseline, candidate, rows)

        self.assertIn("# Prefix-Cache Batch-Pressure Comparison", markdown)
        self.assertIn("| Request count | 16 | 32 | +16 |", markdown)
        self.assertEqual(payload["mode"], "prefix-cache-batch-pressure-comparison")
        self.assertEqual(payload["baseline"]["request_count"], 16)
        self.assertEqual(payload["candidate"]["request_count"], 32)
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
