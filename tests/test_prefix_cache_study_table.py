from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_prefix_cache_study_table import (
    build_intervals,
    build_rows,
    render_interval_chart,
)


SUMMARY_JSON = (
    ROOT
    / "results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary-ttft"
    / "prefix-cache-isolated-stability-summary.json"
)


class PrefixCacheStudyTableTests(unittest.TestCase):
    def test_builds_expected_key_result_rows(self) -> None:
        payload = json.loads(SUMMARY_JSON.read_text())

        rows = build_rows(payload)

        self.assertEqual(len(rows), 7)
        by_metric = {row["metric"]: row for row in rows}
        self.assertEqual(
            by_metric["Control direct counter hit rate"]["value"],
            "4.716%",
        )
        self.assertEqual(
            by_metric["Shared direct counter hit rate"]["value"],
            "83.138%",
        )
        self.assertEqual(
            by_metric["Shared-minus-control direct counter delta"]["bootstrap_interval"],
            "78.269 pp to 78.602 pp",
        )
        self.assertEqual(
            by_metric["p95 first-event/TTFT ratio delta"]["bootstrap_interval"],
            "-0.497 to -0.073",
        )
        self.assertIn(
            "no stable throughput claim",
            by_metric["Throughput-ratio delta"]["interpretation"],
        )

    def test_builds_interval_chart_from_key_effects(self) -> None:
        payload = json.loads(SUMMARY_JSON.read_text())

        intervals = build_intervals(payload)
        chart = render_interval_chart(intervals, SUMMARY_JSON)

        self.assertEqual(len(intervals), 5)
        self.assertIn("Direct counter delta", chart)
        self.assertIn("78.269 pp to 78.602 pp", chart)
        self.assertIn("p95 first-event/TTFT ratio delta", chart)
        self.assertIn("-0.497 to -0.073", chart)
        self.assertIn("crosses zero; no stable throughput claim", chart)

    def test_rejects_ambiguous_profile_control_rows(self) -> None:
        payload = json.loads(SUMMARY_JSON.read_text())
        payload["profile_control_rows"] = [
            payload["profile_control_rows"][0],
            payload["profile_control_rows"][0],
        ]

        with self.assertRaisesRegex(ValueError, "exactly one profile-control row"):
            build_rows(payload)


if __name__ == "__main__":
    unittest.main()
