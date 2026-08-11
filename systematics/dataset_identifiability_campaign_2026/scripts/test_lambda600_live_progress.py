#!/usr/bin/env python3
"""Focused tests for the non-decisional lambda600 live reporter."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

import summarize_lambda600_live_progress as live


class LiveProgressTests(unittest.TestCase):
    def test_progress_derivation_marks_incomplete_seed_nonpromotable(self) -> None:
        manifest = {
            "completed_member_count": 1,
            "endpoint_tags_so_far": ["candidate_s303_terminal"],
        }
        runs = pd.DataFrame({
            "seed": [303, 304],
            "cumulative_lbfgs_iterations": [250000, 5000],
        })
        progress = live.derive_progress(manifest, runs, [303, 304, 305])
        self.assertEqual(progress["started_member_count"], 2)
        self.assertEqual(progress["completed_terminal_seeds"], [303])
        self.assertEqual(progress["active_or_incomplete_seeds"], [304])
        self.assertEqual(progress["active_seed"], 304)
        self.assertEqual(progress["unstarted_seeds"], [305])

    def test_terminal_full24_manifest_uses_member_count(self) -> None:
        manifest = {
            "status": "verification_failed",
            "member_count": 3,
            "endpoint_tags": [
                "candidate_s303_terminal",
                "candidate_s304_terminal",
                "candidate_s305_terminal",
            ],
        }
        runs = pd.DataFrame({
            "seed": [303, 304, 305],
            "cumulative_lbfgs_iterations": [250000, 300000, 255000],
        })
        progress = live.derive_progress(manifest, runs, [303, 304, 305])
        self.assertEqual(progress["completed_terminal_member_count"], 3)
        self.assertEqual(progress["active_or_incomplete_seeds"], [])
        self.assertEqual(progress["unstarted_seeds"], [])

    def test_transient_change_is_retried(self) -> None:
        stable = {Path("input"): b"stable"}
        changed = {Path("input"): b"changed"}
        reads = iter((stable, changed, stable, stable))
        sleeps = []
        with mock.patch.object(live, "read_core_bytes",
                               side_effect=lambda paths: next(reads)), \
             mock.patch.object(live, "build_payload",
                               return_value={"provisional": True}):
            payload = live.collect_consistent_payload(
                attempts=2, retry_seconds=.25, sleeper=sleeps.append)
        self.assertEqual(payload["consistent_read_attempt"], 2)
        self.assertEqual(sleeps, [.25])

    def test_atomic_writer_replaces_only_requested_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = root / "HANDOFF.md"
            target = root / "summaries/lambda600_live_progress/summary.json"
            protected.write_text("unchanged\n")
            live.atomic_write_json(target, {"status": "provisional"})
            self.assertEqual(protected.read_text(), "unchanged\n")
            self.assertEqual(json.loads(target.read_text()),
                             {"status": "provisional"})
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_payload_has_explicit_time_protocol_and_warning(self) -> None:
        instant = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            protocol = {
                "candidate": {"start_seeds": [303, 304]},
            }
            manifest = {
                "status": "in_progress", "completed_member_count": 1,
                "endpoint_tags_so_far": ["fit_s303_terminal"],
            }
            sources = {"endpoint_tags": ["source_s303", "source_s304"]}
            columns = {
                "strength": [600.0, 600.0], "seed": [303, 304],
                "cumulative_lbfgs_iterations": [250000, 5000],
                "fnp_drift_from_previous_chunk": [.001, .002],
                "eligible_post_mandatory_confirmation": [True, False],
                "post_mandatory_window_fnp_drift": [.001, float("nan")],
                "stationarity_window_anchor_iterations": [200000, float("nan")],
                "next_stationarity_window_anchor_iterations": [200000, float("nan")],
                "consecutive_quiet_blocks": [10, 0],
                "sensitivity_confirmation_triggered": [False, False],
                "fresh_quiet_blocks_after_sensitivity_trigger": [0, 0],
                "unpenalized_total_chi2": [130., 131.],
                "stationarity_and_fit_pass": [True, False],
                "tag": ["fit_s303_terminal", "fit_s304_5000"],
            }
            runs = pd.DataFrame(columns).to_csv(index=False).encode()
            for tag, shift in (("source_s303", 0.), ("source_s304", 0.),
                               ("fit_s303_terminal", .01),
                               ("fit_s304_5000", .02)):
                folder = outputs / tag
                folder.mkdir()
                pd.DataFrame({
                    "x": [.1, .1, .1, .1], "bT": [.1, 1., 2., 4.],
                    "F_NP": [1.-shift, .8-shift, .5-shift, .2-shift],
                }).to_csv(folder / "fnp_grid.csv", index=False)
            raw = {
                live.PROTOCOL: json.dumps(protocol).encode(),
                live.FULL24_SUMMARY: json.dumps(manifest).encode(),
                live.RUNS: runs,
                live.SOURCES: json.dumps(sources).encode(),
            }
            payload = live.build_payload(raw, outputs=outputs, now=instant)
        self.assertEqual(payload["generated_at_utc"], instant.isoformat())
        self.assertEqual(payload["authoritative_full24_status"], "in_progress")
        self.assertEqual(payload["started_member_count"], 2)
        self.assertFalse(payload["promotable"])
        self.assertIn("NON-PROMOTABLE", payload["warning"])


if __name__ == "__main__":
    unittest.main()
