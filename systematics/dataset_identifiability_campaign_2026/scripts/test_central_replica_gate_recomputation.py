#!/usr/bin/env python3
"""Independent central/replica terminal-gate recomputation tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_lambda600_state_chains as audit


class CentralReplicaGateRecomputationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.outputs = self.root / "outputs"
        self.origin = self.outputs / "central"
        self._write_fnp(self.origin, np.array([1.0, .8, .5]))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _grid(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "x": [0.1, 0.1, 0.1],
            "bT": [0.1, 1.0, 2.0],
            "F_NP": values,
        })

    def _write_fnp(self, run: Path, values: np.ndarray) -> None:
        run.mkdir(parents=True, exist_ok=True)
        self._grid(values).to_csv(run / "fnp_grid.csv", index=False)
        (run / "model_state.pt").write_bytes(b"state")
        (run / "dataset_norms.csv").write_text(
            "dataset,control_norm\nA,1.0\n")
        (run / "accepted_predictions.csv").write_text("row_id\nrow\n")
        (run / "fit_status.json").write_text(json.dumps({
            "final": {
                "unpenalized_total_chi2": 100.0,
                "data_chi2": 90.0,
            },
            "lbfgs": {"closure_evaluations": 3},
        }) + "\n")

    def quiet_replica_ledger(self, blocks: int = 50) -> pd.DataFrame:
        rows = []
        ceiling = audit.N_DATA + 5.0 * np.sqrt(2.0 * audit.N_DATA)
        for chunk in range(1, blocks + 1):
            cumulative = chunk * 5000
            tag = f"replica_r1001_{cumulative}"
            self._write_fnp(
                self.outputs / tag, np.array([1.0, .8, .5]))
            eligible = cumulative > 200_000
            consecutive = max(0, chunk - 40)
            stationarity = cumulative >= 250_000
            rows.append({
                "tag": tag,
                "cumulative_lbfgs_iterations": cumulative,
                "minimum_required_iterations": 50_000,
                "fnp_drift_from_previous_chunk": 0.0,
                "eligible_post_mandatory_confirmation": eligible,
                "post_mandatory_window_fnp_drift": (
                    0.0 if eligible else np.nan),
                "stationarity_window_anchor_iterations": (
                    200_000 if cumulative >= 200_000 else np.nan),
                "next_stationarity_window_anchor_iterations": (
                    200_000 if cumulative >= 200_000 else np.nan),
                "passes_post_mandatory_window_drift_2pct": eligible,
                "passes_drift_0p25pct": True,
                "passes_drift_0p5pct": True,
                "passes_drift_1pct": True,
                "passes_drift_2pct": True,
                "consecutive_quiet_blocks": consecutive,
                "sensitivity_confirmation_triggered": False,
                "fresh_quiet_blocks_after_sensitivity_trigger": 0,
                "unpenalized_total_chi2": 100.0,
                "data_chi2": 90.0,
                "replica_chi2_sanity_ceiling": ceiling,
                "replica_chi2_sanity_pass": True,
                "fit_quality_ceiling_total_chi2": ceiling,
                "fit_quality_gate_pass": True,
                "stationarity_gate_pass": stationarity,
                "endpoint_acceptance_gate_pass": stationarity,
                "fnp_plateau_gate_pass": stationarity,
                "full_horizon_required": False,
                "requested_lbfgs_max_iterations_this_block": 5000,
                "executed_lbfgs_closure_evaluations_this_block": 3,
            })
        return pd.DataFrame(rows)

    def recompute(self, ledger: pd.DataFrame) -> dict:
        ceiling = audit.N_DATA + 5.0 * np.sqrt(2.0 * audit.N_DATA)
        with mock.patch.object(audit, "OUT", self.outputs):
            return audit.recompute_adaptive_chain_gates(
                ledger, origin=self.origin, replica_seed=1001,
                fit_quality_ceiling=ceiling, label="replica_r1001")

    def test_pristine_quiet_chain_passes_at_exact_first_eligible_window(self) -> None:
        result = self.recompute(self.quiet_replica_ledger())
        self.assertTrue(result["terminal_endpoint_acceptance_gate_pass"])
        self.assertEqual(result["terminal_requested_capacity"], 250_000)
        self.assertEqual(result["first_acceptance_requested_capacity"], 250_000)

    def test_tampered_terminal_stationarity_flag_fails_closed(self) -> None:
        ledger = self.quiet_replica_ledger()
        ledger.loc[ledger.index[-1], "stationarity_gate_pass"] = False
        with self.assertRaisesRegex(RuntimeError, "stationarity_gate_pass differs"):
            self.recompute(ledger)

    def test_changed_endpoint_curve_fails_against_recorded_drift(self) -> None:
        ledger = self.quiet_replica_ledger()
        run = self.outputs / str(ledger.iloc[42].tag)
        self._grid(np.array([1.2, .8, .5])).to_csv(
            run / "fnp_grid.csv", index=False)
        with self.assertRaisesRegex(RuntimeError, "adjacent drift"):
            self.recompute(ledger)

    def test_nonpassing_replica_cannot_be_truncated_before_300k(self) -> None:
        ledger = self.quiet_replica_ledger(blocks=49)
        with self.assertRaisesRegex(RuntimeError, "truncated before 300k"):
            self.recompute(ledger)

    def test_tampered_fit_gate_fails_against_endpoint_status(self) -> None:
        ledger = self.quiet_replica_ledger()
        terminal = self.outputs / str(ledger.iloc[-1].tag)
        status = json.loads((terminal / "fit_status.json").read_text())
        status["final"]["unpenalized_total_chi2"] = 500.0
        (terminal / "fit_status.json").write_text(json.dumps(status) + "\n")
        with self.assertRaisesRegex(
                RuntimeError, "replica_chi2_sanity_pass|unpenalized chi2"):
            self.recompute(ledger)

    def test_stale_terminal_summary_boolean_fails_closed(self) -> None:
        summary = {
            "status": "complete",
            "central_fnp_plateau_pass": False,
            "failed_replica_seeds": [],
        }
        with self.assertRaisesRegex(RuntimeError, "Boolean differs"):
            audit.validate_declared_terminal_gate_flags(
                summary, expected_status="complete",
                expected_bools={"central_fnp_plateau_pass": True},
                expected_seed_lists={"failed_replica_seeds": []})

    def test_stale_terminal_summary_failure_list_fails_closed(self) -> None:
        summary = {
            "status": "complete_with_scientific_failures",
            "central_fnp_plateau_pass": True,
            "failed_replica_seeds": [],
        }
        with self.assertRaisesRegex(RuntimeError, "seed list differs"):
            audit.validate_declared_terminal_gate_flags(
                summary,
                expected_status="complete_with_scientific_failures",
                expected_bools={"central_fnp_plateau_pass": True},
                expected_seed_lists={"failed_replica_seeds": [1001]})


if __name__ == "__main__":
    unittest.main()
