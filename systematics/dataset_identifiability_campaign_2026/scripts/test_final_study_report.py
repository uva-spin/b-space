#!/usr/bin/env python3
"""Focused tests for the fixed-challenger terminal report contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load_module():
    spec = importlib.util.spec_from_file_location(
        "tested_final_study_report", HERE / "write_final_study_report.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("write_final_study_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reporter = load_module()


def stability_fixture() -> dict:
    direct = {
        "u": {"bootstrap": .010, "start": .008, "replica": .006, "joint": .007},
        "d": {"bootstrap": .009, "start": .007, "replica": .011, "joint": .008},
    }
    allowance = {flavor: max(values.values())
                 for flavor, values in direct.items()}
    raw = {"u": .065, "d": .070}
    return {
        "status": "complete",
        "start_count": 24,
        "replica_count": 50,
        "coverage_gate_pass": True,
        "band_integrity_gate_pass": True,
        "state_chain_gate_pass": True,
        "endpoint_gate_pass": True,
        "bootstrap_replicates": 300,
        "split_half_replicates": 200,
        "resampling_allowance_semantics": (
            "direct resampling of the exact flavor-local final statistic; "
            "no endpoint-motion conversion factor"
        ),
        "comparison_champion_id": reporter.INCUMBENT_ID,
        "comparison_champion_max_active_relative_full_width":
            dict(reporter.LOCKED_INCUMBENT_WIDTHS),
        "bootstrap_p95_absolute_full_width_statistic_deviation_by_flavor": {
            flavor: values["bootstrap"] for flavor, values in direct.items()},
        "start_split_half_p95_absolute_full_width_statistic_difference_by_flavor": {
            flavor: values["start"] for flavor, values in direct.items()},
        "replica_split_half_p95_absolute_full_width_statistic_difference_by_flavor": {
            flavor: values["replica"] for flavor, values in direct.items()},
        "joint_split_half_p95_absolute_full_width_statistic_difference_by_flavor": {
            flavor: values["joint"] for flavor, values in direct.items()},
        "resampling_full_width_allowance_by_flavor": allowance,
        "raw_final_max_active_relative_full_width": raw,
        "robust_adjusted_max_active_relative_full_width": {
            flavor: raw[flavor] + allowance[flavor] for flavor in reporter.FLAVORS},
        "robust_improvement_gate_by_flavor": {"u": True, "d": True},
    }


def final_fixture() -> dict:
    resampling_components = {
        "u": {"bootstrap": .010, "start": .008, "replica": .006, "joint": .007},
        "d": {"bootstrap": .009, "start": .007, "replica": .011, "joint": .008},
    }
    metrics = {}
    for flavor in reporter.FLAVORS:
        incumbent = reporter.LOCKED_INCUMBENT_WIDTHS[flavor]
        allowance = .010 if flavor == "u" else .011
        metrics[flavor] = {
            "terminal_product_raw_full_width": .065 if flavor == "u" else .070,
            "terminal_anchor_convergence_raw_full_width": .071 if flavor == "u" else .076,
            "joint_convergence_interaction_raw_full_width": .079 if flavor == "u" else .083,
            "corrected_finite_sampling_full_width_margin": allowance,
            "final_statistic_finite_sampling_full_width_margin": allowance,
            "joint_raw_width_plus_corrected_sampling_margin":
                (.079 if flavor == "u" else .083) + allowance,
            "joint_raw_width_plus_final_statistic_sampling_margin":
                (.079 if flavor == "u" else .083) + allowance,
            "immutable_lambda1_width": incumbent,
            "replacement_gate_pass": True,
            "sampling_allowance_statistic_matches_width_statistic": True,
        }
    return {
        "promotion_validation_gate_pass": True,
        "base_product_stability_gate_pass": True,
        "postfit_tail_convergence_gate_pass": True,
        "nested_interaction_validation_gate_pass": True,
        "joint_width_replacement_gate_pass": True,
        "final_joint_sampling_gate_authoritative": True,
        "prior_product_median_sampling_gate_authoritative": False,
        "trained_central_containment_gate_pass": True,
        "formal_confidence_level_assigned": False,
        "one_sigma_claimed": False,
        "scientific_failure_reasons": [],
        "width_metrics_by_flavor": metrics,
        "final_statistic_resampling": {
            "bootstrap_replicates": 300,
            "split_half_replicates": 200,
            "rng_seed": 60024050,
            "full_exact_final_statistic_by_flavor": {"u": .079, "d": .083},
            "bootstrap_p95_absolute_deviation_by_flavor": {
                flavor: values["bootstrap"]
                for flavor, values in resampling_components.items()},
            "start_split_p95_absolute_difference_by_flavor": {
                flavor: values["start"]
                for flavor, values in resampling_components.items()},
            "replica_split_p95_absolute_difference_by_flavor": {
                flavor: values["replica"]
                for flavor, values in resampling_components.items()},
            "joint_split_p95_absolute_difference_by_flavor": {
                flavor: values["joint"]
                for flavor, values in resampling_components.items()},
            "allowance_by_flavor": {"u": .010, "d": .011},
            "statistic": (
                "maximum relative width divided by the absolute fixed "
                "trained-300k central"),
            "correlation_preserved": (
                "terminal and anchor use identical resampled identities"),
            "interaction_resampled": False,
        },
        "prior_product_median_sampling_allowance_diagnostic_by_flavor": {
            "u": .012, "d": .013},
        "artifacts": {
            "fnp_final_envelope": str(
                reporter.BASE / "summaries/lambda600_final_directional_envelope/"
                "fnp_final_directional_envelope.csv"),
            "fig2_bspace_final_envelope": str(
                reporter.BASE / "summaries/lambda600_final_directional_envelope/"
                "fig2_bspace_final_directional_envelope.csv"),
            "fig6_kspace_final_envelope": str(
                reporter.BASE / "summaries/lambda600_final_directional_envelope/"
                "fig6_kspace_final_directional_envelope.csv"),
        },
    }


def render_evidence(outcome: str) -> dict:
    promoted = outcome == reporter.PROMOTED
    stems = reporter.VALIDATED_STEMS if promoted else reporter.DIAGNOSTIC_STEMS
    final = final_fixture()
    if not promoted:
        final["promotion_validation_gate_pass"] = False
        final["base_product_stability_gate_pass"] = False
        final["scientific_failure_reasons"] = [
            "24-start FNP stationarity/fit-preservation gate failed"]
    return {
        "outcome": outcome,
        "decision_validation_mode": "prepublication_complete_graph_revalidated",
        "starts": {
            "status": "complete" if promoted else "verification_failed",
            "failed_seeds": [] if promoted else [306],
        },
        "start_ancestry": {
            "seal": str(
                reporter.BASE / "summaries/lambda600_start_chain_audit/"
                "current_byte_seals/example.json"),
            "launch_time_receipt_checkpoint_count": 9,
            "legacy_pre_receipt_checkpoint_count": 1200,
            "historical_ancestry_limitation": (
                "Legacy checkpoints have path ancestry and a precentral "
                "current-byte seal, not retrospective launch-time proof."),
        },
        "replicas": {
            "status": "complete",
            "failed_replica_seeds": [],
            "central_endpoint_tag": "central_lambda600_300000",
        },
        "product": {"combined_member_count": 1200},
        "postfit": {"status": "complete_postfit_full_tail_transform_validation"},
        "nested": {
            "status": "complete_observed_interaction_decision_sign_stable",
            "completed_pair_count": 6,
        },
        "final_envelope": final,
        "figures": {
            "status": ("final_validated_figures" if promoted
                       else "diagnostic_figures_not_promotable"),
        },
        "comparison": {
            "status": ("complete_validated_candidate_comparison" if promoted
                       else "complete_diagnostic_scientific_failure_comparison"),
            "artifacts": {
                "fig6_png": str(
                    reporter.BASE / "summaries/lambda600_vs_lambda1_diagnostic/"
                    "lambda600_vs_lambda1_fig6_kT_Q10.png"),
            },
        },
        "frozen": {"unchanged_input_count": 32, "registered_input_count": 32},
        "width_metrics": {
            flavor: {
                "terminal_product_raw_full_width": item[
                    "terminal_product_raw_full_width"],
                "terminal_anchor_convergence_raw_full_width": item[
                    "terminal_anchor_convergence_raw_full_width"],
                "joint_convergence_interaction_raw_full_width": item[
                    "joint_convergence_interaction_raw_full_width"],
                "direct_finite_sampling_full_width_margin": item[
                    "corrected_finite_sampling_full_width_margin"],
                "joint_raw_plus_direct_sampling_margin": item[
                    "joint_raw_width_plus_corrected_sampling_margin"],
                "immutable_lambda1_full_width": item["immutable_lambda1_width"],
                "replacement_gate_pass": item["replacement_gate_pass"],
            }
            for flavor, item in final["width_metrics_by_flavor"].items()
        },
        "gates": {
            "start_stationarity_and_fit": promoted,
            "trained_central_stationarity": True,
            "experimental_replica_stationarity_and_agreement": True,
            "base_product_stability": promoted,
            "postfit_tail_transform": True,
            "nested_interaction": True,
            "trained_central_containment": True,
            "joint_width_replacement": True,
            "complete_promotion_gate": promoted,
        },
        "figure_paths": {
            role: reporter.FIGURE_DIR / f"{stem}.pdf"
            for role, stem in stems.items()
        },
    }


class DirectWidthEvidenceTests(unittest.TestCase):
    def test_direct_allowance_and_final_directional_width_are_distinct(self) -> None:
        stability = stability_fixture()
        final = final_fixture()
        observed, passed = reporter.validate_width_evidence(stability, final)
        self.assertTrue(passed)
        self.assertAlmostEqual(
            observed["d"]["direct_finite_sampling_full_width_margin"], .011)
        self.assertAlmostEqual(
            observed["d"]["joint_raw_plus_direct_sampling_margin"], .094)

    def test_non_matching_compatibility_allowance_fails_closed(self) -> None:
        stability = stability_fixture()
        final = final_fixture()
        final["width_metrics_by_flavor"]["u"][
            "corrected_finite_sampling_full_width_margin"] = .020
        with self.assertRaisesRegex(RuntimeError, "compatibility allowance"):
            reporter.validate_width_evidence(stability, final)


class OutcomeSemanticsTests(unittest.TestCase):
    def common_inputs(self):
        starts = {
            "status": "complete",
            "all_starts_fnp_plateaued_and_fit_preserved": True,
        }
        replicas = {
            "central_fnp_plateau_pass": True,
            "all_replicas_fnp_plateaued": True,
        }
        stability = {
            "endpoint_gate_pass": True,
            "diagnostic_figure_gate_pass": True,
            "candidate_stationarity_gate_pass": True,
        }
        postfit = {"promotion_validation_gate_pass": True}
        nested = {"interaction_validation_gate_pass": True}
        final = final_fixture()
        figures = {
            "status": "final_validated_figures",
            "endpoint_gate_pass": True,
            "diagnostic_only": False,
            "formal_confidence_level_assigned": False,
            "one_sigma_claimed": False,
        }
        comparison = {
            "status": "complete_validated_candidate_comparison",
            "comparison_champion_id": reporter.INCUMBENT_ID,
            "candidate_endpoint_gate_pass": True,
            "diagnostic_only": False,
            "legacy_lambda1_fig6_widths_remain_gating": True,
        }
        return starts, replicas, stability, postfit, nested, final, figures, comparison

    def test_promoted_status_requires_every_gate(self) -> None:
        inputs = self.common_inputs()
        gates = reporter.validate_outcome_consistency(
            reporter.PROMOTED, *inputs, width_gate=True)
        self.assertTrue(gates["complete_promotion_gate"])

    def test_promotion_label_fails_when_start_gate_failed(self) -> None:
        inputs = list(self.common_inputs())
        inputs[0] = {
            "status": "verification_failed",
            "all_starts_fnp_plateaued_and_fit_preserved": False,
        }
        with self.assertRaisesRegex(RuntimeError, "terminal status disagrees"):
            reporter.validate_outcome_consistency(
                reporter.PROMOTED, *inputs, width_gate=True)


class ReportTextTests(unittest.TestCase):
    def test_promoted_report_uses_trained_central_and_correct_artifact_names(self) -> None:
        report = reporter.render_report(render_evidence(reporter.PROMOTED))
        self.assertIn("separately trained terminal lambda=600 central", report)
        self.assertIn("updated_fig2_bspace_product_plus_directional_envelope.pdf", report)
        self.assertIn("updated_fig6_kspace_ud_product_plus_directional_envelope.pdf", report)
        self.assertIn("No formal\n1sigma or confidence-level interpretation", report)
        self.assertIn("not described as retrospective proof", report)
        self.assertIn("Every central and\nexperimental-replica checkpoint", report)
        self.assertNotIn("minimum_fitbar_constraint_search", report)
        self.assertNotIn("lambda=675", report)

    def test_rejected_report_is_complete_and_uses_diagnostic_names(self) -> None:
        report = reporter.render_report(render_evidence(reporter.REJECTED))
        self.assertIn("was rejected", report)
        self.assertIn("lambda=1 incumbent remains", report)
        self.assertIn(
            "diagnostic_failed_fig2_bspace_product_plus_directional_envelope.pdf",
            report)
        self.assertIn(
            "diagnostic_failed_fig6_kspace_ud_product_plus_directional_envelope.pdf",
            report)
        self.assertIn("24-start FNP stationarity", report)

    def test_cli_accepts_explicit_prepublication_status(self) -> None:
        args = reporter.parse_args([
            "--decision-status", reporter.REJECTED])
        self.assertEqual(args.decision_status, reporter.REJECTED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
