#!/usr/bin/env python3
"""Focused crash/idempotence tests for campaign-local champion promotion."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import promote_validated_final_champion as promoter


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


class PromotionTransactionTests(unittest.TestCase):
    def paths(self, root: Path) -> dict[str, Path]:
        public = root / "public"
        return {
            "record": root / "registry/candidate.json",
            "current": root / "registry/current.json",
            "summary": public / "summary.json",
            "transaction": root / "registry/.transaction.json",
        }

    def payloads(self) -> tuple[dict, dict, dict]:
        champion = "candidate"
        record = {
            "status": "complete_audited_study_champion_not_frozen_production",
            "champion_id": champion,
            "production_sources_modified": False,
        }
        summary = {
            "status": "complete",
            "champion_id": champion,
            "production_sources_modified": False,
        }
        transaction = {
            "champion_id": champion,
            "immutable_incumbent_id": "incumbent",
        }
        return record, summary, transaction

    def test_retry_completes_after_crash_at_every_commit_boundary(self) -> None:
        for crash_stage in (
                "champion_record", "public_summary",
                "ready_transaction", "current_commit"):
            with self.subTest(crash_stage=crash_stage), \
                    tempfile.TemporaryDirectory() as directory:
                paths = self.paths(Path(directory))
                write_json(paths["current"], {"champion_id": "incumbent"})
                record, summary, transaction = self.payloads()

                def crash(stage: str) -> None:
                    if stage == crash_stage:
                        raise RuntimeError(f"simulated crash after {stage}")

                with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                    promoter.publish_registry_commit(
                        record=record, public_summary=summary,
                        transaction_core=transaction,
                        champion_record=paths["record"],
                        current=paths["current"],
                        public_summary_path=paths["summary"],
                        transaction_path=paths["transaction"],
                        step_hook=crash,
                    )
                if crash_stage != "current_commit":
                    self.assertEqual(
                        json.loads(paths["current"].read_text())["champion_id"],
                        "incumbent",
                    )

                promoter.publish_registry_commit(
                    record=record, public_summary=summary,
                    transaction_core=transaction,
                    champion_record=paths["record"],
                    current=paths["current"],
                    public_summary_path=paths["summary"],
                    transaction_path=paths["transaction"],
                )
                result = promoter.validate_registry_commit(
                    expected_champion_id="candidate",
                    champion_record=paths["record"],
                    current=paths["current"],
                    public_summary_path=paths["summary"],
                    transaction_path=paths["transaction"],
                )
                self.assertEqual(result["status"], "pass")
                self.assertEqual(result["champion_id"], "candidate")
                self.assertFalse(list(Path(directory).rglob("*.tmp.*")))

    def test_commit_validator_rejects_public_or_current_tamper(self) -> None:
        for target in ("summary", "current"):
            with self.subTest(target=target), \
                    tempfile.TemporaryDirectory() as directory:
                paths = self.paths(Path(directory))
                record, summary, transaction = self.payloads()
                promoter.publish_registry_commit(
                    record=record, public_summary=summary,
                    transaction_core=transaction,
                    champion_record=paths["record"],
                    current=paths["current"],
                    public_summary_path=paths["summary"],
                    transaction_path=paths["transaction"],
                )
                paths[target].write_bytes(paths[target].read_bytes() + b" ")
                with self.assertRaises(RuntimeError):
                    promoter.validate_registry_commit(
                        expected_champion_id="candidate",
                        champion_record=paths["record"],
                        current=paths["current"],
                        public_summary_path=paths["summary"],
                        transaction_path=paths["transaction"],
                    )

    def test_every_declared_artifact_is_hash_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.csv"
            artifact.write_text("x\n1\n")
            record = {
                "artifacts": {"band": str(artifact)},
                "artifact_sha256": {"band": promoter.sha256(artifact)},
            }
            promoter.validated_artifact_bindings(
                record, required_keys={"band"})
            artifact.write_text("x\n2\n")
            with self.assertRaisesRegex(RuntimeError, "hash changed"):
                promoter.validated_artifact_bindings(
                    record, required_keys={"band"})


if __name__ == "__main__":
    unittest.main()
