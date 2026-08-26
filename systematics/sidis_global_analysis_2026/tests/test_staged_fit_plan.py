import json
import unittest
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


class StagedFitPlanTests(unittest.TestCase):
    def test_every_registry_source_has_a_non_inventory_stage(self):
        registry = json.loads((BASE / "config/global_sources.json").read_text())
        plan = json.loads((BASE / "config/staged_fit_plan.json").read_text())
        registry_ids = {row["id"] for row in registry["records"]}
        assigned = set()
        for stage in plan["stages"]:
            source_ids = stage["source_ids"]
            if isinstance(source_ids, list):
                assigned.update(source_ids)
        self.assertEqual(assigned, registry_ids)

    def test_discovery_plan_cannot_authorize_fit(self):
        plan = json.loads((BASE / "config/staged_fit_plan.json").read_text())
        self.assertEqual(plan["approved_rows"], 0)
        self.assertFalse(plan["production_authorized"])
        self.assertIn("S1_literature_benchmark", {s["id"] for s in plan["stages"]})


if __name__ == "__main__":
    unittest.main()
