"""Synthetic engineering contracts; no fixture claims to be a real study."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from GNN.data_pipeline import load_csv_graphs
from GNN.gnn_clean import split_scenarios, train_gnn_multi_graph


def write_fixture(root):
    buses, edges, scenarios = [], [], {}
    for scenario in range(10):
        buses.extend([
            {"scenario": scenario, "bus": 1, "load_MW": 10 + scenario, "p_inj_mw": -10 - scenario,
             "load_Mvar": 2, "q_inj_mvar": -2, "voltage_pu": 1.0, "voltage_class": 2},
            {"scenario": scenario, "bus": 2, "load_MW": 0, "p_inj_mw": 10 + scenario,
             "load_Mvar": 0, "q_inj_mvar": 2, "voltage_pu": 1.0, "voltage_class": 2}])
        edges.append({"scenario": scenario, "from_bus": 1, "to_bus": 2, "branch_id": "1-2-A",
                      "in_service": True, "x_pu": 0.1, "length_km": 5, "normal_rating_mva": 100,
                      "emergency_rating_mva": 120, "flow_mva": 40, "thermal_class": 0})
        scenarios[str(scenario)] = {"source_case": "synthetic-only", "source_case_sha256": "a" * 64,
            "split_group_id": str(scenario // 2), "topology_assumptions": "synthetic two-bus fixture",
            "load_dispatch_assumptions": "synthetic, no solver run", "solver_version": "mock",
            "executed_at": "2026-01-01T00:00:00Z", "solver_converged": True,
            "condition": "pre_contingency", "contingency_definition": "none", "rating_basis": "normal"}
    pd.DataFrame(buses).to_csv(root / "bus_scenarios.csv", index=False)
    pd.DataFrame(edges).to_csv(root / "edge_scenarios.csv", index=False)
    manifest = {"schema_version": 1, "data_kind": "raw_physical", "dataset_id": "synthetic-only",
        "source": "unit test fixture, not real simulation",
        "feature_availability": {"role": "pre_study_inputs", "description": "synthetic values at origin"},
        "preprocessing": "none", "source_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
         for name in ("bus_scenarios.csv", "edge_scenarios.csv")},
        "units": {"load_MW": "MW", "p_inj_mw": "MW", "load_Mvar": "Mvar", "q_inj_mvar": "Mvar",
                  "voltage_pu": "pu", "x_pu": "pu", "length_km": "km", "flow_mva": "MVA",
                  "normal_rating_mva": "MVA", "emergency_rating_mva": "MVA"},
        "scenarios": scenarios, "class_thresholds": {"voltage": [0.90, 0.95, 1.05, 1.10], "thermal": [50, 80, 100]},
        "criteria_source": "arbitrary unit-test thresholds, not engineering requirements"}
    (root / "study_manifest.json").write_text(json.dumps(manifest))
    return manifest


class ProvenanceTests(unittest.TestCase):
    def test_raw_thermal_includes_dispatch_and_grouped_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            graphs = load_csv_graphs("thermal", root, require_traceable=True)
            self.assertEqual(graphs[0].x.shape[1], 11)
            self.assertNotIn("flow_mva", graphs[0].feature_names)
            self.assertFalse(np.array_equal(graphs[0].x, graphs[1].x))
            splits = split_scenarios(graphs)
            groups = [set(graphs[i].split_group_id for i in indices) for indices in splits.values()]
            self.assertTrue(all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:]))
            _, history = train_gnn_multi_graph(graphs, epochs=1, batch_size=2)
            self.assertIsNotNone(history.attrs["evaluation"]["study_manifest"])
            self.assertEqual(len(load_csv_graphs("voltage", root)), 10)

    def test_stale_hash_failed_solver_and_inconsistent_labels_are_rejected(self):
        for defect in ("hash", "solver", "criteria", "units", "threshold_type", "availability"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = write_fixture(root)
                if defect == "hash": manifest["source_hashes"]["bus_scenarios.csv"] = "0" * 64
                if defect == "solver": manifest["scenarios"]["0"]["solver_converged"] = False
                if defect == "criteria": manifest["class_thresholds"]["thermal"] = [10, 20, 30]
                if defect == "units": manifest["units"]["load_MW"] = "kW"
                if defect == "threshold_type": manifest["class_thresholds"] = []
                if defect == "availability": manifest["feature_availability"]["role"] = "post_solve_outputs"
                (root / "study_manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    load_csv_graphs("thermal", root, require_traceable=True)

    def test_demo_cannot_be_relabelled_traceable_without_manifest(self):
        with self.assertRaisesRegex(ValueError, "study_manifest"):
            load_csv_graphs("thermal", require_traceable=True)


if __name__ == "__main__":
    unittest.main()
