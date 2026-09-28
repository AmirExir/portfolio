"""Deterministic regressions for graph semantics and scenario evaluation."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from GNN.data_pipeline import build_graphs, thermal_graph, voltage_graph
from GNN.gnn_clean import _evaluate, split_scenarios, train_gnn_multi_graph
from torch_geometric.loader import DataLoader


def buses() -> pd.DataFrame:
    return pd.DataFrame({"bus": [10, 20, 30], "voltage": [-1.2, 0.1, 2.0],
                         "load_MW": [1.0, 2.0, 3.0], "p_inj_mw": [-1.0, -2.0, 3.0],
                         "voltage_class": [1, 2, 4], "scenario": [7, 7, 7]})


def edges() -> pd.DataFrame:
    return pd.DataFrame({"from_bus": [10, 20, 10], "to_bus": [20, 30, 30],
                         "in_service": [True, True, False], "x_pu": [0.1, 0.2, 0.3],
                         "length_km": [1.0, 2.0, 3.0], "loading_percent": [90, 120, 0],
                         "thermal_class": [1, 2, 0], "scenario": [7, 7, 7]})


def training_graphs() -> list[Data]:
    return [Data(x=torch.tensor([[float(i), 1.0], [float(i + 1), 1.0]]),
                 edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([0, 1]),
                 scenario_id=str(i), class_count=2, feature_names=["load", "constant"])
            for i in range(10)]


class DatasetTests(unittest.TestCase):
    def test_supplied_labels_and_no_target_features(self):
        bus_frame, edge_frame = buses(), edges()
        graph = voltage_graph(bus_frame, edge_frame, "7")
        self.assertEqual(graph.y.tolist(), [1, 2, 4])
        self.assertEqual(graph.x.tolist(), [[1, -1, 1], [2, -2, 2], [3, 3, 1]])
        bus_frame["voltage"] = 999
        torch.testing.assert_close(graph.x, voltage_graph(bus_frame, edge_frame, "7").x)

    def test_active_topology_is_bidirectional_and_source_is_preserved(self):
        source = edges()
        original = source.copy(deep=True)
        graph = voltage_graph(buses(), source, "7")
        self.assertEqual(set(map(tuple, graph.edge_index.t().tolist())), {(0, 1), (1, 0), (1, 2), (2, 1)})
        self.assertEqual(graph.excluded_open_branches, 1)
        pd.testing.assert_frame_equal(source, original)

    def test_thermal_does_not_learn_its_target_or_open_lines(self):
        source = edges()
        graph = thermal_graph(source, "7")
        self.assertEqual(graph.y.tolist(), [1, 2])
        self.assertEqual(graph.feature_names, ["x_pu", "length_km"])
        self.assertEqual(graph.edge_index.tolist(), [[0, 1], [1, 0]])
        source["loading_percent"] = 10000
        torch.testing.assert_close(graph.x, thermal_graph(source, "7").x)

    def test_parallel_lines_have_one_message_link_each_direction(self):
        source = edges().iloc[:2].copy()
        source["from_bus"], source["to_bus"] = 10, 20
        graph = thermal_graph(source, "7")
        self.assertEqual(graph.edge_index.tolist(), [[0, 1], [1, 0]])

    def test_invalid_inputs_are_not_silently_filled_or_dropped(self):
        cases = []
        missing = buses().drop(columns="p_inj_mw")
        cases.append((missing, edges()))
        duplicate = pd.concat([buses(), buses().iloc[:1]])
        cases.append((duplicate, edges()))
        bad_endpoint = edges(); bad_endpoint.loc[0, "from_bus"] = 999
        cases.append((buses(), bad_endpoint))
        unknown_status = edges().astype({"in_service": object}); unknown_status.loc[0, "in_service"] = "unknown"
        cases.append((buses(), unknown_status))
        bad_label = buses().astype({"voltage_class": float}); bad_label.loc[0, "voltage_class"] = 1.5
        cases.append((bad_label, edges()))
        nonfinite = buses(); nonfinite.loc[0, "load_MW"] = np.inf
        cases.append((nonfinite, edges()))
        for bus_frame, edge_frame in cases:
            with self.subTest(columns=list(bus_frame.columns)), self.assertRaises(ValueError):
                voltage_graph(bus_frame, edge_frame, "7")

    def test_scenario_mismatch_fails(self):
        source = edges(); source["scenario"] = 8
        with self.assertRaisesRegex(ValueError, "scenario IDs"):
            build_graphs(buses(), source, "voltage")

    def test_disconnected_buses_preserve_empty_topology(self):
        source = edges(); source["in_service"] = False
        graph = voltage_graph(buses(), source, "7")
        self.assertEqual(tuple(graph.edge_index.shape), (2, 0))
        self.assertEqual(graph.x[:, -1].tolist(), [0, 0, 0])


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_split_is_reproducible_and_groups_repeated_scenarios(self):
        graphs = training_graphs()
        graphs.append(graphs[0].clone())
        first, second = split_scenarios(graphs), split_scenarios(graphs)
        self.assertEqual(first, second)
        sets = [{graphs[i].scenario_id for i in indices} for indices in first.values()]
        self.assertTrue(all(not left & right for index, left in enumerate(sets) for right in sets[index + 1:]))
        self.assertEqual(sorted(i for indices in first.values() for i in indices), list(range(11)))

    def test_train_scaling_excludes_holdouts_and_source_is_unchanged(self):
        graphs = training_graphs()
        splits = split_scenarios(graphs)
        expected = torch.cat([graphs[i].x for i in splits["train"]]).mean(dim=0)
        for index in splits["test"] + splits["validation"]:
            graphs[index].x[:, 0] = 10000
        snapshots = [graph.x.clone() for graph in graphs]
        model, history = train_gnn_multi_graph(graphs, epochs=1)
        torch.testing.assert_close(model.feature_mean, expected)
        self.assertEqual(float(model.feature_scale[1]), 1.0)
        for source, snapshot in zip(graphs, snapshots):
            torch.testing.assert_close(source.x, snapshot)
        report = history.attrs["evaluation"]
        self.assertEqual(report["normalization"]["fit_split"], "train")
        self.assertEqual(sum(sum(row) for row in report["test"]["confusion_matrix"]), 4)
        self.assertEqual(report["majority_baseline"]["accuracy"], 0.5)
        self.assertIn("feature_mean", model.state_dict())

    def test_best_checkpoint_is_restored_before_test_metrics(self):
        snapshot = {}
        calls = []
        def fake_evaluation(model, loader, weights, device):
            calls.append(len(calls))
            if len(calls) == 1:
                snapshot.update({key: tensor.detach().clone() for key, tensor in model.state_dict().items()})
            if len(calls) == 3:
                for key, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, snapshot[key])
            return [0.1, 0.9, 0.2][len(calls) - 1], np.array([0, 1]), np.array([0, 1])
        with patch("GNN.gnn_clean._evaluate", side_effect=fake_evaluation):
            _, history = train_gnn_multi_graph(training_graphs(), epochs=2)
        self.assertEqual(len(calls), 3)
        self.assertEqual(history.attrs["evaluation"]["best_epoch"], 1)
        self.assertEqual(history.attrs["evaluation"]["validation"]["val_loss"], 0.1)

    def test_loss_aggregation_independent_of_evaluation_batch_size(self):
        graphs = training_graphs()
        model, _ = train_gnn_multi_graph(graphs, epochs=1)
        weights = torch.tensor([0.25, 0.75])
        first = _evaluate(model, DataLoader(graphs, batch_size=1), weights, "cpu")
        second = _evaluate(model, DataLoader(graphs, batch_size=3), weights, "cpu")
        self.assertAlmostEqual(first[0], second[0], places=6)
        np.testing.assert_equal(first[2], second[2])

    def test_invalid_training_inputs_fail_before_training(self):
        for kwargs in ({"epochs": 0}, {"batch_size": 0}, {"lr": -1}, {"weight_decay": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                train_gnn_multi_graph(training_graphs(), **kwargs)
        with self.assertRaisesRegex(ValueError, "five"):
            train_gnn_multi_graph(training_graphs()[:4], epochs=1)

    def test_all_four_architectures_train_and_infer(self):
        for name in ("gcn", "gat", "gin", "transformer"):
            with self.subTest(model=name):
                graphs = training_graphs()
                model, history = train_gnn_multi_graph(graphs, epochs=1, model_type=name)
                self.assertEqual(tuple(model(graphs[0].x, graphs[0].edge_index).shape), (2, 2))
                self.assertTrue(np.isfinite(history["val_loss"]).all())


if __name__ == "__main__":
    unittest.main()
