"""Validated scenario graphs from CSV; input feature values remain in source units.

The bundled CSVs contain preprocessed values with no inverse transform. Their
supplied class labels are preserved; voltage/loading are never predictor inputs.
"""
from pathlib import Path
import hashlib
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

Mode = Literal["voltage", "thermal"]
FEATURES = {
    "voltage": ("load_MW", "p_inj_mw", "active_branch_count"),
    "thermal": ("x_pu", "length_km"),
}
CLASS_COUNTS = {"voltage": 5, "thermal": 4}
SOURCE_NOTE = (
    "The bundled CSV features are already preprocessed and their original scaler "
    "and label thresholds are unavailable. Scores are exploratory: they do not "
    "establish physical accuracy or independence of upstream preprocessing. "
    "Voltage/loading measurements and their derived features are excluded from predictors."
)


def _require(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
    if frame.loc[:, list(columns)].isna().any().any():
        raise ValueError(f"{name} contains missing values in required columns")


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {', '.join(columns)}")
    return values


def _labels(frame: pd.DataFrame, column: str, classes: int) -> torch.Tensor:
    values = _numeric(frame, (column,)).ravel()
    if not np.equal(values, np.floor(values)).all() or ((values < 0) | (values >= classes)).any():
        raise ValueError(f"{column} must contain integer class IDs from 0 to {classes - 1}")
    return torch.tensor(values, dtype=torch.long)


def _active_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Honor explicit equipment status; do not invent status when absent."""
    _require(edges, ("from_bus", "to_bus", "in_service"), "edge data")
    status = edges["in_service"].map(lambda v: str(v).strip().lower())
    if not status.isin(("true", "false", "1", "0")).all():
        raise ValueError("in_service must contain booleans or 0/1 values")
    return edges.loc[status.isin(("true", "1"))]


def _edge_tensor(pairs: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous()


def voltage_graph(buses: pd.DataFrame, edges: pd.DataFrame, scenario_id: str) -> Data:
    """Build a bus graph with supplied labels and both directions of active branches."""
    _require(buses, ("bus", "load_MW", "p_inj_mw", "voltage_class"), "bus data")
    if buses.empty or buses["bus"].duplicated().any():
        raise ValueError("Each scenario must contain unique bus IDs and at least one bus")
    active = _active_edges(edges)
    mapping = {bus: index for index, bus in enumerate(buses["bus"])}
    # Validate even open branch endpoints so malformed source topology cannot disappear.
    if not set(edges["from_bus"]).union(edges["to_bus"]).issubset(mapping):
        raise ValueError("Branch endpoint references a bus missing from its scenario")
    pairs = []
    counts = np.zeros(len(buses))
    for start, end in active[["from_bus", "to_bus"]].itertuples(index=False, name=None):
        i, j = mapping[start], mapping[end]
        pairs.extend(((i, j), (j, i)))
        counts[i] += 1
        counts[j] += 1
    features = np.column_stack((_numeric(buses, ("load_MW", "p_inj_mw")), counts))
    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        y=_labels(buses, "voltage_class", CLASS_COUNTS["voltage"]),
        edge_index=_edge_tensor(pairs), scenario_id=str(scenario_id),
        feature_names=list(FEATURES["voltage"]), class_count=CLASS_COUNTS["voltage"],
        entity_ids=[str(bus) for bus in buses["bus"]],
        excluded_open_branches=len(edges) - len(active),
    )


def thermal_graph(edges: pd.DataFrame, scenario_id: str) -> Data:
    """Build an active-line graph; shared endpoints connect line nodes both ways."""
    _require(edges, ("x_pu", "length_km", "thermal_class"), "edge data")
    active = _active_edges(edges)
    if active.empty:
        raise ValueError(f"Scenario {scenario_id} has no active lines to classify")
    bus_to_lines: dict[object, set[int]] = {}
    entities = []
    for index, (row_id, row) in enumerate(active.iterrows()):
        for bus in (row["from_bus"], row["to_bus"]):
            bus_to_lines.setdefault(bus, set()).add(index)
        entities.append(f"{row['from_bus']} → {row['to_bus']} (source row {row_id})")
    pairs = set()
    for lines in bus_to_lines.values():
        pairs.update((left, right) for left in lines for right in lines if left != right)
    return Data(
        x=torch.tensor(_numeric(active, FEATURES["thermal"]), dtype=torch.float32),
        y=_labels(active, "thermal_class", CLASS_COUNTS["thermal"]),
        edge_index=_edge_tensor(sorted(pairs)), scenario_id=str(scenario_id),
        feature_names=list(FEATURES["thermal"]), class_count=CLASS_COUNTS["thermal"],
        entity_ids=entities, excluded_open_branches=len(edges) - len(active),
    )


def build_graphs(buses: pd.DataFrame, edges: pd.DataFrame, mode: Mode) -> list[Data]:
    """Build one graph per scenario without fitting any preprocessing statistics."""
    if mode not in FEATURES:
        raise ValueError(f"Unknown classification mode: {mode}")
    _require(edges, ("scenario",), "edge data")
    if edges.empty:
        raise ValueError("Edge data must contain at least one scenario")
    if mode == "voltage":
        _require(buses, ("scenario",), "bus data")
        if set(buses.scenario) != set(edges.scenario):
            raise ValueError("Bus and edge scenario IDs must match")
    graphs = []
    for scenario, scenario_edges in edges.groupby("scenario", sort=True):
        if mode == "voltage":
            graph = voltage_graph(buses.loc[buses.scenario == scenario], scenario_edges, str(scenario))
        else:
            graph = thermal_graph(scenario_edges, str(scenario))
        graphs.append(graph)
    return graphs


def load_csv_graphs(mode: Mode, directory: Path | str | None = None) -> list[Data]:
    """Load inspectable CSV sources, never deserialize a pickled graph artifact."""
    source = Path(directory) if directory is not None else Path(__file__).resolve().parent
    paths = [source / "edge_scenarios.csv"]
    if mode == "voltage":
        paths.append(source / "bus_scenarios.csv")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    edges = pd.read_csv(source / "edge_scenarios.csv")
    buses = pd.read_csv(source / "bus_scenarios.csv") if mode == "voltage" else pd.DataFrame()
    graphs = build_graphs(buses, edges, mode)
    for graph in graphs:
        graph.source_hashes = hashes
    return graphs
