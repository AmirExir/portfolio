"""Export validated bus graphs without target features or globally fitted scaling."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

try:
    from .data_pipeline import load_csv_graphs, voltage_graph
except ImportError:
    from data_pipeline import load_csv_graphs, voltage_graph


def voltage_to_class(v: float) -> int:
    """Apply the historical five-bin example only to a finite raw voltage in pu.

    These bins are demonstration categories, not governing voltage criteria.
    CSV import preserves voltage_class and never calls this helper.
    """
    if not np.isfinite(v) or v <= 0:
        raise ValueError("Expected a finite positive physical voltage in pu")
    return int(np.searchsorted([0.95, 0.98, 1.00, 1.02], v, side="right"))


def create_graph_for_scenario(
    bus_sub: pd.DataFrame, edge_sub: pd.DataFrame, scaler: object | None = None
) -> tuple[Data, None]:
    """Compatibility entry point; return unscaled features and supplied class labels."""
    if scaler is not None:
        raise ValueError("Dataset-wide scaling is unsupported; training fits its own scaler on training scenarios")
    scenario_id = str(bus_sub["scenario"].iloc[0]) if "scenario" in bus_sub and len(bus_sub) else "0"
    return voltage_graph(bus_sub, edge_sub, scenario_id), None


def main() -> None:
    """Export new artifacts while preserving the original checked-in datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "graph_scenarios_v2.pt")
    args = parser.parse_args()
    graphs = load_csv_graphs("voltage", args.data_dir)
    with args.output.open("xb") as output:
        torch.save(graphs, output)
    print(f"Saved {len(graphs)} unscaled scenario graphs to {args.output}; only load pickle artifacts you trust.")


if __name__ == "__main__":
    main()
