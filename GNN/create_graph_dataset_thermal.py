"""Export active-line graphs, excluding loading_percent from predictor features."""
import argparse
from pathlib import Path

import torch
from torch_geometric.data import Data

try:
    from .data_pipeline import load_csv_graphs
except ImportError:
    from data_pipeline import load_csv_graphs


def create_thermal_graphs(
    data_dir: Path | str | None = None, output_path: Path | str | None = None
) -> list[Data]:
    """Build validated graphs; optionally export to a new, nonexisting artifact."""
    graphs = load_csv_graphs("thermal", data_dir)
    if output_path is not None:
        with Path(output_path).open("xb") as output:
            torch.save(graphs, output)
    return graphs


def main() -> None:
    """Preserve original artifacts and export an explicitly versioned dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "graph_scenarios_thermal_v2.pt")
    args = parser.parse_args()
    graphs = create_thermal_graphs(args.data_dir, args.output)
    print(f"Saved {len(graphs)} unscaled scenario graphs to {args.output}; only load pickle artifacts you trust.")


if __name__ == "__main__":
    main()
