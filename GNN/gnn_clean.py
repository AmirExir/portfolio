"""
GNN Training for Power Grid Violation Detection
Supports both voltage (bus-level) and thermal (line-level) classification
"""
import json
from datetime import datetime, timezone
import torch_geometric
from pathlib import Path
from typing import Callable
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, TransformerConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


try:
    from .data_pipeline import load_csv_graphs, SOURCE_NOTE
except ImportError:
    from data_pipeline import load_csv_graphs, SOURCE_NOTE


def set_seed(s: int = 42) -> None:
    """Set random seeds for reproducibility"""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class NormalizedModel(nn.Module):
    """Store training-only normalization with the model for consistent inference."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(in_dim))
        self.register_buffer("feature_scale", torch.ones(in_dim))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Transform source feature values using training statistics."""
        return (x - self.feature_mean) / self.feature_scale


class GCN(NormalizedModel):
    """Graph Convolutional Network for node classification"""
    def __init__(self, in_dim, num_classes=2, hidden=64, dropout=0.4, use_relu=True):
        super().__init__(in_dim)
        self.g1 = GCNConv(in_dim, hidden)
        self.g2 = GCNConv(hidden, hidden)
        self.do = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)
        self.use_relu = use_relu
        
    def forward(self, x, edge_index):
        x = self.normalize(x)
        x = self.g1(x, edge_index)
        if self.use_relu:
            x = torch.relu(x)
        x = self.do(x)
        x = self.g2(x, edge_index)
        if self.use_relu:
            x = torch.relu(x)
        x = self.do(x)
        return self.head(x)


class GAT(NormalizedModel):
    """Graph Attention Network for node classification"""
    def __init__(self, in_dim, num_classes=2, hidden=64, dropout=0.4, use_relu=True):
        super().__init__(in_dim)
        self.g1 = GATConv(in_dim, hidden, heads=2, dropout=dropout)
        self.g2 = GATConv(hidden * 2, hidden, heads=1, dropout=dropout)
        self.do = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)
        self.use_relu = use_relu
        
    def forward(self, x, edge_index):
        x = self.normalize(x)
        x = self.g1(x, edge_index)
        if self.use_relu:
            x = torch.relu(x)
        x = self.do(x)
        x = self.g2(x, edge_index)
        if self.use_relu:
            x = torch.relu(x)
        x = self.do(x)
        return self.head(x)


class GIN(NormalizedModel):
    """Graph Isomorphism Network for node classification"""
    def __init__(self, in_dim, num_classes=2, hidden=64, dropout=0.4, use_relu=True):
        super().__init__(in_dim)
        nn1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        nn2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.g1 = GINConv(nn1)
        self.g2 = GINConv(nn2)
        self.do = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)
        self.use_relu = use_relu
        
    def forward(self, x, edge_index):
        x = self.normalize(x)
        x = self.g1(x, edge_index)
        if self.use_relu:
            x = torch.relu(x)
        x = self.do(x)
        x = self.g2(x, edge_index)
        if self.use_relu:
            x = torch.relu(x)
        x = self.do(x)
        return self.head(x)


class GraphTransformer(NormalizedModel):
    """Graph Transformer for node classification"""
    def __init__(self, in_dim, num_classes=2, hidden=64, dropout=0.4, use_relu=True):
        super().__init__(in_dim)
        self.g1 = TransformerConv(in_dim, hidden, heads=2, dropout=dropout)
        self.g2 = TransformerConv(hidden * 2, hidden, heads=1, dropout=dropout)
        self.do = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)
        self.use_relu = use_relu
        
    def forward(self, x, edge_index):
        x = self.normalize(x)
        x = self.g1(x, edge_index)
        if self.use_relu:
            x = torch.relu(x)
        x = self.do(x)
        x = self.g2(x, edge_index)
        if self.use_relu:
            x = torch.relu(x)
        x = self.do(x)
        return self.head(x)


def split_scenarios(graph_list: list[Data], seed: int = 42) -> dict[str, list[int]]:
    """Partition complete scenario groups into approximately 60/20/20 percent."""
    groups: dict[str, list[int]] = {}
    for index, graph in enumerate(graph_list):
        if not hasattr(graph, "scenario_id"):
            raise ValueError("Each graph needs a scenario_id; rebuild from CSV with data_pipeline")
        groups.setdefault(str(graph.scenario_id), []).append(index)
    if len(groups) < 5:
        raise ValueError("At least five distinct scenarios are required for train/validation/test evaluation")
    identifiers = sorted(groups)
    order = np.random.default_rng(seed).permutation(len(identifiers))
    holdout_count = max(1, len(identifiers) // 5)
    assignments = {
        "test": order[:holdout_count],
        "validation": order[holdout_count:2 * holdout_count],
        "train": order[2 * holdout_count:],
    }
    return {name: [i for position in positions for i in groups[identifiers[position]]]
            for name, positions in assignments.items()}


def classification_metrics(truth: np.ndarray, prediction: np.ndarray, classes: int) -> dict:
    """Include all declared classes, supports, and confusion counts in evaluation."""
    labels = list(range(classes))
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision_weighted": float(precision_score(truth, prediction, labels=labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(truth, prediction, labels=labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(truth, prediction, labels=labels, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
        "per_class": classification_report(truth, prediction, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=labels).tolist(),
    }


def _validate_graphs(graph_list: list[Data]) -> tuple[int, int]:
    if not graph_list:
        raise ValueError("The graph dataset is empty")
    in_dim = graph_list[0].num_features
    classes = getattr(graph_list[0], "class_count", None)
    if not isinstance(classes, int) or classes < 2:
        raise ValueError("Graphs must declare class_count; rebuild from CSV with data_pipeline")
    for graph in graph_list:
        if graph.x.ndim != 2 or graph.x.shape[0] == 0 or graph.num_features != in_dim:
            raise ValueError("Graphs need nonempty, consistent feature matrices")
        if not torch.isfinite(graph.x).all():
            raise ValueError("Graph features must be finite")
        if (graph.y.dtype != torch.long or graph.y.ndim != 1 or len(graph.y) != graph.num_nodes
                or (graph.y < 0).any() or (graph.y >= classes).any()):
            raise ValueError("Graph labels must be integer class IDs aligned with nodes")
        if getattr(graph, "class_count", None) != classes:
            raise ValueError("All graphs must use the same class schema")
        if (graph.edge_index.dtype != torch.long or graph.edge_index.ndim != 2
                or graph.edge_index.shape[0] != 2):
            raise ValueError("edge_index must have shape (2, edges) and integer dtype")
        if graph.edge_index.numel() and ((graph.edge_index < 0).any() or (graph.edge_index >= graph.num_nodes).any()):
            raise ValueError("Graph edge index references a missing node")
    return in_dim, classes


def _evaluate(model: nn.Module, loader: DataLoader, weights: torch.Tensor, device: str) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum, denominator = 0.0, 0.0
    predictions, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            loss_sum += F.cross_entropy(logits, batch.y, weight=weights, reduction="sum").item()
            denominator += weights[batch.y].sum().item()
            predictions.append(logits.argmax(dim=-1).cpu())
            labels.append(batch.y.cpu())
    return loss_sum / denominator, torch.cat(labels).numpy(), torch.cat(predictions).numpy()


def train_gnn_multi_graph(
    graph_list: list[Data], epochs: int = 100, lr: float = 1e-3,
    weight_decay: float = 5e-4, seed: int = 42, use_relu: bool = True,
    batch_size: int = 32, model_type: str = "gcn",
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[nn.Module, pd.DataFrame]:
    """Select a checkpoint on validation scenarios and evaluate test scenarios once.

    Returns the existing (model, history) pair. history.attrs['evaluation'] stores
    held-out and majority-baseline metrics, split IDs, preprocessing, and settings.
    Models accept unscaled source features; normalization lives in model buffers.
    """
    if epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0 or not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("epochs/batch_size must be positive, lr positive, and weight_decay nonnegative")
    in_dim, classes = _validate_graphs(graph_list)
    splits = split_scenarios(graph_list, seed)
    set_seed(seed)
    # Keep reporting metadata out of PyG collation; do not modify source graphs.
    datasets = {name: [Data(x=graph_list[i].x, y=graph_list[i].y,
                           edge_index=graph_list[i].edge_index) for i in indices]
                for name, indices in splits.items()}
    loaders = {name: DataLoader(graphs, batch_size=batch_size, shuffle=name == "train")
               for name, graphs in datasets.items()}
    architectures = {"gcn": GCN, "gat": GAT, "gin": GIN, "transformer": GraphTransformer}
    if model_type.lower() not in architectures:
        raise ValueError(f"Unknown model_type '{model_type}'")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = architectures[model_type.lower()](in_dim, num_classes=classes, use_relu=use_relu)
    train_features = torch.cat([graph.x for graph in datasets["train"]])
    model.feature_mean.copy_(train_features.mean(dim=0))
    scale = train_features.std(dim=0, unbiased=False)
    model.feature_scale.copy_(torch.where(scale > 0, scale, torch.ones_like(scale)))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_labels = torch.cat([graph.y for graph in datasets["train"]])
    counts = torch.bincount(train_labels, minlength=classes)
    weights = (1.0 / counts.clamp_min(1).float()).to(device)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        numerator, denominator = 0.0, 0.0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(logits, batch.y, weight=weights)
            if not torch.isfinite(loss):
                raise ValueError("Training produced non-finite loss; inspect source features and learning rate")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_weight = weights[batch.y].sum().item()
            numerator += loss.item() * batch_weight
            denominator += batch_weight
        val_loss, truth, predictions = _evaluate(model, loaders["validation"], weights, device)
        if not np.isfinite(val_loss):
            raise ValueError("Validation produced non-finite loss")
        metrics = classification_metrics(truth, predictions, classes)
        history.append({"epoch": epoch, "train_loss": numerator / denominator, "val_loss": val_loss,
                        "val_acc": metrics["accuracy"], "val_prec": metrics["precision_weighted"],
                        "val_rec": metrics["recall_weighted"], "val_f1": metrics["f1_weighted"],
                        "val_f1_macro": metrics["f1_macro"]})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if progress_callback is not None:
            progress_callback(epoch, epochs)
    model.load_state_dict(best_state)
    test_loss, truth, predictions = _evaluate(model, loaders["test"], weights, device)
    baseline_class = int(counts.argmax())
    evaluation = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_hashes": getattr(graph_list[0], "source_hashes", {}),
        "versions": {"torch": torch.__version__, "torch_geometric": torch_geometric.__version__, "numpy": np.__version__},
        "best_epoch": best_epoch,
        "validation": history[best_epoch - 1],
        "test": {"loss": test_loss, **classification_metrics(truth, predictions, classes)},
        "majority_baseline": {"class": baseline_class, **classification_metrics(truth, np.full_like(truth, baseline_class), classes)},
        "splits": {name: sorted({str(graph_list[i].scenario_id) for i in indices}) for name, indices in splits.items()},
        "feature_names": getattr(graph_list[0], "feature_names", []),
        "normalization": {"mean": model.feature_mean.cpu().tolist(), "scale": model.feature_scale.cpu().tolist(), "fit_split": "train"},
        "missing_training_classes": torch.where(counts == 0)[0].tolist(),
        "settings": {"model": model_type.lower(), "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
                     "seed": seed, "batch_size": batch_size, "use_relu": use_relu},
        "source_limitations": SOURCE_NOTE,
    }
    history_frame = pd.DataFrame(history)
    history_frame.attrs["evaluation"] = evaluation
    return model, history_frame


def main() -> None:
    """Train from CSV and report validation-selected, held-out test metrics."""
    parser = argparse.ArgumentParser(description="Exploratory GNN classification with scenario holdouts")
    parser.add_argument("--mode", choices=["voltage", "thermal"], default="voltage")
    parser.add_argument("--model", choices=["gcn", "gat", "gin", "transformer"], default="gcn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_relu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--report", type=Path, help="Write an evaluation JSON file at a new path")
    args = parser.parse_args()
    graphs = load_csv_graphs(args.mode, args.data_dir)
    _, history = train_gnn_multi_graph(graphs, args.epochs, args.lr, args.weight_decay,
                                       args.seed, args.use_relu, args.batch_size, args.model)
    report = history.attrs["evaluation"]
    print(SOURCE_NOTE)
    print(f"Best validation checkpoint: epoch {report['best_epoch']}")
    print(f"Held-out test accuracy: {report['test']['accuracy']:.2%}; macro F1: {report['test']['f1_macro']:.4f}")
    print(f"Training-majority baseline test accuracy: {report['majority_baseline']['accuracy']:.2%}; macro F1: {report['majority_baseline']['f1_macro']:.4f}")
    if args.report:
        with args.report.open("x") as output:
            json.dump(report, output, indent=2)
        print(f"Evaluation report: {args.report}")


if __name__ == "__main__":
    main()
