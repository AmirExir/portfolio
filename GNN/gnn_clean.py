"""
GNN Training for Power Grid Violation Detection
Supports both voltage (bus-level) and thermal (line-level) classification
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def set_seed(s=42):
    """Set random seeds for reproducibility"""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class GCN(nn.Module):
    """Graph Convolutional Network for node classification"""
    def __init__(self, in_dim, num_classes=2, hidden=64, dropout=0.4, use_relu=True):
        super().__init__()
        self.g1 = GCNConv(in_dim, hidden)
        self.g2 = GCNConv(hidden, hidden)
        self.do = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)
        self.use_relu = use_relu
        
    def forward(self, x, edge_index):
        x = self.g1(x, edge_index)
        if self.use_relu:
            x = torch.relu(x)
        x = self.do(x)
        x = self.g2(x, edge_index)
        if self.use_relu:
            x = torch.relu(x)
        x = self.do(x)
        return self.head(x)


def train_gnn_multi_graph(graph_list, epochs=100, lr=1e-3, weight_decay=5e-4, 
                          seed=42, use_relu=True, batch_size=32):
    """
    Train GNN on multiple graphs using DataLoader.
    
    Args:
        graph_list: List of PyG Data objects (one per scenario)
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        seed: Random seed for reproducibility
        use_relu: Use ReLU activation in GCN
        batch_size: Batch size for DataLoader
        
    Returns:
        model: Trained GCN model
        hist_df: Training history DataFrame
    """
    set_seed(seed)
    
    # Split graphs into train/val (70/30 split)
    train_graphs, val_graphs = train_test_split(
        graph_list, train_size=0.7, random_state=seed, shuffle=True
    )
    
    print(f"Training on {len(train_graphs)} graphs, validating on {len(val_graphs)} graphs")
    
    # Create DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    
    # Get number of classes and features from first graph
    n_classes = int(max(g.y.max().item() for g in graph_list)) + 1
    in_dim = graph_list[0].num_features
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model
    model = GCN(in_dim=in_dim, num_classes=n_classes, hidden=64, use_relu=use_relu).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Compute class weights from training graphs (handle class imbalance)
    all_train_labels = torch.cat([g.y for g in train_graphs])
    counts = torch.bincount(all_train_labels, minlength=n_classes).float()
    alpha = 1.0 / (counts + 1e-6)
    alpha = (alpha / alpha.sum()).to(device)
    
    history = []
    best = (1e9, None)  # (val_loss, state_dict)
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(logits, batch.y, weight=alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses, val_preds, val_trues = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index)
                loss = F.cross_entropy(logits, batch.y, weight=alpha)
                val_losses.append(loss.item())
                val_preds.append(logits.argmax(dim=-1).cpu())
                val_trues.append(batch.y.cpu())
        
        val_preds = torch.cat(val_preds).numpy()
        val_trues = torch.cat(val_trues).numpy()
        
        # Calculate metrics
        val_acc = accuracy_score(val_trues, val_preds)
        val_prec = precision_score(val_trues, val_preds, average='weighted', zero_division=0)
        val_rec = recall_score(val_trues, val_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(val_trues, val_preds, average='weighted', zero_division=0)
        val_f1_macro = f1_score(val_trues, val_preds, average='macro', zero_division=0)
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_prec': val_prec,
            'val_rec': val_rec,
            'val_f1': val_f1,
            'val_f1_macro': val_f1_macro
        })
        
        # Save best model based on validation loss
        if val_loss < best[0]:
            best = (val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()})
    
    # Load best model
    if best[1] is not None:
        model.load_state_dict(best[1])
    
    hist_df = pd.DataFrame(history)
    return model, hist_df


def main():
    parser = argparse.ArgumentParser(description='Train GNN for power grid violation detection')
    parser.add_argument("--mode", choices=["voltage", "thermal"], default="voltage",
                        help="Classification mode: voltage (bus-level) or thermal (line-level)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="L2 regularization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_relu", action="store_true", help="Use ReLU activation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Running in {args.mode.upper()} classification mode")
    print(f"{'='*60}\n")
    
    # Select appropriate dataset file based on mode
    if args.mode == "thermal":
        pt_path = "graph_scenarios_thermal.pt"
    else:
        pt_path = "graph_scenarios.pt"
    
    # Check if dataset exists
    if not os.path.exists(pt_path):
        print(f"❌ Error: Dataset file '{pt_path}' not found!")
        print(f"\nPlease generate the dataset first:")
        if args.mode == "thermal":
            print("  python create_graph_dataset_thermal.py")
        else:
            print("  python create_graph_dataset.py")
        return
    
    # Load preprocessed graph data
    print(f"📂 Loading {pt_path}...")
    data = torch.load(pt_path, weights_only=False)
    
    if not isinstance(data, list):
        print(f"❌ Error: Expected list of graphs, got {type(data)}")
        return
    
    print(f"✓ Loaded {len(data)} graph objects (one per scenario)")
    print(f"  Each graph: {data[0].num_nodes} nodes, {data[0].num_features} features")
    
    # Check class distribution
    all_labels = torch.cat([g.y for g in data])
    unique, counts = torch.unique(all_labels, return_counts=True)
    class_dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"  Class distribution: {class_dist}")
    
    # Train model
    print(f"\n🚀 Training GNN...")
    model, hist_df = train_gnn_multi_graph(
        data,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        use_relu=args.use_relu,
        batch_size=args.batch_size
    )
    
    # Print training results
    print(f"\n{'='*60}")
    print("Training Results")
    print(f"{'='*60}")
    print("\nEpoch | Train Loss | Val Loss | Val Acc | Val Prec | Val Rec | Val F1 | Val F1_macro")
    print("-" * 85)
    for i, row in hist_df.iterrows():
        if int(row["epoch"]) % 10 == 0 or int(row["epoch"]) == args.epochs:
            print(
                f"{int(row['epoch']):4d} | {row['train_loss']:10.4f} | {row['val_loss']:8.4f} | "
                f"{row['val_acc']:7.4f} | {row['val_prec']:8.4f} | {row['val_rec']:7.4f} | "
                f"{row['val_f1']:6.4f} | {row['val_f1_macro']:12.4f}"
            )
    
    # Final performance
    final = hist_df.iloc[-1]
    print(f"\n{'='*60}")
    print("Final Performance (Best Model)")
    print(f"{'='*60}")
    print(f"  Accuracy:       {final['val_acc']:.2%}")
    print(f"  Precision:      {final['val_prec']:.2%}")
    print(f"  Recall:         {final['val_rec']:.2%}")
    print(f"  F1 Score:       {final['val_f1']:.2%}")
    print(f"  Macro F1:       {final['val_f1_macro']:.2%}")
    print(f"{'='*60}\n")
    
    print("✓ Training complete! Model ready for evaluation or export.")


if __name__ == "__main__":
    main()
