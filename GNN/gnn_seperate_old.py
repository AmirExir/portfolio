import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GINConv, TransformerConv
from torch_geometric.utils import add_self_loops
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# -----------------------------
# Helpers
# -----------------------------
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def focal_loss(logits, targets, gamma=2.0, alpha=None):
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()

# ---- Model Selector ----
def get_model(model_type, in_dim, h_dim=64, num_classes=2, dropout=0.4, use_relu=True):
    """
    Returns an instance of the selected GNN model type.
    Supported: 'gcn', 'gat', 'gin', 'transformer'
    """
    if model_type.lower() == "gcn":
        class GCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.g1 = GCNConv(in_dim, h_dim)
                self.g2 = GCNConv(h_dim, h_dim)
                self.do = nn.Dropout(dropout)
                self.head = nn.Linear(h_dim, num_classes)
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
        return GCN()
    elif model_type.lower() == "gat":
        class GAT(nn.Module):
            def __init__(self):
                super().__init__()
                self.g1 = GATConv(in_dim, h_dim, heads=2, dropout=dropout)
                self.g2 = GATConv(h_dim * 2, h_dim, heads=1, dropout=dropout)
                self.do = nn.Dropout(dropout)
                self.head = nn.Linear(h_dim, num_classes)
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
        return GAT()
    elif model_type.lower() == "gin":
        class GIN(nn.Module):
            def __init__(self):
                super().__init__()
                nn1 = nn.Sequential(nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))
                nn2 = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))
                self.g1 = GINConv(nn1)
                self.g2 = GINConv(nn2)
                self.do = nn.Dropout(dropout)
                self.head = nn.Linear(h_dim, num_classes)
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
        return GIN()
    elif model_type.lower() == "transformer":
        class Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.g1 = TransformerConv(in_dim, h_dim, heads=2, dropout=dropout)
                self.g2 = TransformerConv(h_dim * 2, h_dim, heads=1, dropout=dropout)
                self.do = nn.Dropout(dropout)
                self.head = nn.Linear(h_dim, num_classes)
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
        return Transformer()
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Supported: gcn, gat, gin, transformer")


def _stratified_indices(y_np, train_frac=0.7, seed=42):
    try:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
        (train_idx_np, val_idx_np), = sss.split(np.zeros_like(y_np), y_np)
        return train_idx_np, val_idx_np
    except ValueError as e:
        if "The least populated class in y has only" in str(e):
            # Fallback: random split
            from sklearn.model_selection import train_test_split
            idx = np.arange(len(y_np))
            train_idx_np, val_idx_np = train_test_split(idx, train_size=train_frac, random_state=seed, shuffle=True)
            return train_idx_np, val_idx_np
        else:
            raise

# -----------------------------
# GCN Model Definition (Standalone)
# -----------------------------
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

def train_gnn_multi_graph(graph_list, epochs=300, lr=1e-2, weight_decay=5e-4, seed=42, use_relu=True, batch_size=32):
    """
    Train GNN on multiple graphs using DataLoader.
    Args:
        graph_list: List of PyG Data objects
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        seed: Random seed
        use_relu: Use ReLU activation
        batch_size: Batch size for DataLoader
    Returns:
        model, history_df, best_threshold
    """
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import train_test_split
    
    set_seed(seed)
    
    # Split graphs into train/val
    train_graphs, val_graphs = train_test_split(
        graph_list, 
        train_size=0.7, 
        random_state=seed,
        shuffle=True
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
    
    # Compute class weights from training graphs
    all_train_labels = torch.cat([g.y for g in train_graphs])
    counts = torch.bincount(all_train_labels, minlength=n_classes).float()
    alpha = 1.0 / (counts + 1e-6)
    alpha = (alpha / alpha.sum()).to(device)
    
    history = []
    best = (1e9, None)  # (val_loss, state_dict)
    
    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index)
            # Use regular cross-entropy with class weights instead of focal loss for stability
            loss = F.cross_entropy(logits, batch.y, weight=alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_losses.append(loss.item())
        
        # ---- Validate ----
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
        
        # Save best model
        if val_loss < best[0]:
            best = (val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()})
    
    # Load best model (if any)
    if best[1] is not None:
        model.load_state_dict(best[1])
    hist_df = pd.DataFrame(history)
    
    return model, hist_df, None

def train_gnn(data, epochs=300, lr=1e-2, weight_decay=5e-4, seed=42, use_relu=True):
    set_seed(seed)
    n_classes = int(data.y.max().item()) + 1
    model = GCN(in_dim=data.x.size(1), num_classes=n_classes, hidden=64, use_relu=use_relu).to(data.x.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Stratified split so minority class appears in both sets
    y_np = data.y.cpu().numpy()
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)
    (train_idx_np, val_idx_np), = sss.split(np.zeros_like(y_np), y_np)
    train_idx = torch.tensor(train_idx_np, dtype=torch.long, device=data.x.device)
    val_idx   = torch.tensor(val_idx_np,   dtype=torch.long, device=data.x.device)

    # FOCAL LOSS alpha from TRAIN ONLY
    counts_t = torch.bincount(data.y[train_idx], minlength=n_classes).float()
    alpha = 1.0 / (counts_t + 1e-6)
    alpha = (alpha / alpha.sum()).to(data.x.device)

    history = []
    best = (1e9, None)  # (val_loss, state_dict)

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        logits = model(data.x, data.edge_index)
        loss = focal_loss(
            logits[train_idx],
            data.y[train_idx],
            gamma=2.0,
            alpha=alpha
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        # ---- Eval ----
        model.eval()
        with torch.no_grad():
            logits_val = model(data.x, data.edge_index)[val_idx]

        # validation loss with focal loss
        val_loss = focal_loss(
            logits_val,
            data.y[val_idx],
            gamma=2.0,
            alpha=alpha
        ).item()

        # metrics: use argmax for logging (UI thresholding happens later)
        preds_t = torch.argmax(logits_val, dim=-1)
        yv_t    = data.y[val_idx]

        preds = preds_t.detach().cpu().numpy().astype(int)
        yv    = yv_t.detach().cpu().numpy().astype(int)

        acc  = accuracy_score(yv, preds)
        prec = precision_score(yv, preds, average='macro', zero_division=0)
        rec  = recall_score(yv, preds, average='macro', zero_division=0)
        f1   = f1_score(yv, preds, average='macro')
        f1m  = f1  # macro F1

        history.append((epoch, float(loss.item()), val_loss, acc, prec, rec, f1, f1m))

        if val_loss < best[0]:
            best = (val_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    # Restore best model
    if best[1] is not None:
        model.load_state_dict(best[1])

    # For multiclass, thresholding is not used; just return None for best_th
    best_th = None

    # Convert history to DataFrame for plotting
    hist_df = pd.DataFrame(
        history,
        columns=["epoch", "train_loss", "val_loss", "val_acc", "val_prec", "val_rec", "val_f1", "val_f1_macro"]
    )

    return model, hist_df, train_idx, val_idx, best_th


def to_pyg(edge_index_np, Xn, y):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    x = torch.tensor(Xn, dtype=torch.float, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


# -----------------------------
# Helper: sanitize PyG Data
# -----------------------------
def sanitize_pyg_data(data, add_loops_if_empty=True, verbose_prefix=""):
    """
    Ensure edge_index is valid w.r.t. data.x; remap nodes to 0..N-1; filter/realign train/val indices.
    Operates in-place and returns the same data object.
    """
    # If there are no edges, optionally add self-loops to keep GCN stable
    if data.edge_index is None or data.edge_index.numel() == 0:
        if add_loops_if_empty:
            data.edge_index, _ = add_self_loops(torch.zeros((2, 0), dtype=torch.long, device=data.x.device),
                                                 num_nodes=data.x.size(0))
        return data

    # Filter any out-of-range edges
    max_idx = data.x.size(0)
    mask = (data.edge_index[0] >= 0) & (data.edge_index[1] >= 0) & \
           (data.edge_index[0] < max_idx) & (data.edge_index[1] < max_idx)
    if (~mask).any():
        invalid = int((~mask).sum().item())
        data.edge_index = data.edge_index[:, mask]
        if invalid > 0:
            print(f"⚠️  {verbose_prefix}filtered {invalid} invalid edges; continuing with {data.edge_index.size(1)} edges.")

    # If no edges remain, add self-loops if requested
    if data.edge_index.size(1) == 0:
        if add_loops_if_empty:
            data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
        return data

    # Remap node indices appearing in edges to a consecutive 0..K-1 range
    unique_nodes = torch.unique(data.edge_index)
    old_to_new = {int(n): i for i, n in enumerate(unique_nodes.tolist())}

    # Remap edge_index
    data.edge_index = torch.tensor(
        [[old_to_new[int(s.item())] for s in data.edge_index[0]],
         [old_to_new[int(t.item())] for t in data.edge_index[1]]],
        dtype=torch.long,
        device=data.edge_index.device
    )

    # Slice features/labels to only nodes present in edges
    data.x = data.x[unique_nodes]
    if hasattr(data, "y") and data.y is not None:
        data.y = data.y[unique_nodes]

    # Remap/clip train & val indices if they exist
    def _remap_idx(idx_tensor, name):
        if not hasattr(data, idx_tensor):
            return
        idx = getattr(data, idx_tensor)
        if idx is None:
            return
        kept = []
        for el in idx.tolist():
            if int(el) in old_to_new:
                kept.append(old_to_new[int(el)])
        if len(kept) == 0:
            setattr(data, idx_tensor, torch.empty((0,), dtype=torch.long, device=data.x.device))
        else:
            setattr(data, idx_tensor, torch.tensor(kept, dtype=torch.long, device=data.x.device))

    _remap_idx("train_idx", "train")
    _remap_idx("val_idx", "val")
    return data


# -----------------------------
# Scenario-wise GNN Surrogate Pipeline
# -----------------------------

def load_bus_edge_csvs(bus_path="bus_scenarios.csv", edge_path="edge_scenarios.csv"):
    bus_df = pd.read_csv(bus_path)
    edge_df = pd.read_csv(edge_path)
    bus_df = bus_df.dropna().reset_index(drop=True)
    edge_df = edge_df.dropna().reset_index(drop=True)
    return bus_df, edge_df

# --- Build global graph: bus__scenario labels so each scenario is a component ---
def make_global_graph(bus_df, edge_df, mode="voltage"):
    """
    Builds a global graph for either voltage or thermal classification.
    mode: "voltage" or "thermal"
    Returns edge_index, features, y, scaler, index mapping, scenario array, bus_df, edge_df
    """
    bus_df = bus_df.copy()
    edge_df = edge_df.copy()
    assert "bus" in bus_df.columns and "scenario" in bus_df.columns

    # --- Oversample voltage classes for balance ---
    if mode == "voltage" and "voltage" in bus_df.columns:
        def voltage_to_class(v):
            if v < 0.95: return 0
            elif 0.95 <= v < 0.98: return 1
            elif 0.98 <= v < 1.00: return 2
            elif 1.00 <= v < 1.02: return 3
            else: return 4
        bus_df["voltage_class"] = bus_df["voltage"].apply(voltage_to_class)
        counts = bus_df["voltage_class"].value_counts()
        max_count = counts.max()
        from sklearn.utils import resample
        bus_df_balanced = []
        for c in counts.index:
            subset = bus_df[bus_df["voltage_class"] == c]
            bus_df_balanced.append(
                resample(subset, replace=True, n_samples=max_count, random_state=42)
            )
        bus_df = pd.concat(bus_df_balanced).reset_index(drop=True)
        # --- Remove edges with missing bus references after oversampling ---
        valid_buses = set(bus_df["bus"].astype(str))
        edge_df = edge_df[edge_df["from_bus"].astype(str).isin(valid_buses) & edge_df["to_bus"].astype(str).isin(valid_buses)].reset_index(drop=True)
        # --- Rebuild mapping after filtering edges ---
        bus_df = bus_df.reset_index(drop=True)
        bus_to_idx = {b: i for i, b in enumerate(bus_df["bus"].astype(str))}
        edge_df["from_idx"] = edge_df["from_bus"].astype(str).map(bus_to_idx)
        edge_df["to_idx"]   = edge_df["to_bus"].astype(str).map(bus_to_idx)
        # Remove edges with unmapped buses
        edge_df = edge_df.dropna(subset=["from_idx", "to_idx"]).astype({"from_idx": int, "to_idx": int}).reset_index(drop=True)

    if mode == "voltage":
        # --- Add neighbor_count column ---
        bus_df["neighbor_count"] = bus_df["bus"].map(
            edge_df["from_bus"].value_counts().add(edge_df["to_bus"].value_counts(), fill_value=0)
        ).fillna(0)
        # --- Ensure all cheat features exist; fill with zeros if missing ---
        for feat in ["voltage", "load_MW", "p_inj_mw", "neighbor_count"]:
            if feat not in bus_df.columns:
                bus_df[feat] = 0.0
        # --- Define voltage_class for 5 classes based on voltage ranges ---
        def voltage_to_class(v):
            if v < 0.95:
                return 0
            elif 0.95 <= v < 0.98:
                return 1
            elif 0.98 <= v < 1.00:
                return 2
            elif 1.00 <= v < 1.02:
                return 3
            else:
                return 4
        # If voltage_class not already defined, create it using voltage
        bus_df["voltage_class"] = bus_df["voltage"].apply(voltage_to_class)
        y = bus_df["voltage_class"].fillna(2).to_numpy().astype(int)
        # Features: voltage, load_MW, p_inj_mw, neighbor_count
        features = [c for c in ["voltage", "load_MW", "p_inj_mw", "neighbor_count"] if c in bus_df.columns]
        X = bus_df[features].to_numpy(dtype=float)
        scaler = StandardScaler().fit(X)
        Xn = scaler.transform(X)
        # Edge index: bus-based
        bus_df["bus_scen"] = bus_df["bus"].astype(str) + "__" + bus_df["scenario"].astype(str)
        edge_df["from_bus_scen"] = edge_df["from_bus"].astype(str) + "__" + edge_df["scenario"].astype(str)
        edge_df["to_bus_scen"]   = edge_df["to_bus"].astype(str) + "__" + edge_df["scenario"].astype(str)

        # --- Robust reindexing fix (only keep edges whose endpoints exist in bus_df) ---
        bus_to_idx = {b: i for i, b in enumerate(bus_df["bus_scen"])}
        edge_df = edge_df[
            edge_df["from_bus_scen"].isin(bus_to_idx.keys()) &
            edge_df["to_bus_scen"].isin(bus_to_idx.keys())
        ].reset_index(drop=True)

        src = edge_df["from_bus_scen"].map(bus_to_idx).to_numpy(dtype=int)
        dst = edge_df["to_bus_scen"].map(bus_to_idx).to_numpy(dtype=int)

        num_nodes = len(bus_df)
        mask_in_bounds = (src >= 0) & (dst >= 0) & (src < num_nodes) & (dst < num_nodes)
        invalid_edges = int((~mask_in_bounds).sum())
        if invalid_edges > 0:
            print(f"⚠️  make_global_graph: corrected {invalid_edges} edges referencing invalid nodes.")
        src = src[mask_in_bounds]
        dst = dst[mask_in_bounds]

        edge_index = np.vstack([src, dst])

        # If graph loses all edges, fall back to self-loops so GCNConv remains stable
        if edge_index.size == 0:
            idx = np.arange(num_nodes)
            edge_index = np.vstack([idx, idx])

        scenario_arr = bus_df["scenario"].to_numpy().astype(int)
        return edge_index, Xn, y, scaler, bus_to_idx, scenario_arr, bus_df, edge_df
    elif mode == "thermal":
        # --- Handle thermal_class column creation if missing ---
        if "thermal_class" not in edge_df.columns:
            edge_df["thermal_class"] = 0
        # Target: thermal_class (convert to int if not already)
        if "thermal_class" in edge_df.columns:
            y = edge_df["thermal_class"]
            if not np.issubdtype(y.dtype, np.integer):
                y = y.astype(str)
                classes = sorted(y.unique())
                class_map = {cls: idx for idx, cls in enumerate(classes)}
                y = y.map(class_map)
            y = y.fillna(0).to_numpy().astype(int)
        else:
            y = np.zeros(len(edge_df), dtype=int)
        # Features: x_pu, length_km, loading_percent (drop missing columns safely)
        features = [col for col in ["x_pu", "length_km", "loading_percent"] if col in edge_df.columns]
        if len(features) == 0:
            # fallback: all numeric columns except target and indexes
            features = [c for c in edge_df.columns if c not in ["thermal_class", "from_bus", "to_bus", "scenario", "from_bus_scen", "to_bus_scen"] and pd.api.types.is_numeric_dtype(edge_df[c])]
        if len(features) == 0:
            # fallback: zeros
            X = np.zeros((len(edge_df), 1))
        else:
            X = edge_df[features].to_numpy(dtype=float)
        scaler = StandardScaler().fit(X)
        Xn = scaler.transform(X)
        # Edge index: line-based (from_bus_scen <-> to_bus_scen for each edge)
        edge_df["from_bus_scen"] = edge_df["from_bus"].astype(str) + "__" + edge_df["scenario"].astype(str)
        edge_df["to_bus_scen"]   = edge_df["to_bus"].astype(str) + "__" + edge_df["scenario"].astype(str)
        # Each edge is a node, so assign an index to each edge row
        edge_to_idx = {i: i for i in range(len(edge_df))}
        # For GNN, create adjacency by connecting edges that share a bus within each scenario (line-graph)
        neighbors = []
        for scen, group in edge_df.groupby("scenario"):
            bus_map = {}
            for idx, row in group.iterrows():
                for b in [row["from_bus_scen"], row["to_bus_scen"]]:
                    bus_map.setdefault(b, []).append(idx)
            for edges in bus_map.values():
                if len(edges) > 1:
                    for a in edges:
                        for b in edges:
                            if a != b:
                                neighbors.append((a, b))
        if len(neighbors) == 0:
            edge_index = np.zeros((2, 0), dtype=int)
        else:
            edge_index = np.array(neighbors).T
        # --- Final Safety Filter: ensure valid edge indices ---
        num_nodes = len(edge_df)
        edge_index = edge_index[:, (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)]
        edge_index = np.clip(edge_index, 0, num_nodes - 1)
        scenario_arr = edge_df["scenario"].to_numpy().astype(int)
        return edge_index, Xn, y, scaler, edge_to_idx, scenario_arr, bus_df, edge_df
    else:
        raise ValueError(f"Unknown mode {mode}. Choose 'voltage' or 'thermal'.")


# -----------------------------
# Main function for CLI/VS Code execution
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["voltage", "thermal"], default="voltage",
                        help="Choose whether to train on voltage_class or thermal_class")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--model", choices=["gcn", "gat", "gin", "transformer"], default="gcn", help="Model architecture to use")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_relu", action="store_true", help="Use ReLU activation")
    args = parser.parse_args()
    mode = args.mode
    model_type = args.model
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    seed = args.seed
    use_relu = args.use_relu
    print(f"Running in {mode.upper()} classification mode")

    pt_path = "graph_scenarios.pt"
    if os.path.exists(pt_path):
        print(f"Found {pt_path}. Loading preprocessed graph data...")
        data = torch.load(pt_path)
        if isinstance(data, list):
            # List of Data objects - use multi-graph training
            print(f"Loaded {len(data)} graph objects (one per scenario)")
            print(f"  Each graph: {data[0].num_nodes} nodes, {data[0].num_features} features")
            
            # Check class distribution
            all_labels = torch.cat([g.y for g in data])
            unique, counts = torch.unique(all_labels, return_counts=True)
            print(f"  Overall class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
            
            print("\nTraining GNN on multi-graph dataset...")
            model, hist_df, best_th = train_gnn_multi_graph(
                data,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                seed=seed,
                use_relu=use_relu,
                batch_size=32
            )
        else:
            # Single Data object - use single-graph training
            print("Loaded a single Data object")
            print("Sanitizing loaded PyG Data...")
            sanitize_pyg_data(data, add_loops_if_empty=True, verbose_prefix="Loaded: ")
            print("Training GNN on single global graph...")
            model, hist_df, train_idx, val_idx, best_th = train_gnn(
                data,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                seed=seed,
                use_relu=use_relu
            )
        
        print("\nEpoch | Train Loss | Val Loss | Val Acc | Val Prec | Val Rec | Val F1 | Val F1_macro")
        for i, row in hist_df.iterrows():
            if int(row["epoch"]) % 10 == 0 or int(row["epoch"]) == epochs:
                print(
                    f"{int(row['epoch']):4d} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | "
                    f"{row['val_acc']:.4f} | {row['val_prec']:.4f} | {row['val_rec']:.4f} | {row['val_f1']:.4f} | {row['val_f1_macro']:.4f}"
                )
        print("\nBest model loaded. You can now evaluate or export.")
        return

    # --- Otherwise, load CSVs and build a global graph for scenario-wise training ---
    bus_path = "bus_scenarios.csv"
    edge_path = "edge_scenarios.csv"
    print(f"Loading data: {bus_path}, {edge_path}")
    bus_df_all, edge_df_all = load_bus_edge_csvs(bus_path, edge_path)
    print(f"Loaded {len(bus_df_all)} bus rows, {len(edge_df_all)} edge rows")

    # --- Build global graph ---
    edge_index_np, Xn, y, scaler, idx_map, scenario_arr, bus_df_full, edge_df_full = make_global_graph(bus_df_all, edge_df_all, mode=mode)
    # Print features used
    if mode == "voltage":
        for feat in ["voltage", "load_MW", "p_inj_mw", "neighbor_count"]:
            if feat not in bus_df_full.columns:
                bus_df_full[feat] = 0.0
        feature_names = ["voltage", "load_MW", "p_inj_mw", "neighbor_count"]
        print("Features used for VOLTAGE classification:", feature_names)
        print("Target: voltage_class")
    elif mode == "thermal":
        feature_names = [col for col in ["x_pu", "length_km", "loading_percent"] if col in edge_df_full.columns]
        print("Features used for THERMAL classification:", feature_names if feature_names else "No features found (using zeros)")
        print("Target: thermal_class")
    # --- Show class balance ---
    cls, cls_counts = np.unique(y, return_counts=True)
    class_counts_str = ", ".join([f"{c}: {int(n)}" for c, n in zip(cls, cls_counts)])
    print(f"Class balance: {class_counts_str}")
    if mode == "voltage":
        print("Voltage class label mapping:")
        print("  0: low (<0.95)")
        print("  1: slightly low [0.95–0.98)")
        print("  2: near nominal [0.98–1.00)")
        print("  3: slightly high [1.00–1.02)")
        print("  4: high (≥1.02)")

    # --- Convert to PyG Data and train ---
    data = to_pyg(edge_index_np, Xn, y)
    sanitize_pyg_data(data, add_loops_if_empty=True, verbose_prefix="Global: ")
    print("Training global GNN with stratified split and focal loss...")
    model, hist_df, train_idx, val_idx, best_th = train_gnn(
        data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        use_relu=use_relu
    )
    print("Epoch | Train Loss | Val Loss | Val Acc | Val Prec | Val Rec | Val F1 | Val F1_macro")
    for i, row in hist_df.iterrows():
        if int(row["epoch"]) % 10 == 0 or int(row["epoch"]) == epochs:
            print(
                f"{int(row['epoch']):4d} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | "
                f"{row['val_acc']:.4f} | {row['val_prec']:.4f} | {row['val_rec']:.4f} | {row['val_f1']:.4f} | {row['val_f1_macro']:.4f}"
            )
    print("Best model loaded. You can now evaluate or export.")

if __name__ == "__main__":
    main()
