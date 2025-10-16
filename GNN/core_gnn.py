
import os
import io
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
# -----------------------
# Graph building function
# -----------------------

def focal_loss(logits, targets, gamma=2.0, alpha=None):
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()

# -----------------------------

# Try to import torch + PyG and fail gracefully with instructions
missing = []
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    missing.append("torch / torch.nn")
    torch = None
    nn = None
    F = None

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
except Exception as e:
    Data = None
    GCNConv = None

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

# -----------------------------
# Helpers
# -----------------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    if torch is not None and hasattr(torch, "manual_seed"):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)


def synthetic_14_bus():
    buses = [f'Bus{i}' for i in range(1, 15)]
    branches = [
        ('Bus1','Bus2'),('Bus1','Bus5'),('Bus2','Bus3'),('Bus2','Bus4'),
        ('Bus3','Bus4'),('Bus4','Bus5'),('Bus5','Bus6'),('Bus6','Bus11'),
        ('Bus6','Bus12'),('Bus6','Bus13'),('Bus7','Bus8'),('Bus7','Bus9'),
        ('Bus9','Bus10'),('Bus9','Bus14'),('Bus10','Bus11'),('Bus12','Bus13'),
        ('Bus13','Bus14'),('Bus4','Bus7'),('Bus8','Bus14'),('Bus3','Bus9'),
    ]
    # simple synthetic bus features + injections
    rows = []
    for i, bus in enumerate(buses):
        voltage = round(1.0 + 0.05 * (i % 3), 3)
        load_MW = 50 + 10 * (i % 5)
        alarm_flag = 1 if i % 6 == 0 else 0
        # tiny net injections alternating +/-
        p_inj_mw = (5 if i % 2 == 0 else -5)
        rows.append([bus, voltage, load_MW, alarm_flag, p_inj_mw])
    bus_df = pd.DataFrame(rows, columns=['bus','voltage','load_MW','alarm_flag','p_inj_mw'])

    # simple constant reactance per line
    edge_df = pd.DataFrame(branches, columns=['from_bus','to_bus'])
    

    return bus_df, edge_df

def build_graph(bus_df, edge_df):
    # Ensure bus and edge IDs are string type for mapping
    bus_df = bus_df.copy()
    edge_df = edge_df.copy()
    bus_df['bus'] = bus_df['bus'].astype(str)
    edge_df['from_bus'] = edge_df['from_bus'].astype(str)
    edge_df['to_bus'] = edge_df['to_bus'].astype(str)

    # map bus to index
    bus_to_idx = {b:i for i,b in enumerate(bus_df['bus'])}

    # edge index (directed, as per PyG)
    src = edge_df['from_bus'].map(bus_to_idx).to_numpy()
    dst = edge_df['to_bus'  ].map(bus_to_idx).to_numpy()
    edge_index = np.vstack([src, dst])

    # features (linearized input)
    X = bus_df[['voltage','load_MW']].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    Xn = scaler.transform(X)
    y = bus_df['alarm_flag'].to_numpy().astype(int)

    return edge_index, Xn, y, scaler, bus_to_idx

class GCN(nn.Module):
    def __init__(self, in_dim, h_dim=64, num_classes=2, dropout=0.4, use_relu=True):
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

def _class_weights(y_np, n_classes):
    # inverse-frequency weights for multiclass
    counts = np.bincount(y_np, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    w = inv / inv.sum() * n_classes
    return torch.tensor(w, dtype=torch.float)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_gnn(data, epochs=300, lr=1e-2, weight_decay=5e-4, seed=42, use_relu=True):
    set_seed(seed)
    n_classes = int(data.y.max().item()) + 1
    model = GCN(in_dim=data.x.size(1), num_classes=n_classes, use_relu=use_relu).to(data.x.device)
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

    # ---- Choose a validation threshold that maximizes F1 on the PR curve ----
    # For multiclass, thresholding is not used; just return None for best_th
    best_th = None

    # Convert history to DataFrame for plotting
    hist_df = pd.DataFrame(
        history,
        columns=["epoch", "train_loss", "val_loss", "val_acc", "val_prec", "val_rec", "val_f1", "val_f1_macro"]
    )

    return model, hist_df, train_idx, val_idx, best_th




def train_gnn_cv(
    data,
    epochs=200,
    lr=1e-2,
    weight_decay=5e-4,
    seed=42,
    n_splits=5,
    n_repeats=3,
    use_relu=True
):
    set_seed(seed)
    n_classes = int(data.y.max().item()) + 1
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    y_np = data.y.cpu().numpy()
    fold_stats = []
    best_state = None
    best_val_loss = float('inf')
    best_train_idx = None
    best_val_idx   = None
    best_hist      = None
    best_th        = None

    for fold_id, (tr_np, va_np) in enumerate(rskf.split(np.zeros_like(y_np), y_np), start=1):
        model = GCN(in_dim=data.x.size(1), h_dim=64, num_classes=n_classes, use_relu=use_relu).to(data.x.device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        tr = torch.tensor(tr_np, dtype=torch.long, device=data.x.device)
        va = torch.tensor(va_np, dtype=torch.long, device=data.x.device)

        # ---- Fold-level guard: skip folds with a single-class validation set ----
        yv_fold = data.y[va].detach().cpu().numpy().astype(int)
        if len(np.unique(yv_fold)) < 2:
            continue

        counts_t = torch.bincount(data.y[tr], minlength=n_classes).float()
        alpha = (1.0 / (counts_t + 1e-6))
        alpha = (alpha / alpha.sum()).to(data.x.device)

        hist = []
        for epoch in range(1, epochs+1):
            model.train()
            logits = model(data.x, data.edge_index)
            loss = focal_loss(logits[tr], data.y[tr], gamma=2.0, alpha=alpha)
            opt.zero_grad(); loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                logits_val = model(data.x, data.edge_index)[va]
            val_loss = focal_loss(logits_val, data.y[va], gamma=2.0, alpha=alpha).item()

            preds = logits_val.argmax(dim=-1).detach().cpu().numpy().astype(int)
            yv    = data.y[va].detach().cpu().numpy().astype(int)

            acc  = accuracy_score(yv, preds)
            prec = precision_score(yv, preds, average='macro', zero_division=0)
            rec  = recall_score(yv, preds, average='macro', zero_division=0)
            f1   = f1_score(yv, preds, average='macro')
            f1m  = f1
            hist.append((epoch, float(loss.item()), val_loss, acc, prec, rec, f1, f1m))

        # For multiclass, thresholding is not used; just use argmax
        # Instead, evaluate metrics on argmax
        preds_fold = logits_val.argmax(dim=-1).detach().cpu().numpy().astype(int)
        yv_fold = data.y[va].detach().cpu().numpy().astype(int)
        acc_th  = accuracy_score(yv_fold, preds_fold)
        prec_th = precision_score(yv_fold, preds_fold, average='macro', zero_division=0)
        rec_th  = recall_score(yv_fold, preds_fold, average='macro', zero_division=0)
        f1_th   = f1_score(yv_fold, preds_fold, average='macro')
        fold_stats.append((acc_th, prec_th, rec_th, f1_th))

        if hist[-1][2] < best_val_loss:
            best_val_loss = hist[-1][2]
            best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_train_idx, best_val_idx = tr, va
            best_hist = hist
            best_th   = None

    fold_stats = np.array(fold_stats)
    cv_summary = {
        "n_folds": int(len(fold_stats)),
        "acc_mean": float(fold_stats[:,0].mean()) if len(fold_stats) else 0.0, "acc_std": float(fold_stats[:,0].std(ddof=1)) if len(fold_stats) > 1 else 0.0,
        "prec_mean": float(fold_stats[:,1].mean()) if len(fold_stats) else 0.0, "prec_std": float(fold_stats[:,1].std(ddof=1)) if len(fold_stats) > 1 else 0.0,
        "rec_mean": float(fold_stats[:,2].mean()) if len(fold_stats) else 0.0, "rec_std": float(fold_stats[:,2].std(ddof=1)) if len(fold_stats) > 1 else 0.0,
        "f1_mean": float(fold_stats[:,3].mean()) if len(fold_stats) else 0.0,  "f1_std": float(fold_stats[:,3].std(ddof=1)) if len(fold_stats) > 1 else 0.0,
    }

    if best_state is None:
        st.warning("⚠️ No valid folds produced a trained model — likely due to class imbalance or insufficient samples.")
        model = GCN(in_dim=data.x.size(1), h_dim=64, num_classes=n_classes, use_relu=use_relu).to(data.x.device)
        cv_summary = {
            "n_folds": 0,
            "acc_mean": 0.0, "acc_std": 0.0,
            "prec_mean": 0.0, "prec_std": 0.0,
            "rec_mean": 0.0, "rec_std": 0.0,
            "f1_mean": 0.0, "f1_std": 0.0,
        }
        return model, pd.DataFrame(), None, None, None, cv_summary
    model = GCN(in_dim=data.x.size(1), h_dim=64, num_classes=n_classes, use_relu=use_relu).to(data.x.device)
    model.load_state_dict(best_state)
    hist_df = pd.DataFrame(best_hist, columns=["epoch","train_loss","val_loss","val_acc","val_prec","val_rec","val_f1","val_f1_macro"])
    return model, hist_df, best_train_idx, best_val_idx, best_th, cv_summary

def to_pyg(edge_index_np, Xn, y):
    device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    x = torch.tensor(Xn, dtype=torch.float, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    data = Data(x=x, edge_index=edge_index, y=y)
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

    if mode == "voltage":
        # --- Define voltage_class for 5 classes based on voltage ranges ---
        def voltage_to_class(v):
            if v < 0.95:
                return 0  # low
            elif 0.95 <= v < 0.98:
                return 1  # slightly low
            elif 0.98 <= v < 1.00:
                return 2  # near nominal
            elif 1.00 <= v < 1.02:
                return 3  # slightly high
            else:  # v >= 1.02
                return 4  # high
        # If voltage_class not already defined, create it using voltage
        bus_df["voltage_class"] = bus_df["voltage"].apply(voltage_to_class)
        y = bus_df["voltage_class"].fillna(2).to_numpy().astype(int)
        # Features: voltage, load_MW
        features = ["voltage", "load_MW"]
        X = bus_df[features].to_numpy(dtype=float)
        scaler = StandardScaler().fit(X)
        Xn = scaler.transform(X)
        # Edge index: bus-based
        bus_df["bus_scen"] = bus_df["bus"].astype(str) + "__" + bus_df["scenario"].astype(str)
        edge_df["from_bus_scen"] = edge_df["from_bus"].astype(str) + "__" + edge_df["scenario"].astype(str)
        edge_df["to_bus_scen"]   = edge_df["to_bus"].astype(str) + "__" + edge_df["scenario"].astype(str)
        bus_to_idx = {b: i for i, b in enumerate(bus_df["bus_scen"])}
        src = edge_df["from_bus_scen"].map(bus_to_idx).to_numpy()
        dst = edge_df["to_bus_scen"].map(bus_to_idx).to_numpy()
        edge_index = np.vstack([src, dst])
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
        scenario_arr = edge_df["scenario"].to_numpy().astype(int)
        return edge_index, Xn, y, scaler, edge_to_idx, scenario_arr, bus_df, edge_df
    else:
        raise ValueError(f"Unknown mode {mode}. Choose 'voltage' or 'thermal'.")


# -----------------------------
# Main function for CLI/VS Code execution
# -----------------------------

# -----------------------------
# Per-scenario batching helpers and new training logic
# -----------------------------
def build_data_list(bus_df, edge_df, scenario_ids, mode="voltage", scaler=None):
    """
    For each scenario in scenario_ids, build a PyG Data object.
    Returns a list of Data objects.
    If scaler is not None, use it to transform features (for test set).
    """
    data_list = []
    for scen in scenario_ids:
        bus_sub = bus_df[bus_df["scenario"] == scen].copy()
        edge_sub = edge_df[edge_df["scenario"] == scen].copy()
        if bus_sub.empty or edge_sub.empty:
            continue
        # Use make_global_graph for this scenario only
        edge_index_np, Xn, y, scaler_this, idx_map, scenario_arr, bdf, edf = make_global_graph(bus_sub, edge_sub, mode=mode)
        # Use provided scaler for test set, else fit on the scenario
        if scaler is not None:
            if mode == "voltage":
                X_raw = bus_sub[["voltage", "load_MW"]].to_numpy(dtype=float)
                Xn = scaler.transform(X_raw)
            elif mode == "thermal":
                features = [col for col in ["x_pu", "length_km", "loading_percent"] if col in edge_sub.columns]
                if len(features) == 0:
                    X_raw = np.zeros((len(edge_sub), 1))
                else:
                    X_raw = edge_sub[features].to_numpy(dtype=float)
                Xn = scaler.transform(X_raw)
        data = to_pyg(edge_index_np, Xn, y)
        data_list.append(data)
    return data_list

def train_gnn_batches(data_list, epochs=150, lr=1e-2, weight_decay=1e-3, seed=42, use_relu=True, batch_size=1):
    """
    Train a GCN model on batches of per-scenario graphs.
    data_list: list of torch_geometric.data.Data objects (each is a scenario).
    Returns model, history DataFrame.
    """
    import torch
    from torch_geometric.loader import DataLoader
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Assume all data in data_list have same feature dim and class count
    in_dim = data_list[0].x.size(1)
    # Get number of classes from y
    all_y = torch.cat([d.y for d in data_list])
    n_classes = int(all_y.max().item()) + 1
    model = GCN(in_dim=in_dim, num_classes=n_classes, use_relu=use_relu).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Stratified split for each scenario: for each Data, split its y into train/val
    splits = []
    for data in data_list:
        y_np = data.y.cpu().numpy()
        train_idx_np, val_idx_np = _stratified_indices(y_np, train_frac=0.7, seed=seed)
        splits.append((torch.tensor(train_idx_np, dtype=torch.long), torch.tensor(val_idx_np, dtype=torch.long)))
    # Focal loss alpha: compute from all train indices across all scenarios
    train_y = torch.cat([d.y[s[0]] for d, s in zip(data_list, splits)])
    counts_t = torch.bincount(train_y, minlength=n_classes).float()
    alpha = 1.0 / (counts_t + 1e-6)
    alpha = (alpha / alpha.sum()).to(device)
    # Dataloader for batching scenarios
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    history = []
    best = (1e9, None)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        total_preds = []
        total_true = []
        # For macro F1/acc, accumulate over all val nodes in all scenarios
        # Train on all scenario batches
        for i, batch in enumerate(loader):
            # Each batch is a batch of graphs (scenarios)
            batch = batch.to(device)
            # Find train_idx in this batch: concatenate for all graphs in batch
            batch_train_idx = []
            offset = 0
            for j in range(batch.num_graphs):
                idx = splits[batch.ptr[j].item()][0] if hasattr(batch, "ptr") else splits[j][0]
                batch_train_idx.append(idx + offset)
                offset += batch.batch.eq(j).sum().item()
            train_idx = torch.cat(batch_train_idx).to(device)
            logits = model(batch.x, batch.edge_index)
            loss = focal_loss(logits[train_idx], batch.y[train_idx], gamma=2.0, alpha=alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * train_idx.numel()
            total_items += train_idx.numel()
        # Validation: evaluate on all graphs' val_idx
        model.eval()
        val_loss_sum = 0.0
        val_items = 0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for data, (tr_idx, va_idx) in zip(data_list, splits):
                data = data.to(device)
                logits = model(data.x, data.edge_index)
                val_loss = focal_loss(logits[va_idx], data.y[va_idx], gamma=2.0, alpha=alpha)
                val_loss_sum += val_loss.item() * va_idx.numel()
                val_items += va_idx.numel()
                preds = logits[va_idx].argmax(dim=-1).cpu().numpy()
                true = data.y[va_idx].cpu().numpy()
                val_preds.append(preds)
                val_true.append(true)
        # Aggregate metrics
        train_loss_avg = total_loss / total_items if total_items else 0.0
        val_loss_avg = val_loss_sum / val_items if val_items else 0.0
        val_preds_all = np.concatenate(val_preds)
        val_true_all = np.concatenate(val_true)
        acc = accuracy_score(val_true_all, val_preds_all)
        f1 = f1_score(val_true_all, val_preds_all, average='macro')
        history.append((epoch, train_loss_avg, val_loss_avg, acc, f1))
        if val_loss_avg < best[0]:
            best = (val_loss_avg, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    # Restore best model
    if best[1] is not None:
        model.load_state_dict(best[1])
    import pandas as pd
    hist_df = pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss", "val_acc", "val_f1"])
    return model, hist_df, splits

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["voltage", "thermal"], default="voltage",
                        help="Choose whether to train on voltage_class or thermal_class")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    args = parser.parse_args()
    mode = args.mode
    print(f"Running in {mode.upper()} classification mode")

    # --- Load CSVs ---
    bus_path = "bus_scenarios.csv"
    edge_path = "edge_scenarios.csv"
    print(f"Loading data: {bus_path}, {edge_path}")
    bus_df_all, edge_df_all = load_bus_edge_csvs(bus_path, edge_path)
    print(f"Loaded {len(bus_df_all)} bus rows, {len(edge_df_all)} edge rows")

    # --- Scenario train/test split ---
    if "scenario" in bus_df_all.columns and "scenario" in edge_df_all.columns:
        scenario_ids = sorted(bus_df_all["scenario"].unique())
        from sklearn.model_selection import train_test_split
        train_scenarios, test_scenarios = train_test_split(
            scenario_ids, test_size=0.2, random_state=42
        )
        train_scenarios = set(train_scenarios)
        test_scenarios = set(test_scenarios)
        train_bus_df = bus_df_all[bus_df_all["scenario"].isin(train_scenarios)].copy()
        train_edge_df = edge_df_all[edge_df_all["scenario"].isin(train_scenarios)].copy()
        test_bus_df = bus_df_all[bus_df_all["scenario"].isin(test_scenarios)].copy()
        test_edge_df = edge_df_all[edge_df_all["scenario"].isin(test_scenarios)].copy()
        n_train_scen = len(train_scenarios)
        n_test_scen = len(test_scenarios)
    else:
        train_bus_df = bus_df_all
        train_edge_df = edge_df_all
        test_bus_df = None
        test_edge_df = None
        n_train_scen = n_test_scen = None

    print(f"Training on {n_train_scen} scenarios, testing on {n_test_scen}")

    # Dataset balancing removed: train on original data directly

    # --- Build global graph for training (for feature/target info and scaler) ---
    edge_index_np, Xn, y, scaler, idx_map, scenario_arr, bus_df_full, edge_df_full = make_global_graph(train_bus_df, train_edge_df, mode=mode)
    # Print features used
    if mode == "voltage":
        feature_names = ["voltage", "load_MW"]
        print("Features used for VOLTAGE classification:", feature_names)
        print("Target: voltage_class")
    elif mode == "thermal":
        feature_names = [col for col in ["x_pu", "length_km", "loading_percent"] if col in train_edge_df.columns]
        print("Features used for THERMAL classification:", feature_names if feature_names else "No features found (using zeros)")
        print("Target: thermal_class")

    # --- Show class balance ---
    cls, cls_counts = np.unique(y, return_counts=True)
    class_counts_str = ", ".join([f"{c}: {int(n)}" for c, n in zip(cls, cls_counts)])
    print(f"Class balance: {class_counts_str}")
    # Print voltage class mapping if in voltage mode
    if mode == "voltage":
        print("Voltage class label mapping:")
        print("  0: low (<0.95)")
        print("  1: slightly low [0.95–0.98)")
        print("  2: near nominal [0.98–1.00)")
        print("  3: slightly high [1.00–1.02)")
        print("  4: high (≥1.02)")

    # --- Build per-scenario PyG data list ---
    train_data_list = build_data_list(train_bus_df, train_edge_df, train_scenarios, mode=mode, scaler=scaler)
    # --- Train GNN using per-scenario batching ---
    epochs = args.epochs
    lr = 1e-2
    wd = 1e-3
    seed = 42
    use_relu = True
    print(f"Training GNN (per-scenario batching) for {epochs} epochs (lr={lr}, weight_decay={wd}, seed={seed})...")
    model, hist_df, splits = train_gnn_batches(
        train_data_list, epochs=epochs, lr=lr, weight_decay=wd, seed=seed, use_relu=use_relu, batch_size=1
    )
    # Print progress every 10 epochs
    print("Epoch | Train Loss | Val Loss | Val Acc | Val F1")
    for i, row in hist_df.iterrows():
        if int(row["epoch"]) % 10 == 0 or int(row["epoch"]) == epochs:
            print(f"{int(row['epoch']):4d} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | {row['val_acc']:.4f} | {row['val_f1']:.4f}")

    # --- Evaluate on test set, per scenario ---
    if test_bus_df is not None and test_edge_df is not None and ((mode == "voltage" and not test_bus_df.empty) or (mode == "thermal" and not test_edge_df.empty)):
        test_data_list = build_data_list(test_bus_df, test_edge_df, test_scenarios, mode=mode, scaler=scaler)
        model.eval()
        all_preds = []
        all_true = []
        for data in test_data_list:
            data = data.to(next(model.parameters()).device)
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
            preds = logits.argmax(dim=-1).cpu().numpy()
            true = data.y.cpu().numpy()
            all_preds.append(preds)
            all_true.append(true)
        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)
        print("\nTest Set Evaluation (unseen scenarios, per-scenario evaluation):")
        report = classification_report(all_true, all_preds, digits=3, zero_division=0)
        print(report)
        all_labels = sorted(set(np.unique(all_true)).union(set(np.unique(all_preds))))
        cm = confusion_matrix(all_true, all_preds, labels=all_labels)
        print("Confusion Matrix:")
        print(cm)
        acc = accuracy_score(all_true, all_preds)
        f1 = f1_score(all_true, all_preds, average='macro')
        print(f"Test Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    else:
        print("No test scenarios available for evaluation.")

if __name__ == "__main__":
    main()