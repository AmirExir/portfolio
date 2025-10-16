
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
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    (train_idx_np, val_idx_np), = sss.split(np.zeros_like(y_np), y_np)
    return train_idx_np, val_idx_np

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
        # --- Handle voltage_class column creation if missing ---
        if "voltage_class" not in bus_df.columns:
            bus_df["voltage_class"] = 0
        # Target: voltage_class (convert to int if not already)
        if "voltage_class" in bus_df.columns:
            y = bus_df["voltage_class"]
            # If not integer, convert (e.g., from string labels)
            if not np.issubdtype(y.dtype, np.integer):
                y = y.astype(str)
                # Map to int: "Normal"->0, others->1,2,...
                classes = sorted(y.unique())
                class_map = {cls: idx for idx, cls in enumerate(classes)}
                y = y.map(class_map)
            y = y.fillna(0).to_numpy().astype(int)
        else:
            y = np.zeros(len(bus_df), dtype=int)
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
        # For GNN, we create a graph where each edge is a node, and connect edges sharing a bus in the same scenario (line-graph)
        # For simplicity, connect edges that share a bus_scen
        edge_nodes = edge_df.reset_index()
        neighbors = []
        for i, row_i in edge_nodes.iterrows():
            for j, row_j in edge_nodes.iterrows():
                if i == j:
                    continue
                # If they share a bus_scen in the same scenario, connect
                if row_i["scenario"] == row_j["scenario"]:
                    buses_i = {row_i["from_bus_scen"], row_i["to_bus_scen"]}
                    buses_j = {row_j["from_bus_scen"], row_j["to_bus_scen"]}
                    if len(buses_i & buses_j) > 0:
                        neighbors.append((i, j))
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
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["voltage", "thermal"], default="voltage",
                        help="Choose whether to train on voltage_class or thermal_class")
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

    # --- Build global graph for training ---
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

    # --- Build PyG data ---
    data = to_pyg(edge_index_np, Xn, y)

    # --- Train GNN ---
    epochs = 150
    lr = 1e-2
    wd = 1e-3
    seed = 42
    use_relu = True
    print(f"Training GNN for {epochs} epochs (lr={lr}, weight_decay={wd}, seed={seed})...")

    model, hist_df, train_idx, val_idx, best_th = train_gnn(
        data, epochs=epochs, lr=lr, weight_decay=wd, seed=seed, use_relu=use_relu
    )
    # Print progress every 10 epochs
    print("Epoch | Train Loss | Val Loss | Val Acc | Val F1")
    for i, row in hist_df.iterrows():
        if int(row["epoch"]) % 10 == 0 or int(row["epoch"]) == epochs:
            print(f"{int(row['epoch']):4d} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | {row['val_acc']:.4f} | {row['val_f1']:.4f}")

    # --- Evaluate on test set ---
    if test_bus_df is not None and test_edge_df is not None and ((mode == "voltage" and not test_bus_df.empty) or (mode == "thermal" and not test_edge_df.empty)):
        def make_test_graph(test_bus_df, test_edge_df, scaler, mode="voltage"):
            test_bus_df = test_bus_df.copy()
            test_edge_df = test_edge_df.copy()
            if mode == "voltage":
                # Target: voltage_class
                if "voltage_class" not in test_bus_df.columns:
                    test_bus_df["voltage_class"] = 0
                y_test = test_bus_df["voltage_class"]
                if not np.issubdtype(y_test.dtype, np.integer):
                    y_test = y_test.astype(str)
                    classes = sorted(y_test.unique())
                    class_map = {cls: idx for idx, cls in enumerate(classes)}
                    y_test = y_test.map(class_map)
                y_test = y_test.fillna(0).to_numpy().astype(int)
                test_bus_df["bus_scen"] = test_bus_df["bus"].astype(str) + "__" + test_bus_df["scenario"].astype(str)
                test_edge_df["from_bus_scen"] = test_edge_df["from_bus"].astype(str) + "__" + test_edge_df["scenario"].astype(str)
                test_edge_df["to_bus_scen"]   = test_edge_df["to_bus"].astype(str) + "__" + test_edge_df["scenario"].astype(str)
                bus_to_idx = {b: i for i, b in enumerate(test_bus_df["bus_scen"])}
                src = test_edge_df["from_bus_scen"].map(bus_to_idx).to_numpy()
                dst = test_edge_df["to_bus_scen"].map(bus_to_idx).to_numpy()
                edge_index = np.vstack([src, dst])
                X_test_raw = test_bus_df[["voltage", "load_MW"]].to_numpy(dtype=float)
                Xn_test = scaler.transform(X_test_raw)
                return edge_index, Xn_test, y_test, bus_to_idx
            elif mode == "thermal":
                # Target: thermal_class
                if "thermal_class" not in test_edge_df.columns:
                    test_edge_df["thermal_class"] = 0
                y_test = test_edge_df["thermal_class"]
                if not np.issubdtype(y_test.dtype, np.integer):
                    y_test = y_test.astype(str)
                    classes = sorted(y_test.unique())
                    class_map = {cls: idx for idx, cls in enumerate(classes)}
                    y_test = y_test.map(class_map)
                y_test = y_test.fillna(0).to_numpy().astype(int)
                test_edge_df["from_bus_scen"] = test_edge_df["from_bus"].astype(str) + "__" + test_edge_df["scenario"].astype(str)
                test_edge_df["to_bus_scen"]   = test_edge_df["to_bus"].astype(str) + "__" + test_edge_df["scenario"].astype(str)
                edge_to_idx = {i: i for i in range(len(test_edge_df))}
                edge_nodes = test_edge_df.reset_index()
                neighbors = []
                for i, row_i in edge_nodes.iterrows():
                    for j, row_j in edge_nodes.iterrows():
                        if i == j:
                            continue
                        if row_i["scenario"] == row_j["scenario"]:
                            buses_i = {row_i["from_bus_scen"], row_i["to_bus_scen"]}
                            buses_j = {row_j["from_bus_scen"], row_j["to_bus_scen"]}
                            if len(buses_i & buses_j) > 0:
                                neighbors.append((i, j))
                if len(neighbors) == 0:
                    edge_index = np.zeros((2, 0), dtype=int)
                else:
                    edge_index = np.array(neighbors).T
                features = [col for col in ["x_pu", "length_km", "loading_percent"] if col in test_edge_df.columns]
                if len(features) == 0:
                    X_test = np.zeros((len(test_edge_df), 1))
                else:
                    X_test = test_edge_df[features].to_numpy(dtype=float)
                Xn_test = scaler.transform(X_test)
                return edge_index, Xn_test, y_test, edge_to_idx
            else:
                raise ValueError("Unknown mode")
        edge_index_np_test, Xn_test, y_test, _ = make_test_graph(test_bus_df, test_edge_df, scaler, mode=mode)
        data_test = to_pyg(edge_index_np_test, Xn_test, y_test)
        model.eval()
        with torch.no_grad():
            logits_test = model(data_test.x, data_test.edge_index)
        probs_test = torch.softmax(logits_test, dim=-1).cpu().numpy()
        pred_test = np.argmax(probs_test, axis=-1)
        true_test = data_test.y.cpu().numpy()
        print("\nTest Set Evaluation (unseen scenarios):")
        report = classification_report(true_test, pred_test, digits=3, zero_division=0)
        print(report)
        all_labels = sorted(set(np.unique(true_test)).union(set(np.unique(pred_test))))
        cm = confusion_matrix(true_test, pred_test, labels=all_labels)
        print("Confusion Matrix:")
        print(cm)
        acc = accuracy_score(true_test, pred_test)
        f1 = f1_score(true_test, pred_test, average='macro')
        print(f"Test Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    else:
        print("No test scenarios available for evaluation.")

if __name__ == "__main__":
    main()