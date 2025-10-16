
import os
import io
import time
import random
import numpy as np
import pandas as pd
import streamlit as st
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
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Amir ExirPower Grid GNN (Alarms)", layout="wide")
st.title("⚡ Amir Exir's Power Grid GNN — Node Alarm Classification (GCN + Message Passing)")
st.caption("Nodes = buses | Edges = lines | Features = voltage, load_MW | Target = alarm_flag (0: normal, 1: voltage, 2: thermal, 3: both)")

# -----------------------------
# Sidebar (Settings, instructions, install help)
# -----------------------------
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

# --- Data Import and Preprocessing ---
col1, col2 = st.columns([1, 1])

@st.cache_data(show_spinner=False)
def load_bus_edge_csvs(bus_path="bus_scenarios.csv", edge_path="edge_scenarios.csv"):
    bus_df = pd.read_csv(bus_path)
    edge_df = pd.read_csv(edge_path)
    bus_df = bus_df.dropna().reset_index(drop=True)
    edge_df = edge_df.dropna().reset_index(drop=True)
    return bus_df, edge_df

bus_df_all, edge_df_all = load_bus_edge_csvs()

# --- Build global graph: bus__scenario labels so each scenario is a component ---
def make_global_graph(bus_çdf, edge_df):
    # Each bus gets a unique id: bus__scenario
    bus_df = bus_df.copy()
    edge_df = edge_df.copy()
    assert "bus" in bus_df.columns and "scenario" in bus_df.columns

    # --- Handle alarm_flag column creation if missing ---
    if "alarm_flag" not in bus_df.columns:
        # Use voltage_class and thermal_class to derive alarm_flag if present
        v_alarm = None
        t_alarm_bus = None
        if "voltage_class" in bus_df.columns:
            v_alarm = (bus_df["voltage_class"] != "Normal").astype(int)
        else:
            v_alarm = pd.Series(0, index=bus_df.index)
        # For thermal, propagate line alarms to buses in each scenario
        if "thermal_class" in edge_df.columns and "from_bus" in edge_df.columns and "to_bus" in edge_df.columns and "scenario" in edge_df.columns:
            # Line-level thermal alarm
            t_alarm_line = (edge_df["thermal_class"] != "Normal").astype(int)
            # For each scenario, for each bus, if any incident line is in alarm, mark t_alarm_bus = 1
            t_alarm_bus = pd.Series(0, index=bus_df.index)
            # Build mapping of bus_scenario to index for fast lookup
            bus_df["_bus_scen_key"] = bus_df["bus"].astype(str) + "__" + bus_df["scenario"].astype(str)
            bus_scen_to_idx = {k: i for i, k in enumerate(bus_df["_bus_scen_key"])}
            # For each row in edge_df, if t_alarm_line is 1, set for both from and to bus in that scenario
            for i, row in edge_df.iterrows():
                if t_alarm_line.iloc[i]:
                    scen = row["scenario"]
                    for bus_col in ["from_bus", "to_bus"]:
                        key = str(row[bus_col]) + "__" + str(scen)
                        idx = bus_scen_to_idx.get(key, None)
                        if idx is not None:
                            t_alarm_bus.iloc[idx] = 1
            bus_df = bus_df.drop(columns="_bus_scen_key")
        else:
            t_alarm_bus = pd.Series(0, index=bus_df.index)
        # alarm_flag = v_alarm + 2 * t_alarm_bus
        bus_df["alarm_flag"] = v_alarm + 2 * t_alarm_bus

    bus_df["bus_scen"] = bus_df["bus"].astype(str) + "__" + bus_df["scenario"].astype(str)
    edge_df["from_bus_scen"] = edge_df["from_bus"].astype(str) + "__" + edge_df["scenario"].astype(str)
    edge_df["to_bus_scen"]   = edge_df["to_bus"].astype(str) + "__" + edge_df["scenario"].astype(str)
    # Map unique bus_scen to index
    bus_to_idx = {b: i for i, b in enumerate(bus_df["bus_scen"])}
    src = edge_df["from_bus_scen"].map(bus_to_idx).to_numpy()
    dst = edge_df["to_bus_scen"].map(bus_to_idx).to_numpy()
    edge_index = np.vstack([src, dst])
    # Features and label
    X = bus_df[["voltage", "load_MW"]].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    Xn = scaler.transform(X)
    y = bus_df["alarm_flag"].to_numpy().astype(int)
    scenario_arr = bus_df["scenario"].to_numpy().astype(int)
    return edge_index, Xn, y, scaler, bus_to_idx, scenario_arr, bus_df, edge_df

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

# --- Global graph for training ---
edge_index_np, Xn, y, scaler, bus_to_idx, scenario_arr, bus_df_full, edge_df_full = make_global_graph(train_bus_df, train_edge_df)

# --- Scenario selector and visualization ---
with col1:
    st.subheader("1) Choose Scenario")
    scenario_id_input = st.text_input("Enter Scenario Number", value="0")
    try:
        scenario_id = int(scenario_id_input)
    except ValueError:
        scenario_id = 0
    st.info(
        "Enter a **scenario number** to visualize it. "
        "The app will show the corresponding bus and edge data, "
        "build the topology, and preview the graph for the GNN surrogate."
    )
    # Extract scenario-specific data for visualization
    if "scenario" in bus_df_all.columns:
        bus_df_vis = bus_df_all[bus_df_all["scenario"] == scenario_id].copy()
        edge_df_vis = edge_df_all[edge_df_all["scenario"] == scenario_id].copy()
    else:
        bus_df_vis, edge_df_vis = bus_df_all, edge_df_all
    st.caption(f"Showing scenario {scenario_id} — {len(bus_df_vis)} buses, {len(edge_df_vis)} lines")
    st.dataframe(bus_df_vis, use_container_width=True, height=250, hide_index=True)
    st.dataframe(edge_df_vis, use_container_width=True, height=180, hide_index=True)

with col2:
    st.subheader("2) Graph Preview")
    if bus_df_vis is not None and edge_df_vis is not None and not bus_df_vis.empty and not edge_df_vis.empty:
        # Filter active lines for the chosen scenario
        edges_active = edge_df_vis[edge_df_vis["in_service"] == True] if "in_service" in edge_df_vis.columns else edge_df_vis

        G = nx.Graph()
        G.add_nodes_from(bus_df_vis['bus' if 'bus' in bus_df_vis.columns else 'BUS'])
        G.add_edges_from(list(zip(edges_active['from_bus'], edges_active['to_bus'])))

        pos = nx.kamada_kawai_layout(G)
        fig, ax = plt.subplots(figsize=(24, 16))
        nx.draw(
            G, pos, with_labels=True, node_size=600, ax=ax,
            edge_color='black', width=1.5, node_color='#1f78b4', font_weight='bold'
        )
        ax.set_title(f"Scenario {scenario_id} Topology (in-service lines only)")
        st.pyplot(fig, use_container_width=True)

        # --- Show out-of-service lines ---
        if "in_service" in edge_df_vis.columns:
            out_of_service = edge_df_vis[edge_df_vis["in_service"] == False]
            if not out_of_service.empty:
                st.warning(f"⚠️ {len(out_of_service)} lines are out of service in Scenario {scenario_id}:")
                st.dataframe(
                    out_of_service[["from_bus", "to_bus", "length_km"]] if "length_km" in out_of_service.columns else out_of_service[["from_bus", "to_bus"]],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success(f"✅ All {len(edge_df_vis)} lines are in service in Scenario {scenario_id}.")

# -----------------------------
# Training
# -----------------------------

# -----------------------------
# Training: Scenario-wise GNN surrogate
# -----------------------------
st.subheader("3) Train GNN Surrogate (Scenario-wise)")
if len(missing) > 0 or Data is None or GCNConv is None:
    st.error("PyTorch and/or PyTorch Geometric are not available. See install commands in the sidebar.")
else:
    if train_bus_df is not None and train_edge_df is not None and not train_bus_df.empty and not train_edge_df.empty:
        # Info: show number of scenarios in train/test
        if n_train_scen is not None and n_test_scen is not None:
            st.info(f"Training on {n_train_scen} scenarios, testing on {n_test_scen}")

        # Hyperparameters
        epochs = st.slider("Epochs", 50, 800, 150, step=50)
        lr     = st.select_slider("Learning Rate", options=[1e-3, 3e-3, 1e-2, 3e-2], value=1e-2)
        wd     = st.select_slider("Weight Decay", options=[0.0, 5e-4, 1e-3], value=1e-3)
        seed   = st.number_input("Seed", value=42, step=1)

        # Features for global graph
        data = to_pyg(edge_index_np, Xn, y)
        y_np = data.y.cpu().numpy()
        cls, cls_counts = np.unique(y_np, return_counts=True)
        class_counts_str = ", ".join([f"{c}: {int(n)}" for c, n in zip(cls, cls_counts)])
        st.caption(f"Class balance → {class_counts_str}")

        # Cross-validation controls
        use_cv = st.toggle("Use RepeatedStratifiedKFold CV (avg metrics)", value=True,
                           help="Runs CV, averages metrics, and keeps the best fold's weights for inspection.")
        n_splits  = st.select_slider("CV n_splits",  options=[3, 5, 7, 10], value=5, disabled=not use_cv)
        n_repeats = st.select_slider("CV n_repeats", options=[1, 2, 3, 5],  value=3, disabled=not use_cv)
        linear_gcn = st.toggle(
            "Make GCN linear (no activations)",
            value=False,
            help="Removes ReLUs from the GCN so the mapping is linear in the inputs."
        )

        # --- Baseline: Logistic Regression (no graph) ---
        st.subheader("Baseline: Logistic Regression (no graph)")
        if st.button("⚖️ Train linear baseline", key="lin_base"):
            X_np = Xn.astype(float)
            y_np2 = y.astype(int)
            # Handle missing values safely before training
            if np.isnan(X_np).any():
                st.warning("Detected NaN values in feature matrix — replacing with column means.")
                col_means = np.nanmean(X_np, axis=0)
                inds = np.where(np.isnan(X_np))
                X_np[inds] = np.take(col_means, inds[1])
            bs_splits  = n_splits if use_cv else 5
            bs_repeats = n_repeats if use_cv else 3
            rskf_bs = RepeatedStratifiedKFold(n_splits=bs_splits, n_repeats=bs_repeats, random_state=seed)
            mets = []
            for tr_idx, va_idx in rskf_bs.split(np.zeros_like(y_np2), y_np2):
                Xtr, Xva = X_np[tr_idx], X_np[va_idx]
                ytr, yva = y_np2[tr_idx], y_np2[va_idx]
                clf = LogisticRegression(max_iter=2000, class_weight='balanced')
                clf.fit(Xtr, ytr)
                pr = clf.predict(Xva)
                mets.append([
                    accuracy_score(yva, pr),
                    precision_score(yva, pr, zero_division=0),
                    recall_score(yva,  pr, zero_division=0),
                    f1_score(yva, pr)
                ])
            mets = np.array(mets)
            st.write(
                f"LR CV (mean ± std) — "
                f"Acc: {mets[:,0].mean():.3f}±{mets[:,0].std(ddof=1):.3f}, "
                f"Prec: {mets[:,1].mean():.3f}±{mets[:,1].std(ddof=1):.3f}, "
                f"Rec: {mets[:,2].mean():.3f}±{mets[:,2].std(ddof=1):.3f}, "
                f"F1: {mets[:,3].mean():.3f}±{mets[:,3].std(ddof=1):.3f}"
            )

        # Guardrail: if too few positive samples, cap splits
        if use_cv:
            pos_count = int(cls_counts[cls.tolist().index(1)]) if 1 in cls else 0
            max_splits = max(2, min(n_splits, pos_count))
            if max_splits < n_splits:
                st.warning(
                    f"Reducing CV splits from {n_splits} → {max_splits} because some folds would lack positives. "
                    f"(positives={pos_count})"
                )
                n_splits = max_splits

        # --- Train button ---
        if st.button("🚀 Train Model", type="primary"):
            if use_cv:
                with st.spinner("Training (cross‑validation)..."):
                    model, hist_df, train_idx, val_idx, best_th, cv_summary = train_gnn_cv(
                        data, epochs=epochs, lr=lr, weight_decay=wd, seed=seed,
                        n_splits=n_splits, n_repeats=n_repeats, use_relu=(not linear_gcn)
                    )
                st.success(f"CV done over {cv_summary['n_folds']} folds.")
                st.write(
                    f"**CV (mean ± std)** — "
                    f"Acc: {cv_summary['acc_mean']:.3f}±{cv_summary['acc_std']:.3f}, "
                    f"Prec: {cv_summary['prec_mean']:.3f}±{cv_summary['prec_std']:.3f}, "
                    f"Rec: {cv_summary['rec_mean']:.3f}±{cv_summary['rec_std']:.3f}, "
                    f"F1: {cv_summary['f1_mean']:.3f}±{cv_summary['f1_std']:.3f}"
                )
                st.caption("Precision, recall, and F1 are macro-averaged for multiclass alarms.")
                if cv_summary["n_folds"] == 0 or cv_summary["f1_mean"] < 0.6:
                    st.error(
                        "🚨 The model did not train properly or achieved low performance.\n\n"
                        "Possible reasons:\n"
                        "• Class imbalance (too few alarm samples)\n"
                        "• Too many CV splits for dataset size\n"
                        "• Learning rate or weight decay too high\n"
                        "• GNN not converging\n\n"
                        "Try reducing CV folds, increasing epochs, or adjusting learning rate."
                    )
                    st.stop()
            else:
                with st.spinner("Training..."):
                    model, hist_df, train_idx, val_idx, best_th = train_gnn(
                        data, epochs=epochs, lr=lr, weight_decay=wd, seed=seed, use_relu=(not linear_gcn)
                    )
            if hist_df.empty:
                st.error("Model did not train properly — no valid best state found.")
                st.stop()
            st.success("Training complete. Showing best validation performance observed.")
            st.line_chart(hist_df.set_index("epoch")[["train_loss", "val_loss"]])
            st.line_chart(hist_df.set_index("epoch")[["val_acc", "val_f1", "val_f1_macro"]])

            # ── EVAL ──────────────────────────────────────────────────────────────────────
            # For multiclass, no threshold slider; just use argmax
            st.info("Multiclass alarm: predictions shown for each class (0: normal, 1: voltage, 2: thermal, 3: both).")

            # --- EVALUATE ON TEST SCENARIOS ---
            if test_bus_df is not None and test_edge_df is not None and not test_bus_df.empty:
                # Build test graph (using train scaler)
                def make_test_graph(test_bus_df, test_edge_df, scaler):
                    test_bus_df = test_bus_df.copy()
                    test_edge_df = test_edge_df.copy()
                    test_bus_df["bus_scen"] = test_bus_df["bus"].astype(str) + "__" + test_bus_df["scenario"].astype(str)
                    test_edge_df["from_bus_scen"] = test_edge_df["from_bus"].astype(str) + "__" + test_edge_df["scenario"].astype(str)
                    test_edge_df["to_bus_scen"]   = test_edge_df["to_bus"].astype(str) + "__" + test_edge_df["scenario"].astype(str)
                    bus_to_idx = {b: i for i, b in enumerate(test_bus_df["bus_scen"])}
                    src = test_edge_df["from_bus_scen"].map(bus_to_idx).to_numpy()
                    dst = test_edge_df["to_bus_scen"].map(bus_to_idx).to_numpy()
                    edge_index = np.vstack([src, dst])
                    X_test_raw = test_bus_df[["voltage", "load_MW"]].to_numpy(dtype=float)
                    Xn_test = scaler.transform(X_test_raw)
                    y_test = test_bus_df["alarm_flag"].to_numpy().astype(int)
                    return edge_index, Xn_test, y_test, bus_to_idx
                edge_index_np_test, Xn_test, y_test, _ = make_test_graph(test_bus_df, test_edge_df, scaler)
                data_test = to_pyg(edge_index_np_test, Xn_test, y_test)
                model.eval()
                with torch.no_grad():
                    logits_test = model(data_test.x, data_test.edge_index)
                probs_test = torch.softmax(logits_test, dim=-1).cpu().numpy()
                pred_test = np.argmax(probs_test, axis=-1)
                true_test = data_test.y.cpu().numpy()
                st.subheader("Test Set Evaluation (unseen scenarios)")
                report = classification_report(true_test, pred_test, digits=3, zero_division=0)
                st.markdown(f"```\n{report}\n```")
                # Dynamically get all unique classes in test/true/pred
                all_labels = sorted(set(np.unique(true_test)).union(set(np.unique(pred_test))))
                display_labels = [f"class {i}" for i in all_labels]
                cm = confusion_matrix(true_test, pred_test, labels=all_labels)
                fig, ax = plt.subplots(figsize=(3, 2.2))
                ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(
                    ax=ax, values_format="d", colorbar=False
                )
                ax.set_title("Test Set Confusion Matrix")
                st.pyplot(fig, use_container_width=False)
                test_bus_df_view = test_bus_df.copy()
                # For multiclass, show max prob and predicted class
                test_bus_df_view["pred_alarm_prob"] = probs_test.max(axis=1)
                test_bus_df_view["pred_alarm_label"] = pred_test
                test_bus_df_view["correct"] = (pred_test == true_test).astype(int)
                st.dataframe(
                    test_bus_df_view.sort_values("pred_alarm_prob", ascending=False),
                    use_container_width=True
                )
            else:
                st.info("No test scenarios available for evaluation.")

        # --- SAVE ARTIFACTS ---
        if st.button("💾 Save model + scaler"):
            import pickle
            torch.save(model.state_dict(), "gnn_alarm_model.pt")
            with open("feature_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            st.success("Saved: gnn_alarm_model.pt, feature_scaler.pkl")
            st.download_button("Download model weights", data=open("gnn_alarm_model.pt", "rb").read(), file_name="gnn_alarm_model.pt")
            st.download_button("Download scaler", data=open("feature_scaler.pkl", "rb").read(), file_name="feature_scaler.pkl")
    else:
        st.info("Load/create data first.")