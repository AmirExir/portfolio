
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
st.caption("Nodes = buses | Edges = lines | Features = voltage, load, breaker_status | Target = alarm_flag")

with st.sidebar:
    st.header("Setup")
    st.markdown("""
**Install requirements (terminal):**
```bash
pip install streamlit torch torchvision torchaudio scikit-learn networkx matplotlib pandas
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
```
*(Use the CPU wheel above; switch to CUDA wheel if you have GPU.)*
    """)

    st.divider()
    st.subheader("Data Options")
    use_upload = st.toggle("Upload CSVs (otherwise auto-generate synthetic 14-bus)", value=False)
    st.write("If uploading, provide: **bus_features.csv**, **branch_connections.csv**")
    st.subheader("Features")
    st.info("Using linearized node features (array): voltage, load_MW, breaker_status.")
# -----------------------------
# Helpers
# -----------------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    if torch is not None and hasattr(torch, "manual_seed"):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

def replicate_graph_with_noise(
    bus_df, edge_df, copies=10,
    sigma_v=0.01,        # voltage noise (p.u.)
    sigma_load=3.0,      # MW noise
    sigma_pinj=1.0,      # MW noise
    flip_breaker_p=0.02, # small prob to flip 0/1
    flip_alarm_p=0.02,   # small prob to flip 0/1
    seed=42
):
    """
    Create N disjoint copies of the same topology with small feature noise.
    Node names made unique by appending '__k' to each bus in copy k.
    """
    rng = np.random.default_rng(seed)

    def getcol(df, name):
        # case-insensitive lookup
        m = {c.lower(): c for c in df.columns}
        return m.get(name, name)

    bus_out  = []
    edge_out = []

    for k in range(copies):
        tag = f"__{k}"

        df = bus_df.copy()
        bcol = getcol(df, 'bus')
        df[bcol] = df[bcol].astype(str) + tag

        vcol = getcol(df, 'voltage')
        if vcol in df:
            df[vcol] = df[vcol].astype(float) + rng.normal(0, sigma_v, len(df))

        lcol = getcol(df, 'load_mw')
        if lcol in df:
            df[lcol] = df[lcol].astype(float) + rng.normal(0, sigma_load, len(df))
        lcol2 = getcol(df, 'load_MW')  # support original casing
        if lcol2 in df:
            df[lcol2] = df[lcol2].astype(float) + rng.normal(0, sigma_load, len(df))

        pcol = getcol(df, 'p_inj_mw')
        if pcol in df:
            df[pcol] = df[pcol].astype(float) + rng.normal(0, sigma_pinj, len(df))

        brk = getcol(df, 'breaker_status')
        if brk in df:
            flip = rng.random(len(df)) < flip_breaker_p
            df.loc[flip, brk] = 1 - df.loc[flip, brk].astype(int)

        af = getcol(df, 'alarm_flag')
        if af in df:
            flip = rng.random(len(df)) < flip_alarm_p
            df.loc[flip, af] = 1 - df.loc[flip, af].astype(int)

        bus_out.append(df)

        e = edge_df.copy()
        e['from_bus'] = e['from_bus'].astype(str) + tag
        e['to_bus']   = e['to_bus'].astype(str) + tag
        edge_out.append(e)

    return pd.concat(bus_out, ignore_index=True), pd.concat(edge_out, ignore_index=True)

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
        breaker_status = 1 if i % 4 != 0 else 0
        alarm_flag = 1 if i % 6 == 0 else 0
        # tiny net injections alternating +/-
        p_inj_mw = (5 if i % 2 == 0 else -5)
        rows.append([bus, voltage, load_MW, breaker_status, alarm_flag, p_inj_mw])
    bus_df = pd.DataFrame(rows, columns=['bus','voltage','load_MW','breaker_status','alarm_flag','p_inj_mw'])

    # simple constant reactance per line
    edge_df = pd.DataFrame(branches, columns=['from_bus','to_bus'])
    edge_df['x_pu'] = 0.2  # any positive value works for demo

    return bus_df, edge_df

def build_graph(bus_df, edge_df):
    # map bus to index
    bus_to_idx = {b:i for i,b in enumerate(bus_df['bus'])}

    # edge index (undirected)
    src = edge_df['from_bus'].map(bus_to_idx).to_numpy()
    dst = edge_df['to_bus'  ].map(bus_to_idx).to_numpy()
    edge_index = np.vstack([np.r_[src, dst], np.r_[dst, src]])

    # features (linearized input)
    X = bus_df[['voltage','load_MW','breaker_status']].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    Xn = scaler.transform(X)
    y = bus_df['alarm_flag'].to_numpy().astype(int)

    return edge_index, Xn, y, scaler, bus_to_idx

class GCN(nn.Module):
    def __init__(self, in_dim, h_dim=32, num_classes=2, dropout=0.2):
        super().__init__()
        self.g1 = GCNConv(in_dim, h_dim)
        self.g2 = GCNConv(h_dim, h_dim)
        self.do = nn.Dropout(dropout)
        self.head = nn.Linear(h_dim, num_classes)  # linear classifier head

    def forward(self, x, edge_index):
        x = self.g1(x, edge_index)
        x = torch.relu(x)
        x = self.do(x)
        x = self.g2(x, edge_index)
        x = torch.relu(x)
        x = self.do(x)
        logits = self.head(x)
        return logits


def _stratified_indices(y_np, train_frac=0.7, seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    (train_idx_np, val_idx_np), = sss.split(np.zeros_like(y_np), y_np)
    return train_idx_np, val_idx_np

def _class_weights(y_np):
    # inverse-frequency weights for binary {0,1}
    counts = np.bincount(y_np, minlength=2).astype(float)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    w = inv / inv.sum() * 2.0
    return torch.tensor(w, dtype=torch.float)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_gnn(data, epochs=300, lr=1e-2, weight_decay=5e-4, seed=42):
    set_seed(seed)
    model = GCN(in_dim=data.x.size(1)).to(data.x.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Stratified split so minority class appears in both sets
    y_np = data.y.cpu().numpy()
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)
    (train_idx_np, val_idx_np), = sss.split(np.zeros_like(y_np), y_np)
    train_idx = torch.tensor(train_idx_np, dtype=torch.long, device=data.x.device)
    val_idx   = torch.tensor(val_idx_np,   dtype=torch.long, device=data.x.device)

    # FOCAL LOSS alpha from TRAIN ONLY
    counts_t = torch.bincount(data.y[train_idx], minlength=2).float()
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
        prec = precision_score(yv, preds, average='binary', zero_division=0)
        rec  = recall_score(yv, preds, average='binary', zero_division=0)
        f1   = f1_score(yv, preds, average='binary')
        f1m  = f1_score(yv, preds, average='macro')

        history.append((epoch, float(loss.item()), val_loss, acc, prec, rec, f1, f1m))

        if val_loss < best[0]:
            best = (val_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    # Restore best model
    if best[1] is not None:
        model.load_state_dict(best[1])

    # ---- Choose a validation threshold that maximizes F1 on the PR curve ----
    with torch.no_grad():
        logits_full_val = model(data.x, data.edge_index)[val_idx]
        probs_val_for_th = torch.softmax(logits_full_val, dim=-1)[:, 1].cpu().numpy()
        true_val_for_th  = data.y[val_idx].cpu().numpy()

    precisions, recalls, thresholds = precision_recall_curve(true_val_for_th, probs_val_for_th)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    if thresholds.size > 0:
        best_idx = int(np.nanargmax(f1s[:-1]))  # thresholds aligns with all but last point
        best_th = float(thresholds[best_idx])
    else:
        best_th = 0.5  # fallback when PR cannot be computed

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
    n_repeats=3
):
    set_seed(seed)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    y_np = data.y.cpu().numpy()
    fold_stats = []
    best_state = None
    best_val_loss = float('inf')
    best_train_idx = None
    best_val_idx   = None
    best_hist      = None
    best_th        = 0.5

    for fold_id, (tr_np, va_np) in enumerate(rskf.split(np.zeros_like(y_np), y_np), start=1):
        model = GCN(in_dim=data.x.size(1)).to(data.x.device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        tr = torch.tensor(tr_np, dtype=torch.long, device=data.x.device)
        va = torch.tensor(va_np, dtype=torch.long, device=data.x.device)

        # ---- Fold-level guard: skip folds with a single-class validation set ----
        yv_fold = data.y[va].detach().cpu().numpy().astype(int)
        if len(np.unique(yv_fold)) < 2:
            # This fold can't produce meaningful PR/F1/PR-curve; skip it.
            continue

        counts_t = torch.bincount(data.y[tr], minlength=2).float()
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
            prec = precision_score(yv, preds, zero_division=0)
            rec  = recall_score(yv, preds, zero_division=0)
            f1   = f1_score(yv, preds)
            f1m  = f1_score(yv, preds, average='macro')
            hist.append((epoch, float(loss.item()), val_loss, acc, prec, rec, f1, f1m))

        # pick an F1-maximizing threshold for this fold
        with torch.no_grad():
            logits_val_full = model(data.x, data.edge_index)[va]
            probs_val = torch.softmax(logits_val_full, dim=-1)[:,1].cpu().numpy()
            true_val  = data.y[va].cpu().numpy().astype(int)
        # Defensive check: skip if validation is single-class (should have been caught earlier)
        if len(np.unique(true_val)) < 2:
            continue
        P, R, T = precision_recall_curve(true_val, probs_val)
        F1s = 2*(P*R)/(P+R+1e-12)
        th  = float(T[np.nanargmax(F1s[:-1])]) if T.size>0 else 0.5

        preds_th = (probs_val >= th).astype(int)
        acc_th  = accuracy_score(true_val, preds_th)
        prec_th = precision_score(true_val, preds_th, zero_division=0)
        rec_th  = recall_score(true_val, preds_th, zero_division=0)
        f1_th   = f1_score(true_val, preds_th)
        fold_stats.append((acc_th, prec_th, rec_th, f1_th))

        if hist[-1][2] < best_val_loss:
            best_val_loss = hist[-1][2]
            best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_train_idx, best_val_idx = tr, va
            best_hist = hist
            best_th   = th

    fold_stats = np.array(fold_stats)
    cv_summary = {
        "n_folds": int(len(fold_stats)),
        "acc_mean": float(fold_stats[:,0].mean()), "acc_std": float(fold_stats[:,0].std(ddof=1)),
        "prec_mean": float(fold_stats[:,1].mean()), "prec_std": float(fold_stats[:,1].std(ddof=1)),
        "rec_mean": float(fold_stats[:,2].mean()), "rec_std": float(fold_stats[:,2].std(ddof=1)),
        "f1_mean": float(fold_stats[:,3].mean()),  "f1_std": float(fold_stats[:,3].std(ddof=1)),
    }

    model = GCN(in_dim=data.x.size(1)).to(data.x.device)
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
# Data Input UI
# -----------------------------
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("1) Load or Create Data")
    if use_upload:
        bus_file = st.file_uploader("Upload bus_features.csv", type=["csv"], key="bus_csv")
        edge_file = st.file_uploader("Upload branch_connections.csv", type=["csv"], key="edge_csv")
        if bus_file and edge_file:
            bus_df = pd.read_csv(bus_file)
            edge_df = pd.read_csv(edge_file)
            st.success("✅ CSVs uploaded.")
        else:
            st.info("Upload both CSVs or uncheck 'Upload CSVs' to use synthetic data.")
            bus_df, edge_df = None, None
    else:
        bus_df, edge_df = synthetic_14_bus()
        st.success("✅ Using synthetic 14-bus dataset.")

    if bus_df is not None:
        st.dataframe(bus_df.head(), use_container_width=True)
        st.dataframe(edge_df.head(), use_container_width=True)

        # ---- Augmentation (optional) ----
        st.subheader("Augment (optional)")
        do_aug = st.toggle(
            "Replicate the 14-bus graph with noise",
            value=False,
            help="Creates N disjoint copies with light feature noise to increase samples."
        )
        if do_aug and (bus_df is not None) and (edge_df is not None):
            copies = st.slider("Number of copies (N)", 1, 50, 10, 1)
            c1, c2, c3 = st.columns(3)
            with c1:
                sigma_v = st.number_input("Voltage noise σ (pu)", 0.0, 0.2, 0.01, 0.005)
            with c2:
                sigma_load = st.number_input("Load noise σ (MW)", 0.0, 50.0, 3.0, 0.5)
            with c3:
                sigma_pinj = st.number_input("P_inj noise σ (MW)", 0.0, 50.0, 1.0, 0.5)
            flip_breaker_p = st.number_input("Breaker flip prob", 0.0, 0.5, 0.02, 0.01)
            flip_alarm_p   = st.number_input("Alarm flip prob",   0.0, 0.5, 0.02, 0.01)

            bus_df, edge_df = replicate_graph_with_noise(
                bus_df, edge_df, copies=copies,
                sigma_v=sigma_v, sigma_load=sigma_load, sigma_pinj=sigma_pinj,
                flip_breaker_p=flip_breaker_p, flip_alarm_p=flip_alarm_p, seed=42
            )
            st.success(f"Augmented dataset: {len(bus_df)} buses, {len(edge_df)} branches")
with col2:
    st.subheader("2) Build Graph")
    if bus_df is not None:
        # build graph with basic linearized features
        edge_index_np, Xn, y, scaler, bus_to_idx = build_graph(bus_df, edge_df)
        st.write(f"Nodes: **{len(bus_df)}** | Edges (undirected counted): **{edge_index_np.shape[1]}**")
        st.info("Linearized features (array): voltage, load_MW, breaker_status.")

        # topology preview
        G = nx.Graph()
        G.add_nodes_from(bus_df['bus' if 'bus' in bus_df.columns else 'BUS'])
        G.add_edges_from(list(zip(edge_df['from_bus'], edge_df['to_bus'])))

        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(6,4))
        nx.draw(G, pos, with_labels=True, node_size=600, ax=ax)
        ax.set_title("Topology Preview")
        st.pyplot(fig, use_container_width=True)

# -----------------------------
# Training
# -----------------------------
st.subheader("3) Train GNN (GCN)")
if len(missing) > 0 or Data is None or GCNConv is None:
    st.error("PyTorch and/or PyTorch Geometric are not available. See install commands in the sidebar.")
else:
    if bus_df is not None:
        # Hyperparameters (set BEFORE the button so they persist in Streamlit state)
        epochs = st.slider("Epochs", 50, 800, 300, step=50)
        lr     = st.select_slider("Learning Rate", options=[1e-3, 3e-3, 1e-2, 3e-2], value=1e-2)
        wd     = st.select_slider("Weight Decay", options=[0.0, 5e-4, 1e-3], value=5e-4)
        seed   = st.number_input("Seed", value=42, step=1)

        # Build the PyG object once, outside the button.
        data = to_pyg(edge_index_np, Xn, y)

        # (Optional) show class balance and a suggestion for CV splits
        y_np = data.y.cpu().numpy()
        cls, cls_counts = np.unique(y_np, return_counts=True)
        st.caption(f"Class balance → 0: {int(cls_counts[cls.tolist().index(0)] if 0 in cls else 0)}, "
                   f"1: {int(cls_counts[cls.tolist().index(1)] if 1 in cls else 0)}")

        # Cross‑validation controls live OUTSIDE the button so the user can set them first
        use_cv = st.toggle("Use RepeatedStratifiedKFold CV (avg metrics)", value=True,
                           help="Runs CV, averages metrics, and keeps the best fold's weights for inspection.")
        n_splits  = st.select_slider("CV n_splits",  options=[3, 5, 7, 10], value=5, disabled=not use_cv)
        n_repeats = st.select_slider("CV n_repeats", options=[1, 2, 3, 5],  value=3, disabled=not use_cv)

        # Guard: if there are very few positive samples, cap the number of splits
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
                        n_splits=n_splits, n_repeats=n_repeats
                    )
                st.success(f"CV done over {cv_summary['n_folds']} folds.")
                st.write(
                    f"**CV (mean ± std)** — "
                    f"Acc: {cv_summary['acc_mean']:.3f}±{cv_summary['acc_std']:.3f}, "
                    f"Prec: {cv_summary['prec_mean']:.3f}±{cv_summary['prec_std']:.3f}, "
                    f"Rec: {cv_summary['rec_mean']:.3f}±{cv_summary['rec_std']:.3f}, "
                    f"F1: {cv_summary['f1_mean']:.3f}±{cv_summary['f1_std']:.3f}"
                )
            else:
                with st.spinner("Training..."):
                    model, hist_df, train_idx, val_idx, best_th = train_gnn(
                        data, epochs=epochs, lr=lr, weight_decay=wd, seed=seed
                    )

            


            st.success("Training complete. Showing best validation performance observed.")
            st.line_chart(hist_df.set_index("epoch")[["train_loss","val_loss"]])
            st.line_chart(hist_df.set_index("epoch")[["val_acc","val_f1","val_f1_macro"]])

            # Final report on validation nodes
            # Put model in eval mode and disable gradient tracking
            # ── EVAL ──────────────────────────────────────────────────────────────────────
            # Add a threshold slider to trade precision/recall for alarms
            th_default = float(np.clip(best_th, 0.1, 0.9))
            th = st.slider("Decision threshold (alarm)", 0.1, 0.9, th_default, 0.05, help=f"PR-curve suggested threshold: {best_th:.3f}")

            # Show a simple PR curve from validation to explain the suggested threshold
            with torch.no_grad():
                val_logits_for_pr = model(data.x, data.edge_index)[val_idx]
                pr_probs = torch.softmax(val_logits_for_pr, dim=-1)[:, 1].cpu().numpy()
                pr_true  = data.y[val_idx].cpu().numpy()
            p_vals, r_vals, _ = precision_recall_curve(pr_true, pr_probs)
            fig_pr, ax_pr = plt.subplots(figsize=(4, 3))
            ax_pr.plot(r_vals, p_vals)
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title("Validation Precision–Recall")
            st.pyplot(fig_pr, use_container_width=False)

            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)

            # --- Validation metrics (use the SAME indices returned by train_gnn) ---
            val_indices = val_idx.cpu().numpy()
            probs_val = torch.softmax(logits[val_idx], dim=-1)[:, 1].cpu().numpy()
            pred_val  = (probs_val >= th).astype(int)
            true_val  = data.y[val_idx].cpu().numpy()

            from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

            # Classification report
            report = classification_report(true_val, pred_val, digits=3, zero_division=0)
            st.code(report, language="text")

            # Confusion matrix
            cm = confusion_matrix(true_val, pred_val, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(4, 3))
            ConfusionMatrixDisplay(cm, display_labels=["no alarm (0)", "alarm (1)"]).plot(
                ax=ax, values_format="d", colorbar=False
            )
            ax.set_title("Validation Confusion Matrix")
            st.pyplot(fig, use_container_width=False)

            # --- Prediction table (all buses) ---
            probs_all = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            pred_all  = (probs_all >= th).astype(int)

            bus_df_view = bus_df.copy()
            bus_df_view["pred_alarm_prob"]  = probs_all
            bus_df_view["pred_alarm_label"] = pred_all

            # mark which rows are in validation split + if correct (align by val_idx order)
            bus_df_view["is_val"] = 0
            bus_df_view.loc[val_indices, "is_val"] = 1
            # correctness only defined for validation rows; align using val_indices order
            bus_df_view.loc[val_indices, "correct"] = (pred_all[val_indices] == true_val).astype(int)

            st.dataframe(
                bus_df_view.sort_values("pred_alarm_prob", ascending=False),
                use_container_width=True
            )

        # ── SAVE ARTIFACTS (optional) ────────────────────────────────────────────────
        if st.button("💾 Save model + scaler"):
            import pickle
            torch.save(model.state_dict(), "gnn_alarm_model.pt")
            with open("feature_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            st.success("Saved: gnn_alarm_model.pt, feature_scaler.pkl")
            st.download_button("Download model weights", data=open("gnn_alarm_model.pt","rb").read(), file_name="gnn_alarm_model.pt")
            st.download_button("Download scaler", data=open("feature_scaler.pkl","rb").read(), file_name="feature_scaler.pkl")  
    else:
        st.info("Load/create data first.")
