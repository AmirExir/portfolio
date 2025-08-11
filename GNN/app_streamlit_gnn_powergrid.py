
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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import torch
import torch.nn.functional as F

# -----------------------
# Graph building function
# -----------------------

def focal_loss(logits, targets, gamma=2.0, alpha=None):
    """
    Focal loss for class imbalance.
    logits: [N, C] raw output of model
    targets: [N] integer class labels
    gamma: focusing parameter
    alpha: class weighting (tensor of size [C] or None)
    """
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce)  # softmax probability of the true class
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()

def build_graph_dc(bus_df, edge_df, slack_bus='Bus1'):
    # normalize columns locally so this function is robust
    bus_df  = bus_df.rename(columns=str.lower).copy()
    edge_df = edge_df.rename(columns=str.lower).copy()

    bus_to_idx = {b:i for i,b in enumerate(bus_df['bus'])}
    n = len(bus_df)

    # Edge index (undirected)
    src = edge_df['from_bus'].map(bus_to_idx).to_numpy()
    dst = edge_df['to_bus'  ].map(bus_to_idx).to_numpy()
    edge_index = np.vstack([np.r_[src, dst], np.r_[dst, src]])

    # Build B' (susceptance) matrix
    B = np.zeros((n, n), dtype=float)
    for _, row in edge_df.iterrows():
        i, j = bus_to_idx[row['from_bus']], bus_to_idx[row['to_bus']]
        x = float(row['x_pu'])
        b = -1.0 / x
        B[i, j] += b; B[j, i] += b
        B[i, i] -= b; B[j, j] -= b

    # Slack removal and solve
    s = bus_to_idx[slack_bus]
    mask = np.ones(n, dtype=bool); mask[s] = False
    B_red = B[mask][:, mask]

    P = bus_df['p_inj_mw'].to_numpy(float)
    P = P - np.mean(P)
    P_red = P[mask]

    theta = np.zeros(n, dtype=float)
    theta_red = np.linalg.solve(B_red, P_red)
    theta[mask] = theta_red
    theta[s] = 0.0

    # Node flow stats
    flow_abs_sum = np.zeros(n, dtype=float)
    degree = np.zeros(n, dtype=int)
    for _, row in edge_df.iterrows():
        i, j = bus_to_idx[row['from_bus']], bus_to_idx[row['to_bus']]
        x = float(row['x_pu'])
        pij = (theta[i] - theta[j]) / x
        flow_abs_sum[i] += abs(pij); flow_abs_sum[j] += abs(pij)
        degree[i] += 1; degree[j] += 1

    # Features (note: load_mw is lowercased)
    X = np.c_[
        bus_df[['voltage','load_mw','breaker_status']].to_numpy(float),
        theta.reshape(-1,1),
        flow_abs_sum.reshape(-1,1),
        degree.reshape(-1,1)
    ]
    scaler = StandardScaler().fit(X)
    Xn = scaler.transform(X)

    y = bus_df['alarm_flag'].to_numpy().astype(int)
    return edge_index, Xn, y, scaler, bus_to_idx, theta, flow_abs_sum, degree
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
    # --- NEW: DC features toggle ---
    st.subheader("Features")
    use_dc = st.toggle(
        "Add DC power-flow features (θ, |flow| sum, degree)",
        value=False,
        help="Needs columns: bus_features.csv → p_inj_mw; branch_connections.csv → x_pu"
    )
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

    # # simple random split
    # num_nodes = data.num_nodes
    # perm = np.random.permutation(num_nodes)
    # split = int(0.7 * num_nodes)
    # train_idx = torch.tensor(perm[:split], dtype=torch.long, device=data.x.device)
    # val_idx   = torch.tensor(perm[split:], dtype=torch.long, device=data.x.device)
     
    # stratified split by label so minority class appears in both sets
    y_np = data.y.cpu().numpy()
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)
    (train_idx_np, val_idx_np), = sss.split(np.zeros_like(y_np), y_np)
    train_idx = torch.tensor(train_idx_np, dtype=torch.long, device=data.x.device)
    val_idx   = torch.tensor(val_idx_np,   dtype=torch.long, device=data.x.device)
    # class weights from the training subset only
    train_labels_np = y_np[train_idx_np]
    counts = np.bincount(train_labels_np, minlength=2).astype(float)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = torch.tensor(inv / inv.sum() * 2.0, dtype=torch.float, device=data.x.device)
    # ✅ FOCAL LOSS alpha from TRAIN ONLY
    counts_t = torch.bincount(data.y[train_idx], minlength=2).float()
    alpha = 1.0 / (counts_t + 1e-6)
    alpha = (alpha / alpha.sum()).to(data.x.device)                     
    history = []
    best = (1e9, None)  # val loss, state
    for epoch in range(1, epochs+1):
        # Train
        model.train()
        logits = model(data.x, data.edge_index)
        #loss = F.cross_entropy(logits[train_idx], data.y[train_idx], weight=weights)
        for epoch in range(1, epochs+1):
            model.train()
            logits = model(data.x, data.edge_index)
            loss = focal_loss(
                logits[train_idx], 
                data.y[train_idx], 
                gamma=2.0, 
                alpha=alpha        # ✅ pass alpha
            )
            opt.zero_grad(); loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                logits_val = model(data.x, data.edge_index)[val_idx]
            val_loss = focal_loss(
                logits_val, 
                data.y[val_idx], 
                gamma=2.0, 
                alpha=alpha        # ✅ same alpha for val
            ).item()

        acc  = accuracy_score(yv, preds)
        prec = precision_score(yv, preds, average='binary', zero_division=0)
        rec  = recall_score(yv, preds, average='binary', zero_division=0)
        f1   = f1_score(yv, preds, average='binary')
        f1m  = f1_score(yv, preds, average='macro')

        history.append((epoch, float(loss.item()), val_loss, acc, prec, rec, f1, f1m))

        if val_loss < best[0]:
            best = (val_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    # Restore best model
    model.load_state_dict(best[1])

    # Convert history to DataFrame for plotting
    hist_df = pd.DataFrame(
        history,
        columns=["epoch", "train_loss", "val_loss", "val_acc", "val_prec", "val_rec", "val_f1", "val_f1_macro"]
    )

    st.subheader("Validation Metrics Over Epochs")
    st.line_chart(hist_df.set_index("epoch")[["val_acc", "val_f1", "val_f1_macro"]])

    return model, hist_df, train_idx, val_idx

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

with col2:
    st.subheader("2) Build Graph")
    if bus_df is not None:
        # choose which builder to use
        if use_dc:
            need_bus_cols = {'bus','voltage','load_mw','breaker_status','alarm_flag','p_inj_mw'}
            need_edge_cols = {'from_bus','to_bus','x_pu'}

            bus_cols_l = set(c.lower() for c in bus_df.columns)
            edge_cols_l = set(c.lower() for c in edge_df.columns)

            if not need_bus_cols.issubset(bus_cols_l) or not need_edge_cols.issubset(edge_cols_l):
                st.error(
                    "DC features requested, but required columns are missing. "
                    "bus_features.csv needs 'p_inj_mw'; branch_connections.csv needs 'x_pu'. "
                    "Using the basic graph instead."
                )
                edge_index_np, Xn, y, scaler, bus_to_idx = build_graph(bus_df, edge_df)
                used_dc = False
            else:
                # normalize column names to lowercase for the DC path
                bus_df.columns = [c.lower() for c in bus_df.columns]
                edge_df.columns = [c.lower() for c in edge_df.columns]
                edge_index_np, Xn, y, scaler, bus_to_idx, theta_raw, flow_sum_raw, degree_raw = \
                    build_graph_dc(bus_df, edge_df, slack_bus='Bus1')
                used_dc = True
        else:
            edge_index_np, Xn, y, scaler, bus_to_idx = build_graph(bus_df, edge_df)
            used_dc = False

        st.write(f"Nodes: **{len(bus_df)}** | Edges (undirected counted): **{edge_index_np.shape[1]}**")
        if used_dc:
            st.success("DC power-flow features added: bus angles θ, |flow| sum per node, degree.")
        else:
            st.info("Using basic features: voltage, load_MW, breaker_status.")

        # topology preview
        G = nx.Graph()
        G.add_nodes_from(bus_df['bus' if 'bus' in bus_df.columns else 'BUS'])
        G.add_edges_from(list(zip(edge_df['from_bus'], edge_df['to_bus'])))

        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(6,4))
        nx.draw(G, pos, with_labels=True, node_size=600, ax=ax)
        ax.set_title("Topology Preview")
        st.pyplot(fig, use_container_width=True)

        if used_dc:
            # ✅ Use the values you already unpacked from build_graph_dc
            preview = pd.DataFrame({
                'bus': bus_df['bus'],
                'theta (rad, DC)': theta_raw,
                '|flow| sum': flow_sum_raw,
                'degree (count)': degree_raw.astype(int)
            })
            st.dataframe(preview.round(4), use_container_width=True)
        else:
            st.info("Using basic features: voltage, load_MW, breaker_status.")

# -----------------------------
# Training
# -----------------------------
st.subheader("3) Train GNN (GCN)")
if len(missing) > 0 or Data is None or GCNConv is None:
    st.error("PyTorch and/or PyTorch Geometric are not available. See install commands in the sidebar.")
else:
    if bus_df is not None:
        epochs = st.slider("Epochs", 50, 800, 300, step=50)
        lr = st.select_slider("Learning Rate", options=[1e-3, 3e-3, 1e-2, 3e-2], value=1e-2)
        wd = st.select_slider("Weight Decay", options=[0.0, 5e-4, 1e-3], value=5e-4)
        seed = st.number_input("Seed", value=42, step=1)

        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training..."):
                data = to_pyg(edge_index_np, Xn, y)
                model, history, train_idx, val_idx = train_gnn(data, epochs=epochs, lr=lr, weight_decay=wd, seed=seed)

            st.success("Training complete. Showing best validation performance observed.")
            hist_df = pd.DataFrame(history, columns=["epoch","train_loss","val_loss","val_acc","val_f1"])
            st.line_chart(hist_df.set_index("epoch")[["train_loss","val_loss"]])
            st.line_chart(hist_df.set_index("epoch")[["val_acc","val_f1"]])

            # Final report on validation nodes
            # Put model in eval mode and disable gradient tracking
# ── EVAL ──────────────────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)

        # Probability threshold (assumes you created a Streamlit slider earlier named `th`)
        # e.g., th = st.slider("Decision threshold (alarm)", 0.1, 0.9, 0.35, 0.05)

        # --- Validation metrics (use the SAME indices returned by train_gnn) ---
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

        # mark which rows are in validation split + if correct (for those rows)
        mask_val = np.zeros(len(bus_df_view), dtype=bool)
        mask_val[val_idx.cpu().numpy()] = True
        bus_df_view["is_val"] = mask_val.astype(int)
        bus_df_view.loc[mask_val, "correct"] = (pred_all[mask_val] == true_val).astype(int)

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
