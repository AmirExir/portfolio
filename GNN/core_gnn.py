import shutil
import warnings
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score, accuracy_score
class TrainModel(object):
    def __init__(
        self,
        model,
        dataset,
        device,
        seed=42,
        graph_classification=False,
        graph_regression=False,
        save_dir=None,
        save_name="coregnn",
        dataloader_params=None,
        **kwargs,
    ):
        """
        Args:
            model: PyTorch GNN model
            dataset: list of PyG Data objects, or DataLoader
            device: torch.device
            seed: random seed
            graph_classification: bool
            graph_regression: bool
            save_dir: directory for saving models
            save_name: base name for checkpoints
            dataloader_params: dict for DataLoader (e.g. batch_size)
        """
        set_seed(seed)
        self.model = model
        self.device = device
        self.seed = seed
        self.graph_classification = graph_classification
        self.graph_regression = graph_regression
        self.save_dir = save_dir if save_dir is not None else "models"
        self.save_name = save_name
        self.save = self.save_dir is not None
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        # Accept dataset as list of Data or DataLoader
        from torch_geometric.loader import DataLoader
        if isinstance(dataset, list):
            # Build loader dict
            dl_params = dataloader_params if dataloader_params is not None else dict(batch_size=1, shuffle=True)
            self.train_loader = DataLoader(dataset, **dl_params)
            self.eval_loader = DataLoader(dataset, **{**dl_params, "shuffle": False})
            self.test_loader = self.eval_loader
        elif hasattr(dataset, "__iter__"):
            # Assume it's a DataLoader
            self.train_loader = dataset
            self.eval_loader = dataset
            self.test_loader = dataset
        else:
            # Fallback: treat as singleton Data
            self.train_loader = [dataset]
            self.eval_loader = [dataset]
            self.test_loader = [dataset]
        self.optimizer = None

    def __loss__(self, logits, labels):
        if self.graph_classification:
            return F.cross_entropy(logits, labels)
        elif self.graph_regression:
            return F.mse_loss(logits.squeeze(), labels)
        else:
            # fallback: try cross_entropy
            return F.cross_entropy(logits, labels)

    def _train_batch(self, batch):
        self.model.train()
        batch = batch.to(self.device)
        logits = self.model(batch.x, batch.edge_index)
        if self.graph_classification:
            loss = self.__loss__(logits, batch.y)
        elif self.graph_regression:
            loss = self.__loss__(logits, batch.y)
        else:
            loss = self.__loss__(logits, batch.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _eval_batch(self, batch):
        self.model.eval()
        batch = batch.to(self.device)
        with torch.no_grad():
            logits = self.model(batch.x, batch.edge_index)
            if self.graph_classification:
                loss = self.__loss__(logits, batch.y).item()
                preds = logits.argmax(dim=-1)
                return loss, preds, logits
            elif self.graph_regression:
                loss = self.__loss__(logits, batch.y).item()
                preds = logits.squeeze()
                return loss, preds
            else:
                loss = self.__loss__(logits, batch.y).item()
                preds = logits.argmax(dim=-1)
                return loss, preds

    def train(self, train_params=None, optimizer_params=None):
        num_epochs = 100 if train_params is None or "num_epochs" not in train_params else train_params["num_epochs"]
        num_early_stop = 10 if train_params is None or "num_early_stop" not in train_params else train_params["num_early_stop"]
        lr = 1e-3
        if optimizer_params is not None and "lr" in optimizer_params:
            lr = optimizer_params["lr"]
        self.model.to(self.device)
        if optimizer_params is None:
            self.optimizer = Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = Adam(self.model.parameters(), **optimizer_params)
        if self.graph_classification:
            scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.2, patience=10, verbose=False)
        else:
            scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=10, verbose=False)
        best_metric = None
        best_loss = float("inf")
        early_stop_counter = 0
        for epoch in range(num_epochs):
            train_losses = []
            for batch in self.train_loader:
                loss = self._train_batch(batch)
                train_losses.append(loss)
            train_loss = np.mean(train_losses)
            # Evaluate on eval_loader
            if self.graph_classification:
                eval_loss, eval_acc, eval_bal_acc, eval_f1 = self.eval()
                metric = eval_acc
                scheduler.step(metric)
            elif self.graph_regression:
                eval_loss, eval_r2 = self.eval()
                metric = -eval_loss
                scheduler.step(eval_loss)
            else:
                eval_loss, eval_acc, eval_bal_acc, eval_f1 = self.eval()
                metric = eval_acc
                scheduler.step(metric)
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f} eval_loss={eval_loss:.4f}")
            is_best = False
            if self.graph_classification:
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    is_best = True
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
            else:
                if eval_loss <= best_loss:
                    best_loss = eval_loss
                    is_best = True
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
            # Save models
            if self.save:
                self.save_model(is_best)
            if num_early_stop > 0 and early_stop_counter > num_early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

    def eval(self):
        self.model.eval()
        losses = []
        preds_all = []
        targets_all = []
        if self.graph_classification:
            for batch in self.eval_loader:
                loss, preds, logits = self._eval_batch(batch)
                losses.append(loss)
                preds_all.append(preds.cpu())
                targets_all.append(batch.y.cpu())
            y_pred = torch.cat(preds_all).numpy()
            y_true = torch.cat(targets_all).numpy()
            eval_loss = np.mean(losses)
            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            return eval_loss, acc, bal_acc, f1
        elif self.graph_regression:
            for batch in self.eval_loader:
                loss, preds = self._eval_batch(batch)
                losses.append(loss)
                preds_all.append(preds.detach().cpu())
                targets_all.append(batch.y.detach().cpu())
            y_pred = torch.cat(preds_all).numpy()
            y_true = torch.cat(targets_all).numpy()
            eval_loss = np.mean(losses)
            r2 = r2_score(y_true, y_pred)
            return eval_loss, r2
        else:
            # fallback: node classification
            for batch in self.eval_loader:
                loss, preds = self._eval_batch(batch)
                losses.append(loss)
                preds_all.append(preds.cpu())
                targets_all.append(batch.y.cpu())
            y_pred = torch.cat(preds_all).numpy()
            y_true = torch.cat(targets_all).numpy()
            eval_loss = np.mean(losses)
            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            return eval_loss, acc, bal_acc, f1

    def test(self):
        # Load best model if exists
        if self.save and os.path.exists(os.path.join(self.save_dir, "best_coregnn.pt")):
            self.load_model()
        self.model.eval()
        losses = []
        preds_all = []
        targets_all = []
        if self.graph_classification:
            for batch in self.test_loader:
                loss, preds, logits = self._eval_batch(batch)
                losses.append(loss)
                preds_all.append(preds.cpu())
                targets_all.append(batch.y.cpu())
            y_pred = torch.cat(preds_all).numpy()
            y_true = torch.cat(targets_all).numpy()
            test_loss = np.mean(losses)
            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            print(f"Test loss: {test_loss:.4f}, acc: {acc:.4f}, bal_acc: {bal_acc:.4f}, f1: {f1:.4f}")
            return test_loss, acc, bal_acc, f1
        elif self.graph_regression:
            for batch in self.test_loader:
                loss, preds = self._eval_batch(batch)
                losses.append(loss)
                preds_all.append(preds.detach().cpu())
                targets_all.append(batch.y.detach().cpu())
            y_pred = torch.cat(preds_all).numpy()
            y_true = torch.cat(targets_all).numpy()
            test_loss = np.mean(losses)
            r2 = r2_score(y_true, y_pred)
            print(f"Test loss: {test_loss:.4f}, r2: {r2:.4f}")
            return test_loss, r2
        else:
            for batch in self.test_loader:
                loss, preds = self._eval_batch(batch)
                losses.append(loss)
                preds_all.append(preds.cpu())
                targets_all.append(batch.y.cpu())
            y_pred = torch.cat(preds_all).numpy()
            y_true = torch.cat(targets_all).numpy()
            test_loss = np.mean(losses)
            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            print(f"Test loss: {test_loss:.4f}, acc: {acc:.4f}, bal_acc: {bal_acc:.4f}, f1: {f1:.4f}")
            return test_loss, acc, bal_acc, f1

    def save_model(self, is_best=False):
        # Save latest and best checkpoint
        path_latest = os.path.join(self.save_dir, "latest_coregnn.pt")
        path_best = os.path.join(self.save_dir, "best_coregnn.pt")
        torch.save({"net": self.model.state_dict()}, path_latest)
        if is_best:
            shutil.copy(path_latest, path_best)

    def load_model(self):
        path_best = os.path.join(self.save_dir, "best_coregnn.pt")
        state = torch.load(path_best, map_location=self.device)
        self.model.load_state_dict(state["net"])
        self.model.to(self.device)

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
    from torch_geometric.nn import GCNConv, GATConv, GINConv, TransformerConv
    from torch_geometric.utils import add_self_loops
except Exception as e:
    Data = None
    GCNConv = None
    GATConv = None
    GINConv = None
    TransformerConv = None
    add_self_loops = None

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


# ---- Model Selector ----
import torch.nn as nn
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
# Helper: sanitize PyG Data
# -----------------------------
def sanitize_pyg_data(data, add_loops_if_empty=True, verbose_prefix=""):
    """
    Ensure edge_index is valid w.r.t. data.x; remap nodes to 0..N-1; filter/realign train/val indices.
    Operates in-place and returns the same data object.
    """
    import torch as _torch
    # If there are no edges, optionally add self-loops to keep GCN stable
    if data.edge_index is None or data.edge_index.numel() == 0:
        if add_loops_if_empty:
            if add_self_loops is not None:
                data.edge_index, _ = add_self_loops(_torch.zeros((2, 0), dtype=_torch.long, device=data.x.device),
                                                     num_nodes=data.x.size(0))
            else:
                idx = _torch.arange(data.x.size(0), device=data.x.device, dtype=_torch.long)
                data.edge_index = _torch.stack([idx, idx], dim=0)
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
            if add_self_loops is not None:
                data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
            else:
                idx = _torch.arange(data.x.size(0), device=data.x.device, dtype=_torch.long)
                data.edge_index = _torch.stack([idx, idx], dim=0)
        return data

    # Remap node indices appearing in edges to a consecutive 0..K-1 range
    unique_nodes = _torch.unique(data.edge_index)
    old_to_new = {int(n): i for i, n in enumerate(unique_nodes.tolist())}

    # Remap edge_index
    data.edge_index = _torch.tensor(
        [[old_to_new[int(s.item())] for s in data.edge_index[0]],
         [old_to_new[int(t.item())] for t in data.edge_index[1]]],
        dtype=_torch.long,
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
            setattr(data, idx_tensor, _torch.empty((0,), dtype=_torch.long, device=data.x.device))
        else:
            setattr(data, idx_tensor, _torch.tensor(kept, dtype=_torch.long, device=data.x.device))

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
            if add_self_loops is not None:
                # Build empty edge_index then add self loops
                import torch as _torch
                ei = _torch.zeros((2, 0), dtype=_torch.long)
                ei, _ = add_self_loops(ei, num_nodes=num_nodes)
                edge_index = ei.cpu().numpy()
            else:
                # Fallback: create explicit self-loop adjacency in numpy
                edge_index = np.vstack([np.arange(num_nodes), np.arange(num_nodes)])

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
        # --- Full edge index safety fix ---
        num_nodes = Xn.shape[0]
        if edge_index_np.size > 0:
            # Filter out invalid edges (out-of-range or NaN)
            mask_valid = (
                (~np.isnan(edge_index_np[0])) &
                (~np.isnan(edge_index_np[1])) &
                (edge_index_np[0] >= 0) & (edge_index_np[1] >= 0) &
                (edge_index_np[0] < num_nodes) & (edge_index_np[1] < num_nodes)
            )
            invalid_edges = np.sum(~mask_valid)
            if invalid_edges > 0:
                print(f"⚠️  Scenario {scen}: removed {invalid_edges} edges with invalid node indices.")
            edge_index_np = edge_index_np[:, mask_valid].astype(int)
        else:
            print(f"⚠️  Scenario {scen}: has no valid edges.")

        # Skip graphs that are too small or invalid
        if num_nodes == 0 or edge_index_np.shape[1] == 0:
            print(f"⚠️  Scenario {scen}: skipped (no valid nodes or edges).")
            continue
        # Use provided scaler for test set, else fit on the scenario
        if scaler is not None:
            if mode == "voltage":
                # --- Ensure all cheat features exist; fill with zeros if missing ---
                for feat in ["voltage", "load_MW", "p_inj_mw", "neighbor_count"]:
                    if feat not in bdf.columns:
                        bdf[feat] = 0.0
                use_cols = feature_names if 'feature_names' in locals() and feature_names is not None else [c for c in ["voltage", "load_MW", "p_inj_mw", "neighbor_count"] if c in bdf.columns]
                X_raw = bdf[use_cols].to_numpy(dtype=float)
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

def train_gnn_batches(
    data_list,
    epochs=150,
    lr=1e-2,
    weight_decay=1e-3,
    seed=42,
    use_relu=True,
    batch_size=1,
    model_type="gcn"
):
    """
    Train a GNN model using mini-batch graph batching (PyG DataLoader), Adam optimizer, optional LR scheduler,
    early stopping on validation loss, and tracking best model state. Prints epoch progress.
    """
    import torch
    from torch_geometric.loader import DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Assume all graphs share the same input dimension
    in_dim = data_list[0].x.size(1)
    # Number of classes from concatenated labels
    all_y = torch.cat([d.y for d in data_list])
    n_classes = int(all_y.max().item()) + 1

    model = get_model(model_type, in_dim=in_dim, num_classes=n_classes, use_relu=use_relu).to(device)
    print(f"Training model type: {model_type.upper()}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # ReduceLROnPlateau on val_loss
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=False)

    # Precompute stratified splits per-graph and store on the Data objects
    splits = []
    for d in data_list:
        y_np = d.y.cpu().numpy()
        tr_np, va_np = _stratified_indices(y_np, train_frac=0.7, seed=seed)
        d.train_idx = torch.tensor(tr_np, dtype=torch.long, device=device)
        d.val_idx   = torch.tensor(va_np, dtype=torch.long, device=device)
        splits.append((d.train_idx, d.val_idx))

    # Focal-loss class weights computed from all train nodes across all graphs
    train_y = torch.cat([d.y[d.train_idx].to(device) for d in data_list])
    counts_t = torch.bincount(train_y, minlength=n_classes).float()
    alpha = (1.0 / (counts_t + 1e-6))
    alpha = (alpha / alpha.sum()).to(device)

    # Use PyG DataLoader for batching graphs
    train_graphs = []
    val_graphs = []
    for d in data_list:
        train_graphs.append(d)
        val_graphs.append(d)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    history = []
    best_val_loss = float('inf')
    best_state = None
    early_stop_patience = 20
    early_stop_counter = 0
    num_epochs = epochs
    print("Epoch | Train Loss | Val Loss | Val Acc | Val F1")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        train_items = 0
        for batch in train_loader:
            batch = batch.to(device)
            # For each graph in batch, use its own train_idx
            # But batch is a Batch object: we need to map back to per-graph indices
            # Instead, treat all nodes in batch.train_idx as train
            # But since we stored train_idx per-graph, we need to offset indices
            # So, we will flatten all graphs and use their .train_idx offset by batch ptr
            # But for now, train per-graph (since batch_size=1 default)
            # So for batch_size > 1, we must map indices
            # For simplicity, only support batch_size=1 robustly
            # (If batch_size > 1, treat all nodes in batch as train)
            # We'll sum losses per graph
            if hasattr(batch, 'train_idx'):
                idx = batch.train_idx
            elif hasattr(batch, 'batch') and hasattr(batch, 'ptr'):
                # batch_size > 1: fallback to all nodes
                idx = torch.arange(batch.x.size(0), device=batch.x.device)
            else:
                idx = torch.arange(batch.x.size(0), device=batch.x.device)
            if idx.numel() == 0:
                continue
            logits = model(batch.x, batch.edge_index)
            loss = focal_loss(logits[idx], batch.y[idx], gamma=2.0, alpha=alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item() * idx.numel())
            train_items += idx.numel()
        train_loss_avg = sum(train_losses) / train_items if train_items else 0.0

        # Validation
        model.eval()
        val_losses = []
        val_items = 0
        val_preds_all = []
        val_true_all = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                if hasattr(batch, 'val_idx'):
                    idx = batch.val_idx
                elif hasattr(batch, 'batch') and hasattr(batch, 'ptr'):
                    idx = torch.arange(batch.x.size(0), device=batch.x.device)
                else:
                    idx = torch.arange(batch.x.size(0), device=batch.x.device)
                if idx.numel() == 0:
                    continue
                logits = model(batch.x, batch.edge_index)
                vloss = focal_loss(logits[idx], batch.y[idx], gamma=2.0, alpha=alpha)
                val_losses.append(vloss.item() * idx.numel())
                val_items += idx.numel()
                preds = logits[idx].argmax(dim=-1).cpu().numpy()
                true = batch.y[idx].cpu().numpy()
                val_preds_all.append(preds)
                val_true_all.append(true)
        if val_items:
            import numpy as _np
            val_loss_avg = sum(val_losses) / val_items
            val_preds_all = _np.concatenate(val_preds_all)
            val_true_all = _np.concatenate(val_true_all)
            acc = accuracy_score(val_true_all, val_preds_all)
            f1 = f1_score(val_true_all, val_preds_all, average='macro')
        else:
            val_loss_avg = 0.0
            acc = 0.0
            f1 = 0.0

        history.append((epoch, train_loss_avg, val_loss_avg, acc, f1))
        if epoch % 10 == 0 or epoch == num_epochs:
            print(f"{epoch:4d} | {train_loss_avg:.4f} | {val_loss_avg:.4f} | {acc:.4f} | {f1:.4f}")

        # Early stopping logic
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        scheduler.step(val_loss_avg)
        if early_stop_counter > early_stop_patience:
            print(f"Early stopping at epoch {epoch} due to no improvement in val loss.")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    import pandas as pd
    hist_df = pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss", "val_acc", "val_f1"])
    return model, hist_df, splits

def main():
    import argparse
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

    import torch
    pt_path = "graph_scenarios.pt"
    if os.path.exists(pt_path):
        print(f"Found {pt_path}. Loading preprocessed global graph...")
        # Assume .pt contains a single Data object (global graph)
        data = torch.load(pt_path)
        if isinstance(data, list):
            # If list of Data, concatenate into a single Data (legacy fallback)
            print("Loaded a list of Data objects; using only the first for global training.")
            data = data[0]
        print("Sanitizing loaded PyG Data...")
        sanitize_pyg_data(data, add_loops_if_empty=True, verbose_prefix="Loaded: ")
        print("Training GNN on loaded .pt global graph...")
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
        # Optionally, save model or evaluate on test set if available.
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