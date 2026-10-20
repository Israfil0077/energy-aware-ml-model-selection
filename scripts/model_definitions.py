"""The six models, their tuning grids, and the LSTM helpers.

Every notebook creates models through make_model(), so the defaults and the
random seed handling are identical across Phases 3-9.
"""
import os
import time

import numpy as np

import config as C

# ====================================================================== classical models
TABULAR_MODELS = ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM"]


def make_model(name: str, seed: int = C.RANDOM_SEED, **params):
    """Return an untrained model with library defaults (plus any params given)."""
    if name == "Linear Regression":
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**params)
    if name == "Decision Tree":
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor(random_state=seed, **params)
    if name == "Random Forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=seed, n_jobs=C.N_JOBS, **params)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, n_jobs=C.N_JOBS, tree_method="hist", **params)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=seed, n_jobs=C.N_JOBS, verbose=-1, **params)
    raise ValueError(f"unknown model: {name}")


# Grids for Phase 4 (GridSearchCV + TimeSeriesSplit(3)). Kept deliberately small
# because free Colab has 2 CPU cores. The Random Forest grid (agreed 2026-10-04) has
# 5 points instead of 9: full-depth trees are only tried with 50 trees, because
# 100-200 full-depth trees take 13-26 min per fit on free Colab.
PARAM_GRIDS = {
    "Decision Tree": {"max_depth": [5, 10, 15, None], "min_samples_split": [2, 10, 50]},
    "Random Forest": [{"n_estimators": [50], "max_depth": [10, 20, None]},
                      {"n_estimators": [100], "max_depth": [10, 20]}],
    "XGBoost": {"n_estimators": [100, 300], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]},
    "LightGBM": {"n_estimators": [100, 300], "max_depth": [4, 8, -1], "learning_rate": [0.05, 0.1]},
}


# ====================================================================== LSTM
# Settings agreed by the team on 2026-10-04.
LSTM_SEQ_FEATURES = ["cpu_usage_pct", "mem_usage_pct"]   # per-step inputs, scaled /100
LSTM_DEFAULTS = {
    "seq_len": 12,          # 12 readings x 5 min = the last hour
    "hidden_size": 64,
    "num_layers": 1,
    "lr": 1e-3,
    "batch_size": 256,
    "max_epochs": 15,
    "patience": 3,          # early stopping on a chronological validation slice
    "val_fraction": 0.10,   # last 10% of each VM's training period
}
# Phase 4 LSTM search: 4 settings, chosen on the validation slice (not a full grid).
LSTM_TUNING_GRID = [
    {"hidden_size": h, "seq_len": s} for h in (32, 64) for s in (6, 12)
]


def make_sequences(featured_df, seq_len: int):
    """Turn the per-row feature table into LSTM windows.

    For each row t of each VM, the window holds the per-step inputs for rows
    t-seq_len+1 .. t (history only) and the label is the same target the tree
    models predict (CPU% at t+1). Windows may reach back into the training
    period when t is a test row; that is past information, so not leakage.
    Returns X (n, seq_len, n_feat), y (n,), and the row index each window belongs to.
    """
    Xs, ys, idx = [], [], []
    for _, g in featured_df.groupby("vm_id", sort=False):
        vals = g[LSTM_SEQ_FEATURES].to_numpy(dtype=np.float32) / 100.0
        tgt = g[C.TARGET].to_numpy(dtype=np.float32) / 100.0
        n = len(g)
        if n < seq_len:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(vals, (seq_len, vals.shape[1]))[:, 0]
        Xs.append(windows)
        ys.append(tgt[seq_len - 1:])
        idx.append(g.index.to_numpy()[seq_len - 1:])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(idx)


def lstm_datasets(features_main, seq_len: int, val_fraction: float = LSTM_DEFAULTS["val_fraction"]):
    """Split LSTM windows into train / validation / test using the Phase 2 'split' column.

    - test: windows whose target row is a test row (same rows, same order as test.parquet)
    - val:  windows whose target row is in the last `val_fraction` of each VM's
            training period (used only for early stopping and LSTM tuning)
    - train: the remaining training windows
    """
    X, y, idx = make_sequences(features_main, seq_len)
    split = features_main["split"].to_numpy()[idx]
    tr = features_main[features_main["split"] == "train"]
    pos = tr.groupby("vm_id").cumcount()
    n = tr.groupby("vm_id")["vm_id"].transform("size")
    val_rows = tr.index[(pos >= np.floor(n * (1 - val_fraction))).to_numpy()]
    is_val = np.isin(idx, val_rows)
    is_test = split == "test"
    is_train = (split == "train") & ~is_val
    return {
        "train": (X[is_train], y[is_train]),
        "val": (X[is_val], y[is_val]),
        "test": (X[is_test], y[is_test], idx[is_test]),
    }


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _torch():
    import torch
    torch.set_num_threads(os.cpu_count() or 1)
    return torch


def build_lstm(n_features: int, hidden_size: int, num_layers: int, seed: int):
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)

    class LSTMRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    return LSTMRegressor()


def train_lstm(X_train, y_train, X_val=None, y_val=None, seed: int = C.RANDOM_SEED,
               max_epochs=None, log_fn=print, **overrides):
    """Train the LSTM with Adam + MSE, early stopping on (X_val, y_val) if given.
    Returns (model, history) where history lists per-epoch losses and times."""
    torch = _torch()
    p = {**LSTM_DEFAULTS, **overrides}
    if max_epochs is not None:
        p["max_epochs"] = max_epochs
    np.random.seed(seed)
    model = build_lstm(X_train.shape[2], p["hidden_size"], p["num_layers"], seed)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
    loss_fn = torch.nn.MSELoss()
    Xt, yt = torch.from_numpy(X_train), torch.from_numpy(y_train)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad_epochs, history = np.inf, None, 0, []
    for epoch in range(1, p["max_epochs"] + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(Xt), p["batch_size"]):
            b = perm[i:i + p["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        train_loss = total / len(Xt)
        val_loss = None
        if X_val is not None:
            val_loss = float(np.mean((predict_lstm(model, X_val) / 100.0 - y_val) ** 2))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "seconds": time.perf_counter() - t0})
        log_fn(f"  epoch {epoch:>2}: train_mse={train_loss:.5f}"
               + (f" val_mse={val_loss:.5f}" if val_loss is not None else "")
               + f" ({history[-1]['seconds']:.1f}s)")
        if val_loss is not None:
            if val_loss < best - 1e-6:
                best, bad_epochs = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= p["patience"]:
                    log_fn(f"  early stop at epoch {epoch} (best val_mse={best:.5f})")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_lstm(model, X, batch_size: int = 4096):
    """Predict CPU% (already scaled back to 0-100)."""
    torch = _torch()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(torch.from_numpy(X[i:i + batch_size])).numpy())
    return np.concatenate(out) * 100.0
