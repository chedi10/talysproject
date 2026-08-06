from __future__ import annotations

"""
Train Deep Tabular Network (replaces sklearn as production tabular model).

Usage:
    python -m src.models.deep_tabular.train
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    FEATURES_FILE,
    MODELS_DIR,
    MODEL_METADATA_FILE,
    BEST_MODEL_FILE,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.models.deep_tabular.model import DeepTabularNet

# Categorical columns in feature matrix (integer-encoded)
CAT_COLS = ["cycle_enc", "objet_enc", "profession_enc"]
NUMERIC_EXCLUDE = set(CAT_COLS)


def _prepare_data(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in {TARGET_COLUMN, "credit_id"}]
    cat_cols = [c for c in CAT_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X = df[feature_cols]
    y = df[TARGET_COLUMN].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    x_num_train = X_train[num_cols].astype(np.float32).values
    x_num_test = X_test[num_cols].astype(np.float32).values
    x_cat_train = X_train[cat_cols].astype(np.int64).values
    x_cat_test = X_test[cat_cols].astype(np.int64).values

    cat_cardinalities = [int(X[c].max()) + 1 for c in cat_cols]

    return {
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_cardinalities": cat_cardinalities,
        "x_num_train": x_num_train,
        "x_num_test": x_num_test,
        "x_cat_train": x_cat_train,
        "x_cat_test": x_cat_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def train_deep_tabular(epochs: int = 40, batch_size: int = 256, lr: float = 1e-3):
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Run feature engineering first: {FEATURES_FILE}")

    df = pd.read_parquet(FEATURES_FILE)
    data = _prepare_data(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_STATE)

    model = DeepTabularNet(
        n_numeric=len(data["num_cols"]),
        cat_cardinalities=data["cat_cardinalities"],
        embed_dim=8,
        hidden_dims=[256, 128, 64],
        dropout=0.25,
    ).to(device)

    n_pos = float((data["y_train"] == 1).sum())
    n_neg = float((data["y_train"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(
        torch.tensor(data["x_num_train"]),
        torch.tensor(data["x_cat_train"]),
        torch.tensor(data["y_train"], dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_auc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xn, xc, yb in loader:
            xn, xc, yb = xn.to(device), xc.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xn, xc), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()

        model.eval()
        with torch.no_grad():
            xn = torch.tensor(data["x_num_test"], device=device)
            xc = torch.tensor(data["x_cat_test"], device=device)
            proba = torch.sigmoid(model(xn, xc)).cpu().numpy()
        auc = float(roc_auc_score(data["y_test"], proba))
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"[DeepTabular] epoch {epoch}/{epochs} loss={np.mean(losses):.4f} AUC={auc:.4f}")

    ap = float(average_precision_score(data["y_test"], proba))
    artifact = MODELS_DIR / "deep_tabular.pt"
    torch.save(
        {
            "state_dict": best_state or model.state_dict(),
            "num_cols": data["num_cols"],
            "cat_cols": data["cat_cols"],
            "cat_cardinalities": data["cat_cardinalities"],
            "feature_cols": data["feature_cols"],
            "model_type": "deep_tabular",
        },
        artifact,
    )

    meta = {
        "best_model_name": "Deep Tabular Network (MLP+Embeddings)",
        "model_type": "deep_tabular",
        "auc_roc": round(best_auc, 4),
        "avg_precision": round(ap, 4),
        "feature_columns": data["feature_cols"],
        "num_cols": data["num_cols"],
        "cat_cols": data["cat_cols"],
        "cat_cardinalities": data["cat_cardinalities"],
        "target_column": TARGET_COLUMN,
        "artifact": str(artifact),
    }
    with open(MODEL_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Pointer for legacy joblib path — API checks deep_tabular.pt first
    print(f"Deep Tabular saved at {artifact} (AUC={best_auc:.4f})")
    print(f"Metadata updated at {MODEL_METADATA_FILE}")
    return meta


if __name__ == "__main__":
    train_deep_tabular()
