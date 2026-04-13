from __future__ import annotations

"""
Train baseline LSTM and GRU models on transaction sequences.

Usage:
    python -m src.models.sequential.train
"""

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import MODELS_DIR, RANDOM_STATE
from src.models.sequential.data import build_sequence_dataset


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required for LSTM/GRU training.\n"
            "Install it in your venv, e.g.:\n"
            "  pip install torch"
        ) from e


def _train_one(model_name: str, rnn_type: str, ds, epochs: int = 8, batch_size: int = 128):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from src.models.sequential.model import RecurrentCreditRiskModel

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = torch.tensor(ds.X_train, dtype=torch.float32)
    y_train = torch.tensor(ds.y_train, dtype=torch.float32)
    X_test = torch.tensor(ds.X_test, dtype=torch.float32)
    y_test = torch.tensor(ds.y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    model = RecurrentCreditRiskModel(
        input_dim=ds.input_dim,
        hidden_dim=64,
        num_layers=1,
        dropout=0.2,
        rnn_type=rnn_type,
    ).to(device)

    # handle class imbalance with pos_weight
    n_pos = float((ds.y_train == 1).sum())
    n_neg = float((ds.y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        print(f"[{model_name}] epoch {epoch}/{epochs} - loss: {np.mean(losses):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device)).cpu().numpy()
        proba = 1.0 / (1.0 + np.exp(-logits))

    auc = float(roc_auc_score(ds.y_test, proba))
    ap = float(average_precision_score(ds.y_test, proba))
    print(f"[{model_name}] AUC={auc:.4f} AP={ap:.4f}")

    out_path = MODELS_DIR / f"sequential_{rnn_type}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": ds.input_dim,
            "seq_len": ds.seq_len,
            "rnn_type": rnn_type,
        },
        out_path,
    )
    print(f"[{model_name}] saved at {out_path}")

    return {"model_name": model_name, "auc_roc": auc, "avg_precision": ap, "artifact": str(out_path)}


def train_sequential_baselines(seq_len: int = 30, epochs: int = 8):
    _require_torch()

    print("Building sequence dataset...")
    ds = build_sequence_dataset(seq_len=seq_len)
    print(
        f"Dataset ready: X_train={ds.X_train.shape}, X_test={ds.X_test.shape}, "
        f"input_dim={ds.input_dim}, seq_len={ds.seq_len}"
    )

    results = []
    results.append(_train_one("LSTM baseline", "lstm", ds, epochs=epochs))
    results.append(_train_one("GRU baseline", "gru", ds, epochs=epochs))

    best = max(results, key=lambda x: x["auc_roc"])
    meta_path = MODELS_DIR / "sequential_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best}, f, indent=2)

    print(f"Best sequential model: {best['model_name']} (AUC={best['auc_roc']:.4f})")
    print(f"Sequential metadata saved at {meta_path}")


if __name__ == "__main__":
    train_sequential_baselines(seq_len=30, epochs=8)

