from __future__ import annotations

"""
Train GraphSAGE (node classification) on the client relation graph (REF-08 §1.1.4).

Usage:
    python -m src.models.graph.train
"""

import json
from pathlib import Path

import numpy as np

from src.config import MODELS_DIR, RANDOM_STATE
from src.models.graph.data import build_graph_dataset


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required for GraphSAGE training.\n"
            "Install it in your venv, e.g.:\n"
            "  pip install torch"
        ) from e


def train_graphsage(epochs: int = 40, lr: float = 1e-3, hidden_dim: int = 64, dropout: float = 0.2):
    _require_torch()
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score

    from src.models.graph.model import GraphSAGEClassifier

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    ds = build_graph_dataset()
    x = torch.tensor(ds.x, dtype=torch.float32)
    y = torch.tensor(ds.y, dtype=torch.float32)
    edge_index = torch.tensor(ds.edge_index, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)

    model = GraphSAGEClassifier(in_dim=x.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)

    # Class imbalance
    y_train = ds.y[ds.train_idx]
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_idx = torch.tensor(ds.train_idx, dtype=torch.long, device=device)
    test_idx = torch.tensor(ds.test_idx, dtype=torch.long, device=device)

    best_auc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(x, edge_index)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            logits_eval = model(x, edge_index)
            proba = torch.sigmoid(logits_eval).detach().cpu().numpy()

        y_test = ds.y[ds.test_idx]
        p_test = proba[ds.test_idx]
        auc = float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else 0.5
        ap = float(average_precision_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else 0.0

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[GraphSAGE] epoch {epoch}/{epochs} loss={loss.item():.4f} AUC={auc:.4f} AP={ap:.4f}")

    artifact = MODELS_DIR / "graphsage.pt"
    torch.save(
        {
            "state_dict": best_state if best_state is not None else model.state_dict(),
            "in_dim": int(x.shape[1]),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "client_ids": ds.client_ids.tolist(),
        },
        artifact,
    )

    meta = {
        "model_name": "GraphSAGE",
        "artifact": str(artifact),
        "best_auc": round(float(best_auc), 4),
        "features": ["age", "revenu_mensuel", "profession_enc", "kyc_score"],
        "label_strategy": "client default = max(en_defaut) over all credits",
    }
    meta_path = MODELS_DIR / "graph_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"GraphSAGE saved at {artifact}")
    print(f"Graph metadata saved at {meta_path}")


if __name__ == "__main__":
    train_graphsage()

