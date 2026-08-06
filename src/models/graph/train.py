from __future__ import annotations



"""

Train GAT (Graph Attention Network) on enriched client relation graph.



Usage:

    python -m src.models.graph.train

"""



import json



import numpy as np



from src.config import MODELS_DIR, RANDOM_STATE

from src.models.graph.data import build_graph_dataset





def _require_torch():

    try:

        import torch  # noqa: F401

    except ImportError as e:

        raise RuntimeError("PyTorch required. Install: pip install torch") from e





def train_gat(epochs: int = 80, lr: float = 8e-4, hidden_dim: int = 96, n_heads: int = 4, dropout: float = 0.25):

    _require_torch()

    import torch

    from sklearn.metrics import roc_auc_score, average_precision_score

    from src.models.graph.gat_model import GATClassifier



    torch.manual_seed(RANDOM_STATE)

    np.random.seed(RANDOM_STATE)



    ds = build_graph_dataset()

    x = torch.tensor(ds.x, dtype=torch.float32)

    y = torch.tensor(ds.y, dtype=torch.float32)

    edge_index = torch.tensor(ds.edge_index, dtype=torch.long)

    edge_weight = torch.tensor(ds.edge_weight, dtype=torch.float32)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y, edge_index, edge_weight = x.to(device), y.to(device), edge_index.to(device), edge_weight.to(device)



    model = GATClassifier(in_dim=x.shape[1], hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout).to(device)



    y_train = ds.y[ds.train_idx]

    n_pos = float((y_train == 1).sum())

    n_neg = float((y_train == 0).sum())

    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)



    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    train_idx = torch.tensor(ds.train_idx, dtype=torch.long, device=device)



    best_auc = -1.0

    best_state = None

    patience = 15

    stale = 0



    for epoch in range(1, epochs + 1):

        model.train()

        optim.zero_grad()

        logits = model(x, edge_index, edge_weight)

        loss = criterion(logits[train_idx], y[train_idx])

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optim.step()

        scheduler.step()



        model.eval()

        with torch.no_grad():

            proba = torch.sigmoid(model(x, edge_index, edge_weight)).detach().cpu().numpy()



        y_test = ds.y[ds.test_idx]

        p_test = proba[ds.test_idx]

        auc = float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else 0.5

        ap = float(average_precision_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else 0.0



        if auc > best_auc:

            best_auc = auc

            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            stale = 0

        else:

            stale += 1



        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:

            print(f"[GAT] epoch {epoch}/{epochs} loss={loss.item():.4f} AUC={auc:.4f} AP={ap:.4f}")



        if stale >= patience and epoch > 30:

            print(f"[GAT] early stop at epoch {epoch}")

            break



    artifact = MODELS_DIR / "graph_gat.pt"

    torch.save(

        {

            "state_dict": best_state if best_state is not None else model.state_dict(),

            "in_dim": int(x.shape[1]),

            "hidden_dim": int(hidden_dim),

            "n_heads": int(n_heads),

            "dropout": float(dropout),

            "client_ids": ds.client_ids.tolist(),

            "feature_names": ds.feature_names,

            "model_type": "gat",

            "edge_weighted": True,

        },

        artifact,

    )



    meta = {

        "model_name": "GAT (Graph Attention Network)",

        "model_type": "gat",

        "artifact": str(artifact.relative_to(MODELS_DIR.parent)) if artifact.exists() else "models/graph_gat.pt",

        "best_auc": round(float(best_auc), 4),

        "features": ds.feature_names,

        "n_edges": int(ds.edge_index.shape[1]),

        "n_nodes": int(len(ds.client_ids)),

        "edge_weighted": True,

    }

    meta_path = MODELS_DIR / "graph_metadata.json"

    with open(meta_path, "w", encoding="utf-8") as f:

        json.dump(meta, f, indent=2)



    print(f"GAT saved at {artifact} (AUC={best_auc:.4f}, edges={meta['n_edges']})")





if __name__ == "__main__":

    train_gat()

