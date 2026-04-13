from __future__ import annotations

"""
Build a graph dataset for GraphSAGE (node classification).

Nodes   : clients (client_id)
Edges   : relations.csv (source_client_id -> target_client_id)
Labels  : derived from credits.csv (client-level default label)
Features: client attributes + estimated kyc_score

This is a lightweight implementation (no PyG) to keep dependencies minimal.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import RAW_CLIENTS, RAW_CREDITS, RAW_RELATIONS, TEST_SIZE, RANDOM_STATE, CAT_PROFESSION
from src.kyc.score import compute_kyc_score_row


@dataclass
class GraphDataset:
    # Mapping
    client_ids: np.ndarray               # (N,)
    client_id_to_idx: Dict[int, int]

    # Graph structure
    edge_index: np.ndarray               # (2, E) int64, undirected
    neighbors: list[np.ndarray]          # length N, each int64 neighbor indices

    # Node features + labels
    x: np.ndarray                        # (N, F) float32
    y: np.ndarray                        # (N,) int64 (0/1)

    # Splits
    train_idx: np.ndarray                # (n_train,) int64
    test_idx: np.ndarray                 # (n_test,) int64


def _profession_to_enc_series(prof: pd.Series) -> pd.Series:
    m = {p: i for i, p in enumerate(CAT_PROFESSION)}
    return prof.map(lambda v: int(m.get(str(v), 0)))


def _build_client_labels(credits: pd.DataFrame) -> pd.DataFrame:
    """
    Client-level label from credit-level labels.

    Strategy: label client as 1 if they ever defaulted on any credit; else 0.
    """
    cr = credits[["client_id", "en_defaut"]].copy()
    cr["client_id"] = cr["client_id"].astype(int)
    cr["en_defaut"] = cr["en_defaut"].astype(int)
    y = cr.groupby("client_id")["en_defaut"].max().rename("y").reset_index()
    return y


def _make_undirected_edges(rel: pd.DataFrame) -> pd.DataFrame:
    e = rel[["source_client_id", "target_client_id"]].copy()
    e["source_client_id"] = e["source_client_id"].astype(int)
    e["target_client_id"] = e["target_client_id"].astype(int)
    # Add reverse edges to make graph undirected
    rev = e.rename(columns={"source_client_id": "target_client_id", "target_client_id": "source_client_id"})
    e2 = pd.concat([e, rev], ignore_index=True)
    e2 = e2.drop_duplicates()
    return e2


def _build_neighbors(n_nodes: int, edge_index: np.ndarray) -> list[np.ndarray]:
    src = edge_index[0]
    dst = edge_index[1]
    neigh = [[] for _ in range(n_nodes)]
    for s, d in zip(src.tolist(), dst.tolist()):
        neigh[int(s)].append(int(d))
    return [np.array(v, dtype=np.int64) if len(v) else np.zeros((0,), dtype=np.int64) for v in neigh]


def build_graph_dataset() -> GraphDataset:
    clients = pd.read_csv(RAW_CLIENTS, dtype={"cin": str})
    credits = pd.read_csv(RAW_CREDITS, parse_dates=["date_debut"])
    rel = pd.read_csv(RAW_RELATIONS)

    # --- Node set ---
    clients = clients.drop_duplicates(subset=["client_id"]).copy()
    clients["client_id"] = clients["client_id"].astype(int)
    clients = clients.sort_values("client_id")

    client_ids = clients["client_id"].to_numpy(dtype=np.int64)
    client_id_to_idx = {int(cid): int(i) for i, cid in enumerate(client_ids.tolist())}
    n = int(len(client_ids))

    # --- Labels (client-level) ---
    y_df = _build_client_labels(credits)
    clients = clients.merge(y_df, on="client_id", how="left")
    clients["y"] = clients["y"].fillna(0).astype(int)

    # --- Node features ---
    clients["profession_enc"] = _profession_to_enc_series(clients["profession"])
    # Compute kyc_score from row (age, revenu_mensuel, profession string)
    clients["kyc_score"] = clients.apply(compute_kyc_score_row, axis=1)

    # Features: age, revenu, profession_enc, kyc_score
    x = np.stack(
        [
            clients["age"].astype(float).to_numpy(),
            clients["revenu_mensuel"].astype(float).to_numpy(),
            clients["profession_enc"].astype(float).to_numpy(),
            clients["kyc_score"].astype(float).to_numpy(),
        ],
        axis=1,
    ).astype(np.float32)

    y = clients["y"].to_numpy(dtype=np.int64)

    # --- Edges ---
    rel_u = _make_undirected_edges(rel)
    # Keep only edges where both endpoints are known clients
    rel_u = rel_u[
        rel_u["source_client_id"].isin(client_id_to_idx.keys())
        & rel_u["target_client_id"].isin(client_id_to_idx.keys())
    ].copy()

    src_idx = rel_u["source_client_id"].map(client_id_to_idx).to_numpy(dtype=np.int64)
    dst_idx = rel_u["target_client_id"].map(client_id_to_idx).to_numpy(dtype=np.int64)
    edge_index = np.stack([src_idx, dst_idx], axis=0).astype(np.int64)

    neighbors = _build_neighbors(n_nodes=n, edge_index=edge_index)

    # --- Train/test split on nodes (stratified) ---
    rng = np.random.default_rng(RANDOM_STATE)
    idx_all = np.arange(n, dtype=np.int64)

    # Simple stratified split (no sklearn dependency here)
    pos = idx_all[y == 1]
    neg = idx_all[y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_test = int(round(len(pos) * TEST_SIZE))
    n_neg_test = int(round(len(neg) * TEST_SIZE))
    test_idx = np.concatenate([pos[:n_pos_test], neg[:n_neg_test]]).astype(np.int64)
    train_idx = np.setdiff1d(idx_all, test_idx).astype(np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return GraphDataset(
        client_ids=client_ids,
        client_id_to_idx=client_id_to_idx,
        edge_index=edge_index,
        neighbors=neighbors,
        x=x,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
    )

