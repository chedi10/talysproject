from __future__ import annotations

"""
Base de données graphique enrichie pour le modèle GAT.

Nœuds  : clients
Arêtes  : relations métier (GARANT, FAMILLE, BUSINESS, CO_VILLE, CO_RETARD)
Features: profil + crédit + remboursements + transactions + agrégats voisins
Poids   : risk_relation normalisé selon type et comportement
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.config import (
    CAT_PROFESSION,
    CAT_STATUT_KYC,
    RAW_CLIENTS,
    RAW_CREDITS,
    RAW_RELATIONS,
    RAW_REMBOURSEMENTS,
    RAW_TRANSACTIONS,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.kyc.score import compute_kyc_score_row

REL_TYPES = ("GARANT", "FAMILLE", "BUSINESS", "CO_VILLE", "CO_RETARD")
REL_TYPE_WEIGHT = {
    "GARANT": 1.0,
    "FAMILLE": 0.75,
    "BUSINESS": 0.55,
    "CO_VILLE": 0.25,
    "CO_RETARD": 0.65,
}

FEATURE_NAMES = [
    "age_norm",
    "revenu_norm",
    "profession_enc_norm",
    "kyc_score_norm",
    "statut_kyc_enc_norm",
    "n_credits_norm",
    "avg_dti_norm",
    "max_dti_norm",
    "pct_late_remb_norm",
    "avg_retard_norm",
    "n_suspect_tx_norm",
    "log_degree_norm",
    "neighbor_avg_dti_norm",
    "neighbor_avg_retard_norm",
    "neighbor_risk_weight_norm",
]


@dataclass
class GraphDataset:
    client_ids: np.ndarray
    client_id_to_idx: Dict[int, int]
    edge_index: np.ndarray
    edge_weight: np.ndarray
    neighbors: list[np.ndarray]
    x: np.ndarray
    y: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))


def _enc_map(values: list[str]) -> dict[str, int]:
    return {v: i for i, v in enumerate(values)}


def _zscore(col: np.ndarray) -> np.ndarray:
    col = col.astype(np.float32)
    mu = float(col.mean())
    sigma = float(col.std())
    if sigma < 1e-6:
        return np.zeros_like(col, dtype=np.float32)
    return ((col - mu) / sigma).astype(np.float32)


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clients = pd.read_csv(RAW_CLIENTS, dtype={"cin": str})
    credits = pd.read_csv(RAW_CREDITS, parse_dates=["date_debut"])
    remb = pd.read_csv(RAW_REMBOURSEMENTS)
    tx = pd.read_csv(RAW_TRANSACTIONS)
    rel = pd.read_csv(RAW_RELATIONS)
    return clients, credits, remb, tx, rel


def build_structural_relations(
    clients: pd.DataFrame,
    credits: pd.DataFrame,
    remb: pd.DataFrame,
    *,
    max_edges: int = 25000,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Construit un graphe relationnel réaliste (microfinance).

    - GARANT : lien garant → emprunteur sur crédits (contagion si défaut)
    - FAMILLE : même nom de famille + même ville
    - BUSINESS : Commerçant / Indépendant dans la même ville
    - CO_VILLE : voisins faibles (même ville, revenus proches)
    - CO_RETARD : clients avec profils de retard similaires
    """
    rng = np.random.default_rng(seed)
    clients = clients.drop_duplicates("client_id").copy()
    clients["client_id"] = clients["client_id"].astype(int)

    default_by_client = credits.groupby("client_id")["en_defaut"].max().to_dict()
    credit_stats = credits.groupby("client_id").agg(
        n_credits=("credit_id", "count"),
        avg_dti=("dti", "mean"),
        max_montant=("montant", "max"),
    )
    remb_stats = remb.groupby("client_id").agg(
        avg_retard=("retard_jours", "mean"),
        pct_late=("retard_jours", lambda s: float((s > 0).mean())),
    )

    rows: list[dict] = []
    seen: set[tuple[int, int, str]] = set()
    rel_id = 1

    def _add(src: int, tgt: int, rtype: str, risk: int) -> bool:
        nonlocal rel_id
        if src == tgt:
            return False
        key = (min(src, tgt), max(src, tgt), rtype)
        if key in seen:
            return False
        seen.add(key)
        rows.append(
            {
                "relation_id": rel_id,
                "source_client_id": int(src),
                "target_client_id": int(tgt),
                "type_relation": rtype,
                "risk_relation": int(np.clip(risk, 1, 100)),
            }
        )
        rel_id += 1
        return True

    client_ids = clients["client_id"].tolist()
    by_id = clients.set_index("client_id")

    # ── GARANT : garant → emprunteur (lié aux crédits) ───────────────────────
    business_jobs = {"Commerçant", "Indépendant", "Cadre", "Fonctionnaire"}
    for _, cr in credits.iterrows():
        borrower = int(cr["client_id"])
        if borrower not in by_id.index:
            continue
        b_row = by_id.loc[borrower]
        same_city = clients[
            (clients["ville"] == b_row["ville"])
            & (clients["client_id"] != borrower)
            & (clients["revenu_mensuel"] >= b_row["revenu_mensuel"] * 0.9)
        ]
        if same_city.empty:
            same_city = clients[clients["client_id"] != borrower]
        garant_pool = same_city[same_city["profession"].isin(business_jobs)]
        if garant_pool.empty:
            garant_pool = same_city
        garant = int(rng.choice(garant_pool["client_id"].tolist()))
        defaulted = int(default_by_client.get(borrower, 0))
        dti = float(cr.get("dti", 0.3))
        base_risk = 35 + int(dti * 40)
        if defaulted:
            base_risk = rng.integers(72, 100)
        _add(garant, borrower, "GARANT", base_risk)

    # ── FAMILLE : même nom + même ville ──────────────────────────────────────
    fam_groups = clients.groupby(["nom", "ville"])["client_id"].apply(list)
    for members in fam_groups:
        if len(members) < 2:
            continue
        for i, src in enumerate(members):
            for tgt in members[i + 1 : i + 4]:
                fam_default = max(default_by_client.get(src, 0), default_by_client.get(tgt, 0))
                risk = rng.integers(55, 95) if fam_default else rng.integers(10, 45)
                _add(int(src), int(tgt), "FAMILLE", int(risk))

    # ── BUSINESS : commerçants / indépendants même ville ─────────────────────
    for ville, grp in clients[clients["profession"].isin(["Commerçant", "Indépendant"])].groupby("ville"):
        ids = grp["client_id"].tolist()
        if len(ids) < 2:
            continue
        rng.shuffle(ids)
        for i in range(0, min(len(ids) - 1, 80)):
            src, tgt = int(ids[i]), int(ids[i + 1])
            biz_def = max(default_by_client.get(src, 0), default_by_client.get(tgt, 0))
            risk = rng.integers(50, 88) if biz_def else rng.integers(15, 55)
            _add(src, tgt, "BUSINESS", int(risk))

    # ── CO_VILLE : liens faibles géographiques ───────────────────────────────
    for ville, grp in clients.groupby("ville"):
        ids = grp["client_id"].tolist()
        if len(ids) < 3:
            continue
        sample_n = min(len(ids), 40)
        sampled = rng.choice(ids, size=sample_n, replace=False)
        for src in sampled[:20]:
            tgt = int(rng.choice([i for i in ids if i != src]))
            risk = rng.integers(8, 35)
            _add(int(src), tgt, "CO_VILLE", int(risk))

    # ── CO_RETARD : profils de retard similaires ─────────────────────────────
    remb_enriched = remb_stats.reset_index()
    remb_enriched = remb_enriched[remb_enriched["avg_retard"] > 5].sort_values("avg_retard")
    late_ids = remb_enriched["client_id"].astype(int).tolist()
    for i in range(0, min(len(late_ids) - 1, 1200)):
        src = int(late_ids[i])
        tgt = int(late_ids[min(i + 1, len(late_ids) - 1)])
        if src == tgt:
            continue
        ar = float(remb_stats.loc[src, "avg_retard"]) if src in remb_stats.index else 0
        risk = int(np.clip(30 + ar * 1.2, 20, 98))
        _add(src, tgt, "CO_RETARD", risk)

    df = pd.DataFrame(rows)
    if len(df) > max_edges:
        # Prioriser GARANT > FAMILLE > CO_RETARD > BUSINESS > CO_VILLE
        priority = {"GARANT": 0, "FAMILLE": 1, "CO_RETARD": 2, "BUSINESS": 3, "CO_VILLE": 4}
        df["_prio"] = df["type_relation"].map(priority)
        df = df.sort_values(["_prio", "risk_relation"], ascending=[True, False]).head(max_edges)
        df = df.drop(columns=["_prio"]).reset_index(drop=True)
        df["relation_id"] = np.arange(1, len(df) + 1)

    return df


def _make_undirected_edges(rel: pd.DataFrame, client_id_to_idx: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rel = rel.copy()
    rel["source_client_id"] = rel["source_client_id"].astype(int)
    rel["target_client_id"] = rel["target_client_id"].astype(int)
    rel = rel[
        rel["source_client_id"].isin(client_id_to_idx)
        & rel["target_client_id"].isin(client_id_to_idx)
    ]
    type_w = rel["type_relation"].map(lambda t: REL_TYPE_WEIGHT.get(str(t), 0.4)).astype(float)
    risk_w = rel["risk_relation"].astype(float) / 100.0
    weight = (type_w * (0.4 + 0.6 * risk_w)).astype(np.float32)

    src = rel["source_client_id"].map(client_id_to_idx).to_numpy(dtype=np.int64)
    dst = rel["target_client_id"].map(client_id_to_idx).to_numpy(dtype=np.int64)

    rev_src, rev_dst = dst.copy(), src.copy()
    edge_index = np.stack(
        [np.concatenate([src, rev_src]), np.concatenate([dst, rev_dst])],
        axis=0,
    ).astype(np.int64)
    edge_weight = np.concatenate([weight, weight]).astype(np.float32)
    return edge_index, edge_weight


def _build_neighbors(n_nodes: int, edge_index: np.ndarray) -> list[np.ndarray]:
    src, dst = edge_index[0], edge_index[1]
    neigh: list[list[int]] = [[] for _ in range(n_nodes)]
    for s, d in zip(src.tolist(), dst.tolist()):
        neigh[int(s)].append(int(d))
    return [np.array(v, dtype=np.int64) if v else np.zeros((0,), dtype=np.int64) for v in neigh]


def _client_behavior_features(
    clients: pd.DataFrame,
    credits: pd.DataFrame,
    remb: pd.DataFrame,
    tx: pd.DataFrame,
) -> pd.DataFrame:
    prof_enc = _enc_map(CAT_PROFESSION)
    kyc_enc = _enc_map(CAT_STATUT_KYC)

    base = clients.drop_duplicates("client_id").copy()
    base["client_id"] = base["client_id"].astype(int)
    base["profession_enc"] = base["profession"].map(lambda p: prof_enc.get(str(p), 0)).astype(float)
    base["statut_kyc_enc"] = base["statut_kyc"].map(lambda s: kyc_enc.get(str(s), 0)).astype(float)
    base["kyc_score"] = base.apply(compute_kyc_score_row, axis=1).astype(float)

    cr = credits.groupby("client_id").agg(
        n_credits=("credit_id", "count"),
        avg_dti=("dti", "mean"),
        max_dti=("dti", "max"),
        en_defaut=("en_defaut", "max"),
    ).reset_index()

    rb = remb.groupby("client_id").agg(
        pct_late_remb=("retard_jours", lambda s: float((s > 0).mean())),
        avg_retard=("retard_jours", "mean"),
    ).reset_index()

    txs = tx.groupby("client_id").agg(
        n_tx=("transaction_id", "count"),
        n_suspect_tx=("suspect", "sum"),
    ).reset_index()

    feat = base.merge(cr, on="client_id", how="left")
    feat = feat.merge(rb, on="client_id", how="left")
    feat = feat.merge(txs, on="client_id", how="left")
    for col, default in [
        ("n_credits", 0), ("avg_dti", 0.3), ("max_dti", 0.3), ("en_defaut", 0),
        ("pct_late_remb", 0), ("avg_retard", 0), ("n_tx", 0), ("n_suspect_tx", 0),
    ]:
        feat[col] = feat[col].fillna(default)
    return feat.sort_values("client_id").reset_index(drop=True)


def _neighbor_aggregates(
    n: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    node_values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    src, dst = edge_index[0], edge_index[1]
    out: dict[str, np.ndarray] = {}
    for key, values in node_values.items():
        agg = np.zeros(n, dtype=np.float32)
        wsum = np.zeros(n, dtype=np.float32)
        np.add.at(agg, dst, values[src] * edge_weight)
        np.add.at(wsum, dst, edge_weight)
        out[key] = np.divide(agg, np.maximum(wsum, 1e-6))
    return out


def build_client_graph(
    *,
    relations: pd.DataFrame | None = None,
    rebuild_relations: bool = False,
) -> GraphDataset:
    """Construit le graphe complet avec features enrichies."""
    clients, credits, remb, tx, rel_csv = _load_tables()

    if rebuild_relations or relations is None:
        rel = build_structural_relations(clients, credits, remb)
    else:
        rel = relations

    feat = _client_behavior_features(clients, credits, remb, tx)
    client_ids = feat["client_id"].to_numpy(dtype=np.int64)
    client_id_to_idx = {int(cid): i for i, cid in enumerate(client_ids.tolist())}
    n = len(client_ids)

    edge_index, edge_weight = _make_undirected_edges(rel, client_id_to_idx)
    neighbors = _build_neighbors(n, edge_index)

    degree = np.array([len(nb) for nb in neighbors], dtype=np.float32)
    log_degree = np.log1p(degree)

    neighbor_vals = {
        "avg_dti": feat["avg_dti"].to_numpy(dtype=np.float32),
        "avg_retard": feat["avg_retard"].to_numpy(dtype=np.float32),
    }
    neigh_agg = _neighbor_aggregates(n, edge_index, edge_weight, neighbor_vals)
    neighbor_risk = _neighbor_aggregates(n, edge_index, edge_weight, {"risk": edge_weight})["risk"]

    raw_cols = {
        "age_norm": feat["age"].to_numpy(dtype=np.float32),
        "revenu_norm": feat["revenu_mensuel"].to_numpy(dtype=np.float32),
        "profession_enc_norm": feat["profession_enc"].to_numpy(dtype=np.float32),
        "kyc_score_norm": feat["kyc_score"].to_numpy(dtype=np.float32),
        "statut_kyc_enc_norm": feat["statut_kyc_enc"].to_numpy(dtype=np.float32),
        "n_credits_norm": feat["n_credits"].to_numpy(dtype=np.float32),
        "avg_dti_norm": feat["avg_dti"].to_numpy(dtype=np.float32),
        "max_dti_norm": feat["max_dti"].to_numpy(dtype=np.float32),
        "pct_late_remb_norm": feat["pct_late_remb"].to_numpy(dtype=np.float32),
        "avg_retard_norm": feat["avg_retard"].to_numpy(dtype=np.float32),
        "n_suspect_tx_norm": feat["n_suspect_tx"].to_numpy(dtype=np.float32),
        "log_degree_norm": log_degree,
        "neighbor_avg_dti_norm": neigh_agg["avg_dti"],
        "neighbor_avg_retard_norm": neigh_agg["avg_retard"],
        "neighbor_risk_weight_norm": neighbor_risk,
    }
    x = np.stack([_zscore(raw_cols[name]) for name in FEATURE_NAMES], axis=1).astype(np.float32)
    y = feat["en_defaut"].to_numpy(dtype=np.int64)

    rng = np.random.default_rng(RANDOM_STATE)
    idx_all = np.arange(n, dtype=np.int64)
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
        edge_weight=edge_weight,
        neighbors=neighbors,
        x=x,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        feature_names=list(FEATURE_NAMES),
    )


def save_relations_csv(df: pd.DataFrame, path=None) -> Path:
    from src.config import RAW_RELATIONS

    out = Path(path) if path else RAW_RELATIONS
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
