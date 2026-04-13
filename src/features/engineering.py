"""
engineering.py – Feature Engineering pipeline.

Aggregates remboursements, transactions, relations, client info and
credit info into a single model-ready feature matrix keyed by credit_id.

Usage:
    # As a module
    python -m src.features.engineering

    # Or in code
    from src.features.engineering import build_features
    df_features = build_features()
"""
import pandas as pd
import numpy as np
from src.config import (
    FEATURES_FILE,
    TARGET_COLUMN,
    MODELS_DIR,
    CAT_CYCLE,
    CAT_OBJET,
    CAT_PROFESSION,
)
from src.data.loader import load_raw_data
from src.data.cleaner import clean_and_save, load_processed
from src.kyc.score import fit_kyc_scorer, compute_kyc_score_row


# ─── 1. Remboursement features (per credit) ───────────────────────────────────
def _remb_features(remb: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate repayment behaviour per credit.

    Features:
        avg_retard      – average delay in days
        max_retard      – worst single payment delay
        std_retard      – volatility of delays
        n_payments      – total number of monthly payments
        n_late          – count of payments with delay > 0
        pct_late        – fraction of late payments
        n_en_retard     – count of payments with statut="EN_RETARD" (delay ≥ 90 days)
    """
    g = remb.groupby("credit_id")["retard_jours"]
    agg = pd.DataFrame({
        "avg_retard":  g.mean(),
        "max_retard":  g.max(),
        "std_retard":  g.std().fillna(0),
        "n_payments":  g.count(),
        "n_late":      remb.groupby("credit_id")["retard_jours"].apply(lambda x: (x > 0).sum()),
    })
    agg["pct_late"] = agg["n_late"] / agg["n_payments"].clip(lower=1)

    # Special: n_en_retard = severe late (≥ 90 days)
    severe = remb[remb["retard_jours"] >= 90].groupby("credit_id").size().rename("n_en_retard")
    agg = agg.join(severe, how="left")
    agg["n_en_retard"] = agg["n_en_retard"].fillna(0).astype(int)

    return agg.reset_index()


# ─── 2. Transaction features (per client) ────────────────────────────────────
def _tx_features(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transactional behaviour per client.

    Features:
        total_depot         – total deposits
        total_retrait       – total withdrawals
        total_remboursement – total repayment transactions
        total_transfert     – total transfers
        n_transactions      – total number of transactions
        n_suspect           – flagged suspicious transactions
        ratio_retrait_depot – spending ratio (withdrawal / deposit)
        avg_tx_amount       – average transaction amount
    """
    base = tx.groupby("client_id").agg(
        n_transactions  = ("transaction_id", "count"),
        n_suspect       = ("suspect", "sum"),
        avg_tx_amount   = ("montant", "mean"),
    )

    # Per-type pivot
    pivot = (
        tx.pivot_table(index="client_id", columns="type", values="montant",
                       aggfunc="sum", fill_value=0)
        .rename(columns=lambda c: f"total_{c.lower()}")
    )
    # Ensure all 4 columns exist even if missing from data
    for col in ["total_depot", "total_retrait", "total_remboursement", "total_transfert"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    agg = base.join(pivot)
    agg["ratio_retrait_depot"] = (
        agg["total_retrait"] / agg["total_depot"].replace(0, np.nan)
    ).fillna(0)

    return agg.reset_index()


# ─── 3. Relation features (per client) ───────────────────────────────────────
def _rel_features(rel: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate graph edge statistics per source client.

    Features:
        max_risk_relation   – highest risk score on any connected edge
        avg_risk_relation   – average risk on all edges
        n_relations         – total number of relationships
        n_garant            – count of GARANT-type relationships
    """
    agg = rel.groupby("source_client_id").agg(
        max_risk_relation = ("risk_relation", "max"),
        avg_risk_relation = ("risk_relation", "mean"),
        n_relations       = ("relation_id", "count"),
    )
    garant = (
        rel[rel["type_relation"] == "GARANT"]
        .groupby("source_client_id")
        .size()
        .rename("n_garant")
    )
    agg = agg.join(garant, how="left")
    agg["n_garant"] = agg["n_garant"].fillna(0).astype(int)

    return agg.reset_index().rename(columns={"source_client_id": "client_id"})


# ─── 4. Ordinal encoders for categoricals (kyc_score computed separately) ─────
def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns. Uses kyc_score (0-100) instead of statut_kyc.
    """
    cycle_map = {v: i for i, v in enumerate(CAT_CYCLE)}
    objet_map = {v: i for i, v in enumerate(CAT_OBJET)}
    prof_map  = {v: i for i, v in enumerate(CAT_PROFESSION)}

    df = df.copy()
    df["cycle_enc"]      = df["cycle"].map(cycle_map).fillna(-1).astype(int)
    df["objet_enc"]      = df["objet"].map(objet_map).fillna(-1).astype(int)
    df["profession_enc"] = df["profession"].map(prof_map).fillna(-1).astype(int)
    return df


# ─── 5. Master function ───────────────────────────────────────────────────────
def build_features(use_cached: bool = False) -> pd.DataFrame:
    """
    Build the full feature matrix.

    Parameters
    ----------
    use_cached : bool
        If True, load processed parquet files instead of re-running cleaning.

    Returns
    -------
    pd.DataFrame  shape ~ (n_credits, ~25 features) with TARGET_COLUMN as last column
    """
    if use_cached:
        try:
            data = load_processed()
            print("📂 Loaded cached processed data.")
        except FileNotFoundError:
            print("⚠️  Cached data not found — loading raw data.")
            use_cached = False

    if not use_cached:
        raw_data = load_raw_data()
        data = clean_and_save(raw_data)

    clients        = data["clients"]
    credits        = data["credits"]
    remb           = data["remboursements"]
    tx             = data["transactions"]
    rel            = data["relations"]

    # ── Aggregate features from side tables ──────────────────────────────────
    print("Computing remboursement features...")
    remb_feats = _remb_features(remb)

    print("Computing transaction features...")
    tx_feats = _tx_features(tx)

    print("Computing relation features...")
    rel_feats = _rel_features(rel)

    # ── Client columns we keep ───────────────────────────────────────────────
    client_cols = credits[["credit_id", "client_id"]].merge(
        clients[["client_id", "age", "revenu_mensuel", "statut_kyc", "profession"]],
        on="client_id", how="left",
    )

    # ── Merge everything on credit_id / client_id ────────────────────────────
    print("Merging all feature groups...")
    df = credits[[
        "credit_id", "client_id",
        "cycle", "objet", "montant", "duree_mois", "dti",
        TARGET_COLUMN,
    ]].copy()

    df = df.merge(remb_feats, on="credit_id", how="left")
    df = df.merge(tx_feats,   on="client_id", how="left")
    df = df.merge(rel_feats,  on="client_id", how="left")
    df = df.merge(
        client_cols.drop(columns="client_id"),
        on="credit_id", how="left",
    )

    # ── KYC score (estimated, not raw statut_kyc) ─────────────────────────────
    kyc_model = MODELS_DIR / "kyc_scorer.joblib"
    if not kyc_model.exists():
        print("Fitting KYC scorer...")
        fit_kyc_scorer(clients)
    df["kyc_score"] = df.apply(compute_kyc_score_row, axis=1)

    # ── Encode categoricals ───────────────────────────────────────────────────
    df = _encode_categoricals(df)

    # ── Fill remaining NaN (clients with no transactions / relations) ─────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # ── Drop raw categorical string cols + IDs (not needed for ML) ───────────
    df = df.drop(columns=[
        "client_id", "cycle", "objet", "statut_kyc", "profession",
    ])

    # ── Persist ──────────────────────────────────────────────────────────────
    df.to_parquet(FEATURES_FILE, index=False)
    print(f"\nFeature matrix saved at {FEATURES_FILE}")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Target distribution: {df[TARGET_COLUMN].value_counts().to_dict()}")

    return df


if __name__ == "__main__":
    build_features(use_cached=False)
