from __future__ import annotations

"""Enrichissement contextuel partagé par les systèmes décisionnels."""

from typing import Any

import pandas as pd


def build_client_profile(*, clients_df: pd.DataFrame, client_id: int, cin: str) -> dict[str, Any]:
    row = clients_df[clients_df["client_id"] == client_id]
    if row.empty:
        return {"cin": cin, "client_id": client_id}
    r = row.iloc[0]
    return {
        "cin": cin,
        "client_id": client_id,
        "nom": str(r.get("nom", "")),
        "prenom": str(r.get("prenom", "")),
        "age": int(r.get("age", 0)),
        "ville": str(r.get("ville", "")),
        "profession": str(r.get("profession", "")),
        "revenu_mensuel": float(r.get("revenu_mensuel", 0)),
        "statut_kyc": str(r.get("statut_kyc", "")),
    }


def build_credit_snapshot(features: dict[str, Any], *, credits_df: pd.DataFrame | None = None, credit_id: int | None = None) -> dict[str, Any]:
    snap = {
        "montant": float(features.get("montant", 0)),
        "duree_mois": int(features.get("duree_mois", 0)),
        "dti": round(float(features.get("dti", 0)), 3),
        "revenu_mensuel": float(features.get("revenu_mensuel", 0)),
        "cycle": int(features.get("cycle_enc", 0)),
        "n_garant": int(features.get("n_garant", 0)),
    }
    if credits_df is not None and credit_id is not None:
        cr = credits_df[credits_df["credit_id"] == credit_id]
        if not cr.empty:
            snap["objet"] = str(cr.iloc[0].get("objet", ""))
            snap["date_debut"] = str(cr.iloc[0].get("date_debut", ""))[:10]
    return snap


def portfolio_medians(features_df: pd.DataFrame, keys: list[str] | None = None) -> dict[str, float]:
    keys = keys or ["dti", "kyc_score", "avg_retard", "max_retard", "pct_late", "montant", "n_suspect"]
    medians: dict[str, float] = {}
    for k in keys:
        if k in features_df.columns:
            medians[k] = round(float(features_df[k].median()), 4)
    return medians


def build_trend_series(
    *,
    client_id: int,
    current_credit_id: int,
    features_df: pd.DataFrame,
    credits_df: pd.DataFrame,
    metrics: list[str] | None = None,
    max_points: int = 5,
) -> list[dict[str, Any]]:
    metrics = metrics or ["dti", "avg_retard", "pct_late", "kyc_score"]
    client_credits = credits_df[credits_df["client_id"] == client_id].copy()
    if "date_debut" in client_credits.columns:
        client_credits = client_credits.sort_values("date_debut")
    credit_ids = client_credits["credit_id"].astype(int).tolist()[-max_points:]

    series: list[dict[str, Any]] = []
    for metric in metrics:
        points = []
        for cid in credit_ids:
            row = features_df[features_df["credit_id"] == cid]
            if row.empty or metric not in row.columns:
                continue
            cr = client_credits[client_credits["credit_id"] == cid]
            points.append({
                "credit_id": int(cid),
                "date": str(cr.iloc[0].get("date_debut", ""))[:10] if not cr.empty else "",
                "value": round(float(row.iloc[0][metric]), 4),
                "is_current": cid == current_credit_id,
            })
        if points:
            series.append({"metric": metric, "label": _metric_label(metric), "points": points})
    return series


def _metric_label(metric: str) -> str:
    labels = {
        "dti": "DTI",
        "avg_retard": "Retard moyen (j)",
        "pct_late": "Taux de retard",
        "kyc_score": "Score KYC",
        "n_suspect": "Transactions suspectes",
        "ratio_retrait_depot": "Ratio retrait/dépôt",
    }
    return labels.get(metric, metric)
