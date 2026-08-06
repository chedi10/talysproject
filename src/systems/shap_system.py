from __future__ import annotations

"""Standalone SHAP system — moteur XAI interne avec contexte enrichi."""

from typing import Any

import numpy as np
import pandas as pd

from src.xai.shap_explainer import explain_classic_model, _top_contributions, _label, FEATURE_LABELS


def _explain_deep_tabular(
    model: Any,
    model_name: str,
    num_cols: list[str],
    cat_cols: list[str],
    features: dict[str, Any],
) -> tuple[dict[str, float], float]:
    from src.models.deep_tabular.predict import predict_proba

    meta = {"num_cols": num_cols, "cat_cols": cat_cols}
    base = predict_proba(model, meta, features)
    contributions: dict[str, float] = {}

    for col in num_cols:
        perturbed = dict(features)
        orig = float(features[col])
        perturbed[col] = orig * 0.5 if orig != 0 else 0.0
        p = predict_proba(model, meta, perturbed)
        contributions[col] = float(p - base)

    for col in cat_cols:
        perturbed = dict(features)
        perturbed[col] = 0
        p = predict_proba(model, meta, perturbed)
        contributions[col] = float(p - base)

    return contributions, float(base)


def _build_driver_details(
    contributions: dict[str, float],
    features: dict[str, Any],
    portfolio_medians: dict[str, float],
    k: int = 8,
) -> list[dict[str, Any]]:
    items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:k]
    details = []
    for feat, impact in items:
        val = features.get(feat)
        med = portfolio_medians.get(feat)
        vs_portfolio = None
        if val is not None and med is not None and med != 0:
            vs_portfolio = round((float(val) - med) / abs(med) * 100, 1)
        details.append({
            "feature": feat,
            "label": _label(feat),
            "impact": round(float(impact), 5),
            "value": round(float(val), 4) if val is not None else None,
            "portfolio_median": med,
            "vs_portfolio_pct": vs_portfolio,
        })
    return details


def run_shap_system(
    *,
    model: Any,
    model_name: str,
    feature_names: list[str],
    features: dict[str, Any],
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    features_df: pd.DataFrame | None = None,
    credits_df: pd.DataFrame | None = None,
    credit_id: int | None = None,
) -> dict[str, Any]:
    medians: dict[str, float] = {}
    if features_df is not None:
        for k in FEATURE_LABELS:
            if k in features_df.columns:
                medians[k] = round(float(features_df[k].median()), 4)

    base_prediction: float | None = None

    if type(model).__name__ == "DeepTabularNet" and num_cols and cat_cols:
        contributions, base_prediction = _explain_deep_tabular(model, model_name, num_cols, cat_cols, features)
        method = "deep_tabular_ablation"
    else:
        x = np.array([float(features[k]) for k in feature_names], dtype=float)
        raw = explain_classic_model(model, model_name, feature_names, x)
        contributions = {item["feature"]: item["impact"] for item in raw["increases_risk"] + raw["decreases_risk"]}
        # Rebuild full contributions from classic explainer output
        all_feats = {fn: 0.0 for fn in feature_names}
        for item in raw["increases_risk"] + raw["decreases_risk"]:
            all_feats[item["feature"]] = item["impact"]
        contributions = all_feats
        method = raw["method"]
        model_name = raw["model_used"]

    increases, decreases = _top_contributions(contributions)
    driver_details = _build_driver_details(contributions, features, medians)

    credit_context: dict[str, Any] = {}
    if credits_df is not None and credit_id is not None:
        cr = credits_df[credits_df["credit_id"] == credit_id]
        if not cr.empty:
            credit_context = {
                "montant": float(features.get("montant", 0)),
                "duree_mois": int(features.get("duree_mois", 0)),
                "dti": round(float(features.get("dti", 0)), 3),
                "objet": str(cr.iloc[0].get("objet", "")),
                "cycle": str(cr.iloc[0].get("cycle", "")),
            }

    return {
        "method": method,
        "model_used": model_name,
        "base_prediction": base_prediction,
        "increases_risk": increases,
        "decreases_risk": decreases,
        "driver_details": driver_details,
        "credit_context": credit_context,
        "summary": (
            f"Variables qui augmentent le risque: {', '.join(x['label'] for x in increases[:3]) or '—'}. "
            f"Variables qui le diminuent: {', '.join(x['label'] for x in decreases[:3]) or '—'}."
        ),
    }
