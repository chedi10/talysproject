from __future__ import annotations

"""SHAP / coefficient-based feature contributions for classic model."""

from typing import Any

import numpy as np

# Human-readable French labels for features
FEATURE_LABELS: dict[str, str] = {
    "montant": "Montant du crédit",
    "duree_mois": "Durée (mois)",
    "dti": "Ratio d'endettement (DTI)",
    "cycle_enc": "Cycle crédit",
    "objet_enc": "Objet du crédit",
    "age": "Âge client",
    "revenu_mensuel": "Revenu mensuel",
    "profession_enc": "Profession",
    "kyc_score": "Score KYC",
    "avg_retard": "Retard moyen (jours)",
    "max_retard": "Retard maximum (jours)",
    "std_retard": "Volatilité des retards",
    "n_payments": "Nombre d'échéances",
    "n_late": "Paiements en retard",
    "pct_late": "Taux de retard",
    "n_en_retard": "Retards sévères (≥90j)",
    "n_transactions": "Nombre transactions",
    "n_suspect": "Transactions suspectes",
    "avg_tx_amount": "Montant moyen transaction",
    "total_depot": "Total dépôts",
    "total_retrait": "Total retraits",
    "total_remboursement": "Total remboursements",
    "total_transfert": "Total transferts",
    "ratio_retrait_depot": "Ratio retrait/dépôt",
    "max_risk_relation": "Risque relation max",
    "avg_risk_relation": "Risque relation moyen",
    "n_relations": "Nombre relations",
    "n_garant": "Nombre garants",
}


def _label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def _top_contributions(values: dict[str, float], k: int = 6) -> tuple[list[dict], list[dict]]:
    items = [{"feature": fn, "label": _label(fn), "impact": round(float(v), 5)} for fn, v in values.items()]
    increases = sorted([x for x in items if x["impact"] > 0], key=lambda x: -x["impact"])[:k]
    decreases = sorted([x for x in items if x["impact"] < 0], key=lambda x: x["impact"])[:k]
    return increases, decreases


def explain_classic_model(
    model: Any,
    model_name: str,
    feature_names: list[str],
    x_row: np.ndarray,
) -> dict[str, Any]:
    """
    Return SHAP-like contributions for a single prediction.
    Positive impact = increases default risk probability.
    """
    x = np.asarray(x_row, dtype=float).reshape(1, -1)
    method = "coefficient_approx"

    try:
        import shap  # noqa: F401
        has_shap = True
    except ImportError:
        has_shap = False

    contributions: dict[str, float] = {}

    # Pipeline (Logistic Regression + scaler)
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        scaler = model.named_steps.get("scaler")
        clf = model.named_steps["clf"]
        x_in = scaler.transform(x) if scaler is not None else x
        if hasattr(clf, "coef_"):
            coef = clf.coef_.ravel()
            for i, name in enumerate(feature_names):
                contributions[name] = float(coef[i] * x_in[0, i])
            method = "logistic_coefficients"
        elif has_shap:
            try:
                explainer = shap.Explainer(clf, x_in)
                sv = explainer(x_in)
                vals = sv.values[0] if hasattr(sv, "values") else sv[0]
                for i, name in enumerate(feature_names):
                    contributions[name] = float(vals[i])
                method = "shap_linear"
            except Exception:
                pass

    # Tree models (RF, XGBoost) — raw features
    elif hasattr(model, "feature_importances_") or "XGB" in model_name or "Forest" in model_name:
        if has_shap:
            try:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(x)
                if isinstance(sv, list):
                    vals = sv[1][0] if len(sv) > 1 else sv[0][0]
                else:
                    vals = sv[0]
                for i, name in enumerate(feature_names):
                    contributions[name] = float(vals[i])
                method = "shap_tree"
            except Exception:
                pass
        if not contributions:
            # Fallback: feature_importances * normalized feature value
            imp = getattr(model, "feature_importances_", np.ones(len(feature_names)) / len(feature_names))
            x_norm = x[0] / (np.abs(x[0]).max() + 1e-9)
            for i, name in enumerate(feature_names):
                contributions[name] = float(imp[i] * x_norm[i])
            method = "importance_proxy"

    if not contributions:
        for i, name in enumerate(feature_names):
            contributions[name] = 0.0
        method = "unavailable"

    increases, decreases = _top_contributions(contributions)
    return {
        "method": method,
        "model_used": model_name,
        "increases_risk": increases,
        "decreases_risk": decreases,
        "summary": (
            f"Variables qui augmentent le risque: {', '.join(x['label'] for x in increases[:3]) or '—'}. "
            f"Variables qui le diminuent: {', '.join(x['label'] for x in decreases[:3]) or '—'}."
        ),
    }
