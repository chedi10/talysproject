from __future__ import annotations

"""Fusion ensemble — Deep Tabular + GAT + Transformer (pondération AUC + vote)."""

from dataclasses import dataclass
from typing import Any, Literal

RiskLevel = Literal["FAIBLE", "MODERE", "ELEVE"]

# Pondération basée sur l'AUC de validation (métadonnées modèles)
DEFAULT_AUC_WEIGHTS: dict[str, float] = {
    "classic": 1.0,
    "sequential": 0.7583,
    "graph": 0.8173,
}

MODEL_LABELS: dict[str, str] = {
    "classic": "Deep Tabular (MLP + Embeddings)",
    "sequential": "Temporal Transformer",
    "graph": "GAT — Graph Attention",
}


def risk_level_from_proba(proba: float) -> RiskLevel:
    if proba < 0.30:
        return "FAIBLE"
    if proba < 0.60:
        return "MODERE"
    return "ELEVE"


@dataclass
class ModelScoreInput:
    key: str
    name: str
    proba: float | None
    available: bool = True
    error: str | None = None


def compute_ensemble(
    scores: list[ModelScoreInput],
    *,
    weights: dict[str, float] | None = None,
    method: str = "weighted_auc",
) -> dict[str, Any]:
    """Combine les scores disponibles en probabilité pondérée + vote majoritaire."""
    weights = weights or DEFAULT_AUC_WEIGHTS
    available = [s for s in scores if s.available and s.proba is not None]
    if not available:
        raise ValueError("no_models_available")

    total_w = sum(weights.get(s.key, 1 / 3) for s in available)
    weighted_proba = sum(float(s.proba) * weights.get(s.key, 1 / 3) for s in available) / total_w

    vote_default = sum(1 for s in available if float(s.proba) >= 0.5)
    vote_non_default = len(available) - vote_default

    if vote_default == len(available) or vote_non_default == len(available):
        agreement = "unanimous"
    elif max(vote_default, vote_non_default) >= 2:
        agreement = "majority"
    else:
        agreement = "split"

    ensemble_proba = round(weighted_proba, 4)
    risk = risk_level_from_proba(ensemble_proba)

    model_rows: list[dict[str, Any]] = []
    for s in scores:
        w = weights.get(s.key, 1 / 3) if s.available and s.proba is not None else 0.0
        row: dict[str, Any] = {
            "model_key": s.key,
            "model_name": s.name,
            "weight": round(w / total_w, 4) if total_w > 0 and s.available and s.proba is not None else 0.0,
            "available": s.available and s.proba is not None,
            "error": s.error,
        }
        if s.available and s.proba is not None:
            p = round(float(s.proba), 4)
            row.update(
                {
                    "default_proba": p,
                    "risk_level": risk_level_from_proba(p),
                    "prediction": int(p >= 0.5),
                }
            )
        model_rows.append(row)

    return {
        "default_proba": ensemble_proba,
        "risk_level": risk,
        "prediction": int(ensemble_proba >= 0.5),
        "method": method,
        "model_used": "Ensemble — Deep Tabular + GAT + Transformer",
        "models": model_rows,
        "vote_default": vote_default,
        "vote_non_default": vote_non_default,
        "agreement": agreement,
        "models_available": len(available),
        "models_total": len(scores),
    }
