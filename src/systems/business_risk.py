from __future__ import annotations

"""Score de risque institutionnel — calculé depuis les données métier, sans modèle ML."""

from typing import Any


def compute_institutional_risk(features: dict[str, Any]) -> dict[str, Any]:
    """
    Évalue le risque crédit à partir des données client/crédit en base.
    Indépendant des modèles ML (Classic / Transformer / GAT).
    """
    dti = float(features.get("dti", 0))
    kyc = float(features.get("kyc_score", 70))
    pct_late = float(features.get("pct_late", 0))
    max_retard = float(features.get("max_retard", 0))
    avg_retard = float(features.get("avg_retard", 0))
    n_suspect = int(features.get("n_suspect", 0))
    ratio_rd = float(features.get("ratio_retrait_depot", 0))
    max_rel = float(features.get("max_risk_relation", 0))
    n_garant = int(features.get("n_garant", 0))
    n_en_retard = int(features.get("n_en_retard", 0))

    score = 0.0
    factors: list[str] = []

    if dti > 0.50:
        score += 0.22
        factors.append("DTI élevé")
    elif dti > 0.35:
        score += 0.10
        factors.append("DTI modéré")

    if kyc < 50:
        score += 0.20
        factors.append("KYC faible")
    elif kyc < 65:
        score += 0.08

    if max_retard >= 30:
        score += 0.18
        factors.append("Retards importants")
    elif avg_retard >= 10:
        score += 0.08

    if pct_late >= 0.25:
        score += 0.12
        factors.append("Taux de retard élevé")

    if n_en_retard >= 1:
        score += 0.15
        factors.append("Retards sévères")

    if n_suspect >= 2:
        score += 0.14
        factors.append("Transactions suspectes")

    if ratio_rd > 0.85:
        score += 0.08

    if max_rel >= 60:
        score += 0.10
        factors.append("Réseau à risque")

    if n_garant == 0 and score > 0.35:
        score += 0.05

    score = min(0.98, score)
    proba = round(score, 4)

    if proba < 0.30:
        risk_level = "FAIBLE"
    elif proba < 0.60:
        risk_level = "MODERE"
    else:
        risk_level = "ELEVE"

    return {
        "institutional_score": proba,
        "risk_level": risk_level,
        "risk_factors": factors,
        "summary": f"Score institutionnel {proba:.0%} ({risk_level}) — basé sur KYC, DTI, retards et comportement.",
    }
