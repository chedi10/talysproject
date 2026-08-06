from __future__ import annotations

"""Standalone AI Recommendation system — orchestrates rules + EWS as separate inputs."""

from typing import Any

from src.business.recommendation import generate_recommendation
from src.systems.early_warning_system import run_early_warning_system
from src.systems.rules_system import run_rules_system


def run_recommendation_system(
    *,
    features: dict[str, Any],
    default_proba: float,
    risk_level: str,
    kyc_score: float,
    client_id: int,
    credit_id: int,
    features_df,
    credits_df,
    shap_summary: str | None = None,
) -> dict[str, Any]:
    rules = run_rules_system(features=features, default_proba=default_proba, risk_level=risk_level)
    ews = run_early_warning_system(
        client_id=client_id,
        credit_id=credit_id,
        default_proba=default_proba,
        risk_level=risk_level,
        features_df=features_df,
        credits_df=credits_df,
    )
    return generate_recommendation(
        default_proba=default_proba,
        risk_level=risk_level,
        kyc_score=kyc_score,
        features=features,
        business_rules=rules,
        early_warning=ews,
        shap_summary=shap_summary,
    )
