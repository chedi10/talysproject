from __future__ import annotations

"""Orchestration des 4 systèmes décisionnels — 100% indépendants des modèles ML."""

from typing import Any

import pandas as pd

from src.business.early_warning import evaluate_early_warnings
from src.business.recommendation import generate_recommendation
from src.business.rules_engine import evaluate_business_rules
from src.systems.business_risk import compute_institutional_risk
from src.systems.context_enrichment import build_client_profile, build_credit_snapshot
from src.systems.shap_system import run_shap_system


def enrich_context(
    ctx: dict[str, Any],
    *,
    clients_df: pd.DataFrame,
    credits_df: pd.DataFrame,
) -> dict[str, Any]:
    profile = build_client_profile(clients_df=clients_df, client_id=ctx["client_id"], cin=ctx["cin"])
    snapshot = build_credit_snapshot(ctx["features"], credits_df=credits_df, credit_id=ctx["credit_id"])
    return {**ctx, "client_profile": profile, "credit_snapshot": snapshot}


def run_rules_standalone(ctx: dict[str, Any]) -> dict[str, Any]:
    risk = compute_institutional_risk(ctx["features"])
    result = evaluate_business_rules(
        ctx["features"],
        risk_level=risk["risk_level"],
        default_proba=risk["institutional_score"],
    )
    result["credit_snapshot"] = ctx.get("credit_snapshot", {})
    return result


def run_ews_standalone(
    ctx: dict[str, Any],
    features_df: pd.DataFrame,
    credits_df: pd.DataFrame,
    remb_df: pd.DataFrame | None = None,
    tx_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    risk = compute_institutional_risk(ctx["features"])
    return evaluate_early_warnings(
        client_id=ctx["client_id"],
        current_credit_id=ctx["credit_id"],
        current_proba=risk["institutional_score"],
        current_risk=risk["risk_level"],
        features_df=features_df,
        credits_df=credits_df,
        remb_df=remb_df,
        tx_df=tx_df,
    )


def run_recommendation_standalone(
    ctx: dict[str, Any],
    features_df: pd.DataFrame,
    credits_df: pd.DataFrame,
    remb_df: pd.DataFrame | None = None,
    tx_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    risk = compute_institutional_risk(ctx["features"])
    rules = run_rules_standalone(ctx)
    ews = run_ews_standalone(ctx, features_df, credits_df, remb_df, tx_df)
    shap_summary = None
    if risk.get("risk_factors"):
        shap_summary = f"Facteurs dominants: {', '.join(risk['risk_factors'][:3])}."
    return generate_recommendation(
        default_proba=risk["institutional_score"],
        risk_level=risk["risk_level"],
        kyc_score=ctx["kyc_score"],
        features=ctx["features"],
        business_rules=rules,
        early_warning=ews,
        shap_summary=shap_summary,
    )


def run_shap_standalone(
    ctx: dict[str, Any],
    model: Any,
    model_name: str,
    meta: dict,
    features_df: pd.DataFrame | None = None,
    credits_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    return run_shap_system(
        model=model,
        model_name=model_name,
        feature_names=ctx["feature_order"],
        features=ctx["features"],
        num_cols=meta.get("num_cols"),
        cat_cols=meta.get("cat_cols"),
        features_df=features_df,
        credits_df=credits_df,
        credit_id=ctx["credit_id"],
    )
