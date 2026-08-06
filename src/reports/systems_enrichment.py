from __future__ import annotations

"""Enrichissement rapports/chat avec les systèmes décisionnels institutionnels."""

from typing import Any

from src.systems.orchestrator import (
    run_ews_standalone,
    run_recommendation_standalone,
    run_rules_standalone,
)


def fetch_systems_from_ctx(
    ctx: dict[str, Any],
    features_df,
    credits_df,
    remb_df=None,
    tx_df=None,
) -> dict[str, Any]:
    """Agrège rules + EWS + recommendation à partir d'un contexte client déjà résolu."""
    rules = run_rules_standalone(ctx)
    ews = run_ews_standalone(ctx, features_df, credits_df, remb_df, tx_df)
    reco = run_recommendation_standalone(ctx, features_df, credits_df, remb_df, tx_df)

    triggered_rules = [r for r in rules.get("rules", []) if r.get("triggered")]

    return {
        "institutional_score": ctx.get("institutional_score"),
        "risk_level": ctx.get("risk_level"),
        "risk_factors": ctx.get("risk_factors", []),
        "client_profile": ctx.get("client_profile", {}),
        "credit_snapshot": ctx.get("credit_snapshot", {}),
        "rules": {
            "summary": rules.get("summary"),
            "triggered_count": rules.get("triggered_count", 0),
            "compliance_score": rules.get("compliance_score", 100),
            "highest_severity": rules.get("highest_severity"),
            "triggered": [
                {"rule_id": r["rule_id"], "name": r["name"], "severity": r["severity"], "message": r["message"]}
                for r in triggered_rules[:8]
            ],
        },
        "early_warning": {
            "summary": ews.get("summary"),
            "alert_count": ews.get("alert_count", 0),
            "critical_count": ews.get("critical_count", 0),
            "watchlist_priority": ews.get("watchlist_priority", "NONE"),
            "degradation_detected": ews.get("degradation_detected", False),
            "alerts": ews.get("alerts", [])[:6],
        },
        "recommendation": {
            "decision": reco.get("decision"),
            "decision_label": reco.get("decision_label"),
            "confidence": reco.get("confidence"),
            "justification": reco.get("justification", "")[:500],
            "suggested_montant": reco.get("suggested_montant"),
            "monitoring_frequency": reco.get("monitoring_frequency"),
            "conditions": reco.get("conditions", []),
            "recommended_actions": reco.get("recommended_actions", [])[:5],
        },
    }
