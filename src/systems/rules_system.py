from __future__ import annotations

"""Standalone Business Rules Engine system."""

from typing import Any

from src.business.rules_engine import evaluate_business_rules


def run_rules_system(
    *,
    features: dict[str, Any],
    default_proba: float,
    risk_level: str,
) -> dict[str, Any]:
    return evaluate_business_rules(features, risk_level=risk_level, default_proba=default_proba)
