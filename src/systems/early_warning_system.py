from __future__ import annotations

"""Standalone Early Warning System."""

from typing import Any

import pandas as pd

from src.business.early_warning import evaluate_early_warnings


def run_early_warning_system(
    *,
    client_id: int,
    credit_id: int,
    default_proba: float,
    risk_level: str,
    features_df: pd.DataFrame,
    credits_df: pd.DataFrame,
) -> dict[str, Any]:
    return evaluate_early_warnings(
        client_id=client_id,
        current_credit_id=credit_id,
        current_proba=default_proba,
        current_risk=risk_level,
        features_df=features_df,
        credits_df=credits_df,
    )
