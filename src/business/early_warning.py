from __future__ import annotations

"""Early Warning System — détection proactive de dégradation."""

from typing import Any, Literal

import pandas as pd

from src.systems.context_enrichment import build_trend_series

Severity = Literal["INFO", "WARNING", "CRITICAL"]


def _risk_rank(level: str) -> int:
    return {"FAIBLE": 1, "MODERE": 2, "ELEVE": 3}.get(str(level).upper(), 0)


def _proba_to_risk(proba: float) -> str:
    if proba < 0.30:
        return "FAIBLE"
    if proba < 0.60:
        return "MODERE"
    return "ELEVE"


def _watchlist_priority(critical_count: int, alert_count: int) -> str:
    if critical_count >= 2:
        return "HIGH"
    if critical_count >= 1 or alert_count >= 3:
        return "MEDIUM"
    if alert_count >= 1:
        return "LOW"
    return "NONE"


def evaluate_early_warnings(
    *,
    client_id: int,
    current_credit_id: int,
    current_proba: float,
    current_risk: str,
    features_df: pd.DataFrame,
    credits_df: pd.DataFrame,
    remb_df: pd.DataFrame | None = None,
    tx_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    alerts: list[dict[str, Any]] = []

    client_credits = credits_df[credits_df["client_id"] == client_id].copy()
    if "date_debut" in client_credits.columns:
        client_credits = client_credits.sort_values("date_debut")

    credit_ids = client_credits["credit_id"].astype(int).tolist()
    hist_ids = [cid for cid in credit_ids if cid != current_credit_id]

    hist_features = features_df[features_df["credit_id"].isin(hist_ids)] if hist_ids else pd.DataFrame()
    current_row = features_df[features_df["credit_id"] == current_credit_id]
    if current_row.empty:
        return {
            "alerts": [], "degradation_detected": False,
            "summary": "Données insuffisantes.", "alert_count": 0,
            "critical_count": 0, "watchlist_priority": "NONE",
            "trend_series": [], "n_credits_historique": 0,
        }

    cur = current_row.iloc[0]
    cur_retard = float(cur.get("avg_retard", 0))
    cur_max_retard = float(cur.get("max_retard", 0))
    cur_suspect = int(cur.get("n_suspect", 0))
    cur_pct_late = float(cur.get("pct_late", 0))
    cur_dti = float(cur.get("dti", 0))
    cur_kyc = float(cur.get("kyc_score", 0))
    cur_ratio_rd = float(cur.get("ratio_retrait_depot", 0))

    if not hist_features.empty:
        hist_avg_retard = float(hist_features["avg_retard"].mean())
        hist_max_retard = float(hist_features["max_retard"].max())
        hist_pct_late = float(hist_features["pct_late"].mean())
        hist_dti = float(hist_features["dti"].mean()) if "dti" in hist_features.columns else 0
        hist_kyc = float(hist_features["kyc_score"].mean()) if "kyc_score" in hist_features.columns else 0
        hist_ratio_rd = float(hist_features["ratio_retrait_depot"].mean()) if "ratio_retrait_depot" in hist_features.columns else 0

        if cur_retard > hist_avg_retard + 5:
            alerts.append({
                "code": "retard_croissant",
                "severity": "WARNING" if cur_retard < hist_avg_retard + 15 else "CRITICAL",
                "message": f"Retard moyen en hausse: {cur_retard:.1f}j vs historique {hist_avg_retard:.1f}j.",
                "metric": "avg_retard", "current": cur_retard, "baseline": hist_avg_retard,
            })

        if cur_max_retard > hist_max_retard + 10:
            alerts.append({
                "code": "max_retard_degrade",
                "severity": "WARNING",
                "message": f"Retard max dégradé: {cur_max_retard:.0f}j vs historique {hist_max_retard:.0f}j.",
                "metric": "max_retard", "current": cur_max_retard, "baseline": hist_max_retard,
            })

        if cur_pct_late > hist_pct_late + 0.10:
            alerts.append({
                "code": "pct_late_hausse",
                "severity": "WARNING",
                "message": f"Taux de retard en hausse: {cur_pct_late:.0%} vs {hist_pct_late:.0%}.",
                "metric": "pct_late", "current": cur_pct_late, "baseline": hist_pct_late,
            })

        if cur_dti > hist_dti + 0.08:
            alerts.append({
                "code": "dti_degrade",
                "severity": "WARNING" if cur_dti < 0.55 else "CRITICAL",
                "message": f"DTI en hausse: {cur_dti:.0%} vs historique {hist_dti:.0%}.",
                "metric": "dti", "current": round(cur_dti, 3), "baseline": round(hist_dti, 3),
            })

        if cur_kyc < hist_kyc - 10:
            alerts.append({
                "code": "kyc_degrade",
                "severity": "WARNING",
                "message": f"Score KYC en baisse: {cur_kyc:.0f} vs historique {hist_kyc:.0f}.",
                "metric": "kyc_score", "current": cur_kyc, "baseline": hist_kyc,
            })

        if cur_ratio_rd > hist_ratio_rd + 0.15:
            alerts.append({
                "code": "ratio_rd_degrade",
                "severity": "WARNING",
                "message": f"Ratio retrait/dépôt dégradé: {cur_ratio_rd:.0%} vs {hist_ratio_rd:.0%}.",
                "metric": "ratio_retrait_depot", "current": round(cur_ratio_rd, 3), "baseline": round(hist_ratio_rd, 3),
            })

        if "en_defaut" in hist_features.columns:
            past_default_rate = float(hist_features["en_defaut"].mean())
            if past_default_rate >= 0.5 and current_proba >= 0.40:
                alerts.append({
                    "code": "historique_defaut",
                    "severity": "CRITICAL",
                    "message": "Historique de défaut sur crédits antérieurs + risque actuel élevé.",
                    "metric": "en_defaut_rate", "current": current_proba, "baseline": past_default_rate,
                })

    if len(credit_ids) >= 2 and hist_ids:
        last_hist_id = hist_ids[-1]
        last_row = features_df[features_df["credit_id"] == last_hist_id]
        if not last_row.empty:
            prev_proba = min(0.95, float(last_row.iloc[0].get("pct_late", 0)) * 0.8 + float(last_row.iloc[0].get("avg_retard", 0)) / 100)
            prev_risk = _proba_to_risk(prev_proba)
            if _risk_rank(current_risk) > _risk_rank(prev_risk):
                alerts.append({
                    "code": "passage_risque_superieur",
                    "severity": "CRITICAL",
                    "message": f"Passage {prev_risk} → {current_risk} vs crédit précédent.",
                    "metric": "risk_level", "current": current_risk, "baseline": prev_risk,
                })
            if current_proba > prev_proba + 0.15:
                alerts.append({
                    "code": "score_hausse",
                    "severity": "WARNING",
                    "message": f"Score institutionnel en hausse (+{(current_proba - prev_proba):.0%}).",
                    "metric": "institutional_score", "current": round(current_proba, 4), "baseline": round(prev_proba, 4),
                })

    if cur_suspect >= 1:
        alerts.append({
            "code": "transactions_suspectes",
            "severity": "WARNING" if cur_suspect < 3 else "CRITICAL",
            "message": f"{cur_suspect} transaction(s) suspecte(s) sur la période.",
            "metric": "n_suspect", "current": cur_suspect,
        })

    # Remboursements récents sévères
    if remb_df is not None and not remb_df.empty:
        client_remb = remb_df[remb_df["client_id"] == client_id]
        severe = client_remb[client_remb["retard_jours"] >= 90] if "retard_jours" in client_remb.columns else pd.DataFrame()
        if len(severe) >= 1:
            alerts.append({
                "code": "remb_severe_recent",
                "severity": "CRITICAL",
                "message": f"{len(severe)} échéance(s) en retard sévère (≥90j) détectée(s).",
                "metric": "n_severe_late", "current": len(severe),
            })

    # Hausse transactions suspectes vs historique
    if tx_df is not None and not tx_df.empty and not hist_features.empty:
        hist_suspect = float(hist_features["n_suspect"].mean()) if "n_suspect" in hist_features.columns else 0
        if cur_suspect > hist_suspect + 1:
            alerts.append({
                "code": "tx_suspect_hausse",
                "severity": "WARNING",
                "message": f"Transactions suspectes en hausse: {cur_suspect} vs moyenne {hist_suspect:.1f}.",
                "metric": "n_suspect", "current": cur_suspect, "baseline": round(hist_suspect, 1),
            })

    if current_risk == "ELEVE":
        alerts.append({
            "code": "niveau_risque_eleve",
            "severity": "CRITICAL",
            "message": "Niveau de risque ELEVE — alerte précoce activée.",
            "metric": "risk_level", "current": current_risk,
        })

    degradation = any(a["severity"] in ("WARNING", "CRITICAL") for a in alerts)
    critical_count = sum(1 for a in alerts if a["severity"] == "CRITICAL")
    trend_series = build_trend_series(
        client_id=client_id,
        current_credit_id=current_credit_id,
        features_df=features_df,
        credits_df=credits_df,
    )

    return {
        "alerts": alerts,
        "alert_count": len(alerts),
        "critical_count": critical_count,
        "degradation_detected": degradation,
        "watchlist_priority": _watchlist_priority(critical_count, len(alerts)),
        "trend_series": trend_series,
        "n_credits_historique": len(hist_ids),
        "summary": (
            f"{len(alerts)} signal(aux) d'alerte précoce"
            + (f" — priorité watchlist {_watchlist_priority(critical_count, len(alerts))}." if alerts else ".")
            + (" Dégradation détectée." if degradation else " Pas de dégradation majeure.")
        ),
    }
