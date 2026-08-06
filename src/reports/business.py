from __future__ import annotations

"""Règles métier avancées pour rapports et réponses agent."""

from typing import Any


def _worst_risk(*levels: str | None) -> str:
    order = {"ELEVE": 3, "MODERE": 2, "FAIBLE": 1}
    best = "FAIBLE"
    score = 0
    for lv in levels:
        if not lv:
            continue
        s = order.get(str(lv).upper(), 0)
        if s > score:
            score = s
            best = str(lv).upper()
    return best


def build_keywords(
    *,
    cin: str,
    classic: dict[str, Any],
    sequential: dict[str, Any],
    graph: dict[str, Any],
) -> list[str]:
    kyc = classic.get("kyc_score") or sequential.get("kyc_score") or graph.get("kyc_score")
    worst = _worst_risk(classic.get("risk_level"), sequential.get("risk_level"), graph.get("risk_level"))
    keys = [
        "microfinance",
        "scoring credit",
        f"CIN {cin}",
        f"risque {worst}",
        "KYC",
        "probabilite defaut",
    ]
    if kyc is not None:
        keys.append(f"score KYC {kyc}")
    if sequential.get("n_credits", 0) and int(sequential.get("n_credits", 0)) > 1:
        keys.append("multi-credits")
    if graph.get("model_used"):
        keys.append("GraphSAGE")
    if sequential.get("model_used"):
        keys.append("LSTM/GRU")
    return keys[:12]


def business_recommendation(
    *,
    worst_risk: str,
    kyc_score: float | None,
    default_proba: float | None,
) -> dict[str, str]:
    risk = str(worst_risk or "MODERE").upper()
    kyc = float(kyc_score or 0)
    proba = float(default_proba or 0)

    if risk == "ELEVE" or proba >= 0.6:
        return {
            "decision": "Revue comité crédit obligatoire",
            "actions": "Renforcer garanties, plafonner l'exposition, plan de recouvrement.",
            "surveillance": "Hebdomadaire pendant 90 jours.",
            "kyc_note": "Revalidation KYC immédiate." if kyc < 50 else "Contrôle KYC renforcé.",
        }
    if risk == "MODERE" or proba >= 0.3:
        return {
            "decision": "Accord sous conditions",
            "actions": "Suivi retards, limiter nouveau décaissement, clause de revue à 3 mois.",
            "surveillance": "Mensuelle.",
            "kyc_note": "Vérifier pièces KYC si score < 60." if kyc < 60 else "KYC acceptable sous surveillance.",
        }
    return {
        "decision": "Profil acceptable — validation standard",
        "actions": "Surveillance classique, pas de dérogation requise.",
        "surveillance": "Trimestrielle.",
        "kyc_note": "KYC conforme." if kyc >= 60 else "Compléter dossier KYC malgré risque faible.",
    }


def _risk_label_fr(level: str | None) -> str:
    mapping = {"FAIBLE": "Faible", "MODERE": "Modéré", "ELEVE": "Élevé"}
    return mapping.get(str(level or "").upper(), str(level or "—"))


def build_executive_summary(
    *,
    cin: str,
    worst_risk: str,
    kyc_score: float | None,
    default_proba: float | None,
    models_agree: bool,
    profile: dict[str, Any] | None = None,
    systems: dict[str, Any] | None = None,
) -> str:
    """Résumé exécutif rédigé pour comité crédit (PDF / rapport)."""
    rec = business_recommendation(worst_risk=worst_risk, kyc_score=kyc_score, default_proba=default_proba)
    risk_fr = _risk_label_fr(worst_risk)
    proba_pct = f"{float(default_proba or 0) * 100:.1f} %" if default_proba is not None else "—"
    kyc = float(kyc_score) if kyc_score is not None else None

    client = ""
    if profile and profile.get("nom"):
        client = f"{profile.get('prenom', '')} {profile.get('nom', '')}".strip()
        if profile.get("ville"):
            client += f", {profile['ville']}"

    intro = (
        f"Le présent dossier synthétise l'analyse du client CIN {cin}"
        + (f" ({client})" if client else "")
        + f". Le profil présente un niveau de risque crédit **{risk_fr}**, "
        f"avec une probabilité de défaut estimée à **{proba_pct}** par les modèles prédictifs."
    )

    if kyc is not None:
        kyc_qual = "solide" if kyc >= 70 else "à surveiller" if kyc >= 50 else "fragile"
        intro += f" Le score KYC ({kyc:.0f}/100) est jugé **{kyc_qual}**."

    if models_agree:
        model_para = (
            "Les trois approches analytiques (tabular, séquentiel et graphe) convergent, "
            "renforçant la fiabilité de la lecture globale du dossier."
        )
    else:
        model_para = (
            "Les modèles prédictifs présentent des écarts de lecture. "
            "Il est recommandé d'arbitrer en comité en s'appuyant sur l'historique comportemental "
            "et les signaux réseau, au-delà du score tabulaire seul."
        )

    inst_para = ""
    if systems and systems.get("institutional_score") is not None:
        inst_score = float(systems["institutional_score"])
        inst_risk = _risk_label_fr(systems.get("risk_level"))
        compliance = systems.get("rules", {}).get("compliance_score", "—")
        watchlist = systems.get("early_warning", {}).get("watchlist_priority", "NONE")
        inst_para = (
            f" Les systèmes décisionnels institutionnels confirment un score de risque à **{inst_score:.1%}** "
            f"(niveau {inst_risk}), une conformité réglementaire de **{compliance}/100** "
            f"et une priorité watchlist **{watchlist}**."
        )

    decision_para = (
        f"**Recommandation opérationnelle :** {rec['decision']}. "
        f"{rec['actions']} Surveillance proposée : {rec['surveillance']}. {rec['kyc_note']}"
    )

    return " ".join(p for p in (intro, model_para + inst_para, decision_para) if p)


def build_conclusion(
    *,
    cin: str,
    worst_risk: str,
    kyc_score: float | None,
    default_proba: float | None,
    models_agree: bool,
) -> str:
    rec = business_recommendation(worst_risk=worst_risk, kyc_score=kyc_score, default_proba=default_proba)
    agree = (
        "Les trois modèles convergent vers un même ordre de grandeur de risque."
        if models_agree
        else "Les modèles divergent : privilégier la vue séquentielle (comportement) et graphe (réseau) en comité."
    )
    return (
        f"Pour le client CIN {cin}, la synthèse métier retient un niveau {_risk_label_fr(worst_risk)} "
        f"(probabilité de défaut {default_proba}). {agree} "
        f"Décision proposée : {rec['decision']}. "
        f"{rec['kyc_note']} Décision finale : analyste crédit / comité (hors automatisation IA)."
    )


def models_agreement(classic: dict, sequential: dict, graph: dict) -> bool:
    levels = {str(classic.get("risk_level", "")).upper(), str(sequential.get("risk_level", "")).upper(), str(graph.get("risk_level", "")).upper()}
    levels.discard("")
    return len(levels) <= 1


def summarize_model_line(label: str, obj: dict[str, Any]) -> str:
    if not obj:
        return f"- {label} : non disponible"
    return (
        f"- **{label}** : risque `{obj.get('risk_level', '—')}` | "
        f"proba `{obj.get('default_proba', '—')}` | KYC `{obj.get('kyc_score', '—')}` | "
        f"modèle `{obj.get('model_used', '—')}`"
    )
