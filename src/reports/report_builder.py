from __future__ import annotations

"""Génération structurée de rapports Markdown (ML + systèmes institutionnels + RAG)."""

from datetime import datetime, timezone
from typing import Any

from src.reports.business import (
    build_conclusion,
    build_executive_summary,
    build_keywords,
    business_recommendation,
    models_agreement,
    summarize_model_line,
    _worst_risk,
)
from src.rag.context import format_rag_references_section


def build_structured_report(
    *,
    cin: str,
    classic: dict[str, Any],
    sequential: dict[str, Any],
    graph: dict[str, Any],
    sources: list[dict[str, Any]],
    analyst_username: str | None = None,
    systems: dict[str, Any] | None = None,
) -> dict[str, Any]:
    worst = _worst_risk(classic.get("risk_level"), sequential.get("risk_level"), graph.get("risk_level"))
    proba = max(
        float(classic.get("default_proba") or 0),
        float(sequential.get("default_proba") or 0),
        float(graph.get("default_proba") or 0),
    )
    kyc = classic.get("kyc_score") or sequential.get("kyc_score") or graph.get("kyc_score")
    keywords = build_keywords(cin=cin, classic=classic, sequential=sequential, graph=graph)
    if systems:
        keywords.extend(["score institutionnel", "conformite", "early warning", "recommandation IA"])
        keywords = keywords[:14]

    rec = business_recommendation(worst_risk=worst, kyc_score=float(kyc) if kyc is not None else None, default_proba=proba)
    if systems and systems.get("recommendation"):
        inst_reco = systems["recommendation"]
        rec = {
            "decision": inst_reco.get("decision_label") or inst_reco.get("decision", rec["decision"]),
            "actions": "; ".join(inst_reco.get("recommended_actions", [])[:3]) or rec["actions"],
            "surveillance": inst_reco.get("monitoring_frequency", rec["surveillance"]),
            "kyc_note": rec["kyc_note"],
        }

    models_agree = models_agreement(classic, sequential, graph)
    conclusion = build_conclusion(
        cin=cin,
        worst_risk=worst,
        kyc_score=float(kyc) if kyc is not None else None,
        default_proba=round(proba, 4),
        models_agree=models_agree,
    )
    executive_summary = build_executive_summary(
        cin=cin,
        worst_risk=worst,
        kyc_score=float(kyc) if kyc is not None else None,
        default_proba=round(proba, 4),
        models_agree=models_agree,
        profile=(systems or {}).get("client_profile"),
        systems=systems,
    )
    if systems:
        inst_score = systems.get("institutional_score")
        inst_risk = systems.get("risk_level")
        if inst_score is not None:
            conclusion += (
                f" Score institutionnel (systèmes autonomes) : {float(inst_score):.1%} ({inst_risk})."
            )

    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    profile = (systems or {}).get("client_profile") or {}

    return {
        "cin": cin,
        "generated_at": now,
        "analyst": analyst_username or "—",
        "kyc_score": kyc,
        "worst_risk": worst,
        "worst_proba": round(proba, 4),
        "keywords": keywords,
        "classic": classic,
        "sequential": sequential,
        "graph": graph,
        "systems": systems or {},
        "client_profile": profile,
        "recommendation": rec,
        "executive_summary": executive_summary,
        "conclusion": conclusion,
        "sources": sources,
        "models_agree": models_agree,
    }


def structured_to_markdown(data: dict[str, Any]) -> str:
    rec = data["recommendation"]
    kw = ", ".join(f"`{k}`" for k in data["keywords"])
    refs = format_rag_references_section(data.get("sources") or [])
    systems = data.get("systems") or {}
    profile = data.get("client_profile") or {}

    lines = [
        f"# Rapport d'analyse crédit — CIN {data['cin']}",
        "",
        f"**Date** : {data['generated_at']}  ",
        f"**Analyste** : {data['analyst']}  ",
    ]
    if profile.get("nom"):
        lines.append(f"**Client** : {profile.get('prenom', '')} {profile.get('nom', '')} — {profile.get('ville', '')}  ")
    lines.extend([
        f"**Synthèse ML** : risque **{data['worst_risk']}** | proba max **{data['worst_proba']}** | KYC **{data.get('kyc_score', '—')}**",
    ])
    if systems.get("institutional_score") is not None:
        lines.append(
            f"**Synthèse institutionnelle** : score **{float(systems['institutional_score']):.1%}** | "
            f"risque **{systems.get('risk_level', '—')}** | conformité **{systems.get('rules', {}).get('compliance_score', '—')}/100**"
        )
    exec_sum = data.get("executive_summary") or data["conclusion"]
    lines.extend(["", "## Mots-clés", kw, "", "## Résumé exécutif", exec_sum, ""])

    if systems:
        lines.extend(["## Systèmes décisionnels institutionnels", ""])
        rules = systems.get("rules") or {}
        ews = systems.get("early_warning") or {}
        inst_reco = systems.get("recommendation") or {}
        lines.extend([
            f"### Business Rules — conformité {rules.get('compliance_score', '—')}/100",
            rules.get("summary", "—"),
            "",
            f"### Early Warning — watchlist **{ews.get('watchlist_priority', 'NONE')}**",
            ews.get("summary", "—"),
            "",
            f"### Recommandation IA — **{inst_reco.get('decision_label', '—')}**",
            inst_reco.get("justification", "—")[:400],
            "",
        ])
        if inst_reco.get("recommended_actions"):
            lines.append("**Actions** :")
            lines.extend(f"- {a}" for a in inst_reco["recommended_actions"])
            lines.append("")

    lines.extend([
        "## Résultats par modèle ML",
        summarize_model_line("Deep Tabular (classique)", data.get("classic") or {}),
        summarize_model_line("Temporal Transformer (séquentiel)", data.get("sequential") or {}),
        summarize_model_line("GAT (graphe)", data.get("graph") or {}),
        "",
        "## Analyse métier",
        f"- **Décision proposée** : {rec['decision']}",
        f"- **Actions** : {rec['actions']}",
        f"- **Surveillance** : {rec['surveillance']}",
        f"- **KYC** : {rec['kyc_note']}",
        "",
        "## Justification",
        "- Scores issus des modèles ML (source numérique prédictive).",
        "- Systèmes institutionnels autonomes (règles, alertes, recommandation).",
        "- Enrichissement documentaire via RAG (REF-08, règles métier).",
        "- Convergence modèles : " + ("oui" if data.get("models_agree") else "non — arbitrage humain recommandé"),
        "",
        "## Recommandation",
        rec["decision"] + ". " + rec["actions"],
        "",
        "## Conclusion",
        data["conclusion"],
        "",
        "## Références documentaires (RAG)",
        refs,
        "",
        "## Disclaimer",
        "- Données synthétiques / démonstration académique.",
        "- Ne remplace pas une décision humaine ni une validation conformité.",
        "- Pas de conseil juridique ou réglementaire.",
    ])
    return "\n".join(lines)
