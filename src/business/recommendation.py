from __future__ import annotations

"""AI Recommendation Engine — décision crédit assistée avec plan d'action."""

from typing import Any, Literal

Decision = Literal[
    "ACCEPTER",
    "ACCEPTER_AVEC_GARANTIE",
    "REDUIRE_MONTANT",
    "DEMANDER_GARANT",
    "REFUSER",
]


def _personalized_actions(
    decision: Decision,
    *,
    montant: float,
    dti: float,
    n_garant: int,
    suggested_montant: float | None,
    monitoring: str,
) -> list[str]:
    actions: list[str] = []
    if decision == "ACCEPTER":
        actions = [
            "Valider le dossier et émettre l'offre de crédit",
            f"Surveillance {monitoring} — prochaine revue dans 3 mois",
            f"Plafond d'exposition: {montant:,.0f} TND",
        ]
    elif decision == "ACCEPTER_AVEC_GARANTIE":
        actions = [
            "Contrat avec clause de revue à 3 mois",
            f"Suivi {monitoring} des échéances",
            f"Garant enregistré ({n_garant}) — vérifier solvabilité",
            "Plafond d'exposition avec marge de sécurité 15%",
        ]
    elif decision == "REDUIRE_MONTANT":
        target = suggested_montant or montant * 0.75
        actions = [
            f"Proposer montant réduit: {target:,.0f} TND (vs {montant:,.0f} TND demandé)",
            f"Recalculer DTI cible < 40% (actuel {dti:.0%})",
            "Revue comité crédit si montant > plafond interne",
        ]
    elif decision == "DEMANDER_GARANT":
        actions = [
            "Mettre le dossier en attente — garant obligatoire",
            "Exiger garant solvable avec KYC ≥ 65",
            "Contrôle renforcé des justificatifs de revenu",
        ]
    else:
        actions = [
            "Refus motivé au client avec courrier officiel",
            "Archiver dossier avec motifs réglementaires",
            "Proposer restructuration si éligible (contact collections)",
        ]
    return actions


def generate_recommendation(
    *,
    default_proba: float,
    risk_level: str,
    kyc_score: float,
    features: dict[str, Any],
    business_rules: dict[str, Any],
    early_warning: dict[str, Any],
    shap_summary: str | None = None,
) -> dict[str, Any]:
    risk = str(risk_level).upper()
    proba = float(default_proba)
    kyc = float(kyc_score)
    dti = float(features.get("dti", 0))
    n_garant = int(features.get("n_garant", 0))
    max_retard = float(features.get("max_retard", 0))
    montant = float(features.get("montant", 0))
    revenu = float(features.get("revenu_mensuel", 0))

    rules_triggered = business_rules.get("triggered_count", 0)
    triggered_ids = business_rules.get("triggered_rule_ids", [])
    requires_manual = business_rules.get("requires_manual_review", False)
    critical_rules = business_rules.get("highest_severity") == "CRITICAL"
    ew_critical = early_warning.get("critical_count", 0) > 0
    ew_degrade = early_warning.get("degradation_detected", False)
    ew_codes = [a.get("code", "") for a in early_warning.get("alerts", [])]

    decision: Decision
    confidence: float
    justification_parts: list[str] = []
    conditions: list[str] = []

    if proba >= 0.65 or risk == "ELEVE" or critical_rules or ew_critical:
        if n_garant == 0 and max_retard >= 30:
            decision = "REFUSER"
            confidence = 0.88
            justification_parts.append("Risque très élevé avec retards significatifs et absence de garant.")
            conditions.append("Dossier non éligible sans restructuration")
        elif n_garant == 0:
            decision = "DEMANDER_GARANT"
            confidence = 0.82
            justification_parts.append("Risque élevé — garant ou co-emprunteur requis avant décaissement.")
            conditions.append("Garant avec KYC ≥ 65 et revenu ≥ 1.5× mensualité")
        else:
            decision = "ACCEPTER_AVEC_GARANTIE"
            confidence = 0.75
            justification_parts.append("Risque élevé atténué par garant existant — conditions renforcées.")
            conditions.append("Clause de déchéance du terme si 2 retards consécutifs")
    elif risk == "MODERE" or proba >= 0.30 or requires_manual or ew_degrade:
        if dti > 0.45 or kyc < 55:
            decision = "REDUIRE_MONTANT"
            confidence = 0.78
            justification_parts.append("Profil modéré avec DTI ou KYC contraints — plafonnement recommandé.")
            conditions.append(f"DTI cible < 40% (actuel {dti:.0%})")
        elif n_garant == 0 and rules_triggered >= 2:
            decision = "DEMANDER_GARANT"
            confidence = 0.72
            justification_parts.append("Plusieurs règles métier déclenchées — garant recommandé.")
            conditions.append("Garant solvable exigé")
        else:
            decision = "ACCEPTER_AVEC_GARANTIE"
            confidence = 0.70
            justification_parts.append("Risque modéré — acceptation sous conditions de suivi renforcé.")
            conditions.append("Suivi mensuel pendant 6 mois")
    else:
        decision = "ACCEPTER"
        confidence = 0.85
        justification_parts.append("Profil faible risque, KYC et comportement de remboursement acceptables.")

    # Montant suggéré
    suggested_montant: float | None = None
    montant_reduction_pct: float | None = None
    if decision == "REDUIRE_MONTANT":
        factor = 0.70 if dti > 0.50 else 0.80
        suggested_montant = round(montant * factor, 0)
        montant_reduction_pct = round((1 - factor) * 100, 0)
    elif dti > 0.40 and revenu > 0:
        max_by_dti = revenu * 0.40 * int(features.get("duree_mois", 12))
        if max_by_dti < montant:
            suggested_montant = round(max_by_dti, 0)

    monitoring = "mensuel" if ew_critical or ew_degrade else "trimestriel"

    if shap_summary:
        justification_parts.append(shap_summary)
    if business_rules.get("summary"):
        justification_parts.append(business_rules["summary"])
    if early_warning.get("summary") and ew_degrade:
        justification_parts.append(early_warning["summary"])

    labels = {
        "ACCEPTER": "✔ Accepter",
        "ACCEPTER_AVEC_GARANTIE": "✔ Accepter avec garantie / conditions",
        "REDUIRE_MONTANT": "✔ Réduire le montant",
        "DEMANDER_GARANT": "✔ Demander un garant",
        "REFUSER": "✔ Refuser",
    }

    return {
        "decision": decision,
        "decision_label": labels[decision],
        "confidence": round(confidence, 2),
        "justification": " ".join(justification_parts),
        "recommended_actions": _personalized_actions(
            decision, montant=montant, dti=dti, n_garant=n_garant,
            suggested_montant=suggested_montant, monitoring=monitoring,
        ),
        "requires_manual_validation": requires_manual or decision in ("REFUSER", "DEMANDER_GARANT", "REDUIRE_MONTANT"),
        "suggested_montant": suggested_montant,
        "montant_reduction_pct": montant_reduction_pct,
        "suggested_dti_target": 0.40 if dti > 0.40 else None,
        "monitoring_frequency": monitoring,
        "conditions": conditions,
        "contributing_factors": {
            "rules": triggered_ids,
            "ews": ew_codes,
            "compliance_score": business_rules.get("compliance_score", 100),
        },
    }
