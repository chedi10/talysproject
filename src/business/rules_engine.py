from __future__ import annotations

"""Business Rules Engine — politique crédit institutionnelle."""

from typing import Any, Literal

Severity = Literal["INFO", "WARNING", "CRITICAL"]
Action = Literal["none", "alert", "manual_review", "block"]

POLICY_REFS = {
    "dti_elevé": "POL-CRD-001",
    "kyc_faible": "POL-KYC-002",
    "retard_important": "POL-REM-003",
    "taux_retard_eleve": "POL-REM-004",
    "transactions_suspectes": "POL-AML-005",
    "ratio_retrait_depot": "POL-TRE-006",
    "risque_relationnel": "POL-REL-007",
    "sans_garant_risque_eleve": "POL-GAR-008",
    "proba_critique": "POL-RIS-009",
    "montant_elevé": "POL-CRD-010",
    "retard_severe": "POL-REM-011",
    "volatilite_retards": "POL-REM-012",
    "premier_cycle": "POL-CRD-013",
}


def _rule(
    rule_id: str,
    name: str,
    *,
    triggered: bool,
    severity: Severity,
    action: Action,
    message: str,
    value: Any = None,
    threshold: Any = None,
    policy_ref: str = "",
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "name": name,
        "triggered": triggered,
        "severity": severity if triggered else "INFO",
        "action": action if triggered else "none",
        "message": message if triggered else f"OK — {name}",
        "value": value,
        "threshold": threshold,
        "policy_ref": policy_ref or POLICY_REFS.get(rule_id, ""),
    }


def evaluate_business_rules(features: dict[str, Any], *, risk_level: str, default_proba: float) -> dict[str, Any]:
    dti = float(features.get("dti", 0))
    kyc = float(features.get("kyc_score", 0))
    max_retard = float(features.get("max_retard", 0))
    avg_retard = float(features.get("avg_retard", 0))
    std_retard = float(features.get("std_retard", 0))
    pct_late = float(features.get("pct_late", 0))
    n_suspect = int(features.get("n_suspect", 0))
    n_garant = int(features.get("n_garant", 0))
    n_en_retard = int(features.get("n_en_retard", 0))
    ratio_rd = float(features.get("ratio_retrait_depot", 0))
    max_rel = float(features.get("max_risk_relation", 0))
    montant = float(features.get("montant", 0))
    revenu = float(features.get("revenu_mensuel", 0))
    cycle = int(features.get("cycle_enc", 0))

    montant_ratio = montant / max(revenu, 1.0)

    rules = [
        _rule(
            "dti_elevé", "DTI élevé",
            triggered=dti > 0.50,
            severity="CRITICAL" if dti > 0.65 else "WARNING",
            action="manual_review" if dti > 0.50 else "none",
            message=f"DTI élevé ({dti:.0%}) — capacité de remboursement tendue.",
            value=round(dti, 3), threshold=0.50,
        ),
        _rule(
            "kyc_faible", "Score KYC insuffisant",
            triggered=kyc < 50,
            severity="CRITICAL" if kyc < 35 else "WARNING",
            action="manual_review" if kyc < 50 else "none",
            message=f"Score KYC faible ({kyc:.0f}/100) — vérification identité requise.",
            value=round(kyc, 1), threshold=50,
        ),
        _rule(
            "retard_important", "Retard de paiement important",
            triggered=max_retard >= 30 or avg_retard >= 15,
            severity="CRITICAL" if max_retard >= 60 else "WARNING",
            action="manual_review",
            message=f"Retards significatifs (max {max_retard:.0f}j, moy {avg_retard:.1f}j).",
            value={"max_retard": max_retard, "avg_retard": avg_retard},
            threshold={"max_retard": 30},
        ),
        _rule(
            "taux_retard_eleve", "Taux de retard élevé",
            triggered=pct_late >= 0.25,
            severity="WARNING", action="alert",
            message=f"Taux de retard {pct_late:.0%} — comportement de remboursement dégradé.",
            value=round(pct_late, 3), threshold=0.25,
        ),
        _rule(
            "transactions_suspectes", "Transactions suspectes",
            triggered=n_suspect >= 2,
            severity="CRITICAL" if n_suspect >= 5 else "WARNING",
            action="manual_review",
            message=f"{n_suspect} transaction(s) suspecte(s) détectée(s).",
            value=n_suspect, threshold=2,
        ),
        _rule(
            "ratio_retrait_depot", "Ratio retrait/dépôt élevé",
            triggered=ratio_rd > 0.85,
            severity="WARNING", action="alert",
            message=f"Ratio retrait/dépôt {ratio_rd:.0%} — pression de trésorerie.",
            value=round(ratio_rd, 3), threshold=0.85,
        ),
        _rule(
            "risque_relationnel", "Réseau relationnel à risque",
            triggered=max_rel >= 60,
            severity="WARNING", action="alert",
            message=f"Relation à haut risque (max {max_rel:.0f}).",
            value=max_rel, threshold=60,
        ),
        _rule(
            "sans_garant_risque_eleve", "Risque élevé sans garant",
            triggered=risk_level == "ELEVE" and n_garant == 0,
            severity="CRITICAL", action="manual_review",
            message="Risque ELEVE sans garant enregistré.",
            value={"risk_level": risk_level, "n_garant": n_garant},
        ),
        _rule(
            "proba_critique", "Score institutionnel critique",
            triggered=default_proba >= 0.60,
            severity="CRITICAL", action="block",
            message=f"Score institutionnel {default_proba:.1%} — seuil critique dépassé.",
            value=round(default_proba, 4), threshold=0.60,
        ),
        _rule(
            "montant_elevé", "Montant disproportionné vs revenu",
            triggered=montant_ratio > 3.0,
            severity="CRITICAL" if montant_ratio > 5.0 else "WARNING",
            action="manual_review" if montant_ratio > 3.0 else "none",
            message=f"Montant {montant:,.0f} TND = {montant_ratio:.1f}× revenu mensuel.",
            value=round(montant_ratio, 2), threshold=3.0,
        ),
        _rule(
            "retard_severe", "Retards sévères (≥90j)",
            triggered=n_en_retard >= 1,
            severity="CRITICAL", action="block",
            message=f"{n_en_retard} échéance(s) en retard sévère (≥90 jours).",
            value=n_en_retard, threshold=1,
        ),
        _rule(
            "volatilite_retards", "Volatilité des retards",
            triggered=std_retard >= 20,
            severity="WARNING", action="alert",
            message=f"Volatilité des retards élevée (σ={std_retard:.1f}j).",
            value=round(std_retard, 1), threshold=20,
        ),
        _rule(
            "premier_cycle", "Premier cycle + DTI contraint",
            triggered=cycle == 0 and dti > 0.40,
            severity="WARNING", action="alert",
            message=f"Premier crédit avec DTI {dti:.0%} — profil novice à risque.",
            value={"cycle": cycle, "dti": round(dti, 3)}, threshold={"dti": 0.40},
        ),
    ]

    triggered = [r for r in rules if r["triggered"]]
    requires_manual = any(r["action"] in ("manual_review", "block") for r in triggered)
    highest = "INFO"
    for r in triggered:
        if r["severity"] == "CRITICAL":
            highest = "CRITICAL"
            break
        if r["severity"] == "WARNING":
            highest = "WARNING"

    # Score de conformité (100 = parfait)
    penalty = sum(
        25 if r["severity"] == "CRITICAL" else 12 if r["severity"] == "WARNING" else 0
        for r in triggered
    )
    compliance_score = max(0, 100 - penalty)

    return {
        "rules": rules,
        "triggered_count": len(triggered),
        "triggered_rule_ids": [r["rule_id"] for r in triggered],
        "requires_manual_review": requires_manual,
        "highest_severity": highest,
        "compliance_score": compliance_score,
        "summary": (
            f"{len(triggered)} règle(s) métier déclenchée(s) — conformité {compliance_score}/100."
            + (" Revue manuelle requise." if requires_manual else " Profil conforme aux règles.")
        ),
    }
