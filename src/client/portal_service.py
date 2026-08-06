"""Données portail client — profil, alertes, synthèse crédit."""

from __future__ import annotations

from src.api.schemas import ClientAlert, ClientCreditSummary, ClientProfileResponse
from src.db.repository import get_client_credits, get_client_remboursements, get_client_transactions_summary
from src.kyc.score import compute_kyc_score_row


def build_client_profile_response(*, cin: str, client_id: int, row: dict) -> ClientProfileResponse:
    credits = get_client_credits(int(client_id))
    kyc_score = round(float(compute_kyc_score_row(row)), 2)

    n_def = sum(1 for c in credits if int(c.get("en_defaut") or 0))
    n_act = len(credits) - n_def
    montant = sum(float(c.get("montant") or 0) for c in credits)
    dti_vals = [float(c.get("dti") or 0) for c in credits]
    dti_moy = round(sum(dti_vals) / max(len(dti_vals), 1), 3)

    remb = get_client_remboursements(int(client_id))
    n_remb = len(remb)
    n_late = sum(1 for r in remb if int(r.get("retard_jours") or 0) > 0)
    taux_retard = round(n_late / max(n_remb, 1), 3)

    prochaine = None
    for r in sorted(remb, key=lambda x: str(x.get("date_echeance", ""))):
        if str(r.get("statut", "")).upper() != "PAYE" or int(r.get("retard_jours") or 0) > 0:
            prochaine = str(r.get("date_echeance", ""))[:10]
            break

    alerts: list[ClientAlert] = []
    sk = str(row.get("statut_kyc", ""))
    if sk == "RISQUE":
        alerts.append(ClientAlert(level="danger", title="KYC à risque", message="Votre dossier KYC nécessite une mise à jour urgente."))
    elif sk == "A_VERIFIER":
        alerts.append(ClientAlert(level="warning", title="KYC à vérifier", message="Merci de fournir les pièces manquantes à votre agence."))
    if n_def > 0:
        alerts.append(ClientAlert(level="danger", title="Crédit en défaut", message=f"{n_def} crédit(s) en situation de défaut — contactez votre agent."))
    if taux_retard > 0.25:
        alerts.append(ClientAlert(level="warning", title="Retards de paiement", message=f"Taux de retard élevé ({taux_retard * 100:.0f} %) — planifiez vos échéances."))
    tx_sum = get_client_transactions_summary(int(client_id))
    if int(tx_sum.get("n_suspect") or 0) > 0:
        alerts.append(ClientAlert(level="warning", title="Transactions suspectes", message="Des opérations atypiques ont été détectées sur votre compte."))
    if not alerts:
        alerts.append(ClientAlert(level="info", title="Dossier à jour", message="Aucune alerte active. Votre relation avec Talys est en bonne voie."))

    if n_def > 0 or sk == "RISQUE":
        sante = "FRAGILE"
    elif sk == "A_VERIFIER" or taux_retard > 0.2:
        sante = "A_SURVEILLER"
    elif kyc_score >= 70 and taux_retard < 0.1:
        sante = "EXCELLENT"
    else:
        sante = "BON"

    return ClientProfileResponse(
        cin=cin,
        client_id=int(client_id),
        nom=row["nom"],
        prenom=row["prenom"],
        age=int(row["age"]),
        ville=row["ville"],
        profession=row["profession"],
        revenu_mensuel=float(row["revenu_mensuel"]),
        statut_kyc=row["statut_kyc"],
        kyc_score=kyc_score,
        credits=credits,
        credit_summary=ClientCreditSummary(
            total=len(credits),
            actifs=n_act,
            en_defaut=n_def,
            montant_total=round(montant, 2),
            dti_moyen=dti_moy,
        ),
        alerts=alerts,
        sante_dossier=sante,
        prochaine_echeance=prochaine,
        taux_retard=taux_retard,
    )


def profile_to_context(profile: ClientProfileResponse) -> dict:
    """Contexte sérialisable pour le chatbot client."""
    return profile.model_dump()
