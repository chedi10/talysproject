from __future__ import annotations

"""
Assistant conversationnel client Talys — dossier personnel + démarches + solutions + FAQ.

Couvre : profil, crédits, KYC, paiements, alertes, demande crédit, retards,
défaut, documents, réclamations, droits client. RAG + guides structurés intégrés.
"""

import os
from typing import Any, Callable, Literal

ClientIntent = Literal[
    "summary",
    "profile",
    "credits",
    "kyc",
    "payments",
    "alerts",
    "contact",
    "demarche_credit",
    "demarche_kyc",
    "solution_retard",
    "solution_defaut",
    "documents",
    "faq",
    "aide",
    "general",
]

SANTE_LABELS = {
    "EXCELLENT": "excellent",
    "BON": "bon",
    "A_SURVEILLER": "à surveiller",
    "FRAGILE": "fragile",
}


def _detect_client_intent(text: str, history: list[dict[str, str]] | None = None) -> ClientIntent:
    t = (text or "").lower()

    if any(k in t for k in ["nouveau crédit", "nouveau credit", "demande crédit", "demande credit", "emprunter", "souscrire", "obtenir un prêt", "obtenir un pret", "faire un crédit"]):
        return "demarche_credit"
    if any(k in t for k in ["rééchelon", "reechelon", "retard", "impayé", "impaye", "difficulté", "difficulte", "ne peux pas payer", "report", "pénalité", "penalite"]):
        return "solution_retard"
    if any(k in t for k in ["défaut", "defaut", "recouvrement", "contentieux", "impayés", "impayes"]):
        return "solution_defaut"
    if any(k in t for k in ["document", "pièce", "piece", "papier", "justificatif", "dossier à fournir"]):
        return "documents"
    if any(k in t for k in ["mettre à jour kyc", "maj kyc", "actualiser kyc", "kyc expir", "pièce identité", "piece identite"]) or (
        "kyc" in t and any(k in t for k in ["mettre", "update", "modifier", "démarche", "demarche", "comment"])
    ):
        return "demarche_kyc"
    if any(k in t for k in ["tout", "complet", "résume", "resume", "synthèse", "synthese", "global", "mon dossier"]):
        return "summary"
    if any(k in t for k in ["crédit", "credit", "prêt", "pret", "montant", "encours"]) and not any(k in t for k in ["demande", "nouveau", "obtenir"]):
        return "credits"
    if "kyc" in t or any(k in t for k in ["identité", "identite", "conformité", "conformite"]):
        return "kyc"
    if any(k in t for k in ["échéance", "echeance", "payer", "paiement", "rembours", "mensualité", "mensualite", "virement"]):
        return "payments"
    if any(k in t for k in ["alerte", "alert", "notification", "problème dossier"]):
        return "alerts"
    if any(k in t for k in ["contact", "agent", "conseiller", "agence", "téléphone", "telephone", "email", "joindre", "horaire"]):
        return "contact"
    if any(k in t for k in ["contester", "réclamation", "reclamation", "recours", "droits", "plainte"]):
        return "faq"
    if any(k in t for k in ["bonjour", "salut", "hello", "aide-moi", "aide moi", "aide", "help", "que peux", "que puis"]):
        return "aide"
    if any(k in t for k in ["comment ça marche", "comment ca marche", "explique", "c'est quoi", "quest ce", "pourquoi", "microfinance", "dti", "glossaire"]):
        return "faq"
    if history and len(t.split()) <= 4:
        for msg in reversed(history or []):
            if msg.get("role") == "user":
                return _detect_client_intent(msg.get("content", ""), None)
    return "general"


def client_suggested_prompts(profile: dict) -> list[str]:
    prompts = [
        "Résume mon dossier complet",
        "Comment demander un nouveau crédit ?",
        "J'ai un retard de paiement, que faire ?",
        "Quels documents pour un crédit ?",
        "Mon KYC est-il conforme ?",
        "Comment contacter mon agent ?",
    ]
    summary = profile.get("credit_summary") or {}
    if int(summary.get("en_defaut") or 0) > 0:
        prompts.insert(2, "Mon crédit est en défaut, quelles solutions ?")
    if profile.get("prochaine_echeance"):
        prompts.insert(1, "Quand est ma prochaine échéance ?")
    return prompts[:8]


def _format_credits_block(profile: dict) -> str:
    credits = profile.get("credits") or []
    summary = profile.get("credit_summary") or {}
    if not credits:
        return "Vous n'avez aucun crédit enregistré. Vous pouvez demander **comment souscrire un premier crédit**."
    lines = [
        f"Vous avez **{summary.get('total', len(credits))} crédit(s)** : "
        f"**{summary.get('actifs', 0)} actif(s)** et **{summary.get('en_defaut', 0)} en défaut**.",
        f"Encours total : **{summary.get('montant_total', 0):,.0f} TND** · DTI moyen **{float(summary.get('dti_moyen', 0)) * 100:.0f} %**.",
        "",
    ]
    for c in credits[:6]:
        statut = "En défaut" if c.get("en_defaut") else "Actif"
        lines.append(
            f"- **{c.get('objet', 'Crédit')}** — {float(c.get('montant', 0)):,.0f} TND, "
            f"{c.get('duree_mois')} mois → **{statut}**"
        )
    return "\n".join(lines)


def _format_alerts_block(profile: dict) -> str:
    alerts = profile.get("alerts") or []
    if not alerts:
        return "Aucune alerte active."
    lines = []
    for a in alerts:
        icon = "🔴" if a.get("level") == "danger" else "🟠" if a.get("level") == "warning" else "ℹ️"
        lines.append(f"{icon} **{a.get('title')}** — {a.get('message')}")
    return "\n".join(lines)


def _personalized_tip(profile: dict) -> str:
    tips = []
    sk = profile.get("statut_kyc", "")
    if sk != "OK":
        tips.append("Mettez à jour votre KYC en agence (demandez « démarche KYC »).")
    if float(profile.get("taux_retard") or 0) > 0.15:
        tips.append("Anticipez vos échéances ou demandez un rééchelonnement.")
    if int((profile.get("credit_summary") or {}).get("en_defaut") or 0) > 0:
        tips.append("Contactez collections pour un plan de recouvrement amiable.")
    if not tips:
        tips.append("Votre dossier est sain — vous pouvez envisager un nouveau crédit si besoin.")
    return " **Conseil personnalisé :** " + " ".join(tips)


def _procedure_demarche_credit(profile: dict) -> str:
    ville = profile.get("ville", "votre ville")
    return "\n".join([
        "## Demander un nouveau crédit — démarche Talys",
        "",
        "### Les 6 étapes",
        "1. **Contact agence** — Rendez-vous avec CIN + justificatif de revenu",
        "2. **Analyse dossier** — KYC, capacité de remboursement (DTI ≤ 40 % recommandé)",
        "3. **Choix produit** — Consommation, santé, éducation, micro-entreprise, logement",
        "4. **Proposition** — Montant et durée adaptés à vos revenus",
        "5. **Décision** — Sous 48–72 h ouvrées",
        "6. **Décaissement** — Après signature du contrat",
        "",
        "### Votre situation actuelle",
        _format_credits_block(profile),
        "",
        f"Votre KYC : **{profile.get('statut_kyc')}** ({profile.get('kyc_score')}/100).",
        _personalized_tip(profile),
        "",
        f"📍 Agence **{ville}** — Lun–Ven 8h30–17h · Demandez « documents crédit » pour la liste complète.",
    ])


def _procedure_demarche_kyc(profile: dict) -> str:
    sk = profile.get("statut_kyc", "")
    steps = [
        "1. CIN originale + copie",
        "2. Justificatif domicile (< 3 mois)",
        "3. Justificatif revenu à jour",
        "4. Dépôt en agence — traitement 24–48 h",
    ]
    return "\n".join([
        "## Mise à jour KYC — démarche",
        "",
        f"Votre statut actuel : **{sk}** (score {profile.get('kyc_score')}/100)",
        "",
        "### Étapes",
        *[f"- {s}" for s in steps],
        "",
        "**Pourquoi ?** Obligation réglementaire anti-fraude. Un KYC à jour facilite vos demandes de crédit.",
        _personalized_tip(profile),
    ])


def _procedure_solution_retard(profile: dict) -> str:
    return "\n".join([
        "## Retard de paiement — solutions",
        "",
        f"Votre taux de retard actuel : **{float(profile.get('taux_retard', 0)) * 100:.0f} %**",
        f"Prochaine échéance : **{profile.get('prochaine_echeance') or '—'}**",
        "",
        "### Que faire ? (par ordre de priorité)",
        "1. **Appelez votre agent AVANT la date limite** — 71 000 000",
        "2. **Demandez un report** — Possible 1×/an (max 15 jours)",
        "3. **Rééchelonnement** — Si baisse temporaire de revenus (étude dossier)",
        "4. **Plan de médiation** — Rendez-vous collections",
        "",
        "### À éviter",
        "- Ignorer les rappels · Payer partiellement sans accord · Multiplier les retards",
        "",
        _personalized_tip(profile),
    ])


def _procedure_solution_defaut(profile: dict) -> str:
    n_def = int((profile.get("credit_summary") or {}).get("en_defaut") or 0)
    return "\n".join([
        "## Crédit en défaut — solutions Talys",
        "",
        f"Crédit(s) en défaut sur votre dossier : **{n_def}**",
        "",
        "### Options disponibles",
        "1. **Plan de recouvrement amiable** — Échéancier négocié",
        "2. **Renégociation** — Mensualités réduites temporairement",
        "3. **Garant supplémentaire** — Si disponible",
        "4. **Comité crédit** — Dossiers complexes",
        "",
        "### Démarche immédiate",
        "- Contactez l'agence sous **48 h** avec proposition écrite de remboursement",
        "- Conservez tous les reçus de paiement partiel",
        "",
        _personalized_tip(profile),
    ])


def _procedure_documents() -> str:
    return "\n".join([
        "## Documents pour un crédit Talys",
        "",
        "### Obligatoires",
        "- CIN valide (originale + copie)",
        "- Justificatif de domicile (< 3 mois)",
        "- Justificatif de revenu (fiche paie, attestation, relevé activité)",
        "",
        "### Selon le type de crédit",
        "- **Santé / Éducation** — Devis ou facture proforma",
        "- **Micro-entreprise** — Registre commerce ou attestation activité",
        "- **Consommation** — Parfois relevé bancaire 3 mois",
        "",
        "### Mise à jour KYC",
        "- Mêmes pièces d'identité + domicile si changement d'adresse",
    ])


def _build_answer(intent: ClientIntent, profile: dict, user_message: str, rag_block: str = "") -> str:
    prenom = profile.get("prenom", "")
    nom = profile.get("nom", "")
    sante = SANTE_LABELS.get(str(profile.get("sante_dossier", "")), "—")

    if intent == "demarche_credit":
        base = _procedure_demarche_credit(profile)
    elif intent == "demarche_kyc":
        base = _procedure_demarche_kyc(profile)
    elif intent == "solution_retard":
        base = _procedure_solution_retard(profile)
    elif intent == "solution_defaut":
        base = _procedure_solution_defaut(profile)
    elif intent == "documents":
        base = _procedure_documents()
    elif intent == "summary":
        base = "\n".join([
            f"## Votre dossier Talys — {prenom} {nom}",
            "",
            f"**Santé :** {sante.title()} · **KYC** {profile.get('kyc_score')}/100 ({profile.get('statut_kyc')})",
            f"**{profile.get('ville')}** · {profile.get('profession')} · {float(profile.get('revenu_mensuel', 0)):,.0f} TND/mois",
            "",
            "### Crédits", _format_credits_block(profile),
            "",
            "### Paiements",
            f"Retard {float(profile.get('taux_retard', 0)) * 100:.0f} %"
            + (f" · Échéance **{profile.get('prochaine_echeance')}**" if profile.get("prochaine_echeance") else ""),
            "",
            "### Alertes", _format_alerts_block(profile),
            "",
            _personalized_tip(profile),
        ])
    elif intent == "profile":
        base = "\n".join([
            f"Bonjour **{prenom}** !",
            "",
            f"- **CIN :** {profile.get('cin')} · **{profile.get('age')} ans**",
            f"- **{profile.get('ville')}** · {profile.get('profession')}",
            f"- **Revenu :** {float(profile.get('revenu_mensuel', 0)):,.0f} TND/mois",
            f"- **KYC :** {profile.get('statut_kyc')} ({profile.get('kyc_score')}/100)",
            f"- **Santé dossier :** {sante.title()}",
        ])
    elif intent == "credits":
        base = "## Vos crédits\n\n" + _format_credits_block(profile) + _personalized_tip(profile)
    elif intent == "kyc":
        base = _procedure_demarche_kyc(profile) if profile.get("statut_kyc") != "OK" else "\n".join([
            "## Votre KYC",
            f"Statut **{profile.get('statut_kyc')}** — score **{profile.get('kyc_score')}/100**.",
            "Votre dossier est conforme. Aucune action immédiate.",
        ])
    elif intent == "payments":
        base = _procedure_solution_retard(profile) if float(profile.get("taux_retard") or 0) > 0.1 else "\n".join([
            "## Vos remboursements",
            f"Prochaine échéance : **{profile.get('prochaine_echeance') or '—'}**",
            f"Taux de retard : **{float(profile.get('taux_retard', 0)) * 100:.0f} %** — bon comportement !",
            "",
            "**Modes de paiement :** agence, virement (réf. crédit obligatoire), prélèvement sur demande.",
        ])
    elif intent == "alerts":
        base = "## Alertes\n\n" + _format_alerts_block(profile) + "\n\n" + _personalized_tip(profile)
    elif intent == "contact":
        base = "\n".join([
            "## Contacter Talys",
            f"- **Agence {profile.get('ville', 'Tunis')}** — Lun–Ven 8h30–17h, Sam 8h30–12h30",
            "- **Tél :** 71 000 000 · **Email :** contact@talys.local",
            "- Préparez CIN + n° crédit · Délai réponse 24 h",
        ])
    elif intent == "aide":
        base = "\n".join([
            f"Bonjour **{prenom}** ! Je suis **Talys Assistant**, votre guide microfinance.",
            "",
            "Je peux vous aider sur :",
            "",
            "| Sujet | Exemples |",
            "|-------|----------|",
            "| **Votre dossier** | Résumé, crédits, KYC, échéances, alertes |",
            "| **Démarches** | Nouveau crédit, mise à jour KYC, documents |",
            "| **Solutions** | Retard de paiement, crédit en défaut, rééchelonnement |",
            "| **Informations** | DTI, types de crédits, droits, réclamations |",
            "",
            "Posez votre question en langage naturel — je m'adapte à votre situation.",
        ])
    elif intent == "faq":
        base = "## Informations & FAQ\n\n"
        if rag_block:
            base += rag_block + "\n\n---\n\n"
        base += (
            "**DTI** = part des revenus consacrée au remboursement (idéal < 40 %).\n\n"
            "**Réclamation :** demande écrite en agence → réexamen 5 jours → recours comité si besoin.\n\n"
            "**Remboursement anticipé :** possible après 6 mois sans pénalité (sauf clause contrat)."
        )
    else:
        base = (
            f"Bonjour **{prenom}** ! Je peux vous guider sur votre **dossier**, les **démarches crédit/KYC**, "
            "les **solutions en cas de retard**, les **documents requis** et vos **droits**.\n\n"
            "Exemples : « Comment demander un crédit ? », « J'ai un retard », « Quels documents ? »"
        )

    return base


def _rag_queries_for_intent(intent: ClientIntent, message: str, profile: dict) -> list[str]:
    cin = str(profile.get("cin", ""))
    base = [
        "guide client Talys microfinance démarche crédit documents",
        "KYC mise à jour pièces justificatives client",
        "retard paiement rééchelonnement recouvrement solutions",
        "types crédit consommation santé éducation micro-entreprise",
        "FAQ client droits réclamation DTI échéance",
    ]
    intent_queries = {
        "demarche_credit": ["demande nouveau crédit étapes éligibilité documents décaissement"],
        "demarche_kyc": ["KYC statut OK A_VERIFIER RISQUE mise à jour"],
        "solution_retard": ["retard remboursement report rééchelonnement médiation"],
        "solution_defaut": ["crédit défaut recouvrement renégociation garant"],
        "documents": ["documents justificatifs CIN domicile revenu devis"],
        "faq": [message[:80], "glossaire microfinance DTI KYC client"],
        "payments": ["mode paiement échéance virement agence"],
        "aide": ["guide client assistant démarches solutions"],
    }
    extra = intent_queries.get(intent, [message[:100] if message else "aide client"])
    if cin:
        base.append(f"client CIN {cin}")
    return extra + base


def _maybe_llm_enhance(base_answer: str, profile: dict, user_message: str, rag_block: str, intent: str) -> str:
    if os.getenv("DISABLE_LLM", "").strip().lower() in {"1", "true", "yes"}:
        return base_answer
    try:
        import httpx
        from src.llm.client import _ollama_base_url, _ollama_model

        prompt = "\n".join([
            "Tu es Talys Assistant, conseiller digital microfinance en Tunisie.",
            "Mission : aider le client sur SON dossier ET les démarches/solutions générales (crédit, KYC, retards, documents).",
            "",
            "Règles STRICTES :",
            "- Ton chaleureux, professionnel, en français.",
            "- Ne jamais révéler scores ML, probas défaut, données d'autres clients.",
            "- Pour les démarches : donner des étapes claires numérotées si pertinent.",
            "- Personnaliser avec les données client quand utile.",
            "- Markdown concis (max 20 lignes). Proposer une action concrète en fin de réponse.",
            "",
            f"Intent détecté : {intent}",
            f"Question : {user_message}",
            "",
            "Données client (autorisées) :",
            f"- {profile.get('prenom')} {profile.get('nom')}, CIN {profile.get('cin')}, {profile.get('ville')}",
            f"- KYC {profile.get('statut_kyc')} ({profile.get('kyc_score')}/100), santé {profile.get('sante_dossier')}",
            f"- Crédits: {profile.get('credit_summary')}, retard {profile.get('taux_retard')}, échéance {profile.get('prochaine_echeance')}",
            f"- Alertes: {[a.get('title') for a in (profile.get('alerts') or [])]}",
            "",
            "Brouillon (reformuler, enrichir si besoin, ne pas inventer) :",
            base_answer[:2500],
        ])
        if rag_block:
            prompt += f"\n\nBase documentaire RAG :\n{rag_block[:2000]}"

        payload = {
            "model": _ollama_model(),
            "messages": [
                {"role": "system", "content": "Talys Assistant — guide client microfinance, démarches et solutions."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.4},
        }
        with httpx.Client(timeout=50.0) as client:
            r = client.post(f"{_ollama_base_url()}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        msg = (data.get("message") or {}).get("content") or data.get("response")
        if msg and len(str(msg).strip()) > 50:
            return str(msg).strip()
    except Exception:
        pass
    return base_answer


class ClientChatbot:
    """Assistant client complet : dossier + démarches + solutions + RAG."""

    def __init__(self, rag_retrieve: Callable[..., list[dict[str, Any]]] | None = None):
        self._rag_retrieve = rag_retrieve

    def invoke(
        self,
        *,
        message: str,
        profile: dict,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        history = conversation_history or []
        intent = _detect_client_intent(message, history)
        rag_sources: list[dict[str, Any]] = []
        rag_block = ""

        rag_intents = {
            "faq", "general", "aide", "demarche_credit", "demarche_kyc",
            "solution_retard", "solution_defaut", "documents", "payments", "summary",
        }
        if self._rag_retrieve and intent in rag_intents:
            try:
                queries = _rag_queries_for_intent(intent, message, profile)
                seen: set[str] = set()
                for q in queries[:6]:
                    for chunk in self._rag_retrieve(q, k=2):
                        key = f"{chunk.get('source')}#{chunk.get('chunk_id')}"
                        if key not in seen:
                            seen.add(key)
                            rag_sources.append(chunk)
                if rag_sources:
                    from src.rag.context import format_rag_sources_for_prompt
                    rag_block = format_rag_sources_for_prompt(rag_sources, max_chars=2000)
            except Exception:
                pass

        base = _build_answer(intent, profile, message, rag_block)
        answer = _maybe_llm_enhance(base, profile, message, rag_block, intent)

        return {
            "intent": intent,
            "answer": answer,
            "rag_sources": rag_sources[:6],
            "suggested_prompts": client_suggested_prompts(profile),
        }
