# Règles métier — Scoring risque crédit (microfinance)

## Niveaux de risque
- **FAIBLE** : probabilité de défaut faible, profil stable, KYC conforme, peu de retards.
- **MODERE** : vigilance accrue, suivi renforcé des échéances et du comportement transactionnel.
- **ELEVE** : probabilité de défaut élevée, retards répétés ou signaux réseau défavorables — décision humaine obligatoire.

## Score KYC (0–100)
- Score calculé à partir du profil client (statut KYC, revenu, âge, profession).
- Un KYC faible ne remplace pas le score crédit : les deux dimensions sont complémentaires.
- Toute décision doit recouper KYC et risque de défaut.

## Modèles et usage
- **Classique (tabulaire)** : montant, DTI, retards agrégés, transactions, réseau — référence rapide.
- **Séquentiel (LSTM/GRU)** : historique transactions + remboursements dans le temps — détecte la dégradation progressive.
- **Graphe (GraphSAGE)** : risque relationnel (garants, contreparties) — signale la contagion ou l’isolement.

## Recommandations types
- FAIBLE : validation sous surveillance standard.
- MODERE : conditions renforcées, plafond ou garantie, revue à 3 mois.
- ELEVE : refus ou restructuration avec comité crédit ; ne pas automatiser l’accord.

## Conformité
- Données synthétiques en démonstration académique.
- Le rapport IA (RAG + LLM) **n’altère jamais** les scores numériques des modèles.
- Décision finale : analyste crédit / comité, pas le chatbot.
