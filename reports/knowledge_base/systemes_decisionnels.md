# Systèmes décisionnels Talys — Guide agent conversationnel

## Architecture
L'application Talys sépare clairement :
- **Modèles ML** (Deep Tabular, Temporal Transformer, GAT) : scoring prédictif via endpoints /explain/*
- **Systèmes institutionnels autonomes** : SHAP, Business Rules, Early Warning, Recommandation IA

Les systèmes institutionnels utilisent un **score métier** calculé depuis KYC, DTI, retards, transactions et historique crédit — sans sélection de modèle ML.

## Business Rules Engine
- 13 règles métier : DTI, KYC, retards, transactions suspectes, montant vs revenu, retards sévères ≥90j
- Score de conformité /100
- Références politique : POL-CRD-001 à POL-CRD-013
- Actions : alert, manual_review, block

## Early Warning System
- Surveillance proactive vs historique client
- Priorité watchlist : NONE, LOW, MEDIUM, HIGH
- Alertes : dégradation DTI/KYC, retards croissants, transactions suspectes, retards sévères
- Tendances sur les derniers crédits

## Recommandation IA
- Décisions : ACCEPTER, ACCEPTER_AVEC_GARANTIE, REDUIRE_MONTANT, DEMANDER_GARANT, REFUSER
- Montant suggéré, DTI cible, conditions contractuelles
- Plan d'action personnalisé, fréquence de suivi

## SHAP / Explainable AI
- Facteurs explicatifs par client
- Comparaison valeur client vs médiane portefeuille
- Indépendant du choix utilisateur de modèle ML

## Chat — intents supportés
- **classic_score** : scoring Deep Tabular
- **sequential_score** : Temporal Transformer
- **graph_score** : GAT Graph Attention
- **compare_models** : comparaison des 3 modèles
- **full_report** : rapport complet Markdown + RAG
- **institutional** : analyse systèmes décisionnels (règles + alertes + recommandation)

## RAG — sources documentaires
- REF-08 : spécifications scoring crédit
- Règles métier microfinance
- Critères score et CRISP-DM

## Rapport PDF
Le rapport comité crédit inclut : KPIs ML, systèmes institutionnels, recommandation, références RAG, disclaimer.
