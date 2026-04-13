# Critères de score et mini section CRISP-DM

## 1) Critères utilisés pour le calcul du score

Le scoring de défaut (`en_defaut`) s'appuie sur trois familles de critères.

### A. KYC / Profil client
- `statut_kyc_enc` (OK / A_VERIFIER / RISQUE)
- `revenu_mensuel`
- `age`
- `profession_enc`

### B. Historique transactionnel
- `n_transactions`, `n_suspect`, `avg_tx_amount`
- `total_depot`, `total_retrait`, `total_remboursement`, `total_transfert`
- `ratio_retrait_depot`

### C. Comportement de remboursement
- `avg_retard`, `max_retard`, `std_retard`
- `n_payments`, `n_late`, `pct_late`
- `n_en_retard` (retards severes)

### D. Réseau relationnel (graph)
- `max_risk_relation`, `avg_risk_relation`
- `n_relations`, `n_garant`

### E. Variables crédit
- `montant`, `duree_mois`, `dti`
- `cycle_enc`, `objet_enc`

## 2) Dataset et structure

Sources brutes (`data/raw/`):
- `clients.csv`
- `credits.csv`
- `transactions.csv`
- `remboursements.csv`
- `relations.csv`

Pipeline:
1. `src/data/loader.py`: chargement + typage
2. `src/data/cleaner.py`: nettoyage + sauvegarde parquet (`data/processed/`)
3. `src/features/engineering.py`: agrégations + encodage + dataset final (`data/features/features.parquet`)

## 3) Mini section CRISP-DM (ce qui est fait)

### 3.1 Business Understanding
- Objectif: estimer le risque de défaut et fournir une explication lisible.

### 3.2 Data Understanding
- Données synthétiques Faker couvrant profil client, crédits, transactions, remboursements, relations.

### 3.3 Data Preparation
- Nettoyage, validation de types, agrégations multi-tables, encodage des variables catégorielles.

### 3.4 Modeling
- Modèles classiques implémentés:
  - Logistic Regression
  - Random Forest
  - XGBoost

### 3.5 Evaluation
- Métriques: AUC-ROC, Average Precision, classification report.
- Artefacts de visualisation: ROC + confusion matrix.

### 3.6 Deployment
- API FastAPI:
  - `POST /predict`
  - `POST /explain`
  - `POST /explain/by-cin`
- LLM local via Ollama pour l'explication textuelle.

## 4) Limites actuelles

1. Dataset synthétique (pas encore validé métier final).
2. Distribution des probabilités souvent extrême (peu de cas `MODERE`).
3. Pas encore de modèle séquentiel (LSTM/GRU).
4. Pas encore de GNN/GraphSAGE.
5. Pas encore de RAG ni d'orchestration LangChain/LangGraph.
6. Pas encore d'intégration ICM finale.
