# Credit Default Risk Prediction — PFE Talys

## 🎯 Objectif
Système de prédiction du risque de défaut de crédit basé sur des données de microfinance tunisienne (5 000 clients, ~15 000 crédits).

## 📁 Structure du projet

```
Talys_pfe/
├── data/
│   ├── raw/               ← CSVs bruts (générés par generate_data.py)
│   ├── processed/         ← Données nettoyées (Parquet)
│   └── features/          ← Matrice de features (features.parquet)
├── notebooks/
│   └── 01_eda.ipynb       ← Analyse exploratoire des données
├── src/
│   ├── config.py          ← Chemins et constantes centralisés
│   ├── data/
│   │   ├── loader.py      ← Chargement des 5 CSVs
│   │   └── cleaner.py     ← Nettoyage + export Parquet
│   ├── features/
│   │   └── engineering.py ← Construction de la matrice de features
│   ├── models/
│   │   ├── train.py       ← Entraînement LR / RF / XGBoost
│   │   └── evaluate.py    ← Métriques + courbes ROC
│   └── api/
│       ├── main.py        ← Application FastAPI
│       └── schemas.py     ← Modèles Pydantic
├── models/                ← Modèles sauvegardés (.joblib)
├── reports/figures/       ← Graphiques EDA + ROC
├── requirements.txt
└── README.md
```

## ⚡ Démarrage rapide

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. (Optionnel) Regénérer les données
```bash
python generate_data.py
```

### 3. Construire la matrice de features
```bash
python -m src.features.engineering
```

### 4. Entraîner les modèles
```bash
python -m src.models.train
```

### 5. Lancer l'API
```bash
uvicorn src.api.main:app --reload --port 8000
```
→ Ouvrir : http://localhost:8000/docs

## 🔑 Variables (Target : `en_defaut`)

| Table | Colonnes clés |
|---|---|
| clients.csv | age, revenu_mensuel, statut_kyc, profession |
| credits.csv | montant, duree_mois, dti, cycle, **en_defaut** |
| remboursements.csv | retard_jours, statut |
| transactions.csv | type, montant, suspect |
| relations.csv | type_relation, risk_relation |

## 🤖 Modèles comparés
- Logistic Regression (baseline)
- Random Forest (300 arbres)
- XGBoost (400 boosters)

Métrique principale : **AUC-ROC**. Le meilleur modèle est sauvegardé automatiquement.

## 📊 API Endpoint

```
POST /predict
Content-Type: application/json

{
  "montant": 5000,
  "dti": 0.45,
  "revenu_mensuel": 2000,
  ...
}
```

Réponse :
```json
{
  "prediction": 0,
  "default_proba": 0.12,
  "risk_level": "FAIBLE",
  "model_used": "XGBoost"
}
```
