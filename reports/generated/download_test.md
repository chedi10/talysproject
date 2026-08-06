## Rapport de scoring — CIN 88710263

### Contexte
- Client analysé: `88710263`
- Analyse consolidée via modèles classique, séquentiel et graphe.

### Observations
| Modèle | Default proba | Risk level | KYC |
|---|---:|---|---:|
| Classique | 0.0003 | FAIBLE | 49.73 |
| Séquentiel | 0.7206 | ELEVE | 49.73 |
| Graphe | 0.9826 | ELEVE | 49.73 |

- Classique: proba=0.0003 | risque=FAIBLE | kyc=49.73 | model=Logistic Regression
- Séquentiel: proba=0.7206 | risque=ELEVE | kyc=49.73 | model=GRU baseline (GRU) (GRU)
- Graphe: proba=0.9826 | risque=ELEVE | kyc=49.73 | model=GraphSAGE

### Risques
- Niveau consolidé orienté par le pire cas séquentiel: `ELEVE` (proba=0.7206).
- Vérifier la cohérence entre risque transactionnel et risque réseau.

### Recommandation
- Décision proposée: revue manuelle si risque `MODERE/ELEVE`, sinon validation sous surveillance.
- Actions: contrôle KYC, suivi retards, monitoring relationnel.

### Références
- `knowledge_base\ref08_extrait.txt#0` (score=0.386)
- `criteres_score_et_crispdm.md#0` (score=0.077)
- `criteres_score_et_crispdm.md#2` (score=0.074)
- `criteres_score_et_crispdm.md#1` (score=0.049)
- `lstm_gru_next_steps.md#0` (score=0.040)

### Annexes (JSON)

```json
{
  "classic": {
    "prediction": 0,
    "default_proba": 0.0003,
    "risk_level": "FAIBLE",
    "model_used": "Logistic Regression",
    "message": "Explication LLM désactivée (mode rapide).",
    "cin": "88710263",
    "credit_id": 2897,
    "kyc_score": 49.73
  },
  "sequential": {
    "cin": "88710263",
    "kyc_score": 49.73,
    "prediction": 1,
    "default_proba": 0.7206,
    "risk_level": "ELEVE",
    "model_used": "GRU baseline (GRU) (GRU)",
    "message": "Explication LLM désactivée (mode rapide).",
    "credits": [
      {
        "credit_id": 2897,
        "prediction": 1,
        "default_proba": 0.7206,
        "risk_level": "ELEVE"
      },
      {
        "credit_id": 2904,
        "prediction": 1,
        "default_proba": 0.704,
        "risk_level": "ELEVE"
      },
      {
        "credit_id": 2893,
        "prediction": 1,
        "default_proba": 0.69,
        "risk_level": "ELEVE"
      },
      {
        "credit_id": 2895,
        "prediction": 1,
        "default_proba": 0.7075,
        "risk_level": "ELEVE"
      },
      {
        "credit_id": 2900,
        "prediction": 1,
        "default_proba": 0.7144,
        "risk_level": "ELEVE"
      },
      {
        "credit_id": 2902,
        "prediction": 1,
        "default_proba": 0.6966,
        "risk_level": "ELEVE"
      },
      {
        "credit_id": 2903,
        "prediction": 0,
        "default_proba": 0.3803,
        "risk_level": "MODERE"
      },
      {
        "credit_id": 2899,
        "prediction": 0,
        "default_proba": 0.3071,
        "risk_level": "MODERE"
      },
      {
        "credit_id": 2901,
        "prediction": 0,
        "default_proba": 0.2922,
        "risk_level": "FAIBLE"
      },
      {
        "credit_id": 2894,
        "prediction": 0,
        "default_proba": 0.2922,
        "risk_level": "FAIBLE"
      },
      {
        "credit_id": 2898,
        "prediction": 0,
        "default_proba": 0.2922,
        "risk_level": "FAIBLE"
      },
      {
        "credit_id": 2896,
        "prediction": 0,
        "default_proba": 0.2774,
        "risk_level": "FAIBLE"
      }
    ],
    "n_credits": 12
  },
  "graph": {
    "cin": "88710263",
    "kyc_score": 49.73,
    "prediction": 1,
    "default_proba": 0.9826,
    "risk_level": "ELEVE",
    "model_used": "GraphSAGE",
    "message": "Explication LLM désactivée (mode rapide)."
  }
}
```
