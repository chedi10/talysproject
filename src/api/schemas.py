"""
schemas.py – Pydantic models for FastAPI request and response.
"""
from pydantic import BaseModel, Field
from typing import Literal
from typing import Optional


class CreditRequest(BaseModel):
    """
    Input features for a single credit default risk prediction.
    All numeric features must match what was used during training.
    """
    # ── Credit-level features ──────────────────────────────────────────────
    montant:    float = Field(..., gt=0,         example=5000.0,  description="Loan amount (TND)")
    duree_mois: int   = Field(..., ge=1, le=120, example=12,      description="Loan duration in months")
    dti:        float = Field(..., ge=0,  le=2,  example=0.45,    description="Debt-to-Income ratio")

    # Encoded as integers (matching config.py: 0-indexed from the list)
    cycle_enc:      int = Field(..., ge=0, le=3, example=0,  description="Cycle (0=CYCLE_1 … 3=CYCLE_4)")
    objet_enc:      int = Field(..., ge=0, le=4, example=0,  description="Purpose (0=CONSOMMATION … 4=LOGEMENT)")

    # ── Client features ───────────────────────────────────────────────────
    age:            int   = Field(..., ge=18, le=80,    example=35,    description="Client age")
    revenu_mensuel: float = Field(..., ge=0,            example=2000.0, description="Monthly income (TND)")
    profession_enc: int   = Field(..., ge=0, le=7,      example=2,     description="Profession (0=Etudiant … 7=Retraité)")
    kyc_score: Optional[float] = Field(None, ge=0, le=100, description="KYC score 0–100. If omitted, computed from age/revenu/profession.")

    # ── Repayment behaviour features ─────────────────────────────────────
    avg_retard:   float = Field(0.0, ge=0, example=3.2,  description="Average payment delay (days)")
    max_retard:   float = Field(0.0, ge=0, example=10.0, description="Max payment delay (days)")
    std_retard:   float = Field(0.0, ge=0, example=2.1,  description="Std dev of payment delays")
    n_payments:   int   = Field(0,   ge=0, example=12,   description="Number of monthly payments recorded")
    n_late:       int   = Field(0,   ge=0, example=1,    description="Count of late payments")
    pct_late:     float = Field(0.0, ge=0, le=1, example=0.08, description="Fraction of late payments")
    n_en_retard:  int   = Field(0,   ge=0, example=0,    description="Payments with delay ≥ 90 days")

    # ── Transaction features ──────────────────────────────────────────────
    n_transactions:     int   = Field(0,   ge=0, example=70,   description="Total transactions by client")
    n_suspect:          int   = Field(0,   ge=0, example=0,    description="Flagged suspicious transactions")
    avg_tx_amount:      float = Field(0.0, ge=0, example=800.0,description="Average transaction amount")
    total_depot:        float = Field(0.0, ge=0, example=15000.0, description="Total deposits (TND)")
    total_retrait:      float = Field(0.0, ge=0, example=9000.0,  description="Total withdrawals (TND)")
    total_remboursement:float = Field(0.0, ge=0, example=5000.0,  description="Total repayment transactions")
    total_transfert:    float = Field(0.0, ge=0, example=4000.0,  description="Total transfers")
    ratio_retrait_depot:float = Field(0.0, ge=0, example=0.6,     description="Withdrawal / Deposit ratio")

    # ── Relation/graph features ───────────────────────────────────────────
    max_risk_relation: float = Field(0.0, ge=0, le=100, example=40.0, description="Highest risk on any relation edge")
    avg_risk_relation: float = Field(0.0, ge=0, le=100, example=30.0, description="Average relation risk")
    n_relations:       int   = Field(0,   ge=0,          example=3,    description="Total number of client relations")
    n_garant:          int   = Field(0,   ge=0,          example=0,    description="Number of GARANT relationships")

    class Config:
        json_schema_extra = {
            "example": {
                "montant": 5000.0,
                "duree_mois": 12,
                "dti": 0.45,
                "cycle_enc": 0,
                "objet_enc": 0,
                "age": 35,
                "revenu_mensuel": 2000.0,
                "profession_enc": 2,
                "avg_retard": 3.2,
                "max_retard": 10.0,
                "std_retard": 2.1,
                "n_payments": 12,
                "n_late": 1,
                "pct_late": 0.08,
                "n_en_retard": 0,
                "n_transactions": 70,
                "n_suspect": 0,
                "avg_tx_amount": 800.0,
                "total_depot": 15000.0,
                "total_retrait": 9000.0,
                "total_remboursement": 5000.0,
                "total_transfert": 4000.0,
                "ratio_retrait_depot": 0.6,
                "max_risk_relation": 40.0,
                "avg_risk_relation": 30.0,
                "n_relations": 3,
                "n_garant": 0,
            }
        }


class PredictionResponse(BaseModel):
    """API response for a single prediction."""
    prediction:    int   = Field(..., description="0 = Non-défaut  |  1 = Défaut")
    default_proba: float = Field(..., description="Probability of default (0..1)")
    risk_level:    Literal["FAIBLE", "MODERE", "ELEVE"] = Field(
        ..., description="Risk tier: FAIBLE < 30% | MODERE 30–60% | ELEVE > 60%"
    )
    model_used:    str   = Field(..., description="Name of the model that made the prediction")


class PredictionByCinResponse(PredictionResponse):
    """
    Same as PredictionResponse, plus CIN, chosen credit_id, and KYC score.
    """
    cin: str = Field(..., description="Client CIN used for lookup")
    credit_id: int = Field(..., description="Credit ID used to compute the score")
    kyc_score: float = Field(..., ge=0, le=100, description="Score KYC (0–100) estimé pour ce client")


class ExplanationResponse(BaseModel):
    """
    Response combining the raw prediction and a natural-language explanation
    generated by a Large Language Model (LLM).
    """
    prediction:    int   = Field(..., description="0 = Non-défaut  |  1 = Défaut")
    default_proba: float = Field(..., description="Probability of default (0..1)")
    risk_level:    Literal["FAIBLE", "MODERE", "ELEVE"] = Field(
        ..., description="Risk tier: FAIBLE < 30% | MODERE 30–60% | ELEVE > 60%"
    )
    model_used:    str   = Field(..., description="Name of the model that made the prediction")
    message:       str   = Field(..., description="French explanation of why the profile is risky or not")


class CinRequest(BaseModel):
    """
    Request model to score/explain a client by CIN.
    If credit_id is not provided, the API will pick the most recent credit for this CIN.
    """
    cin: str = Field(..., min_length=6, max_length=32, example="01234567", description="Client CIN (from clients.csv)")
    credit_id: Optional[int] = Field(None, ge=1, example=123, description="Optional credit_id to use for this CIN")


class ExplanationByCinResponse(ExplanationResponse):
    """
    REF-08 §1.1.5 – Sorties des modèles :
    • Score KYC | Score de risque crédit | Probabilité de défaut | Explication (XAI)
    """
    cin: str = Field(..., description="Client CIN used for lookup")
    credit_id: int = Field(..., description="Credit ID used to compute the score")
    kyc_score: float = Field(..., ge=0, le=100, description="Score KYC (0–100) estimé pour ce client")


class CreditExplanationItem(BaseModel):
    """Per-credit score item for multi-credit CIN responses."""
    credit_id: int = Field(..., description="Credit ID scored")
    prediction: int = Field(..., description="0 = Non-défaut  |  1 = Défaut")
    default_proba: float = Field(..., description="Probability of default (0..1)")
    risk_level: Literal["FAIBLE", "MODERE", "ELEVE"] = Field(..., description="Risk tier from default_proba")


class SequentialByCinResponse(BaseModel):
    """
    Sequential (LSTM/GRU) prediction response using only CIN as input.
    REF-08 §1.1.5 : inclut Score KYC.
    """
    cin: str = Field(..., description="Client CIN used for lookup")
    kyc_score: float = Field(..., ge=0, le=100, description="Score KYC (0–100) estimé pour ce client")
    prediction: int = Field(..., description="0 = Non-défaut  |  1 = Défaut")
    default_proba: float = Field(..., description="Probability of default (0..1)")
    risk_level: Literal["FAIBLE", "MODERE", "ELEVE"] = Field(
        ..., description="Risk tier: FAIBLE < 30% | MODERE 30–60% | ELEVE > 60%"
    )
    model_used: str = Field(..., description="Sequential model name (LSTM/GRU)")


class SequentialExplanationByCinResponse(SequentialByCinResponse):
    """
    Sequential prediction + natural-language explanation from local LLM.
    """
    message: str = Field(..., description="French explanation for sequential risk result")


class SequentialExplanationByCinAllCreditsResponse(SequentialExplanationByCinResponse):
    """
    Professional mode (sequential): if client has multiple credits, score ALL credits and
    return the worst-case summary + per-credit list.
    """
    credits: list[CreditExplanationItem] = Field(..., description="Per-credit sequential scores for this CIN")
    n_credits: int = Field(..., ge=1, description="Number of credits considered for this CIN")


class GraphByCinResponse(BaseModel):
    """
    Graph (GNN/GraphSAGE) prediction response using only CIN as input.
    REF-08 §1.1.4 + §1.1.5 : inclut Score KYC.
    """
    cin: str = Field(..., description="Client CIN used for lookup")
    kyc_score: float = Field(..., ge=0, le=100, description="Score KYC (0–100) estimé pour ce client")
    prediction: int = Field(..., description="0 = Non-défaut  |  1 = Défaut")
    default_proba: float = Field(..., description="Probability of default (0..1)")
    risk_level: Literal["FAIBLE", "MODERE", "ELEVE"] = Field(
        ..., description="Risk tier: FAIBLE < 30% | MODERE 30–60% | ELEVE > 60%"
    )
    model_used: str = Field(..., description="Graph model name (GraphSAGE)")


class GraphExplanationByCinResponse(GraphByCinResponse):
    """
    Graph prediction + natural-language explanation from local LLM.
    """
    message: str = Field(..., description="French explanation for graph risk result")


class HealthResponse(BaseModel):
    status:     str
    model_name: str
    version:    str = "1.0.0"
