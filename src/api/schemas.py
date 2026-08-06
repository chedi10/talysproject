"""
schemas.py – Pydantic models for FastAPI request and response.
"""
from pydantic import BaseModel, Field
from typing import Any, Literal
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


class ShapFeatureImpact(BaseModel):
    feature: str
    label: str
    impact: float = Field(..., description="Positive = augmente le risque de défaut")


class ShapDriverDetail(BaseModel):
    feature: str
    label: str
    impact: float
    value: Optional[float] = None
    portfolio_median: Optional[float] = None
    vs_portfolio_pct: Optional[float] = None


class CreditContext(BaseModel):
    montant: Optional[float] = None
    duree_mois: Optional[int] = None
    dti: Optional[float] = None
    objet: Optional[str] = None
    cycle: Optional[str] = None


class ShapExplanation(BaseModel):
    method: str
    model_used: str
    increases_risk: list[ShapFeatureImpact]
    decreases_risk: list[ShapFeatureImpact]
    summary: str
    base_prediction: Optional[float] = None
    driver_details: list[ShapDriverDetail] = Field(default_factory=list)
    credit_context: CreditContext = Field(default_factory=CreditContext)


class BusinessRuleResult(BaseModel):
    rule_id: str
    name: str
    triggered: bool
    severity: Literal["INFO", "WARNING", "CRITICAL"]
    action: Literal["none", "alert", "manual_review", "block"]
    message: str
    value: Optional[float | int | str | dict] = None
    threshold: Optional[float | int | str | dict] = None
    policy_ref: str = ""


class BusinessRulesBlock(BaseModel):
    rules: list[BusinessRuleResult]
    triggered_count: int
    triggered_rule_ids: list[str] = Field(default_factory=list)
    requires_manual_review: bool
    highest_severity: Literal["INFO", "WARNING", "CRITICAL"]
    compliance_score: int = Field(100, ge=0, le=100)
    summary: str
    credit_snapshot: dict = Field(default_factory=dict)


class EarlyWarningAlert(BaseModel):
    code: str
    severity: Literal["INFO", "WARNING", "CRITICAL"]
    message: str
    metric: str
    current: float | int | str
    baseline: Optional[float | int | str] = None


class TrendPoint(BaseModel):
    credit_id: int
    date: str = ""
    value: float
    is_current: bool = False


class TrendSeries(BaseModel):
    metric: str
    label: str
    points: list[TrendPoint]


class EarlyWarningBlock(BaseModel):
    alerts: list[EarlyWarningAlert]
    alert_count: int
    critical_count: int
    degradation_detected: bool
    summary: str
    watchlist_priority: Literal["NONE", "LOW", "MEDIUM", "HIGH"] = "NONE"
    trend_series: list[TrendSeries] = Field(default_factory=list)
    n_credits_historique: int = 0


class ContributingFactors(BaseModel):
    rules: list[str] = Field(default_factory=list)
    ews: list[str] = Field(default_factory=list)
    compliance_score: int = 100


class AiRecommendation(BaseModel):
    decision: Literal[
        "ACCEPTER",
        "ACCEPTER_AVEC_GARANTIE",
        "REDUIRE_MONTANT",
        "DEMANDER_GARANT",
        "REFUSER",
    ]
    decision_label: str
    confidence: float = Field(..., ge=0, le=1)
    justification: str
    recommended_actions: list[str]
    requires_manual_validation: bool
    suggested_montant: Optional[float] = None
    montant_reduction_pct: Optional[float] = None
    suggested_dti_target: Optional[float] = None
    monitoring_frequency: str = "trimestriel"
    conditions: list[str] = Field(default_factory=list)
    contributing_factors: ContributingFactors = Field(default_factory=ContributingFactors)


class ClientProfile(BaseModel):
    cin: str
    client_id: int
    nom: str = ""
    prenom: str = ""
    age: int = 0
    ville: str = ""
    profession: str = ""
    revenu_mensuel: float = 0
    statut_kyc: str = ""


class SystemClientContext(BaseModel):
    """Contexte client pour systèmes décisionnels — score institutionnel, sans modèle ML."""
    cin: str
    credit_id: int
    kyc_score: float
    institutional_score: float = Field(..., ge=0, le=1, description="Score de risque institutionnel (données métier)")
    risk_level: Literal["FAIBLE", "MODERE", "ELEVE"]
    risk_factors: list[str] = Field(default_factory=list)
    client_profile: ClientProfile = Field(default_factory=lambda: ClientProfile(cin="", client_id=0))
    credit_snapshot: dict = Field(default_factory=dict)


class ShapSystemResponse(SystemClientContext):
    system: Literal["shap"] = "shap"
    shap: ShapExplanation


class RulesSystemResponse(SystemClientContext):
    system: Literal["business_rules"] = "business_rules"
    business_rules: BusinessRulesBlock


class EarlyWarningSystemResponse(SystemClientContext):
    system: Literal["early_warning"] = "early_warning"
    early_warning: EarlyWarningBlock


class RecommendationSystemResponse(SystemClientContext):
    system: Literal["recommendation"] = "recommendation"
    recommendation: AiRecommendation


class ExplanationByCinResponse(ExplanationResponse):
    """
    REF-08 §1.1.5 – Sorties du modèle classique uniquement (score + LLM).
    Les systèmes SHAP / règles / EWS / recommandation sont sur /systems/*.
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


class GraphNetworkNode(BaseModel):
    id: str
    client_id: int
    cin: str = ""
    label: str
    ville: str = ""
    is_center: bool = False
    en_defaut: bool | None = None
    default_proba: float | None = None
    risk_level: Literal["FAIBLE", "MODERE", "ELEVE"] | None = None


class GraphNetworkEdge(BaseModel):
    id: str
    source: str
    target: str
    type_relation: str
    risk_relation: int = 0
    color: str = "#94A3B8"
    label: str = ""
    intra: bool = False


class GraphNetworkSnapshot(BaseModel):
    nodes: list[GraphNetworkNode] = Field(default_factory=list)
    edges: list[GraphNetworkEdge] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    legend: list[dict[str, str]] = Field(default_factory=list)


class GraphExplanationByCinResponse(GraphByCinResponse):
    """
    Graph prediction + natural-language explanation from local LLM.
    """
    message: str = Field(..., description="French explanation for graph risk result")
    network: GraphNetworkSnapshot | None = Field(
        None, description="Ego-network visualization data (client + neighbors)"
    )


class EnsembleModelScore(BaseModel):
    model_key: Literal["classic", "sequential", "graph"]
    model_name: str
    weight: float = Field(..., ge=0, le=1, description="Poids normalisé dans l'ensemble (basé AUC)")
    available: bool = True
    default_proba: float | None = None
    risk_level: Literal["FAIBLE", "MODERE", "ELEVE"] | None = None
    prediction: int | None = None
    error: str | None = None


class EnsembleExplanationByCinResponse(BaseModel):
    """Score unifié — fusion pondérée Deep Tabular + GAT + Transformer."""
    cin: str
    credit_id: int | None = Field(None, description="Crédit utilisé pour le score tabulaire")
    kyc_score: float = Field(..., ge=0, le=100)
    default_proba: float = Field(..., description="Probabilité ensemble (pondération AUC)")
    risk_level: Literal["FAIBLE", "MODERE", "ELEVE"]
    prediction: int
    model_used: str
    method: str = Field("weighted_auc", description="Méthode de fusion")
    models: list[EnsembleModelScore]
    vote_default: int = Field(..., ge=0, description="Modèles votant « défaut »")
    vote_non_default: int = Field(..., ge=0, description="Modèles votant « non-défaut »")
    agreement: Literal["unanimous", "majority", "split"]
    models_available: int
    models_total: int = 3
    message: str
    network: GraphNetworkSnapshot | None = None


class HealthResponse(BaseModel):
    status:     str
    model_name: str
    version:    str = "1.0.0"


class ChatRequest(BaseModel):
    """
    Agent conversationnel (REF-08 §1.2).
    Provide a user message (free text) + a session_id for conversational memory.
    """
    session_id: str = Field(..., min_length=3, max_length=64, example="demo-1")
    message: str = Field(..., min_length=1, max_length=4000, example="Analyse le CIN 88710263 en séquentiel.")


class RagSourceItem(BaseModel):
    source: str
    chunk_id: int = 0
    score: float = 0.0
    text: str = ""


class ChatStructuredResult(BaseModel):
    kyc_score: Optional[float] = None
    default_proba: Optional[float] = None
    risk_level: Optional[str] = None
    model_used: Optional[str] = None
    institutional_score: Optional[float] = None
    institutional_risk: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    model_selected: Literal["classic", "sequential", "graph"] | None = None
    cin: str | None = None
    intent: Literal[
        "classic_score", "sequential_score", "graph_score",
        "full_report", "compare_models", "institutional",
    ] | None = None
    answer: str
    rag_sources: list[RagSourceItem] = Field(default_factory=list)
    structured: Optional[ChatStructuredResult] = None
    systems: Optional[dict] = None
    report_available: bool = False
    suggested_prompts: list[str] = Field(default_factory=list)


class ReportRequest(BaseModel):
    """Generate a Markdown report from CIN using RAG."""
    cin: str = Field(..., min_length=6, max_length=32, example="88710263")


class ReportResponse(BaseModel):
    cin: str
    markdown: str
    sources: list[dict]
    structured: Optional[dict] = None


class ReportDownloadRequest(BaseModel):
    """Generate and download a report file from CIN."""
    cin: str = Field(..., min_length=6, max_length=32, example="88710263")
    format: Literal["md", "pdf"] = Field("md", description="Export format")


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32, example="agent1")
    email: str = Field(..., min_length=5, max_length=120, example="agent1@talys.local")
    password: str = Field(..., min_length=6, max_length=128, example="agent123")
    role: Literal["client", "agent"] = Field("agent", description="Rôle du compte (admin créé par un admin)")
    cin: Optional[str] = Field(None, min_length=6, max_length=32, description="CIN obligatoire pour role=client")


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32, example="agent1")
    password: str = Field(..., min_length=6, max_length=128, example="agent123")


class UserPublic(BaseModel):
    id: str
    username: str
    email: str
    role: Literal["client", "agent", "admin"]
    cin: Optional[str] = None
    client_id: Optional[int] = None
    created_at: str | None = None


class AdminCreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    email: str = Field(..., min_length=5, max_length=120)
    password: str = Field(..., min_length=6, max_length=128)
    role: Literal["client", "agent", "admin"] = "agent"
    cin: Optional[str] = None


class ClientCreditSummary(BaseModel):
    total: int = 0
    actifs: int = 0
    en_defaut: int = 0
    montant_total: float = 0
    dti_moyen: float = 0


class ClientAlert(BaseModel):
    level: Literal["info", "warning", "danger"] = "info"
    title: str
    message: str


class ClientProfileResponse(BaseModel):
    cin: str
    client_id: int
    nom: str
    prenom: str
    age: int
    ville: str
    profession: str
    revenu_mensuel: float
    statut_kyc: str
    kyc_score: float = 0
    credits: list[dict]
    credit_summary: ClientCreditSummary = Field(default_factory=ClientCreditSummary)
    alerts: list[ClientAlert] = Field(default_factory=list)
    sante_dossier: Literal["EXCELLENT", "BON", "A_SURVEILLER", "FRAGILE"] = "BON"
    prochaine_echeance: str | None = None
    taux_retard: float = 0


class ClientChatRequest(BaseModel):
    session_id: str = Field(default="client-assistant", min_length=3, max_length=64)
    message: str = Field(..., min_length=1, max_length=2000, example="Résume mon dossier complet")


class ClientChatResponse(BaseModel):
    session_id: str
    intent: Literal[
        "summary", "profile", "credits", "kyc", "payments", "alerts", "contact",
        "demarche_credit", "demarche_kyc", "solution_retard", "solution_defaut",
        "documents", "faq", "aide", "general",
    ]
    answer: str
    rag_sources: list[RagSourceItem] = Field(default_factory=list)
    suggested_prompts: list[str] = Field(default_factory=list)


class SystemStatsResponse(BaseModel):
    database: str
    clients: int
    credits: int
    transactions: int
    users: int
    activity_log: int
    model_loaded: bool
    model_name: str
    relations: int = 0
    remboursements: int = 0
    chat_sessions: int = 0
    credits_en_defaut: int = 0
    default_rate: float = 0
    activity_last_7_days: int = 0
    users_by_role: dict[str, int] = Field(default_factory=dict)
    kyc_breakdown: dict[str, int] = Field(default_factory=dict)
    activity_by_action: dict[str, int] = Field(default_factory=dict)
    graph_model: str | None = None
    graph_auc: float | None = None


class AuthResponse(BaseModel):
    token: str
    user: UserPublic


class ActivityRecord(BaseModel):
    id: str
    user_id: str
    username: str
    role: str
    action: str
    cin: str | None = None
    model: str | None = None
    intent: str | None = None
    message_preview: str | None = None
    session_id: str | None = None
    created_at: str


class ActivityListResponse(BaseModel):
    items: list[ActivityRecord]
    total: int
    scope: Literal["mine", "all"]


class ChatSessionSummary(BaseModel):
    user_id: str
    username: str
    session_id: str
    message_count: int
    updated_at: str | None = None
    created_at: str | None = None
    title: str | None = None
    last_cin: str | None = None
    last_intent: str | None = None
    last_preview: str | None = None


class ChatSessionListResponse(BaseModel):
    items: list[ChatSessionSummary]
    scope: Literal["mine", "all"]


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[dict[str, str]]
    scope: Literal["mine", "all"]
