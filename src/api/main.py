"""
main.py – FastAPI application for credit default risk prediction.

Run with:
    uvicorn src.api.main:app --reload --port 8000

Then open: http://localhost:8000/docs
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.config import (
    BEST_MODEL_FILE,
    MODEL_METADATA_FILE,
    FEATURES_FILE,
    RAW_CLIENTS,
    RAW_CREDITS,
    RAW_REMBOURSEMENTS,
    RAW_TRANSACTIONS,
    MODELS_DIR,
)
from src.api.schemas import (
    CreditRequest,
    PredictionResponse,
    PredictionByCinResponse,
    HealthResponse,
    ExplanationResponse,
    CinRequest,
    ExplanationByCinResponse,
    CreditExplanationItem,
    SequentialByCinResponse,
    SequentialExplanationByCinResponse,
    SequentialExplanationByCinAllCreditsResponse,
    GraphByCinResponse,
    GraphExplanationByCinResponse,
)
from src.llm.client import generate_risk_explanation
from src.kyc.score import compute_kyc_score, compute_kyc_score_row

# Load environment variables from .env (local settings)
load_dotenv()

# ─── Load model and metadata at startup ──────────────────────────────────────
_model = None
_metadata: dict = {}


def _load_artifacts():
    global _model, _metadata
    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found at {BEST_MODEL_FILE}. "
            "Run `python -m src.models.train` first."
        )
    _model = joblib.load(BEST_MODEL_FILE)
    if MODEL_METADATA_FILE.exists():
        with open(MODEL_METADATA_FILE) as f:
            _metadata = json.load(f)


_load_artifacts()

# ─── Lightweight lookup tables for CIN → credit_id → features ────────────────
_clients_df: pd.DataFrame | None = None
_credits_df: pd.DataFrame | None = None
_features_df: pd.DataFrame | None = None
_tx_df: pd.DataFrame | None = None
_remb_df: pd.DataFrame | None = None
_tx_amount_max: float = 1.0
_remb_amount_max: float = 1.0


def _load_lookup_tables():
    """
    Load small tables needed for CIN-based scoring:
    - clients.csv: cin → client_id
    - credits.csv: client_id → credit_id (+ date_debut for picking most recent)
    - features.parquet: credit_id → feature row used by the model
    """
    global _clients_df, _credits_df, _features_df, _tx_df, _remb_df, _tx_amount_max, _remb_amount_max
    if _clients_df is None:
        _clients_df = pd.read_csv(RAW_CLIENTS, dtype={"cin": str})
    if _credits_df is None:
        _credits_df = pd.read_csv(RAW_CREDITS, parse_dates=["date_debut"])
    if _tx_df is None:
        _tx_df = pd.read_csv(RAW_TRANSACTIONS, parse_dates=["date"])
        _tx_df["client_id"] = _tx_df["client_id"].astype(int)
        _tx_df["montant"] = _tx_df["montant"].astype(float)
        _tx_df["suspect"] = _tx_df["suspect"].astype(int)
        _tx_amount_max = float(_tx_df["montant"].max()) if len(_tx_df) else 1.0
    if _remb_df is None:
        _remb_df = pd.read_csv(RAW_REMBOURSEMENTS, parse_dates=["date_echeance", "date_paiement"])
        _remb_df["client_id"] = _remb_df["client_id"].astype(int)
        _remb_df["montant_du"] = _remb_df["montant_du"].astype(float)
        _remb_df["retard_jours"] = _remb_df["retard_jours"].astype(int)
        _remb_amount_max = float(_remb_df["montant_du"].max()) if len(_remb_df) else 1.0
    if _features_df is None:
        if not FEATURES_FILE.exists():
            raise FileNotFoundError(
                f"Feature matrix not found at {FEATURES_FILE}. "
                "Run `python -m src.features.engineering` first."
            )
        _features_df = pd.read_parquet(FEATURES_FILE)


def _normalize_cin(cin: str) -> str:
    return "".join(str(cin).split()).strip()

def _risk_level_from_proba(proba: float) -> str:
    if proba < 0.30:
        return "FAIBLE"
    if proba < 0.60:
        return "MODERE"
    return "ELEVE"


# ─── Sequential model (LSTM/GRU) artifacts ───────────────────────────────────
_seq_model = None
_seq_device = None
_seq_seq_len = 30
_seq_input_dim = 6
_seq_model_name = "Sequential"

# ─── Graph model (GraphSAGE) artifacts ───────────────────────────────────────
_graph_model = None
_graph_edge_index = None
_graph_x = None
_graph_client_id_to_idx: dict[int, int] | None = None
_graph_model_name = "GraphSAGE"


def _load_graph_artifacts():
    global _graph_model, _graph_edge_index, _graph_x, _graph_client_id_to_idx, _graph_model_name
    if _graph_model is not None:
        return

    try:
        import torch
    except ImportError as e:
        raise HTTPException(status_code=500, detail="PyTorch is not installed in this environment.") from e

    ckpt_path = MODELS_DIR / "graphsage.pt"
    if not ckpt_path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"GraphSAGE checkpoint not found at {ckpt_path}. "
                "Run `python -m src.models.graph.train` first."
            ),
        )

    from src.models.graph.model import GraphSAGEClassifier
    from src.models.graph.data import build_graph_dataset

    ds = build_graph_dataset()
    ckpt = torch.load(ckpt_path, map_location="cpu")

    in_dim = int(ckpt.get("in_dim", ds.x.shape[1]))
    hidden_dim = int(ckpt.get("hidden_dim", 64))
    dropout = float(ckpt.get("dropout", 0.2))

    model = GraphSAGEClassifier(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    state = ckpt.get("state_dict")
    if state:
        model.load_state_dict(state)
    model.eval()

    _graph_model = model
    _graph_edge_index = torch.tensor(ds.edge_index, dtype=torch.long)
    _graph_x = torch.tensor(ds.x, dtype=torch.float32)
    _graph_client_id_to_idx = ds.client_id_to_idx
    _graph_model_name = "GraphSAGE"


def _load_sequential_artifacts():
    global _seq_model, _seq_device, _seq_seq_len, _seq_input_dim, _seq_model_name
    if _seq_model is not None:
        return

    try:
        import torch
    except ImportError as e:
        raise HTTPException(status_code=500, detail="PyTorch is not installed in this environment.") from e

    from src.models.sequential.model import RecurrentCreditRiskModel

    meta_path = MODELS_DIR / "sequential_metadata.json"
    ckpt_path = MODELS_DIR / "sequential_lstm.pt"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        best = meta.get("best", {})
        artifact = best.get("artifact")
        if artifact:
            ckpt_path = Path(artifact)
            _seq_model_name = best.get("model_name", "Sequential")

    if not ckpt_path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Sequential checkpoint not found at {ckpt_path}. "
                "Run `python -m src.models.sequential.train` first."
            ),
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    _seq_input_dim = int(ckpt.get("input_dim", 6))
    _seq_seq_len = int(ckpt.get("seq_len", 30))
    rnn_type = str(ckpt.get("rnn_type", "lstm"))
    _seq_model_name = f"{_seq_model_name} ({rnn_type.upper()})"

    model = RecurrentCreditRiskModel(
        input_dim=_seq_input_dim,
        hidden_dim=64,
        num_layers=1,
        dropout=0.2,
        rnn_type=rnn_type,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _seq_model = model
    _seq_device = torch.device("cpu")


def _tx_to_vec(row: pd.Series) -> np.ndarray:
    # Must match src.models.sequential.data vector layout
    types = ["DEPOT", "RETRAIT", "REMBOURSEMENT", "TRANSFERT"]
    remb_status = ["PAYE", "EN_RETARD"]

    one_hot_tx = np.zeros(4, dtype=np.float32)
    t = str(row.get("type", ""))
    if t in types:
        one_hot_tx[types.index(t)] = 1.0

    amount_scaled = float(row["montant"]) / max(_tx_amount_max, 1.0)
    base = np.array([amount_scaled, float(row["suspect"]), 0.0, 0.0], dtype=np.float32)
    remb_oh = np.zeros(len(remb_status), dtype=np.float32)
    return np.concatenate([base, remb_oh, one_hot_tx], axis=0)


def _remb_to_vec(row: pd.Series) -> np.ndarray:
    types = ["DEPOT", "RETRAIT", "REMBOURSEMENT", "TRANSFERT"]
    remb_status = ["PAYE", "EN_RETARD"]

    one_hot_tx = np.zeros(len(types), dtype=np.float32)
    statut = str(row.get("statut", "PAYE"))
    if int(row.get("retard_jours", 0)) > 0 and statut != "PAYE":
        statut = "EN_RETARD"
    remb_oh = np.zeros(len(remb_status), dtype=np.float32)
    remb_oh[remb_status.index(statut) if statut in remb_status else 0] = 1.0

    amount_scaled = float(row["montant_du"]) / max(_remb_amount_max, 1.0)
    retard_scaled = float(row.get("retard_jours", 0)) / 365.0
    base = np.array([amount_scaled, 0.0, 1.0, retard_scaled], dtype=np.float32)
    return np.concatenate([base, remb_oh, one_hot_tx], axis=0)


def _build_seq_for_client(client_id: int) -> np.ndarray:
    assert _tx_df is not None and _remb_df is not None

    tx = _tx_df[_tx_df["client_id"] == client_id].copy()
    remb = _remb_df[_remb_df["client_id"] == client_id].copy()

    events: list[tuple[pd.Timestamp, np.ndarray]] = []
    if len(tx):
        tx = tx.sort_values("date")
        events += [(d, _tx_to_vec(r)) for d, (_, r) in zip(tx["date"], tx.iterrows())]
    if len(remb):
        remb = remb.sort_values("date_echeance")
        events += [(d, _remb_to_vec(r)) for d, (_, r) in zip(remb["date_echeance"], remb.iterrows())]

    events = sorted(events, key=lambda t: t[0])
    seq = [v for _, v in events]

    arr = np.zeros((_seq_seq_len, _seq_input_dim), dtype=np.float32)
    if seq:
        tail = seq[-_seq_seq_len :]
        arr[-len(tail) :, :] = np.stack(tail)
    return arr


def _build_seq_for_client_credit(client_id: int, credit_start: pd.Timestamp) -> np.ndarray:
    """
    Build a sequential input using only events up to the credit start date.
    This enables per-credit sequential scoring for clients with multiple credits.
    """
    assert _tx_df is not None and _remb_df is not None

    tx = _tx_df[_tx_df["client_id"] == client_id].copy()
    remb = _remb_df[_remb_df["client_id"] == client_id].copy()

    events: list[tuple[pd.Timestamp, np.ndarray]] = []
    if len(tx):
        tx = tx[tx["date"] <= credit_start].sort_values("date")
        events += [(d, _tx_to_vec(r)) for d, (_, r) in zip(tx["date"], tx.iterrows())]
    if len(remb):
        remb = remb[remb["date_echeance"] <= credit_start].sort_values("date_echeance")
        events += [(d, _remb_to_vec(r)) for d, (_, r) in zip(remb["date_echeance"], remb.iterrows())]

    events = sorted(events, key=lambda t: t[0])
    seq = [v for _, v in events]

    arr = np.zeros((_seq_seq_len, _seq_input_dim), dtype=np.float32)
    if seq:
        tail = seq[-_seq_seq_len :]
        arr[-len(tail) :, :] = np.stack(tail)
    return arr


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Default Risk API",
    description=(
        "Predicts the probability that a microfinance credit will default. "
        "Built with scikit-learn / XGBoost, served with FastAPI."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Returns API health status and loaded model name."""
    return HealthResponse(
        status="ok",
        model_name=_metadata.get("best_model_name", "unknown"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"], include_in_schema=False)
def predict(request: CreditRequest):
    """
    Predict default risk for a single credit application.

    Returns:
    - **prediction**: 0 (non-default) or 1 (default)
    - **default_proba**: probability between 0 and 1
    - **risk_level**: FAIBLE / MODERE / ELEVE
    - **model_used**: name of the model
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Build feature vector in the exact order used during training
    feature_order = _metadata.get("feature_columns", _default_feature_order())
    input_dict = request.model_dump()

    # Compute kyc_score if not provided (REF-08: estimate KYC from client attributes)
    if input_dict.get("kyc_score") is None:
        input_dict["kyc_score"] = compute_kyc_score(
            age=input_dict["age"],
            revenu_mensuel=input_dict["revenu_mensuel"],
            profession_enc=input_dict["profession_enc"],
        )

    try:
        X = np.array([[input_dict[col] for col in feature_order]], dtype=float)
    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Missing feature: {e}. Expected columns: {feature_order}",
        )

    proba = float(_model.predict_proba(X)[0][1])
    prediction = int(proba >= 0.5)

    if proba < 0.30:
        risk_level = "FAIBLE"
    elif proba < 0.60:
        risk_level = "MODERE"
    else:
        risk_level = "ELEVE"

    return PredictionResponse(
        prediction=prediction,
        default_proba=round(proba, 4),
        risk_level=risk_level,
        model_used=_metadata.get("best_model_name", "unknown"),
    )


@app.post("/predict/batch", tags=["Prediction"], include_in_schema=False)
def predict_batch(requests: list[CreditRequest]):
    """
    Predict default risk for multiple credits in one call.
    Returns a list of PredictionResponse objects.
    """
    return [predict(r) for r in requests]


@app.post("/predict/by-cin", response_model=PredictionByCinResponse, tags=["Prediction"], include_in_schema=False)
def predict_by_cin(payload: CinRequest):
    """
    Predict default risk for the classic (tabular) model using only CIN (+ optional credit_id).
    """
    _load_lookup_tables()
    assert _clients_df is not None and _credits_df is not None and _features_df is not None

    cin = _normalize_cin(payload.cin)

    # 1) Find client_id
    match = _clients_df[_clients_df["cin"].astype(str) == cin]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"CIN not found: {cin}")
    client_id = int(match.iloc[0]["client_id"])

    # 2) Select credit_id
    client_credits = _credits_df[_credits_df["client_id"] == client_id].copy()
    if client_credits.empty:
        raise HTTPException(status_code=404, detail=f"No credits found for CIN: {cin}")

    if payload.credit_id is not None:
        credit_id = int(payload.credit_id)
        if credit_id not in set(client_credits["credit_id"].astype(int).tolist()):
            raise HTTPException(
                status_code=404,
                detail=f"credit_id {credit_id} does not belong to CIN {cin}",
            )
    else:
        client_credits = client_credits.sort_values("date_debut", ascending=False)
        credit_id = int(client_credits.iloc[0]["credit_id"])

    # 3) Fetch feature row for that credit_id
    feat_row = _features_df[_features_df["credit_id"] == credit_id]
    if feat_row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Features not found for credit_id {credit_id}. Rebuild features.parquet.",
        )

    feature_order = _metadata.get("feature_columns", _default_feature_order())
    row = feat_row.iloc[0].to_dict()

    try:
        req_dict = {k: float(row[k]) for k in feature_order}
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature mismatch: missing {e} in features.parquet",
        )

    # Use existing /predict logic (it will compute kyc_score if missing)
    credit_request = CreditRequest(**req_dict)
    base = predict(credit_request)

    kyc_score_val = float(row.get("kyc_score", credit_request.kyc_score or 0.0))

    return PredictionByCinResponse(
        cin=cin,
        credit_id=credit_id,
        kyc_score=round(float(kyc_score_val), 2),
        prediction=base.prediction,
        default_proba=base.default_proba,
        risk_level=base.risk_level,
        model_used=base.model_used,
    )


@app.post("/explain", response_model=ExplanationResponse, tags=["Prediction", "LLM"])
def explain_risk(request: CreditRequest):
    """
    Same inputs as /predict, but returns in plus:
    - un message en langage naturel généré par un LLM local (Ollama)
      qui explique pourquoi le profil est risqué ou non.
    """
    base = predict(request)

    # Construire un petit résumé de features à envoyer au LLM.
    # On ne passe que quelques champs lisibles pour éviter un prompt trop long.
    # Richer summary for classic scoring explanation (uses engineered features)
    features_for_llm = {
        "Montant du crédit": request.montant,
        "Durée (mois)": request.duree_mois,
        "DTI": request.dti,
        "Revenu mensuel": request.revenu_mensuel,
        "Âge": request.age,
        "Score KYC": round(request.kyc_score, 2) if request.kyc_score is not None else None,
        "Probabilité de défaut": base.default_proba,
        "Niveau de risque": base.risk_level,
        # Repayment behaviour
        "Retard moyen (jours)": round(float(request.avg_retard), 2),
        "Retard max (jours)": round(float(request.max_retard), 2),
        "Paiements (n)": int(request.n_payments),
        "Paiements en retard (n)": int(request.n_late),
        "Paiements sévères (>=90j) (n)": int(request.n_en_retard),
        "Taux de retard": round(float(request.pct_late), 3),
        # Transactions behaviour
        "Transactions (n)": int(request.n_transactions),
        "Transactions suspectes (n)": int(request.n_suspect),
        "Montant moyen transaction": round(float(request.avg_tx_amount), 2),
        "Total dépôts": round(float(request.total_depot), 2),
        "Total retraits": round(float(request.total_retrait), 2),
        "Total transferts": round(float(request.total_transfert), 2),
        "Ratio retrait/dépôt": round(float(request.ratio_retrait_depot), 3),
        # Network behaviour
        "Relations (n)": int(request.n_relations),
        "Garants (n)": int(request.n_garant),
        "Risque relation max": round(float(request.max_risk_relation), 2),
        "Risque relation moyen": round(float(request.avg_risk_relation), 2),
    }
    # Remove None to keep prompt clean
    features_for_llm = {k: v for k, v in features_for_llm.items() if v is not None}

    try:
        message = generate_risk_explanation(
            risk_level=base.risk_level,
            default_proba=base.default_proba,
            features=features_for_llm,
        )
    except Exception as exc:
        # En cas de problème LLM, on renvoie au moins le score.
        message = (
            "Impossible d'obtenir une explication détaillée pour le moment "
            f"(erreur LLM: {exc}). Voici néanmoins le niveau de risque: "
            f"{base.risk_level} avec probabilité {base.default_proba}."
        )

    return ExplanationResponse(
        prediction=base.prediction,
        default_proba=base.default_proba,
        risk_level=base.risk_level,
        model_used=base.model_used,
        message=message,
    )


@app.post("/explain/by-cin", response_model=ExplanationByCinResponse, tags=["Prediction", "LLM"])
def explain_risk_by_cin(payload: CinRequest):
    """
    Provide the same result as /explain, but you only send a CIN.
    Classic model: pick the most recent credit by default (credit_id is optional).
    """
    _load_lookup_tables()
    assert _clients_df is not None and _credits_df is not None and _features_df is not None

    cin = _normalize_cin(payload.cin)

    # 1) Find client_id
    match = _clients_df[_clients_df["cin"].astype(str) == cin]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"CIN not found: {cin}")
    client_id = int(match.iloc[0]["client_id"])

    # 2) Select credit_id
    client_credits = _credits_df[_credits_df["client_id"] == client_id].copy()
    if client_credits.empty:
        raise HTTPException(status_code=404, detail=f"No credits found for CIN: {cin}")

    if payload.credit_id is not None:
        credit_id = int(payload.credit_id)
        if credit_id not in set(client_credits["credit_id"].astype(int).tolist()):
            raise HTTPException(
                status_code=404,
                detail=f"credit_id {credit_id} does not belong to CIN {cin}",
            )
    else:
        client_credits = client_credits.sort_values("date_debut", ascending=False)
        credit_id = int(client_credits.iloc[0]["credit_id"])

    feat_row = _features_df[_features_df["credit_id"] == credit_id]
    if feat_row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Features not found for credit_id {credit_id}. Rebuild features.parquet.",
        )

    feature_order = _metadata.get("feature_columns", _default_feature_order())
    row = feat_row.iloc[0].to_dict()
    try:
        req_dict = {k: float(row[k]) for k in feature_order}
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Feature mismatch: missing {e} in features.parquet")

    credit_request = CreditRequest(**req_dict)
    explained = explain_risk(credit_request)
    kyc_score_val = float(row.get("kyc_score", 0.0))

    return ExplanationByCinResponse(
        cin=cin,
        credit_id=credit_id,
        kyc_score=round(kyc_score_val, 2),
        prediction=explained.prediction,
        default_proba=explained.default_proba,
        risk_level=explained.risk_level,
        model_used=explained.model_used,
        message=explained.message,
    )


@app.post("/predict/sequential/by-cin", response_model=SequentialByCinResponse, tags=["Prediction", "Sequential"], include_in_schema=False)
def predict_sequential_by_cin(payload: CinRequest):
    """
    Predict default risk with the sequential (LSTM/GRU) model using only CIN.
    """
    _load_lookup_tables()
    _load_sequential_artifacts()
    assert _clients_df is not None and _seq_model is not None

    cin = _normalize_cin(payload.cin)
    match = _clients_df[_clients_df["cin"].astype(str) == cin]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"CIN not found: {cin}")
    client_row = match.iloc[0]
    client_id = int(client_row["client_id"])

    kyc_score_val = round(compute_kyc_score_row(client_row), 2)

    import torch

    x = _build_seq_for_client(client_id)
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, input_dim)

    with torch.no_grad():
        logits = _seq_model(x_t).cpu().numpy().ravel()[0]
    proba = float(1.0 / (1.0 + np.exp(-logits)))
    prediction = int(proba >= 0.5)
    risk_level = _risk_level_from_proba(proba)

    return SequentialByCinResponse(
        cin=cin,
        kyc_score=kyc_score_val,
        prediction=prediction,
        default_proba=round(proba, 4),
        risk_level=risk_level,
        model_used=_seq_model_name,
    )


@app.post("/predict/graph/by-cin", response_model=GraphByCinResponse, tags=["Prediction", "Graph"], include_in_schema=False)
def predict_graph_by_cin(payload: CinRequest):
    """
    Predict default risk with the graph (GraphSAGE) model using only CIN.
    """
    _load_lookup_tables()
    _load_graph_artifacts()
    assert _clients_df is not None and _graph_model is not None and _graph_client_id_to_idx is not None

    cin = _normalize_cin(payload.cin)
    match = _clients_df[_clients_df["cin"].astype(str) == cin]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"CIN not found: {cin}")
    client_row = match.iloc[0]
    client_id = int(client_row["client_id"])
    idx = _graph_client_id_to_idx.get(client_id)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Client_id {client_id} not found in graph.")

    import torch

    with torch.no_grad():
        logits = _graph_model(_graph_x, _graph_edge_index).cpu().numpy().ravel()
    proba = float(1.0 / (1.0 + np.exp(-float(logits[int(idx)]))))
    prediction = int(proba >= 0.5)
    risk_level = _risk_level_from_proba(proba)
    kyc_score_val = round(compute_kyc_score_row(client_row), 2)

    return GraphByCinResponse(
        cin=cin,
        kyc_score=kyc_score_val,
        prediction=prediction,
        default_proba=round(proba, 4),
        risk_level=risk_level,
        model_used=_graph_model_name,
    )


@app.post("/explain/graph/by-cin", response_model=GraphExplanationByCinResponse, tags=["Prediction", "Graph", "LLM"])
def explain_graph_by_cin(payload: CinRequest):
    """
    Graph prediction + LLM explanation using only CIN.
    """
    _load_lookup_tables()
    assert _clients_df is not None and _tx_df is not None

    base = predict_graph_by_cin(payload)
    cin = _normalize_cin(payload.cin)

    # Enriched graph explanation: add simple network + behaviour stats
    match = _clients_df[_clients_df["cin"].astype(str) == cin]
    client_id = int(match.iloc[0]["client_id"]) if not match.empty else None

    # Degree from graph edges
    degree = None
    try:
        _load_graph_artifacts()
        if client_id is not None and _graph_client_id_to_idx is not None and _graph_edge_index is not None:
            idx = _graph_client_id_to_idx.get(int(client_id))
            if idx is not None:
                ei = _graph_edge_index.cpu().numpy()
                degree = int(((ei[0] == idx).sum() + (ei[1] == idx).sum()))
    except Exception:
        degree = None

    # Credits count for this client
    n_credits = int((_credits_df["client_id"] == int(client_id)).sum()) if client_id is not None else 0

    # Transaction / remboursement summaries (global)
    tx = _tx_df[_tx_df["client_id"] == int(client_id)].copy() if client_id is not None else pd.DataFrame()
    n_tx = int(len(tx)) if len(tx) else 0
    n_sus = int(tx["suspect"].sum()) if n_tx else 0
    total_tx = float(tx["montant"].sum()) if n_tx else 0.0
    avg_tx = float(tx["montant"].mean()) if n_tx else 0.0

    remb = _remb_df[_remb_df["client_id"] == int(client_id)].copy() if client_id is not None else pd.DataFrame()
    n_remb = int(len(remb)) if len(remb) else 0
    n_late = int((remb["retard_jours"] > 0).sum()) if n_remb else 0
    pct_late = float(n_late / max(n_remb, 1)) if n_remb else 0.0
    avg_ret = float(remb["retard_jours"].mean()) if n_remb else 0.0
    max_ret = float(remb["retard_jours"].max()) if n_remb else 0.0

    features_for_llm = {
        "CIN": cin,
        "Score KYC": base.kyc_score,
        "Nombre de crédits": n_credits,
        "Degré réseau (nombre de liens)": degree,
        "Transactions (n)": n_tx,
        "Transactions suspectes (n)": n_sus,
        "Montant total transactions": round(total_tx, 2),
        "Montant moyen transactions": round(avg_tx, 2),
        "Remboursements (n)": n_remb,
        "Remboursements en retard (n)": n_late,
        "Taux de retard remboursement": round(pct_late, 3),
        "Retard moyen (jours)": round(avg_ret, 2),
        "Retard max (jours)": round(max_ret, 2),
        "Probabilité de défaut (graphe)": base.default_proba,
        "Niveau de risque (graphe)": base.risk_level,
    }
    features_for_llm = {k: v for k, v in features_for_llm.items() if v is not None}

    try:
        message = generate_risk_explanation(
            risk_level=base.risk_level,
            default_proba=base.default_proba,
            features=features_for_llm,
        )
    except Exception as exc:
        message = (
            "Impossible d'obtenir une explication détaillée pour le moment "
            f"(erreur LLM: {exc}). Niveau de risque graphe: "
            f"{base.risk_level}, probabilité {base.default_proba}."
        )

    return GraphExplanationByCinResponse(
        cin=base.cin,
        kyc_score=base.kyc_score,
        prediction=base.prediction,
        default_proba=base.default_proba,
        risk_level=base.risk_level,
        model_used=base.model_used,
        message=message,
    )


@app.post(
    "/explain/sequential/by-cin",
    response_model=SequentialExplanationByCinAllCreditsResponse,
    tags=["Prediction", "Sequential", "LLM"],
)
def explain_sequential_by_cin(payload: CinRequest):
    """
    Sequential prediction + LLM explanation using only CIN.
    """
    _load_lookup_tables()
    assert _clients_df is not None and _credits_df is not None and _tx_df is not None and _remb_df is not None

    cin = _normalize_cin(payload.cin)
    match = _clients_df[_clients_df["cin"].astype(str) == cin]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"CIN not found: {cin}")
    client_row = match.iloc[0]
    client_id = int(client_row["client_id"])

    _load_sequential_artifacts()
    assert _seq_model is not None

    kyc_score_val = round(compute_kyc_score_row(client_row), 2)

    # Score ALL credits for this client (professional sequential)
    client_credits = _credits_df[_credits_df["client_id"] == client_id].copy()
    if client_credits.empty:
        raise HTTPException(status_code=404, detail=f"No credits found for CIN: {cin}")

    client_credits = client_credits.sort_values("date_debut", ascending=False)
    per_credit: list[CreditExplanationItem] = []
    worst = None  # (credit_id, proba, risk_level, pred)

    import torch

    for _, cr in client_credits.iterrows():
        credit_id = int(cr["credit_id"])
        start = pd.to_datetime(cr["date_debut"])
        x = _build_seq_for_client_credit(client_id, start)
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = _seq_model(x_t).cpu().numpy().ravel()[0]
        proba = float(1.0 / (1.0 + np.exp(-logits)))
        pred = int(proba >= 0.5)
        risk = _risk_level_from_proba(proba)

        per_credit.append(
            CreditExplanationItem(
                credit_id=credit_id,
                prediction=pred,
                default_proba=round(proba, 4),
                risk_level=risk,
            )
        )
        if (worst is None) or (proba > worst[1]):
            worst = (credit_id, proba, risk, pred)

    assert worst is not None
    worst_credit_id, worst_proba, worst_risk, worst_pred = worst

    # --- Build richer summary for the LLM (transactions + remboursements + credits) ---
    # Global client transaction summary (all history)
    tx_all = _tx_df[_tx_df["client_id"] == client_id].copy()
    n_tx_all = int(len(tx_all))
    n_sus_all = int(tx_all["suspect"].sum()) if n_tx_all else 0
    avg_amt_all = float(tx_all["montant"].mean()) if n_tx_all else 0.0
    total_amt_all = float(tx_all["montant"].sum()) if n_tx_all else 0.0

    # Global client remboursement summary (all history)
    remb_all = _remb_df[_remb_df["client_id"] == client_id].copy()
    n_remb_all = int(len(remb_all))
    n_late_all = int((remb_all["retard_jours"] > 0).sum()) if n_remb_all else 0
    pct_late_all = float(n_late_all / max(n_remb_all, 1))
    avg_ret_all = float(remb_all["retard_jours"].mean()) if n_remb_all else 0.0
    max_ret_all = float(remb_all["retard_jours"].max()) if n_remb_all else 0.0

    # Worst credit context window (events up to date_debut of worst credit)
    worst_row = client_credits[client_credits["credit_id"].astype(int) == int(worst_credit_id)]
    worst_start = pd.to_datetime(worst_row.iloc[0]["date_debut"]) if not worst_row.empty else pd.Timestamp.max

    tx_w = tx_all[tx_all["date"] <= worst_start].copy() if n_tx_all else tx_all
    n_tx_w = int(len(tx_w))
    n_sus_w = int(tx_w["suspect"].sum()) if n_tx_w else 0

    # Totals by type (window)
    totals_by_type = {}
    if n_tx_w:
        for t in ["DEPOT", "RETRAIT", "REMBOURSEMENT", "TRANSFERT"]:
            totals_by_type[t] = float(tx_w.loc[tx_w["type"].astype(str) == t, "montant"].sum())
    else:
        totals_by_type = {t: 0.0 for t in ["DEPOT", "RETRAIT", "REMBOURSEMENT", "TRANSFERT"]}

    remb_w = remb_all[remb_all["date_echeance"] <= worst_start].copy() if n_remb_all else remb_all
    n_remb_w = int(len(remb_w))
    n_late_w = int((remb_w["retard_jours"] > 0).sum()) if n_remb_w else 0
    pct_late_w = float(n_late_w / max(n_remb_w, 1))
    avg_ret_w = float(remb_w["retard_jours"].mean()) if n_remb_w else 0.0
    max_ret_w = float(remb_w["retard_jours"].max()) if n_remb_w else 0.0

    features_for_llm = {
        "CIN": cin,
        "Score KYC": kyc_score_val,
        "Nombre de crédits analysés": len(per_credit),
        "Crédit le plus risqué (credit_id)": worst_credit_id,
        "Probabilité de défaut séquentielle (pire cas)": round(float(worst_proba), 4),
        "Niveau de risque séquentiel (pire cas)": worst_risk,
        "Synthèse transactions (toutes périodes)": f"{n_tx_all} tx | suspectes={n_sus_all} | montant_moyen={round(avg_amt_all,2)} | total={round(total_amt_all,2)}",
        "Synthèse remboursements (toutes périodes)": f"{n_remb_all} remb | en_retard={n_late_all} | pct_retard={round(pct_late_all,2)} | retard_moy={round(avg_ret_all,2)} | retard_max={round(max_ret_all,2)}",
        "Fenêtre du crédit le plus risqué (jusqu'à date_debut)": str(worst_start.date()) if worst_start is not pd.Timestamp.max else "N/A",
        "Transactions (fenêtre)": f"{n_tx_w} tx | suspectes={n_sus_w} | total_DEPOT={round(totals_by_type['DEPOT'],2)} | total_RETRAIT={round(totals_by_type['RETRAIT'],2)} | total_TRANSFERT={round(totals_by_type['TRANSFERT'],2)}",
        "Remboursements (fenêtre)": f"{n_remb_w} remb | en_retard={n_late_w} | pct_retard={round(pct_late_w,2)} | retard_moy={round(avg_ret_w,2)} | retard_max={round(max_ret_w,2)}",
    }

    try:
        message = generate_risk_explanation(
            risk_level=worst_risk,
            default_proba=float(worst_proba),
            features=features_for_llm,
        )
    except Exception as exc:
        message = (
            "Impossible d'obtenir une explication détaillée pour le moment "
            f"(erreur LLM: {exc}). Niveau de risque séquentiel: "
            f"{worst_risk}, probabilité {round(float(worst_proba), 4)}."
        )

    return SequentialExplanationByCinAllCreditsResponse(
        cin=cin,
        kyc_score=kyc_score_val,
        prediction=worst_pred,
        default_proba=round(float(worst_proba), 4),
        risk_level=worst_risk,
        model_used=_seq_model_name,
        message=message,
        credits=per_credit,
        n_credits=len(per_credit),
    )


def _default_feature_order() -> list[str]:
    """Fallback feature order if metadata JSON is missing."""
    return [
        "montant", "duree_mois", "dti",
        "avg_retard", "max_retard", "std_retard", "n_payments",
        "n_late", "pct_late", "n_en_retard",
        "n_transactions", "n_suspect", "avg_tx_amount",
        "total_depot", "total_retrait", "total_remboursement",
        "total_transfert", "ratio_retrait_depot",
        "max_risk_relation", "avg_risk_relation", "n_relations", "n_garant",
        "age", "revenu_mensuel",
        "kyc_score", "cycle_enc", "objet_enc", "profession_enc",
    ]
