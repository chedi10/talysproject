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
import os
import re
import unicodedata
from pathlib import Path
from typing import Annotated, Any
from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.config import (
    BEST_MODEL_FILE,
    MODEL_METADATA_FILE,
    FEATURES_FILE,
    REPORTS_DIR,
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
    GraphNetworkSnapshot,
    EnsembleExplanationByCinResponse,
    EnsembleModelScore,
    ChatRequest,
    ChatResponse,
    ReportRequest,
    ReportResponse,
    ReportDownloadRequest,
    RegisterRequest,
    LoginRequest,
    AuthResponse,
    UserPublic,
    ActivityListResponse,
    ActivityRecord,
    ChatSessionListResponse,
    ChatSessionSummary,
    ChatHistoryResponse,
    ChatStructuredResult,
    RagSourceItem,
    ShapSystemResponse,
    RulesSystemResponse,
    EarlyWarningSystemResponse,
    RecommendationSystemResponse,
    ShapExplanation,
    ShapFeatureImpact,
    ShapDriverDetail,
    CreditContext,
    BusinessRulesBlock,
    BusinessRuleResult,
    EarlyWarningBlock,
    EarlyWarningAlert,
    TrendSeries,
    TrendPoint,
    ContributingFactors,
    ClientProfile,
    AiRecommendation,
    AdminCreateUserRequest,
    ClientProfileResponse,
    ClientChatRequest,
    ClientChatResponse,
    SystemStatsResponse,
)
from src.auth.dependencies import get_current_user, require_admin, require_staff
from src.auth.local_store import (
    append_chat_messages,
    authenticate,
    create_session,
    create_user,
    delete_session,
    ensure_default_admin,
    get_chat_messages,
    get_conversation_context,
    list_activity,
    list_chat_sessions,
    list_users,
    log_activity,
    public_user,
)
from src.auth.dependencies import _extract_bearer
from src.llm.client import generate_risk_explanation
from src.kyc.score import compute_kyc_score, compute_kyc_score_row
from src.rag.index import retrieve_multi as rag_retrieve_multi
from src.rag.context import (
    build_rag_query,
    format_rag_sources_for_prompt,
    format_rag_references_section,
)
from src.agent.langgraph_workflow import CreditAgentOrchestrator
from src.reports.report_builder import build_structured_report, structured_to_markdown
from src.reports.systems_enrichment import fetch_systems_from_ctx
from src.reports.pdf_export import export_structured_pdf
from src.systems.business_risk import compute_institutional_risk
from src.systems.ensemble_scoring import (
    MODEL_LABELS,
    ModelScoreInput,
    compute_ensemble,
)
from src.systems.orchestrator import (
    enrich_context,
    run_ews_standalone,
    run_recommendation_standalone,
    run_rules_standalone,
    run_shap_standalone,
)
from src.client.portal_service import build_client_profile_response, profile_to_context
from src.agent.client_chatbot import ClientChatbot
from src.db.seed import bootstrap_database
from src.db.repository import (
    load_clients_df,
    load_credits_df,
    load_transactions_df,
    load_remboursements_df,
    get_client_by_cin,
    db_stats,
    admin_extended_stats,
)

# Load environment variables from .env (local settings)
load_dotenv()

# ─── Load model and metadata at startup ──────────────────────────────────────
_model = None
_deep_tabular_model = None
_model_backend: str = "none"
_metadata: dict = {}


def _load_artifacts():
    global _model, _deep_tabular_model, _model_backend, _metadata
    if MODEL_METADATA_FILE.exists():
        with open(MODEL_METADATA_FILE, encoding="utf-8") as f:
            _metadata = json.load(f)

    deep_path = MODELS_DIR / "deep_tabular.pt"
    if deep_path.exists():
        try:
            import torch
            from src.models.deep_tabular.predict import load_deep_tabular

            _deep_tabular_model, ckpt = load_deep_tabular(deep_path)
            _model_backend = "deep_tabular"
            _metadata.setdefault("best_model_name", "Deep Tabular Network (MLP+Embeddings)")
            _metadata.setdefault("model_type", "deep_tabular")
            _metadata.setdefault("feature_columns", ckpt.get("feature_cols", _metadata.get("feature_columns", [])))
            _metadata["num_cols"] = ckpt.get("num_cols", [])
            _metadata["cat_cols"] = ckpt.get("cat_cols", [])
            return
        except Exception as exc:
            print(f"Warning: could not load deep tabular model: {exc}")

    if BEST_MODEL_FILE.exists():
        _model = joblib.load(BEST_MODEL_FILE)
        _model_backend = "sklearn"
        if not _metadata and MODEL_METADATA_FILE.exists():
            with open(MODEL_METADATA_FILE, encoding="utf-8") as f:
                _metadata = json.load(f)
        return

    print(
        "Warning: no ML model found. Run:\n"
        "  python -m src.features.engineering\n"
        "  python -m src.models.deep_train"
    )


try:
    _load_artifacts()
except Exception as e:
    print(f"Model load warning: {e}")

# ─── Lightweight lookup tables for CIN → credit_id → features ────────────────
_clients_df: pd.DataFrame | None = None
_credits_df: pd.DataFrame | None = None
_features_df: pd.DataFrame | None = None
_tx_df: pd.DataFrame | None = None
_remb_df: pd.DataFrame | None = None
_tx_amount_max: float = 1.0
_remb_amount_max: float = 1.0

# ─── Agent conversationnel (LangGraph) memory ────────────────────────────────
_agent_orchestrator: CreditAgentOrchestrator | None = None


def _load_lookup_tables():
    """
    Load tables from SQLite for CIN-based scoring.
    """
    global _clients_df, _credits_df, _features_df, _tx_df, _remb_df, _tx_amount_max, _remb_amount_max
    if _clients_df is None:
        _clients_df = load_clients_df()
    if _credits_df is None:
        _credits_df = load_credits_df()
    if _tx_df is None:
        _tx_df = load_transactions_df()
        _tx_df["client_id"] = _tx_df["client_id"].astype(int)
        _tx_df["montant"] = _tx_df["montant"].astype(float)
        _tx_df["suspect"] = _tx_df["suspect"].astype(int)
        _tx_amount_max = float(_tx_df["montant"].max()) if len(_tx_df) else 1.0
    if _remb_df is None:
        _remb_df = load_remboursements_df()
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


def _rag_context_for_scoring(
    *,
    risk_level: str | None = None,
    model_used: str | None = None,
    kyc_score: float | None = None,
    cin: str | None = None,
    user_message: str | None = None,
    intent: str | None = None,
    max_chars: int = 2000,
) -> tuple[str, list[dict]]:
    """
    Retrieve curated docs and format excerpts for LLM prompts (explanations/reports).
    Does not affect ML scores.
    """
    try:
        queries = build_rag_query(
            cin=cin,
            user_message=user_message,
            intent=intent,
            risk_level=risk_level,
            model_used=model_used,
            kyc_score=kyc_score,
        )
        sources = rag_retrieve_multi(queries, k=5)
        return format_rag_sources_for_prompt(sources, max_chars=max_chars), sources
    except Exception:
        return "", []


def _extract_cin_from_text(text: str) -> str | None:
    # CIN in our synthetic data is numeric (8 digits), but allow 6-32 digits/spaces
    m = re.search(r"\b(\d{6,32})\b", text)
    if not m:
        return None
    return _normalize_cin(m.group(1))


def _select_model_from_text(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["graph", "graphe", "gnn", "graphsage", "réseau", "reseau"]):
        return "graph"
    if any(k in t for k in ["séquent", "sequent", "lstm", "gru", "transaction", "transactions", "remboursement"]):
        return "sequential"
    return "classic"


def _format_agent_answer(model_selected: str, result: dict) -> str:
    # Provide a short, structured answer; the LLM will enrich it.
    parts = [
        f"Modèle choisi: {model_selected}",
        f"CIN: {result.get('cin')}",
        f"Score KYC: {result.get('kyc_score')}",
        f"Probabilité de défaut: {result.get('default_proba')}",
        f"Niveau de risque: {result.get('risk_level')}",
    ]
    if "n_credits" in result:
        parts.append(f"Nombre de crédits analysés: {result.get('n_credits')}")
    return "\n".join(parts)

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
_graph_edge_weight = None
_graph_x = None
_graph_client_id_to_idx: dict[int, int] | None = None
_graph_client_ids: np.ndarray | None = None
_graph_dataset = None
_graph_model_name = "GraphSAGE"


def _load_graph_artifacts():
    global _graph_model, _graph_edge_index, _graph_edge_weight, _graph_x, _graph_client_id_to_idx
    global _graph_client_ids, _graph_dataset, _graph_model_name
    if _graph_model is not None:
        return

    try:
        import torch
    except ImportError as e:
        raise HTTPException(status_code=500, detail="PyTorch is not installed in this environment.") from e

    meta_path = MODELS_DIR / "graph_metadata.json"
    ckpt_path = MODELS_DIR / "graph_gat.pt"
    model_type = "gat"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        artifact = meta.get("artifact")
        if artifact:
            p = Path(artifact)
            ckpt_path = p if p.is_absolute() else MODELS_DIR / p.name
        model_type = meta.get("model_type", "gat")
    if not ckpt_path.exists():
        ckpt_path = MODELS_DIR / "graphsage.pt"
        model_type = "graphsage"

    if not ckpt_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Graph checkpoint not found. Run `python -m src.models.graph.train` first.",
        )

    from src.models.graph.data import build_graph_dataset

    ds = build_graph_dataset()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_type = ckpt.get("model_type", model_type)

    in_dim = int(ckpt.get("in_dim", ds.x.shape[1]))
    hidden_dim = int(ckpt.get("hidden_dim", 64))
    dropout = float(ckpt.get("dropout", 0.2))

    if model_type == "gat":
        from src.models.graph.gat_model import GATClassifier

        n_heads = int(ckpt.get("n_heads", 4))
        model = GATClassifier(in_dim=in_dim, hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout)
        _graph_model_name = "GAT (Graph Attention Network)"
    else:
        from src.models.graph.model import GraphSAGEClassifier

        model = GraphSAGEClassifier(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        _graph_model_name = "GraphSAGE"

    state = ckpt.get("state_dict")
    if state:
        model.load_state_dict(state)
    model.eval()

    _graph_model = model
    _graph_edge_index = torch.tensor(ds.edge_index, dtype=torch.long)
    _graph_edge_weight = torch.tensor(ds.edge_weight, dtype=torch.float32)
    _graph_x = torch.tensor(ds.x, dtype=torch.float32)
    _graph_client_id_to_idx = ds.client_id_to_idx
    _graph_client_ids = ds.client_ids
    _graph_dataset = ds


def _client_default_labels() -> dict[int, int]:
    assert _credits_df is not None
    return _credits_df.groupby("client_id")["en_defaut"].max().astype(int).to_dict()


def _build_graph_network_snapshot(client_id: int) -> GraphNetworkSnapshot | None:
    """Sous-graphe ego pour visualisation UI."""
    try:
        _load_graph_artifacts()
        _load_lookup_tables()
        assert (
            _graph_model is not None
            and _graph_dataset is not None
            and _clients_df is not None
            and _graph_client_id_to_idx is not None
        )
        import torch
        from src.config import RAW_RELATIONS
        from src.graph.network import build_ego_network

        ds = _graph_dataset
        idx_to_cid = {i: int(ds.client_ids[i]) for i in range(len(ds.client_ids))}

        with torch.no_grad():
            if _graph_edge_weight is not None:
                logits = _graph_model(_graph_x, _graph_edge_index, _graph_edge_weight).cpu().numpy().ravel()
            else:
                logits = _graph_model(_graph_x, _graph_edge_index).cpu().numpy().ravel()
        probas = 1.0 / (1.0 + np.exp(-logits))
        node_probas = {int(ds.client_ids[i]): float(probas[i]) for i in range(len(ds.client_ids))}

        rel_df = pd.read_csv(RAW_RELATIONS)
        net = build_ego_network(
            center_client_id=int(client_id),
            edge_index=ds.edge_index,
            edge_weight=ds.edge_weight,
            client_ids=ds.client_ids,
            idx_to_client_id=idx_to_cid,
            client_id_to_idx=ds.client_id_to_idx,
            clients_df=_clients_df,
            relations_df=rel_df,
            node_probas=node_probas,
            node_labels=_client_default_labels(),
        )
        if not net.get("nodes"):
            return None
        return GraphNetworkSnapshot(**net)
    except Exception:
        return None


def _load_sequential_artifacts():
    global _seq_model, _seq_device, _seq_seq_len, _seq_input_dim, _seq_model_name
    if _seq_model is not None:
        return

    try:
        import torch
    except ImportError as e:
        raise HTTPException(status_code=500, detail="PyTorch is not installed in this environment.") from e

    meta_path = MODELS_DIR / "sequential_metadata.json"
    ckpt_path = MODELS_DIR / "sequential_transformer.pt"
    model_type = "transformer"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        best = meta.get("best", {})
        artifact = best.get("artifact")
        if artifact:
            ckpt_path = Path(artifact)
            _seq_model_name = best.get("model_name", "Sequential DL")
        model_type = best.get("model_type", "transformer")

    if not ckpt_path.exists():
        ckpt_path = MODELS_DIR / "sequential_gru.pt"
        model_type = "gru"

    if not ckpt_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Sequential checkpoint not found. Run `python -m src.models.sequential.train` first.",
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    _seq_input_dim = int(ckpt.get("input_dim", 10))
    _seq_seq_len = int(ckpt.get("seq_len", 30))
    model_type = ckpt.get("model_type", model_type)

    if model_type == "transformer":
        from src.models.sequential.transformer_model import TransformerCreditRiskModel

        model = TransformerCreditRiskModel(
            input_dim=_seq_input_dim,
            d_model=64,
            nhead=4,
            num_layers=2,
            seq_len=_seq_seq_len,
        )
        _seq_model_name = "Temporal Transformer (Deep Learning)"
    else:
        from src.models.sequential.model import RecurrentCreditRiskModel

        rnn_type = str(ckpt.get("rnn_type", "gru"))
        model = RecurrentCreditRiskModel(
            input_dim=_seq_input_dim,
            hidden_dim=64,
            num_layers=1,
            dropout=0.2,
            rnn_type=rnn_type,
        )
        _seq_model_name = f"RNN {rnn_type.upper()} (legacy)"

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


@app.on_event("startup")
def _bootstrap_auth_store() -> None:
    bootstrap_database()


def _record_activity(user: dict[str, Any], action: str, **kwargs: Any) -> None:
    try:
        log_activity(user=user, action=action, **kwargs)
    except Exception:
        pass


@app.post("/auth/register", response_model=AuthResponse, tags=["Auth"])
def auth_register(payload: RegisterRequest):
    """Créer un compte client ou agent."""
    role = payload.role
    cin = payload.cin.strip() if payload.cin else None
    client_id = None

    if role == "client":
        if not cin:
            raise HTTPException(status_code=422, detail="CIN requis pour un compte client.")
        _load_lookup_tables()
        row = get_client_by_cin(_normalize_cin(cin))
        if not row:
            raise HTTPException(status_code=404, detail=f"CIN inconnu dans la base : {cin}")
        client_id = int(row["client_id"])
        cin = str(row["cin"])

    try:
        user = create_user(
            username=payload.username.strip(),
            email=payload.email.strip(),
            password=payload.password,
            role=role,
            cin=cin,
            client_id=client_id,
        )
    except ValueError as exc:
        code = str(exc)
        if code == "username_taken":
            raise HTTPException(status_code=409, detail="Nom d'utilisateur déjà utilisé.") from exc
        if code == "username_too_short":
            raise HTTPException(status_code=422, detail="Nom d'utilisateur trop court (min 3).") from exc
        if code == "password_too_short":
            raise HTTPException(status_code=422, detail="Mot de passe trop court (min 6).") from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    token = create_session(user["id"])
    return AuthResponse(token=token, user=UserPublic(**user))


@app.post("/auth/users", response_model=UserPublic, tags=["Auth"])
def auth_create_user(
    payload: AdminCreateUserRequest,
    admin: Annotated[dict[str, Any], Depends(require_admin)],
):
    """Admin — créer un utilisateur (client, agent ou admin)."""
    _ = admin
    cin = payload.cin.strip() if payload.cin else None
    client_id = None
    if payload.role == "client":
        if not cin:
            raise HTTPException(status_code=422, detail="CIN requis pour un compte client.")
        row = get_client_by_cin(_normalize_cin(cin))
        if not row:
            raise HTTPException(status_code=404, detail=f"CIN inconnu : {cin}")
        client_id = int(row["client_id"])
        cin = str(row["cin"])
    try:
        user = create_user(
            username=payload.username.strip(),
            email=payload.email.strip(),
            password=payload.password,
            role=payload.role,
            cin=cin,
            client_id=client_id,
        )
    except ValueError as exc:
        if str(exc) == "username_taken":
            raise HTTPException(status_code=409, detail="Nom d'utilisateur déjà utilisé.")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return UserPublic(**user)


@app.get("/client/profile", response_model=ClientProfileResponse, tags=["Client"])
def client_profile(user: Annotated[dict[str, Any], Depends(get_current_user)]):
    """Portail client — profil enrichi, alertes et synthèse crédit."""
    if user.get("role") != "client":
        raise HTTPException(status_code=403, detail="Réservé aux clients.")
    cin = user.get("cin")
    client_id = user.get("client_id")
    if not cin or not client_id:
        raise HTTPException(status_code=404, detail="Compte client non lié à un CIN.")
    row = get_client_by_cin(cin)
    if not row:
        raise HTTPException(status_code=404, detail="Client introuvable.")
    return build_client_profile_response(cin=cin, client_id=int(client_id), row=row)


_client_chatbot: ClientChatbot | None = None


def _get_client_chatbot() -> ClientChatbot:
    global _client_chatbot
    if _client_chatbot is None:
        _client_chatbot = ClientChatbot(
            rag_retrieve=lambda q, k=2: rag_retrieve_multi([q], k=k),
        )
    return _client_chatbot


@app.post("/client/chat", response_model=ClientChatResponse, tags=["Client"])
def client_chat(
    payload: ClientChatRequest,
    user: Annotated[dict[str, Any], Depends(get_current_user)],
):
    """Assistant conversationnel client — dossier, crédits, KYC, FAQ (données personnelles uniquement)."""
    if user.get("role") != "client":
        raise HTTPException(status_code=403, detail="Réservé aux clients.")
    cin = user.get("cin")
    client_id = user.get("client_id")
    if not cin or not client_id:
        raise HTTPException(status_code=404, detail="Compte client non lié à un CIN.")
    row = get_client_by_cin(cin)
    if not row:
        raise HTTPException(status_code=404, detail="Client introuvable.")

    profile = build_client_profile_response(cin=cin, client_id=int(client_id), row=row)
    ctx = profile_to_context(profile)
    session_id = payload.session_id.strip() or f"client-{user['id'][:8]}"
    user_msg = payload.message.strip()

    history = get_conversation_context(user_id=user["id"], session_id=session_id, limit=6)
    out = _get_client_chatbot().invoke(message=user_msg, profile=ctx, conversation_history=history)

    answer = str(out.get("answer", "")).strip()
    append_chat_messages(
        user_id=user["id"],
        session_id=session_id,
        messages=[
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ],
        cin=cin,
        intent=str(out.get("intent", "general")),
        title=f"Client · {user_msg[:50]}",
    )
    _record_activity(user, "client_chat", cin=cin, intent=out.get("intent"), message=user_msg, session_id=session_id)

    rag_sources = [
        RagSourceItem(
            source=str(s.get("source", "")),
            chunk_id=int(s.get("chunk_id", 0)),
            score=float(s.get("score", 0)),
            text=str(s.get("text", ""))[:400],
        )
        for s in (out.get("rag_sources") or [])
    ]

    return ClientChatResponse(
        session_id=session_id,
        intent=out.get("intent", "general"),
        answer=answer,
        rag_sources=rag_sources,
        suggested_prompts=out.get("suggested_prompts") or [],
    )


@app.get("/admin/stats", response_model=SystemStatsResponse, tags=["Admin"])
def admin_stats(admin: Annotated[dict[str, Any], Depends(require_admin)]):
    """Tableau de bord admin — statistiques système enrichies."""
    _ = admin
    stats = admin_extended_stats()
    graph_auc = None
    graph_model = None
    meta_path = MODELS_DIR / "graph_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                gmeta = json.load(f)
            graph_auc = gmeta.get("best_auc")
            graph_model = gmeta.get("model_name")
        except Exception:
            pass
    return SystemStatsResponse(
        database="SQLite",
        clients=stats.get("clients", 0),
        credits=stats.get("credits", 0),
        transactions=stats.get("transactions", 0),
        users=stats.get("users", 0),
        activity_log=stats.get("activity_log", 0),
        model_loaded=(_model is not None or _deep_tabular_model is not None),
        model_name=_metadata.get("best_model_name", "unknown"),
        relations=stats.get("relations", 0),
        remboursements=stats.get("remboursements", 0),
        chat_sessions=stats.get("chat_sessions", 0),
        credits_en_defaut=stats.get("credits_en_defaut", 0),
        default_rate=float(stats.get("default_rate", 0)),
        activity_last_7_days=stats.get("activity_last_7_days", 0),
        users_by_role=stats.get("users_by_role", {}),
        kyc_breakdown=stats.get("kyc_breakdown", {}),
        activity_by_action=stats.get("activity_by_action", {}),
        graph_model=graph_model,
        graph_auc=graph_auc,
    )


@app.post("/auth/login", response_model=AuthResponse, tags=["Auth"])
def auth_login(payload: LoginRequest):
    """Connexion — retourne un token de session."""
    user_raw = authenticate(payload.username.strip(), payload.password)
    if not user_raw:
        raise HTTPException(status_code=401, detail="Identifiants invalides.")
    user = public_user(user_raw)
    token = create_session(user["id"])
    return AuthResponse(token=token, user=UserPublic(**user))


@app.post("/auth/logout", tags=["Auth"])
def auth_logout(
    user: Annotated[dict[str, Any], Depends(get_current_user)],
    authorization: Annotated[str | None, Header()] = None,
):
    token = _extract_bearer(authorization)
    if token:
        delete_session(token)
    return {"status": "ok", "username": user["username"]}


@app.get("/auth/me", response_model=UserPublic, tags=["Auth"])
def auth_me(user: Annotated[dict[str, Any], Depends(get_current_user)]):
    return UserPublic(**user)


@app.get("/auth/history", response_model=ActivityListResponse, tags=["Auth"])
def auth_history(
    user: Annotated[dict[str, Any], Depends(get_current_user)],
    limit: int = 100,
):
    items = list_activity(user=user, limit=limit)
    scope = "all" if user.get("role") == "admin" else "mine"
    return ActivityListResponse(
        items=[ActivityRecord(**i) for i in items],
        total=len(items),
        scope=scope,
    )


@app.get("/auth/users", response_model=list[UserPublic], tags=["Auth"])
def auth_users(admin: Annotated[dict[str, Any], Depends(require_admin)]):
    _ = admin
    return [UserPublic(**u) for u in list_users()]


@app.get("/auth/chat/sessions", response_model=ChatSessionListResponse, tags=["Auth"])
def auth_chat_sessions(
    user: Annotated[dict[str, Any], Depends(get_current_user)],
    limit: int = 50,
):
    items = list_chat_sessions(user=user, limit=limit)
    scope = "all" if user.get("role") == "admin" else "mine"
    return ChatSessionListResponse(
        items=[ChatSessionSummary(**i) for i in items],
        scope=scope,
    )


@app.get("/auth/chat/sessions/{session_id}", response_model=ChatHistoryResponse, tags=["Auth"])
def auth_chat_history(
    session_id: str,
    user: Annotated[dict[str, Any], Depends(get_current_user)],
):
    messages = get_chat_messages(user=user, session_id=session_id)
    scope = "all" if user.get("role") == "admin" else "mine"
    return ChatHistoryResponse(session_id=session_id.strip(), messages=messages, scope=scope)


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
    if _model is None and _deep_tabular_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run python -m src.models.deep_train")

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

    if _model_backend == "deep_tabular" and _deep_tabular_model is not None:
        from src.models.deep_tabular.predict import predict_proba as deep_predict_proba

        proba = deep_predict_proba(_deep_tabular_model, _metadata, input_dict)
    else:
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

    if os.getenv("DISABLE_LLM", "").strip() in {"1", "true", "True", "yes", "YES"}:
        message = "Explication LLM désactivée (mode rapide)."
    else:
        try:
            rag_ctx, _ = _rag_context_for_scoring(
                risk_level=base.risk_level,
                model_used=base.model_used,
                kyc_score=request.kyc_score,
            )
            message = generate_risk_explanation(
                risk_level=base.risk_level,
                default_proba=base.default_proba,
                features=features_for_llm,
                rag_context=rag_ctx or None,
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


def _resolve_cin_credit_context(payload: CinRequest) -> dict[str, Any]:
    """Resolve CIN → client_id, credit_id, feature row dict."""
    _load_lookup_tables()
    assert _clients_df is not None and _credits_df is not None and _features_df is not None

    cin = _normalize_cin(payload.cin)
    match = _clients_df[_clients_df["cin"].astype(str) == cin]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"CIN not found: {cin}")
    client_id = int(match.iloc[0]["client_id"])

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

    return {
        "cin": cin,
        "client_id": client_id,
        "credit_id": credit_id,
        "feature_order": feature_order,
        "features": req_dict,
        "kyc_score": float(row.get("kyc_score", 0.0)),
    }


def _build_system_context(payload: CinRequest) -> dict[str, Any]:
    """Contexte client + score institutionnel — indépendant des modèles ML."""
    _load_lookup_tables()
    assert _clients_df is not None and _credits_df is not None
    ctx = _resolve_cin_credit_context(payload)
    risk = compute_institutional_risk(ctx["features"])
    ctx = {**ctx, **risk}
    return enrich_context(ctx, clients_df=_clients_df, credits_df=_credits_df)


def _system_response_base(ctx: dict[str, Any]) -> dict[str, Any]:
    profile = ctx.get("client_profile", {})
    return {
        "cin": ctx["cin"],
        "credit_id": ctx["credit_id"],
        "kyc_score": round(ctx["kyc_score"], 2),
        "institutional_score": ctx["institutional_score"],
        "risk_level": ctx["risk_level"],
        "risk_factors": ctx.get("risk_factors", []),
        "client_profile": ClientProfile(
            cin=ctx["cin"],
            client_id=ctx["client_id"],
            nom=profile.get("nom", ""),
            prenom=profile.get("prenom", ""),
            age=int(profile.get("age", 0)),
            ville=profile.get("ville", ""),
            profession=profile.get("profession", ""),
            revenu_mensuel=float(profile.get("revenu_mensuel", 0)),
            statut_kyc=profile.get("statut_kyc", ""),
        ),
        "credit_snapshot": ctx.get("credit_snapshot", {}),
    }


def _explain_risk_by_cin_core(payload: CinRequest) -> ExplanationByCinResponse:
    """
    Classic explain by CIN (core logic — score + LLM only).
    """
    ctx = _resolve_cin_credit_context(payload)
    credit_request = CreditRequest(**ctx["features"])
    explained = explain_risk(credit_request)

    return ExplanationByCinResponse(
        cin=ctx["cin"],
        credit_id=ctx["credit_id"],
        kyc_score=round(ctx["kyc_score"], 2),
        prediction=explained.prediction,
        default_proba=explained.default_proba,
        risk_level=explained.risk_level,
        model_used=explained.model_used,
        message=explained.message,
    )


@app.post("/explain/by-cin", response_model=ExplanationByCinResponse, tags=["Prediction", "LLM"])
def explain_risk_by_cin(
    payload: CinRequest,
    user: Annotated[dict[str, Any], Depends(get_current_user)],
):
    """Modèle classique — score + explication LLM uniquement."""
    out = _explain_risk_by_cin_core(payload)
    _record_activity(
        user,
        "explain_classic",
        cin=out.cin,
        model=out.model_used,
        extra={"risk_level": out.risk_level, "default_proba": out.default_proba},
    )
    return out


# ─── Standalone decision systems (separate from ML model endpoints) ───────────


@app.post("/systems/shap/by-cin", response_model=ShapSystemResponse, tags=["Systems", "XAI"])
def system_shap_by_cin(
    payload: CinRequest,
    user: Annotated[dict[str, Any], Depends(require_staff)],
):
    """Système 1 — Explainable AI (SHAP). Autonome, sans sélection de modèle ML."""
    _load_lookup_tables()
    if _deep_tabular_model is None and _model is None:
        raise HTTPException(status_code=503, detail="Moteur XAI non disponible.")
    ctx = _build_system_context(payload)
    shap_model = _deep_tabular_model if _deep_tabular_model is not None else _model
    raw = run_shap_standalone(
        ctx, model=shap_model, model_name="XAI Engine", meta=_metadata,
        features_df=_features_df, credits_df=_credits_df,
    )
    _record_activity(user, "system_shap", cin=ctx["cin"], model="xai", extra={"risk_level": ctx["risk_level"]})
    cc = raw.get("credit_context") or {}
    return ShapSystemResponse(
        **_system_response_base(ctx),
        shap=ShapExplanation(
            method=raw["method"],
            model_used=raw["model_used"],
            base_prediction=raw.get("base_prediction"),
            increases_risk=[ShapFeatureImpact(**x) for x in raw["increases_risk"]],
            decreases_risk=[ShapFeatureImpact(**x) for x in raw["decreases_risk"]],
            driver_details=[ShapDriverDetail(**x) for x in raw.get("driver_details", [])],
            credit_context=CreditContext(**cc),
            summary=raw["summary"],
        ),
    )


@app.post("/systems/rules/by-cin", response_model=RulesSystemResponse, tags=["Systems", "Business"])
def system_rules_by_cin(
    payload: CinRequest,
    user: Annotated[dict[str, Any], Depends(require_staff)],
):
    """Système 2 — Business Rules Engine (conformité & politique crédit)."""
    ctx = _build_system_context(payload)
    raw = run_rules_standalone(ctx)
    _record_activity(user, "system_rules", cin=ctx["cin"], model="institutional")
    return RulesSystemResponse(
        **_system_response_base(ctx),
        business_rules=BusinessRulesBlock(
            rules=[BusinessRuleResult(**r) for r in raw["rules"]],
            triggered_count=raw["triggered_count"],
            triggered_rule_ids=raw.get("triggered_rule_ids", []),
            requires_manual_review=raw["requires_manual_review"],
            highest_severity=raw["highest_severity"],
            compliance_score=raw.get("compliance_score", 100),
            summary=raw["summary"],
            credit_snapshot=raw.get("credit_snapshot", ctx.get("credit_snapshot", {})),
        ),
    )


@app.post("/systems/early-warning/by-cin", response_model=EarlyWarningSystemResponse, tags=["Systems", "Business"])
def system_early_warning_by_cin(
    payload: CinRequest,
    user: Annotated[dict[str, Any], Depends(require_staff)],
):
    """Système 3 — Early Warning (surveillance comportementale)."""
    _load_lookup_tables()
    ctx = _build_system_context(payload)
    raw = run_ews_standalone(ctx, _features_df, _credits_df, _remb_df, _tx_df)
    _record_activity(user, "system_ews", cin=ctx["cin"], model="institutional")
    return EarlyWarningSystemResponse(
        **_system_response_base(ctx),
        early_warning=EarlyWarningBlock(
            alerts=[EarlyWarningAlert(**a) for a in raw["alerts"]],
            alert_count=raw["alert_count"],
            critical_count=raw["critical_count"],
            degradation_detected=raw["degradation_detected"],
            watchlist_priority=raw.get("watchlist_priority", "NONE"),
            trend_series=[
                TrendSeries(
                    metric=ts["metric"],
                    label=ts["label"],
                    points=[TrendPoint(**p) for p in ts["points"]],
                )
                for ts in raw.get("trend_series", [])
            ],
            n_credits_historique=raw.get("n_credits_historique", 0),
            summary=raw["summary"],
        ),
    )


@app.post("/systems/recommendation/by-cin", response_model=RecommendationSystemResponse, tags=["Systems", "Business"])
def system_recommendation_by_cin(
    payload: CinRequest,
    user: Annotated[dict[str, Any], Depends(require_staff)],
):
    """Système 4 — Recommandation IA (décision assistée)."""
    _load_lookup_tables()
    ctx = _build_system_context(payload)
    raw = run_recommendation_standalone(ctx, _features_df, _credits_df, _remb_df, _tx_df)
    _record_activity(
        user,
        "system_recommendation",
        cin=ctx["cin"],
        model="institutional",
        extra={"decision": raw["decision"]},
    )
    cf = raw.get("contributing_factors", {})
    return RecommendationSystemResponse(
        **_system_response_base(ctx),
        recommendation=AiRecommendation(
            decision=raw["decision"],
            decision_label=raw["decision_label"],
            confidence=raw["confidence"],
            justification=raw["justification"],
            recommended_actions=raw["recommended_actions"],
            requires_manual_validation=raw["requires_manual_validation"],
            suggested_montant=raw.get("suggested_montant"),
            montant_reduction_pct=raw.get("montant_reduction_pct"),
            suggested_dti_target=raw.get("suggested_dti_target"),
            monitoring_frequency=raw.get("monitoring_frequency", "trimestriel"),
            conditions=raw.get("conditions", []),
            contributing_factors=ContributingFactors(**cf) if cf else ContributingFactors(),
        ),
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
        if _graph_edge_weight is not None and hasattr(_graph_model, "forward"):
            logits = _graph_model(_graph_x, _graph_edge_index, _graph_edge_weight).cpu().numpy().ravel()
        else:
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


def _explain_graph_by_cin_core(payload: CinRequest) -> GraphExplanationByCinResponse:
    """
    Graph prediction + LLM explanation using only CIN (core logic).
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

    network_snapshot = _build_graph_network_snapshot(int(client_id)) if client_id is not None else None
    if network_snapshot:
        features_for_llm["Voisins affichés (réseau)"] = network_snapshot.stats.get("displayed_neighbors")
        features_for_llm["Degré réseau total"] = network_snapshot.stats.get("degree")

    if os.getenv("DISABLE_LLM", "").strip() in {"1", "true", "True", "yes", "YES"}:
        message = "Explication LLM désactivée (mode rapide)."
    else:
        try:
            rag_ctx, _ = _rag_context_for_scoring(
                risk_level=base.risk_level,
                model_used=base.model_used,
                kyc_score=base.kyc_score,
                cin=cin,
            )
            message = generate_risk_explanation(
                risk_level=base.risk_level,
                default_proba=base.default_proba,
                features=features_for_llm,
                rag_context=rag_ctx or None,
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
        network=network_snapshot,
    )


@app.post("/explain/graph/by-cin", response_model=GraphExplanationByCinResponse, tags=["Prediction", "Graph", "LLM"])
def explain_graph_by_cin(
    payload: CinRequest,
    user: Annotated[dict[str, Any], Depends(get_current_user)],
):
    out = _explain_graph_by_cin_core(payload)
    _record_activity(
        user,
        "explain_graph",
        cin=out.cin,
        model=out.model_used,
        extra={"risk_level": out.risk_level, "default_proba": out.default_proba},
    )
    return out


def _sequential_worst_proba(payload: CinRequest) -> float:
    """Probabilité séquentielle la plus défavorable (pire crédit du client)."""
    _load_lookup_tables()
    _load_sequential_artifacts()
    assert _clients_df is not None and _credits_df is not None and _seq_model is not None

    cin = _normalize_cin(payload.cin)
    match = _clients_df[_clients_df["cin"].astype(str) == cin]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"CIN not found: {cin}")
    client_id = int(match.iloc[0]["client_id"])

    client_credits = _credits_df[_credits_df["client_id"] == client_id].copy()
    if client_credits.empty:
        raise HTTPException(status_code=404, detail=f"No credits found for CIN: {cin}")

    import torch

    worst_proba = 0.0
    for _, cr in client_credits.iterrows():
        start = pd.to_datetime(cr["date_debut"])
        x = _build_seq_for_client_credit(client_id, start)
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = _seq_model(x_t).cpu().numpy().ravel()[0]
        proba = float(1.0 / (1.0 + np.exp(-logits)))
        worst_proba = max(worst_proba, proba)
    return worst_proba


def _collect_ensemble_model_scores(payload: CinRequest) -> tuple[list[ModelScoreInput], dict[str, Any]]:
    """Exécute les 3 modèles et retourne les scores individuels + contexte."""
    ctx = _resolve_cin_credit_context(payload)
    scores: list[ModelScoreInput] = []
    extras: dict[str, Any] = {"cin": ctx["cin"], "credit_id": ctx["credit_id"], "kyc_score": ctx["kyc_score"]}

    try:
        classic = _explain_risk_by_cin_core(payload)
        scores.append(
            ModelScoreInput(
                key="classic",
                name=MODEL_LABELS["classic"],
                proba=float(classic.default_proba),
            )
        )
    except HTTPException as exc:
        scores.append(
            ModelScoreInput(
                key="classic",
                name=MODEL_LABELS["classic"],
                proba=None,
                available=False,
                error=str(exc.detail),
            )
        )
    except Exception as exc:
        scores.append(
            ModelScoreInput(
                key="classic",
                name=MODEL_LABELS["classic"],
                proba=None,
                available=False,
                error=str(exc),
            )
        )

    try:
        seq_proba = _sequential_worst_proba(payload)
        _load_sequential_artifacts()
        seq_name = _seq_model_name or MODEL_LABELS["sequential"]
        scores.append(ModelScoreInput(key="sequential", name=seq_name, proba=seq_proba))
    except HTTPException as exc:
        scores.append(
            ModelScoreInput(
                key="sequential",
                name=MODEL_LABELS["sequential"],
                proba=None,
                available=False,
                error=str(exc.detail),
            )
        )
    except Exception as exc:
        scores.append(
            ModelScoreInput(
                key="sequential",
                name=MODEL_LABELS["sequential"],
                proba=None,
                available=False,
                error=str(exc),
            )
        )

    try:
        graph = predict_graph_by_cin(payload)
        scores.append(
            ModelScoreInput(
                key="graph",
                name=graph.model_used or MODEL_LABELS["graph"],
                proba=float(graph.default_proba),
            )
        )
        extras["graph_cin"] = graph.cin
    except HTTPException as exc:
        scores.append(
            ModelScoreInput(
                key="graph",
                name=MODEL_LABELS["graph"],
                proba=None,
                available=False,
                error=str(exc.detail),
            )
        )
    except Exception as exc:
        scores.append(
            ModelScoreInput(
                key="graph",
                name=MODEL_LABELS["graph"],
                proba=None,
                available=False,
                error=str(exc),
            )
        )

    return scores, extras


def _explain_ensemble_by_cin_core(payload: CinRequest) -> EnsembleExplanationByCinResponse:
    """Fusion ensemble + explication LLM."""
    scores, extras = _collect_ensemble_model_scores(payload)
    try:
        merged = compute_ensemble(scores)
    except ValueError as exc:
        if str(exc) == "no_models_available":
            raise HTTPException(status_code=503, detail="Aucun modèle ML disponible pour l'ensemble.") from exc
        raise

    cin = extras["cin"]
    kyc_score = round(float(extras["kyc_score"]), 2)
    network_snapshot = None
    try:
        _load_lookup_tables()
        match = _clients_df[_clients_df["cin"].astype(str) == _normalize_cin(cin)]
        if not match.empty:
            network_snapshot = _build_graph_network_snapshot(int(match.iloc[0]["client_id"]))
    except Exception:
        network_snapshot = None

    model_lines = []
    for m in merged["models"]:
        if m.get("available"):
            model_lines.append(
                f"- {m['model_name']} (poids {m['weight']*100:.0f} %) : "
                f"{m['default_proba']:.1%} — {m['risk_level']}"
            )
        else:
            model_lines.append(f"- {m['model_name']} : indisponible ({m.get('error') or 'erreur'})")

    features_for_llm = {
        "CIN": cin,
        "Score KYC": kyc_score,
        "Score ensemble (pondéré AUC)": merged["default_proba"],
        "Niveau de risque ensemble": merged["risk_level"],
        "Vote défaut / non-défaut": f"{merged['vote_default']} / {merged['vote_non_default']}",
        "Accord entre modèles": merged["agreement"],
        "Modèles disponibles": f"{merged['models_available']}/{merged['models_total']}",
        "Détail par modèle": "\n".join(model_lines),
    }

    if os.getenv("DISABLE_LLM", "").strip() in {"1", "true", "True", "yes", "YES"}:
        message = (
            f"Score ensemble : {merged['default_proba']:.1%} ({merged['risk_level']}). "
            f"Vote : {merged['vote_default']} défaut / {merged['vote_non_default']} non-défaut."
        )
    else:
        try:
            rag_ctx, _ = _rag_context_for_scoring(
                risk_level=merged["risk_level"],
                model_used=merged["model_used"],
                kyc_score=kyc_score,
                cin=cin,
            )
            message = generate_risk_explanation(
                risk_level=merged["risk_level"],
                default_proba=merged["default_proba"],
                features=features_for_llm,
                rag_context=rag_ctx or None,
            )
        except Exception as exc:
            message = (
                f"Score ensemble : {merged['default_proba']:.1%} ({merged['risk_level']}). "
                f"Vote : {merged['vote_default']} défaut / {merged['vote_non_default']} non-défaut. "
                f"(LLM indisponible : {exc})"
            )

    return EnsembleExplanationByCinResponse(
        cin=cin,
        credit_id=extras.get("credit_id"),
        kyc_score=kyc_score,
        default_proba=merged["default_proba"],
        risk_level=merged["risk_level"],
        prediction=merged["prediction"],
        model_used=merged["model_used"],
        method=merged["method"],
        models=[EnsembleModelScore(**m) for m in merged["models"]],
        vote_default=merged["vote_default"],
        vote_non_default=merged["vote_non_default"],
        agreement=merged["agreement"],
        models_available=merged["models_available"],
        models_total=merged["models_total"],
        message=message,
        network=network_snapshot,
    )


@app.post("/explain/ensemble/by-cin", response_model=EnsembleExplanationByCinResponse, tags=["Prediction", "Ensemble", "LLM"])
def explain_ensemble_by_cin(
    payload: CinRequest,
    user: Annotated[dict[str, Any], Depends(get_current_user)],
):
    """Score unifié — fusion pondérée Deep Tabular + GAT + Transformer + explication LLM."""
    out = _explain_ensemble_by_cin_core(payload)
    _record_activity(
        user,
        "explain_ensemble",
        cin=out.cin,
        model=out.model_used,
        extra={
            "risk_level": out.risk_level,
            "default_proba": out.default_proba,
            "agreement": out.agreement,
            "vote_default": out.vote_default,
        },
    )
    return out


def _explain_sequential_by_cin_core(payload: CinRequest) -> SequentialExplanationByCinAllCreditsResponse:
    """
    Sequential prediction + LLM explanation using only CIN (core logic).
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

    if os.getenv("DISABLE_LLM", "").strip() in {"1", "true", "True", "yes", "YES"}:
        message = "Explication LLM désactivée (mode rapide)."
    else:
        try:
            rag_ctx, _ = _rag_context_for_scoring(
                risk_level=worst_risk,
                model_used=_seq_model_name,
                kyc_score=kyc_score_val,
                cin=cin,
            )
            message = generate_risk_explanation(
                risk_level=worst_risk,
                default_proba=float(worst_proba),
                features=features_for_llm,
                rag_context=rag_ctx or None,
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


@app.post(
    "/explain/sequential/by-cin",
    response_model=SequentialExplanationByCinAllCreditsResponse,
    tags=["Prediction", "Sequential", "LLM"],
)
def explain_sequential_by_cin(
    payload: CinRequest,
    user: Annotated[dict[str, Any], Depends(get_current_user)],
):
    out = _explain_sequential_by_cin_core(payload)
    _record_activity(
        user,
        "explain_sequential",
        cin=out.cin,
        model=out.model_used,
        extra={
            "risk_level": out.risk_level,
            "default_proba": out.default_proba,
            "n_credits": out.n_credits,
        },
    )
    return out


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


def _run_classic_for_orchestrator(cin: str, credit_id: int | None):
    old_disable = os.getenv("DISABLE_LLM")
    os.environ["DISABLE_LLM"] = "1"
    try:
        return _explain_risk_by_cin_core(CinRequest(cin=cin, credit_id=credit_id))
    finally:
        if old_disable is None:
            os.environ.pop("DISABLE_LLM", None)
        else:
            os.environ["DISABLE_LLM"] = old_disable


def _run_sequential_for_orchestrator(cin: str, credit_id: int | None):
    old_disable = os.getenv("DISABLE_LLM")
    os.environ["DISABLE_LLM"] = "1"
    try:
        return _explain_sequential_by_cin_core(CinRequest(cin=cin, credit_id=credit_id))
    finally:
        if old_disable is None:
            os.environ.pop("DISABLE_LLM", None)
        else:
            os.environ["DISABLE_LLM"] = old_disable


def _run_graph_for_orchestrator(cin: str, credit_id: int | None):
    old_disable = os.getenv("DISABLE_LLM")
    os.environ["DISABLE_LLM"] = "1"
    try:
        return _explain_graph_by_cin_core(CinRequest(cin=cin, credit_id=credit_id))
    finally:
        if old_disable is None:
            os.environ.pop("DISABLE_LLM", None)
        else:
            os.environ["DISABLE_LLM"] = old_disable


def _rag_retrieve_for_orchestrator(queries: list[str], k: int) -> list[dict]:
    return rag_retrieve_multi(queries or build_rag_query(), k=k)


def _run_systems_for_orchestrator(cin: str) -> dict[str, Any]:
    ctx = _build_system_context(CinRequest(cin=cin))
    return fetch_systems_from_ctx(ctx, _features_df, _credits_df, _remb_df, _tx_df)


def _build_chat_structured(out: dict[str, Any]) -> ChatStructuredResult | None:
    intent = out.get("intent")
    if intent == "institutional":
        systems = out.get("systems_result") or {}
        profile = systems.get("client_profile") or {}
        return ChatStructuredResult(
            institutional_score=systems.get("institutional_score"),
            institutional_risk=systems.get("risk_level"),
            kyc_score=profile.get("kyc_score") if profile.get("kyc_score") else None,
        )
    if intent == "sequential_score":
        primary = out.get("sequential_result") or {}
    elif intent == "graph_score":
        primary = out.get("graph_result") or {}
    else:
        primary = out.get("classic_result") or {}
    if not primary:
        return None
    return ChatStructuredResult(
        kyc_score=primary.get("kyc_score"),
        default_proba=primary.get("default_proba"),
        risk_level=primary.get("risk_level"),
        model_used=primary.get("model_used") or out.get("model_selected"),
    )


def _suggested_prompts(cin: str | None) -> list[str]:
    c = cin or "88710263"
    return [
        f"Analyse institutionnelle du CIN {c}",
        f"Rapport complet pour le CIN {c}",
        f"Compare les modèles ML pour {c}",
        f"Score séquentiel du CIN {c}",
    ]


def _get_agent_orchestrator() -> CreditAgentOrchestrator:
    global _agent_orchestrator
    if _agent_orchestrator is None:
        _agent_orchestrator = CreditAgentOrchestrator(
            run_classic=_run_classic_for_orchestrator,
            run_sequential=_run_sequential_for_orchestrator,
            run_graph=_run_graph_for_orchestrator,
            rag_retrieve=_rag_retrieve_for_orchestrator,
            run_systems=_run_systems_for_orchestrator,
        )
    return _agent_orchestrator


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
def chat(
    payload: ChatRequest,
    user: Annotated[dict[str, Any], Depends(get_current_user)],
):
    """
    Professional LangGraph orchestration (auth required).
    """
    _load_lookup_tables()
    session_id = payload.session_id.strip()
    user_msg = payload.message.strip()

    graph = _get_agent_orchestrator()
    history_ctx = get_conversation_context(user_id=user["id"], session_id=session_id, limit=8)
    out = graph.invoke(session_id=session_id, message=user_msg, conversation_history=history_ctx)

    answer = str(out.get("final_answer", "")).strip()
    cin = out.get("cin")
    model_selected = out.get("model_selected")
    intent = out.get("intent")

    append_chat_messages(
        user_id=user["id"],
        session_id=session_id,
        messages=[
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ],
        cin=str(cin) if cin else None,
        intent=str(intent) if intent else None,
        title=user_msg[:60] + ("…" if len(user_msg) > 60 else ""),
    )

    _record_activity(
        user,
        "chat",
        cin=str(cin) if cin else None,
        model=model_selected,
        intent=intent,
        message=user_msg,
        session_id=session_id,
    )

    rag_sources = [
        RagSourceItem(
            source=str(s.get("source", "")),
            chunk_id=int(s.get("chunk_id", 0)),
            score=float(s.get("score", 0)),
            text=str(s.get("text", ""))[:500],
        )
        for s in (out.get("rag_sources") or [])
    ]

    return ChatResponse(
        session_id=session_id,
        model_selected=model_selected,
        cin=cin,
        intent=intent,
        answer=answer,
        rag_sources=rag_sources,
        structured=_build_chat_structured(out),
        systems=out.get("systems_result"),
        report_available=bool(out.get("report_markdown")),
        suggested_prompts=_suggested_prompts(str(cin) if cin else None),
    )


@app.post("/report/by-cin", response_model=ReportResponse, tags=["Report", "RAG"])
def report_by_cin(
    payload: ReportRequest,
    user: Annotated[dict[str, Any], Depends(get_current_user)],
):
    """
    Generate a Markdown report from CIN, augmented by local document retrieval (RAG).
    """
    cin = _normalize_cin(payload.cin)
    markdown, sources, _, structured = _generate_report_markdown(cin, analyst_username=user.get("username"))
    _record_activity(user, "report", cin=cin, message=f"report {cin}")
    return ReportResponse(cin=cin, markdown=markdown, sources=sources, structured=structured)


def _generate_report_markdown(cin: str, *, analyst_username: str | None = None) -> tuple[str, list[dict], Path, dict]:
    _load_lookup_tables()

    old_disable = os.getenv("DISABLE_LLM")
    os.environ["DISABLE_LLM"] = "1"
    try:
        classic = _explain_risk_by_cin_core(CinRequest(cin=cin)).model_dump()
        seq = _explain_sequential_by_cin_core(CinRequest(cin=cin)).model_dump()
        graph = _explain_graph_by_cin_core(CinRequest(cin=cin)).model_dump()
    finally:
        if old_disable is None:
            os.environ.pop("DISABLE_LLM", None)
        else:
            os.environ["DISABLE_LLM"] = old_disable

    queries = build_rag_query(
        cin=cin,
        risk_level=str(seq.get("risk_level") or classic.get("risk_level")),
        model_used="classique + séquentiel + graphe",
        kyc_score=classic.get("kyc_score"),
        extra_topics=[
            "rapport risque crédit mots-clés conclusion recommandation",
            f"probabilité défaut {classic.get('default_proba')}",
        ],
    )
    sources = rag_retrieve_multi(queries, k=8)
    ctx = _build_system_context(CinRequest(cin=cin))
    systems = fetch_systems_from_ctx(ctx, _features_df, _credits_df, _remb_df, _tx_df)
    structured = build_structured_report(
        cin=cin,
        classic=classic,
        sequential=seq,
        graph=graph,
        sources=sources,
        analyst_username=analyst_username,
        systems=systems,
    )

    rag_block = format_rag_sources_for_prompt(sources, max_chars=4500)
    systems_block = json.dumps(systems, ensure_ascii=False)
    report_prompt = f"""
Tu es analyste risque crédit microfinance (Talys).
Génère un rapport Markdown professionnel en français.
Sections OBLIGATOIRES (##): Mots-clés, Résumé exécutif, Systèmes institutionnels,
Résultat classique, Résultat séquentiel, Résultat graphe, Analyse métier, Recommandation, Conclusion, Références.

Conclusion imposée (reformuler): {structured['conclusion']}
Mots-clés: {", ".join(structured['keywords'])}

Systèmes institutionnels: {systems_block}

JSON classique: {json.dumps(classic, ensure_ascii=False)}
JSON séquentiel: {json.dumps(seq, ensure_ascii=False)}
JSON graphe: {json.dumps(graph, ensure_ascii=False)}

{rag_block}
"""

    markdown: str | None = None
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            temperature=0.25,
            sync_client_kwargs={"timeout": 45.0},
            async_client_kwargs={"timeout": 45.0},
        )
        msg = llm.invoke([SystemMessage(content="Rapport comité crédit, sections strictes."), HumanMessage(content=report_prompt)])
        markdown = str(getattr(msg, "content", msg)).strip()
    except Exception:
        markdown = structured_to_markdown(structured)

    out_dir = REPORTS_DIR / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"report_{cin}.md"
    out_path.write_text(markdown, encoding="utf-8")
    return markdown, sources, out_path, structured


@app.post("/report/by-cin/download", tags=["Report", "RAG"])
def download_report_by_cin(
    payload: ReportDownloadRequest,
    user: Annotated[dict[str, Any], Depends(get_current_user)],
):
    """
    Generate a report by CIN and return a downloadable file (.md or .pdf).
    """
    cin = _normalize_cin(payload.cin)
    _, _, md_path, structured = _generate_report_markdown(cin, analyst_username=user.get("username"))
    _record_activity(user, "report_download", cin=cin, extra={"format": payload.format})

    fmt = payload.format.lower().strip()
    if fmt == "md":
        return FileResponse(
            path=str(md_path),
            media_type="text/markdown; charset=utf-8",
            filename=md_path.name,
        )

    pdf_path = md_path.with_suffix(".pdf")
    logo_path = REPORTS_DIR / "assets" / "talys_logo.png"
    try:
        export_structured_pdf(structured, pdf_path, logo_path=logo_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur export PDF: {exc}") from exc
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )
