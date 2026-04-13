"""
KYC score estimation – computes a 0–100 score from client attributes.

The KYC model is trained to predict statut_kyc from (age, revenu, profession).
At inference, we estimate the KYC score without using statut_kyc directly.

Usage:
    python -m src.kyc.score   # fit and save the KYC model
    from src.kyc.score import compute_kyc_score
    score = compute_kyc_score(age=35, revenu_mensuel=2000, profession_enc=2)
"""
from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, CAT_PROFESSION, CAT_STATUT_KYC

KYC_MODEL_FILE = MODELS_DIR / "kyc_scorer.joblib"
KYC_METADATA_FILE = MODELS_DIR / "kyc_metadata.json"

_kyc_pipeline = None


def _profession_to_enc(profession: str) -> int:
    """Map profession string to 0–7."""
    prof_map = {p: i for i, p in enumerate(CAT_PROFESSION)}
    return int(prof_map.get(str(profession), 0))


def _statut_kyc_to_class(statut: str) -> int:
    """Map statut_kyc to class: RISQUE=0, A_VERIFIER=1, OK=2."""
    m = {"RISQUE": 0, "A_VERIFIER": 1, "OK": 2}
    return m.get(str(statut), 1)


def fit_kyc_scorer(clients: pd.DataFrame) -> Path:
    """
    Train a KYC scorer to predict statut_kyc from age, revenu_mensuel, profession.
    Saves a pipeline (scaler + classifier) that outputs a 0–100 score.
    """
    df = clients[["age", "revenu_mensuel", "profession", "statut_kyc"]].dropna()
    df = df[df["statut_kyc"].isin(CAT_STATUT_KYC)]

    X = pd.DataFrame({
        "age": df["age"].astype(float),
        "revenu_mensuel": df["revenu_mensuel"].astype(float),
        "profession_enc": df["profession"].map(lambda p: _profession_to_enc(p)),
    })
    y = df["statut_kyc"].map(_statut_kyc_to_class).values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_scaled, y)

    pipeline = {"scaler": scaler, "model": clf}
    joblib.dump(pipeline, KYC_MODEL_FILE)

    meta = {
        "feature_order": ["age", "revenu_mensuel", "profession_enc"],
        "description": "KYC score 0-100 estimated from client attributes",
    }
    with open(KYC_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"KYC scorer saved at {KYC_MODEL_FILE}")
    return KYC_MODEL_FILE


def _load_kyc_pipeline():
    global _kyc_pipeline
    if _kyc_pipeline is None:
        if not KYC_MODEL_FILE.exists():
            raise FileNotFoundError(
                f"KYC model not found at {KYC_MODEL_FILE}. "
                "Run `python -m src.kyc.score` first."
            )
        _kyc_pipeline = joblib.load(KYC_MODEL_FILE)
    return _kyc_pipeline


def compute_kyc_score(
    age: float,
    revenu_mensuel: float,
    profession_enc: int,
) -> float:
    """
    Estimate KYC score (0–100) from client attributes.

    Parameters
    ----------
    age : float
    revenu_mensuel : float
    profession_enc : int
        Profession encoding 0–7 (see config CAT_PROFESSION)

    Returns
    -------
    float
        KYC score in [0, 100]
    """
    pipe = _load_kyc_pipeline()
    X = np.array([[float(age), float(revenu_mensuel), int(profession_enc)]], dtype=np.float32)
    X_scaled = pipe["scaler"].transform(X)
    proba = pipe["model"].predict_proba(X_scaled)[0]
    # score = 100*P(OK) + 50*P(A_VERIFIER) + 0*P(RISQUE), classes 0=RISQUE, 1=A_VERIFIER, 2=OK
    if proba.shape[0] >= 3:
        raw = 100 * proba[2] + 50 * proba[1] + 0 * proba[0]
    else:
        raw = 100 * proba[-1]
    return round(float(np.clip(raw, 0, 100)), 2)


def compute_kyc_score_row(row: pd.Series) -> float:
    """Compute KYC score from a client row (with profession string)."""
    prof_enc = _profession_to_enc(row.get("profession", "Employé"))
    return compute_kyc_score(
        age=float(row["age"]),
        revenu_mensuel=float(row["revenu_mensuel"]),
        profession_enc=prof_enc,
    )


if __name__ == "__main__":
    from src.data.loader import load_raw_data
    from src.data.cleaner import clean_and_save

    raw = load_raw_data()
    data = clean_and_save(raw)
    fit_kyc_scorer(data["clients"])
