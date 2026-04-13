"""
train.py – Train and compare Logistic Regression, Random Forest, XGBoost.

Usage:
    python -m src.models.train
"""
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    FEATURES_FILE, MODELS_DIR, BEST_MODEL_FILE,
    MODEL_METADATA_FILE, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
)
from src.models.evaluate import evaluate_model, compare_models


# ─── Feature columns (everything except credit_id and target) ────────────────
def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {TARGET_COLUMN, "credit_id"}
    return [c for c in df.columns if c not in drop]


# ─── Model definitions ────────────────────────────────────────────────────────
def _build_models(scale_pos_weight: float) -> dict:
    """
    Returns a dict of {name: sklearn Pipeline or estimator}.
    scale_pos_weight = n_negative / n_positive (for XGBoost).
    class_weight='balanced' handles imbalance in LR and RF.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,   # handles class imbalance
            eval_metric="auc",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ─── Main training function ───────────────────────────────────────────────────
def train():
    # 1. Load feature matrix
    print("Loading feature matrix...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    feature_cols = _get_feature_cols(df)
    X = df[feature_cols].values
    y = df[TARGET_COLUMN].values

    # Class imbalance ratio for XGBoost
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = round(n_neg / n_pos, 2)
    print(f"   Class balance  : {n_neg:,} non-défauts  |  {n_pos:,} défauts  "
          f"(scale_pos_weight={scale_pos_weight})")

    # 2. Train/test split – stratified to preserve default rate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"   Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}\n")

    models = _build_models(scale_pos_weight)
    results = []

    # 3. Train, evaluate, and save each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        result = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(result)

        # Save individual model
        model_path = MODELS_DIR / f"{name.replace(' ', '_').lower()}.joblib"
        joblib.dump(model, model_path)
        print(f"  Model saved at {model_path}")

    # 4. Compare and pick best
    compare_models(results)
    best = max(results, key=lambda r: r["auc_roc"])
    best_name = best["model_name"]
    best_model_src = MODELS_DIR / f"{best_name.replace(' ', '_').lower()}.joblib"

    # Save best model alias
    best_model = joblib.load(best_model_src)
    joblib.dump(best_model, BEST_MODEL_FILE)
    print(f"\nBest model: {best_name}  (AUC={best['auc_roc']})")
    print(f"   Saved at {BEST_MODEL_FILE}")

    # 5. Save metadata JSON (used by the API)
    metadata = {
        "best_model_name": best_name,
        "auc_roc":         best["auc_roc"],
        "avg_precision":   best["avg_precision"],
        "feature_columns": feature_cols,
        "target_column":   TARGET_COLUMN,
        "train_rows":      int(X_train.shape[0]),
        "test_rows":       int(X_test.shape[0]),
        "default_rate":    round(float(n_pos / len(y)), 4),
    }
    with open(MODEL_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved at {MODEL_METADATA_FILE}")


if __name__ == "__main__":
    # Run feature engineering first if features.parquet doesn't exist
    if not FEATURES_FILE.exists():
        # Avoid Unicode icons that break on Windows cp1252 consoles
        print("Feature matrix not found - running feature engineering first...\n")
        from src.features.engineering import build_features
        build_features()

    train()
