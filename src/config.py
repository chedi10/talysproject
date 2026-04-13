"""
config.py – Central configuration for the credit default risk project.
All paths and constants are defined here and imported everywhere else.
"""
from pathlib import Path

# ─── Root ───────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent   # Talys_pfe/

# ─── Data directories ────────────────────────────────────────────────────────
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
FEATURES_DIR   = DATA_DIR / "features"

# ─── Output directories ──────────────────────────────────────────────────────
MODELS_DIR     = ROOT_DIR / "models"
REPORTS_DIR    = ROOT_DIR / "reports"
FIGURES_DIR    = REPORTS_DIR / "figures"

# ─── Create missing directories at import time ───────────────────────────────
for _d in [PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Raw file paths ──────────────────────────────────────────────────────────
RAW_CLIENTS        = RAW_DIR / "clients.csv"
RAW_CREDITS        = RAW_DIR / "credits.csv"
RAW_REMBOURSEMENTS = RAW_DIR / "remboursements.csv"
RAW_TRANSACTIONS   = RAW_DIR / "transactions.csv"
RAW_RELATIONS      = RAW_DIR / "relations.csv"

# ─── Feature matrix output ───────────────────────────────────────────────────
FEATURES_FILE  = FEATURES_DIR / "features.parquet"

# ─── Model artifacts ─────────────────────────────────────────────────────────
BEST_MODEL_FILE     = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# ─── Training ────────────────────────────────────────────────────────────────
TARGET_COLUMN  = "en_defaut"
TEST_SIZE      = 0.20
RANDOM_STATE   = 42

# ─── Categorical columns and their possible values (from generate_data.py) ───
CAT_STATUT_KYC = ["OK", "A_VERIFIER", "RISQUE"]
CAT_CYCLE      = ["CYCLE_1", "CYCLE_2", "CYCLE_3", "CYCLE_4"]
CAT_OBJET      = ["CONSOMMATION", "MICRO_ENTREPRISE", "SANTE", "EDUCATION", "LOGEMENT"]
CAT_PROFESSION = ["Etudiant", "Ouvrier", "Employé", "Commerçant", "Fonctionnaire",
                   "Cadre", "Indépendant", "Retraité"]
CAT_VILLE      = ["Tunis", "Ariana", "Ben Arous", "Sousse", "Sfax",
                  "Nabeul", "Bizerte", "Monastir", "Kairouan", "Gabès"]
