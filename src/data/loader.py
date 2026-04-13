"""
loader.py – Load and type-cast the 5 raw CSV files.

Usage:
    from src.data.loader import load_raw_data
    data = load_raw_data()
    df_clients = data["clients"]
"""
import pandas as pd
from src.config import (
    RAW_CLIENTS, RAW_CREDITS, RAW_REMBOURSEMENTS,
    RAW_TRANSACTIONS, RAW_RELATIONS,
)


def load_raw_data() -> dict[str, pd.DataFrame]:
    """
    Load all 5 raw CSV files and apply basic type casting.

    Returns
    -------
    dict with keys: clients, credits, remboursements, transactions, relations
    """
    # Avoid Unicode icons that break on Windows cp1252 consoles
    print("Loading raw CSV files...")

    # ── Clients ──────────────────────────────────────────────────────────────
    clients = pd.read_csv(RAW_CLIENTS, parse_dates=["date_creation"])
    clients["client_id"]       = clients["client_id"].astype(int)
    clients["age"]             = clients["age"].astype(int)
    clients["revenu_mensuel"]  = clients["revenu_mensuel"].astype(float)
    clients["statut_kyc"]      = clients["statut_kyc"].astype("category")
    clients["profession"]      = clients["profession"].astype("category")
    clients["ville"]           = clients["ville"].astype("category")
    print(f"   clients        : {clients.shape[0]:>7,} rows × {clients.shape[1]} cols")

    # ── Credits ──────────────────────────────────────────────────────────────
    credits = pd.read_csv(RAW_CREDITS, parse_dates=["date_debut"])
    credits["credit_id"]  = credits["credit_id"].astype(int)
    credits["client_id"]  = credits["client_id"].astype(int)
    credits["montant"]    = credits["montant"].astype(float)
    credits["duree_mois"] = credits["duree_mois"].astype(int)
    credits["dti"]        = credits["dti"].astype(float)
    credits["en_defaut"]  = credits["en_defaut"].astype(int)
    credits["cycle"]      = credits["cycle"].astype("category")
    credits["objet"]      = credits["objet"].astype("category")
    print(f"   credits        : {credits.shape[0]:>7,} rows × {credits.shape[1]} cols  "
          f"| défauts: {credits['en_defaut'].mean()*100:.1f}%")

    # ── Remboursements ───────────────────────────────────────────────────────
    remb = pd.read_csv(
        RAW_REMBOURSEMENTS,
        parse_dates=["date_echeance", "date_paiement"],
    )
    remb["remb_id"]     = remb["remb_id"].astype(int)
    remb["credit_id"]   = remb["credit_id"].astype(int)
    remb["client_id"]   = remb["client_id"].astype(int)
    remb["retard_jours"] = remb["retard_jours"].astype(int)
    remb["montant_du"]  = remb["montant_du"].astype(float)
    remb["statut"]      = remb["statut"].astype("category")
    print(f"   remboursements : {remb.shape[0]:>7,} rows × {remb.shape[1]} cols")

    # ── Transactions ──────────────────────────────────────────────────────────
    tx = pd.read_csv(RAW_TRANSACTIONS, parse_dates=["date"])
    tx["transaction_id"] = tx["transaction_id"].astype(int)
    tx["client_id"]      = tx["client_id"].astype(int)
    tx["montant"]        = tx["montant"].astype(float)
    tx["suspect"]        = tx["suspect"].astype(int)
    tx["type"]           = tx["type"].astype("category")
    print(f"   transactions   : {tx.shape[0]:>7,} rows × {tx.shape[1]} cols")

    # ── Relations ─────────────────────────────────────────────────────────────
    rel = pd.read_csv(RAW_RELATIONS)
    rel["relation_id"]       = rel["relation_id"].astype(int)
    rel["source_client_id"]  = rel["source_client_id"].astype(int)
    rel["target_client_id"]  = rel["target_client_id"].astype(int)
    rel["risk_relation"]     = rel["risk_relation"].astype(int)
    rel["type_relation"]     = rel["type_relation"].astype("category")
    print(f"   relations      : {rel.shape[0]:>7,} rows × {rel.shape[1]} cols")

    # Final status message without Unicode icons (Windows-safe)
    print("All files loaded.\n")
    return {
        "clients":        clients,
        "credits":        credits,
        "remboursements": remb,
        "transactions":   tx,
        "relations":      rel,
    }
