"""Data access — load business tables as pandas DataFrames from SQLite."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from src.db.engine import engine


def load_clients_df() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM clients", engine, dtype={"cin": str})


def load_credits_df() -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM credits", engine)
    if "date_debut" in df.columns:
        df["date_debut"] = pd.to_datetime(df["date_debut"])
    return df


def load_transactions_df() -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM transactions", engine)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_remboursements_df() -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM remboursements", engine)
    for col in ("date_echeance", "date_paiement"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def load_relations_df() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM relations", engine)


def get_client_by_cin(cin: str) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM clients WHERE cin = :cin LIMIT 1"),
            {"cin": cin},
        ).mappings().first()
    return dict(row) if row else None


def get_client_credits(client_id: int) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM credits WHERE client_id = :cid ORDER BY date_debut DESC"),
            {"cid": client_id},
        ).mappings().all()
    return [dict(r) for r in rows]


def db_stats() -> dict:
    with engine.connect() as conn:
        stats = {}
        for table in ("clients", "credits", "transactions", "remboursements", "relations", "users", "activity_log", "chat_sessions"):
            n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            stats[table] = int(n or 0)
    return stats


def admin_extended_stats() -> dict:
    """Statistiques enrichies pour le tableau de bord admin."""
    base = db_stats()
    with engine.connect() as conn:
        users_by_role = {
            str(r[0]): int(r[1])
            for r in conn.execute(text("SELECT role, COUNT(*) FROM users GROUP BY role")).all()
        }
        credits_default = int(conn.execute(text("SELECT COUNT(*) FROM credits WHERE en_defaut = 1")).scalar() or 0)
        credits_total = int(conn.execute(text("SELECT COUNT(*) FROM credits")).scalar() or 0)
        kyc_breakdown = {
            str(r[0]): int(r[1])
            for r in conn.execute(text("SELECT statut_kyc, COUNT(*) FROM clients GROUP BY statut_kyc")).all()
        }
        activity_7d = int(
            conn.execute(
                text("SELECT COUNT(*) FROM activity_log WHERE created_at >= datetime('now', '-7 days')")
            ).scalar()
            or 0
        )
        action_rows = conn.execute(
            text(
                "SELECT action, COUNT(*) AS n FROM activity_log "
                "GROUP BY action ORDER BY n DESC LIMIT 8"
            )
        ).all()
        activity_by_action = {str(r[0]): int(r[1]) for r in action_rows}

    default_rate = round(credits_default / max(credits_total, 1), 4)
    return {
        **base,
        "users_by_role": users_by_role,
        "credits_en_defaut": credits_default,
        "default_rate": default_rate,
        "kyc_breakdown": kyc_breakdown,
        "activity_last_7_days": activity_7d,
        "activity_by_action": activity_by_action,
    }


def get_client_remboursements(client_id: int) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM remboursements WHERE client_id = :cid "
                "ORDER BY date_echeance DESC LIMIT 120"
            ),
            {"cid": client_id},
        ).mappings().all()
    return [dict(r) for r in rows]


def get_client_transactions_summary(client_id: int) -> dict:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT COUNT(*) AS n_tx, COALESCE(SUM(suspect), 0) AS n_suspect "
                "FROM transactions WHERE client_id = :cid"
            ),
            {"cid": client_id},
        ).mappings().first()
    return dict(row) if row else {"n_tx": 0, "n_suspect": 0}
