"""Seed SQLite from CSV + migrate legacy JSON auth."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select

from src.config import (
    LOCAL_DATA_DIR,
    RAW_CLIENTS,
    RAW_CREDITS,
    RAW_RELATIONS,
    RAW_REMBOURSEMENTS,
    RAW_TRANSACTIONS,
)
from src.db.engine import SessionLocal, init_db
from src.db.models import Client, Credit, Relation, Remboursement, Transaction, User


def _iso(val) -> str:
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


def seed_business_data(force: bool = False) -> dict[str, int]:
    init_db()
    db = SessionLocal()
    try:
        count = db.scalar(select(func.count()).select_from(Client)) or 0
        if count > 0 and not force:
            return {"clients": count, "skipped": True}

        if force:
            for table in (Remboursement, Transaction, Relation, Credit, Client):
                db.query(table).delete()
            db.commit()

        clients = pd.read_csv(RAW_CLIENTS, parse_dates=["date_creation"])
        credits = pd.read_csv(RAW_CREDITS, parse_dates=["date_debut"])
        remb = pd.read_csv(RAW_REMBOURSEMENTS, parse_dates=["date_echeance", "date_paiement"])
        tx = pd.read_csv(RAW_TRANSACTIONS, parse_dates=["date"])
        rel = pd.read_csv(RAW_RELATIONS)

        db.bulk_insert_mappings(
            Client,
            [
                {
                    "client_id": int(r.client_id),
                    "cin": str(r.cin),
                    "nom": str(r.nom),
                    "prenom": str(r.prenom),
                    "age": int(r.age),
                    "ville": str(r.ville),
                    "profession": str(r.profession),
                    "revenu_mensuel": float(r.revenu_mensuel),
                    "date_creation": _iso(r.date_creation),
                    "statut_kyc": str(r.statut_kyc),
                }
                for r in clients.itertuples(index=False)
            ],
        )
        db.bulk_insert_mappings(
            Credit,
            [
                {
                    "credit_id": int(r.credit_id),
                    "client_id": int(r.client_id),
                    "cycle": str(r.cycle),
                    "objet": str(r.objet),
                    "montant": float(r.montant),
                    "duree_mois": int(r.duree_mois),
                    "dti": float(r.dti),
                    "date_debut": _iso(r.date_debut),
                    "en_defaut": int(r.en_defaut),
                }
                for r in credits.itertuples(index=False)
            ],
        )
        db.bulk_insert_mappings(
            Remboursement,
            [
                {
                    "remb_id": int(r.remb_id),
                    "credit_id": int(r.credit_id),
                    "client_id": int(r.client_id),
                    "mois": int(r.mois),
                    "montant_du": float(r.montant_du),
                    "date_echeance": _iso(r.date_echeance),
                    "date_paiement": _iso(r.date_paiement),
                    "retard_jours": int(r.retard_jours),
                    "statut": str(r.statut),
                }
                for r in remb.itertuples(index=False)
            ],
        )
        db.bulk_insert_mappings(
            Transaction,
            [
                {
                    "transaction_id": int(r.transaction_id),
                    "client_id": int(r.client_id),
                    "type": str(r.type),
                    "montant": float(r.montant),
                    "date": _iso(r.date),
                    "suspect": int(r.suspect),
                }
                for r in tx.itertuples(index=False)
            ],
        )
        db.bulk_insert_mappings(
            Relation,
            [
                {
                    "relation_id": int(r.relation_id),
                    "source_client_id": int(r.source_client_id),
                    "target_client_id": int(r.target_client_id),
                    "type_relation": str(r.type_relation),
                    "risk_relation": int(r.risk_relation),
                }
                for r in rel.itertuples(index=False)
            ],
        )
        db.commit()
        return {
            "clients": len(clients),
            "credits": len(credits),
            "remboursements": len(remb),
            "transactions": len(tx),
            "relations": len(rel),
            "skipped": False,
        }
    finally:
        db.close()


def _migrate_json_auth() -> None:
    """Import users from legacy JSON if DB empty (sessions not migrated — re-login required)."""
    from datetime import datetime, timezone

    users_file = LOCAL_DATA_DIR / "users.json"
    if not users_file.exists():
        return

    db = SessionLocal()
    try:
        if db.scalar(select(func.count()).select_from(User)):
            return

        users = json.loads(users_file.read_text(encoding="utf-8"))
        for u in users:
            role = u.get("role", "agent")
            if role not in ("client", "agent", "admin"):
                role = "agent"
            db.add(
                User(
                    id=u["id"],
                    username=u["username"],
                    email=u.get("email", ""),
                    role=role,
                    salt=u["salt"],
                    password_hash=u["password_hash"],
                    client_id=u.get("client_id"),
                    cin=u.get("cin"),
                    created_at=u.get("created_at") or datetime.now(timezone.utc).isoformat(),
                )
            )
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def bootstrap_database() -> dict:
    """Create tables, seed business CSV, ensure default admin."""
    import os
    import uuid
    from datetime import datetime, timezone

    from src.auth.local_store import hash_password

    init_db()
    stats = seed_business_data(force=False)
    _migrate_json_auth()

    from src.db.repository import get_client_by_cin

    db = SessionLocal()
    try:
        if not db.scalar(select(func.count()).select_from(User)):
            admin_password = os.getenv("TALYS_ADMIN_PASSWORD", "admin123")
            salt, pwd_hash = hash_password(admin_password)
            db.add(
                User(
                    id=str(uuid.uuid4()),
                    username="admin",
                    email="admin@talys.local",
                    role="admin",
                    salt=salt,
                    password_hash=pwd_hash,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            db.commit()
            stats["admin_created"] = True

        demo = db.scalar(select(User).where(User.username == "client_demo"))
        if not demo:
            row = get_client_by_cin("88710263")
            if row:
                salt, pwd_hash = hash_password("client123")
                db.add(
                    User(
                        id=str(uuid.uuid4()),
                        username="client_demo",
                        email="client_demo@talys.local",
                        role="client",
                        salt=salt,
                        password_hash=pwd_hash,
                        cin=str(row["cin"]),
                        client_id=int(row["client_id"]),
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                )
                db.commit()
                stats["client_demo_created"] = True
    finally:
        db.close()

    return stats
