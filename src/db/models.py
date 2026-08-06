"""SQLAlchemy models — business + auth tables."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ─── Business domain ─────────────────────────────────────────────────────────


class Client(Base):
    __tablename__ = "clients"

    client_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cin: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    nom: Mapped[str] = mapped_column(String(120))
    prenom: Mapped[str] = mapped_column(String(120))
    age: Mapped[int] = mapped_column(Integer)
    ville: Mapped[str] = mapped_column(String(64))
    profession: Mapped[str] = mapped_column(String(64))
    revenu_mensuel: Mapped[float] = mapped_column(Float)
    date_creation: Mapped[str] = mapped_column(String(32))
    statut_kyc: Mapped[str] = mapped_column(String(32))

    credits: Mapped[list["Credit"]] = relationship(back_populates="client")


class Credit(Base):
    __tablename__ = "credits"

    credit_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.client_id"), index=True)
    cycle: Mapped[str] = mapped_column(String(32))
    objet: Mapped[str] = mapped_column(String(64))
    montant: Mapped[float] = mapped_column(Float)
    duree_mois: Mapped[int] = mapped_column(Integer)
    dti: Mapped[float] = mapped_column(Float)
    date_debut: Mapped[str] = mapped_column(String(32))
    en_defaut: Mapped[int] = mapped_column(Integer, default=0)

    client: Mapped["Client"] = relationship(back_populates="credits")


class Remboursement(Base):
    __tablename__ = "remboursements"

    remb_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    credit_id: Mapped[int] = mapped_column(Integer, index=True)
    client_id: Mapped[int] = mapped_column(Integer, index=True)
    mois: Mapped[int] = mapped_column(Integer)
    montant_du: Mapped[float] = mapped_column(Float)
    date_echeance: Mapped[str] = mapped_column(String(32))
    date_paiement: Mapped[str] = mapped_column(String(32))
    retard_jours: Mapped[int] = mapped_column(Integer)
    statut: Mapped[str] = mapped_column(String(32))


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[int] = mapped_column(Integer, index=True)
    type: Mapped[str] = mapped_column(String(32))
    montant: Mapped[float] = mapped_column(Float)
    date: Mapped[str] = mapped_column(String(32))
    suspect: Mapped[int] = mapped_column(Integer, default=0)


class Relation(Base):
    __tablename__ = "relations"

    relation_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_client_id: Mapped[int] = mapped_column(Integer, index=True)
    target_client_id: Mapped[int] = mapped_column(Integer, index=True)
    type_relation: Mapped[str] = mapped_column(String(32))
    risk_relation: Mapped[int] = mapped_column(Integer)


# ─── Auth & audit ────────────────────────────────────────────────────────────


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(16), index=True)  # client | agent | admin
    salt: Mapped[str] = mapped_column(String(64))
    password_hash: Mapped[str] = mapped_column(String(128))
    client_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("clients.client_id"), nullable=True)
    cin: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    created_at: Mapped[str] = mapped_column(String(32))


class Session(Base):
    __tablename__ = "sessions"

    token: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    expires_at: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[str] = mapped_column(String(32))


class ActivityLog(Base):
    __tablename__ = "activity_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    username: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(16))
    action: Mapped[str] = mapped_column(String(64))
    cin: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message_preview: Mapped[str | None] = mapped_column(String(240), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    extra_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(32), index=True)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (UniqueConstraint("user_id", "session_id", name="uq_user_session"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    session_id: Mapped[str] = mapped_column(String(128))
    title: Mapped[str | None] = mapped_column(String(120), nullable=True)
    last_cin: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[str] = mapped_column(String(32))
    updated_at: Mapped[str] = mapped_column(String(32))

    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(32))

    session: Mapped["ChatSession"] = relationship(back_populates="messages")
