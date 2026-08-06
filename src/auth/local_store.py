from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from sqlalchemy import desc, func, select

from src.db.engine import SessionLocal
from src.db.models import ActivityLog, ChatMessage, ChatSession, Session, User

Role = Literal["client", "agent", "admin"]

_lock = threading.Lock()
SESSION_DAYS = int(os.getenv("AUTH_SESSION_DAYS", "14"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return salt, digest.hex()


def verify_password(password: str, salt: str, password_hash: str) -> bool:
    _, candidate = hash_password(password, salt)
    return secrets.compare_digest(candidate, password_hash)


def _user_to_public(u: User) -> dict[str, Any]:
    return {
        "id": u.id,
        "username": u.username,
        "email": u.email,
        "role": u.role,
        "cin": u.cin,
        "client_id": u.client_id,
        "created_at": u.created_at,
    }


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user.get("email", ""),
        "role": user.get("role", "agent"),
        "cin": user.get("cin"),
        "client_id": user.get("client_id"),
        "created_at": user.get("created_at"),
    }


def ensure_default_admin() -> None:
    """Handled by bootstrap_database(); kept for API compatibility."""
    pass


def list_users() -> list[dict[str, Any]]:
    db = SessionLocal()
    try:
        users = db.scalars(select(User).order_by(User.created_at)).all()
        return [_user_to_public(u) for u in users]
    finally:
        db.close()


def get_user_by_username(username: str) -> dict[str, Any] | None:
    uname = username.strip().lower()
    db = SessionLocal()
    try:
        for u in db.scalars(select(User)).all():
            if u.username.lower() == uname:
                return {
                    "id": u.id,
                    "username": u.username,
                    "email": u.email,
                    "role": u.role,
                    "salt": u.salt,
                    "password_hash": u.password_hash,
                    "cin": u.cin,
                    "client_id": u.client_id,
                    "created_at": u.created_at,
                }
    finally:
        db.close()
    return None


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    db = SessionLocal()
    try:
        u = db.get(User, user_id)
        if not u:
            return None
        return {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "role": u.role,
            "salt": u.salt,
            "password_hash": u.password_hash,
            "cin": u.cin,
            "client_id": u.client_id,
            "created_at": u.created_at,
        }
    finally:
        db.close()


def create_user(
    *,
    username: str,
    email: str,
    password: str,
    role: Role = "agent",
    cin: str | None = None,
    client_id: int | None = None,
) -> dict[str, Any]:
    uname = username.strip()
    if len(uname) < 3:
        raise ValueError("username_too_short")
    if len(password) < 6:
        raise ValueError("password_too_short")
    if get_user_by_username(uname):
        raise ValueError("username_taken")

    salt, pwd_hash = hash_password(password)
    user_id = str(uuid.uuid4())
    created_at = _now_iso()
    user = User(
        id=user_id,
        username=uname,
        email=email.strip(),
        role=role,
        salt=salt,
        password_hash=pwd_hash,
        cin=cin,
        client_id=client_id,
        created_at=created_at,
    )
    with _lock:
        db = SessionLocal()
        try:
            db.add(user)
            db.commit()
        finally:
            db.close()
    return public_user(
        {
            "id": user_id,
            "username": uname,
            "email": email.strip(),
            "role": role,
            "cin": cin,
            "client_id": client_id,
            "created_at": created_at,
        }
    )


def authenticate(username: str, password: str) -> dict[str, Any] | None:
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["salt"], user["password_hash"]):
        return None
    return user


def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)).isoformat()
    with _lock:
        db = SessionLocal()
        try:
            db.add(Session(token=token, user_id=user_id, expires_at=expires, created_at=_now_iso()))
            db.commit()
        finally:
            db.close()
    return token


def delete_session(token: str) -> None:
    with _lock:
        db = SessionLocal()
        try:
            row = db.get(Session, token)
            if row:
                db.delete(row)
                db.commit()
        finally:
            db.close()


def resolve_session(token: str) -> dict[str, Any] | None:
    if not token:
        return None
    db = SessionLocal()
    try:
        entry = db.get(Session, token)
        if not entry:
            return None
        try:
            expires = datetime.fromisoformat(entry.expires_at)
        except Exception:
            db.delete(entry)
            db.commit()
            return None
        if expires < datetime.now(timezone.utc):
            db.delete(entry)
            db.commit()
            return None
        user = db.get(User, entry.user_id)
        if not user:
            return None
        return public_user(_user_to_public(user))
    finally:
        db.close()


def log_activity(
    *,
    user: dict[str, Any],
    action: str,
    cin: str | None = None,
    model: str | None = None,
    intent: str | None = None,
    message: str | None = None,
    session_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = ActivityLog(
        id=str(uuid.uuid4()),
        user_id=user["id"],
        username=user["username"],
        role=user.get("role", "agent"),
        action=action,
        cin=cin,
        model=model,
        intent=intent,
        message_preview=(message or "")[:240] if message else None,
        session_id=session_id,
        extra_json=json.dumps(extra or {}, ensure_ascii=False),
        created_at=_now_iso(),
    )
    with _lock:
        db = SessionLocal()
        try:
            db.add(record)
            count = db.scalar(select(func.count()).select_from(ActivityLog)) or 0
            if count > 5000:
                oldest = db.scalars(
                    select(ActivityLog).order_by(ActivityLog.created_at).limit(count - 5000)
                ).all()
                for row in oldest:
                    db.delete(row)
            db.commit()
        finally:
            db.close()
    return {
        "id": record.id,
        "user_id": record.user_id,
        "username": record.username,
        "role": record.role,
        "action": record.action,
        "cin": record.cin,
        "model": record.model,
        "intent": record.intent,
        "message_preview": record.message_preview,
        "session_id": record.session_id,
        "created_at": record.created_at,
    }


def list_activity(*, user: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
    db = SessionLocal()
    try:
        q = select(ActivityLog).order_by(desc(ActivityLog.created_at))
        if user.get("role") != "admin":
            q = q.where(ActivityLog.user_id == user["id"])
        rows = db.scalars(q.limit(max(1, min(limit, 500)))).all()
        return [
            {
                "id": r.id,
                "user_id": r.user_id,
                "username": r.username,
                "role": r.role,
                "action": r.action,
                "cin": r.cin,
                "model": r.model,
                "intent": r.intent,
                "message_preview": r.message_preview,
                "session_id": r.session_id,
                "created_at": r.created_at,
            }
            for r in rows
        ]
    finally:
        db.close()


def _get_or_create_chat_session(db, user_id: str, session_id: str) -> ChatSession:
    cs = db.scalar(
        select(ChatSession).where(ChatSession.user_id == user_id, ChatSession.session_id == session_id)
    )
    if cs:
        return cs
    cs = ChatSession(
        user_id=user_id,
        session_id=session_id,
        created_at=_now_iso(),
        updated_at=_now_iso(),
    )
    db.add(cs)
    db.flush()
    return cs


def append_chat_messages(
    *,
    user_id: str,
    session_id: str,
    messages: list[dict[str, str]],
    cin: str | None = None,
    intent: str | None = None,
    title: str | None = None,
) -> None:
    if not messages:
        return
    with _lock:
        db = SessionLocal()
        try:
            cs = _get_or_create_chat_session(db, user_id, session_id.strip())
            for m in messages:
                db.add(
                    ChatMessage(
                        chat_session_id=cs.id,
                        role=m.get("role", "user"),
                        content=m.get("content", ""),
                        created_at=_now_iso(),
                    )
                )
            msg_count = db.scalar(
                select(func.count()).select_from(ChatMessage).where(ChatMessage.chat_session_id == cs.id)
            ) or 0
            if msg_count > 200:
                old = db.scalars(
                    select(ChatMessage)
                    .where(ChatMessage.chat_session_id == cs.id)
                    .order_by(ChatMessage.id)
                    .limit(msg_count - 200)
                ).all()
                for row in old:
                    db.delete(row)
            cs.updated_at = _now_iso()
            if cin:
                cs.last_cin = cin
            if intent:
                cs.last_intent = intent
            if title:
                cs.title = title
            elif not cs.title and messages:
                first_user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
                cs.title = (first_user[:60] + "…") if len(first_user) > 60 else (first_user or f"Session {session_id}")
            db.commit()
        finally:
            db.close()


def get_conversation_context(*, user_id: str, session_id: str, limit: int = 8) -> list[dict[str, str]]:
    db = SessionLocal()
    try:
        cs = db.scalar(
            select(ChatSession).where(ChatSession.user_id == user_id, ChatSession.session_id == session_id.strip())
        )
        if not cs:
            return []
        msgs = db.scalars(
            select(ChatMessage).where(ChatMessage.chat_session_id == cs.id).order_by(ChatMessage.id)
        ).all()
        return [{"role": m.role, "content": m.content} for m in msgs[-max(1, min(limit, 20)) :]]
    finally:
        db.close()


def get_chat_messages(*, user: dict[str, Any], session_id: str) -> list[dict[str, str]]:
    db = SessionLocal()
    try:
        cs = db.scalar(select(ChatSession).where(ChatSession.session_id == session_id.strip()))
        if not cs:
            return []
        if user.get("role") != "admin" and cs.user_id != user["id"]:
            return []
        msgs = db.scalars(
            select(ChatMessage).where(ChatMessage.chat_session_id == cs.id).order_by(ChatMessage.id)
        ).all()
        return [{"role": m.role, "content": m.content} for m in msgs]
    finally:
        db.close()


def list_chat_sessions(*, user: dict[str, Any], limit: int = 50) -> list[dict[str, Any]]:
    db = SessionLocal()
    try:
        q = select(ChatSession).order_by(desc(ChatSession.updated_at))
        if user.get("role") != "admin":
            q = q.where(ChatSession.user_id == user["id"])
        sessions = db.scalars(q.limit(max(1, min(limit, 200)))).all()
        users = {u.id: u for u in db.scalars(select(User)).all()}
        rows: list[dict[str, Any]] = []
        for cs in sessions:
            u = users.get(cs.user_id)
            msgs = db.scalars(
                select(ChatMessage).where(ChatMessage.chat_session_id == cs.id).order_by(desc(ChatMessage.id)).limit(1)
            ).all()
            last_preview = msgs[0].content[:120] if msgs else None
            msg_count = db.scalar(
                select(func.count()).select_from(ChatMessage).where(ChatMessage.chat_session_id == cs.id)
            ) or 0
            rows.append(
                {
                    "user_id": cs.user_id,
                    "username": u.username if u else "?",
                    "session_id": cs.session_id,
                    "message_count": msg_count,
                    "updated_at": cs.updated_at,
                    "created_at": cs.created_at,
                    "title": cs.title,
                    "last_cin": cs.last_cin,
                    "last_intent": cs.last_intent,
                    "last_preview": last_preview,
                }
            )
        return rows
    finally:
        db.close()
