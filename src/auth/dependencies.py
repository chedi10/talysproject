from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException

from src.auth.local_store import public_user, resolve_session


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    token = _extract_bearer(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authentification requise.")
    user = resolve_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session invalide ou expirée.")
    return user


def require_admin(user: Annotated[dict[str, Any], Depends(get_current_user)]) -> dict[str, Any]:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Accès réservé à l'administrateur.")
    return user


def require_staff(user: Annotated[dict[str, Any], Depends(get_current_user)]) -> dict[str, Any]:
    """Agent ou admin — accès scoring / chat."""
    if user.get("role") not in ("agent", "admin"):
        raise HTTPException(status_code=403, detail="Accès réservé aux agents et administrateurs.")
    return user
