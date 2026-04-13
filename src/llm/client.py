from __future__ import annotations

"""
Lightweight wrapper around a LOCAL Ollama server (no external API).

Configuration via environment variables:
    - OLLAMA_BASE_URL (default: http://127.0.0.1:11434)
    - OLLAMA_MODEL    (default: llama3.1)
"""

import os
from typing import Dict, Any
import httpx


def build_risk_prompt(
    risk_level: str,
    default_proba: float,
    features: Dict[str, Any],
) -> str:
    """
    Build a French prompt explaining the credit risk profile.

    The LLM does NOT decide the risk itself; it only explains the
    result produced by the structured ML model.
    """
    lines = [
        "Tu es un analyste crédit dans une institution de microfinance.",
        "On te donne le résultat d'un modèle de scoring de défaut, ainsi que quelques caractéristiques du client.",
        "Tu dois produire une explication courte et claire en français pour un conseiller humain.",
        "",
        f"Niveau de risque calculé par le modèle: {risk_level}",
        f"Probabilité de défaut estimée: {default_proba:.3f}",
        "",
        "Caractéristiques principales du crédit / client :",
    ]
    for k, v in features.items():
        lines.append(f"- {k}: {v}")

    lines += [
        "",
        "Tâche :",
        "- Indique si le profil est risqué ou non (en t'appuyant sur le niveau de risque fourni).",
        "- Explique en 3 à 6 phrases maximum les raisons probables du niveau de risque,",
        "  en te basant uniquement sur les informations fournies (ne pas inventer d'autres données).",
        "- Utilise un ton professionnel, sans jargon statistique compliqué.",
    ]

    return "\n".join(lines)


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def _ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.2")


def generate_risk_explanation(
    risk_level: str,
    default_proba: float,
    features: Dict[str, Any],
) -> str:
    """
    Call the local Ollama server to generate a natural-language explanation.

    Requires Ollama running locally, e.g.:
        ollama serve
        ollama pull llama3.1
    """
    prompt = build_risk_prompt(risk_level, default_proba, features)

    payload = {
        "model": _ollama_model(),
        "messages": [
            {"role": "system", "content": "Tu es un expert en risque de crédit."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.4,
        },
    }

    # Ollama uses /api/chat on newer versions; some installations expose /api/generate instead.
    url = f"{_ollama_base_url()}/api/chat"
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, json=payload)
            if r.status_code == 404:
                # Fallback to /api/generate
                gen_url = f"{_ollama_base_url()}/api/generate"
                gen_payload = {
                    "model": _ollama_model(),
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.4},
                }
                r = client.post(gen_url, json=gen_payload)
            r.raise_for_status()
            data = r.json()
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. "
            "Start it with `ollama serve` (or ensure it's running)."
        ) from e

    # Ollama chat returns: {"message": {"role": "...", "content": "..."}, ...}
    # Ollama generate returns: {"response": "...", ...}
    msg = (data.get("message") or {}).get("content") or data.get("response")
    if not msg:
        raise RuntimeError(f"Unexpected Ollama response format: {data}")
    return str(msg).strip()

