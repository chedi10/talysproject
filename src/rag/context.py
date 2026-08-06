from __future__ import annotations

"""
Build RAG queries and format retrieved chunks for LLM prompts.
"""

from typing import Any


def build_rag_query(
    *,
    cin: str | None = None,
    user_message: str | None = None,
    intent: str | None = None,
    risk_level: str | None = None,
    model_used: str | None = None,
    kyc_score: float | None = None,
    extra_topics: list[str] | None = None,
) -> list[str]:
    """
    Return focused sub-queries (multi-query retrieval works better than one long pipe string).
    """
    queries: list[str] = []

    base = [
        "REF-08 score KYC probabilité défaut risque crédit",
        "explication scoring modèle Deep Tabular Transformer GAT",
        "microfinance recommandation niveau risque FAIBLE MODERE ELEVE",
        "systèmes décisionnels SHAP business rules early warning",
        "conformité politique crédit DTI KYC retards transactions",
        "rapport comité crédit mots-clés conclusion recommandation",
    ]
    queries.extend(base)

    if risk_level:
        queries.append(f"niveau risque {risk_level} règles métier recommandation")
    if model_used:
        queries.append(f"modèle {model_used} explication comportement crédit")
    if kyc_score is not None:
        queries.append(f"score KYC {kyc_score:.0f} conformité client")
    if intent:
        queries.append(f"intent {intent} analyse crédit")
        if intent == "institutional":
            queries.extend([
                "business rules engine conformité politique crédit",
                "early warning watchlist dégradation comportementale",
                "recommandation IA décision comité crédit",
            ])
        if intent in ("full_report", "compare_models"):
            queries.append("rapport complet scoring ML systèmes institutionnels")
    if cin:
        queries.append(f"analyse client CIN {cin}")
    if user_message:
        # Keep only meaningful tokens from user message (avoid polluting with digits-only CIN).
        tokens = [
            t
            for t in user_message.lower().split()
            if len(t) > 3 and not t.isdigit()
        ]
        if tokens:
            queries.append(" ".join(tokens[:12]))

    if extra_topics:
        for t in extra_topics:
            if t and t.strip():
                queries.append(t.strip())

    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        qn = " ".join(q.split())
        if qn not in seen:
            seen.add(qn)
            out.append(qn)
    return out


def format_rag_sources_for_prompt(sources: list[dict[str, Any]], *, max_chars: int = 4000) -> str:
    """Format retrieved chunks with citations for LLM consumption."""
    if not sources:
        return "Aucun extrait documentaire pertinent (base de connaissances vide ou scores trop faibles)."

    lines: list[str] = ["## Extraits documentaires (RAG — ne pas inventer au-delà de ces textes)"]
    used = len(lines[0])
    for i, s in enumerate(sources, start=1):
        src = s.get("source", "?")
        cid = s.get("chunk_id", 0)
        score = float(s.get("score", 0.0))
        text = str(s.get("text", "")).strip()
        block = (
            f"\n### [{i}] {src}#{cid} (pertinence={score:.3f})\n{text}\n"
        )
        if used + len(block) > max_chars:
            lines.append("\n_(extraits tronqués pour limite de contexte)_")
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines)


def format_rag_references_section(sources: list[dict[str, Any]]) -> str:
    """Short bibliography for Markdown reports."""
    if not sources:
        return "- Aucune référence documentaire."
    return "\n".join(
        [
            f"- `{s.get('source')}#{s.get('chunk_id')}` (pertinence={float(s.get('score', 0)):.3f})"
            for s in sources
        ]
    )
