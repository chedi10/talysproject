from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from src.rag.context import build_rag_query, format_rag_sources_for_prompt, format_rag_references_section
from src.reports.business import build_conclusion, build_keywords, business_recommendation, _worst_risk


Intent = Literal["classic_score", "sequential_score", "graph_score", "full_report", "compare_models", "institutional"]
ReportIntent = Literal["full_report", "compare_models"]


class AgentState(TypedDict, total=False):
    session_id: str
    user_message: str
    conversation_history: list[dict[str, str]]
    cin: str | None
    credit_id: int | None
    intent: Intent
    route: Literal["classic", "sequential", "graph", "full_report", "compare_models", "institutional"]
    classic_result: dict[str, Any] | None
    sequential_result: dict[str, Any] | None
    graph_result: dict[str, Any] | None
    systems_result: dict[str, Any] | None
    rag_sources: list[dict[str, Any]]
    report_markdown: str
    final_answer: str
    model_selected: Literal["classic", "sequential", "graph"] | None
    errors: list[str]


def _extract_cin(text: str) -> str | None:
    m = re.search(r"\b(\d{6,32})\b", text or "")
    return m.group(1) if m else None


def _extract_credit_id(text: str) -> int | None:
    m = re.search(r"(?:credit[_\s-]?id|id[_\s-]?credit)\s*[:=]?\s*(\d+)", text or "", flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _detect_intent(text: str, history: list[dict[str, str]] | None = None) -> Intent:
    t = (text or "").lower()
    if any(k in t for k in ["institution", "règles", "regles", "rules", "conformité", "conformite", "systèmes", "systemes"]):
        return "institutional"
    if any(k in t for k in ["alerte", "early warning", "watchlist", "surveillance"]):
        return "institutional"
    if any(k in t for k in ["recommandation institution", "décision comité", "decision comite"]):
        return "institutional"
    if any(k in t for k in ["compar", "compare", "comparer", "vs", "versus"]):
        return "compare_models"
    if any(k in t for k in ["rapport", "report", "global", "complet", "full"]):
        return "full_report"
    if any(k in t for k in ["graph", "graphe", "gnn", "gat", "graphsage", "reseau", "réseau"]):
        return "graph_score"
    if any(k in t for k in ["sequent", "séquent", "lstm", "gru", "transformer", "transaction", "remboursement"]):
        return "sequential_score"
    # Héritage intent depuis historique si message court ("idem", "pareil", "séquentiel")
    if history and len(t.split()) <= 4:
        for msg in reversed(history):
            if msg.get("role") != "user":
                continue
            prev = _detect_intent(msg.get("content", ""), None)
            if prev != "classic_score":
                return prev
            break
    return "classic_score"


def _is_report_intent(intent: str | None) -> bool:
    return intent in ("full_report", "compare_models")


def _safe_dump(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return dict(obj)


def _primary_result(state: AgentState) -> dict[str, Any]:
    intent = state.get("intent", "classic_score")
    if intent == "sequential_score":
        return state.get("sequential_result") or {}
    if intent == "graph_score":
        return state.get("graph_result") or {}
    return state.get("classic_result") or {}


def _format_conversational_answer(state: AgentState) -> str:
    intent = state.get("intent", "classic_score")
    cin = state.get("cin")
    primary = _primary_result(state)
    classic = state.get("classic_result") or {}
    seq = state.get("sequential_result") or {}
    graph = state.get("graph_result") or {}

    if not primary and not cin:
        return "Je n'ai pas trouvé de CIN. Exemple : « Analyse le CIN 88710263 en séquentiel »."

    kyc = primary.get("kyc_score") or classic.get("kyc_score")
    proba = primary.get("default_proba")
    risk = primary.get("risk_level")
    model = primary.get("model_used") or state.get("model_selected")

    rec = business_recommendation(
        worst_risk=str(risk or "MODERE"),
        kyc_score=float(kyc) if kyc is not None else None,
        default_proba=float(proba) if proba is not None else None,
    )

    keywords = build_keywords(
        cin=str(cin or "?"),
        classic=classic or primary,
        sequential=seq or primary,
        graph=graph or primary,
    )

    lines = [
        f"## Analyse — CIN `{cin}`",
        "",
        f"**Intent** : `{intent}` | **Modèle** : `{model}`",
        "",
        "### Indicateurs clés",
        f"| KPI | Valeur |",
        f"|-----|--------|",
        f"| Score KYC | {kyc} |",
        f"| Probabilité de défaut | {proba} |",
        f"| Niveau de risque | **{risk}** |",
        "",
        "### Mots-clés métier",
        ", ".join(f"`{k}`" for k in keywords[:8]),
        "",
        "### Recommandation",
        f"- **Décision** : {rec['decision']}",
        f"- **Actions** : {rec['actions']}",
        f"- **Surveillance** : {rec['surveillance']}",
        "",
        "### Conclusion",
        build_conclusion(
            cin=str(cin),
            worst_risk=str(risk or "MODERE"),
            kyc_score=float(kyc) if kyc is not None else None,
            default_proba=float(proba) if proba is not None else None,
            models_agree=False,
        ),
    ]

    if intent == "compare_models" or (seq and graph and classic):
        lines.insert(
            8,
            "\n".join(
                [
                    "### Comparaison rapide",
                    f"- Classique : {classic.get('risk_level')} ({classic.get('default_proba')})",
                    f"- Séquentiel : {seq.get('risk_level')} ({seq.get('default_proba')})",
                    f"- Graphe : {graph.get('risk_level')} ({graph.get('default_proba')})",
                    "",
                ]
            ),
        )

    sources = state.get("rag_sources") or []
    if sources:
        lines.extend(["", "### Références", format_rag_references_section(sources)])

    return "\n".join(lines)


def _format_institutional_answer(state: AgentState) -> str:
    cin = state.get("cin")
    systems = state.get("systems_result") or {}
    if not systems:
        return f"Aucune analyse institutionnelle disponible pour CIN `{cin}`."

    rules = systems.get("rules") or {}
    ews = systems.get("early_warning") or {}
    reco = systems.get("recommendation") or {}
    profile = systems.get("client_profile") or {}

    lines = [
        f"## Analyse institutionnelle — CIN `{cin}`",
        "",
    ]
    if profile.get("nom"):
        lines.append(f"**Client** : {profile.get('prenom', '')} {profile.get('nom', '')} ({profile.get('ville', '')})")
        lines.append("")

    lines.extend([
        "### Score institutionnel",
        f"| Indicateur | Valeur |",
        f"|------------|--------|",
        f"| Score institutionnel | **{float(systems.get('institutional_score', 0)):.1%}** |",
        f"| Niveau de risque | **{systems.get('risk_level', '—')}** |",
        f"| Conformité | **{rules.get('compliance_score', '—')}/100** |",
        f"| Watchlist EWS | **{ews.get('watchlist_priority', 'NONE')}** |",
        "",
        "### Business Rules",
        rules.get("summary", "—"),
        "",
    ])

    triggered = rules.get("triggered") or []
    if triggered:
        lines.append("**Règles déclenchées** :")
        for r in triggered[:5]:
            lines.append(f"- [{r.get('severity')}] {r.get('name')} — {r.get('message')}")
        lines.append("")

    lines.extend([
        "### Early Warning",
        ews.get("summary", "—"),
        "",
        "### Recommandation IA",
        f"**{reco.get('decision_label', '—')}** (confiance {float(reco.get('confidence', 0)):.0%})",
        "",
        reco.get("justification", "")[:400],
        "",
    ])

    actions = reco.get("recommended_actions") or []
    if actions:
        lines.append("**Plan d'action** :")
        lines.extend(f"- {a}" for a in actions[:4])

    sources = state.get("rag_sources") or []
    if sources:
        lines.extend(["", "### Références RAG", format_rag_references_section(sources)])

    return "\n".join(lines)


def _llm_enhance_answer(base_answer: str, state: AgentState) -> str:
    """Enrichit la réponse conversationnelle via Ollama + RAG (best-effort)."""
    if _is_report_intent(state.get("intent")):
        return base_answer
    sources = state.get("rag_sources") or []
    if not sources:
        return base_answer
    rag_block = format_rag_sources_for_prompt(sources, max_chars=2000)
    prompt = f"""
Tu es l'assistant crédit Talys (microfinance). Reformule la réponse suivante en français professionnel,
en t'appuyant UNIQUEMENT sur les faits présents et les extraits RAG. Ne invente aucun chiffre.

Question utilisateur: {state.get('user_message', '')}

Réponse de base (faits):
{base_answer}

{rag_block}

Réponds en Markdown concis (sections ### si utile). Garde tous les chiffres identiques.
"""
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            temperature=0.15,
            sync_client_kwargs={"timeout": 30.0},
        )
        msg = llm.invoke([
            SystemMessage(content="Assistant crédit Talys, ton professionnel, pas d'invention."),
            HumanMessage(content=prompt),
        ])
        enhanced = str(getattr(msg, "content", msg)).strip()
        return enhanced if len(enhanced) > 80 else base_answer
    except Exception:
        return base_answer


class CreditAgentOrchestrator:
    """LangGraph orchestration — routing métier + conversations + rapports complets."""

    def __init__(
        self,
        *,
        run_classic: Callable[[str, int | None], Any],
        run_sequential: Callable[[str, int | None], Any],
        run_graph: Callable[[str, int | None], Any],
        rag_retrieve: Callable[[list[str], int], list[dict[str, Any]]],
        run_systems: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self._run_classic = run_classic
        self._run_sequential = run_sequential
        self._run_graph = run_graph
        self._rag_retrieve = rag_retrieve
        self._run_systems = run_systems
        self._graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("ExtractCIN", self._node_extract_cin)
        g.add_node("DetectIntent", self._node_detect_intent)
        g.add_node("ChoosePath", self._node_choose_path)
        g.add_node("RunClassic", self._node_run_classic)
        g.add_node("RunSequential", self._node_run_sequential)
        g.add_node("RunGraph", self._node_run_graph)
        g.add_node("RunFullReport", self._node_run_full_report)
        g.add_node("RunCompareModels", self._node_run_compare_models)
        g.add_node("RunInstitutional", self._node_run_institutional)
        g.add_node("RetrieveDocs", self._node_retrieve_docs)
        g.add_node("WriteReport", self._node_write_report)
        g.add_node("Respond", self._node_respond)

        g.add_edge(START, "ExtractCIN")
        g.add_edge("ExtractCIN", "DetectIntent")
        g.add_edge("DetectIntent", "ChoosePath")

        g.add_conditional_edges(
            "ChoosePath",
            self._route_from_state,
            {
                "classic": "RunClassic",
                "sequential": "RunSequential",
                "graph": "RunGraph",
                "full_report": "RunFullReport",
                "compare_models": "RunCompareModels",
                "institutional": "RunInstitutional",
            },
        )

        for run_node in ("RunClassic", "RunSequential", "RunGraph", "RunFullReport", "RunCompareModels", "RunInstitutional"):
            g.add_edge(run_node, "RetrieveDocs")

        g.add_conditional_edges(
            "RetrieveDocs",
            self._route_after_rag,
            {"write_report": "WriteReport", "respond": "Respond"},
        )
        g.add_edge("WriteReport", "Respond")
        g.add_edge("Respond", END)
        return g.compile()

    def invoke(
        self,
        *,
        session_id: str,
        message: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        init: AgentState = {
            "session_id": session_id.strip(),
            "user_message": message.strip(),
            "conversation_history": conversation_history or [],
            "errors": [],
            "rag_sources": [],
            "classic_result": None,
            "sequential_result": None,
            "graph_result": None,
            "systems_result": None,
            "model_selected": None,
        }
        out = self._graph.invoke(init)
        return dict(out)

    def _route_after_rag(self, state: AgentState) -> str:
        return "write_report" if _is_report_intent(state.get("intent")) else "respond"

    def _node_extract_cin(self, state: AgentState) -> AgentState:
        user_msg = state.get("user_message", "")
        cin = _extract_cin(user_msg)
        credit_id = _extract_credit_id(user_msg)
        if not cin and state.get("conversation_history"):
            for msg in reversed(state["conversation_history"]):
                if msg.get("role") != "user":
                    continue
                cin = _extract_cin(msg.get("content", ""))
                if cin:
                    break
        errors = list(state.get("errors", []))
        if not cin:
            errors.append("missing_cin")
        return {"cin": cin, "credit_id": credit_id, "errors": errors}

    def _node_detect_intent(self, state: AgentState) -> AgentState:
        return {"intent": _detect_intent(state.get("user_message", ""), state.get("conversation_history"))}

    def _node_choose_path(self, state: AgentState) -> AgentState:
        intent = state.get("intent", "classic_score")
        route_map = {
            "classic_score": "classic",
            "sequential_score": "sequential",
            "graph_score": "graph",
            "full_report": "full_report",
            "compare_models": "compare_models",
            "institutional": "institutional",
        }
        return {"route": route_map[intent]}

    def _route_from_state(self, state: AgentState) -> str:
        return state.get("route", "classic")

    def _node_run_classic(self, state: AgentState) -> AgentState:
        errors = list(state.get("errors", []))
        cin = state.get("cin")
        if not cin:
            return {"errors": errors}
        try:
            res = _safe_dump(self._run_classic(cin, state.get("credit_id")))
            return {"classic_result": res, "model_selected": "classic"}
        except Exception as exc:
            errors.append(f"classic_failed:{exc}")
            return {"errors": errors}

    def _node_run_sequential(self, state: AgentState) -> AgentState:
        errors = list(state.get("errors", []))
        cin = state.get("cin")
        if not cin:
            return {"errors": errors}
        try:
            res = _safe_dump(self._run_sequential(cin, state.get("credit_id")))
            return {"sequential_result": res, "model_selected": "sequential"}
        except Exception as exc:
            errors.append(f"sequential_failed:{exc}")
            return {"errors": errors}

    def _node_run_graph(self, state: AgentState) -> AgentState:
        errors = list(state.get("errors", []))
        cin = state.get("cin")
        if not cin:
            return {"errors": errors}
        try:
            res = _safe_dump(self._run_graph(cin, state.get("credit_id")))
            return {"graph_result": res, "model_selected": "graph"}
        except Exception as exc:
            errors.append(f"graph_failed:{exc}")
            return {"errors": errors}

    def _node_run_full_report(self, state: AgentState) -> AgentState:
        updates: AgentState = {}
        updates.update(self._node_run_classic(state))
        state_merged = {**state, **updates}
        updates.update(self._node_run_sequential(state_merged))
        state_merged = {**state, **updates}
        updates.update(self._node_run_graph(state_merged))
        updates["model_selected"] = "classic"
        if self._run_systems and state.get("cin"):
            try:
                updates["systems_result"] = self._run_systems(state["cin"])
            except Exception:
                pass
        return updates

    def _node_run_compare_models(self, state: AgentState) -> AgentState:
        return self._node_run_full_report(state)

    def _node_run_institutional(self, state: AgentState) -> AgentState:
        errors = list(state.get("errors", []))
        cin = state.get("cin")
        if not cin or not self._run_systems:
            errors.append("systems_unavailable")
            return {"errors": errors}
        try:
            systems = self._run_systems(cin)
            return {"systems_result": systems, "model_selected": None}
        except Exception as exc:
            errors.append(f"systems_failed:{exc}")
            return {"errors": errors}

    def _node_retrieve_docs(self, state: AgentState) -> AgentState:
        errors = list(state.get("errors", []))
        cin = state.get("cin")
        intent = state.get("intent", "classic_score")
        classic = state.get("classic_result") or {}
        seq = state.get("sequential_result") or {}
        graph = state.get("graph_result") or {}

        risk = _worst_risk(
            str(classic.get("risk_level") or ""),
            str(seq.get("risk_level") or ""),
            str(graph.get("risk_level") or ""),
        )
        kyc = classic.get("kyc_score") or seq.get("kyc_score") or graph.get("kyc_score")
        primary = _primary_result(state)

        systems = state.get("systems_result") or {}
        if intent == "institutional" and systems:
            risk = str(systems.get("risk_level") or "")
            kyc = (systems.get("client_profile") or {}).get("kyc_score") or kyc

        queries = build_rag_query(
            cin=cin,
            user_message=state.get("user_message"),
            intent=intent,
            risk_level=str(risk) if risk else None,
            model_used=str(primary.get("model_used") or state.get("model_selected") or ""),
            kyc_score=float(kyc) if kyc is not None else None,
            extra_topics=[f"recommandation risque {risk}"],
        )
        try:
            k = 6 if _is_report_intent(intent) else 4 if intent == "institutional" else 3
            sources = self._rag_retrieve(queries, k)
            return {"rag_sources": sources}
        except Exception as exc:
            errors.append(f"rag_failed:{exc}")
            return {"errors": errors, "rag_sources": []}

    def _node_write_report(self, state: AgentState) -> AgentState:
        errors = list(state.get("errors", []))
        classic = state.get("classic_result") or {}
        seq = state.get("sequential_result") or {}
        graph = state.get("graph_result") or {}
        sources = state.get("rag_sources", [])
        cin = state.get("cin", "?")

        rag_block = format_rag_sources_for_prompt(sources, max_chars=3500)
        refs_block = format_rag_references_section(sources)
        keywords = build_keywords(cin=str(cin), classic=classic, sequential=seq, graph=graph)
        worst = _worst_risk(
            str(classic.get("risk_level") or ""),
            str(seq.get("risk_level") or ""),
            str(graph.get("risk_level") or ""),
        )
        proba = max(float(classic.get("default_proba") or 0), float(seq.get("default_proba") or 0), float(graph.get("default_proba") or 0))
        kyc = classic.get("kyc_score") or seq.get("kyc_score")
        conclusion = build_conclusion(
            cin=str(cin),
            worst_risk=worst,
            kyc_score=float(kyc) if kyc is not None else None,
            default_proba=proba,
            models_agree=len({classic.get("risk_level"), seq.get("risk_level"), graph.get("risk_level")} - {None}) <= 1,
        )

        systems = state.get("systems_result") or {}
        systems_block = json.dumps(systems, ensure_ascii=False) if systems else "Non disponible"

        prompt = f"""
Tu es analyste risque crédit microfinance (Talys).
Rédige un rapport Markdown PROFESSIONNEL en français avec EXACTEMENT ces sections (titres ##):
1. Mots-clés
2. Résumé exécutif
3. Systèmes institutionnels (règles, alertes, recommandation)
4. Résultat classique
5. Résultat séquentiel
6. Résultat graphe
7. Analyse métier
8. Recommandation
9. Conclusion
10. Références

Mots-clés suggérés: {", ".join(keywords)}
Conclusion suggérée (à reformuler légèrement): {conclusion}

Systèmes institutionnels: {systems_block}

Classique: {json.dumps(classic, ensure_ascii=False)}
Séquentiel: {json.dumps(seq, ensure_ascii=False)}
Graphe: {json.dumps(graph, ensure_ascii=False)}

{rag_block}

Références à citer: {refs_block}
"""

        markdown: str | None = None
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3.2"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                temperature=0.2,
                sync_client_kwargs={"timeout": 45.0},
                async_client_kwargs={"timeout": 45.0},
            )
            msg = llm.invoke(
                [
                    SystemMessage(content="Rapports auditables, ton comité crédit, sections obligatoires."),
                    HumanMessage(content=prompt),
                ]
            )
            markdown = str(getattr(msg, "content", msg)).strip()
        except Exception as exc:
            errors.append(f"write_report_llm_failed:{exc}")
            from src.reports.report_builder import build_structured_report, structured_to_markdown

            structured = build_structured_report(
                cin=str(cin),
                classic=classic,
                sequential=seq,
                graph=graph,
                sources=sources,
                systems=state.get("systems_result"),
            )
            markdown = structured_to_markdown(structured)

        return {"report_markdown": markdown, "errors": errors}

    def _node_respond(self, state: AgentState) -> AgentState:
        cin = state.get("cin")
        intent = state.get("intent")
        errors = state.get("errors", [])

        if not cin:
            return {
                "final_answer": (
                    "Je n'ai pas identifié de CIN. "
                    "Indiquez-le dans votre message (ex. « Analyse le CIN 88710263 »)."
                ),
                "model_selected": None,
            }

        if _is_report_intent(intent):
            body = state.get("report_markdown") or "Rapport indisponible."
        elif intent == "institutional":
            body = _format_institutional_answer(state)
        else:
            body = _format_conversational_answer(state)
            body = _llm_enhance_answer(body, state)

        header = f"**Session** `{state.get('session_id')}` | **Intent** `{intent}` | **CIN** `{cin}`"
        if errors:
            header += f"\n_Avertissements: {' ; '.join(errors[:2])}_"

        return {"final_answer": header + "\n\n" + body}
