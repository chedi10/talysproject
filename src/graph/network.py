from __future__ import annotations

"""Extraction du sous-graphe ego (client + voisins) pour visualisation."""

from typing import Any

import numpy as np
import pandas as pd

REL_COLORS = {
    "GARANT": "#F58220",
    "FAMILLE": "#8BC53F",
    "BUSINESS": "#1E4078",
    "CO_RETARD": "#EF4444",
    "CO_VILLE": "#94A3B8",
}


def build_ego_network(
    *,
    center_client_id: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    client_ids: np.ndarray,
    idx_to_client_id: dict[int, int],
    client_id_to_idx: dict[int, int],
    clients_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    node_probas: dict[int, float] | None = None,
    node_labels: dict[int, int] | None = None,
    max_neighbors: int = 14,
) -> dict[str, Any]:
    """
    Construit le réseau du client (ego-network) pour l'UI agent.

    Retourne nœuds, arêtes, stats et légende des types de relation.
    """
    center_idx = client_id_to_idx.get(int(center_client_id))
    if center_idx is None:
        return {"nodes": [], "edges": [], "stats": {}, "legend": _legend()}

    src, dst = edge_index[0], edge_index[1]
    clients = clients_df.drop_duplicates("client_id").set_index("client_id")

    # Arêtes uniques incidentes au centre (graphe non orienté logique)
    neighbor_map: dict[int, list[tuple[str, float, int]]] = {}
    for s, d, w in zip(src.tolist(), dst.tolist(), edge_weight.tolist()):
        if s == center_idx and d != center_idx:
            neighbor_map.setdefault(int(d), []).append(("out", float(w), int(d)))
        elif d == center_idx and s != center_idx:
            neighbor_map.setdefault(int(s), []).append(("in", float(w), int(s)))

    # Enrichir avec type_relation depuis relations.csv
    rel_lookup: dict[tuple[int, int], dict] = {}
    for r in relations_df.itertuples(index=False):
        a, b = int(r.source_client_id), int(r.target_client_id)
        info = {
            "type_relation": str(r.type_relation),
            "risk_relation": int(r.risk_relation),
        }
        rel_lookup[(a, b)] = info
        rel_lookup[(b, a)] = info

    scored_neighbors: list[tuple[int, float, str, int]] = []
    for n_idx, _entries in neighbor_map.items():
        cid = int(idx_to_client_id.get(n_idx, client_ids[n_idx]))
        rel_info = rel_lookup.get((center_client_id, cid)) or rel_lookup.get((cid, center_client_id)) or {}
        rtype = rel_info.get("type_relation", "CO_VILLE")
        risk = rel_info.get("risk_relation", 50)
        max_w = max(e[1] for e in _entries)
        scored_neighbors.append((n_idx, max_w, rtype, risk))

    scored_neighbors.sort(key=lambda x: (-x[3], -x[1]))
    selected = scored_neighbors[:max_neighbors]
    selected_idxs = {center_idx, *(n[0] for n in selected)}

    def _node_payload(idx: int, is_center: bool) -> dict[str, Any]:
        cid = int(idx_to_client_id.get(idx, client_ids[idx]))
        row = clients.loc[cid] if cid in clients.index else None
        proba = (node_probas or {}).get(cid)
        en_defaut = (node_labels or {}).get(cid)
        risk = _risk_from_proba(proba) if proba is not None else None
        if row is not None:
            name = f"{row.get('prenom', '')} {str(row.get('nom', ''))[:1]}.".strip()
            cin = str(row.get("cin", ""))
            ville = str(row.get("ville", ""))
        else:
            name, cin, ville = f"Client {cid}", "", ""
        return {
            "id": f"n{idx}",
            "client_id": cid,
            "cin": cin if is_center else _mask_cin(cin),
            "label": name or f"#{cid}",
            "ville": ville,
            "is_center": is_center,
            "en_defaut": bool(en_defaut) if en_defaut is not None else None,
            "default_proba": round(float(proba), 4) if proba is not None else None,
            "risk_level": risk,
        }

    idx_to_client = {i: int(client_ids[i]) for i in range(len(client_ids))}
    nodes = [_node_payload(center_idx, True)]
    nodes.extend(_node_payload(n[0], False) for n in selected)

    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[int, int]] = set()
    for n_idx, _w, rtype, risk in selected:
        key = (min(center_idx, n_idx), max(center_idx, n_idx))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        n_cid = idx_to_client.get(n_idx, int(client_ids[n_idx]))
        edges.append({
            "id": f"e{center_idx}-{n_idx}",
            "source": f"n{center_idx}",
            "target": f"n{n_idx}",
            "type_relation": rtype,
            "risk_relation": risk,
            "color": REL_COLORS.get(rtype, "#94A3B8"),
            "label": rtype.replace("_", " ").title(),
        })

    # Liens entre voisins (optionnel, max 8)
    intra = 0
    for i, (a_idx, _, _, _) in enumerate(selected):
        for b_idx, _, brtype, brisk in selected[i + 1 : i + 4]:
            if intra >= 8:
                break
            pair = rel_lookup.get((idx_to_client[a_idx], idx_to_client[b_idx]))
            if not pair:
                continue
            key = (min(a_idx, b_idx), max(a_idx, b_idx))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append({
                "id": f"e{a_idx}-{b_idx}",
                "source": f"n{a_idx}",
                "target": f"n{b_idx}",
                "type_relation": pair["type_relation"],
                "risk_relation": pair["risk_relation"],
                "color": REL_COLORS.get(pair["type_relation"], "#CBD5E1"),
                "label": pair["type_relation"],
                "intra": True,
            })
            intra += 1

    type_counts: dict[str, int] = {}
    for e in edges:
        if not e.get("intra"):
            type_counts[e["type_relation"]] = type_counts.get(e["type_relation"], 0) + 1

    stats = {
        "degree": len(neighbor_map),
        "displayed_neighbors": len(selected),
        "relations_by_type": type_counts,
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": stats,
        "legend": _legend(),
    }


def _legend() -> list[dict[str, str]]:
    return [{"type": k, "color": v, "label": k.replace("_", " ").title()} for k, v in REL_COLORS.items()]


def _mask_cin(cin: str) -> str:
    if len(cin) <= 4:
        return cin
    return f"{'*' * (len(cin) - 4)}{cin[-4:]}"


def _risk_from_proba(proba: float | None) -> str | None:
    if proba is None:
        return None
    if proba < 0.30:
        return "FAIBLE"
    if proba <= 0.60:
        return "MODERE"
    return "ELEVE"
