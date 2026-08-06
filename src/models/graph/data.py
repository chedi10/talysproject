from __future__ import annotations

"""
Build a graph dataset for GAT (node classification).

Delegates to src.graph.builder for enriched node features and edge weights.
"""

from src.graph.builder import GraphDataset, build_client_graph

__all__ = ["GraphDataset", "build_graph_dataset"]


def build_graph_dataset(*, rebuild_relations: bool = False) -> GraphDataset:
    return build_client_graph(rebuild_relations=rebuild_relations)
