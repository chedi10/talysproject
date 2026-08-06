"""
Reconstruit relations.csv (graphe métier) et met à jour SQLite.

Usage:
    python -m src.graph.enrich_relations
    python -m src.graph.enrich_relations --retrain
"""

from __future__ import annotations

import argparse

from src.db.engine import SessionLocal, init_db
from src.db.models import Relation
from src.graph.builder import _load_tables, build_structural_relations, save_relations_csv


def reseed_relations(df) -> int:
    init_db()
    db = SessionLocal()
    try:
        db.query(Relation).delete()
        db.commit()
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
                for r in df.itertuples(index=False)
            ],
        )
        db.commit()
        return len(df)
    finally:
        db.close()


def main(retrain: bool = False) -> None:
    clients, credits, remb, _, _ = _load_tables()
    print("Construction du graphe relationnel enrichi...")
    df = build_structural_relations(clients, credits, remb)
    path = save_relations_csv(df)
    print(f"  -> {path} ({len(df)} arêtes)")
    print("  Types:", df["type_relation"].value_counts().to_dict())

    n = reseed_relations(df)
    print(f"  -> SQLite relations: {n} lignes mises à jour")

    if retrain:
        print("Réentraînement GAT...")
        from src.models.graph.train import train_gat

        train_gat(epochs=80, hidden_dim=96, n_heads=4, dropout=0.25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrichir la base graphique Talys")
    parser.add_argument("--retrain", action="store_true", help="Réentraîner le GAT après enrichissement")
    args = parser.parse_args()
    main(retrain=args.retrain)
