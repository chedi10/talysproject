"""
Train all deep learning models (tabular, transformer, GAT).

Usage:
    python -m src.features.engineering   # if features.parquet missing
    python -m src.models.deep_train
"""

from __future__ import annotations


def main():
    print("=" * 60)
    print("1/3 — Deep Tabular Network (MLP + Embeddings)")
    print("=" * 60)
    from src.models.deep_tabular.train import train_deep_tabular
    train_deep_tabular(epochs=40)

    print("\n" + "=" * 60)
    print("2/3 — Temporal Transformer (sequences)")
    print("=" * 60)
    from src.models.sequential.train import train_sequential_baselines
    train_sequential_baselines(seq_len=30, epochs=8)

    print("\n" + "=" * 60)
    print("3/3 — GAT Graph Attention Network")
    print("=" * 60)
    from src.models.graph.train import train_gat
    train_gat(epochs=50)

    print("\nAll deep learning models trained.")


if __name__ == "__main__":
    main()
