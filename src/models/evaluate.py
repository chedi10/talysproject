"""
evaluate.py – Model evaluation utilities.

Generates classification report, ROC curve and confusion matrix.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    average_precision_score,
)
from src.config import FIGURES_DIR


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str = "model",
    save_figures: bool = True,
) -> dict:
    """
    Evaluate a trained classifier and optionally save figures.

    Returns
    -------
    dict with auc_roc, avg_precision, and classification report string
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc     = roc_auc_score(y_test, y_proba)
    avg_pre = average_precision_score(y_test, y_proba)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  AUC-ROC            : {auc:.4f}")
    print(f"  Avg Precision (AP) : {avg_pre:.4f}")
    print()
    report = classification_report(y_test, y_pred, target_names=["Non-défaut", "Défaut"])
    print(report)

    if save_figures:
        _plot_roc(y_test, y_proba, auc, model_name)
        _plot_confusion_matrix(y_test, y_pred, model_name)

    return {
        "model_name":        model_name,
        "auc_roc":           round(auc, 4),
        "avg_precision":     round(avg_pre, 4),
        "classification_report": report,
    }


def _plot_roc(y_test, y_proba, auc: float, model_name: str) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIGURES_DIR / f"roc_{model_name.replace(' ', '_').lower()}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  ROC curve saved at {out}")


def _plot_confusion_matrix(y_test, y_pred, model_name: str) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-défaut", "Défaut"],
        yticklabels=["Non-défaut", "Défaut"],
        ax=ax,
    )
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(f"Confusion Matrix – {model_name}")
    fig.tight_layout()
    out = FIGURES_DIR / f"cm_{model_name.replace(' ', '_').lower()}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  Confusion matrix saved at {out}")


def compare_models(results: list[dict]) -> None:
    """Print a side-by-side comparison of AUC-ROC scores."""
    print("\n" + "="*45)
    print("  MODEL COMPARISON")
    print("="*45)
    sorted_r = sorted(results, key=lambda x: x["auc_roc"], reverse=True)
    for i, r in enumerate(sorted_r):
        marker = "*" if i == 0 else " "
        print(f"  {marker} {r['model_name']:<28} AUC={r['auc_roc']:.4f}  AP={r['avg_precision']:.4f}")
    print("="*45)
