from __future__ import annotations

"""Inference helpers for Deep Tabular Network."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.deep_tabular.model import DeepTabularNet


def load_deep_tabular(checkpoint_path: Path) -> tuple[DeepTabularNet, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = DeepTabularNet(
        n_numeric=len(ckpt["num_cols"]),
        cat_cardinalities=ckpt["cat_cardinalities"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def predict_proba(model: DeepTabularNet, meta: dict, features: dict[str, float]) -> float:
    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]
    x_num = torch.tensor([[float(features[c]) for c in num_cols]], dtype=torch.float32)
    x_cat = torch.tensor([[int(features[c]) for c in cat_cols]], dtype=torch.long)
    with torch.no_grad():
        logit = model(x_num, x_cat).item()
    return float(1.0 / (1.0 + np.exp(-logit)))
