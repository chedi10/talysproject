from __future__ import annotations

"""
Build sequence datasets from transactions for LSTM/GRU training.

Output:
    X: np.ndarray of shape (n_samples, seq_len, input_dim)
    y: np.ndarray of shape (n_samples,)
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_TRANSACTIONS, RAW_REMBOURSEMENTS, RAW_CREDITS, TEST_SIZE, RANDOM_STATE


TX_TYPES = ["DEPOT", "RETRAIT", "REMBOURSEMENT", "TRANSFERT"]
TX_MAP = {name: i for i, name in enumerate(TX_TYPES)}
REMB_STATUTS = ["PAYE", "EN_RETARD"]
REMB_MAP = {name: i for i, name in enumerate(REMB_STATUTS)}


@dataclass
class SequenceDataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    input_dim: int
    seq_len: int


def _tx_row_to_vector(row: pd.Series) -> np.ndarray:
    """
    Convert one transaction row into a fixed-size numeric vector.
    Features:
    - amount (scaled later by global max)
    - suspect flag
    - one-hot type (4 dims)
    """
    one_hot = np.zeros(len(TX_TYPES), dtype=np.float32)
    idx = TX_MAP.get(str(row["type"]), None)
    if idx is not None:
        one_hot[idx] = 1.0

    # Vector layout (shared with remboursement events):
    # [amount_scaled, suspect, is_remb, retard_scaled, remb_status_onehot(2), tx_type_onehot(4)]
    base = np.array([float(row["montant"]), float(row["suspect"]), 0.0, 0.0], dtype=np.float32)
    remb_oh = np.zeros(len(REMB_STATUTS), dtype=np.float32)
    return np.concatenate([base, remb_oh, one_hot], axis=0)


def _remb_row_to_vector(row: pd.Series) -> np.ndarray:
    one_hot_tx = np.zeros(len(TX_TYPES), dtype=np.float32)
    statut = "EN_RETARD" if int(row.get("retard_jours", 0)) > 0 and str(row.get("statut", "")) != "PAYE" else str(row.get("statut", "PAYE"))
    remb_oh = np.zeros(len(REMB_STATUTS), dtype=np.float32)
    remb_oh[REMB_MAP.get(statut, 0)] = 1.0
    retard_scaled = float(row.get("retard_jours", 0)) / 365.0
    base = np.array([float(row["montant_du"]), 0.0, 1.0, float(retard_scaled)], dtype=np.float32)
    return np.concatenate([base, remb_oh, one_hot_tx], axis=0)


def _build_client_event_sequences(tx: pd.DataFrame, remb: pd.DataFrame) -> dict[int, list[np.ndarray]]:
    """
    Build per-client event sequences mixing transactions + remboursements.
    Events are sorted by date.
    """
    by_client: dict[int, list[tuple[pd.Timestamp, np.ndarray]]] = {}

    for client_id, grp in tx.groupby("client_id"):
        grp = grp.sort_values("date")
        by_client[int(client_id)] = [(d, _tx_row_to_vector(r)) for d, (_, r) in zip(grp["date"], grp.iterrows())]

    for client_id, grp in remb.groupby("client_id"):
        grp = grp.sort_values("date_echeance")
        items = [(d, _remb_row_to_vector(r)) for d, (_, r) in zip(grp["date_echeance"], grp.iterrows())]
        cur = by_client.get(int(client_id), [])
        by_client[int(client_id)] = cur + items

    # sort per client
    out: dict[int, list[np.ndarray]] = {}
    for cid, events in by_client.items():
        events = sorted(events, key=lambda t: t[0])
        out[cid] = [v for _, v in events]
    return out


def _pad_or_trim(seq: list[np.ndarray], seq_len: int, input_dim: int) -> np.ndarray:
    arr = np.zeros((seq_len, input_dim), dtype=np.float32)
    if not seq:
        return arr
    tail = seq[-seq_len:]
    arr[-len(tail):, :] = np.stack(tail)
    return arr


def build_sequence_dataset(seq_len: int = 30) -> SequenceDataset:
    """
    Build train/test arrays aligned to credit-level target `en_defaut`.
    For each credit, we use recent transactions of the same client.
    """
    tx = pd.read_csv(RAW_TRANSACTIONS, parse_dates=["date"])
    remb = pd.read_csv(RAW_REMBOURSEMENTS, parse_dates=["date_echeance", "date_paiement"])
    credits = pd.read_csv(RAW_CREDITS, parse_dates=["date_debut"])

    # Keep clean numeric types
    tx["client_id"] = tx["client_id"].astype(int)
    tx["montant"] = tx["montant"].astype(float)
    tx["suspect"] = tx["suspect"].astype(int)

    remb["client_id"] = remb["client_id"].astype(int)
    remb["montant_du"] = remb["montant_du"].astype(float)
    remb["retard_jours"] = remb["retard_jours"].astype(int)

    credits["client_id"] = credits["client_id"].astype(int)
    credits["en_defaut"] = credits["en_defaut"].astype(int)

    # Global scaling for amount to stabilize training
    max_amt = float(tx["montant"].max()) if len(tx) else 1.0
    tx["montant"] = tx["montant"] / max(max_amt, 1.0)

    max_due = float(remb["montant_du"].max()) if len(remb) else 1.0
    remb["montant_du"] = remb["montant_du"] / max(max_due, 1.0)

    by_client = _build_client_event_sequences(tx, remb)

    # amount_scaled, suspect, is_remb, retard_scaled, remb_status_onehot(2), tx_onehot(4)
    input_dim = 4 + len(REMB_STATUTS) + len(TX_TYPES)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for _, cr in credits.iterrows():
        cid = int(cr["client_id"])
        y = int(cr["en_defaut"])
        seq = by_client.get(cid, [])
        X_list.append(_pad_or_trim(seq, seq_len=seq_len, input_dim=input_dim))
        y_list.append(y)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return SequenceDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        input_dim=input_dim,
        seq_len=seq_len,
    )

