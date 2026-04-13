"""
cleaner.py – Data validation, cleaning and export to Parquet.

Usage:
    from src.data.cleaner import clean_and_save
    data_clean = clean_and_save(data_raw)
"""
import pandas as pd
from pathlib import Path
from src.config import PROCESSED_DIR


def _report_nulls(name: str, df: pd.DataFrame) -> None:
    """Print null-value counts for a DataFrame."""
    nulls = df.isnull().sum()
    total_nulls = nulls.sum()
    if total_nulls > 0:
        print(f"  {name}: {total_nulls} null values found")
        print(nulls[nulls > 0].to_string())
    else:
        print(f"  {name}: no nulls")


def clean_clients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the clients DataFrame.
    - Drop duplicates on client_id
    - Assert no nulls in critical columns
    - Cap age to realistic range
    """
    df = df.drop_duplicates(subset="client_id")
    df["age"] = df["age"].clip(18, 80)
    df["revenu_mensuel"] = df["revenu_mensuel"].clip(0, 50_000)
    _report_nulls("clients", df)
    return df


def clean_credits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the credits DataFrame.
    - Drop duplicates on credit_id
    - Clip dti to [0, 2]
    - montant > 0
    """
    df = df.drop_duplicates(subset="credit_id")
    df["dti"] = df["dti"].clip(0.0, 2.0)
    df = df[df["montant"] > 0].copy()
    _report_nulls("credits", df)
    return df


def clean_remboursements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the remboursements DataFrame.
    - Drop duplicates on remb_id
    - Clip retard_jours to [0, 365]
    """
    df = df.drop_duplicates(subset="remb_id")
    df["retard_jours"] = df["retard_jours"].clip(0, 365)
    _report_nulls("remboursements", df)
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the transactions DataFrame.
    - Drop duplicates on transaction_id
    - montant > 0
    """
    df = df.drop_duplicates(subset="transaction_id")
    df = df[df["montant"] > 0].copy()
    _report_nulls("transactions", df)
    return df


def clean_relations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the relations DataFrame.
    - Drop self-loops (source == target)
    - Drop duplicate edges
    """
    df = df[df["source_client_id"] != df["target_client_id"]].copy()
    df = df.drop_duplicates(subset=["source_client_id", "target_client_id", "type_relation"])
    _report_nulls("relations", df)
    return df


def clean_and_save(raw_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Apply all cleaning functions, then persist each DataFrame as Parquet
    under data/processed/.

    Parameters
    ----------
    raw_data : dict returned by loader.load_raw_data()

    Returns
    -------
    dict with same keys, cleaned DataFrames
    """
    # Windows console-safe messages (no emojis)
    print("Cleaning data...")
    cleaned = {
        "clients":        clean_clients(raw_data["clients"]),
        "credits":        clean_credits(raw_data["credits"]),
        "remboursements": clean_remboursements(raw_data["remboursements"]),
        "transactions":   clean_transactions(raw_data["transactions"]),
        "relations":      clean_relations(raw_data["relations"]),
    }

    print("\nSaving cleaned data to parquet...")
    for name, df in cleaned.items():
        out_path: Path = PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"   {out_path.name}  ({df.shape[0]:,} rows)")

    print("Cleaning complete.\n")
    return cleaned


def load_processed() -> dict[str, pd.DataFrame]:
    """
    Load previously saved processed Parquet files (fast re-read).
    """
    names = ["clients", "credits", "remboursements", "transactions", "relations"]
    return {n: pd.read_parquet(PROCESSED_DIR / f"{n}.parquet") for n in names}
