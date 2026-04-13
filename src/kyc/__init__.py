"""
KYC (Know Your Customer) score estimation module.

Estimates a KYC score (0-100) from client attributes.
This score is used as input to the credit risk model.
"""

from src.kyc.score import compute_kyc_score, fit_kyc_scorer

__all__ = ["compute_kyc_score", "fit_kyc_scorer"]
