from __future__ import annotations

"""
PyTorch recurrent classifier used for both LSTM and GRU runs.
"""

import torch
import torch.nn as nn


class RecurrentCreditRiskModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        rnn_type: str = "lstm",  # "lstm" or "gru"
    ):
        super().__init__()
        rnn_type = rnn_type.lower().strip()
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.rnn(x)
        last = out[:, -1, :]  # last timestep embedding
        logits = self.head(last).squeeze(-1)
        return logits

