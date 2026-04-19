from __future__ import annotations


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError(
            "torch is required for lstm_gru_v2. Install with: pip install -e '.[nlp]'"
        ) from exc
    return torch, nn


def build_lstm_gru_model(input_size: int, hidden_size: int = 64, num_layers: int = 2):
    torch, nn = require_torch()

    class StackedLstmGru(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
            self.gru = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.dropout = nn.Dropout(0.2)
            self.head_reg = nn.Linear(hidden_size, 4)
            self.head_cls = nn.Linear(hidden_size, 3)

        def forward(self, x):
            x, _ = self.lstm(x)
            x, _ = self.gru(x)
            x = x[:, -1, :]
            x = self.dropout(x)
            reg = self.head_reg(x)
            cls = self.head_cls(x)
            return reg, cls

    return StackedLstmGru()
