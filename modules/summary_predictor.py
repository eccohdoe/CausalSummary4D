from typing import Dict, Iterable

import torch
import torch.nn as nn


class SlotMLPPredictor(nn.Module):
    def __init__(self, slot_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, slot_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SlotTransformerPredictor(nn.Module):
    def __init__(
        self,
        slot_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        nhead = max(1, min(num_heads, slot_dim))
        while slot_dim % nhead != 0 and nhead > 1:
            nhead -= 1
        layer = nn.TransformerEncoderLayer(
            d_model=slot_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(slot_dim)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(slot_dim, slot_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-2], original_shape[-1])
        x = self.encoder(self.norm(x))
        x = self.proj(x)
        return x.reshape(original_shape)


class MultiHorizonSummaryPredictor(nn.Module):
    def __init__(
        self,
        slot_dim: int,
        horizons: Iterable[int],
        predictor_type: str = 'mlp',
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.horizons = [int(h) for h in horizons]
        self.predictor_type = predictor_type
        self.predictors = nn.ModuleDict()
        self.horizon_embeddings = nn.ParameterDict()
        for horizon in self.horizons:
            if predictor_type == 'mlp':
                module = SlotMLPPredictor(slot_dim=slot_dim, hidden_dim=hidden_dim, dropout=dropout)
            elif predictor_type == 'transformer':
                module = SlotTransformerPredictor(
                    slot_dim=slot_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            else:
                raise ValueError(f'Unsupported predictor_type: {predictor_type}')
            self.predictors[str(horizon)] = module
            self.horizon_embeddings[str(horizon)] = nn.Parameter(torch.randn(1, 1, 1, slot_dim) * 0.02)

    def forward(self, summary_tokens: torch.Tensor) -> Dict[int, torch.Tensor]:
        predictions = {}
        for horizon in self.horizons:
            if summary_tokens.size(1) <= horizon:
                continue
            source = summary_tokens[:, :-horizon] + self.horizon_embeddings[str(horizon)]
            predictions[horizon] = self.predictors[str(horizon)](source)
        return predictions
