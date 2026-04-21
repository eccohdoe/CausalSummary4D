import math
from typing import Optional

import torch
import torch.nn as nn


class SummaryExtractor(nn.Module):
    """Extract causal summary tokens either from pooled states or token sets."""

    def __init__(
        self,
        input_dim: int,
        num_slots: int,
        slot_dim: int,
        mode: str = 'pooled_mlp',
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.mode = mode

        if mode == 'pooled_mlp':
            hidden_dim = hidden_dim or max(input_dim, num_slots * slot_dim)
            self.norm = nn.LayerNorm(input_dim)
            self.project = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_slots * slot_dim),
            )
            self.slot_bias = nn.Parameter(torch.zeros(1, 1, num_slots, slot_dim))
            self.out_norm = nn.LayerNorm(slot_dim)
        elif mode == 'token_query':
            self.token_norm = nn.LayerNorm(input_dim)
            self.key_proj = nn.Linear(input_dim, slot_dim)
            self.value_proj = nn.Linear(input_dim, slot_dim)
            self.query_tokens = nn.Parameter(torch.randn(1, 1, num_slots, slot_dim) * 0.02)
            self.query_norm = nn.LayerNorm(slot_dim)
            self.out_norm = nn.LayerNorm(slot_dim)
            self.out_proj = nn.Sequential(
                nn.Linear(slot_dim, slot_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(slot_dim, slot_dim),
            )
        else:
            raise ValueError(f'Unsupported summary extractor mode: {mode}')

    def forward(self, frame_states: torch.Tensor, token_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mode == 'pooled_mlp':
            return self._forward_pooled(frame_states)
        if self.mode == 'token_query':
            return self._forward_token_query(token_states)
        raise ValueError(f'Unsupported summary extractor mode: {self.mode}')

    def _forward_pooled(self, frame_states: torch.Tensor) -> torch.Tensor:
        bsz, steps, _ = frame_states.shape
        summary = self.project(self.norm(frame_states))
        summary = summary.view(bsz, steps, self.num_slots, self.slot_dim)
        summary = self.out_norm(summary + self.slot_bias)
        return summary

    def _forward_token_query(self, token_states: Optional[torch.Tensor]) -> torch.Tensor:
        if token_states is None:
            raise ValueError('token_states are required when summary_mode=token_query')
        token_states = self.token_norm(token_states)
        keys = self.key_proj(token_states)
        values = self.value_proj(token_states)
        queries = self.query_norm(self.query_tokens.expand(token_states.size(0), token_states.size(1), -1, -1))
        attn = torch.einsum('btkd,btnd->btkn', queries, keys) / math.sqrt(self.slot_dim)
        weight = torch.softmax(attn, dim=-1)
        summary = torch.einsum('btkn,btnd->btkd', weight, values)
        summary = self.out_norm(summary + self.out_proj(summary))
        return summary
