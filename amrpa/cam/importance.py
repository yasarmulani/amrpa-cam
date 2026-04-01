"""
CAM: Learned Importance
------------------------
Computes per-token importance weights for CAM compression.

Design principles:
    - Fully differentiable (softmax, no hard selection)
    - Context-aware: uses Q, K, V, attention pattern, and layer depth
    - Mask-aware: padding tokens get zero importance
    - Initialised to near-uniform so training starts stable

Role in CAM pipeline:
    importance.py → compression.py → memory_bank.py → injection.py
    ^
    Runs first. Output feeds directly into compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .cam_config import CAMConfig


class LearnedImportance(nn.Module):
    """
    Learns which tokens are reasoning-important at each layer.

    Per-token input features:
        Q[i]        what this token queries          (d_k,)
        K[i]        what this token offers           (d_k,)
        V[i]        what information it carries      (d_k,)
        col_attn[i] column mean of attention row     (1,)
                    = how much others attend TO this token
                    = hub signal in reasoning graph
        layer_depth normalized depth 0.0 → 1.0       (1,)

    Output:
        importance  (batch, seq) soft weights, sum=1 over real tokens
                    fully differentiable
    """

    def __init__(self, config: CAMConfig):
        super().__init__()
        self.config = config
        input_dim = config.d_k * 3 + 2   # Q + K + V + col_attn + depth

        self.net = nn.Sequential(
            nn.Linear(input_dim, config.importance_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.importance_hidden, config.importance_hidden // 2),
            nn.ReLU(),
            nn.Linear(config.importance_hidden // 2, 1)
        )

        # Near-zero init → near-uniform importance at start of training
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        Q: torch.Tensor,                        # (batch, seq, d_k)
        K: torch.Tensor,                        # (batch, seq, d_k)
        V: torch.Tensor,                        # (batch, seq, d_k)
        A: torch.Tensor,                        # (batch, seq, seq)
        layer_depth: float,
        attention_mask: Optional[torch.Tensor] = None   # (batch, seq) binary
    ) -> torch.Tensor:
        """
        Returns:
            importance: (batch, seq) differentiable soft weights summing to 1
        """
        batch, seq, _ = Q.shape

        # Column mean: how much other tokens attend TO each token
        col_attn = A.mean(dim=1, keepdim=False).unsqueeze(-1)  # (batch, seq, 1)

        # Layer depth scalar broadcast to all tokens
        depth = torch.full(
            (batch, seq, 1), layer_depth,
            device=Q.device, dtype=Q.dtype
        )

        features = torch.cat([Q, K, V, col_attn, depth], dim=-1)
        logits = self.net(features).squeeze(-1)            # (batch, seq)

        # Mask padding before softmax
        if attention_mask is not None:
            logits = logits.masked_fill(attention_mask == 0, float('-inf'))

        importance = F.softmax(logits, dim=-1)
        importance = torch.nan_to_num(importance, nan=0.0)
        return importance


class HeuristicImportance(nn.Module):
    """
    Non-learned fallback: column mean of attention matrix.
    Use as ablation baseline against LearnedImportance.

    Ablation experiment:
        LearnedImportance vs HeuristicImportance on same task
        → quantifies value of learned compression
    """

    def __init__(self, config: CAMConfig):
        super().__init__()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        A: torch.Tensor,
        layer_depth: float,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        col_attn = A.mean(dim=1)   # (batch, seq)
        if attention_mask is not None:
            col_attn = col_attn.masked_fill(attention_mask == 0, float('-inf'))
        importance = F.softmax(col_attn, dim=-1)
        return torch.nan_to_num(importance, nan=0.0)


def build_importance(config: CAMConfig) -> nn.Module:
    """Factory: returns correct importance module based on config."""
    if config.use_learned_importance:
        return LearnedImportance(config)
    return HeuristicImportance(config)
