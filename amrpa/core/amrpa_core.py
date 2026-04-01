"""
AMRPA Core Mechanism
----------------------
The four claims of AMRPA implemented as a clean nn.Module.

Ported from original full_roberta_amrpa_run.py into library form.

Four claims:
    CLAIM 1: Similarity-based Smart Gatekeeper (gate G)
    CLAIM 2: Dynamic Memory Selection (alpha MLP)
    CLAIM 3: Fading Ink / Memory Decay (gamma^k)
    CLAIM 4: Adaptive Memory Depth (log-scaled window)

This module is architecture-agnostic.
It receives Q, K, V and attention_history, returns memory_bias and metrics.
The adapter (encoder.py / decoder.py) handles layer detection and patching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Dict

from ..config import AMRPAConfig


class AMRPACore(nn.Module):
    """
    Core AMRPA mechanism for one layer.

    Computes memory bias from attention history and injects into
    current layer's attention scores.

    This is a direct port of AMRPA_SelfAttentionWrapper.compute_memory_bias()
    from the original script, refactored into a clean standalone module.

    Args:
        config: AMRPAConfig with all hyperparameters
    """

    def __init__(self, config: AMRPAConfig):
        super().__init__()
        self.config = config
        d_k = config.d_k

        # CLAIM 1: Similarity-based Smart Gatekeeper
        self.gamma_g = nn.Parameter(torch.tensor(config.gate_gamma_init))
        self.bias_g  = nn.Parameter(torch.tensor(config.gate_bias_init))

        # CLAIM 2: Dynamic Memory Selection (alpha MLP)
        self.mlp_alpha = nn.Sequential(
            nn.Linear(d_k * 2, config.d_mlp),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_mlp, 1)
        )

        # Memory transformation
        self.w_mem          = nn.Linear(d_k, d_k, bias=False)
        self.proj_attention = nn.Linear(d_k, d_k, bias=False)

    def adaptive_window_size(self, relative_layer_idx: int) -> int:
        """
        CLAIM 4: Adaptive Memory Depth
        Earlier AMRPA layers look at less history.
        Later AMRPA layers can integrate more.
        """
        if relative_layer_idx <= 2:
            return 1
        elif relative_layer_idx <= 8:
            return math.floor(math.log2(relative_layer_idx)) + 1
        else:
            return 4

    def forward(
        self,
        Q: torch.Tensor,                        # (batch, seq, d_k)
        K: torch.Tensor,                        # (batch, seq, d_k)
        V: torch.Tensor,                        # (batch, seq, d_k)
        attention_history: List[torch.Tensor],  # list of past (batch, seq, seq) attention matrices
        relative_layer_idx: int,                # position within AMRPA layers (1-based)
        causal_mask: Optional[torch.Tensor] = None,  # (seq, seq) for decoder
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute memory bias from attention history.

        Returns:
            memory_bias: (batch, seq, seq) — add to base attention scores
            metrics:     dict of per-sample scalar tensors
        """
        batch_size, seq_len, d_k = Q.shape
        device = Q.device

        # Zero metrics — returned when no memory available
        zero_metrics = self._zero_metrics(batch_size, device)

        # CLAIM 4: need at least layer 2 and some history
        if relative_layer_idx <= 1 or len(attention_history) == 0:
            return torch.zeros(batch_size, seq_len, seq_len, device=device), zero_metrics

        # CLAIM 4: adaptive window
        w = self.adaptive_window_size(relative_layer_idx)
        memory_window = attention_history[-min(relative_layer_idx - 1, w):]

        alpha_scores    = []
        decayed_patterns = []

        # CLAIM 3: Fading Ink — exponential decay on past patterns
        for k, past_attention in enumerate(reversed(memory_window), 1):
            decay_factor = self.config.gamma ** k
            noise        = torch.rand_like(past_attention) * self.config.epsilon
            decayed_A    = decay_factor * past_attention + noise
            decayed_patterns.append(decayed_A)

            # CLAIM 2: Dynamic Memory Selection — alpha MLP scores each past layer
            projected_values = torch.matmul(decayed_A, V)
            proj_A           = self.proj_attention(projected_values)
            alpha_input      = torch.cat([Q, proj_A], dim=-1)
            alpha_score      = self.mlp_alpha(alpha_input)
            alpha_scores.append(alpha_score)

        if not alpha_scores:
            return torch.zeros(batch_size, seq_len, seq_len, device=device), zero_metrics

        # Alpha weights: which past layer matters most
        alpha_tensor  = torch.cat(alpha_scores, dim=-1)            # (batch, seq, w)
        alpha_weights = F.softmax(
            alpha_tensor / self.config.alpha_temperature, dim=-1
        )

        # Alpha diversity metric — entropy of alpha distribution
        token_entropy             = -(alpha_weights * torch.log(alpha_weights + 1e-9)).sum(dim=-1)
        alpha_diversity_per_sample = token_entropy.mean(dim=1)     # (batch,)

        # Weighted combination of decayed attention patterns
        memory_stack = torch.stack(decayed_patterns, dim=-1)       # (batch, seq, seq, w)
        M = torch.sum(
            alpha_weights.unsqueeze(2) * memory_stack, dim=-1
        )                                                           # (batch, seq, seq)

        # Memory context vectors
        MV     = torch.matmul(M, V)                                # (batch, seq, d_k)
        M_proj = self.proj_attention(MV)

        # CLAIM 1: Smart Gatekeeper — similarity-based gate
        sim_score = (Q * M_proj).sum(dim=-1) / math.sqrt(d_k)     # (batch, seq)
        G         = torch.sigmoid(self.gamma_g * sim_score + self.bias_g)

        # Per-sample gate metrics
        gate_impact_per_sample   = G.mean(dim=1)                   # (batch,)
        gate_variance_per_sample = G.var(dim=1)                    # (batch,)

        # Gated memory
        M_transformed = self.w_mem(torch.matmul(M, V))            # (batch, seq, d_k)
        gated_memory  = G.unsqueeze(-1) * M_transformed           # (batch, seq, d_k)

        # Memory contribution metric
        memory_contribution_per_sample = gated_memory.norm(dim=-1).mean(dim=1)

        # Memory bias — injected into attention scores
        memory_bias = torch.matmul(
            gated_memory, K.transpose(-2, -1)
        ) / math.sqrt(d_k)                                         # (batch, seq, seq)

        # Causal mask for decoder
        if causal_mask is not None:
            memory_bias = memory_bias + causal_mask

        metrics = {
            'gate_impact':          gate_impact_per_sample,
            'gate_variance':        gate_variance_per_sample,
            'alpha_diversity':      alpha_diversity_per_sample,
            'memory_contribution':  memory_contribution_per_sample,
            'using_memory':         torch.ones(batch_size, device=device)
        }

        return memory_bias, metrics

    def _zero_metrics(self, batch_size: int, device: torch.device) -> Dict:
        return {
            'gate_impact':         torch.zeros(batch_size, device=device),
            'gate_variance':       torch.zeros(batch_size, device=device),
            'alpha_diversity':     torch.zeros(batch_size, device=device),
            'memory_contribution': torch.zeros(batch_size, device=device),
            'using_memory':        torch.zeros(batch_size, device=device)
        }
