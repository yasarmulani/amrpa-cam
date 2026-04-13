"""
CAM: Injection
---------------
Injects compressed memory into current layer's attention computation.

V2 core principle — operate directly in compressed space:
    OLD (V1): retrieve compressed → reconstruct full (seq,seq) → inject
    NEW (V2): retrieve compressed → inject directly from proj_K, proj_V

This eliminates:
    - Full matrix reconstruction (was the main contradiction)
    - O(seq²) memory cost during injection
    - Noise from zero-filling unknown positions

How injection works in compressed space:
    CompressedMemory stores proj_K (batch, proj_rank, d_k)
                              proj_V (batch, proj_rank, d_k)

    These are importance-weighted projections of past K and V.
    They capture the reasoning-relevant subspace of past attention.

    Injection computes memory bias as:
        MV      = alpha_weighted_sum(proj_V across past steps)
        G       = sigmoid(gamma_g * sim(Q, MV) + bias_g)   ← gate
        gated_M = G ⊙ Wmem(MV)
        bias    = (gated_M @ K^T) / sqrt(d_k)

    Causal constraint (decoder only):
        Applied to final bias via upper-triangle masking.
        proj_K/proj_V from past steps never contain future tokens
        because they were compressed at earlier generation steps.
        So causal safety is guaranteed structurally, not just by masking.

Autoregression safety:
    Token i can only receive memory from steps ≤ current step.
    CAMMemoryBank stores only past steps (store after inject).
    Causal mask on bias prevents within-step future leakage.
    No data leakage by design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

from .cam_config import CAMConfig
from .cam_compression import CompressedMemory


class CAMInjector(nn.Module):
    """
    Injects CAM compressed memories into current layer attention.

    Operates entirely in compressed space (proj_rank × d_k).
    Never reconstructs full (seq, seq) attention matrix.

    Parameters from CAMConfig:
        d_k, gamma, alpha_temperature, gate_gamma_init, gate_bias_init, dropout
    """

    def __init__(self, config: CAMConfig):
        super().__init__()
        self.config = config
        self.d_k = config.d_k
        self.proj_rank = config.proj_rank

        # Gate parameters (same as AMRPA original)
        self.gate_gamma = nn.Parameter(torch.tensor(config.gate_gamma_init))
        self.gate_bias = nn.Parameter(torch.tensor(config.gate_bias_init))

        # Memory transformation (same as AMRPA original)
        self.w_mem = nn.Linear(config.d_k, config.d_k, bias=False)

        # Alpha MLP: selects which past step is most relevant for current query
        # Input: current Q summary + projected past V summary
        self.mlp_alpha = nn.Sequential(
            nn.Linear(config.d_k * 2, config.d_k),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_k, 1)
        )

    def forward(
        self,
        Q: torch.Tensor,                            # (batch, seq, d_k)
        K: torch.Tensor,                            # (batch, seq, d_k)
        V: torch.Tensor,                            # (batch, seq, d_k)
        memories: List[CompressedMemory],           # from memory bank
        layer_depth: float,
        attention_mask: Optional[torch.Tensor] = None   # (batch, seq) binary
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute memory bias from compressed memories.
        Works entirely in compressed space — no full matrix reconstruction.

        Returns:
            memory_bias: (batch, seq, seq) to add to base attention scores
            metrics:     dict of per-sample scalar tensors on device
        """
        batch, seq, _ = Q.shape
        device = Q.device

        zero_bias = torch.zeros(batch, seq, seq, device=device, dtype=Q.dtype)
        zero_metrics = self._zero_metrics(batch, device)

        if not memories:
            return zero_bias, zero_metrics

        # ── Alpha selection: which past step matters for current query ────
        # Q summary: mean over seq dim for efficiency
        Q_summary = Q.mean(dim=1)   # (batch, d_k)

        alpha_scores = []
        mv_list = []

        for k_idx, mem in enumerate(reversed(memories), 1):
            decay = self.config.gamma ** k_idx
        
            # proj_V: (stored_batch, proj_rank, d_k)
            # Trim to current batch size (last batch may be smaller)
            cur_batch = Q_summary.shape[0]
            proj_V = mem.proj_V[:cur_batch]             # trim batch dim
            past_V_summary = proj_V.mean(dim=1) * decay # (cur_batch, d_k)
            mv_list.append(past_V_summary)
        
            # Alpha input: current Q context + past V summary
            alpha_in = torch.cat([Q_summary, past_V_summary], dim=-1)

        # for k_idx, mem in enumerate(reversed(memories), 1):
        #     decay = self.config.gamma ** k_idx

        #     # proj_V: (batch, proj_rank, d_k)
        #     # Summarise past V: mean over proj_rank dim → (batch, d_k)
        #     past_V_summary = mem.proj_V.mean(dim=1) * decay  # (batch, d_k)
        #     mv_list.append(past_V_summary)

        #     # Alpha input: current Q context + past V summary
        #     alpha_in = torch.cat([Q_summary, past_V_summary], dim=-1)
        #     # (batch, d_k*2)
        #     score = self.mlp_alpha(alpha_in)   # (batch, 1)
        #     alpha_scores.append(score)

        # Stack and softmax over past steps
        alpha_tensor = torch.cat(alpha_scores, dim=-1)   # (batch, n_past)
        alpha_weights = F.softmax(
            alpha_tensor / self.config.alpha_temperature, dim=-1
        )   # (batch, n_past)

        # Alpha diversity metric
        token_entropy = -(alpha_weights * torch.log(alpha_weights + 1e-9)).sum(-1)
        # (batch,)

        # Weighted combination of past V summaries
        mv_stack = torch.stack(mv_list, dim=-1)   # (batch, d_k, n_past)
        MV_summary = torch.bmm(
            mv_stack, alpha_weights.unsqueeze(-1)
        ).squeeze(-1)   # (batch, d_k)

        # ── Gate: similarity between current query and memory ────────────
        # Expand MV_summary to all token positions for per-token gating
        MV_expanded = MV_summary.unsqueeze(1).expand(-1, seq, -1)
        # (batch, seq, d_k)

        sim = (Q * MV_expanded).sum(dim=-1) / math.sqrt(self.d_k)
        # (batch, seq)
        G = torch.sigmoid(self.gate_gamma * sim + self.gate_bias)
        # (batch, seq)

        # ── Memory transformation ────────────────────────────────────────
        gated_mem = G.unsqueeze(-1) * self.w_mem(MV_expanded)
        # (batch, seq, d_k)

        # ── Memory bias: inject into attention scores ────────────────────
        # gated_mem @ K^T gives (batch, seq, seq)
        # This adds memory-derived bias to how current tokens attend to each other
        memory_bias = torch.bmm(
            gated_mem,
            K.transpose(1, 2)
        ) / math.sqrt(self.d_k)
        # (batch, seq, seq)

        # ── Causal mask (decoder only) ───────────────────────────────────
        if self.config.causal:
            causal_mask = torch.triu(
                torch.full((seq, seq), float('-inf'), device=device),
                diagonal=1
            )
            memory_bias = memory_bias + causal_mask

        # ── Padding mask ─────────────────────────────────────────────────
        if attention_mask is not None:
            # Mask key positions that are padding
            key_pad = (attention_mask == 0).unsqueeze(1)   # (batch, 1, seq)
            memory_bias = memory_bias.masked_fill(key_pad, float('-inf'))

        # ── Metrics (kept on device) ─────────────────────────────────────
        metrics = {
            'gate_impact':          G.mean(dim=1),           # (batch,)
            'gate_variance':        G.var(dim=1),            # (batch,)
            'alpha_diversity':      token_entropy,           # (batch,)
            'memory_contribution':  gated_mem.norm(dim=-1).mean(dim=1),  # (batch,)
            'using_memory':         torch.ones(batch, device=device)
        }

        return memory_bias, metrics

    def _zero_metrics(self, batch: int, device: torch.device) -> dict:
        return {
            'gate_impact':         torch.zeros(batch, device=device),
            'gate_variance':       torch.zeros(batch, device=device),
            'alpha_diversity':     torch.zeros(batch, device=device),
            'memory_contribution': torch.zeros(batch, device=device),
            'using_memory':        torch.zeros(batch, device=device)
        }
