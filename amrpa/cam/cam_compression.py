"""
CAM: Compression
-----------------
Compresses attention state into fixed-size memory.

V2 changes from V1:
    ✗ Removed: SVD (expensive, unstable in forward pass)
    ✗ Removed: reconstruct_attention() (caused full matrix rebuild — contradiction)
    ✓ Added:   LearnableProjection replacing SVD
    ✓ Added:   CompressedMemory stores K/V projections only, never full matrix
    ✓ Design:  injection operates directly on compressed form, never rebuilds (seq,seq)

Core idea:
    Instead of storing full (seq, seq) attention matrix and reconstructing it,
    store compact (seq, proj_rank) projections of K and V weighted by importance.
    Injection then works directly in this projected space.

This is the key conceptual fix:
    Old: compress A → store → reconstruct A → inject
    New: compress K,V → store → inject directly from K,V projections

Memory cost (fixed regardless of sequence length growth):
    Per CompressedMemory:
        proj_K:     (batch, proj_rank, d_k)
        proj_V:     (batch, proj_rank, d_k)
        importance: (batch, seq)              ← kept for analysis/visualization
        step:       int
        seq_len:    int

    Total per layer per step: proj_rank × d_k × 2 × batch × 4 bytes
    Example: rank=16, d_k=64, batch=32 → 16×64×2×32×4 = 262 KB
    Compare to raw (seq,seq): 384×384×32×4 = 18 MB per layer → 97% reduction
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from .cam_config import CAMConfig


@dataclass
class CompressedMemory:
    """
    Fixed-size compressed representation of attention state at one step.

    Shapes are independent of sequence length (proj_rank << seq).
    This is the key invariant: memory cost does not grow with sequence.

    Fields:
        proj_K:     (batch, proj_rank, d_k)  importance-weighted K projection
        proj_V:     (batch, proj_rank, d_k)  importance-weighted V projection
        importance: (batch, seq)             soft weights (kept for analysis)
        step:       int                      generation step index
        seq_len:    int                      original sequence length at this step
    """
    proj_K: torch.Tensor
    proj_V: torch.Tensor
    importance: torch.Tensor
    step: int
    seq_len: int

    def detach(self) -> 'CompressedMemory':
        """Detach from computation graph for storage in memory bank."""
        return CompressedMemory(
            proj_K=self.proj_K.detach(),
            proj_V=self.proj_V.detach(),
            importance=self.importance.detach(),
            step=self.step,
            seq_len=self.seq_len
        )

    def to(self, device) -> 'CompressedMemory':
        return CompressedMemory(
            proj_K=self.proj_K.to(device),
            proj_V=self.proj_V.to(device),
            importance=self.importance.to(device),
            step=self.step,
            seq_len=self.seq_len
        )

    def memory_bytes(self) -> int:
        total = 0
        for t in [self.proj_K, self.proj_V, self.importance]:
            total += t.numel() * t.element_size()
        return total


class CAMCompressor(nn.Module):
    """
    Compresses (K, V, importance) into fixed-size CompressedMemory.

    Replaces SVD with LearnableProjection:
        proj_matrix: (d_k, proj_rank)
        proj_K[b] = importance[b] @ K[b] @ proj_matrix  → (proj_rank, d_k)
        proj_V[b] = importance[b] @ V[b] @ proj_matrix  → (proj_rank, d_k)

    Why learnable projection over SVD:
        - SVD: expensive O(seq²·d_k), not differentiable w.r.t. input
        - LearnableProjection: O(seq·d_k·rank), fully differentiable
        - Model learns which subspace preserves reasoning-relevant structure
        - Scales to long sequences and large models

    Note: Q is NOT compressed. Q is only used in importance computation
    and in injection (gate computation). It lives in current step only.
    """

    def __init__(self, config: CAMConfig):
        super().__init__()
        self.config = config
        self.proj_rank = config.proj_rank

        # Learnable projection: maps d_k → proj_rank subspace
        # One projection shared for K and V (they live in same space)
        self.proj_matrix = nn.Linear(config.d_k, config.proj_rank, bias=False)

        # Orthogonal init: preserves distances in projected space
        nn.init.orthogonal_(self.proj_matrix.weight)

    def forward(
        self,
        K: torch.Tensor,                   # (batch, seq, d_k)
        V: torch.Tensor,                   # (batch, seq, d_k)
        importance: torch.Tensor,          # (batch, seq) from LearnedImportance
        step: int
    ) -> CompressedMemory:
        """
        Compress K, V into fixed-size memory using importance-weighted projection.

        Operation:
            importance-weighted K: imp^T @ K → (batch, proj_rank, d_k)
            where imp = importance.unsqueeze(-1) weights each token's contribution

        Returns CompressedMemory with fixed shape regardless of seq length.
        """
        batch, seq, d_k = K.shape

        # importance: (batch, seq) → (batch, seq, 1) for broadcasting
        imp = importance.unsqueeze(-1)    # (batch, seq, 1)

        # Importance-weighted K and V: each token contributes proportionally
        # weighted_K: (batch, seq, d_k) with each row scaled by importance
        weighted_K = imp * K              # (batch, seq, d_k)
        weighted_V = imp * V              # (batch, seq, d_k)

        # Project into rank-r subspace: (batch, seq, d_k) → (batch, seq, proj_rank)
        proj_K_seq = self.proj_matrix(weighted_K)   # (batch, seq, proj_rank)
        proj_V_seq = self.proj_matrix(weighted_V)   # (batch, seq, proj_rank)

        # Collapse seq dimension via sum: (batch, seq, proj_rank) → (batch, proj_rank, d_k)
        # This gives us a fixed-size summary independent of seq length
        # Transpose d_k back: we need (batch, proj_rank, d_k) for injection
        proj_K = proj_K_seq.transpose(1, 2) @ K / (seq + 1e-8)
        # proj_K: (batch, proj_rank, d_k)
        proj_V = proj_V_seq.transpose(1, 2) @ V / (seq + 1e-8)
        # proj_V: (batch, proj_rank, d_k)

        return CompressedMemory(
            proj_K=proj_K,
            proj_V=proj_V,
            importance=importance.detach(),  # kept for analysis only
            step=step,
            seq_len=seq
        )
