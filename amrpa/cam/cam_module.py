"""
CAM: Module
------------
Orchestrates the full CAM pipeline for one AMRPA layer.

Pipeline per forward call:
    1. Retrieve stored memories from bank (oldest first)
    2. Inject memories into current attention via CAMInjector
    3. Compress current K, V using importance weights
    4. Store compressed memory into bank

Encoder vs Decoder behaviour:
    Encoder (arch='encoder'):
        - No CAM compression: stores raw K, V projections with uniform importance
        - No causal mask
        - Sequence length fixed: no growing shape problem
        - use_cam_compression = False (from config)

    Decoder (arch='decoder'):
        - CAM compression: LearnedImportance + CAMCompressor
        - Causal mask enforced in injector
        - Sequence grows step by step: fixed-size memory via compression
        - use_cam_compression = True (from config)

    Encoder-Decoder:
        - Encoder side: encoder behaviour, side='encoder'
        - Decoder side: decoder behaviour, side='decoder'
        - Banks isolated via CAMMemoryBankSet

Integration contract with AMRPA wrappers:
    Each AMRPA layer wrapper holds one CAMModule instance.
    Wrapper calls: memory_bias, metrics = cam(Q, K, V, A, layer_depth, mask, side)
    Wrapper is responsible for adding memory_bias to base attention scores.
    CAMMemoryBankSet is shared across all layers of one model.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .cam_config import CAMConfig
from .importance import build_importance
from .cam_compression import CAMCompressor, CompressedMemory
from .memory_bank import CAMMemoryBankSet
from .injection import CAMInjector


class CAMModule(nn.Module):
    """
    Full CAM module for one AMRPA layer.

    One instance per AMRPA layer per model side.

    Args:
        config:     CAMConfig — all hyperparameters
        layer_idx:  index within AMRPA layers (0 = first AMRPA layer from bottom)

    Usage:
        cam = CAMModule(config, layer_idx=0)
        cam.set_memory_bank(shared_bank_set)

        # In forward pass:
        cam.reset()                    # once per sequence, before first step
        bias, metrics = cam(Q, K, V, A, layer_depth, attention_mask)
        # add bias to base attention scores in wrapper
    """

    def __init__(self, config: CAMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Importance: used for both encoder (uniform) and decoder (learned)
        self.importance_net = build_importance(config)

        # Compressor: used for decoder only
        # For encoder, we store a lightweight uniform-importance projection
        self.compressor = CAMCompressor(config)

        # Injector: shared logic for both encoder and decoder
        # Causal mask handled internally based on config.causal
        self.injector = CAMInjector(config)

        # Memory bank reference — set externally by wrapper
        self._bank: Optional[CAMMemoryBankSet] = None
        self._step: int = 0

    def set_memory_bank(self, bank: CAMMemoryBankSet) -> None:
        """
        Attach shared memory bank.
        Called once by the AMRPA wrapper after creating this module.
        """
        self._bank = bank

    def reset(self) -> None:
        """
        Reset step counter for this layer.
        Memory bank reset is handled externally by wrapper (shared across layers).
        """
        self._step = 0

    def forward(
        self,
        Q: torch.Tensor,                            # (batch, seq, d_k)
        K: torch.Tensor,                            # (batch, seq, d_k)
        V: torch.Tensor,                            # (batch, seq, d_k)
        A: torch.Tensor,                            # (batch, seq, seq) base attention
        layer_depth: float,                         # normalized 0.0 → 1.0
        attention_mask: Optional[torch.Tensor] = None,   # (batch, seq) binary
        side: str = 'main'                          # 'main'|'encoder'|'decoder'
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full CAM forward: retrieve → inject → compress → store.

        Returns:
            memory_bias: (batch, seq, seq) — add to base attention logits
            metrics:     dict of per-sample tensors on device
        """
        assert self._bank is not None, \
            "CAMModule: memory bank not set. Call set_memory_bank() first."

        # ── 1. Retrieve stored memories ──────────────────────────────────
        memories = self._bank.get_last_k(
            layer_idx=self.layer_idx,
            k=self.config.window_size,
            side=side
        )

        # ── 2. Inject: compute memory bias ───────────────────────────────
        memory_bias, metrics = self.injector(
            Q=Q, K=K, V=V,
            memories=memories,
            layer_depth=layer_depth,
            attention_mask=attention_mask
        )

        # ── 3. Compress current state ────────────────────────────────────
        if self.config.use_cam_compression:
            # Decoder: learned importance + full compression
            importance = self.importance_net(
                Q=Q, K=K, V=V, A=A,
                layer_depth=layer_depth,
                attention_mask=attention_mask
            )
        else:
            # Encoder: uniform importance (equal weight to all tokens)
            # Compressor still runs but with uniform weights
            # This keeps encoder path consistent without learned overhead
            batch, seq, _ = K.shape
            importance = torch.ones(batch, seq, device=K.device) / seq

        compressed = self.compressor(
            K=K, V=V,
            importance=importance,
            step=self._step
        )

        # ── 4. Store compressed memory ───────────────────────────────────
        self._bank.store(
            layer_idx=self.layer_idx,
            compressed=compressed,
            side=side
        )

        self._step += 1
        return memory_bias, metrics
