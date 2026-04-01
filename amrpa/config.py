"""
AMRPA + CAM Unified Configuration
------------------------------------
Single source of truth for all hyperparameters.

Two sections:
    AMRPAConfig  — AMRPA mechanism hyperparameters
    CAMConfig    — CAM compression hyperparameters (decoder only)

Usage:
    # Encoder (RoBERTa-style)
    config = AMRPAConfig.for_encoder(
        d_model=model.config.hidden_size,
        n_heads=model.config.num_attention_heads
    )

    # Decoder (GPT2-style)
    config = AMRPAConfig.for_decoder(
        d_model=model.config.hidden_size,
        n_heads=model.config.num_attention_heads
    )

    # Auto-detect from HuggingFace config
    config = AMRPAConfig.from_hf_config(model.config, arch='encoder')
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CAMConfig:
    """
    CAM compression hyperparameters.
    Only used when arch='decoder' or arch='encoder_decoder'.
    Ignored for encoder-only models.
    """
    use_cam: bool = True                # False = disable CAM entirely
    proj_rank: int = 16                 # learnable projection rank (replaces SVD)
    window_size: int = 8                # sliding window for stored memories
    importance_hidden: int = 64         # hidden dim of LearnedImportance MLP
    use_learned_importance: bool = True # False = heuristic (ablation baseline)


@dataclass
class AMRPAConfig:
    """
    Full AMRPA + CAM configuration.

    Model dimensions (fill from your model's config):
        d_model:        hidden size of base model
        n_heads:        number of attention heads
        d_k:            per-head dim (auto = d_model // n_heads)

    Architecture:
        arch:           'encoder' | 'decoder' | 'encoder_decoder'
        n_amrpa_layers: how many layers from end to apply AMRPA

    AMRPA mechanism:
        gamma:              temporal decay factor (0 < γ < 1)
        epsilon:            noise added to decayed patterns (stability)
        alpha_temperature:  softmax temperature for layer selection
        gate_gamma_init:    initial gate scale parameter
        gate_bias_init:     initial gate bias parameter
        d_mlp:              hidden dim of alpha MLP

    Training:
        dropout:            dropout in AMRPA MLPs
        freeze_embeddings:  freeze base model embeddings
        freeze_layers:      how many base layers to freeze from bottom
                            -1 = auto (freeze all but last 8)

    Regularization:
        diversity_weight:   weight for alpha diversity loss
        gate_reg_weight:    weight for gate regularization loss
        label_smoothing:    label smoothing for QA loss

    CAM (decoder only):
        cam: CAMConfig object
    """

    # ── Model dimensions ──────────────────────────────────────────────────
    d_model: int = 768
    n_heads: int = 12
    d_k: Optional[int] = None          # auto-computed if None

    # ── Architecture ──────────────────────────────────────────────────────
    arch: str = 'encoder'
    n_amrpa_layers: int = 4

    # ── AMRPA mechanism ───────────────────────────────────────────────────
    gamma: float = 0.9
    epsilon: float = 0.001
    alpha_temperature: float = 0.25
    gate_gamma_init: float = 2.0
    gate_bias_init: float = -0.25
    d_mlp: int = 384

    # ── Training ──────────────────────────────────────────────────────────
    dropout: float = 0.2
    freeze_embeddings: bool = True
    freeze_layers: int = -1             # -1 = auto

    # ── Regularization ────────────────────────────────────────────────────
    diversity_weight: float = 0.005
    gate_reg_weight: float = 0.05
    label_smoothing: float = 0.05

    # ── CAM config (decoder only) ─────────────────────────────────────────
    cam: CAMConfig = field(default_factory=CAMConfig)

    def __post_init__(self):
        if self.d_k is None:
            self.d_k = self.d_model // self.n_heads

        assert self.arch in ('encoder', 'decoder', 'encoder_decoder'), \
            f"arch must be 'encoder', 'decoder', or 'encoder_decoder', got '{self.arch}'"
        assert 0.0 < self.gamma < 1.0, \
            f"gamma must be in (0,1), got {self.gamma}"
        assert self.d_model % self.n_heads == 0, \
            f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"

    @property
    def causal(self) -> bool:
        return self.arch == 'decoder'

    @property
    def use_cam(self) -> bool:
        return self.arch in ('decoder', 'encoder_decoder') and self.cam.use_cam

    @classmethod
    def for_encoder(cls, d_model: int, n_heads: int, **kwargs) -> 'AMRPAConfig':
        """
        Preset for any encoder-only model.
        CAM is disabled (not needed for fixed-length sequences).

        Example:
            config = AMRPAConfig.for_encoder(
                d_model=model.config.hidden_size,
                n_heads=model.config.num_attention_heads
            )
        """
        cam = CAMConfig(use_cam=False)
        return cls(d_model=d_model, n_heads=n_heads, arch='encoder',
                   cam=cam, **kwargs)

    @classmethod
    def for_decoder(cls, d_model: int, n_heads: int, **kwargs) -> 'AMRPAConfig':
        """
        Preset for any decoder-only model.
        CAM is enabled by default.

        Example:
            config = AMRPAConfig.for_decoder(
                d_model=model.config.hidden_size,
                n_heads=model.config.num_attention_heads
            )
        """
        cam = CAMConfig(use_cam=True)
        return cls(d_model=d_model, n_heads=n_heads, arch='decoder',
                   cam=cam, **kwargs)

    @classmethod
    def for_encoder_decoder(cls, d_model: int, n_heads: int,
                             **kwargs) -> 'AMRPAConfig':
        """
        Preset for encoder-decoder models (T5, BART etc).
        CAM enabled on decoder side only.
        """
        cam = CAMConfig(use_cam=True)
        return cls(d_model=d_model, n_heads=n_heads, arch='encoder_decoder',
                   cam=cam, **kwargs)

    @classmethod
    def from_hf_config(cls, hf_config, arch: str, **kwargs) -> 'AMRPAConfig':
        """
        Auto-detect dimensions from any HuggingFace model config.

        Example:
            config = AMRPAConfig.from_hf_config(model.config, arch='encoder')
        """
        d_model = (
            getattr(hf_config, 'hidden_size', None) or
            getattr(hf_config, 'n_embd', None) or
            getattr(hf_config, 'd_model', None)
        )
        n_heads = (
            getattr(hf_config, 'num_attention_heads', None) or
            getattr(hf_config, 'n_head', None) or
            getattr(hf_config, 'num_heads', None)
        )
        assert d_model is not None, \
            "Cannot auto-detect d_model. Pass d_model= explicitly."
        assert n_heads is not None, \
            "Cannot auto-detect n_heads. Pass n_heads= explicitly."

        if arch == 'encoder':
            return cls.for_encoder(d_model=d_model, n_heads=n_heads, **kwargs)
        elif arch == 'decoder':
            return cls.for_decoder(d_model=d_model, n_heads=n_heads, **kwargs)
        else:
            return cls.for_encoder_decoder(d_model=d_model, n_heads=n_heads,
                                           **kwargs)

    def __repr__(self) -> str:
        return (
            f"AMRPAConfig("
            f"arch={self.arch}, "
            f"d_model={self.d_model}, "
            f"n_heads={self.n_heads}, "
            f"d_k={self.d_k}, "
            f"n_amrpa_layers={self.n_amrpa_layers}, "
            f"gamma={self.gamma}, "
            f"use_cam={self.use_cam})"
        )
