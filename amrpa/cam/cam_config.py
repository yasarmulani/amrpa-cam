"""
CAM Configuration
------------------
Single source of truth for all CAM hyperparameters.
Designed to integrate cleanly into the AMRPA + CAM unified library.

Architecture modes:
    'encoder'          - bidirectional, no causal mask, raw attention storage
    'decoder'          - causal, CAM compression, sliding window
    'encoder_decoder'  - both sides, isolated memory banks

Usage:
    # From scratch — user provides dimensions from their model
    config = CAMConfig(
        d_model = model.config.hidden_size,
        n_heads = model.config.num_attention_heads,
        arch    = 'encoder'
    )

    # Architecture presets — user still provides model dimensions
    config = CAMConfig.for_encoder(d_model=768, n_heads=12)
    config = CAMConfig.for_decoder(d_model=1024, n_heads=16)
    config = CAMConfig.for_encoder_decoder(d_model=768, n_heads=12)

    # Auto-detect from HuggingFace config object
    config = CAMConfig.from_hf_config(model.config, arch='encoder')
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CAMConfig:
    """
    All CAM hyperparameters in one place.

    Core dimensions (must match base model — set from model.config):
        d_model:        full hidden size of base model
        n_heads:        number of attention heads in base model
        d_k:            per-head key dimension
                        auto-computed as d_model // n_heads if not set

    Architecture:
        arch:           'encoder', 'decoder', or 'encoder_decoder'
        n_amrpa_layers: how many layers from the end to apply CAM-AMRPA

    Memory:
        window_size:    sliding window W — max past steps stored per layer
                        encoder: window over layers
                        decoder: window over generation steps
        gamma:          temporal decay factor (0 < gamma < 1)
                        higher = trust past more, lower = forget faster

    Compression (decoder only, ignored for encoder):
        proj_rank:      rank of learnable projection replacing SVD
                        lower = more compression, less fidelity
                        recommended: 16-32 for seq<=512, 8 for longer

    Importance network:
        importance_hidden:      hidden dim of LearnedImportance MLP
        use_learned_importance: True = learned, False = heuristic (ablation)

    Alpha selection:
        alpha_temperature:  softmax temperature for past-step selection
                            lower = more selective, higher = more uniform

    Injection:
        gate_gamma_init:    initial value of learnable gate scaling parameter
        gate_bias_init:     initial value of learnable gate bias

    Training:
        dropout:            dropout rate inside CAM MLPs
    """

    # Core dimensions
    d_model: int = 768
    n_heads: int = 12
    d_k: Optional[int] = None      # auto-computed if None

    # Architecture
    arch: str = 'encoder'
    n_amrpa_layers: int = 4

    # Memory
    window_size: int = 8
    gamma: float = 0.9

    # Compression
    proj_rank: int = 16

    # Importance network
    importance_hidden: int = 64
    use_learned_importance: bool = True

    # Alpha selection
    alpha_temperature: float = 0.25

    # Injection
    gate_gamma_init: float = 2.0
    gate_bias_init: float = -0.25

    # Training
    dropout: float = 0.1

    def __post_init__(self):
        if self.d_k is None:
            self.d_k = self.d_model // self.n_heads

        assert self.arch in ('encoder', 'decoder', 'encoder_decoder'), \
            f"arch must be 'encoder', 'decoder', or 'encoder_decoder', got '{self.arch}'"
        assert 0.0 < self.gamma < 1.0, \
            f"gamma must be in (0,1), got {self.gamma}"
        assert self.d_model % self.n_heads == 0, \
            f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
        assert self.proj_rank > 0, \
            f"proj_rank must be positive, got {self.proj_rank}"
        assert self.window_size > 0, \
            f"window_size must be positive, got {self.window_size}"

    @property
    def causal(self) -> bool:
        """Decoder needs causal masking. Encoder does not."""
        return self.arch == 'decoder'

    @property
    def use_cam_compression(self) -> bool:
        """CAM compression needed for decoder. Encoder has fixed sequence length."""
        return self.arch in ('decoder', 'encoder_decoder')

    @classmethod
    def for_encoder(cls, d_model: int, n_heads: int, **kwargs) -> 'CAMConfig':
        """
        Preset for any encoder-only architecture.

        Args:
            d_model: model.config.hidden_size
            n_heads: model.config.num_attention_heads

        Example:
            config = CAMConfig.for_encoder(
                d_model=model.config.hidden_size,
                n_heads=model.config.num_attention_heads
            )
        """
        return cls(d_model=d_model, n_heads=n_heads, arch='encoder', **kwargs)

    @classmethod
    def for_decoder(cls, d_model: int, n_heads: int, **kwargs) -> 'CAMConfig':
        """
        Preset for any decoder-only architecture.

        Args:
            d_model: model.config.hidden_size (or n_embd for GPT-style)
            n_heads: model.config.num_attention_heads (or n_head)

        Example:
            config = CAMConfig.for_decoder(
                d_model=model.config.n_embd,
                n_heads=model.config.n_head
            )
        """
        return cls(d_model=d_model, n_heads=n_heads, arch='decoder', **kwargs)

    @classmethod
    def for_encoder_decoder(cls, d_model: int, n_heads: int,
                             **kwargs) -> 'CAMConfig':
        """
        Preset for any encoder-decoder architecture.

        Args:
            d_model: model.config.hidden_size
            n_heads: model.config.num_attention_heads

        Example:
            config = CAMConfig.for_encoder_decoder(
                d_model=model.config.hidden_size,
                n_heads=model.config.num_attention_heads
            )
        """
        return cls(d_model=d_model, n_heads=n_heads,
                   arch='encoder_decoder', **kwargs)

    @classmethod
    def from_hf_config(cls, hf_config, arch: str, **kwargs) -> 'CAMConfig':
        """
        Auto-detect dimensions from any HuggingFace model config object.
        Handles different attribute naming conventions across model families.

        Args:
            hf_config:  model.config from any HuggingFace model
            arch:       'encoder', 'decoder', or 'encoder_decoder'

        Example:
            cam_config = CAMConfig.from_hf_config(model.config, arch='encoder')
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

        return cls(d_model=d_model, n_heads=n_heads, arch=arch, **kwargs)

    def __repr__(self) -> str:
        return (
            f"CAMConfig("
            f"arch={self.arch}, "
            f"d_model={self.d_model}, "
            f"n_heads={self.n_heads}, "
            f"d_k={self.d_k}, "
            f"n_amrpa_layers={self.n_amrpa_layers}, "
            f"window_size={self.window_size}, "
            f"gamma={self.gamma}, "
            f"proj_rank={self.proj_rank})"
        )
