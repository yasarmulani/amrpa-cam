"""
AMRPAModel — Universal Entry Point
------------------------------------
Single class that routes to correct adapter based on config.arch.

Usage:
    from amrpa import AMRPAModel, AMRPAConfig

    # Encoder
    config = AMRPAConfig.from_hf_config(model.config, arch='encoder')
    model, state = AMRPAModel.wrap(model, config)

    # Decoder
    config = AMRPAConfig.from_hf_config(model.config, arch='decoder')
    model, state = AMRPAModel.wrap(model, config)

    # In training loop
    state.reset()          # call before every forward pass
    outputs = model(...)
    metrics = state.get_metrics()
"""

import torch.nn as nn
from typing import Tuple, Union

from .config import AMRPAConfig
from .adapters.encoder import (
    apply_amrpa_to_encoder, EncoderAMRPAState
)
from .adapters.decoder import (
    apply_amrpa_to_decoder, DecoderAMRPAState
)

AMRPAState = Union[EncoderAMRPAState, DecoderAMRPAState]


class AMRPAModel:
    """
    Universal AMRPA wrapper.

    Does not subclass nn.Module — patches the existing model in-place.
    The user's training loop stays completely unchanged.
    """

    @staticmethod
    def wrap(
        model: nn.Module,
        config: AMRPAConfig
    ) -> Tuple[nn.Module, AMRPAState]:
        """
        Apply AMRPA to any supported model.

        Args:
            model:  Any HuggingFace model
            config: AMRPAConfig specifying architecture and hyperparameters

        Returns:
            model:  patched model (same object, modified in-place)
            state:  state object — call state.reset() before each forward pass

        Example:
            # Encoder
            config = AMRPAConfig.for_encoder(
                d_model=model.config.hidden_size,
                n_heads=model.config.num_attention_heads
            )
            model, state = AMRPAModel.wrap(model, config)

            for batch in dataloader:
                state.reset()
                outputs = model(**batch)
                metrics = state.get_metrics()
        """
        if config.arch == 'encoder':
            return apply_amrpa_to_encoder(model, config)
        elif config.arch == 'decoder':
            return apply_amrpa_to_decoder(model, config)
        elif config.arch == 'encoder_decoder':
            raise NotImplementedError(
                "encoder_decoder support coming in next release. "
                "Use arch='encoder' or arch='decoder' for now."
            )
        else:
            raise ValueError(f"Unknown arch: {config.arch}")

    @staticmethod
    def reset(state: AMRPAState) -> None:
        """Reset AMRPA memory. Call before every forward pass."""
        state.reset()

    @staticmethod
    def get_metrics(state: AMRPAState) -> dict:
        """Get mechanism metrics from last forward pass."""
        return state.get_metrics()
