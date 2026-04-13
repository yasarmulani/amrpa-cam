"""
AMRPAModel — Universal Entry Point
------------------------------------
Single class that wraps any supported model with AMRPA.
Uses registry to detect architecture automatically.

Usage:
    from amrpa import AMRPAModel, AMRPAConfig

    # Works for ANY supported model
    config = AMRPAConfig.from_hf_config(model.config, arch='encoder')
    model, state = AMRPAModel.wrap(model, config)

    for batch in dataloader:
        state.reset()
        outputs = model(**batch)
        metrics = state.get_metrics()
"""

import torch.nn as nn
from typing import Tuple, Union

from .config import AMRPAConfig
from .adapters.universal import apply_amrpa_universal, UniversalAMRPAState
from .adapters.registry import is_supported, get_arch


class AMRPAModel:
    """
    Universal AMRPA wrapper.
    Patches any supported model in-place.
    Training loop stays completely unchanged.
    """

    @staticmethod
    def wrap(
        model: nn.Module,
        config: AMRPAConfig
    ) -> Tuple[nn.Module, UniversalAMRPAState]:
        """
        Apply AMRPA to any supported model.

        Args:
            model:  Any HuggingFace model (encoder, decoder, enc-dec)
            config: AMRPAConfig — use from_hf_config() for auto-detection

        Returns:
            model:  patched model
            state:  call state.reset() before every forward pass

        Supported models:
            Encoder:         roberta, bert, deberta
            Decoder:         gpt2, gpt_neo, phi, llama, mistral
            Encoder-Decoder: t5, bart

        Example — Encoder:
            config = AMRPAConfig.from_hf_config(model.config, arch='encoder')
            model, state = AMRPAModel.wrap(model, config)

        Example — Decoder:
            config = AMRPAConfig.from_hf_config(model.config, arch='decoder')
            model, state = AMRPAModel.wrap(model, config)

        Example — Any model auto-detect:
            model_type = model.config.model_type
            arch = get_arch(model_type)
            config = AMRPAConfig.from_hf_config(model.config, arch=arch)
            model, state = AMRPAModel.wrap(model, config)
        """
        return apply_amrpa_universal(model, config)

    @staticmethod
    def reset(state: UniversalAMRPAState) -> None:
        """Reset memory. Call before every forward pass."""
        state.reset()

    @staticmethod
    def get_metrics(state: UniversalAMRPAState) -> dict:
        """Get mechanism metrics from last forward pass."""
        return state.get_metrics()

    @staticmethod
    def is_supported(model) -> bool:
        """Check if model is supported."""
        model_type = model.config.model_type
        return is_supported(model_type)

    @staticmethod
    def memory_summary(state: UniversalAMRPAState) -> dict:
        """Memory usage summary."""
        if state.cam_bank:
            return state.cam_bank.summary()
        return {'arch': state.arch, 'cam': 'not used'}
