from .encoder import apply_amrpa_to_encoder, reset_encoder, EncoderAMRPAState
from .decoder import apply_amrpa_to_decoder, reset_decoder, DecoderAMRPAState
from .universal import apply_amrpa_universal, UniversalAMRPAState
from .registry import MODEL_REGISTRY, get_model_info, is_supported, get_arch

__all__ = [
    # Universal (recommended)
    'apply_amrpa_universal',
    'UniversalAMRPAState',
    # Registry
    'MODEL_REGISTRY',
    'get_model_info',
    'is_supported',
    'get_arch',
    # Legacy specific adapters
    'apply_amrpa_to_encoder',
    'reset_encoder',
    'EncoderAMRPAState',
    'apply_amrpa_to_decoder',
    'reset_decoder',
    'DecoderAMRPAState',
]
