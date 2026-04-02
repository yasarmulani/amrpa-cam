"""
AMRPA + CAM Library
---------------------
Adaptive Multi-layer Recursive Preconditioned Attention
with Compressed Attention Memory.

Quick start:

    # Encoder (span extraction, classification)
    from amrpa import AMRPAModel, AMRPAConfig

    config = AMRPAConfig.from_hf_config(model.config, arch='encoder')
    model, state = AMRPAModel.wrap(model, config)

    for batch in dataloader:
        state.reset()
        outputs = model(**batch)

    # Decoder (generation)
    config = AMRPAConfig.from_hf_config(model.config, arch='decoder')
    model, state = AMRPAModel.wrap(model, config)

    # Ready-to-use QA model (RoBERTa + AMRPA)
    from amrpa import AMRPAForQA

    config = AMRPAConfig.for_encoder(d_model=768, n_heads=12)
    model  = AMRPAForQA(config, model_name='roberta-base')
"""

from .config import AMRPAConfig, CAMConfig
from .model import AMRPAModel
from .models.qa_model import AMRPAForQA
from .adapters.encoder import apply_amrpa_to_encoder, reset_encoder
from .adapters.decoder import apply_amrpa_to_decoder, reset_decoder
from .utils import print_flops_summary, plot_flops_comparison
from .training import (
    PreprocessedQADataset,
    train_epoch,
    evaluate,
    build_optimizer,
    compute_exact_match,
    compute_f1,
    compute_rouge_l,
    normalize_answer,
    get_best_span
)

__version__ = '1.0.0'

__all__ = [
    # Config
    'AMRPAConfig',
    'CAMConfig',
    # Main entry point
    'AMRPAModel',
    # Ready-to-use models
    'AMRPAForQA',
    # Adapters
    'apply_amrpa_to_encoder',
    'reset_encoder',
    'apply_amrpa_to_decoder',
    'reset_decoder',
    # Training utilities
    'PreprocessedQADataset',
    'train_epoch',
    'evaluate',
    'build_optimizer',
    'compute_exact_match',
    'compute_f1',
    'compute_rouge_l',
    'normalize_answer',
    'get_best_span',
    'print_flops_summary',
    'plot_flops_comparison',
    'compute_rouge_l',
]
