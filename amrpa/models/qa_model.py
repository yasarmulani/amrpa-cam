"""
AMRPA QA Model
---------------
RoBERTa + AMRPA for extractive span-based QA.

Direct port of RoBERTa_AMRPA_QA from original script,
now using the clean library components.

This is the model that achieved:
    HotpotQA:      +22.9% F1
    2WikiMultiHop: +10.0% F1
    MuSiQue:       +19.8% F1

Usage:
    from amrpa.models.qa_model import AMRPAForQA

    model = AMRPAForQA(config)
    start_logits, end_logits = model(input_ids, attention_mask)
    start_logits, end_logits, metrics = model(input_ids, attention_mask,
                                               return_metrics=True)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict

from ..config import AMRPAConfig
from ..adapters.encoder import apply_amrpa_to_encoder, EncoderAMRPAState


class AMRPAForQA(nn.Module):
    """
    RoBERTa (or any encoder) + AMRPA for extractive span QA.

    Predicts start and end token positions of answer span.
    Compatible with HotpotQA preprocessed .pt files from original script.

    Args:
        config: AMRPAConfig with arch='encoder'
        model_name: HuggingFace model name (default: 'roberta-base')
    """

    def __init__(
        self,
        config: AMRPAConfig,
        model_name: str = 'roberta-base'
    ):
        super().__init__()
        assert config.arch == 'encoder', \
            "AMRPAForQA requires arch='encoder'"

        from transformers import RobertaModel
        print(f"\nBuilding AMRPAForQA ({model_name})...")
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.config  = config

        # Apply AMRPA to encoder layers
        self.roberta, self._amrpa_state = apply_amrpa_to_encoder(
            self.roberta, config
        )

        # QA head: predict start and end positions
        self.qa_outputs = nn.Linear(config.d_model, 2)
        print(f"✓ AMRPAForQA ready")

    @property
    def amrpa_state(self) -> EncoderAMRPAState:
        return self._amrpa_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_metrics: bool = False
    ):
        """
        Forward pass.

        Args:
            input_ids:      (batch, seq)
            attention_mask: (batch, seq)
            return_metrics: if True, returns AMRPA mechanism metrics

        Returns:
            start_logits: (batch, seq)
            end_logits:   (batch, seq)
            metrics_list: list of dicts (only if return_metrics=True)
        """
        # Reset AMRPA memory for each forward pass
        self._amrpa_state.reset()

        outputs        = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False
        )
        sequence_output = outputs.last_hidden_state  # (batch, seq, d_model)

        logits       = self.qa_outputs(sequence_output)
        start_logits = logits[:, :, 0]               # (batch, seq)
        end_logits   = logits[:, :, 1]               # (batch, seq)

        if return_metrics:
            metrics = self._amrpa_state.get_metrics()
            return start_logits, end_logits, metrics

        return start_logits, end_logits
