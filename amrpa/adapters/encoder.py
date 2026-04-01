"""
AMRPA Encoder Adapter
-----------------------
Patches the last N attention layers of any encoder model with AMRPA.

Ported and refactored from AMRPA_SelfAttentionWrapper in original script.
Now uses AMRPACore for the mechanism and AMRPAConfig for all hyperparameters.

Supports: RoBERTa, BERT, DeBERTa (any model with encoder.layer[i].attention.self)

Usage:
    from amrpa.adapters.encoder import apply_amrpa_to_encoder, reset_encoder

    model, state = apply_amrpa_to_encoder(roberta_model, config)

    # In training loop — call once per batch
    reset_encoder(state)
    outputs = model(input_ids, attention_mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Dict

from ..config import AMRPAConfig
from ..core import AMRPACore


# ── Layer detection ───────────────────────────────────────────────────────────

def _get_encoder_layers(model: nn.Module) -> List[nn.Module]:
    """Get encoder transformer blocks from any supported encoder model."""
    # Handle task-specific heads (e.g. RobertaForQuestionAnswering)
    base = (
        getattr(model, 'roberta', None) or
        getattr(model, 'bert', None) or
        getattr(model, 'deberta', None) or
        model
    )
    if hasattr(base, 'encoder') and hasattr(base.encoder, 'layer'):
        return list(base.encoder.layer)
    raise ValueError(
        "Cannot find encoder layers. "
        "Expected model.encoder.layer or model.roberta/bert.encoder.layer"
    )


def _get_self_attn(block: nn.Module) -> nn.Module:
    if hasattr(block, 'attention') and hasattr(block.attention, 'self'):
        return block.attention.self
    raise ValueError("Cannot find self-attention in encoder block.")


def _set_self_attn(block: nn.Module, new_attn: nn.Module):
    if hasattr(block, 'attention') and hasattr(block.attention, 'self'):
        block.attention.self = new_attn
    else:
        raise ValueError("Cannot set self-attention in encoder block.")


# ── Per-layer wrapper ─────────────────────────────────────────────────────────

class EncoderAMRPALayer(nn.Module):
    """
    Replaces one encoder self-attention layer with AMRPA.

    Preserves original QKV projections from the base model.
    Adds AMRPA memory bias on top of base attention scores.

    Attention history is shared across all AMRPA layers via a list reference
    set before each forward pass (same pattern as original script).
    """

    def __init__(
        self,
        original_self_attn: nn.Module,
        amrpa_core: AMRPACore,
        layer_idx: int,         # absolute layer index in model
        amrpa_layer_idx: int,   # index within AMRPA layers (0-based)
        n_amrpa_layers: int,
        config: AMRPAConfig
    ):
        super().__init__()
        self.original        = original_self_attn
        self.amrpa           = amrpa_core
        self.layer_idx       = layer_idx
        self.amrpa_layer_idx = amrpa_layer_idx
        self.n_amrpa_layers  = n_amrpa_layers
        self.config          = config

        self.d_model  = original_self_attn.query.in_features
        self.d_k      = self.d_model  # encoder uses full d_model, not per-head d_k
        self.n_heads  = config.n_heads
        self.head_dim = self.d_model // self.n_heads

        # Shared attention history — set by model forward() before each pass
        self.attention_history: Optional[List] = None
        self.last_metrics: Dict = {}

    def reset_metrics(self):
        self.last_metrics = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        past_key_values=None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple:
        batch_size, seq_len, _ = hidden_states.shape

        # QKV from original layer
        Q = self.original.query(hidden_states)  # (batch, seq, d_model)
        K = self.original.key(hidden_states)
        V = self.original.value(hidden_states)

        # Base attention scores (full d_model, not per-head)
        base_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply HuggingFace attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attention_mask = attention_mask.squeeze(1)
            if attention_mask.dim() == 3 and attention_mask.size(1) == 1:
                attention_mask = attention_mask.expand(-1, seq_len, -1)
            base_scores = base_scores + attention_mask

        # AMRPA memory bias
        # relative_layer_idx is 1-based within AMRPA layers
        relative_idx = self.amrpa_layer_idx + 1

        memory_bias, metrics = self.amrpa(
            Q=Q, K=K, V=V,
            attention_history=self.attention_history or [],
            relative_layer_idx=relative_idx,
            causal_mask=None   # encoder: no causal mask
        )

        final_scores    = base_scores + memory_bias
        attention_probs = F.softmax(final_scores, dim=-1)
        context_layer   = torch.matmul(attention_probs, V)

        # Append to shared history
        if self.attention_history is not None:
            self.attention_history.append(attention_probs.detach())

        self.last_metrics = {k: v.detach().cpu() for k, v in metrics.items()}

        if output_attentions:
            return (context_layer, attention_probs)
        return (context_layer,)


# ── Encoder state (returned to user) ─────────────────────────────────────────

class EncoderAMRPAState:
    """Holds references needed for memory reset between batches."""
    def __init__(self, layers: List[EncoderAMRPALayer]):
        self.layers = layers

    def reset(self):
        """Call at start of every forward pass / batch."""
        for layer in self.layers:
            layer.attention_history = []
            layer.reset_metrics()

    def get_metrics(self) -> Dict:
        """Aggregate metrics from all AMRPA layers."""
        all_metrics = [l.last_metrics for l in self.layers if l.last_metrics]
        if not all_metrics:
            return {}
        keys = set().union(*[m.keys() for m in all_metrics])
        merged = {}
        for k in keys:
            vals = [m[k] for m in all_metrics if k in m]
            if vals:
                merged[k] = torch.stack(vals).mean(0)
        return merged


# ── Main apply function ───────────────────────────────────────────────────────

def apply_amrpa_to_encoder(
    model: nn.Module,
    config: AMRPAConfig
) -> Tuple[nn.Module, EncoderAMRPAState]:
    """
    Apply AMRPA to last config.n_amrpa_layers of an encoder model.

    Args:
        model:  Any HuggingFace encoder model (RoBERTa, BERT, DeBERTa)
                or task-specific head model (e.g. RobertaForQuestionAnswering)
        config: AMRPAConfig with arch='encoder'

    Returns:
        model:  patched model (in-place)
        state:  EncoderAMRPAState — call state.reset() before each forward pass

    Example:
        config = AMRPAConfig.from_hf_config(model.config, arch='encoder')
        model, state = apply_amrpa_to_encoder(model, config)

        for batch in dataloader:
            state.reset()
            outputs = model(**batch)
            metrics = state.get_metrics()
    """
    assert config.arch == 'encoder', \
        f"Expected arch='encoder', got '{config.arch}'"

    all_layers  = _get_encoder_layers(model)
    total_layers = len(all_layers)
    start_layer  = total_layers - config.n_amrpa_layers

    # Freeze components as in original script
    base = (
        getattr(model, 'roberta', None) or
        getattr(model, 'bert', None) or
        model
    )
    if config.freeze_embeddings and hasattr(base, 'embeddings'):
        for param in base.embeddings.parameters():
            param.requires_grad = False

    freeze_until = config.freeze_layers
    if freeze_until == -1:
        freeze_until = max(0, total_layers - 8)

    for i, layer in enumerate(all_layers):
        requires_grad = (i >= freeze_until)
        for param in layer.parameters():
            param.requires_grad = requires_grad

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    # Encoder Q,K,V projections output d_model (768) not d_k (64)
    # Original script used self.d_k = self.d_model for same reason
    import copy as _copy
    encoder_core_config = _copy.copy(config)
    encoder_core_config.d_k = config.d_model

    amrpa_layers = []
    print(f"\nApplying AMRPA to encoder (last {config.n_amrpa_layers} layers):")

    for i in range(start_layer, total_layers):
        amrpa_idx        = i - start_layer
        block            = all_layers[i]
        original_self    = _get_self_attn(block)

        amrpa_core = AMRPACore(encoder_core_config).to(device)

        amrpa_layer = EncoderAMRPALayer(
            original_self_attn=original_self,
            amrpa_core=amrpa_core,
            layer_idx=i,
            amrpa_layer_idx=amrpa_idx,
            n_amrpa_layers=config.n_amrpa_layers,
            config=config
        ).to(device)

        _set_self_attn(block, amrpa_layer)
        amrpa_layers.append(amrpa_layer)

        n_params = sum(p.numel() for p in amrpa_core.parameters())
        print(f"  ✓ Layer {i} (AMRPA layer {amrpa_idx+1}): patched "
              f"[{n_params:,} AMRPA params]")

    total_amrpa = sum(
        p.numel() for l in amrpa_layers for p in l.amrpa.parameters()
    )
    print(f"\n  Total AMRPA params added: {total_amrpa:,}")

    state = EncoderAMRPAState(amrpa_layers)
    return model, state


def reset_encoder(state: EncoderAMRPAState):
    """Convenience function. Call before every forward pass."""
    state.reset()
