"""
AMRPA Decoder Adapter
-----------------------
Patches the last N attention layers of any decoder model with AMRPA + CAM.

Decoder differences from encoder:
    1. Causal mask — token i cannot use memory about token j > i
    2. CAM compression — sequence length grows during generation,
       so we compress attention into fixed-size memory
    3. GPT2 uses c_attn (combined QKV) instead of separate projections

Supports: GPT2, GPT2-medium (any GPT2-style model)

Usage:
    from amrpa.adapters.decoder import apply_amrpa_to_decoder, reset_decoder

    model, state = apply_amrpa_to_decoder(gpt2_model, config)

    for batch in dataloader:
        reset_decoder(state)
        outputs = model(**batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Dict

from ..config import AMRPAConfig
from ..core import AMRPACore
from ..cam import CAMModule, CAMMemoryBankSet
from ..cam.cam_config import CAMConfig as InternalCAMConfig


# ── Layer detection ───────────────────────────────────────────────────────────

def _get_decoder_layers(model: nn.Module) -> List[nn.Module]:
    """Get transformer blocks from GPT2-style decoder model."""
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)
    elif hasattr(model, 'h'):
        return list(model.h)
    raise ValueError(
        "Cannot find decoder layers. "
        "Expected model.transformer.h or model.h (GPT2-style)"
    )


# ── Per-layer wrapper ─────────────────────────────────────────────────────────

class DecoderAMRPALayer(nn.Module):
    """
    Replaces one GPT2 decoder attention layer with AMRPA + CAM.

    AMRPA provides cross-layer reasoning memory.
    CAM compresses that memory for bounded storage during generation.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        amrpa_core: AMRPACore,
        cam_module: CAMModule,
        layer_idx: int,
        amrpa_layer_idx: int,
        n_amrpa_layers: int,
        config: AMRPAConfig
    ):
        super().__init__()
        self.original        = original_attn
        self.amrpa           = amrpa_core
        self.cam             = cam_module
        self.layer_idx       = layer_idx
        self.amrpa_layer_idx = amrpa_layer_idx
        self.n_amrpa_layers  = n_amrpa_layers
        self.config          = config

        self.embed_dim = original_attn.embed_dim
        self.num_heads = original_attn.num_heads
        self.head_dim  = self.embed_dim // self.num_heads
        self.scale     = 1.0 / math.sqrt(self.head_dim)

        self.last_metrics: Dict = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs
    ):
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # GPT2 combined QKV projection
        qkv = self.original.c_attn(hidden_states)
        Q, K, V = qkv.split(self.embed_dim, dim=2)

        if layer_past is not None:
            past_K, past_V = layer_past
            K = torch.cat([past_K, K], dim=1)
            V = torch.cat([past_V, V], dim=1)

        present   = (K, V) if use_cache else None
        kv_seq_len = K.shape[1]

        # Base attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mask_exp = attention_mask.expand(
                    batch_size, self.num_heads, seq_len, kv_seq_len
                ).reshape(batch_size, seq_len, kv_seq_len)
                attn_scores = attn_scores + mask_exp

        # Build causal mask for AMRPA memory bias
        causal_mask = torch.triu(
            torch.full((seq_len, kv_seq_len), float('-inf'), device=device),
            diagonal=1
        )

        # AMRPA: get memory bias from CAM bank
        # CAM uses mean-head Q, K, V for memory computation
        Q_cam = Q[:, :, :self.config.d_k]
        K_cam = K[:, :, :self.config.d_k]
        V_cam = V[:, :, :self.config.d_k]

        with torch.no_grad():
            base_A_cam = F.softmax(
                torch.matmul(Q_cam, K_cam.transpose(-2, -1)) * self.scale,
                dim=-1
            )

        layer_depth = self.amrpa_layer_idx / max(self.n_amrpa_layers, 1)
        cam_mask    = torch.ones(batch_size, seq_len, device=device)

        cam_bias, cam_metrics = self.cam(
            Q=Q_cam, K=K_cam, V=V_cam,
            A=base_A_cam,
            layer_depth=layer_depth,
            attention_mask=cam_mask
        )

        # Add memory bias (causal mask already applied inside CAM for decoder)
        final_scores = attn_scores + cam_bias

        final_attn   = F.softmax(final_scores, dim=-1)
        context      = torch.matmul(final_attn, V)

        output = self.original.c_proj(context)
        output = self.original.resid_dropout(output)

        self.last_metrics = {k: v.detach() for k, v in cam_metrics.items()}

        outputs = (output, present)
        if output_attentions:
            outputs += (final_attn,)
        return outputs


# ── Decoder state ─────────────────────────────────────────────────────────────

class DecoderAMRPAState:
    """Holds references needed for memory reset between sequences."""

    def __init__(
        self,
        layers: List[DecoderAMRPALayer],
        cam_bank: CAMMemoryBankSet
    ):
        self.layers   = layers
        self.cam_bank = cam_bank

    def reset(self):
        """Call at start of every new sequence."""
        self.cam_bank.reset()
        for layer in self.layers:
            layer.cam.reset()

    def get_metrics(self) -> Dict:
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

def apply_amrpa_to_decoder(
    model: nn.Module,
    config: AMRPAConfig
) -> Tuple[nn.Module, DecoderAMRPAState]:
    """
    Apply AMRPA + CAM to last config.n_amrpa_layers of a decoder model.

    Args:
        model:  GPT2 or compatible decoder model
        config: AMRPAConfig with arch='decoder'

    Returns:
        model:  patched model (in-place)
        state:  DecoderAMRPAState — call state.reset() before each sequence

    Example:
        config = AMRPAConfig.from_hf_config(model.config, arch='decoder')
        model, state = apply_amrpa_to_decoder(model, config)

        for batch in dataloader:
            state.reset()
            outputs = model(**batch)
    """
    assert config.arch == 'decoder', \
        f"Expected arch='decoder', got '{config.arch}'"

    # Find the transformer blocks
    # Handle GPT2LMHeadModel vs GPT2Model
    if hasattr(model, 'transformer'):
        transformer = model.transformer
    else:
        transformer = model

    all_layers   = _get_decoder_layers(transformer)
    total_layers = len(all_layers)
    start_layer  = total_layers - config.n_amrpa_layers

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    # Build internal CAMConfig from AMRPAConfig.cam
    internal_cam_config = InternalCAMConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        arch='decoder',
        n_amrpa_layers=config.n_amrpa_layers,
        window_size=config.cam.window_size,
        proj_rank=config.cam.proj_rank,
        importance_hidden=config.cam.importance_hidden,
        use_learned_importance=config.cam.use_learned_importance,
        gamma=config.gamma,
        alpha_temperature=config.alpha_temperature,
        gate_gamma_init=config.gate_gamma_init,
        gate_bias_init=config.gate_bias_init,
        dropout=config.dropout
    )

    # Shared CAM memory bank
    cam_bank = CAMMemoryBankSet(
        n_amrpa_layers=config.n_amrpa_layers,
        window_size=config.cam.window_size,
        arch='decoder'
    )

    amrpa_layers = []
    print(f"\nApplying AMRPA + CAM to decoder (last {config.n_amrpa_layers} layers):")

    for i in range(start_layer, total_layers):
        amrpa_idx     = i - start_layer
        block         = all_layers[i]
        original_attn = block.attn

        amrpa_core = AMRPACore(config).to(device)

        cam_module = CAMModule(
            config=internal_cam_config,
            layer_idx=amrpa_idx
        ).to(device)
        cam_module.set_memory_bank(cam_bank)

        amrpa_layer = DecoderAMRPALayer(
            original_attn=original_attn,
            amrpa_core=amrpa_core,
            cam_module=cam_module,
            layer_idx=i,
            amrpa_layer_idx=amrpa_idx,
            n_amrpa_layers=config.n_amrpa_layers,
            config=config
        ).to(device)

        block.attn = amrpa_layer
        amrpa_layers.append(amrpa_layer)

        n_params = sum(p.numel() for p in amrpa_core.parameters())
        n_cam    = sum(p.numel() for p in cam_module.parameters())
        print(f"  ✓ Layer {i} (AMRPA layer {amrpa_idx+1}): patched "
              f"[AMRPA={n_params:,}, CAM={n_cam:,} params]")

    state = DecoderAMRPAState(amrpa_layers, cam_bank)
    return model, state


def reset_decoder(state: DecoderAMRPAState):
    """Convenience function. Call before every new sequence."""
    state.reset()
