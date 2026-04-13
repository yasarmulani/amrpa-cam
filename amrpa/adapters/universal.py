"""
AMRPA Universal Adapter
------------------------
Single adapter that works for any model in the registry.
Replaces the hardcoded encoder.py and decoder.py approach.

Handles:
    - Combined QKV (GPT2 style: c_attn → split)
    - Separate QKV (BERT/LLaMA style: q_proj, k_proj, v_proj)
    - Causal masking (decoder)
    - Bidirectional (encoder)
    - Nested attention modules (GPT-Neo)

Usage:
    from amrpa.adapters.universal import apply_amrpa_universal

    model, state = apply_amrpa_universal(model, config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Dict, Any

from ..config import AMRPAConfig
from ..core import AMRPACore
from ..cam import CAMModule, CAMMemoryBankSet
from ..cam.cam_config import CAMConfig as InternalCAMConfig
from .registry import get_model_info


# ── Utility: navigate nested attributes ──────────────────────────────────────

def _get_nested_attr(obj, path: str):
    """Get nested attribute using dot notation. e.g. 'attention.self'"""
    parts = path.split('.')
    for part in parts:
        obj = getattr(obj, part)
    return obj


def _set_nested_attr(obj, path: str, value):
    """Set nested attribute using dot notation."""
    parts = path.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _get_layers(model: nn.Module, layers_path: List[str]) -> List[nn.Module]:
    """Navigate model to get list of transformer blocks."""
    obj = model
    for attr in layers_path:
        obj = getattr(obj, attr)
    return list(obj)


# ── QKV extraction ───────────────────────────────────────────────────────────

def extract_qkv(
    attn_module: nn.Module,
    hidden_states: torch.Tensor,
    qkv_style: str,
    embed_dim: int,
    model_type: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract Q, K, V from attention module regardless of style.

    Combined: c_attn projects to 3*embed_dim, then split
    Separate: q_proj, k_proj, v_proj project independently
    """
    if qkv_style == 'combined':
        # GPT2 style
        qkv = attn_module.c_attn(hidden_states)
        Q, K, V = qkv.split(embed_dim, dim=2)

    elif qkv_style == 'separate':
        # BERT/RoBERTa/LLaMA style
        if hasattr(attn_module, 'query'):
            # BERT/RoBERTa
            Q = attn_module.query(hidden_states)
            K = attn_module.key(hidden_states)
            V = attn_module.value(hidden_states)
        elif hasattr(attn_module, 'q_proj'):
            # LLaMA/Mistral/Phi/GPT-Neo style
            Q = attn_module.q_proj(hidden_states)
            K = attn_module.k_proj(hidden_states)
            V = attn_module.v_proj(hidden_states)
        elif hasattr(attn_module, 'q'):
            # T5 style
            Q = attn_module.q(hidden_states)
            K = attn_module.k(hidden_states)
            V = attn_module.v(hidden_states)
        else:
            raise ValueError(
                f"Cannot find QKV projections in {type(attn_module).__name__}. "
                f"Expected query/key/value or q_proj/k_proj/v_proj or q/k/v"
            )
    else:
        raise ValueError(f"Unknown qkv_style: {qkv_style}")

    return Q, K, V


def get_embed_dim(attn_module: nn.Module, qkv_style: str) -> int:
    """Get embedding dimension from attention module."""
    if qkv_style == 'combined':
        if hasattr(attn_module, 'embed_dim'):
            return attn_module.embed_dim
        if hasattr(attn_module, 'c_attn'):
            return attn_module.c_attn.weight.shape[1]
    else:
        if hasattr(attn_module, 'query'):
            return attn_module.query.in_features
        if hasattr(attn_module, 'q_proj'):
            return attn_module.q_proj.in_features
        if hasattr(attn_module, 'q'):
            return attn_module.q.in_features
    raise ValueError(f"Cannot determine embed_dim from {type(attn_module)}")


def get_num_heads(attn_module: nn.Module, model_type: str) -> int:
    """Get number of attention heads."""
    for attr in ['num_heads', 'num_attention_heads', 'n_head',
                 'num_key_value_heads']:
        if hasattr(attn_module, attr):
            return getattr(attn_module, attr)
    # Try parent module config
    if hasattr(attn_module, 'query'):
        # BERT/RoBERTa: infer from weight shape
        d_model = attn_module.query.in_features
        d_out   = attn_module.query.out_features
        # typical: out_features == d_model, each head = d_model // n_heads
        # we cannot infer n_heads without config, default to 12
        return 12
    return 12  # safe default


# ── Universal AMRPA Layer ─────────────────────────────────────────────────────

class UniversalAMRPALayer(nn.Module):
    """
    Single AMRPA layer that works for any model architecture.

    Wraps any attention module and injects AMRPA + CAM memory bias.
    Handles both combined and separate QKV projections.
    Handles both causal (decoder) and bidirectional (encoder) attention.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        amrpa_core: AMRPACore,
        cam_module: Optional[CAMModule],
        config: AMRPAConfig,
        model_info: dict,
        amrpa_layer_idx: int,
        n_amrpa_layers: int,
        model_type: str,
    ):
        super().__init__()
        self.original        = original_attn
        self.amrpa           = amrpa_core
        self.cam             = cam_module
        self.config          = config
        self.model_info      = model_info
        self.amrpa_layer_idx = amrpa_layer_idx
        self.n_amrpa_layers  = n_amrpa_layers
        self.model_type      = model_type

        self.qkv_style = model_info['qkv_style']
        self.causal    = config.causal

        self.embed_dim = get_embed_dim(original_attn, self.qkv_style)
        self.num_heads = get_num_heads(original_attn, model_type)
        self.head_dim  = self.embed_dim // self.num_heads
        self.scale     = 1.0 / math.sqrt(self.head_dim)

        # Shared attention history for encoder (no CAM)
        self.attention_history: Optional[List] = None
        self.last_metrics: Dict = {}

    def _apply_attention_mask(
        self,
        attn_scores: torch.Tensor,
        attention_mask,
        batch_size: int,
        seq_len: int,
        kv_seq_len: int
    ) -> torch.Tensor:
        """Apply attention mask regardless of its shape."""
        if attention_mask is None:
            return attn_scores

        if attention_mask.dim() == 4:
            # (batch, 1, seq, kv_seq)
            attn_scores = attn_scores + attention_mask[:, 0, :seq_len, :kv_seq_len]
        elif attention_mask.dim() == 3:
            attn_scores = attn_scores + attention_mask[:, :seq_len, :kv_seq_len]
        elif attention_mask.dim() == 2:
            attn_scores = attn_scores + attention_mask[:, None, :kv_seq_len]

        return attn_scores

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """
        Universal forward pass.
        Extracts Q,K,V → computes base attention → injects AMRPA memory →
        returns (context, ...) matching original module's output format.
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # ── Extract Q, K, V ──────────────────────────────────────────────────
        Q, K, V = extract_qkv(
            self.original, hidden_states,
            self.qkv_style, self.embed_dim, self.model_type
        )

        kv_seq_len = K.shape[1]

        # ── Base attention scores ─────────────────────────────────────────────
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_scores = self._apply_attention_mask(
            attn_scores, attention_mask, batch_size, seq_len, kv_seq_len
        )

        # ── Causal mask (decoder only) ────────────────────────────────────────
        if self.causal:
            causal_mask = torch.triu(
                torch.full((seq_len, kv_seq_len), float('-inf'), device=device),
                diagonal=1
            )
            attn_scores = attn_scores + causal_mask

        # ── AMRPA memory injection ────────────────────────────────────────────
        Q_cam = Q[:, :, :self.config.d_k]
        K_cam = K[:, :, :self.config.d_k]
        V_cam = V[:, :, :self.config.d_k]

        layer_depth = self.amrpa_layer_idx / max(self.n_amrpa_layers, 1)

        if self.cam is not None:
            # Decoder: use CAM for compressed memory
            with torch.no_grad():
                base_A = F.softmax(
                    torch.matmul(Q_cam, K_cam.transpose(-2, -1)) * self.scale,
                    dim=-1
                )
            cam_mask = torch.ones(batch_size, seq_len, device=device)
            memory_bias, metrics = self.cam(
                Q=Q_cam, K=K_cam, V=V_cam,
                A=base_A,
                layer_depth=layer_depth,
                attention_mask=cam_mask
            )
        else:
            # Encoder: use raw attention history
            history = self.attention_history or []
            relative_idx = self.amrpa_layer_idx + 1
            memory_bias, metrics = self.amrpa(
                Q=Q_cam, K=K_cam, V=V_cam,
                attention_history=history,
                relative_layer_idx=relative_idx,
                causal_mask=None
            )

        # Add memory bias
        if memory_bias.shape[-2:] == attn_scores.shape[-2:]:
            attn_scores = attn_scores + memory_bias

        # ── Final attention + output ──────────────────────────────────────────
        attn_probs = F.softmax(attn_scores, dim=-1)
        context    = torch.matmul(attn_probs, V)

        # Store attention for encoder history
        if self.attention_history is not None:
            self.attention_history.append(attn_probs.detach())

        # Project output
        out_attr = self.model_info.get('out_attr')
        if out_attr and hasattr(self.original, out_attr):
            context = getattr(self.original, out_attr)(context)

        drop_attr = self.model_info.get('dropout_attr')
        if drop_attr and hasattr(self.original, drop_attr):
            context = getattr(self.original, drop_attr)(context)

        self.last_metrics = {k: v.detach().cpu() for k, v in metrics.items()}

        # Return in format expected by HF
        return (context, None)


# ── State objects ─────────────────────────────────────────────────────────────

class UniversalAMRPAState:
    """State object for any architecture."""

    def __init__(
        self,
        layers: List[UniversalAMRPALayer],
        cam_bank: Optional[CAMMemoryBankSet] = None,
        arch: str = 'encoder'
    ):
        self.layers   = layers
        self.cam_bank = cam_bank
        self.arch     = arch

    def reset(self):
        """Reset between sequences. Call before every forward pass."""
        if self.cam_bank is not None:
            self.cam_bank.reset()
        for layer in self.layers:
            if layer.cam is not None:
                layer.cam.reset()
            layer.attention_history = [] if layer.cam is None else None
            layer.last_metrics = {}

    def get_metrics(self) -> Dict:
        all_metrics = [l.last_metrics for l in self.layers if l.last_metrics]
        if not all_metrics:
            return {}
        keys = set().union(*[m.keys() for m in all_metrics])
        merged = {}
        for k in keys:
            vals = [m[k] for m in all_metrics if k in m]
            if vals:
                try:
                    merged[k] = torch.stack(vals).mean(0)
                except Exception:
                    merged[k] = vals[0]
        return merged


# ── Main apply function ───────────────────────────────────────────────────────

def apply_amrpa_universal(
    model: nn.Module,
    config: AMRPAConfig
) -> Tuple[nn.Module, UniversalAMRPAState]:
    """
    Apply AMRPA to any supported model.
    Automatically detects architecture from model.config.model_type.

    Args:
        model:  Any HuggingFace model
        config: AMRPAConfig

    Returns:
        model:  patched model (in-place)
        state:  UniversalAMRPAState

    Usage:
        from amrpa.adapters.universal import apply_amrpa_universal

        model, state = apply_amrpa_universal(model, config)

        for batch in dataloader:
            state.reset()
            outputs = model(**batch)
            metrics = state.get_metrics()
    """
    # Get model type
    base = (getattr(model, 'roberta', None) or
            getattr(model, 'bert', None) or
            getattr(model, 'transformer', None) or
            model)
    model_type = base.config.model_type if hasattr(base, 'config') \
                 else model.config.model_type

    model_info = get_model_info(model_type)
    arch       = model_info['arch']

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    # Build CAM config if needed
    cam_bank = None
    internal_cam_config = None

    if config.use_cam:
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
        cam_bank = CAMMemoryBankSet(
            n_amrpa_layers=config.n_amrpa_layers,
            window_size=config.cam.window_size,
            arch='decoder'
        )

    # Handle encoder-decoder separately
    if arch == 'encoder_decoder':
        return _apply_encoder_decoder(
            model, config, model_info, model_type, device,
            internal_cam_config, cam_bank
        )

    # Get layers for encoder or decoder
    if arch == 'encoder':
        enc_info = model_info
        layers_path = enc_info['layers_path']
        # Handle wrapped models
        root = (getattr(model, 'roberta', None) or
                getattr(model, 'bert', None) or
                getattr(model, 'deberta', None) or
                model)
        all_layers = _get_layers(root, layers_path)
    else:
        layers_path = model_info['layers_path']
        # For decoder, navigate from model root using full path
        all_layers = _get_layers(model, layers_path)

    total_layers = len(all_layers)
    start_layer  = total_layers - config.n_amrpa_layers

    amrpa_layers = []
    print(f"\nApplying AMRPA to {model_type} {arch} "
          f"(last {config.n_amrpa_layers} layers):")

    for i in range(start_layer, total_layers):
        amrpa_idx    = i - start_layer
        block        = all_layers[i]
        attn_attr    = model_info['attn_attr']
        original_attn = _get_nested_attr(block, attn_attr)

        amrpa_core = AMRPACore(config).to(device)

        cam_module = None
        if config.use_cam and cam_bank is not None:
            cam_module = CAMModule(
                config=internal_cam_config,
                layer_idx=amrpa_idx
            ).to(device)
            cam_module.set_memory_bank(cam_bank)

        wrapper = UniversalAMRPALayer(
            original_attn=original_attn,
            amrpa_core=amrpa_core,
            cam_module=cam_module,
            config=config,
            model_info=model_info,
            amrpa_layer_idx=amrpa_idx,
            n_amrpa_layers=config.n_amrpa_layers,
            model_type=model_type,
        ).to(device)

        _set_nested_attr(block, attn_attr, wrapper)
        amrpa_layers.append(wrapper)

        n_amrpa = sum(p.numel() for p in amrpa_core.parameters())
        n_cam   = sum(p.numel() for p in cam_module.parameters()) \
                  if cam_module else 0
        print(f"  ✓ Layer {i} (AMRPA {amrpa_idx+1}): "
              f"[AMRPA={n_amrpa:,}, CAM={n_cam:,}]")

    state = UniversalAMRPAState(amrpa_layers, cam_bank, arch)
    return model, state


def _apply_encoder_decoder(
    model, config, model_info, model_type, device,
    internal_cam_config, cam_bank
):
    """Apply AMRPA to both encoder and decoder sides."""
    all_layers = []

    for side in ('encoder', 'decoder'):
        side_info   = model_info[side]
        layers_path = side_info['layers_path']
        all_side_layers = _get_layers(model, layers_path)
        total       = len(all_side_layers)
        start       = total - config.n_amrpa_layers
        is_causal   = (side == 'decoder')

        print(f"\nApplying AMRPA to {model_type} {side} "
              f"(last {config.n_amrpa_layers} layers):")

        for i in range(start, total):
            amrpa_idx     = i - start
            block         = all_side_layers[i]
            attn_attr     = side_info['attn_attr']
            original_attn = _get_nested_attr(block, attn_attr)

            # Use decoder config for causal side
            cfg_side = config
            if is_causal:
                import copy
                cfg_side = copy.copy(config)
                cfg_side.__dict__['_causal_override'] = True

            amrpa_core = AMRPACore(config).to(device)

            cam_module = None
            if is_causal and config.use_cam and cam_bank is not None:
                cam_module = CAMModule(
                    config=internal_cam_config,
                    layer_idx=amrpa_idx
                ).to(device)
                cam_module.set_memory_bank(cam_bank)

            # Build side-specific model_info
            side_model_info = {
                'arch':         side,
                'layers_path':  side_info['layers_path'],
                'attn_attr':    side_info['attn_attr'],
                'qkv_style':    side_info['qkv_style'],
                'out_attr':     side_info.get('out_attr'),
                'dropout_attr': side_info.get('dropout_attr'),
            }

            wrapper = UniversalAMRPALayer(
                original_attn=original_attn,
                amrpa_core=amrpa_core,
                cam_module=cam_module,
                config=config,
                model_info=side_model_info,
                amrpa_layer_idx=amrpa_idx,
                n_amrpa_layers=config.n_amrpa_layers,
                model_type=model_type,
            ).to(device)

            # Override causal for decoder side
            wrapper.causal = is_causal

            _set_nested_attr(block, attn_attr, wrapper)
            all_layers.append(wrapper)

            n_amrpa = sum(p.numel() for p in amrpa_core.parameters())
            n_cam   = sum(p.numel() for p in cam_module.parameters()) \
                      if cam_module else 0
            print(f"  ✓ {side} Layer {i} (AMRPA {amrpa_idx+1}): "
                  f"[AMRPA={n_amrpa:,}, CAM={n_cam:,}]")

    state = UniversalAMRPAState(all_layers, cam_bank, 'encoder_decoder')
    return model, state
