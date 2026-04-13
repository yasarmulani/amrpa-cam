"""
AMRPA Model Registry
---------------------
Lookup table for all supported model architectures.
Maps HuggingFace model_type to layer access patterns.

Adding a new model:
    1. Find model_type from model.config.model_type
    2. Find layers path by inspecting model architecture
    3. Find attention attr name
    4. Determine QKV style (combined or separate)
    5. Add entry below
"""

# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {

    # ── Encoder Only ──────────────────────────────────────────────────────────

    'roberta': {
        'arch':          'encoder',
        'layers_path':   ['encoder', 'layer'],
        'attn_attr':     'attention.self',
        'qkv_style':     'separate',       # query, key, value projections
        'out_attr':      None,             # encoder output handled by HF
        'dropout_attr':  None,
        'in_features':   'query.in_features',
    },
    'bert': {
        'arch':          'encoder',
        'layers_path':   ['encoder', 'layer'],
        'attn_attr':     'attention.self',
        'qkv_style':     'separate',
        'out_attr':      None,
        'dropout_attr':  None,
        'in_features':   'query.in_features',
    },
    'deberta': {
        'arch':          'encoder',
        'layers_path':   ['encoder', 'layer'],
        'attn_attr':     'attention.self',
        'qkv_style':     'separate',
        'out_attr':      None,
        'dropout_attr':  None,
        'in_features':   'query_proj.in_features',
    },
    'deberta-v2': {
        'arch':          'encoder',
        'layers_path':   ['encoder', 'layer'],
        'attn_attr':     'attention.self',
        'qkv_style':     'separate',
        'out_attr':      None,
        'dropout_attr':  None,
        'in_features':   'query_proj.in_features',
    },

    # ── Decoder Only ──────────────────────────────────────────────────────────

    'gpt2': {
        'arch':          'decoder',
        'layers_path':   ['transformer', 'h'],
        'attn_attr':     'attn',
        'qkv_style':     'combined',       # c_attn → split into Q, K, V
        'out_attr':      'c_proj',
        'dropout_attr':  'resid_dropout',
        'in_features':   None,
    },
    'gpt_neo': {
        'arch':          'decoder',
        'layers_path':   ['transformer', 'h'],
        'attn_attr':     'attn.attention',  # GPT-Neo nests attention
        'qkv_style':     'separate',        # q_proj, k_proj, v_proj
        'out_attr':      'out_proj',
        'dropout_attr':  None,
        'in_features':   None,
    },
    'phi': {
        'arch':          'decoder',
        'layers_path':   ['model', 'layers'],
        'attn_attr':     'self_attn',
        'qkv_style':     'separate',
        'out_attr':      'dense',
        'dropout_attr':  None,
        'in_features':   None,
    },
    'llama': {
        'arch':          'decoder',
        'layers_path':   ['model', 'layers'],
        'attn_attr':     'self_attn',
        'qkv_style':     'separate',
        'out_attr':      'o_proj',
        'dropout_attr':  None,
        'in_features':   None,
    },
    'mistral': {
        'arch':          'decoder',
        'layers_path':   ['model', 'layers'],
        'attn_attr':     'self_attn',
        'qkv_style':     'separate',
        'out_attr':      'o_proj',
        'dropout_attr':  None,
        'in_features':   None,
    },

    # ── Encoder-Decoder ───────────────────────────────────────────────────────

    't5': {
        'arch':          'encoder_decoder',
        'encoder': {
            'layers_path':  ['encoder', 'block'],
            'attn_attr':    'layer.0.SelfAttention',
            'qkv_style':    'separate',    # q, k, v projections
            'out_attr':     'o',
            'dropout_attr': None,
        },
        'decoder': {
            'layers_path':  ['decoder', 'block'],
            'attn_attr':    'layer.0.SelfAttention',
            'qkv_style':    'separate',
            'out_attr':     'o',
            'dropout_attr': None,
        },
    },
    'bart': {
        'arch':          'encoder_decoder',
        'encoder': {
            'layers_path':  ['model', 'encoder', 'layers'],
            'attn_attr':    'self_attn',
            'qkv_style':    'separate',
            'out_attr':     'out_proj',
            'dropout_attr': None,
        },
        'decoder': {
            'layers_path':  ['model', 'decoder', 'layers'],
            'attn_attr':    'self_attn',
            'qkv_style':    'separate',
            'out_attr':     'out_proj',
            'dropout_attr': None,
        },
    },
}


def get_model_info(model_type: str) -> dict:
    """
    Get registry entry for a model type.
    Raises clear error if model not supported.
    """
    info = MODEL_REGISTRY.get(model_type)
    if info is None:
        supported = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model type '{model_type}' not in registry.\n"
            f"Supported: {supported}\n"
            f"To add support, add an entry to amrpa/adapters/registry.py"
        )
    return info


def is_supported(model_type: str) -> bool:
    return model_type in MODEL_REGISTRY


def get_arch(model_type: str) -> str:
    return get_model_info(model_type)['arch']
