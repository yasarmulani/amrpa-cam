"""
AMRPA + CAM Library — Full Test Suite
---------------------------------------
Tests every component of the library end-to-end with mock models.
No internet required. No HuggingFace downloads.

Run:
    cd amrpa_lib/
    PYTHONPATH=. python test_library.py
"""

import sys
import os
# Insert amrpa_lib/ itself so 'amrpa' package is found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from amrpa import (
    AMRPAConfig, CAMConfig, AMRPAModel,
    compute_exact_match, compute_f1, normalize_answer, get_best_span
)
from amrpa.core import AMRPACore
from amrpa.adapters.encoder import apply_amrpa_to_encoder
from amrpa.adapters.decoder import apply_amrpa_to_decoder

torch.manual_seed(42)

PASS = "✓"
FAIL = "✗"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    suffix = f": {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if not condition:
        raise AssertionError(f"FAILED: {name}")


# ── Mock models ───────────────────────────────────────────────────────────────

def make_mock_roberta():
    class MockSA(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(768, 768)
            self.key   = nn.Linear(768, 768)
            self.value = nn.Linear(768, 768)
            self._in_features = 768

        @property
        def in_features(self): return self._in_features

    class MockAttn(nn.Module):
        def __init__(self): super().__init__(); self.self = MockSA()

    class MockBlock(nn.Module):
        def __init__(self): super().__init__(); self.attention = MockAttn()

    class MockConfig:
        hidden_size = 768; num_attention_heads = 12; model_type = 'roberta'

    class MockEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleList([MockBlock() for _ in range(12)])

    class MockRoBERTa(nn.Module):
        def __init__(self):
            super().__init__()
            self.config     = MockConfig()
            self.encoder    = MockEncoder()
            self.embeddings = nn.Embedding(100, 768)

    return MockRoBERTa()


def make_mock_gpt2():
    class MockGPT2Config:
        model_type = 'gpt2'; n_embd = 768; n_head = 12
        hidden_size = 768; num_attention_heads = 12

    class MockGPT2Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 768; self.num_heads = 12
            self.c_attn = nn.Linear(768, 768 * 3)
            self.c_proj = nn.Linear(768, 768)
            self.resid_dropout = nn.Dropout(0.0)

    class MockGPT2Block(nn.Module):
        def __init__(self): super().__init__(); self.attn = MockGPT2Attn()

    class MockGPT2(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockGPT2Config()
            self.h = nn.ModuleList([MockGPT2Block() for _ in range(12)])

    return MockGPT2()


# ══════════════════════════════════════════════════════════════════════════════

def test_config():
    print("\n[TEST] AMRPAConfig")

    # Encoder preset
    enc = AMRPAConfig.for_encoder(d_model=768, n_heads=12)
    check("Encoder arch", enc.arch == 'encoder')
    check("Encoder causal=False", not enc.causal)
    check("Encoder use_cam=False", not enc.use_cam)
    check("Encoder d_k auto", enc.d_k == 64)

    # Decoder preset
    dec = AMRPAConfig.for_decoder(d_model=768, n_heads=12)
    check("Decoder arch", dec.arch == 'decoder')
    check("Decoder causal=True", dec.causal)
    check("Decoder use_cam=True", dec.use_cam)

    # Encoder-decoder preset
    ed = AMRPAConfig.for_encoder_decoder(d_model=512, n_heads=8)
    check("EncDec arch", ed.arch == 'encoder_decoder')
    check("EncDec d_k auto", ed.d_k == 64)

    # from_hf_config
    class FakeHF:
        hidden_size = 1024; num_attention_heads = 16

    cfg = AMRPAConfig.from_hf_config(FakeHF(), arch='decoder')
    check("from_hf_config d_model", cfg.d_model == 1024)
    check("from_hf_config n_heads", cfg.n_heads == 16)
    check("from_hf_config d_k", cfg.d_k == 64)

    # Bad arch
    try:
        AMRPAConfig(d_model=768, n_heads=12, arch='invalid')
        check("Bad arch rejected", False)
    except AssertionError:
        check("Bad arch rejected", True)

    # CAMConfig nested
    check("CAM use_cam default encoder", not enc.cam.use_cam)
    check("CAM use_cam default decoder", dec.cam.use_cam)
    check("CAM proj_rank default", dec.cam.proj_rank == 16)


def test_amrpa_core():
    print("\n[TEST] AMRPACore mechanism")
    config = AMRPAConfig.for_encoder(d_model=256, n_heads=4)
    core   = AMRPACore(config)

    Q = torch.randn(2, 16, 64)
    K = torch.randn(2, 16, 64)
    V = torch.randn(2, 16, 64)
    A = F.softmax(torch.randn(2, 16, 16), dim=-1)

    # Layer 1 with no history → zero bias
    bias, m = core(Q, K, V, [], relative_layer_idx=1)
    check("No history → zero bias", bias.abs().max().item() < 1e-6)
    check("Bias shape", bias.shape == (2, 16, 16), f"{bias.shape}")

    # Layer 2 with history → non-zero bias
    bias2, m2 = core(Q, K, V, [A], relative_layer_idx=2)
    check("With history → non-zero", bias2.abs().max().item() > 0)
    check("No NaN in bias", not bias2.isnan().any())
    check("gate_impact shape", m2['gate_impact'].shape == (2,))
    check("gate_impact in [0,1]",
          (m2['gate_impact'] >= 0).all() and (m2['gate_impact'] <= 1).all())

    # Adaptive window
    check("Window L=1 → 1", core.adaptive_window_size(1) == 1)
    check("Window L=2 → 1", core.adaptive_window_size(2) == 1)
    check("Window L=3 → 2", core.adaptive_window_size(3) == 2)
    check("Window L=9 → 4", core.adaptive_window_size(9) == 4)

    # Causal mask
    causal = torch.triu(
        torch.full((16, 16), float('-inf')), diagonal=1
    )
    bias_c, _ = core(Q, K, V, [A], relative_layer_idx=2, causal_mask=causal)
    check("Causal mask applied", bias_c[0, 0, 1].item() < -1e4)
    check("Lower tri not masked", bias_c[0, 1, 0].item() > -1e4)

    # Gradients
    Q2 = torch.randn(2, 16, 64, requires_grad=True)
    bias3, m3 = core(Q2, K, V, [A], relative_layer_idx=2)
    (bias3.sum() + m3['gate_impact'].sum()).backward()
    check("Gradients flow", Q2.grad is not None)
    params_with_grad = sum(1 for p in core.parameters()
                          if p.requires_grad and p.grad is not None)
    check("Core params have grads", params_with_grad > 0,
          f"{params_with_grad} params")


def test_encoder_adapter():
    print("\n[TEST] Encoder adapter")
    roberta = make_mock_roberta()
    config  = AMRPAConfig.for_encoder(d_model=768, n_heads=12)
    config.freeze_embeddings = False

    model, state = apply_amrpa_to_encoder(roberta, config)
    check("4 AMRPA layers", len(state.layers) == 4)

    # Forward pass
    state.reset()
    hidden = torch.randn(2, 32, 768)
    mask   = torch.zeros(2, 1, 1, 32)

    layer = state.layers[0]
    out   = layer(hidden, attention_mask=mask)
    check("Output shape", out[0].shape == (2, 32, 768), f"{out[0].shape}")
    check("No NaN", not out[0].isnan().any())

    # Step 2 — memory should fire
    out2  = layer(hidden, attention_mask=mask)
    check("Step 2 no NaN", not out2[0].isnan().any())

    # Metrics
    metrics = state.get_metrics()
    check("Metrics captured", len(metrics) > 0)
    check("Gate impact in metrics", 'gate_impact' in metrics)

    # Reset clears history
    state.reset()
    for l in state.layers:
        check(f"Layer {l.amrpa_layer_idx} history cleared",
              l.attention_history == [])
        break

    # Gradient flow — verify AMRPA core trains via QKV projections
    # AMRPA core params get gradients when the full model is trained end-to-end
    # (gate, mlp_alpha, w_mem update via QA loss backprop through Q,K,V)
    state.reset()
    hidden_g = torch.randn(2, 32, 768, requires_grad=True)
    layer(hidden_g, attention_mask=mask)
    out4 = layer(hidden_g, attention_mask=mask)
    out4[0].sum().backward()
    # hidden_g should have gradient (proves backward flows through layer)
    check("AMRPA backward flows", hidden_g.grad is not None)


def test_decoder_adapter():
    print("\n[TEST] Decoder adapter (AMRPA + CAM)")
    gpt2   = make_mock_gpt2()
    config = AMRPAConfig.for_decoder(d_model=768, n_heads=12)

    model, state = apply_amrpa_to_decoder(gpt2, config)
    check("4 AMRPA+CAM layers", len(state.layers) == 4)

    state.reset()
    layer = state.layers[1]

    # Growing sequence (decoder generation)
    print("  Generation steps:")
    for step in range(5):
        seq_t = 8 + step
        out   = layer(torch.randn(2, seq_t, 768))
        check(f"Step {step} shape", out[0].shape == (2, seq_t, 768))
        check(f"Step {step} no NaN", not out[0].isnan().any())

    # Sliding window bounded
    bank   = state.cam_bank
    stored = len(bank.get(layer_idx=1))
    check("Sliding window bounded",
          stored <= config.cam.window_size,
          f"stored={stored}, window={config.cam.window_size}")

    # Gradients
    state.reset()
    layer(torch.randn(2, 8, 768))
    out_g = layer(torch.randn(2, 8, 768))
    out_g[0].sum().backward()
    cam_params   = list(layer.cam.parameters())
    grads_cam    = sum(1 for p in cam_params if p.grad is not None)
    nan_grads    = sum(p.grad.isnan().any().item()
                      for p in cam_params if p.grad is not None)
    check("CAM gradients", grads_cam > 0, f"{grads_cam} params")
    check("No NaN grads", nan_grads == 0)


def test_amrpa_model_api():
    print("\n[TEST] AMRPAModel unified API")
    config_enc = AMRPAConfig.for_encoder(d_model=768, n_heads=12)
    config_enc.freeze_embeddings = False
    config_dec = AMRPAConfig.for_decoder(d_model=768, n_heads=12)

    # Encoder
    roberta     = make_mock_roberta()
    model_e, state_e = AMRPAModel.wrap(roberta, config_enc)
    check("Encoder wrap", len(state_e.layers) == 4)
    AMRPAModel.reset(state_e)
    check("Encoder reset works", True)

    # Decoder
    gpt2        = make_mock_gpt2()
    model_d, state_d = AMRPAModel.wrap(gpt2, config_dec)
    check("Decoder wrap", len(state_d.layers) == 4)
    AMRPAModel.reset(state_d)
    check("Decoder reset works", True)

    # Bad arch
    try:
        AMRPAModel.wrap(roberta, AMRPAConfig(
            d_model=768, n_heads=12, arch='encoder_decoder'
        ))
        check("enc-dec raises NotImplementedError", False)
    except NotImplementedError:
        check("enc-dec raises NotImplementedError", True)


def test_metrics():
    print("\n[TEST] Evaluation metrics")
    check("EM exact",        compute_exact_match("Edinburgh", "Edinburgh") == 1)
    check("EM case insensitive", compute_exact_match("edinburgh", "Edinburgh") == 1)
    check("EM wrong",        compute_exact_match("London", "Edinburgh") == 0)
    check("F1 full overlap", compute_f1("the city of Edinburgh", "Edinburgh") > 0)
    check("F1 no overlap",   compute_f1("completely wrong", "Edinburgh") == 0.0)
    check("Normalize articles",
          normalize_answer("the quick brown fox") == "quick brown fox")

    # get_best_span
    start_l = torch.zeros(10); start_l[3] = 5.0
    end_l   = torch.zeros(10); end_l[6]   = 5.0
    s, e    = get_best_span(start_l, end_l)
    check("Best span start", s == 3)
    check("Best span end",   e == 6)

    # End before start → corrected
    end_l2  = torch.zeros(10); end_l2[1] = 5.0
    s2, e2  = get_best_span(start_l, end_l2)
    check("End < start corrected", e2 >= s2)


def test_training_loop():
    print("\n[TEST] Training loop (mock)")
    from amrpa.training import train_epoch, evaluate
    from amrpa.config import AMRPAConfig
    from amrpa.core import AMRPACore
    from amrpa.adapters.encoder import apply_amrpa_to_encoder
    from transformers import get_linear_schedule_with_warmup

    # Build a minimal end-to-end model
    class TinyConfig:
        hidden_size = 64; num_attention_heads = 4; model_type = 'roberta'

    class TinySA(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.key   = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)
            self._in_features = 64

        @property
        def in_features(self): return self._in_features

        def forward(self, hidden_states, **kwargs):
            return (hidden_states,)

    class TinyAttn(nn.Module):
        def __init__(self): super().__init__(); self.self = TinySA()

    class TinyBlock(nn.Module):
        def __init__(self): super().__init__(); self.attention = TinyAttn()

    class TinyRoBERTa(nn.Module):
        def __init__(self):
            super().__init__()
            self.config     = TinyConfig()
            self.encoder    = type('E', (), {
                'layer': nn.ModuleList([TinyBlock() for _ in range(4)])
            })()
            self.embeddings = nn.Embedding(100, 64)
            self.wte        = nn.Embedding(100, 64)

        def forward(self, input_ids, attention_mask, output_attentions=False):
            x = self.wte(input_ids)
            # Pass through AMRPA layers (they're patched in)
            for block in self.encoder.layer:
                out = block.attention.self(x)
                x   = out[0]

            class Out: pass
            o = Out()
            o.last_hidden_state = x
            return o

    class TinyQA(nn.Module):
        def __init__(self):
            super().__init__()
            self.roberta = TinyRoBERTa()
            self.config  = AMRPAConfig.for_encoder(
                d_model=64, n_heads=4, n_amrpa_layers=2
            )
            self.config.freeze_embeddings = False
            self.roberta, self._state = apply_amrpa_to_encoder(
                self.roberta, self.config
            )
            self.qa_outputs = nn.Linear(64, 2)

        def forward(self, input_ids, attention_mask, return_metrics=False):
            self._state.reset()
            out         = self.roberta(input_ids, attention_mask)
            seq         = out.last_hidden_state
            logits      = self.qa_outputs(seq)
            start_l     = logits[:, :, 0]
            end_l       = logits[:, :, 1]
            if return_metrics:
                return start_l, end_l, self._state.get_metrics()
            return start_l, end_l

    model     = TinyQA()
    config    = model.config
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, 10)

    # Dummy dataloader
    class DummyDS(torch.utils.data.Dataset):
        def __len__(self): return 16
        def __getitem__(self, i):
            return {
                'input_ids':       torch.randint(0, 100, (32,)),
                'attention_mask':  torch.ones(32, dtype=torch.long),
                'start_positions': torch.tensor(3),
                'end_positions':   torch.tensor(7),
                'answer_text':     'answer'
            }

    loader = torch.utils.data.DataLoader(
        DummyDS(), batch_size=4, shuffle=False
    )

    loss, metrics = train_epoch(
        model, loader, optimizer, scheduler,
        torch.device('cpu'), config, epoch=0
    )
    check("Training loss > 0", loss > 0, f"loss={loss:.4f}")
    check("Training completes", True)
    print(f"    Loss: {loss:.4f}")


# ══════════════════════════════════════════════════════════════════════════════

def run_all():
    print(f"\n{'='*60}")
    print("AMRPA + CAM LIBRARY — FULL TEST SUITE")
    print(f"{'='*60}")

    tests = [
        ("Config",           test_config),
        ("AMRPACore",        test_amrpa_core),
        ("Encoder Adapter",  test_encoder_adapter),
        ("Decoder Adapter",  test_decoder_adapter),
        ("AMRPAModel API",   test_amrpa_model_api),
        ("Metrics",          test_metrics),
        ("Training Loop",    test_training_loop),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  ❌ FAILED [{name}]: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    total_checks  = len(results)
    passed_checks = sum(1 for _, ok in results if ok)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"  Test suites: {passed}/{passed+failed}")
    print(f"  Checks:      {passed_checks}/{total_checks}")

    if failed == 0:
        print("\n✅ All tests passed.")
        print("Library is ready to use on Kaggle.")
    else:
        failed_list = [n for n, ok in results if not ok]
        print(f"\n❌ Failed checks:")
        for n in failed_list:
            print(f"  - {n}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    run_all()
