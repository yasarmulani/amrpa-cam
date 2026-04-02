"""
AMRPA + CAM FLOPs Analysis
----------------------------
Computes and visualises compute overhead for AMRPA and CAM.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional


def _linear_flops(in_dim, out_dim, seq, batch=1):
    return 2.0 * batch * seq * in_dim * out_dim

def _matmul_flops(a, b, c, batch=1):
    return 2.0 * batch * a * b * c


def compute_baseline_flops(d_model, n_heads, seq_len, n_layers, batch_size=1):
    d, B, S = d_model, batch_size, seq_len
    per_layer = (
        3 * _linear_flops(d, d, S, B) +
        _matmul_flops(S, d, S, B) +
        _matmul_flops(S, S, d, B) +
        _linear_flops(d, d, S, B)
    )
    total = per_layer * n_layers
    return {
        'per_layer_mflops': per_layer / 1e6,
        'total_mflops':     total / 1e6,
        'total_gflops':     total / 1e9,
        'n_layers':         n_layers
    }


def compute_amrpa_flops(d_model, d_k, d_mlp, n_amrpa_layers, gamma,
                         seq_len, batch_size=1):
    B, S, dk = batch_size, seq_len, d_k

    def adaptive_window(l):
        if l <= 2:   return 1
        elif l <= 8: return math.floor(math.log2(l)) + 1
        else:        return 4

    total_flops = 0.0
    per_layer_breakdown = []

    for rel_idx in range(1, n_amrpa_layers + 1):
        if rel_idx == 1:
            layer_flops = 0.0
        else:
            w = adaptive_window(rel_idx)
            proj_attn  = _matmul_flops(S, S, dk, B) + _linear_flops(dk, dk, S, B)
            alpha_mlp  = (_linear_flops(dk * 2, d_mlp, S, B) +
                          _linear_flops(d_mlp, 1, S, B))
            gate_sim   = 2.0 * B * S * dk
            w_mem      = _linear_flops(dk, dk, S, B)
            mem_bias   = _matmul_flops(S, dk, S, B)
            per_step   = proj_attn + alpha_mlp + gate_sim + w_mem + mem_bias
            layer_flops = per_step * w
        total_flops += layer_flops
        per_layer_breakdown.append(layer_flops / 1e6)

    return {
        'total_mflops':     total_flops / 1e6,
        'total_gflops':     total_flops / 1e9,
        'per_layer_mflops': per_layer_breakdown,
        'n_amrpa_layers':   n_amrpa_layers
    }


def compute_cam_flops(d_k, proj_rank, importance_hidden, n_amrpa_layers,
                       window_size, seq_len, batch_size=1):
    B, S, dk, r, h = batch_size, seq_len, d_k, proj_rank, importance_hidden

    importance_mlp = (
        _linear_flops(dk * 3 + 2, h, S, B) +
        _linear_flops(h, h // 2, S, B) +
        _linear_flops(h // 2, 1, S, B)
    )
    proj_ops   = _linear_flops(dk, r, S, B) * 2
    alpha_ops  = (_linear_flops(dk * 2, dk, 1, B) +
                  _linear_flops(dk, 1, 1, B)) * window_size
    gate_ops   = 2.0 * B * S * dk
    per_step   = importance_mlp + proj_ops + alpha_ops + gate_ops
    total      = per_step * n_amrpa_layers

    naive_bytes = B * S * S * 4
    cam_bytes   = B * r * dk * 2 * 4
    reduction   = naive_bytes / max(cam_bytes, 1)

    return {
        'total_mflops':       total / 1e6,
        'total_gflops':       total / 1e9,
        'memory_reduction_x': reduction,
        'naive_storage_mb':   (naive_bytes * window_size * n_amrpa_layers) / (1024**2),
        'cam_storage_mb':     (cam_bytes   * window_size * n_amrpa_layers) / (1024**2),
        'n_amrpa_layers':     n_amrpa_layers
    }


def print_flops_summary(config, seq_len=384, batch_size=1, side='main'):
    """
    Print FLOPs breakdown.
    For encoder-decoder call twice:
        print_flops_summary(config, side='encoder')
        print_flops_summary(config, side='decoder')
    """
    label = {'main': 'AMRPA', 'encoder': 'ENCODER SIDE',
             'decoder': 'DECODER SIDE'}.get(side, 'AMRPA')
    total_layers = 12

    print(f"\n{'═'*60}")
    print(f"📊 FLOPs Analysis: {label}")
    print(f"   seq={seq_len}, batch={batch_size}, "
          f"d_model={config.d_model}, n_heads={config.n_heads}")
    print(f"{'─'*60}")

    baseline = compute_baseline_flops(
        config.d_model, config.n_heads, seq_len, total_layers, batch_size
    )
    amrpa = compute_amrpa_flops(
        d_model=config.d_model, d_k=config.d_model,
        d_mlp=config.d_mlp,
        n_amrpa_layers=config.n_amrpa_layers,
        gamma=config.gamma, seq_len=seq_len, batch_size=batch_size
    )

    total_mf         = baseline['total_mflops'] + amrpa['total_mflops']
    amrpa_overhead   = (amrpa['total_mflops'] / baseline['total_mflops']) * 100

    print(f"  Baseline attention:      {baseline['total_gflops']:.4f} GFLOPs")
    print(f"  AMRPA extra:            +{amrpa['total_mflops']:.2f} MFLOPs")
    print(f"  AMRPA overhead:         +{amrpa_overhead:.2f}%")
    print(f"  Total with AMRPA:        {total_mf/1e3:.4f} GFLOPs")

    result = {
        'baseline_gflops':    baseline['total_gflops'],
        'amrpa_mflops':       amrpa['total_mflops'],
        'amrpa_overhead_pct': amrpa_overhead,
        'total_gflops':       total_mf / 1e3
    }

    if config.use_cam:
        cam = compute_cam_flops(
            d_k=config.d_k,
            proj_rank=config.cam.proj_rank,
            importance_hidden=config.cam.importance_hidden,
            n_amrpa_layers=config.n_amrpa_layers,
            window_size=config.cam.window_size,
            seq_len=seq_len, batch_size=batch_size
        )
        cam_overhead = (cam['total_mflops'] / baseline['total_mflops']) * 100
        total_mf    += cam['total_mflops']

        print(f"\n  CAM extra:              +{cam['total_mflops']:.2f} MFLOPs")
        print(f"  CAM overhead:           +{cam_overhead:.2f}%")
        print(f"  Total AMRPA+CAM:         {total_mf/1e3:.4f} GFLOPs")
        print(f"\n  CAM memory reduction:    {cam['memory_reduction_x']:.0f}x")
        print(f"  Naive storage:           {cam['naive_storage_mb']:.1f} MB")
        print(f"  CAM storage:             {cam['cam_storage_mb']:.3f} MB")

        result.update({
            'cam_mflops':            cam['total_mflops'],
            'cam_overhead_pct':      cam_overhead,
            'total_with_cam_gflops': total_mf / 1e3,
            'memory_reduction_x':    cam['memory_reduction_x'],
            'naive_storage_mb':      cam['naive_storage_mb'],
            'cam_storage_mb':        cam['cam_storage_mb']
        })

    print(f"{'═'*60}")
    return result


def plot_flops_comparison(config, save_path='flops_comparison.png',
                           seq_lengths=None, side='main'):
    """
    Plot FLOPs breakdown across sequence lengths. Saves PNG.
    For encoder-decoder call twice with different save_path and side.
    """
    if seq_lengths is None:
        seq_lengths = [64, 128, 192, 256, 320, 384]

    total_layers = 12
    baseline_mf_list, amrpa_mf_list, cam_mf_list, overhead_pct_list = [], [], [], []

    for S in seq_lengths:
        bl = compute_baseline_flops(config.d_model, config.n_heads, S, total_layers)
        am = compute_amrpa_flops(d_model=config.d_model, d_k=config.d_model,
                                  d_mlp=config.d_mlp,
                                  n_amrpa_layers=config.n_amrpa_layers,
                                  gamma=config.gamma, seq_len=S)
        total_mf = bl['total_mflops'] + am['total_mflops']
        cam_mf   = 0.0

        if config.use_cam:
            cm = compute_cam_flops(d_k=config.d_k,
                                    proj_rank=config.cam.proj_rank,
                                    importance_hidden=config.cam.importance_hidden,
                                    n_amrpa_layers=config.n_amrpa_layers,
                                    window_size=config.cam.window_size,
                                    seq_len=S)
            cam_mf    = cm['total_mflops']
            total_mf += cam_mf

        baseline_mf_list.append(bl['total_mflops'])
        amrpa_mf_list.append(am['total_mflops'])
        cam_mf_list.append(cam_mf)
        overhead_pct_list.append(
            ((total_mf - bl['total_mflops']) / bl['total_mflops']) * 100
        )

    label = {'main': 'AMRPA', 'encoder': 'Encoder Side',
             'decoder': 'Decoder Side'}.get(side, 'AMRPA')
    x     = np.arange(len(seq_lengths))
    xlabs = [str(s) for s in seq_lengths]

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)
    fig.suptitle(f'FLOPs: Baseline vs {label}', fontsize=15, fontweight='bold')

    # 1. Stacked bars
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x, baseline_mf_list, color='steelblue', label='Baseline')
    ax1.bar(x, amrpa_mf_list, bottom=baseline_mf_list,
            color='orange', label='AMRPA')
    if config.use_cam:
        b2 = [a + b for a, b in zip(baseline_mf_list, amrpa_mf_list)]
        ax1.bar(x, cam_mf_list, bottom=b2, color='crimson', label='CAM')
    ax1.set_title('Stacked MFLOPs'); ax1.set_xlabel('Seq Length')
    ax1.set_ylabel('MFLOPs'); ax1.set_xticks(x); ax1.set_xticklabels(xlabs)
    ax1.legend(); ax1.grid(True, alpha=0.3, axis='y')

    # 2. Overhead %
    ax2 = fig.add_subplot(gs[0, 1])
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(seq_lengths)))
    bars   = ax2.bar(x, overhead_pct_list, color=colors)
    for bar, pct in zip(bars, overhead_pct_list):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'+{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    ax2.set_title('Overhead (%)'); ax2.set_xlabel('Seq Length')
    ax2.set_ylabel('Overhead (%)'); ax2.set_xticks(x); ax2.set_xticklabels(xlabs)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. FLOPs scaling log
    ax3 = fig.add_subplot(gs[0, 2])
    total_gf = [(b + a + c) / 1e3
                for b, a, c in zip(baseline_mf_list, amrpa_mf_list, cam_mf_list)]
    baseline_gf = [b / 1e3 for b in baseline_mf_list]
    ax3.plot(seq_lengths, baseline_gf, 'o-', color='steelblue',
             linewidth=2, label='Baseline')
    ax3.plot(seq_lengths, total_gf, 's-', color='crimson',
             linewidth=2, label=f'+ {label}')
    ax3.set_yscale('log'); ax3.set_title('FLOPs Scaling (log)')
    ax3.set_xlabel('Seq Length'); ax3.set_ylabel('GFLOPs (log)')
    ax3.legend(); ax3.grid(True, alpha=0.3)

    # 4. Per-layer AMRPA
    ax4 = fig.add_subplot(gs[1, 0])
    per_l = compute_amrpa_flops(d_model=config.d_model, d_k=config.d_model,
                                 d_mlp=config.d_mlp,
                                 n_amrpa_layers=config.n_amrpa_layers,
                                 gamma=config.gamma, seq_len=384)['per_layer_mflops']
    ax4.bar([f'L{i+1}' for i in range(len(per_l))], per_l, color='orange')
    ax4.set_title('AMRPA MFLOPs per Layer (seq=384)')
    ax4.set_xlabel('AMRPA Layer'); ax4.set_ylabel('MFLOPs')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Memory or decay
    ax5 = fig.add_subplot(gs[1, 1])
    if config.use_cam:
        naive_mb, cam_mb = [], []
        for S in seq_lengths:
            cm = compute_cam_flops(d_k=config.d_k,
                                    proj_rank=config.cam.proj_rank,
                                    importance_hidden=config.cam.importance_hidden,
                                    n_amrpa_layers=config.n_amrpa_layers,
                                    window_size=config.cam.window_size, seq_len=S)
            naive_mb.append(cm['naive_storage_mb'])
            cam_mb.append(cm['cam_storage_mb'])
        ax5.plot(seq_lengths, naive_mb, 'o-', color='gray', linewidth=2, label='Naive')
        ax5.plot(seq_lengths, cam_mb,   's-', color='teal', linewidth=2, label='CAM')
        ax5.set_title('Memory: Naive vs CAM'); ax5.set_xlabel('Seq Length')
        ax5.set_ylabel('MB'); ax5.legend(); ax5.grid(True, alpha=0.3)
    else:
        steps = np.arange(1, 11)
        ax5.plot(steps, config.gamma**steps, 'o-', color='darkred', linewidth=2.5)
        ax5.fill_between(steps, 0, config.gamma**steps, alpha=0.3, color='red')
        ax5.set_title(f'Decay γ={config.gamma}')
        ax5.set_xlabel('Steps back'); ax5.set_ylabel('γ^k')
        ax5.grid(True, alpha=0.3)

    # 6. Key numbers table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    bl384 = compute_baseline_flops(config.d_model, config.n_heads, 384, 12)
    am384 = compute_amrpa_flops(d_model=config.d_model, d_k=config.d_model,
                                 d_mlp=config.d_mlp,
                                 n_amrpa_layers=config.n_amrpa_layers,
                                 gamma=config.gamma, seq_len=384)
    rows = [
        ['Baseline (seq=384)',  f"{bl384['total_gflops']:.4f} GFLOPs"],
        ['AMRPA extra',         f"+{am384['total_mflops']:.2f} MFLOPs"],
        ['AMRPA overhead',      f"+{(am384['total_mflops']/bl384['total_mflops'])*100:.2f}%"],
        ['AMRPA layers',        str(config.n_amrpa_layers)],
        ['Gamma',               str(config.gamma)],
    ]
    if config.use_cam:
        cm384 = compute_cam_flops(d_k=config.d_k,
                                   proj_rank=config.cam.proj_rank,
                                   importance_hidden=config.cam.importance_hidden,
                                   n_amrpa_layers=config.n_amrpa_layers,
                                   window_size=config.cam.window_size, seq_len=384)
        rows += [
            ['CAM extra',        f"+{cm384['total_mflops']:.2f} MFLOPs"],
            ['CAM memory saving', f"{cm384['memory_reduction_x']:.0f}x"],
            ['CAM storage',      f"{cm384['cam_storage_mb']:.3f} MB"],
        ]
    table = ax6.table(cellText=rows, colLabels=['Metric', 'Value'],
                      cellLoc='center', loc='center', bbox=[0, 0.1, 1, 0.85])
    table.auto_set_font_size(False); table.set_fontsize(9)
    ax6.set_title(f'Key Numbers — {label}', fontweight='bold')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ FLOPs chart saved: {save_path}")
    plt.close()
