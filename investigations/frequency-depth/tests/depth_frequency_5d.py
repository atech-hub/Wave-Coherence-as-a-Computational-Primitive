"""
Experiment 5d: Maestro Tempo Control

Two questions:
1. Does wider bottleneck = tighter velocity equalization?
   Test MAESTRO_DIM = 4, 8, 16 (default), 32, 64
2. Can we control the maestro's "tempo" by scaling its output?
   Test tempo scalars: 0.25, 0.5, 1.0 (default), 2.0, 4.0

Measures: H/L phase velocity ratio, val loss, per-band velocity profile.
"""

import math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add experiments dir and local tests dir for imports
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', '..', 'experiments'))
sys.path.insert(0, _here)

from phaseC_integrated import (
    GPT, Dataset, download_shakespeare, build_harmonic_table, build_positional_table,
    N_BANDS, N_EMBD, N_HEAD, BLOCK_SIZE, BATCH_SIZE, LEARNING_RATE,
    MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS, DEVICE,
    PROG_STAGES, MAG_FREE_STEP, estimate_loss, KerrODE,
)
from depth_frequency_5a import extract_layer_states, decompose_bands, phase_velocity


# ─── Modified Maestro with configurable dim and tempo ────────────────

class MaestroTempo(nn.Module):
    def __init__(self, dim=16, tempo=1.0):
        super().__init__()
        self.squeeze = nn.Linear(N_EMBD, dim)
        self.process = nn.Sequential(nn.GELU(), nn.Linear(dim, N_EMBD))
        self.tempo = tempo

    def forward(self, x_flat):
        return self.tempo * self.process(self.squeeze(x_flat))


class KerrMaestroTempo(nn.Module):
    def __init__(self, maestro_dim=16, tempo=1.0):
        super().__init__()
        self.kerr = KerrODE(n_steps=8)
        self.maestro = MaestroTempo(dim=maestro_dim, tempo=tempo)
        self.out_proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(B*T, C)
        kerr_out = self.kerr(x_flat.view(B*T, N_BANDS, 2)).reshape(B*T, C)
        return self.out_proj(kerr_out + self.maestro(x_flat)).view(B, T, C)


# ─── Monkey-patch GPT to use our custom FFN ──────────────────────────

def build_custom_model(dataset, maestro_dim=16, tempo=1.0):
    """Build a GPT with custom maestro dim and tempo."""
    model = GPT(dataset.vocab_size, mode="kerr", use_maestro=True, use_mag=False).to(DEVICE)

    # Replace each block's FFN with our custom version
    for block in model.blocks:
        block.ffn = KerrMaestroTempo(maestro_dim=maestro_dim, tempo=tempo).to(DEVICE)

    return model


# ─── Train and measure ───────────────────────────────────────────────

def train_and_measure(dataset, maestro_dim=16, tempo=1.0, label=""):
    """Train a model with given maestro config, measure depth-axis velocity."""
    torch.manual_seed(42)
    model = build_custom_model(dataset, maestro_dim=maestro_dim, tempo=tempo)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Count maestro params
    maestro_params = sum(
        p.numel() for block in model.blocks
        for p in block.ffn.maestro.parameters()
    )

    print(f"\n  --- {label} (dim={maestro_dim}, tempo={tempo}) ---")
    print(f"  Maestro params per layer: {maestro_params // len(model.blocks)}")

    for i in range(MAX_ITERS):
        # Curriculum
        for step_thresh, nb in PROG_STAGES:
            if i >= step_thresh:
                model.n_bands_active = nb

        if i % 500 == 0:
            losses = estimate_loss(model, dataset)
            print(f"    step {i:>5} | val {losses['val']:.4f}")

        x, y = dataset.get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.n_bands_active = N_BANDS
    final = estimate_loss(model, dataset)
    val_loss = final['val']
    print(f"    Done. Val: {val_loss:.4f}")

    # Extract depth-axis frequency
    states = extract_layer_states(model, dataset, n_batches=5)
    magnitude, phase = decompose_bands(states)
    delta = phase_velocity(phase)
    per_band_vel = np.abs(delta).mean(axis=(0, 1))  # (n_bands,)

    low = per_band_vel[:N_BANDS//2].mean()
    high = per_band_vel[N_BANDS//2:].mean()
    ratio = high / low if low > 0 else float('inf')
    mean_vel = per_band_vel.mean()
    vel_std = per_band_vel.std()
    vel_cv = vel_std / mean_vel * 100 if mean_vel > 0 else 0

    print(f"    H/L ratio: {ratio:.4f}  Mean vel: {mean_vel:.4f}  Vel CV: {vel_cv:.1f}%")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        'val_loss': val_loss,
        'ratio': ratio,
        'mean_vel': mean_vel,
        'vel_cv': vel_cv,
        'per_band_vel': per_band_vel,
        'low_vel': low,
        'high_vel': high,
        'maestro_params_per_layer': maestro_params // 4,
    }


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Experiment 5d: Maestro Tempo Control")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"  Dataset: {len(text)} chars, {dataset.vocab_size} vocab")

    results = {}

    # Part 1: Bottleneck width sweep (tempo = 1.0)
    print(f"\n{'='*70}")
    print(f"  PART 1: Bottleneck Width Sweep (tempo=1.0)")
    print(f"{'='*70}")

    widths = [4, 8, 16, 32, 64]
    for dim in widths:
        label = f"dim={dim}"
        r = train_and_measure(dataset, maestro_dim=dim, tempo=1.0, label=label)
        results[f"width_{dim}"] = r

    # Part 2: Tempo scalar sweep (dim = 16)
    print(f"\n{'='*70}")
    print(f"  PART 2: Tempo Scalar Sweep (dim=16)")
    print(f"{'='*70}")

    tempos = [0.25, 0.5, 1.0, 2.0, 4.0]
    for tempo in tempos:
        label = f"tempo={tempo}"
        # dim=16 at tempo=1.0 already run above, reuse
        if tempo == 1.0:
            results[f"tempo_{tempo}"] = results["width_16"]
            print(f"\n  --- tempo=1.0 — reusing width_16 result ---")
            print(f"    Val: {results['width_16']['val_loss']:.4f}  H/L: {results['width_16']['ratio']:.4f}")
            continue
        r = train_and_measure(dataset, maestro_dim=16, tempo=tempo, label=label)
        results[f"tempo_{tempo}"] = r

    # Summary tables
    print(f"\n{'='*70}")
    print(f"  RESULTS: Bottleneck Width")
    print(f"{'='*70}")
    print(f"  {'Dim':>6} {'Params/L':>10} {'Val Loss':>10} {'H/L Ratio':>10} {'Mean Vel':>10} {'Vel CV':>8}")
    for dim in widths:
        r = results[f"width_{dim}"]
        print(f"  {dim:>6} {r['maestro_params_per_layer']:>10} {r['val_loss']:>10.4f} "
              f"{r['ratio']:>10.4f} {r['mean_vel']:>10.4f} {r['vel_cv']:>7.1f}%")

    print(f"\n  Prediction: wider bottleneck = ratio closer to 1.0 (tighter sync)")
    ratios_by_width = [results[f"width_{d}"]['ratio'] for d in widths]
    if all(ratios_by_width[i] >= ratios_by_width[i+1] - 0.01 for i in range(len(ratios_by_width)-1)):
        # Monotonically approaching 1.0 (for ratios > 1) or away from sub-1
        trend = "CONFIRMED" if abs(ratios_by_width[-1] - 1.0) < abs(ratios_by_width[0] - 1.0) else "MIXED"
    else:
        trend = "NOT monotonic"
    closest_to_1 = min(widths, key=lambda d: abs(results[f"width_{d}"]['ratio'] - 1.0))
    print(f"  Closest to 1.0: dim={closest_to_1} (ratio={results[f'width_{closest_to_1}']['ratio']:.4f})")
    print(f"  Trend: {trend}")

    print(f"\n{'='*70}")
    print(f"  RESULTS: Tempo Scalar")
    print(f"{'='*70}")
    print(f"  {'Tempo':>6} {'Val Loss':>10} {'H/L Ratio':>10} {'Mean Vel':>10} {'Vel CV':>8}")
    for tempo in tempos:
        r = results[f"tempo_{tempo}"]
        print(f"  {tempo:>6.2f} {r['val_loss']:>10.4f} {r['ratio']:>10.4f} "
              f"{r['mean_vel']:>10.4f} {r['vel_cv']:>7.1f}%")

    print(f"\n  Key question: does tempo scalar change the H/L ratio?")
    print(f"  - If yes: maestro signal strength directly controls synchronization")
    print(f"  - If no: the ratio is emergent from training, not from signal amplitude")

    # Does tempo affect mean velocity?
    vel_by_tempo = [results[f"tempo_{t}"]['mean_vel'] for t in tempos]
    tempo_vel_corr = np.corrcoef(tempos, vel_by_tempo)[0, 1]
    print(f"\n  Correlation(tempo, mean_velocity): {tempo_vel_corr:.3f}")
    if abs(tempo_vel_corr) > 0.7:
        direction = "faster" if tempo_vel_corr > 0 else "slower"
        print(f"  -> Higher tempo = {direction} overall phase velocity")
    else:
        print(f"  -> Tempo does not strongly control overall velocity")

    # Does tempo affect ratio?
    ratio_by_tempo = [results[f"tempo_{t}"]['ratio'] for t in tempos]
    tempo_ratio_corr = np.corrcoef(tempos, ratio_by_tempo)[0, 1]
    print(f"  Correlation(tempo, H/L_ratio): {tempo_ratio_corr:.3f}")
    if abs(tempo_ratio_corr) > 0.7:
        print(f"  -> Tempo directly controls spectral equalization")
    else:
        print(f"  -> Spectral equalization is emergent, not tempo-driven")

    # Best val loss
    best_width = min(widths, key=lambda d: results[f"width_{d}"]['val_loss'])
    best_tempo = min(tempos, key=lambda t: results[f"tempo_{t}"]['val_loss'])
    print(f"\n  Best val loss by width: dim={best_width} ({results[f'width_{best_width}']['val_loss']:.4f})")
    print(f"  Best val loss by tempo: tempo={best_tempo} ({results[f'tempo_{best_tempo}']['val_loss']:.4f})")

    # Per-band velocity profiles for extreme configs
    print(f"\n{'='*70}")
    print(f"  PER-BAND VELOCITY PROFILES (selected configs)")
    print(f"{'='*70}")
    show_configs = [f"width_{widths[0]}", f"width_{widths[-1]}", f"tempo_{tempos[0]}", f"tempo_{tempos[-1]}"]
    print(f"  {'Bands':<10}", end="")
    for cfg in show_configs:
        print(f"  {cfg:>14}", end="")
    print()
    for g in range(0, N_BANDS, 8):
        print(f"  {g+1:>2}-{min(g+8, N_BANDS):<6}", end="")
        for cfg in show_configs:
            v = results[cfg]['per_band_vel'][g:g+8].mean()
            print(f"  {v:>14.4f}", end="")
        print()

    print(f"\n{'='*70}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  Total time: {time.time() - t0:.1f}s")
