"""
Experiment 5: Depth-Axis Frequency — Does the Model Oscillate Through Layers?

Track how token hidden states change across layers. Extract per-band phase
and magnitude at each depth. Measure phase velocity (delta-phase per layer)
to see if the model has internal frequency structure through depth.

Questions:
1. Does phase advance linearly (constant frequency) or nonlinearly (acceleration)?
2. Do different bands have different depth-frequencies (spectral dispersion)?
3. Do different tokens oscillate at different rates (connects to commitment point)?
4. Does Kerr+Maestro show different depth dynamics than plain Kerr or MLP?

No new training needed — uses a freshly trained model from phaseC_integrated.
"""

import math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add experiments dir for imports
# Add experiments dir and local tests dir for imports
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', '..', 'experiments'))
sys.path.insert(0, _here)

from phaseC_integrated import (
    GPT, Dataset, download_shakespeare, build_harmonic_table, build_positional_table,
    N_BANDS, N_EMBD, N_HEAD, BLOCK_SIZE, BATCH_SIZE, LEARNING_RATE,
    MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS, DEVICE, MAESTRO_DIM,
    PROG_STAGES, MAG_FREE_STEP, estimate_loss,
)

# ─── Train a model (or reuse) ───────────────────────────────────────

def train_model(dataset, mode="kerr", use_maestro=True, curriculum=True):
    """Train a model and return it. Maestro+curriculum by default (best config)."""
    torch.manual_seed(42)
    model = GPT(dataset.vocab_size, mode=mode, use_maestro=use_maestro, use_mag=False).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"  Training {mode} (maestro={use_maestro}, curriculum={curriculum})...")
    for i in range(MAX_ITERS):
        if curriculum and mode != "mlp":
            for step_thresh, nb in PROG_STAGES:
                if i >= step_thresh:
                    model.n_bands_active = nb
        else:
            model.n_bands_active = N_BANDS

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
    print(f"    Done. Final val: {final['val']:.4f}")
    return model


# ─── Extract hidden states at each layer ─────────────────────────────

@torch.no_grad()
def extract_layer_states(model, dataset, n_batches=5):
    """
    Run forward passes and capture hidden states after each block.
    Returns: tensor of shape (n_layers+1, total_tokens, n_embd)
      - Layer 0 = embedding output (before any block)
      - Layer 1..4 = after each block
    """
    model.eval()
    n_layers = len(model.blocks)
    all_states = [[] for _ in range(n_layers + 1)]

    for _ in range(n_batches):
        x, _ = dataset.get_batch("val")
        B, T = x.size()

        # Embedding
        h = model._get_embeddings(x) + model.wpe[:T]
        all_states[0].append(h.reshape(-1, N_EMBD).cpu())

        # Each block
        for i, block in enumerate(model.blocks):
            h = block(h)
            all_states[i + 1].append(h.reshape(-1, N_EMBD).cpu())

    # Concatenate across batches: (total_tokens, n_embd) per layer
    states = torch.stack([torch.cat(s, dim=0) for s in all_states])  # (n_layers+1, tokens, embd)
    model.train()
    return states


# ─── Decompose into per-band phase and magnitude ────────────────────

def decompose_bands(states):
    """
    Convert hidden states to per-band complex representation.

    Input: (n_layers, tokens, n_embd)
    Output: magnitude (n_layers, tokens, n_bands), phase (n_layers, tokens, n_bands)
    """
    n_layers, n_tokens, _ = states.shape
    # Reshape to (n_layers, tokens, n_bands, 2) — pairs of (cos, sin)
    bands = states.view(n_layers, n_tokens, N_BANDS, 2)
    r = bands[:, :, :, 0]  # cos component
    s = bands[:, :, :, 1]  # sin component

    magnitude = torch.sqrt(r**2 + s**2)
    phase = torch.atan2(s, r)  # range [-pi, pi]

    return magnitude, phase


# ─── Measure phase velocity through depth ────────────────────────────

def phase_velocity(phase):
    """
    Compute phase change per layer for each band.

    Input: phase (n_layers, tokens, n_bands)
    Output: delta_phase (n_layers-1, tokens, n_bands) — phase change per layer step
    """
    # Unwrap phase to handle -pi/pi wraparound
    phase_np = phase.numpy()
    # Unwrap along the layer axis (axis=0)
    unwrapped = np.unwrap(phase_np, axis=0)
    # Diff along layer axis
    delta = np.diff(unwrapped, axis=0)  # (n_layers-1, tokens, n_bands)
    return delta


# ─── Analysis ────────────────────────────────────────────────────────

def analyse_depth_frequency(states, label=""):
    """Full analysis of depth-axis frequency structure."""
    magnitude, phase = decompose_bands(states)
    n_layers = states.shape[0]
    layer_names = ["Emb"] + [f"L{i}" for i in range(n_layers - 1)]

    print(f"\n{'='*70}")
    print(f"  Depth-Axis Frequency Analysis: {label}")
    print(f"  {states.shape[0]} depth points, {states.shape[1]} tokens, {N_BANDS} bands")
    print(f"{'='*70}")

    # 1. Magnitude evolution through depth
    print(f"\n  --- Magnitude per layer (mean across tokens) ---")
    print(f"  {'Layer':<6} {'Mean |Z|':>10} {'Std |Z|':>10} {'Min band':>10} {'Max band':>10}")
    for l in range(n_layers):
        mag_l = magnitude[l]  # (tokens, bands)
        mean_per_band = mag_l.mean(dim=0)  # (bands,)
        print(f"  {layer_names[l]:<6} {mean_per_band.mean():>10.4f} {mean_per_band.std():>10.4f} "
              f"{mean_per_band.min():>10.4f} {mean_per_band.max():>10.4f}")

    # 2. Phase velocity — the core measurement
    delta = phase_velocity(phase)  # (n_layers-1, tokens, n_bands)
    print(f"\n  --- Phase velocity (radians per layer) ---")
    print(f"  {'Transition':<12} {'Mean |dp|':>10} {'Std |dp|':>10} {'Median |dp|':>12}")
    for l in range(n_layers - 1):
        d = np.abs(delta[l])  # (tokens, bands)
        print(f"  {layer_names[l]}->{layer_names[l+1]:<8} {d.mean():>10.4f} {d.std():>10.4f} "
              f"{np.median(d):>12.4f}")

    # 3. Per-band phase velocity — spectral dispersion
    print(f"\n  --- Per-band mean phase velocity (|dp|, averaged across tokens and transitions) ---")
    mean_delta_per_band = np.abs(delta).mean(axis=(0, 1))  # (n_bands,)
    # Show in groups of 8
    print(f"  {'Bands':<12} {'Mean |dp|':>10} {'Std across bands in group':>26}")
    for g in range(0, N_BANDS, 8):
        group = mean_delta_per_band[g:g+8]
        print(f"  {g+1:>2}-{min(g+8, N_BANDS):<3}       {group.mean():>10.4f} {group.std():>26.4f}")

    # Low vs high band comparison (Phase 9b split)
    low_bands = mean_delta_per_band[:N_BANDS//2]
    high_bands = mean_delta_per_band[N_BANDS//2:]
    ratio = high_bands.mean() / low_bands.mean() if low_bands.mean() > 0 else float('inf')
    print(f"\n  Low bands (1-{N_BANDS//2}) mean |dp|:  {low_bands.mean():.4f}")
    print(f"  High bands ({N_BANDS//2+1}-{N_BANDS}) mean |dp|: {high_bands.mean():.4f}")
    print(f"  High/Low ratio: {ratio:.3f}")

    # 4. Phase acceleration — is velocity constant or changing?
    if n_layers > 2:
        print(f"\n  --- Phase acceleration (change in velocity between transitions) ---")
        for l in range(n_layers - 2):
            v1 = np.abs(delta[l]).mean()
            v2 = np.abs(delta[l + 1]).mean()
            accel = v2 - v1
            pct = (v2 / v1 - 1) * 100 if v1 > 0 else float('inf')
            print(f"  {layer_names[l+1]}->{layer_names[l+2]} vs {layer_names[l]}->{layer_names[l+1]}: "
                  f"dv = {accel:+.4f} ({pct:+.1f}%)")

    # 5. Token-level variation — do different tokens oscillate differently?
    token_mean_velocity = np.abs(delta).mean(axis=(0, 2))  # mean across transitions and bands per token
    print(f"\n  --- Token-level phase velocity distribution ---")
    print(f"  Mean across tokens: {token_mean_velocity.mean():.4f}")
    print(f"  Std across tokens:  {token_mean_velocity.std():.4f}")
    print(f"  CV:                 {token_mean_velocity.std() / token_mean_velocity.mean() * 100:.1f}%")
    # Quartiles
    q25, q50, q75 = np.percentile(token_mean_velocity, [25, 50, 75])
    print(f"  Quartiles:          Q25={q25:.4f}  Q50={q50:.4f}  Q75={q75:.4f}")
    print(f"  Range:              {token_mean_velocity.min():.4f} — {token_mean_velocity.max():.4f}")

    # 6. Band-transition correlation matrix
    # For each transition, correlate phase velocity across bands
    print(f"\n  --- Cross-transition correlation (do bands that move fast at L0->L1 also move fast at L1->L2?) ---")
    if n_layers > 2:
        for l1 in range(n_layers - 2):
            for l2 in range(l1 + 1, n_layers - 1):
                v1 = np.abs(delta[l1]).mean(axis=0)  # (bands,)
                v2 = np.abs(delta[l2]).mean(axis=0)  # (bands,)
                corr = np.corrcoef(v1, v2)[0, 1]
                print(f"  {layer_names[l1]}->{layer_names[l1+1]} vs "
                      f"{layer_names[l2]}->{layer_names[l2+1]}: r = {corr:.3f}")

    return {
        'magnitude': magnitude,
        'phase': phase,
        'delta_phase': delta,
        'per_band_velocity': mean_delta_per_band,
        'token_velocity': token_mean_velocity,
    }


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Experiment 5: Depth-Axis Frequency")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"  Dataset: {len(text)} chars, {dataset.vocab_size} vocab\n")

    configs = [
        ("Maestro + curriculum (best)", dict(mode="kerr", use_maestro=True, curriculum=True)),
        ("Kerr flat (no maestro)", dict(mode="kerr", use_maestro=False, curriculum=False)),
        ("MLP baseline", dict(mode="mlp", use_maestro=False, curriculum=False)),
    ]

    results = {}
    for label, kwargs in configs:
        print(f"\n  === {label} ===")
        model = train_model(dataset, **kwargs)

        print(f"  Extracting hidden states...")
        states = extract_layer_states(model, dataset, n_batches=5)
        print(f"  States shape: {states.shape}")

        result = analyse_depth_frequency(states, label=label)
        results[label] = result

        # Free GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Cross-model comparison
    print(f"\n{'='*70}")
    print(f"  CROSS-MODEL COMPARISON")
    print(f"{'='*70}")

    print(f"\n  --- Per-band velocity profiles ---")
    print(f"  {'Band group':<12}", end="")
    for label, _ in configs:
        short = label.split("(")[0].strip()
        print(f"  {short:>16}", end="")
    print()

    for g in range(0, N_BANDS, 8):
        print(f"  {g+1:>2}-{min(g+8, N_BANDS):<8}", end="")
        for label, _ in configs:
            v = results[label]['per_band_velocity'][g:g+8].mean()
            print(f"  {v:>16.4f}", end="")
        print()

    print(f"\n  --- Low/High band ratio ---")
    for label, _ in configs:
        low = results[label]['per_band_velocity'][:N_BANDS//2].mean()
        high = results[label]['per_band_velocity'][N_BANDS//2:].mean()
        ratio = high / low if low > 0 else float('inf')
        short = label.split("(")[0].strip()
        print(f"  {short:<25} Low={low:.4f}  High={high:.4f}  Ratio={ratio:.3f}")

    print(f"\n  --- Token velocity CV (how much tokens differ from each other) ---")
    for label, _ in configs:
        tv = results[label]['token_velocity']
        cv = tv.std() / tv.mean() * 100
        short = label.split("(")[0].strip()
        print(f"  {short:<25} CV = {cv:.1f}%")

    print(f"\n{'='*70}")
    print(f"  INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print(f"  - Phase velocity = how much phase changes per layer (radians)")
    print(f"  - Constant velocity across transitions = constant-frequency oscillator")
    print(f"  - Accelerating velocity = deeper layers compute faster")
    print(f"  - High/Low band ratio > 1 = high bands change faster (spectral dispersion)")
    print(f"  - High/Low band ratio = 1 = all bands move at same rate (no dispersion)")
    print(f"  - High token CV = different tokens oscillate at different rates")
    print(f"  - Cross-transition correlation near 1 = same bands are always fast/slow")
    print(f"  - Cross-transition correlation near 0 = which bands are fast changes per layer")
    print(f"{'='*70}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  Total time: {time.time() - t0:.1f}s")
