"""
Experiment 5b: Selective Dispersion — Does the Model Create Its Own Frequency Structure?

Experiment 5 showed Maestro equalizes global phase velocity (ratio 1.013) while
Kerr flat and MLP have built-in dispersion (0.88, 0.86). The hypothesis:
Maestro provides a uniform baseline so the model can create TOKEN-SPECIFIC
dispersion — different tokens get different frequency profiles through depth.

Questions:
1. Do individual tokens show dispersion even when the global average is flat?
2. Do some tokens accelerate while others decelerate? (selective organization)
3. Do token dispersion patterns cluster? (semantic grouping by frequency profile)
4. Does Maestro have HIGHER per-token dispersion variance than Kerr/MLP?
   (uniform global average hiding rich per-token structure = the hypothesis)

Reuses training and extraction from depth_frequency.py.
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
    MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS, DEVICE, MAESTRO_DIM,
    PROG_STAGES, MAG_FREE_STEP, estimate_loss,
)

# Reuse from 5
from depth_frequency_5a import train_model, extract_layer_states, decompose_bands, phase_velocity


# ─── Per-token dispersion ratio ──────────────────────────────────────

def per_token_dispersion(delta):
    """
    For each token, compute its own High/Low band velocity ratio.

    Input: delta (n_transitions, tokens, n_bands)
    Output: ratios (tokens,) — each token's high/low band velocity ratio
    """
    # Mean absolute velocity per band per token, averaged across transitions
    token_band_velocity = np.abs(delta).mean(axis=0)  # (tokens, n_bands)

    low = token_band_velocity[:, :N_BANDS//2].mean(axis=1)   # (tokens,)
    high = token_band_velocity[:, N_BANDS//2:].mean(axis=1)  # (tokens,)

    # Avoid division by zero
    ratios = np.where(low > 1e-8, high / low, 1.0)
    return ratios, token_band_velocity


# ─── Per-token velocity profile clustering ───────────────────────────

def velocity_profile_clustering(token_band_velocity, n_clusters=5):
    """
    Cluster tokens by their band velocity profiles using simple k-means-like approach.
    No sklearn dependency — use iterative assignment.

    Input: token_band_velocity (tokens, n_bands)
    Returns: labels (tokens,), centroids (n_clusters, n_bands)
    """
    n_tokens, n_bands = token_band_velocity.shape

    # Normalize each token's profile to unit norm (compare shape, not scale)
    norms = np.linalg.norm(token_band_velocity, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    profiles = token_band_velocity / norms

    # Initialize centroids with evenly spaced tokens
    idx = np.linspace(0, n_tokens - 1, n_clusters, dtype=int)
    centroids = profiles[idx].copy()

    # 20 iterations of k-means
    for _ in range(20):
        # Assign each token to nearest centroid (cosine ~ euclidean on unit vectors)
        dists = np.zeros((n_tokens, n_clusters))
        for c in range(n_clusters):
            dists[:, c] = np.linalg.norm(profiles - centroids[c], axis=1)
        labels = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for c in range(n_clusters):
            mask = labels == c
            if mask.sum() > 0:
                new_centroids[c] = profiles[mask].mean(axis=0)
                norm = np.linalg.norm(new_centroids[c])
                if norm > 1e-8:
                    new_centroids[c] /= norm
            else:
                new_centroids[c] = centroids[c]
        centroids = new_centroids

    return labels, centroids


# ─── Per-token per-transition velocity ───────────────────────────────

def token_transition_profiles(delta):
    """
    For each token, get its velocity at each transition (averaged across bands).
    Shows if tokens accelerate/decelerate selectively.

    Input: delta (n_transitions, tokens, n_bands)
    Output: profiles (tokens, n_transitions)
    """
    return np.abs(delta).mean(axis=2).T  # (tokens, n_transitions)


# ─── Acceleration diversity ──────────────────────────────────────────

def acceleration_diversity(delta):
    """
    For each token, compute whether it accelerates or decelerates through depth.
    Returns per-token acceleration (positive = speeds up, negative = slows down).

    Input: delta (n_transitions, tokens, n_bands)
    Output: accel (tokens,) — mean acceleration per token
    """
    # Velocity per transition per token (averaged across bands)
    v_per_transition = np.abs(delta).mean(axis=2)  # (n_transitions, tokens)

    if v_per_transition.shape[0] < 2:
        return np.zeros(v_per_transition.shape[1])

    # Acceleration = diff of velocity across transitions
    accel = np.diff(v_per_transition, axis=0)  # (n_transitions-1, tokens)
    return accel.mean(axis=0)  # (tokens,) — mean acceleration


# ─── Analysis ────────────────────────────────────────────────────────

def analyse_selective_dispersion(states, label=""):
    """Full 5b analysis: does the model create per-token frequency structure?"""
    magnitude, phase = decompose_bands(states)
    n_layers = states.shape[0]
    n_tokens = states.shape[1]
    layer_names = ["Emb"] + [f"L{i}" for i in range(n_layers - 1)]
    transition_names = [f"{layer_names[l]}->{layer_names[l+1]}" for l in range(n_layers - 1)]

    delta = phase_velocity(phase)

    print(f"\n{'='*70}")
    print(f"  Experiment 5b: Selective Dispersion Analysis: {label}")
    print(f"  {n_tokens} tokens, {N_BANDS} bands, {n_layers-1} transitions")
    print(f"{'='*70}")

    # 1. Per-token dispersion ratios
    ratios, token_band_vel = per_token_dispersion(delta)
    print(f"\n  --- Per-token High/Low band velocity ratio ---")
    print(f"  Global ratio (from Exp 5):  {token_band_vel[:, N_BANDS//2:].mean() / token_band_vel[:, :N_BANDS//2].mean():.4f}")
    print(f"  Mean per-token ratio:       {ratios.mean():.4f}")
    print(f"  Std per-token ratio:        {ratios.std():.4f}")
    print(f"  CV of per-token ratio:      {ratios.std() / ratios.mean() * 100:.1f}%")
    q5, q25, q50, q75, q95 = np.percentile(ratios, [5, 25, 50, 75, 95])
    print(f"  Percentiles:  5%={q5:.3f}  25%={q25:.3f}  50%={q50:.3f}  75%={q75:.3f}  95%={q95:.3f}")

    # How many tokens have inverted dispersion (high > low)?
    n_inverted = (ratios > 1.0).sum()
    pct_inverted = n_inverted / len(ratios) * 100
    print(f"  Tokens with high > low:     {n_inverted}/{len(ratios)} ({pct_inverted:.1f}%)")

    # 2. Acceleration diversity — do tokens accelerate or decelerate?
    accel = acceleration_diversity(delta)
    print(f"\n  --- Per-token acceleration (positive = speeds up through depth) ---")
    print(f"  Mean acceleration:   {accel.mean():+.4f}")
    print(f"  Std acceleration:    {accel.std():.4f}")
    print(f"  CV:                  {accel.std() / (abs(accel.mean()) + 1e-8) * 100:.1f}%")
    n_accel = (accel > 0).sum()
    n_decel = (accel < 0).sum()
    print(f"  Accelerating tokens: {n_accel} ({n_accel/len(accel)*100:.1f}%)")
    print(f"  Decelerating tokens: {n_decel} ({n_decel/len(accel)*100:.1f}%)")

    # 3. Per-token per-transition velocity profiles
    profiles = token_transition_profiles(delta)  # (tokens, n_transitions)
    print(f"\n  --- Per-token velocity at each transition ---")
    print(f"  {'Transition':<12} {'Mean':>8} {'Std':>8} {'CV':>8} {'Min':>8} {'Max':>8}")
    for t in range(profiles.shape[1]):
        v = profiles[:, t]
        print(f"  {transition_names[t]:<12} {v.mean():>8.4f} {v.std():>8.4f} "
              f"{v.std()/v.mean()*100:>7.1f}% {v.min():>8.4f} {v.max():>8.4f}")

    # 4. Token velocity profile shape diversity
    # Normalize each token's transition profile, then measure spread
    profile_norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    profile_norms = np.where(profile_norms > 1e-8, profile_norms, 1.0)
    normed = profiles / profile_norms

    # Pairwise cosine similarity of a random sample (full matrix too big)
    n_sample = min(2000, n_tokens)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_tokens, n_sample, replace=False)
    sample = normed[sample_idx]

    # Mean pairwise cosine sim
    cos_sim_matrix = sample @ sample.T
    # Upper triangle only
    triu_idx = np.triu_indices(n_sample, k=1)
    cos_sims = cos_sim_matrix[triu_idx]

    print(f"\n  --- Transition profile shape diversity (cosine similarity, {n_sample} token sample) ---")
    print(f"  Mean cosine sim:    {cos_sims.mean():.4f}")
    print(f"  Std cosine sim:     {cos_sims.std():.4f}")
    print(f"  Min cosine sim:     {cos_sims.min():.4f}")
    print(f"  (1.0 = all tokens have identical velocity shape; lower = more diverse)")

    # 5. Velocity profile clustering
    print(f"\n  --- Velocity profile clusters (5 clusters on band velocity shape) ---")
    labels, centroids = velocity_profile_clustering(token_band_vel, n_clusters=5)

    for c in range(5):
        mask = labels == c
        n_in = mask.sum()
        cluster_ratios = ratios[mask]
        cluster_vel = token_band_vel[mask].mean(axis=0)
        low_v = cluster_vel[:N_BANDS//2].mean()
        high_v = cluster_vel[N_BANDS//2:].mean()
        cr = high_v / low_v if low_v > 1e-8 else float('inf')
        print(f"  Cluster {c}: {n_in:>6} tokens ({n_in/n_tokens*100:>5.1f}%) | "
              f"H/L ratio={cr:.3f} | mean |dp|={token_band_vel[mask].mean():.4f} | "
              f"dispersion ratio range: [{cluster_ratios.min():.3f}, {cluster_ratios.max():.3f}]")

    # 6. Band-level per-token velocity variance
    # For each band, how much do tokens differ in velocity?
    print(f"\n  --- Per-band token velocity variance (do some bands let tokens diverge more?) ---")
    band_token_var = token_band_vel.var(axis=0)  # (n_bands,)
    band_token_mean = token_band_vel.mean(axis=0)  # (n_bands,)
    band_cv = np.where(band_token_mean > 1e-8,
                       np.sqrt(band_token_var) / band_token_mean * 100, 0)

    print(f"  {'Bands':<12} {'Mean vel':>10} {'Token CV':>10}")
    for g in range(0, N_BANDS, 8):
        gv = band_token_mean[g:g+8].mean()
        gcv = band_cv[g:g+8].mean()
        print(f"  {g+1:>2}-{min(g+8, N_BANDS):<8} {gv:>10.4f} {gcv:>9.1f}%")

    # Low vs high band token CV
    low_cv = band_cv[:N_BANDS//2].mean()
    high_cv = band_cv[N_BANDS//2:].mean()
    print(f"\n  Low bands (1-{N_BANDS//2}) token CV:  {low_cv:.1f}%")
    print(f"  High bands ({N_BANDS//2+1}-{N_BANDS}) token CV: {high_cv:.1f}%")
    print(f"  (Higher = more token-level differentiation in that band group)")

    return {
        'dispersion_ratios': ratios,
        'acceleration': accel,
        'profiles': profiles,
        'cos_sims': cos_sims,
        'cluster_labels': labels,
        'band_token_cv': band_cv,
        'token_band_velocity': token_band_vel,
    }


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Experiment 5b: Selective Dispersion")
    print("  Does the model create its own frequency structure from a uniform base?")
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

        result = analyse_selective_dispersion(states, label=label)
        results[label] = result

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Cross-model comparison
    print(f"\n{'='*70}")
    print(f"  CROSS-MODEL COMPARISON: SELECTIVE DISPERSION")
    print(f"{'='*70}")

    print(f"\n  --- THE KEY QUESTION: Does Maestro have more per-token dispersion variety? ---")
    print(f"  {'Model':<28} {'Global H/L':>10} {'Per-token H/L CV':>18} {'Accel std':>12}")
    for label, _ in configs:
        r = results[label]
        short = label.split("(")[0].strip()
        global_hl = r['token_band_velocity'][:, N_BANDS//2:].mean() / r['token_band_velocity'][:, :N_BANDS//2].mean()
        ptcv = r['dispersion_ratios'].std() / r['dispersion_ratios'].mean() * 100
        print(f"  {short:<28} {global_hl:>10.3f} {ptcv:>17.1f}% {r['acceleration'].std():>12.4f}")

    print(f"\n  If Maestro has HIGHER per-token H/L CV and/or acceleration std,")
    print(f"  that means it creates more diverse token-specific frequency structures")
    print(f"  from its globally uniform baseline. The model organizes frequencies itself.")

    print(f"\n  --- Transition profile diversity (cosine similarity) ---")
    print(f"  {'Model':<28} {'Mean cos sim':>14} {'Std':>8}")
    for label, _ in configs:
        r = results[label]
        short = label.split("(")[0].strip()
        print(f"  {short:<28} {r['cos_sims'].mean():>14.4f} {r['cos_sims'].std():>8.4f}")

    print(f"\n  Lower mean cosine sim = more diverse velocity profiles across tokens.")
    print(f"  If Maestro is lower, tokens take more different paths through depth.")

    print(f"\n  --- Per-band token differentiation ---")
    print(f"  {'Model':<28} {'Low band CV':>12} {'High band CV':>14} {'Ratio':>8}")
    for label, _ in configs:
        r = results[label]
        short = label.split("(")[0].strip()
        low_cv = r['band_token_cv'][:N_BANDS//2].mean()
        high_cv = r['band_token_cv'][N_BANDS//2:].mean()
        ratio = high_cv / low_cv if low_cv > 0 else float('inf')
        print(f"  {short:<28} {low_cv:>11.1f}% {high_cv:>13.1f}% {ratio:>8.3f}")

    print(f"\n  Higher token CV per band = more token differentiation in that band group.")
    print(f"  If Maestro has higher CV despite uniform global velocity, the model is")
    print(f"  using the uniform substrate to create richer per-token frequency structure.")

    # Accelerating vs decelerating split
    print(f"\n  --- Acceleration/deceleration split ---")
    print(f"  {'Model':<28} {'% accel':>10} {'% decel':>10} {'Balance':>10}")
    for label, _ in configs:
        r = results[label]
        short = label.split("(")[0].strip()
        n = len(r['acceleration'])
        pct_a = (r['acceleration'] > 0).sum() / n * 100
        pct_d = (r['acceleration'] < 0).sum() / n * 100
        balance = abs(pct_a - pct_d)
        print(f"  {short:<28} {pct_a:>9.1f}% {pct_d:>9.1f}% {balance:>9.1f}%")

    print(f"\n  50/50 split = model uses both strategies equally (maximum diversity)")
    print(f"  Skewed split = model prefers one depth-velocity pattern for most tokens")

    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    # Auto-verdict based on data
    maestro_r = results["Maestro + curriculum (best)"]
    kerr_r = results["Kerr flat (no maestro)"]
    mlp_r = results["MLP baseline"]

    m_ptcv = maestro_r['dispersion_ratios'].std() / maestro_r['dispersion_ratios'].mean()
    k_ptcv = kerr_r['dispersion_ratios'].std() / kerr_r['dispersion_ratios'].mean()
    ml_ptcv = mlp_r['dispersion_ratios'].std() / mlp_r['dispersion_ratios'].mean()

    if m_ptcv > k_ptcv and m_ptcv > ml_ptcv:
        print(f"  CONFIRMED: Maestro has highest per-token dispersion diversity")
        print(f"  ({m_ptcv*100:.1f}% vs Kerr {k_ptcv*100:.1f}%, MLP {ml_ptcv*100:.1f}%)")
        print(f"  The model creates its own frequency structure from the uniform base.")
    elif m_ptcv < k_ptcv and m_ptcv < ml_ptcv:
        print(f"  REFUTED: Maestro has LOWEST per-token dispersion diversity")
        print(f"  ({m_ptcv*100:.1f}% vs Kerr {k_ptcv*100:.1f}%, MLP {ml_ptcv*100:.1f}%)")
        print(f"  Uniform global = uniform per-token. No selective organization.")
    else:
        print(f"  MIXED: Maestro per-token CV = {m_ptcv*100:.1f}%,")
        print(f"         Kerr = {k_ptcv*100:.1f}%, MLP = {ml_ptcv*100:.1f}%")
        print(f"  Partial support — check individual metrics above.")

    m_cos = maestro_r['cos_sims'].mean()
    k_cos = kerr_r['cos_sims'].mean()
    ml_cos = mlp_r['cos_sims'].mean()

    if m_cos < k_cos and m_cos < ml_cos:
        print(f"\n  BONUS: Maestro tokens take the most diverse paths through depth")
        print(f"  (cos sim {m_cos:.4f} vs Kerr {k_cos:.4f}, MLP {ml_cos:.4f})")
    elif m_cos > k_cos and m_cos > ml_cos:
        print(f"\n  NOTE: Maestro tokens are MORE similar in transition shape")
        print(f"  (cos sim {m_cos:.4f} vs Kerr {k_cos:.4f}, MLP {ml_cos:.4f})")
        print(f"  Uniform global AND uniform per-token transition patterns.")

    print(f"{'='*70}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  Total time: {time.time() - t0:.1f}s")
