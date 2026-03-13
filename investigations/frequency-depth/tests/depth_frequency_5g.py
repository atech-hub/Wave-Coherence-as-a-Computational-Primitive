"""
Experiment 5g: Do Band Roles Reassign Through Depth?

5c showed that different words have different stable/variable band patterns at L3.
5f showed the universal/word-specific gap grows through depth (+0.045 at Emb -> +0.180 at L3).

But does a band that carries IDENTITY at L0 become a CONTEXT band at L3?
Or is role assignment fixed at embedding time and just amplified through depth?

Method: For each word, compute per-band phase stability at EACH layer independently.
Then measure:
  A) Correlation of stability profiles between layers — high = roles preserved
  B) Number of bands that "flip" category (stable->variable or vice versa)
  C) Whether flips are random or directional (do bands tend to stabilize or destabilize?)
  D) Does the universal/word-specific classification hold at every layer?

If roles reassign: the model actively reorganises band function at each depth.
If roles are fixed: the embedding sets the pattern, layers just amplify it.

Uses Maestro+curriculum model. Same methodology as 5c.
"""

import math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', '..', 'experiments'))
sys.path.insert(0, _here)

from phaseC_integrated import (
    GPT, Dataset, download_shakespeare, build_harmonic_table, build_positional_table,
    N_BANDS, N_EMBD, N_HEAD, BLOCK_SIZE, BATCH_SIZE, LEARNING_RATE,
    MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS, DEVICE, MAESTRO_DIM,
    PROG_STAGES, MAG_FREE_STEP, estimate_loss,
)
from depth_frequency_5a import train_model, decompose_bands
from depth_frequency_5c import extract_token_states, analyse_band_stability, build_test_sets


# ─── Test A: Layer-to-layer stability correlation ────────────────────

def test_layer_correlation(all_results):
    """
    For each word, correlate band stability profiles between consecutive layers.
    High correlation = roles preserved through depth.
    Low correlation = roles reassigned at each layer.
    """
    words = list(all_results.keys())
    layer_names = list(all_results[words[0]].keys())
    n_layers = len(layer_names)

    print(f"\n{'='*70}")
    print(f"  TEST A: Layer-to-Layer Stability Correlation")
    print(f"  (Do the same bands stay stable from layer to layer?)")
    print(f"{'='*70}")

    # Per-word correlation matrix
    print(f"\n  --- Per-word correlation between layers ---")
    for word in words:
        print(f"\n  '{word}':")
        print(f"  {'':6}", end="")
        for ln in layer_names:
            print(f"  {ln:>6}", end="")
        print()

        stabs = []
        for ln in layer_names:
            stabs.append(all_results[word][ln]['phase_stability'])

        for i, ln1 in enumerate(layer_names):
            print(f"  {ln1:<6}", end="")
            for j, ln2 in enumerate(layer_names):
                if i == j:
                    print(f"  {'---':>6}", end="")
                else:
                    r = np.corrcoef(stabs[i], stabs[j])[0, 1]
                    print(f"  {r:>6.3f}", end="")
            print()

    # Summary: consecutive layer correlations
    print(f"\n  --- Consecutive layer correlations (mean across words) ---")
    print(f"  {'Transition':<12} ", end="")
    for word in words:
        print(f" {word:>8}", end="")
    print(f" {'Mean':>8}")

    all_consecutive = []
    for li in range(n_layers - 1):
        ln1, ln2 = layer_names[li], layer_names[li + 1]
        transition = f"{ln1}->{ln2}"
        word_corrs = []
        print(f"  {transition:<12} ", end="")
        for word in words:
            s1 = all_results[word][ln1]['phase_stability']
            s2 = all_results[word][ln2]['phase_stability']
            r = np.corrcoef(s1, s2)[0, 1]
            word_corrs.append(r)
            print(f" {r:>8.3f}", end="")
        mean_r = np.mean(word_corrs)
        all_consecutive.append(mean_r)
        print(f" {mean_r:>8.3f}")

    # Emb->L3 (endpoint) correlation
    print(f"\n  --- Endpoint correlation (Emb -> L3) ---")
    endpoint_corrs = []
    for word in words:
        s_emb = all_results[word][layer_names[0]]['phase_stability']
        s_final = all_results[word][layer_names[-1]]['phase_stability']
        r = np.corrcoef(s_emb, s_final)[0, 1]
        endpoint_corrs.append(r)
        print(f"  {word:<8}: r = {r:.3f}")
    print(f"  Mean endpoint r: {np.mean(endpoint_corrs):.3f}")

    return all_consecutive, endpoint_corrs


# ─── Test B: Band role flips ─────────────────────────────────────────

def test_band_flips(all_results, stable_threshold=0.9, variable_threshold=0.5):
    """
    Count bands that change category between layers.
    Categories: stable (>0.9), variable (<0.5), intermediate.
    A "flip" = stable in one layer, variable in another (or vice versa).
    """
    words = list(all_results.keys())
    layer_names = list(all_results[words[0]].keys())
    n_layers = len(layer_names)

    print(f"\n{'='*70}")
    print(f"  TEST B: Band Role Flips Between Layers")
    print(f"  (How many bands change category: stable->variable or variable->stable?)")
    print(f"{'='*70}")

    print(f"\n  Thresholds: stable > {stable_threshold}, variable < {variable_threshold}")

    for word in words:
        print(f"\n  --- '{word}' ---")

        # Get stability at each layer
        stab_by_layer = []
        for ln in layer_names:
            stab_by_layer.append(all_results[word][ln]['phase_stability'])

        # Track per-band category through layers
        # Count transitions
        stable_to_var = 0
        var_to_stable = 0
        stable_to_mid = 0
        mid_to_stable = 0
        var_to_mid = 0
        mid_to_var = 0

        for b in range(N_BANDS):
            for li in range(n_layers - 1):
                s1 = stab_by_layer[li][b]
                s2 = stab_by_layer[li + 1][b]

                cat1 = 'stable' if s1 > stable_threshold else ('variable' if s1 < variable_threshold else 'mid')
                cat2 = 'stable' if s2 > stable_threshold else ('variable' if s2 < variable_threshold else 'mid')

                if cat1 == 'stable' and cat2 == 'variable':
                    stable_to_var += 1
                elif cat1 == 'variable' and cat2 == 'stable':
                    var_to_stable += 1
                elif cat1 == 'stable' and cat2 == 'mid':
                    stable_to_mid += 1
                elif cat1 == 'mid' and cat2 == 'stable':
                    mid_to_stable += 1
                elif cat1 == 'variable' and cat2 == 'mid':
                    var_to_mid += 1
                elif cat1 == 'mid' and cat2 == 'variable':
                    mid_to_var += 1

        total_transitions = N_BANDS * (n_layers - 1)
        total_flips = stable_to_var + var_to_stable
        total_changes = total_flips + stable_to_mid + mid_to_stable + var_to_mid + mid_to_var

        print(f"  Extreme flips (stable<->variable): {total_flips}/{total_transitions} "
              f"({total_flips/total_transitions*100:.1f}%)")
        print(f"    stable->variable: {stable_to_var}  |  variable->stable: {var_to_stable}")
        print(f"  Category changes (any): {total_changes}/{total_transitions} "
              f"({total_changes/total_transitions*100:.1f}%)")
        print(f"    stable->mid: {stable_to_mid}  |  mid->stable: {mid_to_stable}")
        print(f"    variable->mid: {var_to_mid}  |  mid->variable: {mid_to_var}")

        # Net direction: are bands stabilizing or destabilizing through depth?
        net_stabilize = (var_to_stable + var_to_mid + mid_to_stable) - \
                       (stable_to_var + stable_to_mid + mid_to_var)
        direction = "stabilizing" if net_stabilize > 0 else "destabilizing"
        print(f"  Net direction: {direction} ({net_stabilize:+d})")

    # Overall summary
    print(f"\n  --- Overall Emb vs L3 category comparison ---")
    print(f"  {'Word':<8} {'Emb stable':>12} {'L3 stable':>12} {'Emb var':>10} {'L3 var':>10} {'Flipped':>10}")
    for word in words:
        s_emb = all_results[word][layer_names[0]]['phase_stability']
        s_final = all_results[word][layer_names[-1]]['phase_stability']

        emb_stable = (s_emb > stable_threshold).sum()
        final_stable = (s_final > stable_threshold).sum()
        emb_var = (s_emb < variable_threshold).sum()
        final_var = (s_final < variable_threshold).sum()

        # Bands that were stable at Emb but variable at L3 (or vice versa)
        flipped = ((s_emb > stable_threshold) & (s_final < variable_threshold)).sum() + \
                  ((s_emb < variable_threshold) & (s_final > stable_threshold)).sum()

        print(f"  {word:<8} {emb_stable:>12} {final_stable:>12} {emb_var:>10} {final_var:>10} {flipped:>10}")


# ─── Test C: Universal band classification stability ─────────────────

def test_universal_persistence(all_results):
    """
    5c classified 43 bands as universal (stable for all words at L3).
    Does this classification hold at every layer? Or are different bands
    universal at L0 vs L3?
    """
    words = list(all_results.keys())
    layer_names = list(all_results[words[0]].keys())

    print(f"\n{'='*70}")
    print(f"  TEST C: Universal Band Classification Through Depth")
    print(f"  (Are the same bands universal at every layer?)")
    print(f"{'='*70}")

    # At each layer, classify bands as universal (stable for all words)
    universal_by_layer = {}
    for ln in layer_names:
        stab_matrix = np.array([all_results[w][ln]['phase_stability'] for w in words])
        mean_stab = stab_matrix.mean(axis=0)
        std_stab = stab_matrix.std(axis=0)
        # Universal = high mean stability, low cross-word variance
        universal = (mean_stab > 0.8) & (std_stab < 0.1)
        universal_by_layer[ln] = universal
        n_uni = universal.sum()
        print(f"  {ln}: {n_uni} universal bands (mean_stab>0.8, std<0.1)")

    # Overlap between layers
    print(f"\n  --- Universal band overlap between layers ---")
    print(f"  {'':6}", end="")
    for ln in layer_names:
        print(f"  {ln:>6}", end="")
    print()
    for ln1 in layer_names:
        print(f"  {ln1:<6}", end="")
        u1 = universal_by_layer[ln1]
        for ln2 in layer_names:
            u2 = universal_by_layer[ln2]
            overlap = (u1 & u2).sum()
            print(f"  {overlap:>6}", end="")
        print()

    # Jaccard similarity between layers
    print(f"\n  --- Jaccard similarity of universal band sets ---")
    for i in range(len(layer_names)):
        for j in range(i + 1, len(layer_names)):
            ln1, ln2 = layer_names[i], layer_names[j]
            u1, u2 = universal_by_layer[ln1], universal_by_layer[ln2]
            intersection = (u1 & u2).sum()
            union = (u1 | u2).sum()
            jaccard = intersection / union if union > 0 else 0
            print(f"  {ln1}<->{ln2}: Jaccard = {jaccard:.3f} "
                  f"(overlap={intersection}, union={union})")

    # Bands that are universal at EVERY layer vs only some
    always_universal = np.ones(N_BANDS, dtype=bool)
    ever_universal = np.zeros(N_BANDS, dtype=bool)
    for ln in layer_names:
        always_universal &= universal_by_layer[ln]
        ever_universal |= universal_by_layer[ln]

    print(f"\n  Always universal (every layer): {always_universal.sum()} bands")
    print(f"  Ever universal (any layer):     {ever_universal.sum()} bands")
    print(f"  Never universal:                {(~ever_universal).sum()} bands")

    # Which bands are always universal?
    if always_universal.sum() > 0:
        print(f"  Always-universal band IDs: {np.where(always_universal)[0] + 1}")

    return universal_by_layer


# ─── Test D: Per-band stability trajectories ─────────────────────────

def test_stability_trajectories(all_results):
    """
    Track the full trajectory of each band's stability through depth.
    Classify trajectory shapes: monotonic increase, decrease, U-shape, inverted-U, flat.
    """
    words = list(all_results.keys())
    layer_names = list(all_results[words[0]].keys())
    n_layers = len(layer_names)

    print(f"\n{'='*70}")
    print(f"  TEST D: Per-Band Stability Trajectories")
    print(f"  (How does each band's stability evolve through depth?)")
    print(f"{'='*70}")

    for word in words:
        # (n_layers, n_bands) stability matrix
        stab_traj = np.array([all_results[word][ln]['phase_stability']
                             for ln in layer_names])

        # Classify each band's trajectory
        monotonic_up = 0
        monotonic_down = 0
        u_shape = 0
        inverted_u = 0
        flat = 0
        other = 0

        for b in range(N_BANDS):
            traj = stab_traj[:, b]
            diffs = np.diff(traj)

            # Flat: range < 0.1
            if traj.max() - traj.min() < 0.1:
                flat += 1
                continue

            # Monotonic: all diffs same sign
            if np.all(diffs >= -0.01):  # small tolerance
                monotonic_up += 1
            elif np.all(diffs <= 0.01):
                monotonic_down += 1
            else:
                # Check for U-shape or inverted-U
                min_idx = np.argmin(traj)
                max_idx = np.argmax(traj)

                if 0 < min_idx < n_layers - 1 and traj[0] > traj[min_idx] and traj[-1] > traj[min_idx]:
                    u_shape += 1
                elif 0 < max_idx < n_layers - 1 and traj[0] < traj[max_idx] and traj[-1] < traj[max_idx]:
                    inverted_u += 1
                else:
                    other += 1

        print(f"\n  '{word}' trajectory classification:")
        print(f"    Monotonic increase (stabilizing):  {monotonic_up:>3}/64")
        print(f"    Monotonic decrease (destabilizing): {monotonic_down:>3}/64")
        print(f"    U-shape (dip then recover):        {u_shape:>3}/64")
        print(f"    Inverted-U (peak then decline):    {inverted_u:>3}/64")
        print(f"    Flat (range < 0.1):                {flat:>3}/64")
        print(f"    Other:                             {other:>3}/64")

    # Mean trajectory across all words
    print(f"\n  --- Mean stability trajectory (across all words) ---")
    print(f"  {'Band group':<12}", end="")
    for ln in layer_names:
        print(f"  {ln:>6}", end="")
    print(f"  {'Range':>8}")

    for g in range(0, N_BANDS, 8):
        print(f"  {g+1:>2}-{min(g+8, N_BANDS):<8}", end="")
        vals = []
        for ln in layer_names:
            mean_s = np.mean([all_results[w][ln]['phase_stability'][g:g+8].mean()
                            for w in words])
            vals.append(mean_s)
            print(f"  {mean_s:>6.3f}", end="")
        print(f"  {max(vals)-min(vals):>8.3f}")


# ─── Test E: Nyquist boundary analysis ────────────────────────────────

def test_nyquist_boundary(all_results, universal_by_layer, vocab_size):
    """
    Check whether the Nyquist limit (vocab_size / 2) cleanly separates
    universal (structural) from word-specific (semantic) bands.

    Band k (1-indexed) = harmonic k. Nyquist limit = vocab_size / 2.
    Above-Nyquist bands alias to lower harmonics and are mathematically
    redundant for a vocabulary of this size.

    Also analyse GCD(harmonic, vocab_size) for degeneracy — harmonics
    that share a factor with vocab_size produce fewer distinct values.
    """
    words = list(all_results.keys())
    layer_names = list(all_results[words[0]].keys())
    final = layer_names[-1]
    nyquist = vocab_size / 2.0

    print(f"\n{'='*70}")
    print(f"  TEST E: Nyquist Boundary Analysis")
    print(f"  (Does the Nyquist limit separate structural from semantic bands?)")
    print(f"  Vocab size: {vocab_size}, Nyquist limit: {nyquist}")
    print(f"{'='*70}")

    # Classify each band by Nyquist position and GCD
    below_nyquist = []  # band indices (0-based)
    above_nyquist = []

    print(f"\n  --- Per-band mathematical properties ---")
    print(f"  {'Band':>5} {'Harm':>5} {'Position':>10} {'GCD':>5} {'Distinct':>9} {'Alias of':>9}")

    for b in range(N_BANDS):
        harmonic = b + 1  # band k (0-indexed) = harmonic k+1
        gcd = math.gcd(harmonic, vocab_size)
        n_distinct = vocab_size // gcd
        above = harmonic > nyquist
        alias_of = vocab_size - harmonic if above else None

        if above:
            above_nyquist.append(b)
        else:
            below_nyquist.append(b)

        # Only print interesting bands (above Nyquist, degenerate, or always-universal)
        always_uni = True
        for ln in layer_names:
            if not universal_by_layer[ln][b]:
                always_uni = False
                break

        if above or gcd > 1 or always_uni:
            pos_label = "ABOVE" if above else "below"
            alias_str = f"h{alias_of}" if alias_of else "--"
            marker = " <-- ALWAYS UNI" if always_uni else ""
            print(f"  {b+1:>5} {harmonic:>5} {pos_label:>10} {gcd:>5} {n_distinct:>9} {alias_str:>9}{marker}")

    # Get L3 universal classification
    stab_matrix_L3 = np.array([all_results[w][final]['phase_stability'] for w in words])
    mean_stab_L3 = stab_matrix_L3.mean(axis=0)
    std_stab_L3 = stab_matrix_L3.std(axis=0)
    universal_L3 = (mean_stab_L3 > 0.8) & (std_stab_L3 < 0.1)

    # Count universal bands above vs below Nyquist
    above_arr = np.array(above_nyquist)
    below_arr = np.array(below_nyquist)

    n_above_universal = universal_L3[above_arr].sum() if len(above_arr) > 0 else 0
    n_below_universal = universal_L3[below_arr].sum() if len(below_arr) > 0 else 0

    print(f"\n  --- Nyquist boundary vs universality (L3) ---")
    print(f"  Below Nyquist (bands 1-{int(nyquist)}):  {len(below_arr)} bands, "
          f"{n_below_universal} universal ({n_below_universal/len(below_arr)*100:.1f}%)")
    print(f"  Above Nyquist (bands {int(nyquist)+1}-64): {len(above_arr)} bands, "
          f"{n_above_universal} universal ({n_above_universal/len(above_arr)*100:.1f}%)")

    # Mean stability by Nyquist position
    below_mean_stab = mean_stab_L3[below_arr].mean()
    above_mean_stab = mean_stab_L3[above_arr].mean()
    print(f"\n  Mean stability (L3):")
    print(f"    Below Nyquist: {below_mean_stab:.4f}")
    print(f"    Above Nyquist: {above_mean_stab:.4f}")
    print(f"    Gap: {above_mean_stab - below_mean_stab:+.4f}")

    # Cross-word std by Nyquist position (lower std = more universal)
    below_mean_std = std_stab_L3[below_arr].mean()
    above_mean_std = std_stab_L3[above_arr].mean()
    print(f"  Cross-word std (lower = more universal):")
    print(f"    Below Nyquist: {below_mean_std:.4f}")
    print(f"    Above Nyquist: {above_mean_std:.4f}")

    # GCD analysis: degenerate bands (GCD > 1)
    degenerate = []
    coprime = []
    for b in range(N_BANDS):
        harmonic = b + 1
        gcd = math.gcd(harmonic, vocab_size)
        if gcd > 1:
            degenerate.append(b)
        else:
            coprime.append(b)

    deg_arr = np.array(degenerate)
    cop_arr = np.array(coprime)

    n_deg_universal = universal_L3[deg_arr].sum() if len(deg_arr) > 0 else 0
    n_cop_universal = universal_L3[cop_arr].sum() if len(cop_arr) > 0 else 0

    print(f"\n  --- GCD degeneracy vs universality (L3) ---")
    print(f"  Coprime (GCD=1):     {len(cop_arr)} bands, "
          f"{n_cop_universal} universal ({n_cop_universal/len(cop_arr)*100:.1f}%)")
    print(f"  Degenerate (GCD>1):  {len(deg_arr)} bands, "
          f"{n_deg_universal} universal ({n_deg_universal/len(deg_arr)*100:.1f}%)")

    # Combined: below-Nyquist AND coprime = full semantic capability
    semantic_capable = []
    for b in range(N_BANDS):
        harmonic = b + 1
        if harmonic <= nyquist and math.gcd(harmonic, vocab_size) == 1:
            semantic_capable.append(b)

    sem_arr = np.array(semantic_capable)
    n_sem_universal = universal_L3[sem_arr].sum() if len(sem_arr) > 0 else 0

    print(f"\n  --- Combined: below-Nyquist AND coprime (full semantic capability) ---")
    print(f"  Semantic-capable bands: {len(sem_arr)}")
    print(f"  Of these, universal at L3: {n_sem_universal} ({n_sem_universal/len(sem_arr)*100:.1f}%)")
    print(f"  Of these, word-specific at L3: {len(sem_arr) - n_sem_universal} "
          f"({(len(sem_arr) - n_sem_universal)/len(sem_arr)*100:.1f}%)")

    # What about the 21 word-specific bands from 5c?
    word_specific_L3 = ~universal_L3
    n_ws_semantic = word_specific_L3[sem_arr].sum()
    n_ws_total = word_specific_L3.sum()
    print(f"\n  Word-specific bands at L3: {n_ws_total}")
    print(f"  Of those, semantic-capable: {n_ws_semantic}/{n_ws_total} "
          f"({n_ws_semantic/n_ws_total*100:.1f}%)")

    # Conjugate pair analysis
    print(f"\n  --- Conjugate pair analysis ---")
    print(f"  (Above-Nyquist band N aliases to band {vocab_size}-N)")
    for b in above_arr:
        harmonic = b + 1
        alias_harmonic = vocab_size - harmonic
        alias_band = alias_harmonic - 1  # 0-indexed
        b_uni = "UNI" if universal_L3[b] else "ws"
        a_uni = "UNI" if universal_L3[alias_band] else "ws"
        b_stab = mean_stab_L3[b]
        a_stab = mean_stab_L3[alias_band]
        match = "MATCH" if b_uni == a_uni else "MISMATCH"
        print(f"  Band {b+1:>2} ({b_uni}) <-> Band {alias_band+1:>2} ({a_uni}): "
              f"stab {b_stab:.3f} / {a_stab:.3f} [{match}]")

    # Count conjugate pair matches
    matches = 0
    mismatches = 0
    for b in above_arr:
        harmonic = b + 1
        alias_band = vocab_size - harmonic - 1
        if universal_L3[b] == universal_L3[alias_band]:
            matches += 1
        else:
            mismatches += 1
    print(f"\n  Conjugate pairs with same classification: {matches}/{matches+mismatches}")

    # Always-universal analysis
    always_uni_bands = []
    for b in range(N_BANDS):
        is_always = True
        for ln in layer_names:
            if not universal_by_layer[ln][b]:
                is_always = False
                break
        if is_always:
            always_uni_bands.append(b)

    print(f"\n  --- Always-universal band mathematical properties ---")
    print(f"  {'Band':>5} {'Harm':>5} {'Nyquist':>8} {'GCD':>5} {'Distinct':>9} {'Alias':>6} {'Explanation':>20}")
    for b in always_uni_bands:
        harmonic = b + 1
        gcd = math.gcd(harmonic, vocab_size)
        n_distinct = vocab_size // gcd
        above = harmonic > nyquist
        alias_of = vocab_size - harmonic if above else None

        if above:
            explanation = f"alias of h{alias_of}"
        elif gcd > 1:
            explanation = f"degenerate ({n_distinct} groups)"
        else:
            # Check if it's the conjugate of an above-Nyquist always-universal
            conjugate = vocab_size - harmonic
            if (conjugate - 1) in always_uni_bands:
                explanation = f"paired with b{conjugate}"
            else:
                explanation = "UNEXPLAINED"

        print(f"  {b+1:>5} {harmonic:>5} {'ABOVE' if above else 'below':>8} "
              f"{gcd:>5} {n_distinct:>9} {('h'+str(alias_of)) if alias_of else '--':>6} {explanation:>20}")

    # Summary verdict
    print(f"\n  --- Nyquist verdict ---")
    if n_above_universal / len(above_arr) > 0.8:
        print(f"  STRONG: {n_above_universal/len(above_arr)*100:.0f}% of above-Nyquist bands are universal.")
        if n_ws_semantic / n_ws_total > 0.7:
            print(f"  {n_ws_semantic/n_ws_total*100:.0f}% of word-specific bands are semantic-capable.")
            print(f"  The Nyquist boundary largely determines the structural/semantic divide.")
            print(f"  Optimal band count for vocab {vocab_size}: ~{len(sem_arr)} semantic bands.")
        else:
            print(f"  But only {n_ws_semantic/n_ws_total*100:.0f}% of word-specific bands are semantic-capable.")
            print(f"  The divide is partially Nyquist, partially learned.")
    else:
        print(f"  WEAK: Only {n_above_universal/len(above_arr)*100:.0f}% of above-Nyquist bands are universal.")
        print(f"  The Nyquist boundary does not cleanly determine band roles.")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Experiment 5g: Do Band Roles Reassign Through Depth?")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"  Dataset: {len(text)} chars, {dataset.vocab_size} vocab")

    test_sets = build_test_sets()

    # Verify sentences
    print(f"\n  Verifying sentence construction...")
    errors = 0
    for ts in test_sets:
        word = ts['word']
        for text_str, pos in ts['sentences']:
            actual = text_str[pos:pos+len(word)]
            if actual != word:
                print(f"  ERROR: '{word}' expected at pos {pos}, found '{actual}'")
                errors += 1
    if errors == 0:
        print(f"  All positions verified.")

    # Train model
    print(f"\n  Training Maestro + curriculum model...")
    model = train_model(dataset, mode="kerr", use_maestro=True, curriculum=True)

    # Analyse each word
    all_results = {}
    print(f"\n  Analysing {len(test_sets)} words...")
    for ts in test_sets:
        word = ts['word']
        results = analyse_band_stability(model, dataset, ts)
        all_results[word] = results

    # Run all tests
    consecutive, endpoint = test_layer_correlation(all_results)
    test_band_flips(all_results)
    universal_by_layer = test_universal_persistence(all_results)
    test_stability_trajectories(all_results)
    test_nyquist_boundary(all_results, universal_by_layer, dataset.vocab_size)

    # ─── Verdict ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    mean_consecutive = np.mean(consecutive)
    mean_endpoint = np.mean(endpoint)

    print(f"\n  Mean consecutive-layer correlation: {mean_consecutive:.3f}")
    print(f"  Mean Emb->L3 endpoint correlation:   {mean_endpoint:.3f}")

    if mean_consecutive > 0.7 and mean_endpoint > 0.5:
        print(f"\n  ROLES ARE PRESERVED. Band assignments set at embedding are maintained")
        print(f"  through depth. Layers amplify the existing pattern (more stable bands")
        print(f"  become more stable, variable bands become more variable) rather than")
        print(f"  reassigning which bands carry identity vs context.")
        if mean_endpoint < mean_consecutive:
            print(f"  But endpoint correlation ({mean_endpoint:.3f}) < consecutive ({mean_consecutive:.3f})")
            print(f"  suggests gradual drift — small changes compound over 4 layers.")
    elif mean_consecutive > 0.7 and mean_endpoint < 0.3:
        print(f"\n  GRADUAL REASSIGNMENT. Consecutive layers preserve roles (r={mean_consecutive:.3f})")
        print(f"  but endpoint correlation is low (r={mean_endpoint:.3f}). The model makes small")
        print(f"  adjustments at each layer that compound into large role changes by L3.")
        print(f"  Band function evolves gradually, not abruptly.")
    elif mean_consecutive < 0.5:
        print(f"\n  ACTIVE REORGANISATION. Low consecutive correlation (r={mean_consecutive:.3f})")
        print(f"  means the model actively reassigns band roles at each layer.")
        print(f"  Identity and context aren't fixed properties of bands — they're")
        print(f"  recomputed at every depth.")
    else:
        print(f"\n  MIXED. Moderate correlations suggest partial preservation with")
        print(f"  some reorganisation. Not fully fixed, not fully reassigned.")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  Total time: {time.time() - t0:.1f}s")
