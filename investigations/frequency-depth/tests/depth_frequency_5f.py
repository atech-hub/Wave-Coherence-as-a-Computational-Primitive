"""
Experiment 5f: What Do the 43 Universal Bands Carry?

5c found 43 of 64 bands are stable for ALL words regardless of context.
That's 67% of the spectrum. What are they encoding?

Candidates:
1. Position information — bands change with position in sentence, not with content
2. Character identity — bands change with which character, not where it is
3. Structural scaffolding — bands are roughly constant regardless of position or character

Method:
A. Same character at different positions: "the" at pos 0 vs pos 10 vs pos 20
   If universal bands change -> they encode position
B. Different characters at the same position: 't' vs 'k' vs 's' at pos 4
   If universal bands change -> they encode character identity
C. Same character, same position, different surrounding context
   If universal bands change -> they encode context (contradicts 5c universal label)

Also: compare universal vs word-specific bands on each test to confirm the split.

If 43 bands are overhead, the model does all semantic work in 21 bands.
That's a compression finding connecting to selective band loading.
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
from depth_frequency_5c import extract_token_states


# ─── The 43 universal band indices from 5c (Maestro, L3) ────────────

# These were stable (>0.8 mean, <0.1 std) for all 5 words in Exp 5c
UNIVERSAL_BANDS = np.array([
    1, 2, 6, 8, 9, 10, 11, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 33, 34, 39, 40, 42, 44, 45, 46, 47, 48, 50,
    52, 53, 55, 58, 59, 60, 61, 62
]) - 1  # Convert to 0-indexed

WORD_SPECIFIC_BANDS = np.array([
    i for i in range(N_BANDS) if i not in UNIVERSAL_BANDS
])


# ─── Extract states for multiple sentences ───────────────────────────

@torch.no_grad()
def extract_batch_states(model, dataset, sentences_with_pos):
    """
    Extract hidden states at target positions for a list of (sentence, pos) pairs.
    Returns: (n_sentences, n_layers+1, n_embd)
    """
    all_states = []
    for text, pos in sentences_with_pos:
        states = extract_token_states(model, dataset, text, pos)
        all_states.append(states)
    return torch.stack(all_states)


# ─── Measure band variation across a set of states ──────────────────

def band_variation(states_batch, layer_idx=-1):
    """
    Compute per-band phase and magnitude variation across a batch of states.

    Args:
        states_batch: (n_sentences, n_layers+1, n_embd)
        layer_idx: which layer to analyse (-1 = final)

    Returns: dict with phase_stability, mag_cv per band
    """
    n_layers = states_batch.shape[1]
    if layer_idx < 0:
        layer_idx = n_layers + layer_idx

    # Extract target layer: (n_sentences, n_embd)
    layer_states = states_batch[:, layer_idx, :]

    # Decompose: need shape (1, n_sentences, n_embd) for decompose_bands
    layer_states_3d = layer_states.unsqueeze(0)
    magnitude, phase = decompose_bands(layer_states_3d)
    # magnitude: (1, n_sentences, n_bands), phase: (1, n_sentences, n_bands)
    mag = magnitude[0].numpy()   # (n_sentences, n_bands)
    ph = phase[0].numpy()        # (n_sentences, n_bands)

    # Phase stability: circular resultant length
    cos_ph = np.cos(ph)
    sin_ph = np.sin(ph)
    resultant = np.sqrt(cos_ph.mean(axis=0)**2 + sin_ph.mean(axis=0)**2)

    # Magnitude CV
    mag_mean = mag.mean(axis=0)
    mag_std = mag.std(axis=0)
    mag_cv = np.where(mag_mean > 1e-8, mag_std / mag_mean, 0)

    return {
        'phase_stability': resultant,  # (n_bands,)  1=stable, 0=variable
        'mag_cv': mag_cv,              # (n_bands,)  0=stable, high=variable
        'mag_mean': mag_mean,
        'phase_mean': ph.mean(axis=0),
    }


def print_band_comparison(label, variation, layer_name="L3"):
    """Print universal vs word-specific band comparison."""
    ps = variation['phase_stability']
    mc = variation['mag_cv']

    uni_ps = ps[UNIVERSAL_BANDS].mean()
    ws_ps = ps[WORD_SPECIFIC_BANDS].mean()
    uni_mc = mc[UNIVERSAL_BANDS].mean()
    ws_mc = mc[WORD_SPECIFIC_BANDS].mean()

    print(f"\n  {label} ({layer_name}):")
    print(f"    {'':30} {'Phase stab':>12} {'Mag CV':>10}")
    print(f"    {'Universal bands (43)':30} {uni_ps:>12.4f} {uni_mc:>10.4f}")
    print(f"    {'Word-specific bands (21)':30} {ws_ps:>12.4f} {ws_mc:>10.4f}")
    print(f"    {'Ratio (universal/specific)':30} {uni_ps/ws_ps:>12.4f} {uni_mc/(ws_mc+1e-8):>10.4f}")

    # Count how many bands are highly stable (>0.9)
    uni_stable = (ps[UNIVERSAL_BANDS] > 0.9).sum()
    ws_stable = (ps[WORD_SPECIFIC_BANDS] > 0.9).sum()
    print(f"    Universal stable (>0.9): {uni_stable}/{len(UNIVERSAL_BANDS)}")
    print(f"    Word-spec stable (>0.9): {ws_stable}/{len(WORD_SPECIFIC_BANDS)}")

    return uni_ps, ws_ps, uni_mc, ws_mc


# ─── Test A: Same character at different positions ───────────────────

def test_position_sensitivity(model, dataset):
    """
    Does the same character have different universal band activations at
    different positions? If yes, universal bands encode position.
    """
    print(f"\n{'='*70}")
    print(f"  TEST A: Position Sensitivity")
    print(f"  Same character at different positions in different sentences")
    print(f"{'='*70}")

    # Use common characters at various positions
    # 'e' is the most common char in English — appears everywhere in Shakespeare
    tests = {
        'e': [
            ("the king sits upon the throne", 2),    # 'e' at pos 2
            ("he feared death more than all", 1),    # 'e' at pos 1
            ("a noble death upon the field", 8),     # 'e' at pos 8 (nobl'e')
            ("she spoke of love and more", 2),       # 'e' at pos 2
            ("the sword was broken here", 2),        # 'e' at pos 2
            ("before the dawn we ride forth", 6),    # 'e' at pos 6
        ],
        't': [
            ("the king sits upon the throne", 0),    # 't' at pos 0
            ("to be or not to be that is", 0),       # 't' at pos 0
            ("art thou the man i seek here", 2),     # 't' at pos 2 (ar't')
            ("but soft what light is this", 2),      # 't' at pos 2 (bu't')
            ("let me not to the marriage", 2),       # 't' at pos 2 (le't')
            ("sit upon the ground and tell", 2),     # 't' at pos 2 (si't')
        ],
    }

    # Also test: same character at VERY different positions
    # Build sentences where 'a' appears at pos 0, ~10, ~17
    position_test_raw = [
        "a noble king sat on the throne of gold",
        "the noble a king sat on the throne",
        "the king sat upon a golden throne",
    ]
    position_test = [(s, s.index('a')) for s in position_test_raw]
    print(f"  'a' positions: {[p for _, p in position_test]}")

    results = {}

    # Test each character group
    for char, sentences in tests.items():
        # Verify all positions point to the right character
        valid = []
        for text, pos in sentences:
            if pos < len(text) and text[pos] == char:
                valid.append((text, pos))
        if len(valid) < 3:
            print(f"  Skipping '{char}' — too few valid positions ({len(valid)})")
            continue

        states = extract_batch_states(model, dataset, valid)
        var = band_variation(states)
        results[f"char_{char}"] = var
        print_band_comparison(f"'{char}' at {len(valid)} different positions/contexts", var)

    # Explicit position test with 'a'
    states = extract_batch_states(model, dataset, position_test)
    var = band_variation(states)
    results['position_a'] = var
    print_band_comparison("'a' at positions 0, 10, 19", var)

    return results


# ─── Test B: Different characters at same position ───────────────────

def test_character_sensitivity(model, dataset):
    """
    Do universal bands change when different characters occupy the same position?
    If yes, they encode character identity.
    """
    print(f"\n{'='*70}")
    print(f"  TEST B: Character Identity")
    print(f"  Different characters at the same position (pos 4)")
    print(f"{'='*70}")

    # All sentences have a target character at position 4
    # "xxxx[TARGET]..." — 4 chars then the target
    sentences = [
        ("the king sits on the throne", 4),    # 'k'
        ("the sword was broken apart", 4),     # 's'
        ("the love between them grew", 4),     # 'l'
        ("the death of the young prince", 4),  # 'd'
        ("the hand that held the crown", 4),   # 'h'
        ("the gold was hidden in caves", 4),   # 'g'
        ("the moon shone over the land", 4),   # 'm'
        ("the noble lord came riding", 4),     # 'n'
        ("the fair lady wept for him", 4),     # 'f'
        ("the wall stood tall and proud", 4),  # 'w'
    ]

    # Verify
    chars_found = []
    valid = []
    for text, pos in sentences:
        chars_found.append(text[pos])
        valid.append((text, pos))

    print(f"  Characters at pos 4: {', '.join(chars_found)}")

    states = extract_batch_states(model, dataset, valid)
    var = band_variation(states)
    print_band_comparison(f"{len(valid)} different characters at position 4", var)

    # Also test: different characters at position 0
    sentences_pos0 = [
        ("the king", 0),       # 't'
        ("a sword", 0),        # 'a'
        ("my love", 0),        # 'm'
        ("in death", 0),       # 'i'
        ("by hand", 0),        # 'b'
        ("no more", 0),        # 'n'
        ("he wept", 0),        # 'h'
        ("so fair", 0),        # 's'
        ("or else", 0),        # 'o'
        ("we come", 0),        # 'w'
    ]

    chars0 = [text[0] for text, _ in sentences_pos0]
    print(f"\n  Characters at pos 0: {', '.join(chars0)}")

    states0 = extract_batch_states(model, dataset, sentences_pos0)
    var0 = band_variation(states0)
    print_band_comparison(f"{len(sentences_pos0)} different characters at position 0", var0)

    return {'pos4': var, 'pos0': var0}


# ─── Test C: Same char, same position, different context ─────────────

def test_context_sensitivity(model, dataset):
    """
    Do universal bands change when the same character is at the same position
    but the surrounding context differs? If yes, they're context-sensitive
    (contradicting their 'universal' label from 5c).
    """
    print(f"\n{'='*70}")
    print(f"  TEST C: Context Sensitivity (control)")
    print(f"  Same character, same position, different surrounding context")
    print(f"{'='*70}")

    # 't' at position 0 with very different contexts
    sentences_t0 = [
        ("the king sits upon the throne", 0),
        ("the sword was broken in two", 0),
        ("the love between them ended", 0),
        ("the death came without warning", 0),
        ("the hand that wrote the words", 0),
        ("the moon was full that night", 0),
        ("the fair lady spoke of war", 0),
        ("the noble prince was slain", 0),
    ]

    # Verify all start with 't'
    for text, pos in sentences_t0:
        assert text[pos] == 't', f"Expected 't' at pos {pos}, got '{text[pos]}'"

    states = extract_batch_states(model, dataset, sentences_t0)
    var = band_variation(states)
    print_band_comparison("'t' at pos 0, 8 different contexts", var)

    # 'e' at position 2 with different contexts
    sentences_e2 = [
        ("the king spoke of his reign", 2),
        ("the sword fell to the ground", 2),
        ("the love was lost forever", 2),
        ("the death of hope and dreams", 2),
        ("the moon cast silver light", 2),
        ("the wall crumbled to dust", 2),
    ]

    for text, pos in sentences_e2:
        assert text[pos] == 'e', f"Expected 'e' at pos {pos} in '{text}', got '{text[pos]}'"

    states_e = extract_batch_states(model, dataset, sentences_e2)
    var_e = band_variation(states_e)
    print_band_comparison("'e' at pos 2, 6 different contexts", var_e)

    return {'t_pos0': var, 'e_pos2': var_e}


# ─── Test D: Per-layer analysis of universal bands ───────────────────

def test_depth_profile(model, dataset):
    """
    Track universal band behaviour through all layers.
    Do they become universal (stable) gradually or are they universal from embedding?
    """
    print(f"\n{'='*70}")
    print(f"  TEST D: When Do Universal Bands Become Universal?")
    print(f"  Tracking stability through depth")
    print(f"{'='*70}")

    # Use mixed sentences (different chars, positions, contexts)
    sentences = [
        ("the king sits upon the throne", 4),    # 'k'
        ("the sword was broken apart", 4),        # 's'
        ("my love is like a summer day", 3),      # 'l'
        ("death came swiftly in the night", 0),   # 'd'
        ("her gentle hand upon his cheek", 11),   # 'h'
        ("a noble lord came riding forth", 2),    # 'n'
        ("the fair lady wept for him", 4),        # 'f'
        ("by my own will i stand alone", 0),      # 'b'
    ]

    states = extract_batch_states(model, dataset, sentences)
    n_layers = states.shape[1]
    layer_names = ["Emb"] + [f"L{i}" for i in range(n_layers - 1)]

    print(f"\n  {'Layer':<6} {'Uni phase stab':>16} {'WS phase stab':>16} {'Uni mag CV':>12} {'WS mag CV':>12} {'Gap':>8}")
    for l in range(n_layers):
        var = band_variation(states, layer_idx=l)
        ps = var['phase_stability']
        mc = var['mag_cv']

        uni_ps = ps[UNIVERSAL_BANDS].mean()
        ws_ps = ps[WORD_SPECIFIC_BANDS].mean()
        uni_mc = mc[UNIVERSAL_BANDS].mean()
        ws_mc = mc[WORD_SPECIFIC_BANDS].mean()
        gap = uni_ps - ws_ps

        print(f"  {layer_names[l]:<6} {uni_ps:>16.4f} {ws_ps:>16.4f} "
              f"{uni_mc:>12.4f} {ws_mc:>12.4f} {gap:>+8.4f}")


# ─── Test E: Magnitude structure of universal bands ──────────────────

def test_magnitude_structure(model, dataset):
    """
    Do universal bands have systematically different magnitude than word-specific bands?
    If universal bands are low-magnitude, they might be residual noise.
    If high-magnitude, they carry strong signal.
    """
    print(f"\n{'='*70}")
    print(f"  TEST E: Magnitude Structure")
    print(f"  Are universal bands high-energy or low-energy?")
    print(f"{'='*70}")

    from depth_frequency_5a import extract_layer_states

    states = extract_layer_states(model, dataset, n_batches=5)
    magnitude, phase = decompose_bands(states)
    # magnitude: (n_layers, n_tokens, n_bands)

    n_layers = magnitude.shape[0]
    layer_names = ["Emb"] + [f"L{i}" for i in range(n_layers - 1)]

    print(f"\n  {'Layer':<6} {'Uni mean |Z|':>14} {'WS mean |Z|':>14} {'Ratio':>8} {'Uni frac of total':>20}")
    for l in range(n_layers):
        mag_l = magnitude[l].numpy()  # (tokens, n_bands)
        mean_per_band = mag_l.mean(axis=0)  # (n_bands,)

        uni_mag = mean_per_band[UNIVERSAL_BANDS].mean()
        ws_mag = mean_per_band[WORD_SPECIFIC_BANDS].mean()
        ratio = uni_mag / ws_mag if ws_mag > 0 else float('inf')

        # What fraction of total magnitude energy is in universal bands?
        uni_energy = mean_per_band[UNIVERSAL_BANDS].sum()
        total_energy = mean_per_band.sum()
        uni_frac = uni_energy / total_energy * 100

        print(f"  {layer_names[l]:<6} {uni_mag:>14.4f} {ws_mag:>14.4f} {ratio:>8.3f} {uni_frac:>19.1f}%")

    # Band-by-band at final layer
    mag_final = magnitude[-1].numpy().mean(axis=0)
    print(f"\n  Top 10 highest-magnitude bands at {layer_names[-1]}:")
    top10 = np.argsort(mag_final)[-10:][::-1]
    for b in top10:
        band_type = "UNI" if b in UNIVERSAL_BANDS else "W-S"
        print(f"    b{b+1}: {mag_final[b]:.4f} [{band_type}]")

    print(f"\n  Top 10 lowest-magnitude bands at {layer_names[-1]}:")
    bot10 = np.argsort(mag_final)[:10]
    for b in bot10:
        band_type = "UNI" if b in UNIVERSAL_BANDS else "W-S"
        print(f"    b{b+1}: {mag_final[b]:.4f} [{band_type}]")


# ─── Summary and verdict ─────────────────────────────────────────────

def print_summary(pos_results, char_results, ctx_results):
    """Cross-test summary."""
    print(f"\n{'='*70}")
    print(f"  SUMMARY: What Do Universal Bands Encode?")
    print(f"{'='*70}")

    # Collect universal band phase stability across all tests
    tests = []

    # Position tests
    for key, var in pos_results.items():
        uni = var['phase_stability'][UNIVERSAL_BANDS].mean()
        ws = var['phase_stability'][WORD_SPECIFIC_BANDS].mean()
        tests.append(('Position', key, uni, ws))

    # Character tests
    for key, var in char_results.items():
        uni = var['phase_stability'][UNIVERSAL_BANDS].mean()
        ws = var['phase_stability'][WORD_SPECIFIC_BANDS].mean()
        tests.append(('Character', key, uni, ws))

    # Context tests
    for key, var in ctx_results.items():
        uni = var['phase_stability'][UNIVERSAL_BANDS].mean()
        ws = var['phase_stability'][WORD_SPECIFIC_BANDS].mean()
        tests.append(('Context', key, uni, ws))

    print(f"\n  {'Test type':<12} {'Variant':<20} {'Uni stab':>10} {'WS stab':>10} {'Uni>WS?':>8}")
    for test_type, variant, uni, ws in tests:
        more_stable = "YES" if uni > ws else "NO"
        print(f"  {test_type:<12} {variant:<20} {uni:>10.4f} {ws:>10.4f} {more_stable:>8}")

    # Average across all tests
    uni_avg = np.mean([t[2] for t in tests])
    ws_avg = np.mean([t[3] for t in tests])

    print(f"\n  Average universal stability: {uni_avg:.4f}")
    print(f"  Average word-specific stability: {ws_avg:.4f}")

    # Verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    # Check: are universal bands stable when position changes?
    pos_uni_stabs = [var['phase_stability'][UNIVERSAL_BANDS].mean()
                     for var in pos_results.values()]
    pos_uni_mean = np.mean(pos_uni_stabs)

    # Are universal bands stable when character changes?
    char_uni_stabs = [var['phase_stability'][UNIVERSAL_BANDS].mean()
                      for var in char_results.values()]
    char_uni_mean = np.mean(char_uni_stabs)

    # Are universal bands stable when context changes?
    ctx_uni_stabs = [var['phase_stability'][UNIVERSAL_BANDS].mean()
                     for var in ctx_results.values()]
    ctx_uni_mean = np.mean(ctx_uni_stabs)

    print(f"\n  Universal band stability when varying:")
    print(f"    Position:  {pos_uni_mean:.4f}")
    print(f"    Character: {char_uni_mean:.4f}")
    print(f"    Context:   {ctx_uni_mean:.4f}")

    if pos_uni_mean > 0.8 and char_uni_mean < 0.5:
        print(f"\n  -> Universal bands encode CHARACTER IDENTITY")
        print(f"     Stable across positions, variable across characters.")
        print(f"     They're not overhead — they carry per-char signal.")
    elif pos_uni_mean < 0.5 and char_uni_mean > 0.8:
        print(f"\n  -> Universal bands encode POSITION")
        print(f"     Variable across positions, stable across characters.")
        print(f"     They're positional encodings embedded in the band structure.")
    elif pos_uni_mean > 0.8 and char_uni_mean > 0.8 and ctx_uni_mean > 0.8:
        print(f"\n  -> Universal bands are STRUCTURAL SCAFFOLDING")
        print(f"     Stable across everything. Fixed structure that doesn't change.")
        print(f"     This is compressible — 43 bands of constant template.")
    elif pos_uni_mean < 0.5 and char_uni_mean < 0.5:
        print(f"\n  -> Universal bands respond to BOTH position and character")
        print(f"     Not truly universal — the 5c label may have been too generous.")
        print(f"     These bands vary with everything, just less than word-specific bands.")
    else:
        print(f"\n  -> MIXED: Universal bands encode a combination of factors")
        print(f"     Not purely position, character, or scaffolding.")
        print(f"     Partial sensitivity to multiple inputs.")

    # Compression implication
    if ctx_uni_mean > 0.8:
        ws_count = len(WORD_SPECIFIC_BANDS)
        print(f"\n  COMPRESSION FINDING: Only {ws_count} of {N_BANDS} bands ({ws_count/N_BANDS*100:.0f}%)")
        print(f"  carry context-dependent meaning. The other {len(UNIVERSAL_BANDS)} could be")
        print(f"  reconstructed from a fixed template, reducing effective bandwidth by {len(UNIVERSAL_BANDS)/N_BANDS*100:.0f}%.")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Experiment 5f: What Do the 43 Universal Bands Carry?")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"  Dataset: {len(text)} chars, {dataset.vocab_size} vocab")
    print(f"  Universal bands: {len(UNIVERSAL_BANDS)}, Word-specific: {len(WORD_SPECIFIC_BANDS)}")

    # Train Maestro+curriculum model
    print(f"\n  Training Maestro + curriculum model...")
    model = train_model(dataset, mode="kerr", use_maestro=True, curriculum=True)

    # Run all tests
    pos_results = test_position_sensitivity(model, dataset)
    char_results = test_character_sensitivity(model, dataset)
    ctx_results = test_context_sensitivity(model, dataset)
    test_depth_profile(model, dataset)
    test_magnitude_structure(model, dataset)

    # Summary
    print_summary(pos_results, char_results, ctx_results)

    print(f"\n{'='*70}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  Total time: {time.time() - t0:.1f}s")
