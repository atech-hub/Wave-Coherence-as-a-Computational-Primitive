"""
Experiment 5c: Band Aspect Separation — What Do Individual Bands Encode?

Feed the same object word ("soup", "king", "sword") through different contextual
sentences. Measure which bands change across contexts and which stay stable.

Hypothesis:
- Stable bands = object identity (what the thing IS)
- Changing bands = contextual aspect (what's happening to it, where it is)
- If the split is clean: bands encode separable aspects of meaning
- If distributed: the information is entangled across all bands

This is a character-level model on Shakespeare, so we use words that appear
frequently in the corpus and build minimal contextual frames around them.

Uses the Maestro+curriculum model (best config, uniform band velocity from 5a).
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
from depth_frequency_5a import train_model, decompose_bands


# ─── Extract hidden state for a specific character position ──────────

@torch.no_grad()
def extract_token_states(model, dataset, text_str, target_pos):
    """
    Run a text string through the model and extract hidden states at a
    specific character position, at every layer.

    Args:
        model: trained GPT model
        dataset: Dataset object (for char encoding)
        text_str: the input text string
        target_pos: character index in text_str to extract states for

    Returns:
        states: (n_layers+1, n_embd) — hidden state at target_pos at each depth
    """
    model.eval()
    n_layers = len(model.blocks)

    # Encode text to token IDs
    ids = [dataset.stoi.get(c, 0) for c in text_str]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    B, T = x.size()

    states = []

    # Embedding
    h = model._get_embeddings(x) + model.wpe[:T]
    states.append(h[0, target_pos].cpu())

    # Each block
    for block in model.blocks:
        h = block(h)
        states.append(h[0, target_pos].cpu())

    return torch.stack(states)  # (n_layers+1, n_embd)


# ─── Build contextual sentence sets ─────────────────────────────────

def build_test_sets():
    """
    Build sets of sentences where the same target word appears in different contexts.
    Uses words common in Shakespeare. Each set has the target word at a known position.

    Returns: list of dicts with 'word', 'sentences' (list of (text, target_pos))
    """
    # Build sentences with auto-computed positions (no manual counting)
    raw_sets = {
        'king': [
            "the king sits upon the throne",
            "the king rides into the battle",
            "the king speaks to his people",
            "the king sleeps in his chamber",
            "the king weeps for his lost son",
            "a wise king rules with mercy",
        ],
        'sword': [
            "he drew his sword from the sheath",
            "the sword was broken in the fight",
            "she held the sword above her head",
            "a golden sword lay on the table",
            "the sword cut through the armor",
            "his father gave him the sword",
        ],
        'love': [
            "my love is like a summer day",
            "she spoke of love and nothing more",
            "for love of god i beg of thee",
            "his love was stronger than his fear",
            "in love there is no room for hate",
            "the love between them grew each day",
        ],
        'death': [
            "he feared death more than all else",
            "death came swiftly in the night",
            "she laughed in the face of death",
            "after death there is but silence",
            "the death of the prince was mourned",
            "a noble death upon the battlefield",
        ],
        'hand': [
            "he took her hand and led her forth",
            "with heavy hand he struck the blow",
            "her gentle hand upon his cheek",
            "the hand that wrote these words is gone",
            "by my own hand i seal this fate",
            "his cold hand lay upon the stone",
        ],
    }

    test_sets = []
    for word, sentences in raw_sets.items():
        sent_list = []
        for s in sentences:
            pos = s.index(word)  # auto-find — will raise if not found
            sent_list.append((s, pos))
        test_sets.append({'word': word, 'sentences': sent_list})

    return test_sets


# ─── Analyse band stability across contexts ─────────────────────────

def analyse_band_stability(model, dataset, test_set):
    """
    For one target word, extract hidden states across all contextual sentences.
    Measure per-band stability (variance across contexts).

    Returns dict with per-band and per-layer analysis.
    """
    word = test_set['word']
    sentences = test_set['sentences']
    n_sentences = len(sentences)

    # Collect states: (n_sentences, n_layers+1, n_embd)
    all_states = []
    for text, pos in sentences:
        states = extract_token_states(model, dataset, text, pos)
        all_states.append(states)

    all_states = torch.stack(all_states)  # (n_sentences, n_layers+1, n_embd)
    n_layers_plus1 = all_states.shape[1]

    # Decompose to phase and magnitude per layer
    # Reshape: treat sentences as token dimension
    # (n_layers+1, n_sentences, n_embd)
    states_by_layer = all_states.permute(1, 0, 2)

    magnitude, phase = decompose_bands(states_by_layer)
    # magnitude: (n_layers+1, n_sentences, n_bands)
    # phase: (n_layers+1, n_sentences, n_bands)

    layer_names = ["Emb"] + [f"L{i}" for i in range(n_layers_plus1 - 1)]

    # Per-band stability = 1 - normalized variance across sentences
    # Low variance = stable (same across contexts) = identity band
    # High variance = changing (different across contexts) = contextual band

    results = {}

    for layer_idx in range(n_layers_plus1):
        mag_l = magnitude[layer_idx].numpy()  # (n_sentences, n_bands)
        phase_l = phase[layer_idx].numpy()     # (n_sentences, n_bands)

        # Phase stability: circular variance (1 - |mean of unit vectors|)
        # For each band, compute mean resultant length across sentences
        cos_phase = np.cos(phase_l)  # (n_sentences, n_bands)
        sin_phase = np.sin(phase_l)
        mean_cos = cos_phase.mean(axis=0)  # (n_bands,)
        mean_sin = sin_phase.mean(axis=0)
        resultant_length = np.sqrt(mean_cos**2 + mean_sin**2)  # 1 = perfectly stable
        phase_stability = resultant_length  # range [0, 1], 1 = all sentences agree

        # Magnitude stability: CV across sentences per band
        mag_mean = mag_l.mean(axis=0)  # (n_bands,)
        mag_std = mag_l.std(axis=0)
        mag_cv = np.where(mag_mean > 1e-8, mag_std / mag_mean, 0)  # lower = more stable

        results[layer_names[layer_idx]] = {
            'phase_stability': phase_stability,
            'mag_cv': mag_cv,
            'mag_mean': mag_mean,
        }

    return results


def print_band_analysis(word, results):
    """Print the band stability analysis for one word."""
    n_layers = len(results)
    layer_names = list(results.keys())

    print(f"\n  --- '{word}' across {6} contexts ---")

    # Show final layer (L3) in detail — that's where meaning is most processed
    final = results[layer_names[-1]]
    ps = final['phase_stability']
    mc = final['mag_cv']

    # Find most stable and most variable bands
    most_stable_phase = np.argsort(ps)[-8:][::-1]  # top 8 by phase stability
    most_variable_phase = np.argsort(ps)[:8]         # bottom 8

    most_stable_mag = np.argsort(mc)[:8]              # lowest CV = most stable
    most_variable_mag = np.argsort(mc)[-8:][::-1]     # highest CV = most variable

    print(f"\n  Final layer ({layer_names[-1]}) phase stability (1.0 = identical across contexts):")
    print(f"    Most STABLE bands (identity):   {', '.join(f'b{b+1}({ps[b]:.3f})' for b in most_stable_phase)}")
    print(f"    Most VARIABLE bands (context):  {', '.join(f'b{b+1}({ps[b]:.3f})' for b in most_variable_phase)}")

    print(f"  Final layer ({layer_names[-1]}) magnitude CV (0.0 = identical across contexts):")
    print(f"    Most STABLE bands (identity):   {', '.join(f'b{b+1}({mc[b]:.3f})' for b in most_stable_mag)}")
    print(f"    Most VARIABLE bands (context):  {', '.join(f'b{b+1}({mc[b]:.3f})' for b in most_variable_mag)}")

    # Summary stats per layer
    print(f"\n  Phase stability by layer (mean across bands):")
    print(f"    {'Layer':<6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Stable(>0.9)':>14} {'Variable(<0.5)':>16}")
    for ln in layer_names:
        ps_l = results[ln]['phase_stability']
        n_stable = (ps_l > 0.9).sum()
        n_variable = (ps_l < 0.5).sum()
        print(f"    {ln:<6} {ps_l.mean():>8.4f} {ps_l.std():>8.4f} {ps_l.min():>8.4f} {ps_l.max():>8.4f} "
              f"{n_stable:>14} {n_variable:>16}")

    # Low vs high band split
    print(f"\n  Low vs High band stability ({layer_names[-1]}):")
    low_ps = ps[:N_BANDS//2].mean()
    high_ps = ps[N_BANDS//2:].mean()
    low_mc = mc[:N_BANDS//2].mean()
    high_mc = mc[N_BANDS//2:].mean()
    print(f"    Phase stability:  Low bands={low_ps:.4f}  High bands={high_ps:.4f}")
    print(f"    Magnitude CV:     Low bands={low_mc:.4f}  High bands={high_mc:.4f}")

    return ps, mc


# ─── Cross-word comparison ───────────────────────────────────────────

def cross_word_analysis(all_results):
    """
    Compare band stability patterns across different words.
    If the same bands are stable for ALL words, those bands encode something
    universal (position? frequency? structure?). If different words have
    different stable bands, the stability pattern is word-specific (identity).
    """
    words = list(all_results.keys())
    n_words = len(words)

    # Get final layer phase stability for each word
    final_stabilities = []
    for word in words:
        layer_names = list(all_results[word].keys())
        final = all_results[word][layer_names[-1]]
        final_stabilities.append(final['phase_stability'])

    stab_matrix = np.array(final_stabilities)  # (n_words, n_bands)

    print(f"\n{'='*70}")
    print(f"  CROSS-WORD BAND STABILITY COMPARISON (final layer)")
    print(f"{'='*70}")

    # Correlation between words' stability profiles
    print(f"\n  --- Stability profile correlation between words ---")
    print(f"  (High = same bands stable for both words; Low = different bands per word)")
    print(f"  {'':12}", end="")
    for w in words:
        print(f"  {w:>8}", end="")
    print()
    for i, w1 in enumerate(words):
        print(f"  {w1:<12}", end="")
        for j, w2 in enumerate(words):
            if i == j:
                print(f"  {'---':>8}", end="")
            else:
                corr = np.corrcoef(stab_matrix[i], stab_matrix[j])[0, 1]
                print(f"  {corr:>8.3f}", end="")
        print()

    # Mean correlation
    corrs = []
    for i in range(n_words):
        for j in range(i+1, n_words):
            corrs.append(np.corrcoef(stab_matrix[i], stab_matrix[j])[0, 1])
    mean_corr = np.mean(corrs)
    print(f"\n  Mean cross-word correlation: {mean_corr:.3f}")

    if mean_corr > 0.7:
        print(f"  -> HIGH: Same bands are stable for all words. Band stability is structural, not semantic.")
    elif mean_corr < 0.3:
        print(f"  -> LOW: Different words have different stable bands. Band stability IS word identity.")
    else:
        print(f"  -> MODERATE: Partial overlap. Some bands are structurally stable, others word-specific.")

    # Which bands are universally stable vs word-specific?
    mean_stab = stab_matrix.mean(axis=0)   # (n_bands,) — mean stability across words
    std_stab = stab_matrix.std(axis=0)     # (n_bands,) — how much words disagree

    # Universal = high mean, low std (stable for everyone)
    # Word-specific = moderate mean, high std (stable for some, variable for others)
    # Context-dependent = low mean (variable for everyone)

    universal = (mean_stab > 0.8) & (std_stab < 0.1)
    context_dep = mean_stab < 0.5
    word_specific = (~universal) & (~context_dep) & (std_stab > 0.1)

    print(f"\n  --- Band classification ---")
    print(f"  Universal (stable for all words):     {universal.sum()} bands  {np.where(universal)[0]+1}")
    print(f"  Context-dependent (variable for all): {context_dep.sum()} bands  {np.where(context_dep)[0]+1}")
    print(f"  Word-specific (stable for some):      {word_specific.sum()} bands")

    # Band group breakdown
    print(f"\n  --- Stability by band group (mean across words) ---")
    print(f"  {'Bands':<10} {'Mean stab':>10} {'Cross-word std':>16} {'Classification':>16}")
    for g in range(0, N_BANDS, 8):
        ms = mean_stab[g:g+8].mean()
        ss = std_stab[g:g+8].mean()
        if ms > 0.8 and ss < 0.1:
            cls = "universal"
        elif ms < 0.5:
            cls = "contextual"
        else:
            cls = "mixed"
        print(f"  {g+1:>2}-{min(g+8, N_BANDS):<6} {ms:>10.4f} {ss:>16.4f} {cls:>16}")

    # THE KEY METRIC: ratio of word-specific variance to total variance
    # If bands encode aspects, we expect some variance to be word-specific
    total_var = stab_matrix.var()
    between_word_var = stab_matrix.mean(axis=1).var()  # how much words differ overall
    between_band_var = stab_matrix.mean(axis=0).var()  # how much bands differ overall

    print(f"\n  --- Variance decomposition ---")
    print(f"  Total variance:         {total_var:.6f}")
    print(f"  Between-word variance:  {between_word_var:.6f} ({between_word_var/total_var*100:.1f}%)")
    print(f"  Between-band variance:  {between_band_var:.6f} ({between_band_var/total_var*100:.1f}%)")
    print(f"  (High between-band = bands specialize; High between-word = words differ)")

    return {
        'stab_matrix': stab_matrix,
        'mean_stab': mean_stab,
        'std_stab': std_stab,
        'cross_word_corr': mean_corr,
    }


# ─── Layer evolution: does separation emerge through depth? ──────────

def depth_evolution(all_results):
    """Track how the stability/variability split develops through layers."""
    words = list(all_results.keys())
    layer_names = list(all_results[words[0]].keys())

    print(f"\n{'='*70}")
    print(f"  DEPTH EVOLUTION: When does band specialization emerge?")
    print(f"{'='*70}")

    for ln in layer_names:
        stabs = []
        for word in words:
            stabs.append(all_results[word][ln]['phase_stability'])
        stab_matrix = np.array(stabs)

        mean_stab = stab_matrix.mean()
        n_stable = (stab_matrix > 0.9).sum() / stab_matrix.size * 100
        n_variable = (stab_matrix < 0.5).sum() / stab_matrix.size * 100

        # Cross-word correlation at this layer
        corrs = []
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                corrs.append(np.corrcoef(stab_matrix[i], stab_matrix[j])[0, 1])
        mean_corr = np.mean(corrs)

        print(f"  {ln:<6} | mean stab={mean_stab:.3f} | stable(>0.9)={n_stable:>5.1f}% | "
              f"variable(<0.5)={n_variable:>5.1f}% | cross-word r={mean_corr:.3f}")


# ─── Frequency-stability correlation (the 5a+5c join) ────────────────

def frequency_stability_correlation(model, dataset, all_results, model_label):
    """
    THE KEY QUESTION: do stable bands (identity) have different phase velocity
    through depth than variable bands (context)?

    Prediction: stable bands = slow velocity, variable bands = fast velocity.
    If true, frequency IS the mechanism of identity vs context.
    """
    from depth_frequency_5a import extract_layer_states, phase_velocity

    print(f"\n{'='*70}")
    print(f"  FREQUENCY-STABILITY CORRELATION: {model_label}")
    print(f"  Do stable bands move slower through depth than variable bands?")
    print(f"{'='*70}")

    # Get per-band phase velocity from depth-axis analysis (Exp 5a data)
    states = extract_layer_states(model, dataset, n_batches=5)
    magnitude, phase = decompose_bands(states)
    delta = phase_velocity(phase)
    per_band_velocity = np.abs(delta).mean(axis=(0, 1))  # (n_bands,) mean |dp| per band

    words = list(all_results.keys())
    layer_names = list(all_results[words[0]].keys())
    final_layer = layer_names[-1]

    # Per-word correlation: stability vs velocity for each band
    print(f"\n  --- Per-word: correlation(phase_stability, band_velocity) ---")
    print(f"  {'Word':<10} {'r (stab vs vel)':>16} {'p-direction':>14}")
    all_stab = []
    for word in words:
        stab = all_results[word][final_layer]['phase_stability']  # (n_bands,)
        all_stab.append(stab)
        r = np.corrcoef(stab, per_band_velocity)[0, 1]
        direction = "stable=slow" if r < 0 else "stable=fast"
        print(f"  {word:<10} {r:>16.4f} {direction:>14}")

    # Mean stability across all words vs velocity
    mean_stab = np.mean(all_stab, axis=0)  # (n_bands,)
    r_global = np.corrcoef(mean_stab, per_band_velocity)[0, 1]
    print(f"\n  Global (mean stability across words):")
    print(f"  r = {r_global:.4f}")

    # Split analysis: compare velocity of stable vs variable bands
    # Use median stability as the split point
    median_stab = np.median(mean_stab)
    stable_mask = mean_stab >= median_stab
    variable_mask = mean_stab < median_stab

    vel_stable = per_band_velocity[stable_mask].mean()
    vel_variable = per_band_velocity[variable_mask].mean()
    ratio = vel_variable / vel_stable if vel_stable > 0 else float('inf')

    print(f"\n  --- Split analysis (median split on mean stability) ---")
    print(f"  Stable bands (n={stable_mask.sum()}):   mean velocity = {vel_stable:.4f}")
    print(f"  Variable bands (n={variable_mask.sum()}): mean velocity = {vel_variable:.4f}")
    print(f"  Variable/Stable velocity ratio: {ratio:.4f}")

    if ratio > 1.05:
        print(f"  -> Variable bands move FASTER through depth ({(ratio-1)*100:.1f}% faster)")
        print(f"     Frequency IS the mechanism: fast bands carry context, slow bands carry identity")
    elif ratio < 0.95:
        print(f"  -> Variable bands move SLOWER through depth ({(1-ratio)*100:.1f}% slower)")
        print(f"     OPPOSITE of prediction: context bands are slow, identity bands are fast")
    else:
        print(f"  -> No meaningful velocity difference ({(ratio-1)*100:+.1f}%)")
        print(f"     Frequency and stability are independent mechanisms")

    # Per-word split: for each word, do ITS variable bands move faster?
    print(f"\n  --- Per-word: velocity of stable vs variable bands ---")
    print(f"  {'Word':<10} {'Vel(stable)':>12} {'Vel(variable)':>14} {'Ratio':>8} {'Direction':>14}")
    for word in words:
        stab = all_results[word][final_layer]['phase_stability']
        word_median = np.median(stab)
        s_mask = stab >= word_median
        v_mask = stab < word_median
        vs = per_band_velocity[s_mask].mean()
        vv = per_band_velocity[v_mask].mean()
        wr = vv / vs if vs > 0 else float('inf')
        direction = "ctx=fast" if wr > 1.0 else "id=fast"
        print(f"  {word:<10} {vs:>12.4f} {vv:>14.4f} {wr:>8.4f} {direction:>14}")

    # Magnitude stability vs velocity correlation
    print(f"\n  --- Magnitude CV vs velocity (alternative stability measure) ---")
    all_mag_cv = []
    for word in words:
        mc = all_results[word][final_layer]['mag_cv']
        all_mag_cv.append(mc)
    mean_mag_cv = np.mean(all_mag_cv, axis=0)  # (n_bands,)
    r_mag = np.corrcoef(mean_mag_cv, per_band_velocity)[0, 1]
    print(f"  r(mean_mag_CV, velocity) = {r_mag:.4f}")
    print(f"  (Positive = high-CV bands move faster; Negative = high-CV bands move slower)")

    # Band group breakdown: velocity vs stability side by side
    print(f"\n  --- Band groups: velocity vs stability side by side ---")
    print(f"  {'Bands':<10} {'Mean vel':>10} {'Mean stab':>10} {'Mean mag CV':>12}")
    for g in range(0, N_BANDS, 8):
        v = per_band_velocity[g:g+8].mean()
        s = mean_stab[g:g+8].mean()
        m = mean_mag_cv[g:g+8].mean()
        print(f"  {g+1:>2}-{min(g+8, N_BANDS):<6} {v:>10.4f} {s:>10.4f} {m:>12.4f}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Experiment 5c: Band Aspect Separation")
    print("  Do different bands encode different aspects of meaning?")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"  Dataset: {len(text)} chars, {dataset.vocab_size} vocab")

    # Verify target words exist in vocabulary
    test_sets = build_test_sets()
    print(f"\n  Target words: {', '.join(ts['word'] for ts in test_sets)}")
    for ts in test_sets:
        word = ts['word']
        missing = [c for c in word if c not in dataset.stoi]
        if missing:
            print(f"  WARNING: '{word}' has chars not in vocab: {missing}")

    # Verify target positions are correct
    print(f"\n  Verifying sentence construction...")
    for ts in test_sets:
        word = ts['word']
        for text_str, pos in ts['sentences']:
            # Check that the target word starts at the indicated position
            actual = text_str[pos:pos+len(word)]
            if actual != word:
                print(f"  ERROR: '{word}' expected at pos {pos} in '{text_str}', found '{actual}'")
            else:
                pass  # Good

    # Train Maestro+curriculum model (best config)
    print(f"\n  Training Maestro + curriculum model...")
    model = train_model(dataset, mode="kerr", use_maestro=True, curriculum=True)

    # Also train MLP for comparison
    print(f"\n  Training MLP baseline...")
    model_mlp = train_model(dataset, mode="mlp", use_maestro=False, curriculum=False)

    for model_label, mdl in [("Maestro+curriculum", model), ("MLP baseline", model_mlp)]:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_label}")
        print(f"{'='*70}")

        # Analyse each word
        all_results = {}
        for ts in test_sets:
            word = ts['word']
            results = analyse_band_stability(mdl, dataset, ts)
            all_results[word] = results

            ps, mc = print_band_analysis(word, results)

        # Cross-word comparison
        cross = cross_word_analysis(all_results)

        # Depth evolution
        depth_evolution(all_results)

        # Frequency-stability correlation (5c + 5a join)
        frequency_stability_correlation(mdl, dataset, all_results, model_label)

    print(f"\n{'='*70}")
    print(f"  INTERPRETATION")
    print(f"{'='*70}")
    print(f"  Phase stability near 1.0 = band encodes same thing regardless of context")
    print(f"  Phase stability near 0.0 = band changes completely with context")
    print(f"  Cross-word correlation near 1.0 = structural bands (not semantic)")
    print(f"  Cross-word correlation near 0.0 = each word has unique stable bands")
    print(f"")
    print(f"  The 'cup of soup' hypothesis:")
    print(f"  - If clean separation: some bands = 'what it is', other bands = 'what's happening'")
    print(f"  - If entangled: all bands carry mixed identity + context (no clean split)")
    print(f"  - If structural: stability pattern is about band frequency, not word meaning")
    print(f"")
    print(f"  Frequency-stability prediction:")
    print(f"  - Negative correlation = stable bands are slow, contextual bands are fast")
    print(f"  - Positive correlation = stable bands are fast, contextual bands are slow")
    print(f"  - Zero correlation = frequency and stability are independent")
    print(f"{'='*70}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  Total time: {time.time() - t0:.1f}s")
