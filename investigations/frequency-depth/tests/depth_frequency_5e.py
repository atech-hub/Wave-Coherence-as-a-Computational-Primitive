"""
Experiment 5e: The Death Anomaly — Does Abstraction Level Predict Band Stability?

5c found death had 31/64 stable bands vs 53-60 for king/sword/love/hand.
Hypothesis: abstract words have fewer stable bands because their meaning
is more context-dependent.

Test: concrete nouns vs abstract nouns, each in 6 contextual sentences.
If abstraction predicts stability count, the model encodes abstractness
as contextual dependence — more bands needed to represent "what it means
in THIS context" vs fewer for a concrete thing that IS what it IS.

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
from depth_frequency_5c import extract_token_states, analyse_band_stability


# ─── Word sets: concrete vs abstract ─────────────────────────────────

def build_test_sets():
    """
    Two groups: concrete nouns (physical, tangible) and abstract nouns
    (conceptual, intangible). All common in Shakespeare.
    Auto-compute positions using str.index().
    """
    raw_sets = {
        # CONCRETE — physical objects/beings you can see and touch
        'stone': [
            "the stone lay cold upon the ground",
            "he threw the stone across the river",
            "a heavy stone blocked the way ahead",
            "she carved the stone with careful hands",
            "the stone wall stood for many years",
            "upon the stone he wrote his name",
        ],
        'blood': [
            "the blood ran cold within his veins",
            "she washed the blood from her hands",
            "his blood was royal born and true",
            "the blood of kings shall not be shed",
            "upon the ground the blood did pool",
            "with blood and tears the war was won",
        ],
        'sword': [
            "he drew his sword and charged ahead",
            "the sword was broken in the fight",
            "she held the sword above her head",
            "a golden sword lay on the table",
            "the sword cut through the heavy armor",
            "his father gave him the old sword",
        ],
        'horse': [
            "the horse rode swiftly through the night",
            "he fed the horse with grain and hay",
            "a white horse stood beside the gate",
            "the horse was tired from the journey",
            "she mounted the horse and rode away",
            "upon his horse the knight did sit",
        ],
        'crown': [
            "the crown sat heavy on his head",
            "she placed the crown upon the altar",
            "a golden crown was his reward",
            "the crown fell to the dusty ground",
            "he wore the crown with solemn grace",
            "without the crown he was but a man",
        ],

        # ABSTRACT — concepts, emotions, states you cannot touch
        'death': [
            "he feared death more than all else",
            "death came swiftly in the night",
            "she laughed in the face of death",
            "after death there is but silence",
            "the death of the prince was mourned",
            "a noble death upon the battlefield",
        ],
        'hope': [
            "all hope was lost among the ruins",
            "she held on to hope with all her might",
            "there is no hope for the wicked",
            "with hope and faith they carried on",
            "the last hope faded with the dawn",
            "his hope was stronger than his fear",
        ],
        'grief': [
            "the grief consumed her very soul",
            "he bore his grief in silent pain",
            "such grief was never seen before",
            "with grief and sorrow she did weep",
            "the grief of losing him was great",
            "in grief she wandered through the halls",
        ],
        'fear': [
            "the fear was written on his face",
            "she knew no fear before the battle",
            "with fear and trembling he did speak",
            "the fear of god was in his heart",
            "his fear was greater than his pride",
            "in fear they fled into the woods",
        ],
        'pride': [
            "his pride would be his greatest fall",
            "she spoke with pride about her son",
            "the pride of kings is but a shadow",
            "with pride he raised the battle flag",
            "such pride was never meant to last",
            "in pride and glory he did stand",
        ],
    }

    test_sets = []
    for word, sentences in raw_sets.items():
        sent_list = []
        for s in sentences:
            pos = s.index(word)
            sent_list.append((s, pos))
        test_sets.append({'word': word, 'sentences': sent_list})

    return test_sets


CONCRETE_WORDS = {'stone', 'blood', 'sword', 'horse', 'crown'}
ABSTRACT_WORDS = {'death', 'hope', 'grief', 'fear', 'pride'}


# ─── Analysis ────────────────────────────────────────────────────────

def count_stable_bands(results, layer_name, threshold=0.9):
    """Count bands with phase stability above threshold."""
    ps = results[layer_name]['phase_stability']
    return (ps > threshold).sum()


def count_variable_bands(results, layer_name, threshold=0.5):
    """Count bands with phase stability below threshold."""
    ps = results[layer_name]['phase_stability']
    return (ps < threshold).sum()


def main():
    print("=" * 70)
    print("  Experiment 5e: The Death Anomaly")
    print("  Does abstraction level predict band stability?")
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
    print(f"\n{'='*70}")
    print(f"  PER-WORD RESULTS")
    print(f"{'='*70}")

    layer_names = None
    for ts in test_sets:
        word = ts['word']
        word_type = "CONCRETE" if word in CONCRETE_WORDS else "ABSTRACT"
        results = analyse_band_stability(model, dataset, ts)
        all_results[word] = results

        if layer_names is None:
            layer_names = list(results.keys())

        final = layer_names[-1]
        n_stable = count_stable_bands(results, final)
        n_variable = count_variable_bands(results, final)
        mean_stab = results[final]['phase_stability'].mean()

        print(f"  {word:<8} [{word_type:<8}] | stable(>0.9): {n_stable:>2}/64 | "
              f"variable(<0.5): {n_variable:>2}/64 | mean stab: {mean_stab:.4f}")

    # Group comparison
    print(f"\n{'='*70}")
    print(f"  CONCRETE vs ABSTRACT COMPARISON")
    print(f"{'='*70}")

    final = layer_names[-1]

    concrete_stabs = []
    abstract_stabs = []
    concrete_counts = []
    abstract_counts = []
    concrete_variable = []
    abstract_variable = []

    for word, results in all_results.items():
        ps = results[final]['phase_stability']
        mean_s = ps.mean()
        n_stable = (ps > 0.9).sum()
        n_var = (ps < 0.5).sum()

        if word in CONCRETE_WORDS:
            concrete_stabs.append(mean_s)
            concrete_counts.append(n_stable)
            concrete_variable.append(n_var)
        else:
            abstract_stabs.append(mean_s)
            abstract_counts.append(n_stable)
            abstract_variable.append(n_var)

    print(f"\n  {'':20} {'Mean stab':>10} {'Stable(>0.9)':>14} {'Variable(<0.5)':>16}")
    print(f"  {'Concrete (5 words)':20} {np.mean(concrete_stabs):>10.4f} "
          f"{np.mean(concrete_counts):>14.1f} {np.mean(concrete_variable):>16.1f}")
    print(f"  {'Abstract (5 words)':20} {np.mean(abstract_stabs):>10.4f} "
          f"{np.mean(abstract_counts):>14.1f} {np.mean(abstract_variable):>16.1f}")

    gap = np.mean(concrete_stabs) - np.mean(abstract_stabs)
    print(f"\n  Gap (concrete - abstract): {gap:+.4f}")

    # Per-word detail sorted by stability
    print(f"\n  --- All words ranked by mean stability ---")
    ranked = sorted(all_results.items(),
                    key=lambda x: x[1][final]['phase_stability'].mean(),
                    reverse=True)
    print(f"  {'Rank':>4} {'Word':<8} {'Type':<10} {'Mean stab':>10} {'Stable':>8} {'Variable':>10}")
    for i, (word, results) in enumerate(ranked, 1):
        ps = results[final]['phase_stability']
        word_type = "CONCRETE" if word in CONCRETE_WORDS else "ABSTRACT"
        print(f"  {i:>4} {word:<8} {word_type:<10} {ps.mean():>10.4f} "
              f"{(ps>0.9).sum():>8} {(ps<0.5).sum():>10}")

    # Cross-word stability correlation (do concrete words cluster?)
    print(f"\n{'='*70}")
    print(f"  CROSS-WORD STABILITY CORRELATION")
    print(f"{'='*70}")

    words = list(all_results.keys())
    n_words = len(words)
    stab_matrix = np.array([all_results[w][final]['phase_stability'] for w in words])

    # Within-group vs between-group correlation
    within_concrete = []
    within_abstract = []
    between = []

    for i in range(n_words):
        for j in range(i+1, n_words):
            r = np.corrcoef(stab_matrix[i], stab_matrix[j])[0, 1]
            w1, w2 = words[i], words[j]
            both_concrete = w1 in CONCRETE_WORDS and w2 in CONCRETE_WORDS
            both_abstract = w1 in ABSTRACT_WORDS and w2 in ABSTRACT_WORDS
            if both_concrete:
                within_concrete.append(r)
            elif both_abstract:
                within_abstract.append(r)
            else:
                between.append(r)

    print(f"\n  Within concrete pairs: mean r = {np.mean(within_concrete):.3f} (n={len(within_concrete)})")
    print(f"  Within abstract pairs: mean r = {np.mean(within_abstract):.3f} (n={len(within_abstract)})")
    print(f"  Between groups:        mean r = {np.mean(between):.3f} (n={len(between)})")

    if np.mean(within_concrete) > np.mean(between) and np.mean(within_abstract) > np.mean(between):
        print(f"  -> Concrete and abstract words cluster separately in band stability space")
    elif abs(np.mean(within_concrete) - np.mean(between)) < 0.05:
        print(f"  -> No clustering by abstraction level")

    # Depth evolution: does the concrete/abstract gap grow through depth?
    print(f"\n{'='*70}")
    print(f"  DEPTH EVOLUTION: Does the gap grow through layers?")
    print(f"{'='*70}")

    print(f"\n  {'Layer':<6} {'Concrete':>10} {'Abstract':>10} {'Gap':>10}")
    for ln in layer_names:
        c_stab = np.mean([all_results[w][ln]['phase_stability'].mean()
                         for w in CONCRETE_WORDS])
        a_stab = np.mean([all_results[w][ln]['phase_stability'].mean()
                         for w in ABSTRACT_WORDS])
        print(f"  {ln:<6} {c_stab:>10.4f} {a_stab:>10.4f} {c_stab - a_stab:>+10.4f}")

    # Verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    if gap > 0.05:
        print(f"\n  CONFIRMED: Concrete words have higher band stability than abstract words.")
        print(f"  Gap = {gap:+.4f}. Abstraction level predicts contextual dependence.")
        print(f"  The model encodes abstract concepts with more context-sensitive bands")
        print(f"  because their meaning depends more on surrounding context.")
    elif gap < -0.05:
        print(f"\n  REVERSED: Abstract words are MORE stable than concrete words.")
        print(f"  Gap = {gap:+.4f}. Opposite of prediction.")
    else:
        print(f"\n  NULL: No meaningful difference between concrete and abstract.")
        print(f"  Gap = {gap:+.4f}. Abstraction level does not predict band stability.")
        print(f"  The death anomaly from 5c may be word-specific, not category-level.")

    # Check if death is still an outlier
    death_stab = all_results['death'][final]['phase_stability'].mean()
    other_abstract = [all_results[w][final]['phase_stability'].mean()
                     for w in ABSTRACT_WORDS if w != 'death']
    death_vs_abstract = death_stab - np.mean(other_abstract)
    print(f"\n  Death check: {death_stab:.4f} vs other abstract mean {np.mean(other_abstract):.4f}")
    print(f"  Death offset from abstract group: {death_vs_abstract:+.4f}")
    if abs(death_vs_abstract) > 0.05:
        print(f"  Death is still an outlier even within abstract words.")
    else:
        print(f"  Death is typical for abstract words.")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  Total time: {time.time() - t0:.1f}s")
