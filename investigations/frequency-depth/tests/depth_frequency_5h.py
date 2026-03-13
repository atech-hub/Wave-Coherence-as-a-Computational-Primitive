"""
Experiment 5h: The Love-Hand Correlation

5c found love and hand had r=0.587 in MLP cross-word stability -- by far the
highest pair. Both sit at the concrete/abstract boundary (hand is physical but
defined by action, love is abstract but physically expressed).

Questions:
1. Is the love-hand correlation reproducible? (Could be noise at n=6)
2. Does it appear in Maestro models too, or only MLP?
3. Do other boundary words cluster similarly?
4. Is there a gradient of stability correlation from concrete->boundary->abstract?

Method: 15 words spanning concrete, boundary, and abstract categories.
Each in 6 contexts. Measure cross-word stability correlation at L3.
Build a full correlation matrix and test whether boundary words form a cluster.

Uses Maestro+curriculum and MLP models for comparison.
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


# --- Word sets: concrete / boundary / abstract ---

def build_test_sets():
    """
    Three groups:
    - CONCRETE: purely physical objects
    - BOUNDARY: physical things defined by action/relation, or abstract things with physical expression
    - ABSTRACT: purely conceptual/emotional

    All common in Shakespeare. 6 contexts each. Auto-compute positions.
    """
    raw_sets = {
        # CONCRETE -- purely physical
        'stone': [
            "the stone lay cold upon the ground",
            "he threw the stone across the river",
            "a heavy stone blocked the way ahead",
            "she carved the stone with careful hands",
            "the stone wall stood for many years",
            "upon the stone he wrote his name",
        ],
        'sword': [
            "he drew his sword from the sheath",
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
        'blood': [
            "the blood ran cold within his veins",
            "she washed the blood from her hands",
            "his blood was royal born and true",
            "the blood of kings shall not be shed",
            "upon the ground the blood did pool",
            "with blood and tears the war was won",
        ],

        # BOUNDARY -- physical/action or abstract/embodied
        'hand': [
            "he took her hand and led her forth",
            "with heavy hand he struck the blow",
            "her gentle hand upon his cheek",
            "the hand that wrote these words is gone",
            "by my own hand i seal this fate",
            "his cold hand lay upon the stone",
        ],
        'heart': [
            "his heart was heavy with the news",
            "she held her heart and wept with pain",
            "the heart of the kingdom lay in ruin",
            "with all his heart he loved the queen",
            "a broken heart cannot be mended",
            "the heart knows what the mind cannot",
        ],
        'fire': [
            "the fire burned bright within the hall",
            "she lit the fire against the cold",
            "his fire and passion moved the crowd",
            "the fire of war consumed the land",
            "a fire in her eyes showed her rage",
            "the fire died down to glowing coals",
        ],
        'eye': [
            "her eye was sharp and missed nothing",
            "the eye of the storm was upon them",
            "with a keen eye he watched the door",
            "the eye sees what it wants to see",
            "his eye fell upon the golden ring",
            "she could not meet his steady eye",
        ],
        'voice': [
            "his voice rang out across the hall",
            "she raised her voice above the crowd",
            "the voice of reason called to him",
            "with gentle voice he spoke to her",
            "a voice from the darkness answered",
            "the voice of the people must be heard",
        ],

        # ABSTRACT -- purely conceptual/emotional
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


CONCRETE_WORDS = {'stone', 'sword', 'horse', 'crown', 'blood'}
BOUNDARY_WORDS = {'hand', 'heart', 'fire', 'eye', 'voice'}
ABSTRACT_WORDS = {'death', 'hope', 'grief', 'fear', 'pride'}


def word_category(word):
    if word in CONCRETE_WORDS:
        return 'CONCRETE'
    elif word in BOUNDARY_WORDS:
        return 'BOUNDARY'
    else:
        return 'ABSTRACT'


# --- Analysis ---

def full_correlation_matrix(all_results, layer_name):
    """Build full cross-word stability correlation matrix."""
    words = list(all_results.keys())
    n = len(words)
    stab_matrix = np.array([all_results[w][layer_name]['phase_stability'] for w in words])

    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i, j] = 1.0
            else:
                corr[i, j] = np.corrcoef(stab_matrix[i], stab_matrix[j])[0, 1]
    return corr, words


def group_analysis(corr, words):
    """Compute within-group and between-group mean correlations."""
    n = len(words)
    groups = {}

    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = word_category(words[i]), word_category(words[j])
            pair = tuple(sorted([c1.lower(), c2.lower()]))
            key = f"{pair[0]}-{pair[1]}"
            if key not in groups:
                groups[key] = []
            groups[key].append(corr[i, j])

    return groups


def main():
    print("=" * 70)
    print("  Experiment 5h: The Love-Hand Correlation")
    print("  Do boundary words form a stability cluster?")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"  Dataset: {len(text)} chars, {dataset.vocab_size} vocab")

    test_sets = build_test_sets()

    # Verify
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
        print(f"  All {sum(len(ts['sentences']) for ts in test_sets)} sentences verified.")

    # Train both models
    for model_label, mode, use_maestro, curriculum in [
        ("Maestro+curriculum", "kerr", True, True),
        ("MLP baseline", "mlp", False, False),
    ]:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_label}")
        print(f"{'='*70}")

        print(f"\n  Training {model_label}...")
        model = train_model(dataset, mode=mode, use_maestro=use_maestro, curriculum=curriculum)

        # Analyse each word
        all_results = {}
        print(f"\n  Analysing {len(test_sets)} words...")
        for ts in test_sets:
            word = ts['word']
            results = analyse_band_stability(model, dataset, ts)
            all_results[word] = results

        layer_names = list(all_results[test_sets[0]['word']].keys())
        final = layer_names[-1]

        # Per-word summary
        print(f"\n  --- Per-word stability ({final}) ---")
        print(f"  {'Word':<8} {'Category':<10} {'Mean stab':>10} {'Stable(>0.9)':>14} {'Variable(<0.5)':>16}")
        for ts in test_sets:
            word = ts['word']
            ps = all_results[word][final]['phase_stability']
            cat = word_category(word)
            print(f"  {word:<8} {cat:<10} {ps.mean():>10.4f} "
                  f"{(ps > 0.9).sum():>14} {(ps < 0.5).sum():>16}")

        # Full correlation matrix
        corr, words = full_correlation_matrix(all_results, final)

        print(f"\n  --- Cross-word stability correlation ({final}) ---")
        # Print header
        print(f"  {'':8}", end="")
        for w in words:
            print(f" {w[:5]:>6}", end="")
        print()
        for i, w1 in enumerate(words):
            cat = word_category(w1)[0]  # first letter
            print(f"  {w1:<7}{cat}", end="")
            for j, w2 in enumerate(words):
                if i == j:
                    print(f" {'--':>6}", end="")
                else:
                    print(f" {corr[i,j]:>6.3f}", end="")
            print()

        # Group statistics
        groups = group_analysis(corr, words)

        print(f"\n  --- Group mean correlations ---")
        print(f"  {'Group':<25} {'Mean r':>8} {'n pairs':>8} {'Std':>8}")
        for grp, vals in sorted(groups.items()):
            if vals:
                print(f"  {grp:<25} {np.mean(vals):>8.3f} {len(vals):>8} {np.std(vals):>8.3f}")

        # Key test: does boundary cluster more than between-group?
        within_boundary = groups.get('boundary-boundary', [])
        concrete_abstract = groups.get('abstract-concrete', [])

        print(f"\n  --- Key comparisons ---")
        print(f"  Within-boundary mean r:     {np.mean(within_boundary):.3f}" if within_boundary else "  Within-boundary: no data")
        print(f"  Concrete-abstract mean r:   {np.mean(concrete_abstract):.3f}" if concrete_abstract else "  Concrete-abstract: no data")
        within_concrete = groups.get('concrete-concrete', [])
        within_abstract = groups.get('abstract-abstract', [])
        print(f"  Within-concrete mean r:     {np.mean(within_concrete):.3f}" if within_concrete else "  Within-concrete: no data")
        print(f"  Within-abstract mean r:     {np.mean(within_abstract):.3f}" if within_abstract else "  Within-abstract: no data")

        # Specifically check love->hand equivalent: hand correlations
        hand_idx = words.index('hand') if 'hand' in words else -1
        if hand_idx >= 0:
            print(f"\n  --- 'hand' correlations (the original anomaly) ---")
            hand_corrs = [(words[j], corr[hand_idx, j]) for j in range(len(words)) if j != hand_idx]
            hand_corrs.sort(key=lambda x: -x[1])
            for w, r in hand_corrs:
                cat = word_category(w)
                print(f"    hand <-> {w:<8} [{cat:<10}]: r = {r:.3f}")

        # Category mean stability
        print(f"\n  --- Category mean stability ---")
        for cat_name, cat_set in [('CONCRETE', CONCRETE_WORDS), ('BOUNDARY', BOUNDARY_WORDS), ('ABSTRACT', ABSTRACT_WORDS)]:
            stabs = [all_results[w][final]['phase_stability'].mean() for w in cat_set]
            print(f"  {cat_name:<10}: {np.mean(stabs):.4f} (range {np.min(stabs):.4f} - {np.max(stabs):.4f})")

        # Gradient test: is there a monotonic decrease concrete > boundary > abstract?
        c_mean = np.mean([all_results[w][final]['phase_stability'].mean() for w in CONCRETE_WORDS])
        b_mean = np.mean([all_results[w][final]['phase_stability'].mean() for w in BOUNDARY_WORDS])
        a_mean = np.mean([all_results[w][final]['phase_stability'].mean() for w in ABSTRACT_WORDS])

        print(f"\n  Stability gradient: concrete({c_mean:.4f}) > boundary({b_mean:.4f}) > abstract({a_mean:.4f})?")
        if c_mean > b_mean > a_mean:
            print(f"  -> YES: monotonic decrease. Gap C-B: {c_mean-b_mean:.4f}, B-A: {b_mean-a_mean:.4f}")
        elif c_mean > a_mean and b_mean > a_mean:
            print(f"  -> PARTIAL: concrete and boundary both above abstract, but no C>B ordering")
        else:
            print(f"  -> NO: not monotonic")

        # Top-5 and bottom-5 most correlated pairs
        pairs = []
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                pairs.append((words[i], words[j], corr[i, j],
                             word_category(words[i]), word_category(words[j])))
        pairs.sort(key=lambda x: -x[2])

        print(f"\n  --- Top 10 most correlated pairs ---")
        for w1, w2, r, c1, c2 in pairs[:10]:
            print(f"    {w1:<8} <-> {w2:<8}  r={r:>6.3f}  [{c1[0]}-{c2[0]}]")

        print(f"\n  --- Bottom 5 least correlated pairs ---")
        for w1, w2, r, c1, c2 in pairs[-5:]:
            print(f"    {w1:<8} <-> {w2:<8}  r={r:>6.3f}  [{c1[0]}-{c2[0]}]")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")
    print(f"\n  Examine the results above to determine:")
    print(f"  1. Is the love-hand correlation from 5c reproducible?")
    print(f"     (Note: love is not tested here — replaced by boundary words)")
    print(f"  2. Do boundary words cluster with each other?")
    print(f"  3. Is there a stability gradient: concrete > boundary > abstract?")
    print(f"  4. Does the MLP show different grouping than Maestro?")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  Total time: {time.time() - t0:.1f}s")
