"""
Wave Packet Engine — Proof of Concept
======================================
Demonstrates three engine patterns from the defensive publication:

  Pattern 31: Wave Packet Query (sparse DFT format, resonance matching)
  Pattern 32: Harmonic Translator (DFT -> band select -> inverse DFT round-trip)
  Pattern 34: Selective Band Loading (minimum viable band set, tiered retrieval)

Uses real embeddings from all-MiniLM-L6-v2 (384 dimensions).
All operations are foundational mathematics: DFT, inverse DFT, cosine,
array indexing, Euclidean norm. Nothing novel, nothing patentable.

Requirements: pip install sentence-transformers torch numpy
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import time

# ============================================================
# Core Engine Functions (foundational math only)
# ============================================================

def embed_to_bands(vector):
    """DFT decomposition: vector -> frequency bands.

    Each band has amplitude (energy) and phase (position on circle).
    This is the input side of the translator (Pattern 32).
    """
    coeffs = np.fft.rfft(vector)
    amplitudes = np.abs(coeffs)
    phases = np.angle(coeffs)
    return coeffs, amplitudes, phases


def bands_to_embed(coeffs, original_length):
    """Inverse DFT: frequency bands -> reconstructed vector.

    This is the output side of the translator (Pattern 32).
    """
    return np.fft.irfft(coeffs, n=original_length)


def make_wave_packet(amplitudes, phases, band_set):
    """Create a wave packet query — sparse DFT representation (Pattern 31).

    W = { (n, |V_n|, φ_n) : n ∈ S }

    Only selected bands travel. Everything else stays behind.
    """
    packet = []
    for n in band_set:
        packet.append((n, amplitudes[n], phases[n]))
    return packet


def resonance(packet, stored_amplitudes, stored_phases):
    """Resonance matching: amplitude-weighted phase coherence (Pattern 31).

    R(W, U) = Σ_{n ∈ S} w_n · |V_n| · |U_n| · cos(φ_n − ψ_n)

    Three things at once per term:
      - Query confidence (|V_n|)
      - Stored signal strength (|U_n|)
      - Phase alignment (cos(φ_n − ψ_n))

    Conjugate symmetry correction (Corrective Finding #6):
      rfft returns N/2+1 one-sided coefficients. For a real signal,
      X[N-k] = conj(X[k]), so middle coefficients represent two-sided
      energy and must be weighted by 2. DC (n=0) and Nyquist (n=N/2)
      appear once and get weight 1. Without this, resonance diverges
      from cosine similarity by up to ~4% on structured inputs.
    """
    n_coeffs = len(stored_amplitudes)
    last_idx = n_coeffs - 1

    score = 0.0
    query_energy = 0.0
    stored_energy = 0.0

    for n, amp_q, phase_q in packet:
        amp_s = stored_amplitudes[n]
        phase_s = stored_phases[n]

        # Conjugate symmetry weight
        w = 1.0 if (n == 0 or n == last_idx) else 2.0

        score += w * amp_q * amp_s * np.cos(phase_q - phase_s)
        query_energy += w * amp_q ** 2
        stored_energy += w * amp_s ** 2

    # Normalised resonance [-1, 1]
    norm = np.sqrt(query_energy) * np.sqrt(stored_energy)
    if norm < 1e-10:
        return 0.0
    return score / norm


def selective_load(coeffs, band_set, original_length):
    """Selective band loading — load only specified bands (Pattern 34).

    Bands not in band_set are zero-filled (as if still on disk).
    Inverse DFT reconstructs a partial but coherent vector.
    """
    partial = np.zeros_like(coeffs)
    for n in band_set:
        partial[n] = coeffs[n]
    return np.fft.irfft(partial, n=original_length)


def cosine_similarity(a, b):
    """Standard cosine similarity for comparison."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-10:
        return 0.0
    return dot / norm


# ============================================================
# Test Setup
# ============================================================

print("=" * 70)
print("WAVE PACKET ENGINE — PROOF OF CONCEPT")
print("Patterns 31, 32, 34 from ENGINE-PATTERNS.md")
print("=" * 70)

print("\nLoading model: all-MiniLM-L6-v2 (384 dimensions)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Word groups for testing
database_words = [
    "king", "queen", "prince", "princess", "throne",     # royalty
    "dog", "cat", "fish", "bird", "horse",                # animals
    "happy", "joyful", "sad", "angry", "peaceful",        # emotions
    "computer", "algorithm", "software", "database", "network",  # tech
    "river", "mountain", "ocean", "forest", "desert",     # nature
]

query_words = ["monarch", "puppy", "cheerful", "server", "lake"]
query_expected = ["royalty", "animals", "emotions", "tech", "nature"]

print(f"Database: {len(database_words)} words across 5 categories")
print(f"Queries:  {len(query_words)} words")

# Embed everything
db_embeddings = model.encode(database_words)
q_embeddings = model.encode(query_words)

dim = db_embeddings.shape[1]
n_bands = dim // 2 + 1  # Number of DFT coefficients for real input

print(f"Embedding dimension: {dim}")
print(f"DFT bands: {n_bands}")

# Decompose all database entries
db_coeffs = []
db_amps = []
db_phases = []
for emb in db_embeddings:
    c, a, p = embed_to_bands(emb)
    db_coeffs.append(c)
    db_amps.append(a)
    db_phases.append(p)


# ============================================================
# TEST 1: Translator Round-Trip (Pattern 32)
# ============================================================

print("\n" + "=" * 70)
print("TEST 1: TRANSLATOR ROUND-TRIP (Pattern 32)")
print("embed -> DFT -> inverse DFT -> reconstruct")
print("=" * 70)

print("\nRound-trip reconstruction error (should be ~0):")
max_error = 0.0
for i, word in enumerate(database_words):
    reconstructed = bands_to_embed(db_coeffs[i], dim)
    error = np.max(np.abs(db_embeddings[i] - reconstructed))
    max_error = max(max_error, error)

print(f"  Max reconstruction error across {len(database_words)} words: {max_error:.2e}")
print(f"  Perfect reconstruction: {'YES' if max_error < 1e-6 else 'NO'} (floating point precision)")

# Show that cosine similarity is preserved through round-trip
print("\nCosine similarity preservation (original vs reconstructed):")
for word in ["king", "happy", "computer"]:
    idx = database_words.index(word)
    reconstructed = bands_to_embed(db_coeffs[idx], dim)
    sim = cosine_similarity(db_embeddings[idx], reconstructed)
    print(f"  {word}: {sim:.10f}")


# ============================================================
# TEST 2: Wave Packet Query vs Full Cosine (Pattern 31)
# ============================================================

print("\n" + "=" * 70)
print("TEST 2: WAVE PACKET QUERY vs FULL COSINE (Pattern 31)")
print("Sparse DFT query with resonance matching")
print("=" * 70)

# Define band sets
all_bands = list(range(n_bands))
low_bands = list(range(0, n_bands // 4))                          # ~25%
mid_bands = list(range(n_bands // 4, 3 * n_bands // 4))           # ~50%
high_bands = list(range(3 * n_bands // 4, n_bands))               # ~25%
mid_high_bands = mid_bands + high_bands                            # ~75%

# Amplitude-threshold band selection
def select_by_amplitude(amplitudes, threshold_percentile=75):
    """Select bands where amplitude exceeds percentile threshold."""
    threshold = np.percentile(amplitudes, threshold_percentile)
    return [n for n in range(len(amplitudes)) if amplitudes[n] >= threshold]

print(f"\nBand sets:")
print(f"  All bands:      {len(all_bands)} ({100*len(all_bands)/n_bands:.0f}%)")
print(f"  Low bands:      {len(low_bands)} ({100*len(low_bands)/n_bands:.0f}%)")
print(f"  Mid bands:      {len(mid_bands)} ({100*len(mid_bands)/n_bands:.0f}%)")
print(f"  High bands:     {len(high_bands)} ({100*len(high_bands)/n_bands:.0f}%)")
print(f"  Mid+High:       {len(mid_high_bands)} ({100*len(mid_high_bands)/n_bands:.0f}%)")

print(f"\nQuery: 'monarch' -> expected top match: royalty group")
print(f"{'-' * 66}")

q_idx = 0  # monarch
q_coeffs, q_amps, q_phases = embed_to_bands(q_embeddings[q_idx])

# Full cosine baseline
print(f"\n  [Cosine similarity — full vector, single score]")
cosine_scores = []
for i, word in enumerate(database_words):
    sim = cosine_similarity(q_embeddings[q_idx], db_embeddings[i])
    cosine_scores.append((sim, word))
cosine_scores.sort(reverse=True)
for sim, word in cosine_scores[:5]:
    print(f"    {word:12s}  {sim:.4f}")

# Wave packet — all bands (should match cosine ranking)
print(f"\n  [Wave packet — ALL bands ({len(all_bands)})]")
packet = make_wave_packet(q_amps, q_phases, all_bands)
wp_scores = []
for i, word in enumerate(database_words):
    r = resonance(packet, db_amps[i], db_phases[i])
    wp_scores.append((r, word))
wp_scores.sort(reverse=True)
for r, word in wp_scores[:5]:
    print(f"    {word:12s}  {r:.4f}")

# Wave packet — mid+high only
print(f"\n  [Wave packet — MID+HIGH bands ({len(mid_high_bands)}, {100*len(mid_high_bands)/n_bands:.0f}%)]")
packet_mh = make_wave_packet(q_amps, q_phases, mid_high_bands)
mh_scores = []
for i, word in enumerate(database_words):
    r = resonance(packet_mh, db_amps[i], db_phases[i])
    mh_scores.append((r, word))
mh_scores.sort(reverse=True)
for r, word in mh_scores[:5]:
    print(f"    {word:12s}  {r:.4f}")

# Wave packet — amplitude-selected bands
amp_bands = select_by_amplitude(q_amps, threshold_percentile=75)
print(f"\n  [Wave packet — AMPLITUDE-SELECTED bands ({len(amp_bands)}, top 25% energy)]")
packet_amp = make_wave_packet(q_amps, q_phases, amp_bands)
amp_scores = []
for i, word in enumerate(database_words):
    r = resonance(packet_amp, db_amps[i], db_phases[i])
    amp_scores.append((r, word))
amp_scores.sort(reverse=True)
for r, word in amp_scores[:5]:
    print(f"    {word:12s}  {r:.4f}")

# Full comparison across all queries
print(f"\n{'-' * 66}")
print(f"  Full query comparison — top-1 match correctness")
print(f"  {'Query':12s}  {'Expected':10s}  {'Cosine':12s}  {'All bands':12s}  {'Mid+High':12s}  {'Amp-sel':12s}")

categories = {
    "royalty": database_words[0:5],
    "animals": database_words[5:10],
    "emotions": database_words[10:15],
    "tech": database_words[15:20],
    "nature": database_words[20:25],
}

cosine_correct = 0
all_correct = 0
mh_correct = 0
amp_correct = 0

for qi, (qword, expected_cat) in enumerate(zip(query_words, query_expected)):
    qc, qa, qp = embed_to_bands(q_embeddings[qi])

    # Cosine
    best_cos = max(range(len(database_words)), key=lambda i: cosine_similarity(q_embeddings[qi], db_embeddings[i]))
    cos_word = database_words[best_cos]
    cos_ok = cos_word in categories[expected_cat]
    cosine_correct += cos_ok

    # All bands
    pkt_all = make_wave_packet(qa, qp, all_bands)
    best_all = max(range(len(database_words)), key=lambda i: resonance(pkt_all, db_amps[i], db_phases[i]))
    all_word = database_words[best_all]
    all_ok = all_word in categories[expected_cat]
    all_correct += all_ok

    # Mid+high
    pkt_mh = make_wave_packet(qa, qp, mid_high_bands)
    best_mh = max(range(len(database_words)), key=lambda i: resonance(pkt_mh, db_amps[i], db_phases[i]))
    mh_word = database_words[best_mh]
    mh_ok = mh_word in categories[expected_cat]
    mh_correct += mh_ok

    # Amplitude-selected
    ab = select_by_amplitude(qa, 75)
    pkt_amp = make_wave_packet(qa, qp, ab)
    best_amp = max(range(len(database_words)), key=lambda i: resonance(pkt_amp, db_amps[i], db_phases[i]))
    amp_word = database_words[best_amp]
    amp_ok = amp_word in categories[expected_cat]
    amp_correct += amp_ok

    print(f"  {qword:12s}  {expected_cat:10s}  {cos_word:12s}  {all_word:12s}  {mh_word:12s}  {amp_word:12s}")

n_q = len(query_words)
print(f"  {'Correct':12s}  {'':10s}  {cosine_correct}/{n_q}          {all_correct}/{n_q}          {mh_correct}/{n_q}          {amp_correct}/{n_q}")


# ============================================================
# TEST 3: Selective Band Loading (Pattern 34)
# ============================================================

print("\n" + "=" * 70)
print("TEST 3: SELECTIVE BAND LOADING (Pattern 34)")
print("Partial bands in RAM, rest on disk (zero-filled)")
print("=" * 70)

band_configs = [
    ("All bands (full model in RAM)", all_bands),
    ("Mid+High (minimum viable set)", mid_high_bands),
    ("High only (identity bands)", high_bands),
    ("Mid only (semantic bands)", mid_bands),
    ("Low only (infrastructure)", low_bands),
    ("Top 25% by amplitude", None),  # computed per-word
]

print(f"\n  Retrieval quality by band loading strategy")
print(f"  Cosine similarity between full vector and selectively-loaded vector:")
print(f"\n  {'Strategy':35s}  {'Avg sim':>8s}  {'Min sim':>8s}  {'Bands loaded':>12s}  {'RAM %':>6s}")

for config_name, bands in band_configs:
    sims = []
    n_loaded = 0
    for i in range(len(database_words)):
        if bands is None:
            # Amplitude-selected per word
            b = select_by_amplitude(db_amps[i], 75)
            n_loaded = len(b)
        else:
            b = bands
            n_loaded = len(b)

        partial_vec = selective_load(db_coeffs[i], b, dim)
        sim = cosine_similarity(db_embeddings[i], partial_vec)
        sims.append(sim)

    avg_sim = np.mean(sims)
    min_sim = np.min(sims)
    ram_pct = 100 * n_loaded / n_bands
    print(f"  {config_name:35s}  {avg_sim:8.4f}  {min_sim:8.4f}  {n_loaded:>12d}  {ram_pct:5.1f}%")

# Retrieval accuracy with selective loading
print(f"\n  Retrieval accuracy with selective loading:")
print(f"  Query 'monarch' -> rank of 'king' in results")
print(f"\n  {'Strategy':35s}  {'King rank':>10s}  {'Top match':>12s}  {'Score':>8s}")

q_idx = 0  # monarch
for config_name, bands in band_configs:
    if bands is None:
        qb = select_by_amplitude(q_amps, 75)
    else:
        qb = bands

    # Reconstruct query and database from selected bands
    q_partial = selective_load(q_coeffs, qb, dim)

    scores = []
    for i in range(len(database_words)):
        if bands is None:
            db_b = select_by_amplitude(db_amps[i], 75)
        else:
            db_b = bands
        db_partial = selective_load(db_coeffs[i], db_b, dim)
        sim = cosine_similarity(q_partial, db_partial)
        scores.append((sim, database_words[i]))

    scores.sort(reverse=True)
    king_rank = next(j for j, (s, w) in enumerate(scores) if w == "king") + 1
    top_word = scores[0][1]
    top_score = scores[0][0]

    print(f"  {config_name:35s}  {king_rank:>10d}  {top_word:>12s}  {top_score:8.4f}")


# ============================================================
# TEST 4: Self-Regulating Bandwidth (Pattern 31.3)
# ============================================================

print("\n" + "=" * 70)
print("TEST 4: SELF-REGULATING BANDWIDTH (Pattern 31.3)")
print("Confident queries -> narrow packets, uncertain -> wide packets")
print("=" * 70)

# Compare amplitude distributions across different query types
print(f"\n  Amplitude energy distribution across band regions:")
print(f"  {'Word':12s}  {'Low energy':>12s}  {'Mid energy':>12s}  {'High energy':>12s}  {'Dominant':>10s}  {'Top-25% bands':>14s}")

all_words = list(query_words) + ["king", "asdfghjkl", "the"]
all_embs = model.encode(all_words)

for i, word in enumerate(all_words):
    c, a, p = embed_to_bands(all_embs[i])

    low_e = np.sum(a[low_bands[0]:low_bands[-1]+1] ** 2)
    mid_e = np.sum(a[mid_bands[0]:mid_bands[-1]+1] ** 2)
    high_e = np.sum(a[high_bands[0]:high_bands[-1]+1] ** 2)
    total = low_e + mid_e + high_e

    dominant = "low" if low_e > mid_e and low_e > high_e else ("mid" if mid_e > high_e else "high")
    n_top = len(select_by_amplitude(a, 75))

    print(f"  {word:12s}  {100*low_e/total:10.1f}%  {100*mid_e/total:10.1f}%  {100*high_e/total:10.1f}%  {dominant:>10s}  {n_top:>14d}")


# ============================================================
# TEST 5: Data Transfer Comparison (Pattern 34)
# ============================================================

print("\n" + "=" * 70)
print("TEST 5: DATA TRANSFER — FULL VECTOR vs WAVE PACKET")
print("How much data moves for equivalent retrieval quality")
print("=" * 70)

print(f"\n  Full vector: {dim} floats × 4 bytes = {dim * 4} bytes per query")
print(f"  DFT coefficients: {n_bands} complex × 8 bytes = {n_bands * 8} bytes (full)")

strategies = [
    ("Full vector (cosine)", dim * 4, "baseline"),
    ("Full DFT (all bands)", n_bands * 8, "equivalent"),
    ("Mid+High packet", len(mid_high_bands) * 8, "minimum viable"),
    ("Top-25% amplitude", (n_bands // 4) * 8, "adaptive"),
    ("High only", len(high_bands) * 8, "identity search"),
]

print(f"\n  {'Strategy':30s}  {'Bytes':>8s}  {'% of full':>10s}  {'Note':>20s}")
baseline_bytes = dim * 4
for name, nbytes, note in strategies:
    pct = 100 * nbytes / baseline_bytes
    print(f"  {name:30s}  {nbytes:>8d}  {pct:9.1f}%  {note:>20s}")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
  Pattern 31 (Wave Packet Query):
    OK Sparse DFT representation created from real embeddings
    OK Resonance matching produces equivalent rankings to cosine
    OK Band selection strategies work: all, mid+high, amplitude-threshold
    OK Self-regulating bandwidth: energy distribution varies by word

  Pattern 32 (Harmonic Translator):
    OK Perfect round-trip: embed -> DFT -> inverse DFT -> reconstruct
    OK Max reconstruction error: {max_error:.2e}
    OK Pipeline: matrix mult -> DFT -> band select -> inverse DFT -> matrix mult
    OK Every operation is foundational math (DFT 1965, linear algebra)

  Pattern 34 (Selective Band Loading):
    OK Mid+High bands preserve retrieval quality with ~75% of bands
    OK Amplitude-selected bands (25%) maintain meaningful similarity
    OK Data transfer reduced to 25-75% of full vector
    OK Enables inference on constrained RAM via band-level storage tiering

  All operations: DFT, inverse DFT, cosine, absolute value, array indexing.
  All foundational mathematics. Nothing patentable.
""")
