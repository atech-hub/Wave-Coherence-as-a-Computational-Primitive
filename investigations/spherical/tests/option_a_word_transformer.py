"""
Option A: Word-Level Shakespeare Transformer — Triple-Channel Evaluation

Three training variants:
  1. Frozen: Harmonic phase fixed, magnitude fixed at scale (control)
  2. Magnitude: Harmonic phase frozen, per-token-per-band magnitude trainable (hypothesis)
  3. Baseline: Fully trainable nn.Embedding (ceiling)

Post-training analysis (before any coherence tests):
  1. Does magnitude vary? (global CV)
  2. Does magnitude cluster by meaning? (within-family vs cross-family CV)
  3. Does well membership align with semantics? (Legendre zeros at l=8)
  4. Does within-well magnitude distance correlate with semantic distance?

Then triple-channel evaluation:
  Channel 1: Circle coherence cos(n * delta_phi)
  Channel 2: Well membership agreement (same/different well)
  Channel 3: Within-well magnitude distance

Usage:
    python experiments/option_a_word_transformer.py
"""

import math
import os
import re
import time
import urllib.request
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Configuration
# =============================================================================

N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
N_BANDS = N_EMBD // 2  # 64
BLOCK_SIZE = 128        # word context (shorter than char-level 256)
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_ITERS = 3000
EVAL_INTERVAL = 300
EVAL_ITERS = 50
MIN_WORD_FREQ = 3       # minimum occurrences to be in vocabulary

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Progressive curriculum (3000 steps)
PROG_STAGES = [
    (0,    1000, 8),     # Stage 1: bands 1-8
    (1000, 2000, 24),    # Stage 2: bands 1-24
    (2000, 3000, 64),    # Stage 3: all 64 bands
]

# Semantic families for Shakespeare
SEMANTIC_FAMILIES = {
    "royalty": ["king", "queen", "prince", "duke", "lord", "lady",
                "crown", "throne", "majesty", "royal"],
    "nature": ["sun", "moon", "star", "earth", "sea", "wind",
               "storm", "fire", "water", "night"],
    "emotion": ["love", "hate", "fear", "joy", "grief", "rage",
                "hope", "shame", "pride", "anger"],
    "body": ["heart", "blood", "hand", "eye", "tongue", "face",
             "soul", "bone", "head", "arm"],
    "function": ["the", "and", "but", "of", "to", "in",
                 "is", "that", "for", "with"],
    "speech": ["say", "speak", "tell", "hear", "call", "cry",
               "pray", "swear", "bid", "answer"],
}


# =============================================================================
# Data — Word-Level Tokenizer
# =============================================================================

def download_shakespeare():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    filepath = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(filepath):
        # Check known locations in the repo
        repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "..")
        for alt_path in ["python/data/shakespeare.txt",
                         "rust-transformer/data/shakespeare.txt",
                         "experiments/data/shakespeare.txt"]:
            alt = os.path.join(repo_root, alt_path)
            if os.path.exists(alt):
                filepath = alt
                break
        else:
            os.makedirs(data_dir, exist_ok=True)
            print("  Downloading Shakespeare...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, filepath)
    with open(filepath, "r") as f:
        return f.read()


def tokenize_text(text):
    """Split text into words, lowercase, keep common punctuation as tokens."""
    # Split on whitespace, then separate punctuation
    raw_tokens = text.lower().split()
    tokens = []
    for tok in raw_tokens:
        # Separate leading/trailing punctuation
        m = re.match(r"^([^a-z]*)([a-z]+(?:'[a-z]+)?)([^a-z]*)$", tok)
        if m:
            if m.group(1):
                tokens.append(m.group(1))
            tokens.append(m.group(2))
            if m.group(3):
                tokens.append(m.group(3))
        else:
            tokens.append(tok)
    return tokens


class WordDataset:
    def __init__(self, text, min_freq=MIN_WORD_FREQ):
        tokens = tokenize_text(text)
        counts = Counter(tokens)

        # Build vocabulary: words with freq >= min_freq
        vocab_words = sorted([w for w, c in counts.items() if c >= min_freq])
        self.unk_token = "<unk>"
        self.vocab = [self.unk_token] + vocab_words
        self.vocab_size = len(self.vocab)
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}

        # Store word frequencies (for frequency confound analysis)
        self.word_freq = {self.stoi[w]: counts[w] for w in vocab_words if w in self.stoi}

        # Encode
        unk_id = self.stoi[self.unk_token]
        data = [self.stoi.get(t, unk_id) for t in tokens]
        n_unk = sum(1 for d in data if d == unk_id)

        print(f"  Tokenizer: {len(tokens)} tokens, {self.vocab_size} vocab "
              f"(min_freq={min_freq}), {n_unk} UNK ({100*n_unk/len(tokens):.1f}%)")

        n = int(0.9 * len(data))
        self.train_data = torch.tensor(data[:n], dtype=torch.long)
        self.val_data = torch.tensor(data[n:], dtype=torch.long)

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
        return x.to(DEVICE), y.to(DEVICE)

    def get_family_ids(self):
        """Return dict of family_name -> list of token IDs present in vocab."""
        families = {}
        for name, words in SEMANTIC_FAMILIES.items():
            ids = [self.stoi[w] for w in words if w in self.stoi]
            if len(ids) >= 3:
                families[name] = ids
        return families


# =============================================================================
# Harmonic Embeddings
# =============================================================================

def build_harmonic_table(vocab_size, n_embd):
    """Phase-only harmonic table (unit magnitude per band)."""
    n_harmonics = n_embd // 2
    scale = 1.0 / math.sqrt(n_harmonics)
    table = torch.zeros(vocab_size, n_embd)
    for c in range(vocab_size):
        theta = c * 2.0 * math.pi / vocab_size
        for h in range(n_harmonics):
            n = h + 1
            phase = n * theta
            table[c, h * 2] = math.cos(phase) * scale
            table[c, h * 2 + 1] = math.sin(phase) * scale
    return table


def build_positional_table(max_len, n_embd):
    n_harmonics = n_embd // 2
    scale = 1.0 / math.sqrt(n_harmonics)
    table = torch.zeros(max_len, n_embd)
    for pos in range(max_len):
        for h in range(n_harmonics):
            freq = 1.0 / (10000.0 ** (2.0 * h / n_embd))
            phase = pos * freq
            table[pos, h * 2] = math.cos(phase) * scale
            table[pos, h * 2 + 1] = math.sin(phase) * scale
    return table


class MagnitudeEmbedding(nn.Module):
    """Frozen harmonic phase + trainable per-token-per-band magnitude.

    Phase direction is fixed (cos/sin from harmonic table).
    Magnitude (scalar per band per token) is trainable, init at 1.0.
    Forward: emb[w, 2k] = mag[w, k] * phase[w, 2k]
             emb[w, 2k+1] = mag[w, k] * phase[w, 2k+1]
    """

    def __init__(self, vocab_size, n_embd):
        super().__init__()
        n_bands = n_embd // 2
        # Frozen phase (unit-magnitude harmonic directions)
        self.register_buffer("phase_table",
                             build_harmonic_table(vocab_size, n_embd))
        # Trainable magnitude: one scalar per token per band, init=1.0
        self.mag = nn.Parameter(torch.ones(vocab_size, n_bands))
        self.n_bands = n_bands

    def forward(self, idx):
        phase = self.phase_table[idx]              # (B, T, n_embd)
        mag = self.mag[idx]                        # (B, T, n_bands)
        # Expand mag to match cos/sin pairs
        mag_expanded = mag.unsqueeze(-1).expand(-1, -1, -1, 2)
        mag_flat = mag_expanded.reshape(mag.shape[0], mag.shape[1], -1)
        return phase * mag_flat


class PhaseOnlyEmbedding(nn.Module):
    """Trainable phase angles + frozen uniform magnitude.

    Each token has one trainable phase angle per harmonic band.
    Magnitude is fixed at 1/sqrt(n_bands) for all tokens.
    This lets the optimizer organise tokens on the circle (semantic phase)
    without any magnitude freedom.

    Comparison:
      phase_only (this) vs baseline: does magnitude ADD semantic value?
      phase_only (this) vs magnitude: phase leads, magnitude follows?
    """

    def __init__(self, vocab_size, n_embd):
        super().__init__()
        n_bands = n_embd // 2
        self.n_bands = n_bands
        self.scale = 1.0 / math.sqrt(n_bands)
        # Initialize phase angles from harmonic grid
        phases = torch.zeros(vocab_size, n_bands)
        for w in range(vocab_size):
            theta = w * 2.0 * math.pi / vocab_size
            for k in range(n_bands):
                phases[w, k] = (k + 1) * theta
        self.phase = nn.Parameter(phases)  # trainable

    def forward(self, idx):
        ph = self.phase[idx]                       # (B, T, n_bands)
        cos_part = torch.cos(ph) * self.scale      # (B, T, n_bands)
        sin_part = torch.sin(ph) * self.scale      # (B, T, n_bands)
        # Interleave cos/sin pairs: [cos0, sin0, cos1, sin1, ...]
        return torch.stack([cos_part, sin_part], dim=-1).reshape(
            *ph.shape[:-1], -1)


# =============================================================================
# Model
# =============================================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.n_head = N_HEAD
        self.n_embd = N_EMBD
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
            .view(1, 1, BLOCK_SIZE, BLOCK_SIZE),
        )

    def forward(self, x):
        B, T, C = x.size()
        head_dim = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.ffn = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class WordGPT(nn.Module):
    def __init__(self, vocab_size, mode="frozen"):
        super().__init__()
        self.mode = mode
        self.vocab_size = vocab_size

        # Embedding layer depends on mode
        if mode == "frozen":
            self.register_buffer("wte", build_harmonic_table(vocab_size, N_EMBD))
            self.embed_fn = lambda idx: F.embedding(idx, self.wte)
        elif mode == "magnitude":
            self.mag_embed = MagnitudeEmbedding(vocab_size, N_EMBD)
            self.embed_fn = self.mag_embed.forward
        elif mode == "phase_only":
            self.phase_embed = PhaseOnlyEmbedding(vocab_size, N_EMBD)
            self.embed_fn = self.phase_embed.forward
        elif mode == "baseline":
            self.wte = nn.Embedding(vocab_size, N_EMBD)
            self.embed_fn = self.wte.forward
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.register_buffer("wpe", build_positional_table(BLOCK_SIZE, N_EMBD))
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)

        # Band mask for progressive curriculum
        self.register_buffer("band_mask", torch.ones(N_EMBD))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  {mode} model: {n_params:,} trainable parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_active_bands(self, n_active):
        mask = torch.zeros(N_EMBD, device=self.band_mask.device)
        for h in range(min(n_active, N_BANDS)):
            mask[h * 2] = 1.0
            mask[h * 2 + 1] = 1.0
        self.band_mask = mask

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.embed_fn(idx)
        pos_emb = self.wpe[:T]
        x = (tok_emb + pos_emb) * self.band_mask

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_embeddings(self):
        """Extract the final embedding table as (vocab_size, n_embd) tensor."""
        if self.mode == "frozen":
            return self.wte.clone()
        elif self.mode == "magnitude":
            with torch.no_grad():
                idx = torch.arange(self.vocab_size, device=self.mag_embed.phase_table.device)
                phase = self.mag_embed.phase_table[idx]
                mag = self.mag_embed.mag[idx]
                mag_expanded = mag.unsqueeze(-1).expand(-1, -1, 2)
                mag_flat = mag_expanded.reshape(self.vocab_size, -1)
                return (phase * mag_flat).clone()
        elif self.mode == "phase_only":
            with torch.no_grad():
                idx = torch.arange(self.vocab_size,
                                   device=self.phase_embed.phase.device)
                return self.phase_embed(idx.unsqueeze(0)).squeeze(0).clone()
        elif self.mode == "baseline":
            return self.wte.weight.detach().clone()


# =============================================================================
# Training
# =============================================================================

@torch.no_grad()
def estimate_loss(model, dataset):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = dataset.get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_stage(iter_num):
    for start, end, bands in PROG_STAGES:
        if start <= iter_num < end:
            return bands
    return 64


def train_variant(mode, dataset):
    print(f"\n{'=' * 70}")
    print(f"  Training: {mode}")
    for start, end, bands in PROG_STAGES:
        print(f"    Steps {start}-{end}: bands 1-{bands}")
    print(f"{'=' * 70}")

    model = WordGPT(dataset.vocab_size, mode=mode).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    history = []
    start_time = time.time()
    current_bands = 64

    for iter_num in range(MAX_ITERS):
        stage_bands = get_stage(iter_num)
        if stage_bands != current_bands:
            model.set_active_bands(stage_bands)
            current_bands = stage_bands
            print(f"  >>> Stage change at step {iter_num}: bands 1-{stage_bands}")

        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            saved_mask = model.band_mask.clone()
            model.set_active_bands(64)
            losses = estimate_loss(model, dataset)
            if current_bands < 64:
                model.band_mask = saved_mask

            elapsed = time.time() - start_time
            print(f"  step {iter_num:>5} | train {losses['train']:.4f}"
                  f" | val {losses['val']:.4f} | {elapsed:.1f}s"
                  f" [bands 1-{current_bands}]")
            history.append((iter_num, losses["train"], losses["val"]))

            if math.isnan(losses["train"]):
                print("  !!! NaN detected -- stopping.")
                break

        x, y = dataset.get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    total = time.time() - start_time
    final_val = history[-1][2] if history else float("nan")
    print(f"  Training complete in {total:.1f}s, final val loss: {final_val:.4f}")

    return model, history


# =============================================================================
# Post-Training Analysis
# =============================================================================

def extract_magnitudes(embeddings):
    """Extract per-band magnitude and phase from embedding table.
    Returns magnitudes (vocab, n_bands) and phases (vocab, n_bands)."""
    vocab_size, n_embd = embeddings.shape
    n_bands = n_embd // 2
    cos_vals = embeddings[:, 0::2]  # (vocab, n_bands)
    sin_vals = embeddings[:, 1::2]  # (vocab, n_bands)
    magnitudes = torch.sqrt(cos_vals ** 2 + sin_vals ** 2)
    phases = torch.atan2(sin_vals, cos_vals)
    return magnitudes, phases


def legendre(l, x):
    """Legendre polynomial P_l(x) via recurrence."""
    if l == 0:
        return torch.ones_like(x)
    if l == 1:
        return x.clone()
    p_prev = torch.ones_like(x)
    p_curr = x.clone()
    for k in range(1, l):
        p_next = ((2 * k + 1) * x * p_curr - k * p_prev) / (k + 1)
        p_prev = p_curr
        p_curr = p_next
    return p_curr


def find_legendre_zeros(l, n_scan=10000):
    """Find zeros of P_l(x) in (-1, 1), return as theta = arccos(x) sorted ascending."""
    x_vals = torch.linspace(-0.9999, 0.9999, n_scan)
    p_vals = legendre(l, x_vals)
    zeros = []
    for i in range(1, n_scan):
        if p_vals[i - 1] * p_vals[i] < 0:
            # Bisect
            lo, hi = x_vals[i - 1].item(), x_vals[i].item()
            for _ in range(60):
                mid = (lo + hi) / 2
                pm = legendre(l, torch.tensor([mid]))[0].item()
                p_lo = legendre(l, torch.tensor([lo]))[0].item()
                if p_lo * pm < 0:
                    hi = mid
                else:
                    lo = mid
            zeros.append(math.acos((lo + hi) / 2))
    return sorted(zeros)


def assign_wells(elevations, boundaries):
    """Assign tokens to wells based on elevation and boundary theta values."""
    wells = torch.zeros(len(elevations), dtype=torch.long)
    for i, theta in enumerate(elevations):
        w = 0
        for b in boundaries:
            if theta >= b:
                w += 1
            else:
                break
        wells[i] = w
    return wells


def analyze_magnitudes(model, dataset):
    """Full post-training magnitude analysis."""
    print(f"\n{'=' * 70}")
    print(f"  POST-TRAINING ANALYSIS: {model.mode}")
    print(f"{'=' * 70}")

    embeddings = model.get_embeddings().cpu()
    magnitudes, phases = extract_magnitudes(embeddings)
    vocab_size = magnitudes.shape[0]
    n_bands = magnitudes.shape[1]

    # ---- Measurement 1: Does magnitude vary? ----
    print("\n  --- Measurement 1: Does magnitude vary? ---")
    mean_mags = magnitudes.mean(dim=1)  # per-token mean across bands
    global_mean = mean_mags.mean().item()
    global_std = mean_mags.std().item()
    global_cv = (global_std / global_mean * 100) if global_mean > 1e-10 else 0.0

    band_cvs = []
    for k in range(n_bands):
        bm = magnitudes[:, k]
        bc = (bm.std().item() / bm.mean().item() * 100) if bm.mean().item() > 1e-10 else 0.0
        band_cvs.append(bc)

    print(f"    Mean magnitude per token: mean={global_mean:.6f}, std={global_std:.6f}")
    print(f"    Global CV (mean-across-bands): {global_cv:.1f}%")
    print(f"    Per-band CV: avg={sum(band_cvs)/len(band_cvs):.1f}%, "
          f"min={min(band_cvs):.1f}%, max={max(band_cvs):.1f}%")

    if model.mode == "magnitude":
        raw_mag = model.mag_embed.mag.detach().cpu()
        rm_mean = raw_mag.mean().item()
        rm_std = raw_mag.std().item()
        rm_cv = (rm_std / rm_mean * 100) if rm_mean > 1e-10 else 0.0
        print(f"    Raw trainable magnitude: mean={rm_mean:.4f}, std={rm_std:.4f}, CV={rm_cv:.1f}%")

    mag_varies = global_cv >= 0.5

    if not mag_varies:
        print(f"    VERDICT: Magnitude is uniform ({global_cv:.1f}% CV).")

    # ---- Phase displacement (for phase_only mode) ----
    if model.mode == "phase_only":
        # Measure how much the optimizer moved phases from the harmonic grid
        grid_phases = torch.zeros(vocab_size, n_bands)
        for w in range(vocab_size):
            theta = w * 2.0 * math.pi / vocab_size
            for k in range(n_bands):
                grid_phases[w, k] = (k + 1) * theta
        # Phase displacement per token (circular distance)
        phase_diff = phases - grid_phases
        # Wrap to [-pi, pi]
        phase_diff = (phase_diff + math.pi) % (2 * math.pi) - math.pi
        mean_disp = phase_diff.abs().mean().item()
        max_disp = phase_diff.abs().max().item()
        disp_degrees = math.degrees(mean_disp)
        print(f"\n    Phase displacement from harmonic grid:")
        print(f"      Mean: {mean_disp:.4f} rad ({disp_degrees:.1f} deg)")
        print(f"      Max:  {max_disp:.4f} rad ({math.degrees(max_disp):.1f} deg)")
        if mean_disp > 0.01:
            print(f"    Optimizer actively reorganised phases.")
        else:
            print(f"    Phases barely moved from harmonic grid.")

    if not mag_varies and model.mode not in ("phase_only", "baseline"):
        # Skip detailed magnitude analysis for frozen mode
        return {
            "embeddings": embeddings, "magnitudes": magnitudes,
            "phases": phases, "mean_mags": mean_mags, "wells": None,
            "n_wells": 0, "global_cv": global_cv, "mag_varies": False,
        }

    families = dataset.get_family_ids()

    # ---- Measurements 2-4: magnitude clustering (skip if uniform) ----
    if mag_varies:
        print("\n  --- Measurement 2: Does magnitude cluster by meaning? ---")
        if not families:
            print("    No semantic families found in vocabulary.")
        else:
            print(f"    {len(families)} families with tokens in vocab:")
            within_cvs = []
            for fname, fids in families.items():
                fmags = mean_mags[fids]
                fm = fmags.mean().item()
                fs = fmags.std().item()
                fcv = (fs / fm * 100) if fm > 1e-10 else 0.0
                within_cvs.append(fcv)
                words = [dataset.itos[i] for i in fids]
                print(f"      {fname:>10} ({len(fids)} words): "
                      f"mean={fm:.6f} std={fs:.6f} CV={fcv:.1f}%  [{', '.join(words[:6])}]")

            avg_within_cv = sum(within_cvs) / len(within_cvs) if within_cvs else 0
            print(f"\n    Within-family CV avg: {avg_within_cv:.1f}%")
            print(f"    Cross-family (global) CV:  {global_cv:.1f}%")

            if avg_within_cv < global_cv * 0.5:
                print(f"    SIGNAL: Within-family CV is {global_cv/avg_within_cv:.1f}x tighter than global.")
                print(f"    Magnitude carries semantic information.")
            elif avg_within_cv < global_cv * 0.8:
                print(f"    WEAK SIGNAL: Within-family CV is somewhat tighter ({avg_within_cv:.1f}% vs {global_cv:.1f}%).")
            else:
                print(f"    NO SIGNAL: Within-family CV ({avg_within_cv:.1f}%) ~ global ({global_cv:.1f}%).")
                print(f"    Magnitude does not cluster by semantic family.")

        print("\n  --- Measurement 3: Well membership alignment (l=8) ---")
        l_val = 8
        boundaries = find_legendre_zeros(l_val)
        n_wells = len(boundaries) + 1

        mm_min = mean_mags.min().item()
        mm_max = mean_mags.max().item()
        mm_range = mm_max - mm_min
        if mm_range > 1e-12:
            elevations = [math.pi * (m.item() - mm_min) / mm_range for m in mean_mags]
        else:
            elevations = [math.pi / 2] * vocab_size

        wells = assign_wells(elevations, boundaries)
        well_sizes = [(wells == w).sum().item() for w in range(n_wells)]
        print(f"    {n_wells} wells, sizes: {well_sizes}")

        if families:
            print(f"\n    Family well membership:")
            import random
            random.seed(42)
            n_random_trials = 1000

            for fname, fids in families.items():
                family_wells = wells[fids].tolist()
                well_counts = Counter(family_wells)
                most_common_well, most_common_count = well_counts.most_common(1)[0]
                concentration = most_common_count / len(fids)

                random_concentrations = []
                for _ in range(n_random_trials):
                    rand_ids = random.sample(range(vocab_size), len(fids))
                    rand_wells = wells[rand_ids].tolist()
                    rand_counts = Counter(rand_wells)
                    rand_best = rand_counts.most_common(1)[0][1]
                    random_concentrations.append(rand_best / len(fids))
                rand_mean = sum(random_concentrations) / len(random_concentrations)
                p_value = sum(1 for rc in random_concentrations if rc >= concentration) / n_random_trials

                sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
                print(f"      {fname:>10}: {most_common_count}/{len(fids)} in W{most_common_well} "
                      f"({concentration:.0%}), random={rand_mean:.0%}, p={p_value:.3f} {sig}")

        print("\n  --- Measurement 4: Within-well magnitude distance ---")
        if families:
            print(f"    For same-well pairs within each family:")
            for fname, fids in families.items():
                family_wells_list = wells[fids].tolist()
                family_mags = mean_mags[fids]
                same_well_dists = []
                diff_well_dists = []
                for i in range(len(fids)):
                    for j in range(i + 1, len(fids)):
                        d = abs(family_mags[i].item() - family_mags[j].item())
                        if family_wells_list[i] == family_wells_list[j]:
                            same_well_dists.append(d)
                        else:
                            diff_well_dists.append(d)
                sw_mean = sum(same_well_dists) / len(same_well_dists) if same_well_dists else 0
                dw_mean = sum(diff_well_dists) / len(diff_well_dists) if diff_well_dists else 0
                ratio = (dw_mean / sw_mean) if sw_mean > 1e-15 else 0
                print(f"      {fname:>10}: same-well={sw_mean:.6f} ({len(same_well_dists)} pairs), "
                      f"diff-well={dw_mean:.6f} ({len(diff_well_dists)} pairs), ratio={ratio:.1f}x")
    else:
        wells = None
        n_wells = 0

    # ---- Measurement 5: Phase-based semantic clustering ----
    # (Applies to phase_only and baseline -- any mode where phases may differ)
    if model.mode in ("phase_only", "baseline") and families:
        print(f"\n  --- Measurement 5: Phase-based semantic clustering ---")
        # For each family, compute mean pairwise phase distance (band 0)
        # Within-family vs cross-family
        family_ids_all = set()
        for fids in families.values():
            family_ids_all.update(fids)
        other_ids = [i for i in range(vocab_size) if i not in family_ids_all and i != 0]
        import random
        random.seed(99)
        cross_sample = random.sample(other_ids, min(100, len(other_ids)))

        print(f"    Phase coherence cos(1 * delta_phi) on band 0:")
        print(f"    {'Family':>10}  {'Within':>10}  {'Cross':>10}  {'Ratio':>8}")
        for fname, fids in families.items():
            within_cohs = []
            for i in range(len(fids)):
                for j in range(i + 1, len(fids)):
                    dphi = phases[fids[i], 0].item() - phases[fids[j], 0].item()
                    within_cohs.append(math.cos(dphi))
            cross_cohs = []
            for fi in fids:
                for oi in cross_sample:
                    dphi = phases[fi, 0].item() - phases[oi, 0].item()
                    cross_cohs.append(math.cos(dphi))
            w_mean = sum(within_cohs) / len(within_cohs) if within_cohs else 0
            c_mean = sum(cross_cohs) / len(cross_cohs) if cross_cohs else 0
            ratio = abs(w_mean / c_mean) if abs(c_mean) > 1e-10 else 0
            print(f"    {fname:>10}  {w_mean:>10.6f}  {c_mean:>10.6f}  {ratio:>8.2f}")

        # Multi-band phase distance (average across all bands)
        print(f"\n    Multi-band phase distance (avg over {n_bands} bands):")
        print(f"    {'Family':>10}  {'Within':>10}  {'Cross':>10}  {'Ratio':>8}")
        for fname, fids in families.items():
            within_dists = []
            for i in range(len(fids)):
                for j in range(i + 1, len(fids)):
                    band_dists = []
                    for k in range(n_bands):
                        dphi = phases[fids[i], k].item() - phases[fids[j], k].item()
                        dphi = (dphi + math.pi) % (2 * math.pi) - math.pi
                        band_dists.append(abs(dphi))
                    within_dists.append(sum(band_dists) / len(band_dists))
            cross_dists = []
            for fi in fids:
                for oi in cross_sample[:30]:  # smaller sample for speed
                    band_dists = []
                    for k in range(n_bands):
                        dphi = phases[fi, k].item() - phases[oi, k].item()
                        dphi = (dphi + math.pi) % (2 * math.pi) - math.pi
                        band_dists.append(abs(dphi))
                    cross_dists.append(sum(band_dists) / len(band_dists))
            w_mean = sum(within_dists) / len(within_dists) if within_dists else 0
            c_mean = sum(cross_dists) / len(cross_dists) if cross_dists else 0
            ratio = c_mean / w_mean if w_mean > 1e-10 else 0
            print(f"    {fname:>10}  {w_mean:>10.4f}  {c_mean:>10.4f}  {ratio:>8.2f}x")

    # ---- Measurement 6: Frequency confound check ----
    # Do high-frequency CONTENT words cluster in the same well as function words?
    # If yes: magnitude clustering = frequency effect (gradient pressure)
    # If no: magnitude clustering = linguistic role (grammatical class)
    if model.mode == "baseline" and wells is not None and hasattr(dataset, 'word_freq'):
        print(f"\n  --- Measurement 6: Frequency confound resolution ---")

        # Find the function-word well (the one 9/10 landed in)
        func_ids = families.get("function", [])
        if func_ids:
            func_wells = wells[func_ids].tolist()
            func_well = Counter(func_wells).most_common(1)[0][0]

            # Get all word frequencies, sort descending
            freq_list = [(wid, freq) for wid, freq in dataset.word_freq.items()
                         if wid != 0]  # skip <unk>
            freq_list.sort(key=lambda x: x[1], reverse=True)

            # Function word IDs (to exclude from content words)
            func_set = set(func_ids)

            # Top-20 most frequent CONTENT words (not in function family)
            top_content = [(wid, f) for wid, f in freq_list
                           if wid not in func_set][:20]

            # Also show function words for comparison
            func_freq = [(wid, dataset.word_freq.get(wid, 0)) for wid in func_ids]
            func_freq.sort(key=lambda x: x[1], reverse=True)

            print(f"    Function-word well: W{func_well}")
            print(f"\n    Function words (reference):")
            for wid, freq in func_freq:
                w = wells[wid].item()
                match = "<--" if w == func_well else ""
                print(f"      {dataset.itos[wid]:>12} freq={freq:>5}  W{w} {match}")

            print(f"\n    Top-20 frequent CONTENT words:")
            n_in_func_well = 0
            for wid, freq in top_content:
                w = wells[wid].item()
                match = "<--" if w == func_well else ""
                if w == func_well:
                    n_in_func_well += 1
                print(f"      {dataset.itos[wid]:>12} freq={freq:>5}  W{w} {match}")

            frac = n_in_func_well / len(top_content) if top_content else 0
            # Expected fraction (well size / vocab)
            well_size = (wells == func_well).sum().item()
            expected = well_size / vocab_size

            print(f"\n    Content words in function-word well: "
                  f"{n_in_func_well}/{len(top_content)} ({frac:.0%})")
            print(f"    Expected by chance: {expected:.0%}")

            if frac > expected * 1.5:
                print(f"    FREQUENCY EFFECT: High-freq content words also cluster "
                      f"in W{func_well}.")
                print(f"    Magnitude clustering reflects frequency, not grammatical role.")
            elif frac < expected * 0.7:
                print(f"    LINGUISTIC ROLE: High-freq content words AVOID W{func_well}.")
                print(f"    Function words cluster by role, not by frequency.")
            else:
                print(f"    INCONCLUSIVE: Content words land in W{func_well} "
                      f"at ~chance rate ({frac:.0%} vs {expected:.0%}).")

    # ---- Return data for triple-channel evaluation ----
    return {
        "embeddings": embeddings,
        "magnitudes": magnitudes,
        "phases": phases,
        "mean_mags": mean_mags,
        "wells": wells,
        "n_wells": n_wells,
        "global_cv": global_cv,
        "mag_varies": mag_varies,
    }


# =============================================================================
# Triple-Channel Evaluation
# =============================================================================

def triple_channel_eval(analysis_data, dataset):
    """Evaluate three independent channels on semantic families."""
    if analysis_data is None:
        print("\n  Skipping triple-channel eval (no magnitude variation).")
        return

    print(f"\n{'=' * 70}")
    print(f"  TRIPLE-CHANNEL EVALUATION")
    print(f"{'=' * 70}")

    families = dataset.get_family_ids()
    if not families:
        print("  No semantic families in vocab. Cannot evaluate.")
        return

    phases = analysis_data["phases"]
    mean_mags = analysis_data["mean_mags"]
    wells = analysis_data["wells"]
    vocab_size = phases.shape[0]

    # Build "all other" token set for contrast
    family_ids_all = set()
    for fids in families.values():
        family_ids_all.update(fids)
    other_ids = [i for i in range(vocab_size) if i not in family_ids_all and i != 0]

    print("\n  Channel 1: Circle coherence cos(n * delta_phi)")
    print(f"  {'Family':>10}  {'Within':>10}  {'Cross':>10}  {'Ratio':>8}")
    print(f"  {'':>10}  {'(mean)':>10}  {'(mean)':>10}  {'':>8}")

    for fname, fids in families.items():
        # Within-family circle coherence (n=1, band 0 phase)
        within_cohs = []
        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                dphi = phases[fids[i], 0].item() - phases[fids[j], 0].item()
                within_cohs.append(math.cos(dphi))

        # Cross-family: family vs random others
        import random
        random.seed(123)
        cross_sample = random.sample(other_ids, min(50, len(other_ids)))
        cross_cohs = []
        for fi in fids:
            for oi in cross_sample:
                dphi = phases[fi, 0].item() - phases[oi, 0].item()
                cross_cohs.append(math.cos(dphi))

        w_mean = sum(within_cohs) / len(within_cohs) if within_cohs else 0
        c_mean = sum(cross_cohs) / len(cross_cohs) if cross_cohs else 0
        ratio = abs(w_mean / c_mean) if abs(c_mean) > 1e-10 else 0
        print(f"  {fname:>10}  {w_mean:>10.6f}  {c_mean:>10.6f}  {ratio:>8.2f}")

    if wells is not None:
        print("\n  Channel 2: Well membership agreement")
        print(f"  {'Family':>10}  {'Same-well':>12}  {'Expected':>12}  {'Lift':>8}")

        for fname, fids in families.items():
            n_same = 0
            n_total = 0
            for i in range(len(fids)):
                for j in range(i + 1, len(fids)):
                    if wells[fids[i]] == wells[fids[j]]:
                        n_same += 1
                    n_total += 1
            frac_same = n_same / n_total if n_total > 0 else 0

            well_sizes = [(wells == w).sum().item() for w in range(analysis_data["n_wells"])]
            total_pairs = vocab_size * (vocab_size - 1) / 2
            expected_same = sum(s * (s - 1) / 2 for s in well_sizes) / total_pairs if total_pairs > 0 else 0

            lift = frac_same / expected_same if expected_same > 1e-10 else 0
            print(f"  {fname:>10}  {frac_same:>12.3f}  {expected_same:>12.3f}  {lift:>8.2f}x")

        print("\n  Channel 3: Within-well magnitude distance")
        print(f"  {'Family':>10}  {'Intra-mag':>12}  {'Inter-mag':>12}  {'Ratio':>8}")

        for fname, fids in families.items():
            intra_dists = []
            inter_dists = []
            for i in range(len(fids)):
                for j in range(i + 1, len(fids)):
                    d = abs(mean_mags[fids[i]].item() - mean_mags[fids[j]].item())
                    if wells[fids[i]] == wells[fids[j]]:
                        intra_dists.append(d)
                    else:
                        inter_dists.append(d)

            intra_m = sum(intra_dists) / len(intra_dists) if intra_dists else 0
            inter_m = sum(inter_dists) / len(inter_dists) if inter_dists else 0
            ratio = inter_m / intra_m if intra_m > 1e-15 else 0
            print(f"  {fname:>10}  {intra_m:>12.6f}  {inter_m:>12.6f}  {ratio:>8.2f}x")
    else:
        print("\n  Channels 2-3: Skipped (uniform magnitude, no well structure)")


# =============================================================================
# Band-Split Analysis: Low harmonics (lens) vs High harmonics (boundary)
# =============================================================================

def band_split_analysis(analysis_data, dataset):
    """Test whether low harmonics detect and high harmonics verify/contain.

    Split cos(n * delta_phi) into two bands:
      Low (n=1-6): broad resonance, relationship detection
      High (n=7-15): narrow resonance, boundary enforcement

    If high band is a boundary: moderate low-band pairs split into
    two populations on the high band (confirmed vs rejected).
    If both bands do the same job: they correlate strongly.
    """
    if analysis_data is None:
        return

    print(f"\n{'=' * 70}")
    print(f"  BAND-SPLIT ANALYSIS: Low harmonics (lens) vs High harmonics (boundary)")
    print(f"{'=' * 70}")

    phases = analysis_data["phases"]
    vocab_size = phases.shape[0]
    n_bands = phases.shape[1]
    families = dataset.get_family_ids()
    if not families:
        print("  No semantic families. Cannot evaluate.")
        return

    # Build cross-family sample
    family_ids_all = set()
    for fids in families.values():
        family_ids_all.update(fids)
    other_ids = [i for i in range(vocab_size) if i not in family_ids_all and i != 0]
    import random
    random.seed(77)
    cross_sample = random.sample(other_ids, min(200, len(other_ids)))

    LOW_BANDS = range(0, min(6, n_bands))    # n=1-6
    HIGH_BANDS = range(6, min(15, n_bands))  # n=7-15

    def band_coherence(id_a, id_b, bands):
        """Mean cos(n * delta_phi) over specified bands."""
        vals = []
        for k in bands:
            dphi = phases[id_a, k].item() - phases[id_b, k].item()
            vals.append(math.cos((k + 1) * dphi))
        return sum(vals) / len(vals) if vals else 0

    # ---- Per-family: low vs high band scores ----
    print(f"\n  Per-family mean coherence by band:")
    print(f"  {'Family':>10}  {'Type':>8}  {'Low(1-6)':>10}  {'High(7-15)':>10}  {'Ratio':>8}")

    all_within_low = []
    all_within_high = []
    all_cross_low = []
    all_cross_high = []

    for fname, fids in families.items():
        # Within-family pairs
        w_low, w_high = [], []
        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                w_low.append(band_coherence(fids[i], fids[j], LOW_BANDS))
                w_high.append(band_coherence(fids[i], fids[j], HIGH_BANDS))

        # Cross-family pairs
        c_low, c_high = [], []
        for fi in fids:
            for oi in cross_sample[:50]:
                c_low.append(band_coherence(fi, oi, LOW_BANDS))
                c_high.append(band_coherence(fi, oi, HIGH_BANDS))

        wl = sum(w_low) / len(w_low) if w_low else 0
        wh = sum(w_high) / len(w_high) if w_high else 0
        cl = sum(c_low) / len(c_low) if c_low else 0
        ch = sum(c_high) / len(c_high) if c_high else 0
        r_low = abs(wl / cl) if abs(cl) > 1e-10 else 0
        r_high = abs(wh / ch) if abs(ch) > 1e-10 else 0

        print(f"  {fname:>10}  {'within':>8}  {wl:>10.6f}  {wh:>10.6f}")
        print(f"  {'':>10}  {'cross':>8}  {cl:>10.6f}  {ch:>10.6f}")
        print(f"  {'':>10}  {'ratio':>8}  {r_low:>10.1f}x  {r_high:>10.1f}x")

        all_within_low.extend(w_low)
        all_within_high.extend(w_high)
        all_cross_low.extend(c_low)
        all_cross_high.extend(c_high)

    # ---- Correlation between low and high bands ----
    print(f"\n  Low-High correlation (all within-family pairs):")
    n_pairs = len(all_within_low)
    if n_pairs > 2:
        mean_l = sum(all_within_low) / n_pairs
        mean_h = sum(all_within_high) / n_pairs
        cov = sum((l - mean_l) * (h - mean_h) for l, h in
                  zip(all_within_low, all_within_high)) / n_pairs
        std_l = (sum((l - mean_l)**2 for l in all_within_low) / n_pairs) ** 0.5
        std_h = (sum((h - mean_h)**2 for h in all_within_high) / n_pairs) ** 0.5
        corr = cov / (std_l * std_h) if std_l > 1e-10 and std_h > 1e-10 else 0
        print(f"    r = {corr:.4f} (n={n_pairs} pairs)")
        if abs(corr) > 0.7:
            print(f"    STRONG CORRELATION: Both bands doing the same job.")
        elif abs(corr) > 0.3:
            print(f"    MODERATE CORRELATION: Partially independent roles.")
        else:
            print(f"    WEAK/NO CORRELATION: Bands carry independent information.")

    # ---- Key test: variance of high band at different low-band levels ----
    print(f"\n  Boundary test: high-band variance by low-band score")
    print(f"  (If high band enforces boundaries, moderate low-band pairs")
    print(f"   should show HIGH variance on high band -- split into confirmed/rejected)")

    # Combine within + cross pairs for full range
    all_low = all_within_low + all_cross_low
    all_high = all_within_high + all_cross_high

    # Bin by low-band score
    bins = [(-1.0, -0.3, "low-band < -0.3"),
            (-0.3, -0.05, "low-band [-0.3,-0.05]"),
            (-0.05, 0.05, "low-band [-0.05,0.05]"),
            (0.05, 0.3, "low-band [0.05,0.3]"),
            (0.3, 1.0, "low-band > 0.3")]

    print(f"  {'Bin':>25}  {'Count':>6}  {'High mean':>10}  {'High std':>10}  {'High CV':>8}")
    for lo, hi, label in bins:
        subset_high = [h for l, h in zip(all_low, all_high) if lo <= l < hi]
        if len(subset_high) >= 5:
            hm = sum(subset_high) / len(subset_high)
            hs = (sum((h - hm)**2 for h in subset_high) / len(subset_high)) ** 0.5
            hcv = abs(hs / hm * 100) if abs(hm) > 1e-10 else 0
            print(f"  {label:>25}  {len(subset_high):>6}  {hm:>10.6f}  {hs:>10.6f}  {hcv:>7.1f}%")
        else:
            print(f"  {label:>25}  {len(subset_high):>6}  (too few)")

    # ---- Interpretation ----
    print(f"\n  Interpretation:")
    print(f"    If high-band std is HIGHEST in the moderate low-band bins:")
    print(f"    -> High harmonics act as boundaries (splitting ambiguous pairs)")
    print(f"    If high-band std is UNIFORM across bins:")
    print(f"    -> Both bands do the same detection job at different scales")


# =============================================================================
# Main
# =============================================================================

def main():
    torch.manual_seed(42)
    print("=== Option A: Word-Level Shakespeare Transformer ===")
    print(f"  Device: {DEVICE}")
    print(f"  Architecture: {N_LAYER}L/{N_HEAD}H/{N_EMBD}D, context={BLOCK_SIZE} words")
    print(f"  Training: {MAX_ITERS} steps, progressive curriculum")
    print()

    # Load data
    text = download_shakespeare()
    dataset = WordDataset(text)
    print(f"  Train: {len(dataset.train_data):,} tokens, Val: {len(dataset.val_data):,} tokens")

    # Show which semantic families survived the vocabulary cut
    families = dataset.get_family_ids()
    print(f"\n  Semantic families in vocab:")
    for fname, fids in families.items():
        words = [dataset.itos[i] for i in fids]
        print(f"    {fname}: {words}")

    # =========================================================================
    # Train all three variants
    # =========================================================================
    results = {}

    ALL_MODES = ["frozen", "magnitude", "phase_only", "baseline"]

    for mode in ALL_MODES:
        model, history = train_variant(mode, dataset)
        results[mode] = {"model": model, "history": history}

    # =========================================================================
    # Summary: Training results
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  TRAINING SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Mode':>12}  {'Final Val':>10}  {'Params':>12}")
    print(f"  {'':>12}  {'Loss':>10}  {'':>12}")

    for mode in ALL_MODES:
        h = results[mode]["history"]
        final_val = h[-1][2] if h else float("nan")
        n_params = sum(p.numel() for p in results[mode]["model"].parameters()
                       if p.requires_grad)
        print(f"  {mode:>12}  {final_val:>10.4f}  {n_params:>12,}")

    # =========================================================================
    # Post-training analysis for each variant
    # =========================================================================
    analysis = {}
    for mode in ALL_MODES:
        analysis[mode] = analyze_magnitudes(results[mode]["model"], dataset)

    # =========================================================================
    # Triple-channel evaluation
    # =========================================================================
    for mode in ["magnitude", "phase_only", "baseline"]:
        if analysis[mode] is not None:
            print(f"\n  --- Triple-Channel for {mode} ---")
            triple_channel_eval(analysis[mode], dataset)

    # =========================================================================
    # Band-split analysis (baseline only — needs learned phase structure)
    # =========================================================================
    if analysis.get("baseline") is not None:
        band_split_analysis(analysis["baseline"], dataset)

    # =========================================================================
    # Final Verdict
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  FINAL VERDICT")
    print(f"{'=' * 70}")

    frozen_val = results["frozen"]["history"][-1][2] if results["frozen"]["history"] else float("nan")
    mag_val = results["magnitude"]["history"][-1][2] if results["magnitude"]["history"] else float("nan")
    phase_val = results["phase_only"]["history"][-1][2] if results["phase_only"]["history"] else float("nan")
    base_val = results["baseline"]["history"][-1][2] if results["baseline"]["history"] else float("nan")

    print(f"\n  Loss comparison:")
    print(f"    Frozen (phase+mag frozen):    {frozen_val:.4f}")
    print(f"    Magnitude (phase frozen):     {mag_val:.4f}")
    print(f"    Phase-only (mag frozen):      {phase_val:.4f}")
    print(f"    Baseline (both free):         {base_val:.4f}")

    # --- Key question 1: Does magnitude carry semantics? ---
    mag_data = analysis.get("magnitude")
    print(f"\n  Q1: Does magnitude carry semantics (phase frozen)?")
    if mag_data and mag_data["global_cv"] > 0.1:
        print(f"    Magnitude CV: {mag_data['global_cv']:.1f}% -- optimizer uses the freedom")
        print(f"    But check Measurement 2 above for semantic clustering.")
    else:
        print(f"    Magnitude barely varies. No semantic signal possible.")

    # --- Key question 2: Does phase carry semantics? ---
    phase_data = analysis.get("phase_only")
    print(f"\n  Q2: Does phase carry semantics (mag frozen)?")
    if phase_data:
        print(f"    Check Measurement 5 above for phase-based semantic clustering.")
        print(f"    Loss: {phase_val:.4f} (vs frozen {frozen_val:.4f}, "
              f"delta {(phase_val/frozen_val - 1)*100:+.1f}%)")
    else:
        print(f"    Phase-only analysis not available.")

    # --- Key question 3: Does magnitude ADD to phase? ---
    print(f"\n  Q3: Does magnitude add semantic value on top of phase?")
    base_data = analysis.get("baseline")
    if phase_data and base_data:
        print(f"    Compare phase_only vs baseline circle coherence ratios above.")
        gap = (base_val / phase_val - 1) * 100
        print(f"    Loss gap: baseline is {gap:+.1f}% vs phase-only")
        if base_val < phase_val * 0.98:
            print(f"    Magnitude freedom improves loss -- may add semantic structure.")
        elif base_val > phase_val * 1.02:
            print(f"    Magnitude freedom HURTS -- harmonic structure regularises.")
        else:
            print(f"    Losses similar -- magnitude adds little beyond phase.")

    # --- Key question 4: magnitude vs phase_only ---
    print(f"\n  Q4: Phase freedom vs magnitude freedom (same param budget)?")
    print(f"    Magnitude (phase frozen): {mag_val:.4f}")
    print(f"    Phase-only (mag frozen):  {phase_val:.4f}")
    gap_mp = abs(mag_val - phase_val) / min(mag_val, phase_val) * 100
    if gap_mp < 1.0:
        print(f"    EQUIVALENT ({gap_mp:.2f}% gap). Type of freedom doesn't matter.")
    elif mag_val < phase_val:
        print(f"    Magnitude freedom wins by {gap_mp:.1f}%.")
    else:
        print(f"    Phase freedom wins by {gap_mp:.1f}%.")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
