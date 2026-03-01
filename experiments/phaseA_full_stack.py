"""
Phase A: Full Stack Integration Test

Assemble all validated components into one model and train on Shakespeare.
Does the combined stack perform within margin of the component-level predictions?

Components:
  - Embeddings: Harmonic frozen (no gradient) -- Test 25, Phase 17
  - Layer 0: PerBandLinear (analytical, no ODE) -- Phase 22b
  - Layers 1-3: Kerr-ODE with RK4 solver, no clamp -- Phase 22d
  - Attention: Standard learned Q/K -- Phase 18/19 boundary
  - Training: Progressive curriculum (bands 1-8, 1-24, 1-64) -- Phase 6

Two modes:
  1. MLP baseline (standard training, frozen harmonic embeddings)
  2. Full stack (all components active simultaneously)

Expected range:
  - MLP baseline: ~1.69 val loss
  - Full stack: ~1.81 (93.5% of MLP from Phase 22d ceiling)
  - If 1.69-1.85: stack validated (components don't interfere)
  - If > 1.85: components interfere
  - If < 1.81: components synergise

Usage:
    python experiments/phaseA_full_stack.py
"""

import math
import os
import time
import urllib.request

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
BLOCK_SIZE = 256
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_ITERS = 2000
EVAL_INTERVAL = 200
EVAL_ITERS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Progressive curriculum schedule (adapted from Phase 6 for 2000 steps)
PROG_STAGES = [
    (0,    667,  8),    # Stage 1: bands 1-8
    (667,  1334, 24),   # Stage 2: bands 1-24
    (1334, 2000, 64),   # Stage 3: all 64 bands
]


# =============================================================================
# Harmonic Embedding
# =============================================================================

def build_harmonic_table(vocab_size, n_embd):
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


# =============================================================================
# Per-Band Linear Layer (L0 replacement, from Phase 22b)
# =============================================================================

class PerBandLinear(nn.Module):
    """Learned 2x2 transform per band + output projection.
    Replaces L0's ODE with an analytical linear operation."""

    def __init__(self, n_bands=N_BANDS, n_embd=N_EMBD):
        super().__init__()
        self.n_bands = n_bands
        self.n_embd = n_embd
        self.band_w = nn.Parameter(torch.zeros(n_bands, 2, 2))
        with torch.no_grad():
            for k in range(n_bands):
                self.band_w.data[k] = torch.eye(2)
        self.band_b = nn.Parameter(torch.zeros(n_bands, 2))
        self.out_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        bt = B * T
        bands = x.view(bt, self.n_bands, 2)
        out = torch.einsum('bni,nij->bnj', bands, self.band_w) + self.band_b
        out = out.reshape(bt, C)
        out = self.out_proj(out)
        return out.view(B, T, C)


# =============================================================================
# Kerr-ODE Layer with RK4 (from Phase 22d)
# =============================================================================

class KerrODELayer(nn.Module):
    def __init__(self, n_bands=N_BANDS, n_embd=N_EMBD, n_steps=8):
        super().__init__()
        self.n_bands = n_bands
        self.n_embd = n_embd
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps

        self._gamma_raw = nn.Parameter(
            torch.full((n_bands,), math.log(math.exp(0.1) - 1)))
        omega_init = torch.arange(1, n_bands + 1, dtype=torch.float32) / n_bands
        self.omega = nn.Parameter(omega_init)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('neighbor_kernel',
                             torch.tensor([[[1.0, 1.0, 0.0, 1.0, 1.0]]]))

        # Range tracking
        self.register_buffer('max_magnitude', torch.zeros(n_bands))

    @property
    def gamma(self):
        return F.softplus(self._gamma_raw)

    def _derivative(self, r, s, gamma):
        mag_sq = r * r + s * s
        neighbor_sum = F.conv1d(
            mag_sq.unsqueeze(1), self.neighbor_kernel, padding=2
        ).squeeze(1)
        phi = self.omega + self.alpha * mag_sq + self.beta * neighbor_sum
        dr_dt = -gamma * r - phi * s
        ds_dt = -gamma * s + phi * r
        return dr_dt, ds_dt

    def _rk4_step(self, r, s, dt, gamma):
        dr1, ds1 = self._derivative(r, s, gamma)
        r2 = r + 0.5 * dt * dr1
        s2 = s + 0.5 * dt * ds1
        dr2, ds2 = self._derivative(r2, s2, gamma)
        r3 = r + 0.5 * dt * dr2
        s3 = s + 0.5 * dt * ds2
        dr3, ds3 = self._derivative(r3, s3, gamma)
        r4 = r + dt * dr3
        s4 = s + dt * ds3
        dr4, ds4 = self._derivative(r4, s4, gamma)
        r_new = r + (dt / 6.0) * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4)
        s_new = s + (dt / 6.0) * (ds1 + 2.0 * ds2 + 2.0 * ds3 + ds4)
        return r_new, s_new

    def forward(self, x):
        B, T, C = x.size()
        bt = B * T
        bands = x.view(bt, self.n_bands, 2)
        r = bands[:, :, 0].contiguous()
        s = bands[:, :, 1].contiguous()

        dt = self.dt
        gamma = self.gamma

        for _ in range(self.n_steps):
            r, s = self._rk4_step(r, s, dt, gamma)

            if self.training:
                with torch.no_grad():
                    mag = torch.sqrt(r * r + s * s)
                    batch_max = mag.max(dim=0).values
                    self.max_magnitude = torch.max(self.max_magnitude, batch_max)

        out = torch.stack([r, s], dim=2).reshape(bt, C)
        out = self.out_proj(out)
        return out.view(B, T, C)


# =============================================================================
# Standard MLP
# =============================================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


# =============================================================================
# Attention
# =============================================================================

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


# =============================================================================
# Block and Model
# =============================================================================

class Block(nn.Module):
    def __init__(self, ffn_module):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.ffn = ffn_module

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, mode="mlp"):
        super().__init__()
        self.mode = mode
        self.n_bands = N_BANDS

        # Frozen harmonic embeddings (both modes use these)
        self.register_buffer("wte", build_harmonic_table(vocab_size, N_EMBD))
        self.register_buffer("wpe", build_positional_table(BLOCK_SIZE, N_EMBD))

        # Build layer stack based on mode
        if mode == "mlp":
            layers = [Block(MLP()) for _ in range(N_LAYER)]
        elif mode == "full_stack":
            ffn_modules = [
                PerBandLinear(),          # L0: analytical linear
                KerrODELayer(n_steps=8),  # L1: Kerr-ODE RK4
                KerrODELayer(n_steps=8),  # L2: Kerr-ODE RK4
                KerrODELayer(n_steps=8),  # L3: Kerr-ODE RK4
            ]
            layers = [Block(ffn) for ffn in ffn_modules]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.blocks = nn.ModuleList(layers)
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)

        # Band mask for progressive curriculum (starts as all-ones)
        self.register_buffer("band_mask", torch.ones(N_EMBD))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("out_proj.weight"):
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

    def set_active_bands(self, n_active):
        """Set how many harmonic bands are active (for progressive curriculum).
        Masks out bands beyond n_active by zeroing their cos/sin pairs."""
        mask = torch.zeros(N_EMBD, device=self.band_mask.device)
        for h in range(min(n_active, N_BANDS)):
            mask[h * 2] = 1.0
            mask[h * 2 + 1] = 1.0
        self.band_mask = mask

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = F.embedding(idx, self.wte)
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


# =============================================================================
# Data
# =============================================================================

def download_shakespeare():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    filepath = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(filepath):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "python", "data", "shakespeare.txt")
        if os.path.exists(alt):
            filepath = alt
        else:
            os.makedirs(data_dir, exist_ok=True)
            print("  Downloading Shakespeare...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, filepath)
    with open(filepath, "r") as f:
        return f.read()


class Dataset:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        data = [self.stoi[c] for c in text]
        n = int(0.9 * len(data))
        self.train_data = torch.tensor(data[:n], dtype=torch.long)
        self.val_data = torch.tensor(data[n:], dtype=torch.long)

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
        return x.to(DEVICE), y.to(DEVICE)


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
    """Return number of active bands for the current iteration."""
    for start, end, bands in PROG_STAGES:
        if start <= iter_num < end:
            return bands
    return 64


def train_mode(mode_name, mode, use_progressive, dataset):
    print(f"\n{'=' * 70}")
    print(f"  Training: {mode_name}")
    if use_progressive:
        for start, end, bands in PROG_STAGES:
            print(f"    Steps {start}-{end}: bands 1-{bands}")
    else:
        print(f"    Standard training (all bands)")
    print(f"{'=' * 70}")

    model = GPT(dataset.vocab_size, mode=mode).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    history = []
    start_time = time.time()
    current_bands = 64

    for iter_num in range(MAX_ITERS):
        # Update progressive curriculum
        if use_progressive:
            stage_bands = get_stage(iter_num)
            if stage_bands != current_bands:
                model.set_active_bands(stage_bands)
                current_bands = stage_bands
                print(f"  >>> Stage change at step {iter_num}:"
                      f" now using bands 1-{stage_bands}")

        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            # For evaluation, always use all bands
            saved_mask = model.band_mask.clone()
            model.set_active_bands(64)
            losses = estimate_loss(model, dataset)
            if use_progressive and current_bands < 64:
                model.band_mask = saved_mask

            elapsed = time.time() - start_time
            extra = f" [bands 1-{current_bands}]" if use_progressive else ""
            print(f"  step {iter_num:>5} | train {losses['train']:.4f}"
                  f" | val {losses['val']:.4f} | {elapsed:.1f}s{extra}")

            # Report Kerr-ODE range for full stack
            if mode == "full_stack":
                for i, block in enumerate(model.blocks):
                    if isinstance(block.ffn, KerrODELayer):
                        mag_max = block.ffn.max_magnitude.max().item()
                        mag_mean = block.ffn.max_magnitude.mean().item()
                        alpha = block.ffn.alpha.item()
                        gamma = block.ffn.gamma
                        print(f"    L{i}: peak_mag={mag_max:.2f}"
                              f" mean_mag={mag_mean:.2f}"
                              f" alpha={alpha:.4f}"
                              f" gamma=[{gamma.min().item():.3f},"
                              f"{gamma.max().item():.3f}]")
                    elif isinstance(block.ffn, PerBandLinear):
                        w = block.ffn.band_w.detach()
                        frob = torch.sqrt((w * w).sum(dim=(1, 2))).mean().item()
                        print(f"    L{i}: PerBandLinear"
                              f" avg_frob={frob:.4f}")

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
    print(f"  Training complete in {total:.1f}s")

    # Band energy diagnostic: how much energy in each frequency band at output
    model.eval()
    model.set_active_bands(64)
    with torch.no_grad():
        x, _ = dataset.get_batch("val")
        tok_emb = F.embedding(x, model.wte)
        pos_emb = model.wpe[:x.size(1)]
        h = tok_emb + pos_emb
        for block in model.blocks:
            h = block(h)
        h = model.ln_f(h)

        # Compute energy per band
        B, T, C = h.size()
        bands = h.view(B * T, N_BANDS, 2)
        band_energy = (bands[:, :, 0] ** 2 + bands[:, :, 1] ** 2).mean(dim=0)
        total_energy = band_energy.sum().item()
        top_bands = band_energy.topk(8)

        print(f"\n  Band energy distribution (output):")
        print(f"    Total energy: {total_energy:.4f}")
        print(f"    Top 8 bands (of 64):")
        for idx, val in zip(top_bands.indices.tolist(), top_bands.values.tolist()):
            pct = val / total_energy * 100
            print(f"      band {idx+1:>3}: {val:.4f} ({pct:.1f}%)")
        top8_pct = top_bands.values.sum().item() / total_energy * 100
        print(f"    Top 8 total: {top8_pct:.1f}% of energy")

    return {
        "mode_name": mode_name,
        "mode": mode,
        "progressive": use_progressive,
        "history": history,
        "n_params": n_params,
        "time": total,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  Phase A: Full Stack Integration Test")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text)} characters, {dataset.vocab_size} unique")
    print(f"  Train: {len(dataset.train_data)} | Val: {len(dataset.val_data)}")
    print(f"  Model: {N_LAYER} layers, {N_HEAD} heads, {N_EMBD} dim, {N_BANDS} bands")
    print(f"  Framework: PyTorch {torch.__version__}")

    print(f"\n  Components under test:")
    print(f"    Embeddings: frozen harmonic (both modes)")
    print(f"    MLP baseline: standard 4-layer MLP")
    print(f"    Full stack: PerBandLinear L0 + Kerr-ODE RK4 L1-L3 + progressive")

    print(f"\n  Expected range:")
    print(f"    MLP baseline: ~1.69")
    print(f"    Full stack:   ~1.81 (93.5% of MLP from Phase 22d ceiling)")
    print(f"    If 1.69-1.85: stack validated")
    print(f"    If > 1.85:    components interfere")
    print(f"    If < 1.81:    components synergise")

    # Train both modes
    result_mlp = train_mode(
        "MLP baseline", "mlp", use_progressive=False, dataset=dataset)
    result_stack = train_mode(
        "Full stack (prog)", "full_stack", use_progressive=True, dataset=dataset)

    # =========================================================================
    # Comparison
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: Full Stack Integration Test")
    print(f"{'=' * 70}")

    mlp_val = result_mlp["history"][-1][2]
    stack_val = result_stack["history"][-1][2]
    gap_pct = (stack_val / mlp_val - 1) * 100
    perf_pct = (1 - (stack_val - mlp_val) / mlp_val) * 100

    print(f"\n  {'Mode':<28} {'Val':>8} {'Train':>8} {'Params':>10} {'Time':>7}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*10} {'-'*7}")
    for r in [result_mlp, result_stack]:
        if not r["history"]:
            continue
        _, train_l, val_l = r["history"][-1]
        print(f"  {r['mode_name']:<28} {val_l:>8.4f} {train_l:>8.4f}"
              f" {r['n_params']:>10,} {r['time']:>6.0f}s")

    print(f"\n  Gap: {gap_pct:+.2f}%")
    print(f"  Performance ratio: {perf_pct:.1f}% of MLP")
    print(f"  Parameter ratio: {result_stack['n_params']/result_mlp['n_params']*100:.1f}%"
          f" ({result_mlp['n_params']-result_stack['n_params']:,} fewer)")

    # Convergence comparison
    print(f"\n  Convergence:")
    print(f"  {'Step':>6}  {'MLP':>10}  {'Full stack':>10}  {'Gap':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")
    min_len = min(len(result_mlp["history"]), len(result_stack["history"]))
    for i in range(min_len):
        step = result_mlp["history"][i][0]
        m_val = result_mlp["history"][i][2]
        s_val = result_stack["history"][i][2]
        g = (s_val / m_val - 1) * 100
        print(f"  {step:>6}  {m_val:>10.4f}  {s_val:>10.4f}  {g:>+7.2f}%")

    # =========================================================================
    # Verdict
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}")

    if stack_val > 1.85:
        print(f"\n  FAIL: Components interfere")
        print(f"  Val loss {stack_val:.4f} exceeds 1.85 threshold")
        print(f"  The combined stack is worse than components predicted")
    elif stack_val < 1.81:
        print(f"\n  SYNERGY: Components help each other")
        print(f"  Val loss {stack_val:.4f} beats the 1.81 component ceiling")
        print(f"  The combined stack is better than expected")
    else:
        print(f"\n  VALIDATED: Stack performs within margin")
        print(f"  Val loss {stack_val:.4f} is within [1.69, 1.85]")
        print(f"  Components work together without interfering")

    print(f"\n  Bottom line:")
    print(f"    {perf_pct:.1f}% of MLP performance")
    print(f"    {result_stack['n_params']/result_mlp['n_params']*100:.1f}%"
          f" of MLP parameters")
    print(f"    Frozen embeddings + analytical L0 + Kerr-ODE RK4 L1-L3"
          f" + progressive curriculum")
    print(f"    The infrastructure {'holds' if stack_val < 1.85 else 'needs work'}.")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
