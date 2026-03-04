"""
Phase B: Integration Sweep — Three Spherical Coherence Integration Points

Runs 7 variants against MLP baseline on char-level Shakespeare (4L/4H/128D):
  1. mlp           -- MLP baseline (ceiling)
  2. full_stack    -- Frozen everything + Kerr-ODE (Phase A architecture)
  3. mag_stack     -- Frozen phase + trainable magnitude (Integration 1)
  4. band_stack    -- Band-aware routing: L0 -> bands 1-8, L1-3 -> bands 9-64
  5. two_stage     -- Magnitude frozen for steps 0-1334, freed for 1334-2000
  6. band_mag      -- Band routing + trainable magnitude
  7. band_two      -- Band routing + two-stage magnitude

Integration points tested (from spherical coherence investigation, v2.20.0):
  1. Trainable magnitude — per-token-per-band magnitude parameter (from Option A
     coupling principle: 3.7% word-level regularisation, 2.25% char-level)
  2. Band-aware routing — restrict L0 to low bands, L1-3 to high bands (from
     band-split finding: r=0.05 orthogonality between low/high harmonics)
  3. Two-stage training — freeze magnitude during phase organisation, free after
     phase stabilises (from coupling principle: phase first, magnitude second)

Results (RTX 4070 Ti, CUDA, 2000 steps, batch 64, lr 3e-4):

  Rank  Variant              Val Loss  % of MLP  vs Full Stack
  ----  -------------------  --------  --------  -------------
    1   MLP baseline          1.6932   100.0%     -6.45%
    2   Two-stage             1.7752    95.2%     -1.91%   <-- WINNER
    3   Mag stack             1.7800    94.9%     -1.64%
    4   Full stack            1.8098    93.1%     baseline
    5   Band + mag            1.9511    84.8%     +7.81%
    6   Band + two-stage      1.9659    83.9%     +8.63%
    7   Band stack            1.9754    83.3%     +9.15%

Key findings:
  - Two-stage wins: 95.2% of MLP at 43.1% params (+1.91% over frozen)
  - Band routing HURTS: all band variants -7 to -9% (L0 needs full spectrum)
  - Magnitude CV: two-stage 2.46% (surgical) vs mag_stack 6.92% (exploratory)
  - Constrained freedom produces more precise optimisation
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
BAND_SPLIT = 8  # L0 handles bands 1-8, L1-L3 handle bands 9-64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROG_STAGES = [
    (0,    667,  8),
    (667,  1334, 24),
    (1334, 2000, 64),
]

# Step at which two-stage unfreezes magnitude
TWO_STAGE_UNFREEZE = 1334


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
            table[c, h * 2] = math.cos(n * theta) * scale
            table[c, h * 2 + 1] = math.sin(n * theta) * scale
    return table


def build_positional_table(max_len, n_embd):
    n_harmonics = n_embd // 2
    scale = 1.0 / math.sqrt(n_harmonics)
    table = torch.zeros(max_len, n_embd)
    for pos in range(max_len):
        for h in range(n_harmonics):
            freq = 1.0 / (10000.0 ** (2.0 * h / n_embd))
            table[pos, h * 2] = math.cos(pos * freq) * scale
            table[pos, h * 2 + 1] = math.sin(pos * freq) * scale
    return table


def build_band_mask(n_embd, band_start, band_end):
    """Build a mask that selects bands [band_start, band_end) (0-indexed)."""
    mask = torch.zeros(n_embd)
    for h in range(band_start, band_end):
        mask[h * 2] = 1.0
        mask[h * 2 + 1] = 1.0
    return mask


# =============================================================================
# Per-Band Linear Layer (L0)
# =============================================================================

class PerBandLinear(nn.Module):
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
# Kerr-ODE Layer with RK4
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
        self.omega = nn.Parameter(
            torch.arange(1, n_bands + 1, dtype=torch.float32) / n_bands)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('neighbor_kernel',
                             torch.tensor([[[1.0, 1.0, 0.0, 1.0, 1.0]]]))
        self.register_buffer('max_magnitude', torch.zeros(n_bands))

    @property
    def gamma(self):
        return F.softplus(self._gamma_raw)

    def _derivative(self, r, s, gamma):
        mag_sq = r * r + s * s
        neighbor_sum = F.conv1d(
            mag_sq.unsqueeze(1), self.neighbor_kernel, padding=2).squeeze(1)
        phi = self.omega + self.alpha * mag_sq + self.beta * neighbor_sum
        return -gamma * r - phi * s, -gamma * s + phi * r

    def _rk4_step(self, r, s, dt, gamma):
        dr1, ds1 = self._derivative(r, s, gamma)
        dr2, ds2 = self._derivative(r + 0.5*dt*dr1, s + 0.5*dt*ds1, gamma)
        dr3, ds3 = self._derivative(r + 0.5*dt*dr2, s + 0.5*dt*ds2, gamma)
        dr4, ds4 = self._derivative(r + dt*dr3, s + dt*ds3, gamma)
        return (r + (dt/6)*(dr1 + 2*dr2 + 2*dr3 + dr4),
                s + (dt/6)*(ds1 + 2*ds2 + 2*ds3 + ds4))

    def forward(self, x):
        B, T, C = x.size()
        bt = B * T
        bands = x.view(bt, self.n_bands, 2)
        r, s = bands[:, :, 0].contiguous(), bands[:, :, 1].contiguous()
        dt, gamma = self.dt, self.gamma
        for _ in range(self.n_steps):
            r, s = self._rk4_step(r, s, dt, gamma)
            if self.training:
                with torch.no_grad():
                    mag = torch.sqrt(r*r + s*s)
                    self.max_magnitude = torch.max(
                        self.max_magnitude, mag.max(dim=0).values)
        out = torch.stack([r, s], dim=2).reshape(bt, C)
        return self.out_proj(out).view(B, T, C)


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
            .view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, x):
        B, T, C = x.size()
        hd = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


# =============================================================================
# Block and Model
# =============================================================================

class Block(nn.Module):
    def __init__(self, ffn_module, band_route_mask=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.ffn = ffn_module
        if band_route_mask is not None:
            self.register_buffer("band_route", band_route_mask)
        else:
            self.band_route = None

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        ffn_out = self.ffn(self.ln_2(x))
        if self.band_route is not None:
            ffn_out = ffn_out * self.band_route
        x = x + ffn_out
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, mode="mlp"):
        super().__init__()
        self.mode = mode
        self.vocab_size = vocab_size
        self.n_bands = N_BANDS

        self.register_buffer("wte", build_harmonic_table(vocab_size, N_EMBD))
        self.register_buffer("wpe", build_positional_table(BLOCK_SIZE, N_EMBD))

        # Trainable magnitude for modes that use it
        use_mag = mode in ("mag_stack", "band_mag", "two_stage", "band_two")
        if use_mag:
            freeze_mag = mode in ("two_stage", "band_two")
            self.tok_mag = nn.Parameter(
                torch.ones(vocab_size, N_BANDS),
                requires_grad=not freeze_mag)
        else:
            self.tok_mag = None

        # Band routing masks
        use_band = mode in ("band_stack", "band_mag", "band_two")
        low_mask = build_band_mask(N_EMBD, 0, BAND_SPLIT) if use_band else None
        high_mask = build_band_mask(N_EMBD, BAND_SPLIT, N_BANDS) if use_band else None

        # Build layer stack
        if mode == "mlp":
            layers = [Block(MLP()) for _ in range(N_LAYER)]
        elif mode in ("full_stack", "mag_stack", "two_stage"):
            ffn_modules = [
                PerBandLinear(), KerrODELayer(n_steps=8),
                KerrODELayer(n_steps=8), KerrODELayer(n_steps=8),
            ]
            layers = [Block(ffn) for ffn in ffn_modules]
        elif mode in ("band_stack", "band_mag", "band_two"):
            layers = [
                Block(PerBandLinear(), band_route_mask=low_mask),
                Block(KerrODELayer(n_steps=8), band_route_mask=high_mask),
                Block(KerrODELayer(n_steps=8), band_route_mask=high_mask),
                Block(KerrODELayer(n_steps=8), band_route_mask=high_mask),
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.blocks = nn.ModuleList(layers)
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.register_buffer("band_mask", torch.ones(N_EMBD))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

        # Reset tok_mag after _init_weights (keep at 1.0)
        if self.tok_mag is not None:
            self.tok_mag.data.fill_(1.0)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_mag = self.tok_mag.numel() if self.tok_mag is not None else 0
        mag_info = f" (mag: {n_mag:,}, frozen={not self.tok_mag.requires_grad})" if n_mag else ""
        band_info = " (band-routed)" if use_band else ""
        print(f"  {mode}: {n_params:,} trainable params{mag_info}{band_info}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def set_active_bands(self, n_active):
        mask = torch.zeros(N_EMBD, device=self.band_mask.device)
        for h in range(min(n_active, N_BANDS)):
            mask[h * 2] = 1.0
            mask[h * 2 + 1] = 1.0
        self.band_mask = mask

    def unfreeze_magnitude(self):
        """Unfreeze magnitude for two-stage training."""
        if self.tok_mag is not None and not self.tok_mag.requires_grad:
            self.tok_mag.requires_grad_(True)
            return True
        return False

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = F.embedding(idx, self.wte)

        if self.tok_mag is not None:
            mag = self.tok_mag[idx]
            mag_exp = mag.unsqueeze(-1).expand(-1, -1, -1, 2)
            tok_emb = tok_emb * mag_exp.reshape(B, T, N_EMBD)

        x = (tok_emb + self.wpe[:T]) * self.band_mask
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_magnitude_stats(self):
        if self.tok_mag is None:
            return None
        mag = self.tok_mag.detach().cpu()
        band_stds = mag.std(dim=0)
        band_means = mag.mean(dim=0)
        band_cvs = (band_stds / band_means * 100).clamp(max=999)
        gm = mag.mean().item()
        gs = mag.std().item()
        return {
            "global_cv": (gs / gm * 100) if gm > 1e-10 else 0,
            "early_cv": band_cvs[:BAND_SPLIT].mean().item(),
            "mid_cv": band_cvs[BAND_SPLIT:24].mean().item(),
            "late_cv": band_cvs[24:].mean().item(),
            "global_mean": gm,
        }


# =============================================================================
# Data
# =============================================================================

def download_shakespeare():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "experiments", "data")
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
    for start, end, bands in PROG_STAGES:
        if start <= iter_num < end:
            return bands
    return 64


def train_variant(name, mode, dataset):
    use_prog = mode != "mlp"
    is_two_stage = mode in ("two_stage", "band_two")

    print(f"\n{'=' * 70}")
    print(f"  Training: {name} (mode={mode})")
    if use_prog:
        for s, e, b in PROG_STAGES:
            print(f"    Steps {s}-{e}: bands 1-{b}")
    if is_two_stage:
        print(f"    Magnitude unfreezes at step {TWO_STAGE_UNFREEZE}")
    print(f"{'=' * 70}")

    model = GPT(dataset.vocab_size, mode=mode).to(DEVICE)

    # For two-stage modes, exclude frozen tok_mag from initial optimizer
    # (add_param_group later would fail if it's already tracked)
    if is_two_stage and model.tok_mag is not None:
        params = [p for n, p in model.named_parameters()
                  if p.requires_grad and n != "tok_mag"]
    else:
        params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    history = []
    start_time = time.time()
    current_bands = 64
    mag_unfrozen = False

    for iter_num in range(MAX_ITERS):
        if use_prog:
            stage_bands = get_stage(iter_num)
            if stage_bands != current_bands:
                model.set_active_bands(stage_bands)
                current_bands = stage_bands
                print(f"  >>> Stage change at step {iter_num}: bands 1-{stage_bands}")

                # Log magnitude at transitions
                ms = model.get_magnitude_stats()
                if ms:
                    print(f"    Mag CV: global={ms['global_cv']:.2f}%"
                          f"  early={ms['early_cv']:.2f}%"
                          f"  mid={ms['mid_cv']:.2f}%"
                          f"  late={ms['late_cv']:.2f}%")

        # Two-stage: unfreeze magnitude at the right step
        if is_two_stage and not mag_unfrozen and iter_num >= TWO_STAGE_UNFREEZE:
            if model.unfreeze_magnitude():
                optimizer.add_param_group({"params": [model.tok_mag]})
                mag_unfrozen = True
                n_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
                print(f"  >>> Magnitude UNFROZEN at step {iter_num}"
                      f" ({n_params:,} params now)")

        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            saved_mask = model.band_mask.clone()
            model.set_active_bands(64)
            losses = estimate_loss(model, dataset)
            if use_prog and current_bands < 64:
                model.band_mask = saved_mask

            elapsed = time.time() - start_time
            extra = f" [bands 1-{current_bands}]" if use_prog else ""
            print(f"  step {iter_num:>5} | train {losses['train']:.4f}"
                  f" | val {losses['val']:.4f} | {elapsed:.1f}s{extra}")

            ms = model.get_magnitude_stats()
            if ms:
                print(f"    Mag: CV={ms['global_cv']:.2f}%"
                      f"  mean={ms['global_mean']:.4f}")

            history.append((iter_num, losses["train"], losses["val"]))
            if math.isnan(losses["train"]):
                print("  !!! NaN -- stopping.")
                break

        x, y = dataset.get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    total = time.time() - start_time
    final_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Done in {total:.1f}s")

    # Final magnitude diagnostic
    ms = model.get_magnitude_stats()
    if ms:
        print(f"  Final mag: CV={ms['global_cv']:.2f}%"
              f"  early={ms['early_cv']:.2f}%"
              f"  mid={ms['mid_cv']:.2f}%"
              f"  late={ms['late_cv']:.2f}%")

    return {
        "name": name, "mode": mode, "history": history,
        "n_params": final_params, "time": total,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    torch.manual_seed(42)
    print("=" * 70)
    print("  Integration Sweep: All Variants")
    print(f"  Device: {DEVICE}")
    print(f"  Band split: L0 -> bands 1-{BAND_SPLIT},"
          f" L1-L3 -> bands {BAND_SPLIT+1}-{N_BANDS}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text)} chars, {dataset.vocab_size} vocab")
    print(f"  Model: {N_LAYER}L/{N_HEAD}H/{N_EMBD}D, {MAX_ITERS} steps")

    variants = [
        ("MLP baseline",       "mlp"),
        ("Full stack",         "full_stack"),
        ("Mag stack",          "mag_stack"),
        ("Band stack",         "band_stack"),
        ("Two-stage",          "two_stage"),
        ("Band + mag",         "band_mag"),
        ("Band + two-stage",   "band_two"),
    ]

    results = []
    for name, mode in variants:
        r = train_variant(name, mode, dataset)
        results.append(r)

    # =========================================================================
    # Results Table
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: Integration Sweep")
    print(f"{'=' * 70}")

    mlp_val = results[0]["history"][-1][2]

    print(f"\n  {'Variant':<22} {'Val':>8} {'Train':>8} {'Params':>10}"
          f" {'Time':>6} {'vs MLP':>8} {'vs Full':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*10} {'-'*6} {'-'*8} {'-'*8}")

    full_val = results[1]["history"][-1][2]
    for r in results:
        if not r["history"]:
            continue
        _, tr, vl = r["history"][-1]
        vs_mlp = (vl / mlp_val - 1) * 100
        vs_full = (vl / full_val - 1) * 100
        print(f"  {r['name']:<22} {vl:>8.4f} {tr:>8.4f} {r['n_params']:>10,}"
              f" {r['time']:>5.0f}s {vs_mlp:>+7.2f}% {vs_full:>+7.2f}%")

    # =========================================================================
    # Convergence
    # =========================================================================
    print(f"\n  Convergence (val loss at each eval point):")
    header = f"  {'Step':>6}"
    for r in results:
        header += f"  {r['name'][:10]:>10}"
    print(header)
    print(f"  {'-'*6}" + f"  {'-'*10}" * len(results))

    min_len = min(len(r["history"]) for r in results)
    for i in range(min_len):
        step = results[0]["history"][i][0]
        line = f"  {step:>6}"
        for r in results:
            vl = r["history"][i][2]
            line += f"  {vl:>10.4f}"
        print(line)

    # =========================================================================
    # Verdict
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}")

    # Find best non-MLP variant
    stack_results = results[1:]  # exclude MLP
    best = min(stack_results, key=lambda r: r["history"][-1][2])
    best_val = best["history"][-1][2]
    best_perf = (1 - (best_val - mlp_val) / mlp_val) * 100

    print(f"\n  Best variant: {best['name']}")
    print(f"    Val loss: {best_val:.4f}")
    print(f"    {best_perf:.1f}% of MLP at"
          f" {best['n_params']/results[0]['n_params']*100:.1f}% params")
    print(f"    vs full_stack: {(best_val/full_val - 1)*100:+.2f}%")

    # Rank all
    print(f"\n  Ranking (by val loss):")
    ranked = sorted(results, key=lambda r: r["history"][-1][2])
    for i, r in enumerate(ranked):
        vl = r["history"][-1][2]
        perf = (1 - (vl - mlp_val) / mlp_val) * 100
        print(f"    {i+1}. {r['name']:<22} val={vl:.4f}"
              f"  ({perf:.1f}% of MLP)")

    print(f"\n  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
