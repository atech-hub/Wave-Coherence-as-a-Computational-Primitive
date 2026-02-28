"""
Phase 22: Inverse Kerr -- Understanding the Transform by Reversing It

Principle: to understand what the Kerr-ODE layer does forward, test what
happens in reverse. Feed output back through the ODE with negated dt.
Per-band reconstruction error separates three categories:

  1. REVERSIBLE: low error with full reverse -> spectral remixing.
     Could potentially be replaced by an analytical transform.

  2. IRREVERSIBLE-DAMPING: high error full reverse, low error gamma=0 reverse.
     Energy dissipation is eating information. Structural, not computational.

  3. IRREVERSIBLE-NONLINEAR: high error even with gamma=0 reverse.
     Genuine nonlinear computation. This is where the real work lives.

Method:
  - Train 8-step scalar Kerr-ODE (best from Phase 21)
  - For each layer, capture ODE input/output on validation data
  - Run three reverse passes:
    a) Full reverse (64 steps, fine-grained to minimize numerical error)
    b) Gamma=0 reverse (remove dissipation, keep Kerr terms)
    c) Control: random vector forward+backward for noise floor
  - Categorize each of 64 bands per layer
  - Cross-reference with amplification ratio from forward pass

Usage:
    python experiments/phase22_inverse_kerr.py
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

N_REVERSE_STEPS = 64  # Fine-grained reverse for accuracy
N_ANALYSIS_BATCHES = 10  # Average over multiple batches for stable estimates

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Harmonic Embedding -- frozen, deterministic
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
# Kerr-ODE Layer with exposed ODE integration
# =============================================================================

class KerrODELayer(nn.Module):
    def __init__(self, n_bands=N_BANDS, n_embd=N_EMBD, n_steps=8):
        super().__init__()
        self.n_bands = n_bands
        self.n_embd = n_embd
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps

        self._gamma_raw = nn.Parameter(torch.full((n_bands,), math.log(math.exp(0.1) - 1)))
        omega_init = torch.arange(1, n_bands + 1, dtype=torch.float32) / n_bands
        self.omega = nn.Parameter(omega_init)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('neighbor_kernel',
                             torch.tensor([[[1.0, 1.0, 0.0, 1.0, 1.0]]]))

    @property
    def gamma(self):
        return F.softplus(self._gamma_raw)

    def ode_step(self, r, s, dt, gamma=None):
        """Single ODE integration step. Allows overriding gamma."""
        if gamma is None:
            gamma = self.gamma
        mag_sq = r * r + s * s
        neighbor_sum = F.conv1d(
            mag_sq.unsqueeze(1), self.neighbor_kernel, padding=2
        ).squeeze(1)
        phi = self.omega + self.alpha * mag_sq + self.beta * neighbor_sum
        dr_dt = -gamma * r - phi * s
        ds_dt = -gamma * s + phi * r
        r_new = r + dt * dr_dt
        s_new = s + dt * ds_dt
        return r_new, s_new

    def forward_ode(self, r, s):
        """Run ODE forward, return (r_out, s_out) and whether clamping occurred."""
        dt = self.dt
        clamped = torch.zeros(r.shape[1], dtype=torch.bool, device=r.device)
        for _ in range(self.n_steps):
            r, s = self.ode_step(r, s, dt)
            # Track clamping
            clamped = clamped | (r.abs() > 9.99).any(dim=0) | (s.abs() > 9.99).any(dim=0)
            r = torch.clamp(r, -10.0, 10.0)
            s = torch.clamp(s, -10.0, 10.0)
        return r, s, clamped

    def reverse_ode(self, r, s, n_steps=64, zero_gamma=False):
        """Run ODE backward (negative dt). Optionally zero out gamma.

        Clamps to [-1000, 1000] to prevent NaN while allowing large values.
        Returns (r, s, blew_up) where blew_up is per-band bool.
        """
        dt = -1.0 / n_steps  # negative dt for reverse
        gamma = torch.zeros_like(self.gamma) if zero_gamma else self.gamma
        blew_up = torch.zeros(r.shape[1], dtype=torch.bool, device=r.device)
        for _ in range(n_steps):
            r, s = self.ode_step(r, s, dt, gamma=gamma)
            # Track blow-up before clamping
            blew_up = blew_up | (r.abs() > 999).any(dim=0) | (s.abs() > 999).any(dim=0)
            blew_up = blew_up | torch.isnan(r).any(dim=0) | torch.isnan(s).any(dim=0)
            # Clamp to prevent NaN propagation, but wide enough to measure real error
            r = torch.clamp(torch.nan_to_num(r, nan=0.0), -1000.0, 1000.0)
            s = torch.clamp(torch.nan_to_num(s, nan=0.0), -1000.0, 1000.0)
        return r, s, blew_up

    def forward(self, x):
        B, T, C = x.size()
        bt = B * T
        bands = x.view(bt, self.n_bands, 2)
        r = bands[:, :, 0].contiguous()
        s = bands[:, :, 1].contiguous()
        r, s, _ = self.forward_ode(r, s)
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
    def __init__(self, ffn_type="mlp", n_steps=8):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.ffn_type = ffn_type
        if ffn_type == "mlp":
            self.ffn = MLP()
        elif ffn_type == "kerr":
            self.ffn = KerrODELayer(n_steps=n_steps)
        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}")

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, ffn_type="mlp", n_steps=8):
        super().__init__()
        self.ffn_type = ffn_type
        self.register_buffer("wte", build_harmonic_table(vocab_size, N_EMBD))
        self.register_buffer("wpe", build_positional_table(BLOCK_SIZE, N_EMBD))
        self.blocks = nn.ModuleList(
            [Block(ffn_type=ffn_type, n_steps=n_steps) for _ in range(N_LAYER)]
        )
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  {ffn_type} model: {n_params:,} trainable parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = F.embedding(idx, self.wte)
        pos_emb = self.wpe[:T]
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_layer_inputs(self, idx):
        """Forward pass that captures input to each Kerr-ODE layer."""
        B, T = idx.size()
        tok_emb = F.embedding(idx, self.wte)
        pos_emb = self.wpe[:T]
        x = tok_emb + pos_emb
        layer_inputs = []
        for block in self.blocks:
            x = x + block.attn(block.ln_1(x))
            kerr_input = block.ln_2(x)
            layer_inputs.append(kerr_input)
            x = x + block.ffn(kerr_input)
        return layer_inputs


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


def train_model(mode_name, ffn_type, n_steps, dataset):
    print(f"\n{'=' * 70}")
    print(f"  Training: {mode_name}")
    print(f"{'=' * 70}")

    model = GPT(dataset.vocab_size, ffn_type=ffn_type, n_steps=n_steps).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start = time.time()
    for iter_num in range(MAX_ITERS):
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model, dataset)
            elapsed = time.time() - start
            print(f"  step {iter_num:>5} | train {losses['train']:.4f}"
                  f" | val {losses['val']:.4f} | {elapsed:.1f}s")

            if math.isnan(losses["train"]):
                print("  !!! NaN detected -- stopping.")
                break

        x, y = dataset.get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    total = time.time() - start
    print(f"  Training complete in {total:.1f}s")

    final_losses = estimate_loss(model, dataset)
    return model, final_losses


# =============================================================================
# Reversibility Analysis
# =============================================================================

@torch.no_grad()
def analyze_reversibility(model, dataset):
    """Core analysis: forward + reverse ODE per layer, per band."""
    model.eval()
    print(f"\n{'=' * 70}")
    print(f"  REVERSIBILITY ANALYSIS")
    print(f"  Reverse steps: {N_REVERSE_STEPS} (vs 8 forward)")
    print(f"  Averaging over {N_ANALYSIS_BATCHES} validation batches")
    print(f"{'=' * 70}")

    # Accumulators per layer, per band
    n_layers = len(model.blocks)
    full_errors = [torch.zeros(N_BANDS, device=DEVICE) for _ in range(n_layers)]
    nogamma_errors = [torch.zeros(N_BANDS, device=DEVICE) for _ in range(n_layers)]
    control_errors = [torch.zeros(N_BANDS, device=DEVICE) for _ in range(n_layers)]
    input_magnitudes = [torch.zeros(N_BANDS, device=DEVICE) for _ in range(n_layers)]
    amplification_ratios = [torch.zeros(N_BANDS, device=DEVICE) for _ in range(n_layers)]
    fwd_clamp_counts = [torch.zeros(N_BANDS, device=DEVICE) for _ in range(n_layers)]
    full_blowup_counts = [torch.zeros(N_BANDS, device=DEVICE) for _ in range(n_layers)]
    nogamma_blowup_counts = [torch.zeros(N_BANDS, device=DEVICE) for _ in range(n_layers)]
    ctrl_blowup_counts = [torch.zeros(N_BANDS, device=DEVICE) for _ in range(n_layers)]

    for batch_idx in range(N_ANALYSIS_BATCHES):
        x, _ = dataset.get_batch("val")
        layer_inputs = model.get_layer_inputs(x)

        for layer_idx, block in enumerate(model.blocks):
            if not isinstance(block.ffn, KerrODELayer):
                continue

            kerr = block.ffn
            inp = layer_inputs[layer_idx]
            B, T, C = inp.size()
            bt = B * T
            bands = inp.view(bt, N_BANDS, 2)
            r_in = bands[:, :, 0].contiguous()
            s_in = bands[:, :, 1].contiguous()

            # Input magnitude per band (for normalisation)
            in_mag = torch.sqrt(r_in * r_in + s_in * s_in)  # (bt, n_bands)
            input_magnitudes[layer_idx] += in_mag.mean(dim=0)

            # --- Forward pass ---
            r_out, s_out, clamped = kerr.forward_ode(r_in.clone(), s_in.clone())
            fwd_clamp_counts[layer_idx] += clamped.float()

            # Amplification ratio per band
            out_mag = torch.sqrt(r_out * r_out + s_out * s_out)
            ratio = out_mag / (in_mag + 1e-10)
            amplification_ratios[layer_idx] += ratio.mean(dim=0)

            # --- Full reverse (8x more steps for accuracy) ---
            r_rev, s_rev, blew_full = kerr.reverse_ode(
                r_out.clone(), s_out.clone(),
                n_steps=N_REVERSE_STEPS, zero_gamma=False
            )
            full_blowup_counts[layer_idx] += blew_full.float()
            err_full = torch.sqrt((r_in - r_rev)**2 + (s_in - s_rev)**2)
            full_errors[layer_idx] += err_full.mean(dim=0)

            # --- Gamma=0 reverse (remove dissipation) ---
            r_rev_ng, s_rev_ng, blew_ng = kerr.reverse_ode(
                r_out.clone(), s_out.clone(),
                n_steps=N_REVERSE_STEPS, zero_gamma=True
            )
            nogamma_blowup_counts[layer_idx] += blew_ng.float()
            err_nogamma = torch.sqrt((r_in - r_rev_ng)**2 + (s_in - s_rev_ng)**2)
            nogamma_errors[layer_idx] += err_nogamma.mean(dim=0)

            # --- Control: random vector forward + backward ---
            r_rand = torch.randn_like(r_in) * in_mag.mean()
            s_rand = torch.randn_like(s_in) * in_mag.mean()
            r_fwd, s_fwd, _ = kerr.forward_ode(r_rand.clone(), s_rand.clone())
            r_ctrl, s_ctrl, blew_ctrl = kerr.reverse_ode(
                r_fwd.clone(), s_fwd.clone(),
                n_steps=N_REVERSE_STEPS, zero_gamma=False
            )
            ctrl_blowup_counts[layer_idx] += blew_ctrl.float()
            err_ctrl = torch.sqrt((r_rand - r_ctrl)**2 + (s_rand - s_ctrl)**2)
            control_errors[layer_idx] += err_ctrl.mean(dim=0)

    # Average over batches
    for i in range(n_layers):
        full_errors[i] /= N_ANALYSIS_BATCHES
        nogamma_errors[i] /= N_ANALYSIS_BATCHES
        control_errors[i] /= N_ANALYSIS_BATCHES
        input_magnitudes[i] /= N_ANALYSIS_BATCHES
        amplification_ratios[i] /= N_ANALYSIS_BATCHES
        fwd_clamp_counts[i] /= N_ANALYSIS_BATCHES
        full_blowup_counts[i] /= N_ANALYSIS_BATCHES
        nogamma_blowup_counts[i] /= N_ANALYSIS_BATCHES
        ctrl_blowup_counts[i] /= N_ANALYSIS_BATCHES

    return {
        'full_errors': full_errors,
        'nogamma_errors': nogamma_errors,
        'control_errors': control_errors,
        'input_magnitudes': input_magnitudes,
        'amplification_ratios': amplification_ratios,
        'fwd_clamp_counts': fwd_clamp_counts,
        'full_blowup': full_blowup_counts,
        'nogamma_blowup': nogamma_blowup_counts,
        'ctrl_blowup': ctrl_blowup_counts,
    }


def classify_bands(results):
    """Classify each band into reversible / irreversible-damping / irreversible-nonlinear.

    Uses blow-up tracking as primary signal. A band that blew up during reverse
    is definitively irreversible. For bands that didn't blow up, use error magnitude.
    """
    print(f"\n{'=' * 70}")
    print(f"  BAND CLASSIFICATION")
    print(f"{'=' * 70}")

    n_layers = len(results['full_errors'])
    classifications = []

    for layer_idx in range(n_layers):
        full_err = results['full_errors'][layer_idx]
        nogamma_err = results['nogamma_errors'][layer_idx]
        ctrl_err = results['control_errors'][layer_idx]
        in_mag = results['input_magnitudes'][layer_idx]
        amp_ratio = results['amplification_ratios'][layer_idx]
        fwd_clamp = results['fwd_clamp_counts'][layer_idx]
        full_blew = results['full_blowup'][layer_idx]
        ng_blew = results['nogamma_blowup'][layer_idx]
        ctrl_blew = results['ctrl_blowup'][layer_idx]

        # Blow-up rate (fraction of batches where this band blew up)
        full_blew_any = full_blew > 0.5  # blew up in majority of batches
        ng_blew_any = ng_blew > 0.5
        ctrl_blew_any = ctrl_blew > 0.5

        # Relative errors (normalised by input magnitude) -- only valid for non-blown bands
        rel_full = full_err / (in_mag + 1e-10)
        rel_nogamma = nogamma_err / (in_mag + 1e-10)
        rel_ctrl = ctrl_err / (in_mag + 1e-10)

        # For blown-up bands, set relative error to a large sentinel
        rel_full = torch.where(full_blew_any, torch.tensor(999.0, device=DEVICE), rel_full)
        rel_nogamma = torch.where(ng_blew_any, torch.tensor(999.0, device=DEVICE), rel_nogamma)
        rel_ctrl = torch.where(ctrl_blew_any, torch.tensor(999.0, device=DEVICE), rel_ctrl)

        # Noise floor: for bands where control didn't blow up, use 2x control error
        # For bands where even control blew up, use a fixed threshold (0.5 = 50% relative error)
        noise_floor = torch.where(
            ctrl_blew_any,
            torch.tensor(0.5, device=DEVICE),
            2.0 * rel_ctrl
        )

        # Classification:
        # Bands that blew up on full reverse but NOT on gamma=0 -> damping-driven
        # Bands that blew up on both full and gamma=0 -> nonlinear-driven
        # Bands that didn't blow up and have low error -> reversible
        stable_full = ~full_blew_any & (rel_full < noise_floor)
        reversible = stable_full

        # Blew up with gamma but not without -> damping is the cause
        irrev_damping_blowup = full_blew_any & ~ng_blew_any
        # Didn't blow up, but error too high; gamma=0 fixes it -> damping error
        irrev_damping_error = ~full_blew_any & (rel_full >= noise_floor) & \
                              ~ng_blew_any & (rel_nogamma < noise_floor)
        irrev_damping = irrev_damping_blowup | irrev_damping_error

        # Everything else is nonlinear-irreversible
        irrev_nonlinear = ~reversible & ~irrev_damping

        n_rev = reversible.sum().item()
        n_damp = irrev_damping.sum().item()
        n_nonlin = irrev_nonlinear.sum().item()

        classifications.append({
            'reversible': reversible,
            'irrev_damping': irrev_damping,
            'irrev_nonlinear': irrev_nonlinear,
            'rel_full': rel_full,
            'rel_nogamma': rel_nogamma,
            'rel_ctrl': rel_ctrl,
            'amp_ratio': amp_ratio,
            'fwd_clamp': fwd_clamp,
            'full_blew': full_blew,
            'ng_blew': ng_blew,
        })

        # --- Report ---
        print(f"\n  Layer {layer_idx}:")
        print(f"    Reversible:              {n_rev:>3}/64  ({100*n_rev/64:.1f}%)")
        print(f"    Irreversible (damping):   {n_damp:>3}/64  ({100*n_damp/64:.1f}%)")
        print(f"    Irreversible (nonlinear): {n_nonlin:>3}/64  ({100*n_nonlin/64:.1f}%)")

        # Blow-up stats
        n_full_blew = full_blew_any.sum().item()
        n_ng_blew = ng_blew_any.sum().item()
        n_ctrl_blew = ctrl_blew_any.sum().item()
        print(f"    Blow-up: full_reverse={n_full_blew}/64"
              f"  gamma0_reverse={n_ng_blew}/64"
              f"  control={n_ctrl_blew}/64")
        print(f"    Forward clamping: {(fwd_clamp > 0.5).sum().item()}/64 bands")

        # Error stats (only for non-blown bands)
        stable_mask = ~full_blew_any
        n_stable = stable_mask.sum().item()
        if n_stable > 0:
            stable_full_err = rel_full[stable_mask]
            print(f"    Stable bands ({n_stable}): relative error"
                  f" mean={stable_full_err.mean():.4f}"
                  f" [{stable_full_err.min():.4f}, {stable_full_err.max():.4f}]")

        ng_stable = ~ng_blew_any
        n_ng_stable = ng_stable.sum().item()
        if n_ng_stable > 0:
            stable_ng_err = rel_nogamma[ng_stable]
            print(f"    Gamma=0 stable bands ({n_ng_stable}): relative error"
                  f" mean={stable_ng_err.mean():.4f}"
                  f" [{stable_ng_err.min():.4f}, {stable_ng_err.max():.4f}]")

        # Amplification breakdown by category
        if n_rev > 0:
            rev_amp = amp_ratio[reversible]
            print(f"    Reversible bands -- amplification:"
                  f" mean={rev_amp.mean():.3f} [{rev_amp.min():.3f}, {rev_amp.max():.3f}]")
        if n_damp > 0:
            damp_amp = amp_ratio[irrev_damping]
            print(f"    Damping-irrev bands -- amplification:"
                  f" mean={damp_amp.mean():.3f} [{damp_amp.min():.3f}, {damp_amp.max():.3f}]")
        if n_nonlin > 0:
            nl_amp = amp_ratio[irrev_nonlinear]
            print(f"    Nonlinear-irrev bands -- amplification:"
                  f" mean={nl_amp.mean():.3f} [{nl_amp.min():.3f}, {nl_amp.max():.3f}]")

        # Which bands are in each category?
        if 0 < n_rev <= 20:
            idx = torch.where(reversible)[0].tolist()
            print(f"    Reversible band indices: {idx}")
        if 0 < n_damp <= 20:
            idx = torch.where(irrev_damping)[0].tolist()
            print(f"    Damping-irrev band indices: {idx}")
        if 0 < n_nonlin <= 20:
            idx = torch.where(irrev_nonlinear)[0].tolist()
            print(f"    Nonlinear-irrev band indices: {idx}")

    return classifications


def cross_reference_amplification(classifications):
    """Cross-reference amplified bands with reversibility category."""
    print(f"\n{'=' * 70}")
    print(f"  CROSS-REFERENCE: Amplification vs Reversibility")
    print(f"{'=' * 70}")

    for layer_idx, c in enumerate(classifications):
        amp = c['amp_ratio']

        # Use >1.5x threshold since 8-step Kerr doesn't amplify as aggressively
        high_amp = amp > 1.5
        n_high = high_amp.sum().item()

        if n_high == 0:
            print(f"\n  Layer {layer_idx}: no bands >1.5x amplified")
            continue

        n_rev = (high_amp & c['reversible']).sum().item()
        n_damp = (high_amp & c['irrev_damping']).sum().item()
        n_nonlin = (high_amp & c['irrev_nonlinear']).sum().item()

        print(f"\n  Layer {layer_idx}: {n_high} bands with >1.5x amplification")
        print(f"    Reversible:              {n_rev:>3}/{n_high}"
              f"  -> spectral routing")
        print(f"    Irreversible (damping):   {n_damp:>3}/{n_high}"
              f"  -> energy dissipation")
        print(f"    Irreversible (nonlinear): {n_nonlin:>3}/{n_high}"
              f"  -> genuine nonlinear computation")

        if n_high > 0:
            if n_rev / n_high > 0.7:
                print(f"    >> VERDICT: amplification is mostly spectral routing")
            elif n_nonlin / n_high > 0.7:
                print(f"    >> VERDICT: amplification is mostly nonlinear computation")
            elif n_damp / n_high > 0.7:
                print(f"    >> VERDICT: amplification is mostly damping-related")
            else:
                print(f"    >> VERDICT: mixed -- routing + computation + damping")


def overall_summary(classifications):
    """Summary across all layers with interpretation."""
    print(f"\n{'=' * 70}")
    print(f"  OVERALL SUMMARY")
    print(f"{'=' * 70}")

    total_bands = 64 * len(classifications)
    total_rev = sum(c['reversible'].sum().item() for c in classifications)
    total_damp = sum(c['irrev_damping'].sum().item() for c in classifications)
    total_nonlin = sum(c['irrev_nonlinear'].sum().item() for c in classifications)

    print(f"\n  Across all {len(classifications)} layers ({total_bands} total bands):")
    print(f"    Reversible:              {total_rev:>3}/{total_bands}"
          f"  ({100*total_rev/total_bands:.1f}%)")
    print(f"    Irreversible (damping):   {total_damp:>3}/{total_bands}"
          f"  ({100*total_damp/total_bands:.1f}%)")
    print(f"    Irreversible (nonlinear): {total_nonlin:>3}/{total_bands}"
          f"  ({100*total_nonlin/total_bands:.1f}%)")

    # Layer-by-layer breakdown table
    print(f"\n  {'Layer':<8} {'Reversible':>12} {'Damping':>12} {'Nonlinear':>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for i, c in enumerate(classifications):
        n_r = c['reversible'].sum().item()
        n_d = c['irrev_damping'].sum().item()
        n_n = c['irrev_nonlinear'].sum().item()
        print(f"  L{i:<7} {n_r:>8}/64   {n_d:>8}/64   {n_n:>8}/64")

    # Interpretation
    rev_pct = 100 * total_rev / total_bands
    nonlin_pct = 100 * total_nonlin / total_bands

    print(f"\n  INTERPRETATION:")
    if rev_pct > 70:
        print(f"    >70% reversible ({rev_pct:.1f}%)")
        print(f"    -> Kerr-ODE is mostly doing spectral remixing")
        print(f"    -> Sylvester-style analytical solve could replace most integration")
        print(f"    -> MASSIVE compute savings potential")
    elif nonlin_pct > 70:
        print(f"    >70% irreversible-nonlinear ({nonlin_pct:.1f}%)")
        print(f"    -> Nonlinear dynamics are genuinely essential")
        print(f"    -> RK4 is the right next step (better integration, not different structure)")
    else:
        print(f"    Mixed: {rev_pct:.1f}% reversible, {nonlin_pct:.1f}% nonlinear")
        print(f"    -> Hybrid approach: analytical for reversible, ODE for the rest")
        print(f"    -> Could skip ODE integration for {rev_pct:.0f}% of bands")

    # Depth gradient
    rev_per_layer = [c['reversible'].sum().item() for c in classifications]
    nl_per_layer = [c['irrev_nonlinear'].sum().item() for c in classifications]
    if len(rev_per_layer) >= 2:
        if rev_per_layer[0] > rev_per_layer[-1] + 5:
            print(f"\n    Depth gradient: shallow layers more reversible,"
                  f" deep layers more nonlinear")
            print(f"    -> Consistent with Phase 21 finding: deep layers amplify Kerr")
        elif rev_per_layer[-1] > rev_per_layer[0] + 5:
            print(f"\n    Depth gradient: deep layers more reversible,"
                  f" shallow layers more nonlinear")
            print(f"    -> Unexpected: deep layers doing less nonlinear work")
        else:
            print(f"\n    No strong depth gradient in reversibility")

    print()
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  Phase 22: Inverse Kerr -- Understanding the Transform")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text)} characters, {dataset.vocab_size} unique")
    print(f"  Train: {len(dataset.train_data)} | Val: {len(dataset.val_data)}")
    print(f"  Model: {N_LAYER} layers, {N_HEAD} heads, {N_EMBD} dim, {N_BANDS} bands")
    print(f"  Framework: PyTorch {torch.__version__}")

    print(f"\n  Design:")
    print(f"    Train 8-step scalar Kerr-ODE (best from Phase 21)")
    print(f"    Run ODE forward, then reverse with {N_REVERSE_STEPS} steps")
    print(f"    Three reverse passes: full, gamma=0, control (random)")
    print(f"    Classify each band: reversible / damping / nonlinear")

    # --- Train ---
    kerr_model, kerr_losses = train_model(
        "Kerr-ODE 8-step", "kerr", 8, dataset
    )
    std_model, std_losses = train_model(
        "MLP baseline", "mlp", 8, dataset
    )

    print(f"\n  Trained models:")
    print(f"    MLP baseline: val {std_losses['val']:.4f}")
    print(f"    Kerr-ODE 8s:  val {kerr_losses['val']:.4f}"
          f"  ({(kerr_losses['val']/std_losses['val'] - 1)*100:+.2f}% vs MLP)")

    # Report final Kerr parameters
    print(f"\n  Final Kerr-ODE parameters:")
    for i, block in enumerate(kerr_model.blocks):
        if isinstance(block.ffn, KerrODELayer):
            k = block.ffn
            gamma = k.gamma
            print(f"    L{i}: alpha={k.alpha.item():.4f}  beta={k.beta.item():.4f}"
                  f"  gamma=[{gamma.min().item():.4f},{gamma.max().item():.4f}]"
                  f" avg={gamma.mean().item():.4f}"
                  f"  omega=[{k.omega.min().item():.4f},{k.omega.max().item():.4f}]")

    # --- Reversibility analysis ---
    results = analyze_reversibility(kerr_model, dataset)
    classifications = classify_bands(results)
    cross_reference_amplification(classifications)
    overall_summary(classifications)


if __name__ == "__main__":
    main()
