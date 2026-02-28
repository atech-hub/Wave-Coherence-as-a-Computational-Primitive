"""
Phase 21b: Per-Band Kerr Coefficients with Diagnostic Monitoring

Phase 21 Kerr-ODE achieved 92% of MLP at 12% of FFN parameters, but alpha and
beta were global scalars shared across all 64 bands. The model already
differentiates by depth (L3 amplifies Kerr, L0 suppresses). This experiment
gives each band its own alpha_k and beta_k, testing whether band-level
nonlinear control closes the remaining 8% gap to MLP.

Changes from Phase 21:
  - scalar alpha -> alpha_k (64 learned values, one per band)
  - scalar beta  -> beta_k  (64 learned values, one per band)
  - All initialised to 0.1
  - New parameters: 128 per layer (512 total across 4 layers)
  - Total Kerr-ODE FFN: ~16,768/layer (still 7.8x fewer than MLP's 131,712)

Diagnostic monitoring (logged every 200 iterations):
  1. Per-band Kerr stats: alpha_k, beta_k, gamma_k, omega_k (min/max/mean/std)
  2. Gradient norms per parameter group (flag dead gradients < 1e-7)
  3. Band energy analysis: input |Z_k|^2 vs output |Z_k|^2 per band

Success criteria:
  - Per-band alpha/beta std > 0.02 by iteration 2000 (model used the freedom)
  - Val loss improvement over scalar baseline
  - Non-uniform band energy patterns

Usage:
    python experiments/phase21b_perband_kerr.py
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
# Kerr-ODE Layer -- SCALAR alpha/beta (Phase 21 baseline reproduction)
# =============================================================================

class KerrODEScalar(nn.Module):
    """Phase 21 Kerr-ODE with scalar alpha and beta. Baseline for comparison."""

    def __init__(self, n_bands=N_BANDS, n_embd=N_EMBD, n_steps=4):
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

    def forward(self, x):
        B, T, C = x.size()
        bt = B * T
        bands = x.view(bt, self.n_bands, 2)
        r = bands[:, :, 0].contiguous()
        s = bands[:, :, 1].contiguous()
        dt = self.dt
        gamma = self.gamma

        for _ in range(self.n_steps):
            mag_sq = r * r + s * s
            neighbor_sum = F.conv1d(
                mag_sq.unsqueeze(1), self.neighbor_kernel, padding=2
            ).squeeze(1)
            phi = self.omega + self.alpha * mag_sq + self.beta * neighbor_sum
            dr_dt = -gamma * r - phi * s
            ds_dt = -gamma * s + phi * r
            r = r + dt * dr_dt
            s = s + dt * ds_dt
            r = torch.clamp(r, -10.0, 10.0)
            s = torch.clamp(s, -10.0, 10.0)

        out = torch.stack([r, s], dim=2).reshape(bt, C)
        out = self.out_proj(out)
        return out.view(B, T, C)


# =============================================================================
# Kerr-ODE Layer -- PER-BAND alpha_k / beta_k
# =============================================================================

class KerrODEPerBand(nn.Module):
    """
    Kerr-ODE with per-band nonlinearity coefficients.

    Each of the 64 bands gets its own alpha_k and beta_k, allowing the model
    to learn band-specific Kerr strength. Low-frequency bands may need different
    nonlinear coupling than high-frequency bands.

    Extra parameters vs scalar: 63 alpha + 63 beta = 126 per layer (512 total).
    Negligible impact on total parameter count.
    """

    def __init__(self, n_bands=N_BANDS, n_embd=N_EMBD, n_steps=4):
        super().__init__()
        self.n_bands = n_bands
        self.n_embd = n_embd
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps

        # Per-band damping (softplus-constrained)
        self._gamma_raw = nn.Parameter(torch.full((n_bands,), math.log(math.exp(0.1) - 1)))

        # Per-band resonant frequency
        omega_init = torch.arange(1, n_bands + 1, dtype=torch.float32) / n_bands
        self.omega = nn.Parameter(omega_init)

        # PER-BAND nonlinearity coefficients (the experiment)
        self.alpha = nn.Parameter(torch.full((n_bands,), 0.1))  # self-phase per band
        self.beta = nn.Parameter(torch.full((n_bands,), 0.1))   # cross-phase per band

        # Output projection
        self.out_proj = nn.Linear(n_embd, n_embd)

        # Fixed neighbor kernel for cross-phase modulation
        self.register_buffer('neighbor_kernel',
                             torch.tensor([[[1.0, 1.0, 0.0, 1.0, 1.0]]]))

    @property
    def gamma(self):
        return F.softplus(self._gamma_raw)

    def forward(self, x, return_diagnostics=False):
        B, T, C = x.size()
        bt = B * T
        bands = x.view(bt, self.n_bands, 2)
        r = bands[:, :, 0].contiguous()
        s = bands[:, :, 1].contiguous()

        # Capture input band energy for diagnostics
        if return_diagnostics:
            input_energy = (r * r + s * s).mean(dim=0).detach()  # (n_bands,)

        dt = self.dt
        gamma = self.gamma

        for _ in range(self.n_steps):
            mag_sq = r * r + s * s
            neighbor_sum = F.conv1d(
                mag_sq.unsqueeze(1), self.neighbor_kernel, padding=2
            ).squeeze(1)
            # Per-band alpha and beta -- this is the key change
            phi = self.omega + self.alpha * mag_sq + self.beta * neighbor_sum
            dr_dt = -gamma * r - phi * s
            ds_dt = -gamma * s + phi * r
            r = r + dt * dr_dt
            s = s + dt * ds_dt
            r = torch.clamp(r, -10.0, 10.0)
            s = torch.clamp(s, -10.0, 10.0)

        # Capture output band energy for diagnostics
        if return_diagnostics:
            output_energy = (r * r + s * s).mean(dim=0).detach()  # (n_bands,)

        out = torch.stack([r, s], dim=2).reshape(bt, C)
        out = self.out_proj(out)
        result = out.view(B, T, C)

        if return_diagnostics:
            return result, input_energy, output_energy
        return result

    def get_param_summary(self):
        """Per-band parameter statistics."""
        with torch.no_grad():
            gamma = self.gamma
            return {
                'alpha_mean': self.alpha.mean().item(),
                'alpha_std': self.alpha.std().item(),
                'alpha_min': self.alpha.min().item(),
                'alpha_max': self.alpha.max().item(),
                'beta_mean': self.beta.mean().item(),
                'beta_std': self.beta.std().item(),
                'beta_min': self.beta.min().item(),
                'beta_max': self.beta.max().item(),
                'gamma_mean': gamma.mean().item(),
                'gamma_std': gamma.std().item(),
                'gamma_min': gamma.min().item(),
                'gamma_max': gamma.max().item(),
                'omega_mean': self.omega.mean().item(),
                'omega_std': self.omega.std().item(),
                'omega_min': self.omega.min().item(),
                'omega_max': self.omega.max().item(),
                'proj_norm': self.out_proj.weight.pow(2).sum().sqrt().item(),
            }

    def get_grad_norms(self):
        """Gradient norms per parameter group."""
        result = {}
        for name, param in [('alpha', self.alpha), ('beta', self.beta),
                            ('gamma_raw', self._gamma_raw), ('omega', self.omega)]:
            if param.grad is not None:
                norm = param.grad.norm().item()
                result[name] = norm
                if norm < 1e-7:
                    result[f'{name}_DEAD'] = True
        if self.out_proj.weight.grad is not None:
            result['out_proj_w'] = self.out_proj.weight.grad.norm().item()
        if self.out_proj.bias is not None and self.out_proj.bias.grad is not None:
            result['out_proj_b'] = self.out_proj.bias.grad.norm().item()
        return result


# =============================================================================
# Standard MLP
# =============================================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD)

    def forward(self, x, return_diagnostics=False):
        result = self.c_proj(F.gelu(self.c_fc(x)))
        if return_diagnostics:
            return result, None, None
        return result


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
    def __init__(self, ffn_type="mlp", n_steps=4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.ffn_type = ffn_type
        if ffn_type == "mlp":
            self.ffn = MLP()
        elif ffn_type == "kerr_scalar":
            self.ffn = KerrODEScalar(n_steps=n_steps)
        elif ffn_type == "kerr_perband":
            self.ffn = KerrODEPerBand(n_steps=n_steps)
        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}")

    def forward(self, x, return_diagnostics=False):
        x = x + self.attn(self.ln_1(x))
        if return_diagnostics and hasattr(self.ffn, 'forward'):
            ffn_in = self.ln_2(x)
            ffn_out = self.ffn(ffn_in, return_diagnostics=True)
            if isinstance(ffn_out, tuple):
                ffn_result, in_energy, out_energy = ffn_out
                x = x + ffn_result
                return x, in_energy, out_energy
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, ffn_type="mlp", n_steps=4):
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

        if ffn_type.startswith("kerr"):
            ffn_params = sum(
                p.numel() for name, p in self.named_parameters()
                if "ffn." in name
            )
            ode_only = sum(
                p.numel() for name, p in self.named_parameters()
                if "ffn." in name and "out_proj" not in name
            )
            print(f"  Kerr-ODE FFN params: {ffn_params:,} ({ffn_params // N_LAYER:,} per layer x {N_LAYER})")
            print(f"    ODE params: {ode_only:,} ({ode_only // N_LAYER} per layer)")
            print(f"    Output proj: {ffn_params - ode_only:,} ({(ffn_params - ode_only) // N_LAYER:,} per layer)")
            print(f"  Standard MLP would be: {N_LAYER * 131712:,} FFN params")
            print(f"  Reduction: {N_LAYER * 131712 / ffn_params:.1f}x fewer FFN params")
            print(f"  Integration steps: {n_steps}, dt = {1.0/n_steps:.3f}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx, targets=None, return_diagnostics=False):
        B, T = idx.size()
        tok_emb = F.embedding(idx, self.wte)
        pos_emb = self.wpe[:T]
        x = tok_emb + pos_emb

        diagnostics = []
        for block in self.blocks:
            if return_diagnostics:
                result = block(x, return_diagnostics=True)
                if isinstance(result, tuple):
                    x, in_e, out_e = result
                    diagnostics.append((in_e, out_e))
                else:
                    x = result
                    diagnostics.append((None, None))
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_diagnostics:
            return logits, loss, diagnostics
        return logits, loss


# =============================================================================
# Diagnostics
# =============================================================================

def log_diagnostics(model, iter_num):
    """Log the three diagnostic categories for per-band Kerr layers."""
    if not model.ffn_type.startswith("kerr"):
        return

    print(f"\n  --- Diagnostics @ iter {iter_num} ---")

    for i, block in enumerate(model.blocks):
        ffn = block.ffn
        if not isinstance(ffn, (KerrODEPerBand, KerrODEScalar)):
            continue

        # ---- 1. Per-band parameter stats ----
        if isinstance(ffn, KerrODEPerBand):
            s = ffn.get_param_summary()
            print(f"  L{i} params:")
            print(f"    alpha: mean={s['alpha_mean']:.4f} std={s['alpha_std']:.4f}"
                  f" [{s['alpha_min']:.4f}, {s['alpha_max']:.4f}]")
            print(f"    beta:  mean={s['beta_mean']:.4f} std={s['beta_std']:.4f}"
                  f" [{s['beta_min']:.4f}, {s['beta_max']:.4f}]")
            print(f"    gamma: mean={s['gamma_mean']:.4f} std={s['gamma_std']:.4f}"
                  f" [{s['gamma_min']:.4f}, {s['gamma_max']:.4f}]")
            print(f"    omega: mean={s['omega_mean']:.4f} std={s['omega_std']:.4f}"
                  f" [{s['omega_min']:.4f}, {s['omega_max']:.4f}]")
            print(f"    proj_norm: {s['proj_norm']:.4f}")
        elif isinstance(ffn, KerrODEScalar):
            with torch.no_grad():
                gamma = ffn.gamma
                print(f"  L{i} params: alpha={ffn.alpha.item():.4f}"
                      f"  beta={ffn.beta.item():.4f}"
                      f"  gamma=[{gamma.min().item():.4f},{gamma.max().item():.4f}]"
                      f" avg={gamma.mean().item():.4f}"
                      f"  proj_norm={ffn.out_proj.weight.pow(2).sum().sqrt().item():.4f}")

        # ---- 2. Gradient norms ----
        if isinstance(ffn, KerrODEPerBand):
            g = ffn.get_grad_norms()
            parts = []
            dead = []
            for k, v in g.items():
                if k.endswith('_DEAD'):
                    dead.append(k.replace('_DEAD', ''))
                else:
                    parts.append(f"{k}={v:.2e}")
            print(f"    grads: {', '.join(parts)}")
            if dead:
                print(f"    !! DEAD GRADIENTS: {', '.join(dead)}")

    print()


def log_band_energy(model, dataset):
    """Run one batch with diagnostics to capture band energy in/out."""
    model.eval()
    x, y = dataset.get_batch("val")
    with torch.no_grad():
        _, _, diagnostics = model(x, y, return_diagnostics=True)

    print(f"  Band energy (input -> output, averaged over batch):")
    for i, (in_e, out_e) in enumerate(diagnostics):
        if in_e is None or out_e is None:
            continue
        # Show quartile summary instead of all 64 bands
        ratio = out_e / (in_e + 1e-10)
        print(f"    L{i}: in_energy  mean={in_e.mean():.4f} std={in_e.std():.4f}"
              f" [{in_e.min():.4f}, {in_e.max():.4f}]")
        print(f"         out_energy mean={out_e.mean():.4f} std={out_e.std():.4f}"
              f" [{out_e.min():.4f}, {out_e.max():.4f}]")
        print(f"         ratio      mean={ratio.mean():.4f} std={ratio.std():.4f}"
              f" [{ratio.min():.4f}, {ratio.max():.4f}]")

        # Flag bands with extreme amplification or attenuation
        amplified = (ratio > 2.0).sum().item()
        attenuated = (ratio < 0.5).sum().item()
        if amplified > 0 or attenuated > 0:
            print(f"         {amplified} bands >2x amplified, {attenuated} bands <0.5x attenuated")

    model.train()
    print()


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


def train_mode(mode_name, ffn_type, n_steps, dataset):
    print(f"\n{'=' * 70}")
    print(f"  Training: {mode_name.upper()}")
    if ffn_type == "mlp":
        print(f"  FFN: standard MLP (128->512->128, ~131K params/layer)")
    elif ffn_type == "kerr_scalar":
        print(f"  FFN: Kerr-ODE SCALAR alpha/beta ({n_steps} steps)")
    elif ffn_type == "kerr_perband":
        print(f"  FFN: Kerr-ODE PER-BAND alpha_k/beta_k ({n_steps} steps)")
    print(f"{'=' * 70}")

    model = GPT(dataset.vocab_size, ffn_type=ffn_type, n_steps=n_steps).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ffn_params = sum(
        p.numel() for name, p in model.named_parameters()
        if "ffn." in name
    )

    history = []
    start = time.time()

    for iter_num in range(MAX_ITERS):
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model, dataset)
            elapsed = time.time() - start
            print(f"  step {iter_num:>5} | train {losses['train']:.4f}"
                  f" | val {losses['val']:.4f} | {elapsed:.1f}s")

            # Diagnostics for Kerr modes
            if ffn_type.startswith("kerr"):
                log_diagnostics(model, iter_num)
                if ffn_type == "kerr_perband":
                    log_band_energy(model, dataset)

            history.append((iter_num, losses["train"], losses["val"]))

            if math.isnan(losses["train"]):
                print("  !!! NaN detected -- stopping.")
                break

        x, y = dataset.get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()

        # Gradient check at first iteration
        if iter_num == 0 and ffn_type.startswith("kerr"):
            print("  === GRADIENT CHECK (iter 0) ===")
            for i, block in enumerate(model.blocks):
                if isinstance(block.ffn, KerrODEPerBand):
                    g = block.ffn.get_grad_norms()
                    parts = [f"{k}={v:.2e}" for k, v in g.items() if not k.endswith('_DEAD')]
                    print(f"    L{i}: {', '.join(parts)}")
            print()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    total = time.time() - start
    print(f"  Training complete in {total:.1f}s")

    # Final per-band report for perband mode
    if ffn_type == "kerr_perband":
        print(f"\n  === FINAL PER-BAND PARAMETER ANALYSIS ===")
        for i, block in enumerate(model.blocks):
            if isinstance(block.ffn, KerrODEPerBand):
                s = block.ffn.get_param_summary()
                print(f"\n  Layer {i}:")
                print(f"    alpha: mean={s['alpha_mean']:.4f} std={s['alpha_std']:.4f}"
                      f" [{s['alpha_min']:.4f}, {s['alpha_max']:.4f}]")
                print(f"    beta:  mean={s['beta_mean']:.4f} std={s['beta_std']:.4f}"
                      f" [{s['beta_min']:.4f}, {s['beta_max']:.4f}]")
                print(f"    gamma: mean={s['gamma_mean']:.4f} std={s['gamma_std']:.4f}"
                      f" [{s['gamma_min']:.4f}, {s['gamma_max']:.4f}]")
                print(f"    omega: mean={s['omega_mean']:.4f} std={s['omega_std']:.4f}"
                      f" [{s['omega_min']:.4f}, {s['omega_max']:.4f}]")

                # Success criterion: did bands differentiate?
                if s['alpha_std'] > 0.02:
                    print(f"    -> alpha DIFFERENTIATED (std {s['alpha_std']:.4f} > 0.02)")
                else:
                    print(f"    -> alpha clustered (std {s['alpha_std']:.4f} <= 0.02)")
                if s['beta_std'] > 0.02:
                    print(f"    -> beta DIFFERENTIATED (std {s['beta_std']:.4f} > 0.02)")
                else:
                    print(f"    -> beta clustered (std {s['beta_std']:.4f} <= 0.02)")

    return {
        "mode_name": mode_name,
        "ffn_type": ffn_type,
        "history": history,
        "n_params": n_params,
        "ffn_params": ffn_params,
        "n_steps": n_steps,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  Phase 21b: Per-Band Kerr Coefficients with Diagnostics")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text)} characters, {dataset.vocab_size} unique")
    print(f"  Train: {len(dataset.train_data)} | Val: {len(dataset.val_data)}")
    print(f"  Model: {N_LAYER} layers, {N_HEAD} heads, {N_EMBD} dim, {N_BANDS} bands")
    print(f"  Framework: PyTorch {torch.__version__}")

    print(f"\n  Phase 21 results (scalar alpha/beta):")
    print(f"    Standard MLP:   val ~1.71  (131K FFN params/layer)")
    print(f"    Kerr-ODE 4s:    val ~1.86  (-8.5% vs MLP)")
    print(f"    Kerr-ODE 8s:    val ~1.84  (-7.7% vs MLP)")

    print(f"\n  Phase 21b hypothesis:")
    print(f"    Per-band alpha_k/beta_k lets each band control its own Kerr strength")
    print(f"    Extra params: 128/layer (negligible). Question: does the model use them?")
    print(f"    Success threshold: alpha/beta std > 0.02 across bands by iter 2000")

    # Train three modes
    result_std = train_mode("frozen_standard", "mlp", 4, dataset)
    result_scalar = train_mode("kerr_scalar", "kerr_scalar", 4, dataset)
    result_perband = train_mode("kerr_perband_4s", "kerr_perband", 4, dataset)
    result_deep = train_mode("kerr_perband_8s", "kerr_perband", 8, dataset)

    results = [result_std, result_scalar, result_perband, result_deep]

    # =========================================================================
    # Comparison
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: Final Validation Loss")
    print(f"{'=' * 70}")
    print()
    print(f"  {'Mode':<22} {'Val':>8} {'Train':>8} {'vs MLP':>9}"
          f" {'Total':>7} {'FFN':>7} {'Steps':>5}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*9} {'-'*7} {'-'*7} {'-'*5}")

    std_val = result_std["history"][-1][2]

    for r in results:
        _, train_l, val_l = r["history"][-1]
        if r["mode_name"] == "frozen_standard":
            diff = "baseline"
        else:
            pct = (1.0 - val_l / std_val) * 100.0
            diff = f"{pct:+.2f}%"
        steps_str = str(r["n_steps"]) if r["ffn_type"].startswith("kerr") else "-"
        print(f"  {r['mode_name']:<22} {val_l:>8.4f} {train_l:>8.4f} {diff:>9}"
              f" {r['n_params']//1000:>6}K {r['ffn_params']//1000:>6}K {steps_str:>5}")

    # =========================================================================
    # Convergence
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  CONVERGENCE: Val Loss at Each Checkpoint")
    print(f"{'=' * 70}")
    print()

    min_len = min(len(r["history"]) for r in results)

    header = f"  {'Step':>6}"
    for r in results:
        name = r["mode_name"][:16]
        header += f"  {name:>16}"
    print(header)
    print(f"  {'-' * (6 + 4*18)}")

    for i in range(min_len):
        step = result_std["history"][i][0]
        vals = []
        for r in results:
            v = r["history"][i][2] if i < len(r["history"]) else float('nan')
            vals.append(v)
        line = f"  {step:>6}"
        for v in vals:
            line += f"  {v:>16.4f}"
        print(line)

    # =========================================================================
    # Key result: did per-band alpha/beta help?
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  KEY RESULT: Per-Band vs Scalar Kerr")
    print(f"{'=' * 70}")
    print()

    scalar_val = result_scalar["history"][-1][2]
    perband_val = result_perband["history"][-1][2]
    deep_val = result_deep["history"][-1][2]

    scalar_gap = (scalar_val / std_val - 1.0) * 100
    perband_gap = (perband_val / std_val - 1.0) * 100
    deep_gap = (deep_val / std_val - 1.0) * 100

    print(f"  MLP baseline:       val {std_val:.4f}")
    print(f"  Kerr scalar:        val {scalar_val:.4f}  ({scalar_gap:+.2f}% vs MLP)")
    print(f"  Kerr per-band 4s:   val {perband_val:.4f}  ({perband_gap:+.2f}% vs MLP)")
    print(f"  Kerr per-band 8s:   val {deep_val:.4f}  ({deep_gap:+.2f}% vs MLP)")
    print()

    improvement = scalar_gap - perband_gap
    if improvement > 0.5:
        print(f"  Per-band alpha/beta IMPROVED over scalar by {improvement:.2f}pp")
        print(f"  Remaining gap to MLP: {perband_gap:.2f}%")
    elif improvement > -0.5:
        print(f"  Per-band alpha/beta had NEGLIGIBLE effect ({improvement:+.2f}pp)")
        print(f"  The 8% gap is NOT about per-band nonlinear expressiveness")
    else:
        print(f"  Per-band alpha/beta HURT by {-improvement:.2f}pp")
        print(f"  Extra degrees of freedom confused the optimizer")

    deep_imp = perband_gap - deep_gap
    if abs(deep_imp) > 0.5:
        better = "8-step" if deep_imp > 0 else "4-step"
        print(f"  Integration depth: {better} is {abs(deep_imp):.2f}pp better")
    else:
        print(f"  Integration depth: minimal effect ({deep_imp:+.2f}pp)")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
