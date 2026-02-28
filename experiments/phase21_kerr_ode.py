"""
Phase 21: Kerr-ODE Layer -- Wave-Native Computation Primitive

Phase 20b showed that the LC layer's bottleneck was architectural, not parametric:
per-band nonlinear + cross-band LINEAR cannot match MLP's dense nonlinear
multi-dimensional interaction. The missing primitive is nonlinear multi-band fusion.

The Kerr-ODE layer provides exactly that. Based on the physics of coupled optical
resonators, it integrates a differential equation where:
  - Each harmonic band evolves with learned damping and resonant frequency
  - Self-phase modulation (Kerr effect): |Z_k|^2 * Z_k -- intensity-dependent phase shift
  - Cross-phase modulation: sum(|Z_neighbor|^2) * Z_k -- NONLINEAR cross-band coupling
  - Output projection: Linear(128->128) for global mixing

The Kerr nonlinearity is the key difference from Phase 20b: cross-band interaction
is now NONLINEAR (through |Z_j|^2 terms) rather than linear (matrix multiply).

Parameter budget:
  ODE params: 64 gamma + 64 omega + 1 alpha + 1 beta = 130 per layer
  Output projection: 128x128 + 128 = 16,512 per layer
  Total: 16,642 per layer vs MLP's 131,712 (7.9x reduction)

Usage:
    python experiments/phase21_kerr_ode.py
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
# Kerr-ODE Layer -- the wave-native computation primitive
# =============================================================================

class KerrODELayer(nn.Module):
    """
    Frequency-native FFN replacement using coupled nonlinear oscillator ODE.

    Input x (B, T, 128) is treated as 64 complex harmonic bands:
      Z_k = x[2k] + i*x[2k+1]

    ODE dynamics for each band k:
      dZ_k/dt = -gamma_k * Z_k                    (learned damping)
              + i * omega_k * Z_k                  (learned resonance)
              + i * alpha * |Z_k|^2 * Z_k          (Kerr self-phase modulation)
              + i * beta * sum_neighbors(|Z_j|^2) * Z_k  (cross-phase modulation)

    Expanded to real/imag (r_k, s_k):
      phi_k = omega_k + alpha * |Z_k|^2 + beta * N_k
      dr_k/dt = -gamma_k * r_k - phi_k * s_k
      ds_k/dt = -gamma_k * s_k + phi_k * r_k

    Where N_k = |Z_{k-2}|^2 + |Z_{k-1}|^2 + |Z_{k+1}|^2 + |Z_{k+2}|^2

    Integrated with fixed-step Euler. Followed by learned output projection.
    """

    def __init__(self, n_bands=N_BANDS, n_embd=N_EMBD, n_steps=4):
        super().__init__()
        self.n_bands = n_bands
        self.n_embd = n_embd
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps

        # Per-band ODE parameters
        # gamma stored as raw values, passed through softplus to enforce positivity
        # softplus(0.0) = 0.693, so init raw to -0.8 -> softplus(-0.8) ~ 0.37
        # Actually, init so that softplus(raw) ~ 0.1: inverse_softplus(0.1) = log(exp(0.1)-1) ~ -2.25
        # Simpler: init raw to match desired gamma=0.1 after softplus
        self._gamma_raw = nn.Parameter(torch.full((n_bands,), math.log(math.exp(0.1) - 1)))

        # omega: harmonic order scaled for Euler stability
        # Raw harmonic order (1..64) with dt=0.25 causes Euler blow-up at high bands
        # (growth factor sqrt(1 + (omega*dt)^2) per step).
        # Scale: omega_k = (k+1) / N_BANDS gives range [1/64, 1.0],
        # max phase/step = 0.25 rad, growth factor ~1.03/step. Stable.
        # Model can learn larger values where damping compensates.
        omega_init = torch.arange(1, n_bands + 1, dtype=torch.float32) / n_bands
        self.omega = nn.Parameter(omega_init)

        # Global nonlinearity coefficients
        self.alpha = nn.Parameter(torch.tensor(0.1))  # self-phase modulation
        self.beta = nn.Parameter(torch.tensor(0.1))    # cross-phase modulation

        # Output projection: global cross-band mixing after ODE
        self.out_proj = nn.Linear(n_embd, n_embd)

        # Fixed neighbor kernel for cross-phase modulation
        # Neighbours: k-2, k-1, k+1, k+2 (skip self)
        self.register_buffer(
            'neighbor_kernel',
            torch.tensor([[[1.0, 1.0, 0.0, 1.0, 1.0]]])
        )

        # Parameter counts
        self.ode_param_count = n_bands + n_bands + 1 + 1  # gamma + omega + alpha + beta
        self.proj_param_count = n_embd * n_embd + n_embd
        self.total_param_count = self.ode_param_count + self.proj_param_count

    @property
    def gamma(self):
        """Damping coefficient, guaranteed positive via softplus."""
        return F.softplus(self._gamma_raw)

    def forward(self, x):
        B, T, C = x.size()
        bt = B * T

        # Reshape to bands: (bt, n_bands, 2) -> separate real/imag
        bands = x.view(bt, self.n_bands, 2)
        r = bands[:, :, 0].contiguous()  # (bt, n_bands)
        s = bands[:, :, 1].contiguous()  # (bt, n_bands)

        dt = self.dt
        gamma = self.gamma  # softplus ensures positive damping

        for _ in range(self.n_steps):
            # |Z_k|^2 per band
            mag_sq = r * r + s * s  # (bt, n_bands)

            # Neighbor sum of |Z_j|^2 using conv1d
            neighbor_sum = F.conv1d(
                mag_sq.unsqueeze(1),    # (bt, 1, n_bands)
                self.neighbor_kernel,   # (1, 1, 5)
                padding=2
            ).squeeze(1)                # (bt, n_bands)

            # Total phase modulation per band
            phi = self.omega + self.alpha * mag_sq + self.beta * neighbor_sum

            # ODE: dZ/dt = (-gamma + i*phi) * Z
            dr_dt = -gamma * r - phi * s
            ds_dt = -gamma * s + phi * r

            # Euler step
            r = r + dt * dr_dt
            s = s + dt * ds_dt

            # Stability clamp: prevent magnitude blow-up in deep integration
            r = torch.clamp(r, -10.0, 10.0)
            s = torch.clamp(s, -10.0, 10.0)

        # Reassemble to (B, T, C) and apply output projection
        out = torch.stack([r, s], dim=2).reshape(bt, C)
        out = self.out_proj(out)
        return out.view(B, T, C)

    def get_param_summary(self):
        with torch.no_grad():
            gamma = self.gamma  # uses softplus property
            return {
                'alpha': self.alpha.item(),
                'beta': self.beta.item(),
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

    def get_grad_summary(self):
        result = {}
        for name, param in [('alpha', self.alpha), ('beta', self.beta),
                            ('gamma', self._gamma_raw), ('omega', self.omega)]:
            if param.grad is not None:
                result[f'{name}_grad_rms'] = param.grad.pow(2).mean().sqrt().item()
        if self.out_proj.weight.grad is not None:
            result['proj_grad_rms'] = self.out_proj.weight.grad.pow(2).mean().sqrt().item()
        return result


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
    def __init__(self, ffn_type="mlp", n_steps=4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.ffn_type = ffn_type
        if ffn_type == "mlp":
            self.ffn = MLP()
        elif ffn_type.startswith("kerr"):
            self.ffn = KerrODELayer(n_steps=n_steps)
        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}")

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
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
        # Residual scaling for output projections
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

    def report_kerr_params(self):
        if not self.ffn_type.startswith("kerr"):
            return
        for i, block in enumerate(self.blocks):
            if isinstance(block.ffn, KerrODELayer):
                s = block.ffn.get_param_summary()
                g = block.ffn.get_grad_summary()
                alpha_g = f"  a_grad={g.get('alpha_grad_rms', 0):.2e}" if g else ""
                beta_g = f"  b_grad={g.get('beta_grad_rms', 0):.2e}" if g else ""
                print(f"    L{i}: alpha={s['alpha']:.4f}  beta={s['beta']:.4f}"
                      f"  gamma=[{s['gamma_min']:.3f},{s['gamma_max']:.3f}] avg={s['gamma_mean']:.3f}"
                      f"  omega=[{s['omega_min']:.3f},{s['omega_max']:.3f}] avg={s['omega_mean']:.3f}"
                      f"{alpha_g}{beta_g}")


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
    else:
        print(f"  FFN: Kerr-ODE ({n_steps} steps, dt={1.0/n_steps:.3f})")
        print(f"       Dynamics: damping + resonance + Kerr self-phase + cross-phase")
        print(f"       + output projection Linear(128->128)")
    print(f"{'=' * 70}")

    model = GPT(dataset.vocab_size, ffn_type=ffn_type, n_steps=n_steps).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ffn_params = sum(
        p.numel() for name, p in model.named_parameters()
        if any(k in name for k in ["ffn.", "mlp."])  # handle both naming conventions
    )

    history = []
    start = time.time()

    for iter_num in range(MAX_ITERS):
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model, dataset)
            elapsed = time.time() - start
            print(f"  step {iter_num:>5} | train {losses['train']:.4f}"
                  f" | val {losses['val']:.4f} | {elapsed:.1f}s")

            # Report Kerr params at select checkpoints
            if ffn_type.startswith("kerr"):
                model.report_kerr_params()

            history.append((iter_num, losses["train"], losses["val"]))

            # Check for NaN (Euler stability)
            if math.isnan(losses["train"]):
                print("  !!! NaN detected -- Euler integration unstable. Stopping.")
                break

        x, y = dataset.get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()

        # Gradient check at first iteration
        if iter_num == 0 and ffn_type.startswith("kerr"):
            print("\n  === GRADIENT CHECK (iter 0) ===")
            for i, block in enumerate(model.blocks):
                if isinstance(block.ffn, KerrODELayer):
                    g = block.ffn.get_grad_summary()
                    parts = [f"{k}={v:.2e}" for k, v in g.items()]
                    print(f"    L{i}: {', '.join(parts)}")
            print()

        # Gradient clipping for ODE stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    total = time.time() - start
    print(f"  Training complete in {total:.1f}s")

    # Final detailed Kerr param report
    if ffn_type.startswith("kerr"):
        print(f"\n  Final Kerr-ODE parameters:")
        for i, block in enumerate(model.blocks):
            if isinstance(block.ffn, KerrODELayer):
                s = block.ffn.get_param_summary()
                print(f"    Layer {i}:")
                print(f"      gamma:  mean={s['gamma_mean']:.4f}"
                      f"  std={s['gamma_std']:.4f}"
                      f"  [{s['gamma_min']:.4f}, {s['gamma_max']:.4f}]")
                print(f"      omega:  mean={s['omega_mean']:.4f}"
                      f"  std={s['omega_std']:.4f}"
                      f"  [{s['omega_min']:.4f}, {s['omega_max']:.4f}]")
                print(f"      alpha:  {s['alpha']:.6f}")
                print(f"      beta:   {s['beta']:.6f}")
                print(f"      proj_norm: {s['proj_norm']:.4f}")

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
    print("  Phase 21: Kerr-ODE Layer -- Wave-Native Computation Primitive")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text)} characters, {dataset.vocab_size} unique")
    print(f"  Train: {len(dataset.train_data)} | Val: {len(dataset.val_data)}")
    print(f"  Model: {N_LAYER} layers, {N_HEAD} heads, {N_EMBD} dim, {N_BANDS} bands")
    print(f"  Framework: PyTorch {torch.__version__}")

    print(f"\n  Previous results (Phase 20b):")
    print(f"    Standard MLP:   val ~1.65  (131K FFN params/layer)")
    print(f"    Expanded LC:    val ~2.00  (13.4K FFN params/layer, -21.3%)")
    print(f"    LC bottleneck:  per-band nonlinear + cross-band LINEAR")

    print(f"\n  Kerr-ODE design:")
    print(f"    ODE: damping + resonance + Kerr |Z|^2 self-phase + cross-phase modulation")
    print(f"    Key difference: cross-band coupling is NONLINEAR (|Z_j|^2 terms)")
    print(f"    + output projection Linear(128->128) for global mixing")
    print(f"    ODE params: 130/layer, projection: 16,512/layer, total: 16,642/layer")
    print(f"    vs MLP: 131,712/layer (7.9x reduction)")

    # Train three modes
    result_std = train_mode("frozen_standard", "mlp", 4, dataset)
    result_kerr = train_mode("kerr_ode", "kerr_ode", 4, dataset)
    result_deep = train_mode("kerr_ode_deep", "kerr_ode", 8, dataset)

    results = [result_std, result_kerr, result_deep]

    # =========================================================================
    # Comparison
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: Final Validation Loss")
    print(f"{'=' * 70}")
    print()
    print(f"  {'Mode':<22} {'Val':>8} {'Train':>8} {'vs Std':>9}"
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

    # Find minimum common length
    min_len = min(len(r["history"]) for r in results)

    header = f"  {'Step':>6}"
    for r in results:
        name = r["mode_name"][:16]
        header += f"  {name:>16}"
    header += f"  {'kerr vs std':>12}  {'deep vs std':>12}"
    print(header)
    print(f"  {'-' * (6 + 3*18 + 2*14)}")

    for i in range(min_len):
        step = result_std["history"][i][0]
        std_v = result_std["history"][i][2]
        kerr_v = result_kerr["history"][i][2] if i < len(result_kerr["history"]) else float('nan')
        deep_v = result_deep["history"][i][2] if i < len(result_deep["history"]) else float('nan')

        kerr_pct = (1.0 - kerr_v / std_v) * 100.0 if not math.isnan(kerr_v) else float('nan')
        deep_pct = (1.0 - deep_v / std_v) * 100.0 if not math.isnan(deep_v) else float('nan')

        print(f"  {step:>6}  {std_v:>16.4f}  {kerr_v:>16.4f}  {deep_v:>16.4f}"
              f"  {kerr_pct:>+11.2f}%  {deep_pct:>+11.2f}%")

    # =========================================================================
    # Parameter efficiency
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  PARAMETER EFFICIENCY")
    print(f"{'=' * 70}")

    kerr_val = result_kerr["history"][-1][2]
    deep_val = result_deep["history"][-1][2]
    std_ffn = result_std["ffn_params"]
    kerr_ffn = result_kerr["ffn_params"]

    print(f"\n  FFN params: standard={std_ffn:,}  Kerr-ODE={kerr_ffn:,}  ({std_ffn/kerr_ffn:.1f}x reduction)")

    for name, val in [("kerr_ode", kerr_val), ("kerr_ode_deep", deep_val)]:
        loss_ratio = val / std_val
        param_ratio = std_ffn / kerr_ffn
        print(f"\n  {name}:")
        print(f"    Loss ratio (Kerr/std): {loss_ratio:.4f}")
        print(f"    FFN param ratio: {param_ratio:.1f}x")
        print(f"    Efficiency: {param_ratio / loss_ratio:.1f}x params saved per unit loss")

    # =========================================================================
    # Cross-phase comparison
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  CROSS-PHASE COMPARISON")
    print(f"{'=' * 70}")
    print()
    print(f"  Phase 20  (tiny LC):      148 FFN/layer  ->  23.5% worse than MLP")
    print(f"  Phase 20b (expanded LC):  13,440 FFN/layer  ->  21.3% worse")
    print(f"  Phase 21  (Kerr-ODE 4s):  {kerr_ffn//N_LAYER:,} FFN/layer"
          f"  ->  {(1.0-kerr_val/std_val)*100:+.2f}%")
    print(f"  Phase 21  (Kerr-ODE 8s):  {kerr_ffn//N_LAYER:,} FFN/layer"
          f"  ->  {(1.0-deep_val/std_val)*100:+.2f}%")
    print(f"  Standard MLP:             131,712 FFN/layer  (baseline)")
    print()

    # Key questions
    print(f"  KEY QUESTIONS:")
    kerr_gap = (kerr_val / std_val - 1.0) * 100
    deep_gap = (deep_val / std_val - 1.0) * 100
    lc_gap = 21.3  # from Phase 20b

    print(f"  1. Does Kerr nonlinearity beat LC's linear coupling?")
    if kerr_gap < lc_gap:
        print(f"     YES: Kerr {kerr_gap:.1f}% gap vs LC's {lc_gap:.1f}% gap"
              f" ({lc_gap - kerr_gap:.1f}pp improvement)")
    else:
        print(f"     NO: Kerr {kerr_gap:.1f}% gap vs LC's {lc_gap:.1f}% gap")

    print(f"  2. Can ~16K params match 131K MLP?")
    if kerr_gap < 5:
        print(f"     NEARLY: only {kerr_gap:.1f}% gap with 7.9x fewer FFN params")
    elif kerr_gap < 10:
        print(f"     PARTIALLY: {kerr_gap:.1f}% gap, significant improvement over LC")
    else:
        print(f"     NOT YET: {kerr_gap:.1f}% gap remains substantial")

    print(f"  3. Do parameters differentiate by band?")
    print(f"     (See per-layer parameter reports above)")

    print(f"  4. Does integration depth matter (4 vs 8 steps)?")
    if abs(deep_gap - kerr_gap) > 1.0:
        better = "8-step" if deep_gap < kerr_gap else "4-step"
        print(f"     YES: {better} is {abs(deep_gap - kerr_gap):.1f}pp better")
    else:
        print(f"     MINIMAL: {abs(deep_gap - kerr_gap):.1f}pp difference")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
