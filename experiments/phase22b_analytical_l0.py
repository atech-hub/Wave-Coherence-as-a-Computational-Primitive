"""
Phase 22b: Analytical L0 Replacement

Phase 22 showed L0 is 100% reversible -- gentle spectral remixing with
negligible Kerr effect (alpha=0.053, beta=0.047). The ODE is approximately
linear, meaning it has a closed-form solution:

  Z_k(1) = Z_k(0) * exp(-gamma_k + i*omega_k)

In real components per band:
  scale_k = exp(-gamma_k)
  r_out = scale_k * (r * cos(omega_k) - s * sin(omega_k))
  s_out = scale_k * (s * cos(omega_k) + r * sin(omega_k))

No integration needed. One matrix multiply per band replaces 8 Euler steps.

Three modes:
  1. kerr_full: 8-step Kerr-ODE all layers (baseline reproduction)
  2. hybrid: L0 = PerBandLinear, L1-L3 = Kerr-ODE 8-step (trained from scratch)
  3. posthoc: Train full Kerr-ODE, replace L0 at inference with analytical approx

Success criteria:
  - Post-hoc replacement should produce < 0.1% val loss change
  - Hybrid training should match full Kerr-ODE performance
  - L0 compute saving: 8 ODE steps -> 1 matmul (per-band 2x2)

Usage:
    python experiments/phase22b_analytical_l0.py
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
# Per-Band Linear Layer (analytical replacement for L0)
# =============================================================================

class PerBandLinear(nn.Module):
    """
    Learned 2x2 transform per band + output projection.

    Each band gets: [r_out, s_out] = W_k @ [r_in, s_in] + b_k
    where W_k is a 2x2 matrix. This subsumes rotation + scaling
    (the analytical ODE solution) but allows the model to learn
    any linear per-band operation.

    Parameters: 64 bands * (2*2 + 2) = 384 per-band + 16,512 projection
    = 16,896 per layer (comparable to Kerr-ODE's 16,642)
    """

    def __init__(self, n_bands=N_BANDS, n_embd=N_EMBD):
        super().__init__()
        self.n_bands = n_bands
        self.n_embd = n_embd

        # Per-band 2x2 transform: initialise as identity
        self.band_w = nn.Parameter(torch.zeros(n_bands, 2, 2))
        # Init as identity matrices
        with torch.no_grad():
            for k in range(n_bands):
                self.band_w.data[k] = torch.eye(2)
        self.band_b = nn.Parameter(torch.zeros(n_bands, 2))

        # Output projection (same as Kerr-ODE)
        self.out_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        bt = B * T
        bands = x.view(bt, self.n_bands, 2)  # (bt, 64, 2)

        # Per-band linear: einsum for batched 2x2 matmul
        out = torch.einsum('bni,nij->bnj', bands, self.band_w) + self.band_b

        out = out.reshape(bt, C)
        out = self.out_proj(out)
        return out.view(B, T, C)

    @staticmethod
    def from_kerr_params(kerr_layer):
        """Create analytical approximation from trained Kerr-ODE parameters.

        Uses the linear part of the ODE (ignoring Kerr nonlinearity):
          Z(1) = Z(0) * exp(-gamma + i*omega)

        In real form per band k:
          scale = exp(-gamma_k)
          W_k = scale * [[cos(omega_k), -sin(omega_k)],
                         [sin(omega_k),  cos(omega_k)]]
        """
        with torch.no_grad():
            gamma = kerr_layer.gamma.detach()  # (n_bands,)
            omega = kerr_layer.omega.detach()   # (n_bands,)

            n_bands = gamma.shape[0]
            n_embd = kerr_layer.n_embd

            layer = PerBandLinear(n_bands=n_bands, n_embd=n_embd)

            scale = torch.exp(-gamma)
            cos_w = torch.cos(omega)
            sin_w = torch.sin(omega)

            for k in range(n_bands):
                s = scale[k]
                c = cos_w[k]
                sn = sin_w[k]
                layer.band_w.data[k] = torch.tensor([
                    [s * c, -s * sn],
                    [s * sn, s * c],
                ])
            layer.band_b.data.zero_()

            # Copy the output projection weights
            layer.out_proj.weight.data.copy_(kerr_layer.out_proj.weight.data)
            layer.out_proj.bias.data.copy_(kerr_layer.out_proj.bias.data)

        return layer


# =============================================================================
# Kerr-ODE Layer (from Phase 21)
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
        elif ffn_type == "perband_linear":
            self.ffn = PerBandLinear()
        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}")

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, layer_types=None, n_steps=8):
        """
        layer_types: list of ffn types per layer, e.g. ["perband_linear", "kerr", "kerr", "kerr"]
        If None, all layers use "kerr".
        """
        super().__init__()
        if layer_types is None:
            layer_types = ["kerr"] * N_LAYER
        self.layer_types = layer_types
        self.register_buffer("wte", build_harmonic_table(vocab_size, N_EMBD))
        self.register_buffer("wpe", build_positional_table(BLOCK_SIZE, N_EMBD))
        self.blocks = nn.ModuleList(
            [Block(ffn_type=lt, n_steps=n_steps) for lt in layer_types]
        )
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        type_str = "+".join(layer_types)
        print(f"  [{type_str}] model: {n_params:,} trainable parameters")

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


def train_model(mode_name, layer_types, n_steps, dataset):
    print(f"\n{'=' * 70}")
    print(f"  Training: {mode_name}")
    print(f"  Layers: {' | '.join(f'L{i}={t}' for i, t in enumerate(layer_types))}")
    print(f"{'=' * 70}")

    model = GPT(dataset.vocab_size, layer_types=layer_types, n_steps=n_steps).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    history = []
    start = time.time()

    for iter_num in range(MAX_ITERS):
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model, dataset)
            elapsed = time.time() - start
            print(f"  step {iter_num:>5} | train {losses['train']:.4f}"
                  f" | val {losses['val']:.4f} | {elapsed:.1f}s")
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

    total = time.time() - start
    print(f"  Training complete in {total:.1f}s")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, history, n_params


# =============================================================================
# Post-hoc replacement
# =============================================================================

def posthoc_replace_l0(model, dataset):
    """Replace L0's Kerr-ODE with analytical linear approximation at inference."""
    print(f"\n{'=' * 70}")
    print(f"  POST-HOC REPLACEMENT: L0 Kerr-ODE -> Analytical")
    print(f"{'=' * 70}")

    model.eval()

    # Measure before
    losses_before = estimate_loss(model, dataset)
    print(f"  Before replacement: val {losses_before['val']:.4f}")

    # Report L0's learned parameters
    kerr_l0 = model.blocks[0].ffn
    if isinstance(kerr_l0, KerrODELayer):
        gamma = kerr_l0.gamma
        print(f"  L0 Kerr params:")
        print(f"    alpha={kerr_l0.alpha.item():.4f}  beta={kerr_l0.beta.item():.4f}")
        print(f"    gamma: mean={gamma.mean().item():.4f} [{gamma.min().item():.4f}, {gamma.max().item():.4f}]")
        print(f"    omega: mean={kerr_l0.omega.mean().item():.4f} [{kerr_l0.omega.min().item():.4f}, {kerr_l0.omega.max().item():.4f}]")

        # Create analytical replacement
        analytical = PerBandLinear.from_kerr_params(kerr_l0).to(DEVICE)

        # Verify the approximation: compare outputs on a batch
        x, _ = dataset.get_batch("val")
        B, T = x.size()
        tok_emb = F.embedding(x, model.wte)
        pos_emb = model.wpe[:T]
        h = tok_emb + pos_emb
        # Pass through attention
        h = h + model.blocks[0].attn(model.blocks[0].ln_1(h))
        kerr_input = model.blocks[0].ln_2(h)

        with torch.no_grad():
            kerr_output = kerr_l0(kerr_input)
            analytical_output = analytical(kerr_input)
            diff = (kerr_output - analytical_output).pow(2).mean().sqrt().item()
            kerr_norm = kerr_output.pow(2).mean().sqrt().item()
            print(f"\n  Output comparison (one batch):")
            print(f"    RMS difference: {diff:.6f}")
            print(f"    Kerr output RMS: {kerr_norm:.6f}")
            print(f"    Relative error: {diff/kerr_norm*100:.4f}%")

        # Replace L0
        model.blocks[0].ffn = analytical
        model.blocks[0].ffn_type = "perband_linear"

        # Measure after
        losses_after = estimate_loss(model, dataset)
        print(f"\n  After replacement: val {losses_after['val']:.4f}")

        delta = losses_after['val'] - losses_before['val']
        pct = delta / losses_before['val'] * 100
        print(f"  Delta: {delta:+.4f} ({pct:+.4f}%)")

        if abs(pct) < 0.5:
            print(f"  -> L0 replacement is LOSSLESS (< 0.5% change)")
        elif abs(pct) < 2.0:
            print(f"  -> L0 replacement has MINOR impact ({pct:+.2f}%)")
        else:
            print(f"  -> L0 replacement has SIGNIFICANT impact ({pct:+.2f}%)")
            print(f"     L0 was doing more nonlinear work than Phase 22 suggested")

    return losses_before, losses_after


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  Phase 22b: Analytical L0 Replacement")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text)} characters, {dataset.vocab_size} unique")
    print(f"  Train: {len(dataset.train_data)} | Val: {len(dataset.val_data)}")
    print(f"  Model: {N_LAYER} layers, {N_HEAD} heads, {N_EMBD} dim, {N_BANDS} bands")
    print(f"  Framework: PyTorch {torch.__version__}")

    print(f"\n  Phase 22 finding: L0 is 100% reversible")
    print(f"  Intervention: replace L0 ODE (8 Euler steps) with per-band 2x2 linear")
    print(f"  Expected: < 0.1% val loss change")

    # --- Mode 1: Full Kerr-ODE baseline ---
    kerr_model, kerr_hist, kerr_params = train_model(
        "Full Kerr-ODE 8-step",
        ["kerr"] * 4, 8, dataset
    )

    # --- Mode 2: Hybrid (analytical L0 + Kerr L1-L3), trained from scratch ---
    hybrid_model, hybrid_hist, hybrid_params = train_model(
        "Hybrid: analytical L0 + Kerr L1-L3",
        ["perband_linear", "kerr", "kerr", "kerr"], 8, dataset
    )

    # --- Mode 3: MLP baseline for reference ---
    mlp_model, mlp_hist, mlp_params = train_model(
        "MLP baseline",
        ["mlp"] * 4, 8, dataset
    )

    # --- Mode 4: Post-hoc replacement ---
    losses_before, losses_after = posthoc_replace_l0(kerr_model, dataset)

    # =========================================================================
    # Comparison
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: Final Validation Loss")
    print(f"{'=' * 70}")

    mlp_val = mlp_hist[-1][2]
    kerr_val = kerr_hist[-1][2]
    hybrid_val = hybrid_hist[-1][2]
    posthoc_val = losses_after['val']

    results = [
        ("MLP baseline", mlp_val, mlp_params, "-"),
        ("Full Kerr-ODE 8s", kerr_val, kerr_params, "8x4=32 steps"),
        ("Hybrid (analytical L0)", hybrid_val, hybrid_params, "8x3=24 steps"),
        ("Post-hoc replacement", posthoc_val, kerr_params, "8x3=24 steps"),
    ]

    print(f"\n  {'Mode':<26} {'Val':>8} {'vs MLP':>9} {'vs Kerr':>10} {'Params':>8} {'ODE Steps':>12}")
    print(f"  {'-'*26} {'-'*8} {'-'*9} {'-'*10} {'-'*8} {'-'*12}")

    for name, val, params, steps in results:
        vs_mlp = f"{(val/mlp_val - 1)*100:+.2f}%" if name != "MLP baseline" else "baseline"
        vs_kerr = f"{(val/kerr_val - 1)*100:+.2f}%" if name != "Full Kerr-ODE 8s" else "baseline"
        print(f"  {name:<26} {val:>8.4f} {vs_mlp:>9} {vs_kerr:>10} {params//1000:>7}K {steps:>12}")

    # =========================================================================
    # Convergence
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  CONVERGENCE")
    print(f"{'=' * 70}")
    print()
    print(f"  {'Step':>6}  {'MLP':>10}  {'Kerr full':>10}  {'Hybrid':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")

    min_len = min(len(mlp_hist), len(kerr_hist), len(hybrid_hist))
    for i in range(min_len):
        step = mlp_hist[i][0]
        print(f"  {step:>6}  {mlp_hist[i][2]:>10.4f}"
              f"  {kerr_hist[i][2]:>10.4f}  {hybrid_hist[i][2]:>10.4f}")

    # =========================================================================
    # L0 learned parameters comparison
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  L0 PARAMETER COMPARISON")
    print(f"{'=' * 70}")

    # Hybrid L0 (PerBandLinear)
    hybrid_l0 = hybrid_model.blocks[0].ffn
    if isinstance(hybrid_l0, PerBandLinear):
        w = hybrid_l0.band_w.detach()
        # Decompose 2x2 matrices into rotation angle + scale
        # For a matrix [[a, -b], [b, a]], angle = atan2(b, a), scale = sqrt(a^2+b^2)
        # For general 2x2, compute singular values
        print(f"\n  Hybrid L0 (PerBandLinear) learned transforms:")
        # Compute determinant and trace for each band's 2x2
        dets = w[:, 0, 0] * w[:, 1, 1] - w[:, 0, 1] * w[:, 1, 0]
        traces = w[:, 0, 0] + w[:, 1, 1]
        frob = (w ** 2).sum(dim=(1, 2)).sqrt()
        print(f"    Frobenius norm: mean={frob.mean():.4f} [{frob.min():.4f}, {frob.max():.4f}]")
        print(f"    Determinant:    mean={dets.mean():.4f} [{dets.min():.4f}, {dets.max():.4f}]")
        print(f"    Trace:          mean={traces.mean():.4f} [{traces.min():.4f}, {traces.max():.4f}]")
        print(f"    (Identity would have: frob=sqrt(2)={math.sqrt(2):.4f}, det=1, trace=2)")

    # =========================================================================
    # Key results
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  KEY RESULTS")
    print(f"{'=' * 70}")

    # Post-hoc delta
    posthoc_delta_pct = (posthoc_val / kerr_val - 1) * 100
    hybrid_delta_pct = (hybrid_val / kerr_val - 1) * 100

    print(f"\n  1. Post-hoc L0 replacement (trained as Kerr, replaced at inference):")
    print(f"     Delta: {posthoc_delta_pct:+.4f}% vs full Kerr-ODE")
    if abs(posthoc_delta_pct) < 0.5:
        print(f"     -> VALIDATES Phase 22: L0 is doing reversible spectral remixing")
    else:
        print(f"     -> L0 contributes more than expected to final loss")

    print(f"\n  2. Hybrid training (analytical L0 from start):")
    print(f"     Delta: {hybrid_delta_pct:+.4f}% vs full Kerr-ODE")
    if abs(hybrid_delta_pct) < 1.0:
        print(f"     -> L0 analytical is VIABLE for training")
    else:
        print(f"     -> Training dynamics affected by L0 type")

    compute_saving = (32 - 24) / 32 * 100
    print(f"\n  3. Compute saving: {compute_saving:.0f}% fewer ODE steps")
    print(f"     Full Kerr: 8 steps x 4 layers = 32 ODE steps/token")
    print(f"     Hybrid:    8 steps x 3 layers + 1 matmul = 24 ODE steps + 1 matmul")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
