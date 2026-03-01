"""
Phase 22d: RK4 Integration -- Does Integration Quality Close the Gap?

Phase 22c found:
  - [-50,50] clamp improves 1.61% over [-10,10]
  - Unclamped Euler hurts (-0.51%) with peak magnitudes reaching 178 million
  - The 178M peaks are Euler transient artifacts, not real dynamics

Euler is first-order: error O(dt) globally. RK4 is fourth-order: error O(dt^4).
RK4 requires 4 derivative evaluations per step (4x compute per step), but
should produce stable dynamics without needing clamps to suppress integration
artifacts.

Four modes (all trained from scratch):
  1. MLP baseline
  2. Kerr Euler 8-step [-50,50]  -- Phase 22c best, reproduce
  3. Kerr RK4 8-step [-50,50]    -- better integration, same clamp
  4. Kerr RK4 8-step unclamped   -- does RK4 tame the transient spikes?

Key question: if the 178M spikes are Euler artifacts, RK4 should remove them
and allow unclamped operation. If the spikes persist under RK4, they're real
dynamics, not integration error.

Usage:
    python experiments/phase22d_rk4_integration.py
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
# Kerr-ODE Layer with Euler or RK4 integration
# =============================================================================

class KerrODELayer(nn.Module):
    def __init__(self, n_bands=N_BANDS, n_embd=N_EMBD, n_steps=8,
                 clamp_bound=50.0, integrator="euler"):
        super().__init__()
        self.n_bands = n_bands
        self.n_embd = n_embd
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps
        self.clamp_bound = clamp_bound
        self.integrator = integrator

        self._gamma_raw = nn.Parameter(torch.full((n_bands,), math.log(math.exp(0.1) - 1)))
        omega_init = torch.arange(1, n_bands + 1, dtype=torch.float32) / n_bands
        self.omega = nn.Parameter(omega_init)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('neighbor_kernel',
                             torch.tensor([[[1.0, 1.0, 0.0, 1.0, 1.0]]]))

        # Dynamic range tracking
        self.register_buffer('max_magnitude', torch.zeros(n_bands))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))

    @property
    def gamma(self):
        return F.softplus(self._gamma_raw)

    def _derivative(self, r, s, gamma):
        """Compute dr/dt and ds/dt for the Kerr-ODE system."""
        mag_sq = r * r + s * s
        neighbor_sum = F.conv1d(
            mag_sq.unsqueeze(1), self.neighbor_kernel, padding=2
        ).squeeze(1)
        phi = self.omega + self.alpha * mag_sq + self.beta * neighbor_sum
        dr_dt = -gamma * r - phi * s
        ds_dt = -gamma * s + phi * r
        return dr_dt, ds_dt

    def _euler_step(self, r, s, dt, gamma):
        """Single Euler integration step."""
        dr_dt, ds_dt = self._derivative(r, s, gamma)
        return r + dt * dr_dt, s + dt * ds_dt

    def _rk4_step(self, r, s, dt, gamma):
        """Single RK4 integration step."""
        # k1
        dr1, ds1 = self._derivative(r, s, gamma)
        # k2
        r2 = r + 0.5 * dt * dr1
        s2 = s + 0.5 * dt * ds1
        dr2, ds2 = self._derivative(r2, s2, gamma)
        # k3
        r3 = r + 0.5 * dt * dr2
        s3 = s + 0.5 * dt * ds2
        dr3, ds3 = self._derivative(r3, s3, gamma)
        # k4
        r4 = r + dt * dr3
        s4 = s + dt * ds3
        dr4, ds4 = self._derivative(r4, s4, gamma)
        # Combine
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
        cb = self.clamp_bound
        step_fn = self._rk4_step if self.integrator == "rk4" else self._euler_step

        for _ in range(self.n_steps):
            r, s = step_fn(r, s, dt, gamma)

            # Track dynamic range before clamping
            if self.training:
                with torch.no_grad():
                    mag = torch.sqrt(r * r + s * s)
                    batch_max = mag.max(dim=0).values
                    self.max_magnitude = torch.max(self.max_magnitude, batch_max)
                    self.step_count += 1

            r = torch.clamp(r, -cb, cb)
            s = torch.clamp(s, -cb, cb)

        out = torch.stack([r, s], dim=2).reshape(bt, C)
        out = self.out_proj(out)
        return out.view(B, T, C)

    def get_range_stats(self):
        return {
            'max_mag_mean': self.max_magnitude.mean().item(),
            'max_mag_max': self.max_magnitude.max().item(),
            'max_mag_min': self.max_magnitude.min().item(),
            'bands_above_10': (self.max_magnitude > 10.0).sum().item(),
            'bands_above_50': (self.max_magnitude > 50.0).sum().item(),
            'bands_above_100': (self.max_magnitude > 100.0).sum().item(),
            'bands_above_1000': (self.max_magnitude > 1000.0).sum().item(),
        }

    def reset_range_stats(self):
        self.max_magnitude.zero_()
        self.step_count.zero_()


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
    def __init__(self, ffn_type="mlp", n_steps=8, clamp_bound=50.0,
                 integrator="euler"):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.ffn_type = ffn_type
        if ffn_type == "mlp":
            self.ffn = MLP()
        elif ffn_type == "kerr":
            self.ffn = KerrODELayer(n_steps=n_steps, clamp_bound=clamp_bound,
                                    integrator=integrator)
        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}")

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, ffn_type="mlp", n_steps=8,
                 clamp_bound=50.0, integrator="euler"):
        super().__init__()
        self.ffn_type = ffn_type
        self.register_buffer("wte", build_harmonic_table(vocab_size, N_EMBD))
        self.register_buffer("wpe", build_positional_table(BLOCK_SIZE, N_EMBD))
        self.blocks = nn.ModuleList(
            [Block(ffn_type=ffn_type, n_steps=n_steps, clamp_bound=clamp_bound,
                   integrator=integrator)
             for _ in range(N_LAYER)]
        )
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        integrator_str = f" {integrator}" if ffn_type == "kerr" else ""
        clamp_str = f" clamp={clamp_bound}" if ffn_type == "kerr" else ""
        print(f"  {ffn_type}{integrator_str}{clamp_str} model:"
              f" {n_params:,} trainable parameters")

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


def train_mode(mode_name, ffn_type, n_steps, clamp_bound, integrator, dataset):
    print(f"\n{'=' * 70}")
    print(f"  Training: {mode_name}")
    if ffn_type == "kerr":
        print(f"  Integrator: {integrator}, Steps: {n_steps},"
              f" Clamp: [-{clamp_bound}, {clamp_bound}]")
        deriv_evals = n_steps * (4 if integrator == "rk4" else 1)
        print(f"  Derivative evaluations per forward: {deriv_evals}")
    print(f"{'=' * 70}")

    model = GPT(dataset.vocab_size, ffn_type=ffn_type, n_steps=n_steps,
                clamp_bound=clamp_bound, integrator=integrator).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    history = []
    start = time.time()

    for iter_num in range(MAX_ITERS):
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model, dataset)
            elapsed = time.time() - start
            print(f"  step {iter_num:>5} | train {losses['train']:.4f}"
                  f" | val {losses['val']:.4f} | {elapsed:.1f}s")

            if ffn_type == "kerr":
                for i, block in enumerate(model.blocks):
                    if isinstance(block.ffn, KerrODELayer):
                        s = block.ffn.get_range_stats()
                        gamma = block.ffn.gamma
                        print(f"    L{i}: max_mag=[{s['max_mag_min']:.1f},"
                              f"{s['max_mag_max']:.1f}]"
                              f" avg={s['max_mag_mean']:.1f}"
                              f"  >10:{s['bands_above_10']}/64"
                              f"  >50:{s['bands_above_50']}/64"
                              f"  >1K:{s['bands_above_1000']}/64"
                              f"  alpha={block.ffn.alpha.item():.4f}"
                              f"  gamma=[{gamma.min().item():.3f},"
                              f"{gamma.max().item():.3f}]")

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

    # Final dynamic range
    if ffn_type == "kerr":
        print(f"\n  Final dynamic range:")
        for i, block in enumerate(model.blocks):
            if isinstance(block.ffn, KerrODELayer):
                s = block.ffn.get_range_stats()
                print(f"    L{i}: peak magnitude [{s['max_mag_min']:.2f},"
                      f" {s['max_mag_max']:.2f}]"
                      f" mean={s['max_mag_mean']:.2f}"
                      f"  >50:{s['bands_above_50']}/64"
                      f"  >1K:{s['bands_above_1000']}/64")

    return {
        "mode_name": mode_name,
        "ffn_type": ffn_type,
        "integrator": integrator,
        "n_steps": n_steps,
        "clamp_bound": clamp_bound,
        "history": history,
        "n_params": n_params,
        "model": model,
        "time": total,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  Phase 22d: RK4 Integration -- Does Integration Quality Matter?")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text)} characters, {dataset.vocab_size} unique")
    print(f"  Train: {len(dataset.train_data)} | Val: {len(dataset.val_data)}")
    print(f"  Model: {N_LAYER} layers, {N_HEAD} heads, {N_EMBD} dim, {N_BANDS} bands")
    print(f"  Framework: PyTorch {torch.__version__}")

    print(f"\n  Phase 22c finding: unclamped Euler spikes to 178M magnitude")
    print(f"  Hypothesis: the spikes are Euler artifacts, not real dynamics")
    print(f"  If true, RK4 should stabilize without clamps")

    # Train all modes
    result_mlp = train_mode(
        "MLP baseline", "mlp", 8, 50.0, "euler", dataset)
    result_euler = train_mode(
        "Kerr Euler 8s [-50]", "kerr", 8, 50.0, "euler", dataset)
    result_rk4_clamped = train_mode(
        "Kerr RK4 8s [-50]", "kerr", 8, 50.0, "rk4", dataset)
    result_rk4_unclamped = train_mode(
        "Kerr RK4 8s unclamped", "kerr", 8, 1000.0, "rk4", dataset)

    results = [result_mlp, result_euler, result_rk4_clamped, result_rk4_unclamped]

    # =========================================================================
    # Comparison
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: Final Validation Loss")
    print(f"{'=' * 70}")

    mlp_val = result_mlp["history"][-1][2]
    euler_val = result_euler["history"][-1][2]

    print(f"\n  {'Mode':<26} {'Val':>8} {'Train':>8} {'vs MLP':>9}"
          f" {'vs Euler':>9} {'Time':>7}")
    print(f"  {'-'*26} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*7}")

    for r in results:
        if not r["history"]:
            continue
        _, train_l, val_l = r["history"][-1]
        if r["mode_name"] == "MLP baseline":
            vs_mlp = "baseline"
            vs_euler = "-"
        elif r["mode_name"] == "Kerr Euler 8s [-50]":
            vs_mlp = f"{(val_l/mlp_val - 1)*100:+.2f}%"
            vs_euler = "baseline"
        else:
            vs_mlp = f"{(val_l/mlp_val - 1)*100:+.2f}%"
            vs_euler = f"{(val_l/euler_val - 1)*100:+.2f}%"
        t = f"{r['time']:.0f}s"
        print(f"  {r['mode_name']:<26} {val_l:>8.4f} {train_l:>8.4f}"
              f" {vs_mlp:>9} {vs_euler:>9} {t:>7}")

    # =========================================================================
    # Convergence
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  CONVERGENCE")
    print(f"{'=' * 70}")
    print()
    labels = ["MLP", "Euler [-50]", "RK4 [-50]", "RK4 unclamp"]
    print(f"  {'Step':>6}  " + "  ".join(f"{l:>12}" for l in labels))
    print(f"  {'-'*6}  " + "  ".join(f"{'-'*12}" for _ in labels))

    min_len = min(len(r["history"]) for r in results if r["history"])
    for i in range(min_len):
        step = result_mlp["history"][i][0]
        vals = [r["history"][i][2] if i < len(r["history"]) else float('nan')
                for r in results]
        print(f"  {step:>6}  " + "  ".join(f"{v:>12.4f}" for v in vals))

    # =========================================================================
    # Dynamic Range: Euler vs RK4
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  DYNAMIC RANGE: Euler vs RK4")
    print(f"{'=' * 70}")
    print()
    print(f"  Peak band magnitude reached during training:")
    print(f"  {'':>26} {'L0':>12} {'L1':>12} {'L2':>12} {'L3':>12}")
    print(f"  {'-'*26} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for r in [result_euler, result_rk4_clamped, result_rk4_unclamped]:
        if r["ffn_type"] != "kerr" or not r["history"]:
            continue
        layer_stats = []
        for block in r["model"].blocks:
            if isinstance(block.ffn, KerrODELayer):
                s = block.ffn.get_range_stats()
                layer_stats.append(f"{s['max_mag_max']:>12.1f}")
        print(f"  {r['mode_name']:<26} {''.join(layer_stats)}")

    # =========================================================================
    # Key Questions
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  KEY QUESTIONS")
    print(f"{'=' * 70}")

    if not all(r["history"] for r in results):
        print("\n  Some modes failed (NaN). Cannot fully answer.")
    else:
        euler_v = result_euler["history"][-1][2]
        rk4c_v = result_rk4_clamped["history"][-1][2]
        rk4u_v = result_rk4_unclamped["history"][-1][2]

        imp_rk4c = (euler_v - rk4c_v) / euler_v * 100
        imp_rk4u = (euler_v - rk4u_v) / euler_v * 100

        print(f"\n  Q1: Does RK4 improve over Euler (same clamp)?")
        print(f"      Euler [-50,50]:  {euler_v:.4f}")
        print(f"      RK4   [-50,50]:  {rk4c_v:.4f}  ({imp_rk4c:+.2f}%)")
        if imp_rk4c > 0.5:
            print(f"      YES -- RK4 provides meaningful improvement")
        elif imp_rk4c > 0:
            print(f"      Marginal -- small improvement, integration quality is minor")
        else:
            print(f"      NO -- RK4 does not improve, Euler was sufficient")

        print(f"\n  Q2: Does RK4 tame the unclamped transient spikes?")
        # Check if any layer exceeds 1000 in unclamped RK4
        max_peaks = []
        for block in result_rk4_unclamped["model"].blocks:
            if isinstance(block.ffn, KerrODELayer):
                max_peaks.append(block.ffn.get_range_stats()['max_mag_max'])
        max_peak = max(max_peaks) if max_peaks else 0

        print(f"      Phase 22c unclamped Euler peak: 178,000,000")
        print(f"      Phase 22d unclamped RK4 peak:   {max_peak:,.0f}")
        if max_peak < 1000:
            print(f"      YES -- RK4 eliminates transient spikes (peak {max_peak:.0f} vs 178M)")
            print(f"      The 178M peaks were Euler integration artifacts")
        elif max_peak < 100000:
            print(f"      PARTIALLY -- peak reduced from 178M to {max_peak:,.0f}")
            print(f"      Euler amplified them, but some instability is real")
        else:
            print(f"      NO -- peaks still large ({max_peak:,.0f})")
            print(f"      The instability is in the dynamics, not the integrator")

        print(f"\n  Q3: Does unclamped RK4 beat clamped RK4?")
        print(f"      RK4 [-50,50]:   {rk4c_v:.4f}")
        print(f"      RK4 unclamped:  {rk4u_v:.4f}  ({(rk4u_v/rk4c_v - 1)*100:+.2f}%)")
        if rk4u_v < rk4c_v:
            print(f"      YES -- unclamped is better, the clamp was constraining")
        else:
            print(f"      NO -- clamping still helps, even with RK4")

        print(f"\n  Q4: How much of the MLP gap does RK4 close?")
        gap_euler = (euler_v / mlp_val - 1) * 100
        gap_rk4 = (min(rk4c_v, rk4u_v) / mlp_val - 1) * 100
        best_rk4_name = "clamped" if rk4c_v < rk4u_v else "unclamped"
        closed = gap_euler - gap_rk4
        print(f"      MLP:          {mlp_val:.4f}")
        print(f"      Euler gap:    {gap_euler:+.2f}%")
        print(f"      RK4 gap:      {gap_rk4:+.2f}% (best: {best_rk4_name})")
        print(f"      Gap closed:   {closed:.2f} percentage points")

        # Training time comparison
        print(f"\n  Compute cost:")
        euler_t = result_euler["time"]
        rk4c_t = result_rk4_clamped["time"]
        print(f"      Euler 8s:  {euler_t:.0f}s ({8} deriv evals/step)")
        print(f"      RK4 8s:    {rk4c_t:.0f}s ({32} deriv evals/step)"
              f"  ({rk4c_t/euler_t:.1f}x)")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
