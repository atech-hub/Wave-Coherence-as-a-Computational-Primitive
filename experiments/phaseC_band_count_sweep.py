"""
Frequency Budget Curve — Experiments 3 + 5

Sweep band count (8, 16, 32, 48, 64, 96) to find the minimum bandwidth
for target performance. At each band count, run:
  1. MLP baseline (at that dimension)
  2. Two-stage Kerr-ODE (the Phase B winner)

Architecture: 4L/4H, progressive curriculum, two-stage magnitude training.
N_EMBD = N_BANDS * 2 at each point. Reference: MLP at 64 bands / 128 dim.

This answers: "What's the cost of intelligence?" — how much frequency bandwidth
buys how much capability. If 32 bands gets 90%+ of 64-band performance,
that's a halved model.
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

BAND_COUNTS = [8, 16, 32, 48, 64, 96]

N_LAYER = 4
N_HEAD = 4
BLOCK_SIZE = 256
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_ITERS = 2000
EVAL_INTERVAL = 200
EVAL_ITERS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Two-stage unfreezes magnitude at final stage
TWO_STAGE_UNFREEZE = 1334


# =============================================================================
# Progressive curriculum — adapts to band count
# =============================================================================

def get_prog_stages(n_bands):
    """Build 3-stage progressive curriculum scaled to band count."""
    if n_bands <= 8:
        # Only one stage — all bands from the start
        return [(0, MAX_ITERS, n_bands)]
    elif n_bands <= 16:
        # Two stages
        mid = n_bands // 2
        return [
            (0, 1000, mid),
            (1000, MAX_ITERS, n_bands),
        ]
    else:
        # Three stages: ~12.5%, ~37.5%, 100%
        low = max(4, n_bands // 8)
        mid = max(low + 1, n_bands * 3 // 8)
        return [
            (0, 667, low),
            (667, 1334, mid),
            (1334, MAX_ITERS, n_bands),
        ]


def get_stage(iter_num, stages):
    for start, end, bands in stages:
        if start <= iter_num < end:
            return bands
    return stages[-1][2]


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


# =============================================================================
# Per-Band Linear Layer (L0)
# =============================================================================

class PerBandLinear(nn.Module):
    def __init__(self, n_bands, n_embd):
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
    def __init__(self, n_bands, n_embd, n_steps=8):
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
        out = torch.stack([r, s], dim=2).reshape(bt, C)
        return self.out_proj(out).view(B, T, C)


# =============================================================================
# Standard MLP
# =============================================================================

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


# =============================================================================
# Attention
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
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
    def __init__(self, n_embd, n_head, ffn_module):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.ffn = ffn_module

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_bands, mode="mlp"):
        super().__init__()
        self.mode = mode
        self.vocab_size = vocab_size
        self.n_bands = n_bands
        n_embd = n_bands * 2

        self.register_buffer("wte", build_harmonic_table(vocab_size, n_embd))
        self.register_buffer("wpe", build_positional_table(BLOCK_SIZE, n_embd))

        # Trainable magnitude for two-stage
        if mode == "two_stage":
            self.tok_mag = nn.Parameter(
                torch.ones(vocab_size, n_bands), requires_grad=False)
        else:
            self.tok_mag = None

        # Build layer stack
        if mode == "mlp":
            layers = [Block(n_embd, N_HEAD, MLP(n_embd))
                      for _ in range(N_LAYER)]
        elif mode in ("two_stage", "frozen"):
            layers = [
                Block(n_embd, N_HEAD, PerBandLinear(n_bands, n_embd)),
                Block(n_embd, N_HEAD, KerrODELayer(n_bands, n_embd, n_steps=8)),
                Block(n_embd, N_HEAD, KerrODELayer(n_bands, n_embd, n_steps=8)),
                Block(n_embd, N_HEAD, KerrODELayer(n_bands, n_embd, n_steps=8)),
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.blocks = nn.ModuleList(layers)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.register_buffer("band_mask", torch.ones(n_embd))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

        # Reset tok_mag after _init_weights
        if self.tok_mag is not None:
            self.tok_mag.data.fill_(1.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def set_active_bands(self, n_active):
        n_embd = self.n_bands * 2
        mask = torch.zeros(n_embd, device=self.band_mask.device)
        for h in range(min(n_active, self.n_bands)):
            mask[h * 2] = 1.0
            mask[h * 2 + 1] = 1.0
        self.band_mask = mask

    def unfreeze_magnitude(self):
        if self.tok_mag is not None and not self.tok_mag.requires_grad:
            self.tok_mag.requires_grad_(True)
            return True
        return False

    def forward(self, idx, targets=None):
        B, T = idx.size()
        n_embd = self.n_bands * 2
        tok_emb = F.embedding(idx, self.wte)

        if self.tok_mag is not None:
            mag = self.tok_mag[idx]
            mag_exp = mag.unsqueeze(-1).expand(-1, -1, -1, 2)
            tok_emb = tok_emb * mag_exp.reshape(B, T, n_embd)

        x = (tok_emb + self.wpe[:T]) * self.band_mask
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    def get_magnitude_stats(self):
        if self.tok_mag is None:
            return None
        mag = self.tok_mag.detach().cpu()
        gm = mag.mean().item()
        gs = mag.std().item()
        return {
            "global_cv": (gs / gm * 100) if gm > 1e-10 else 0,
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


def train_variant(n_bands, mode, dataset):
    n_embd = n_bands * 2
    is_two_stage = mode == "two_stage"
    use_prog = mode != "mlp"
    stages = get_prog_stages(n_bands) if use_prog else None

    name = f"{mode}_{n_bands}b"
    print(f"\n  --- {name} ({n_embd}D) ---")
    if stages:
        for s, e, b in stages:
            print(f"    Steps {s}-{e}: bands 1-{b}")

    torch.manual_seed(42)
    model = GPT(dataset.vocab_size, n_bands, mode=mode).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_two_stage and model.tok_mag is not None:
        n_params += model.tok_mag.numel()  # will be unfrozen later
    print(f"    {n_params:,} params (final)")

    # Optimizer — exclude frozen tok_mag for two-stage
    if is_two_stage and model.tok_mag is not None:
        params = [p for n, p in model.named_parameters()
                  if p.requires_grad and n != "tok_mag"]
    else:
        params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)

    history = []
    start_time = time.time()
    current_bands = n_bands
    mag_unfrozen = False

    for iter_num in range(MAX_ITERS):
        if stages:
            stage_bands = get_stage(iter_num, stages)
            if stage_bands != current_bands:
                model.set_active_bands(stage_bands)
                current_bands = stage_bands

        if is_two_stage and not mag_unfrozen and iter_num >= TWO_STAGE_UNFREEZE:
            if model.unfreeze_magnitude():
                optimizer.add_param_group({"params": [model.tok_mag]})
                mag_unfrozen = True

        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            saved_mask = model.band_mask.clone()
            model.set_active_bands(n_bands)
            losses = estimate_loss(model, dataset)
            if use_prog and current_bands < n_bands:
                model.band_mask = saved_mask

            elapsed = time.time() - start_time
            extra = f" [1-{current_bands}]" if use_prog else ""
            print(f"    step {iter_num:>5} | val {losses['val']:.4f}{extra}")
            history.append((iter_num, losses["train"], losses["val"]))
            if math.isnan(losses["train"]):
                print("    !!! NaN -- stopping.")
                break

        x, y = dataset.get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    total = time.time() - start_time
    final_val = history[-1][2] if history else float("inf")

    ms = model.get_magnitude_stats()
    mag_cv = f"  CV={ms['global_cv']:.2f}%" if ms else ""
    print(f"    Done in {total:.1f}s | val={final_val:.4f}{mag_cv}")

    return {
        "name": name, "n_bands": n_bands, "n_embd": n_embd,
        "mode": mode, "n_params": n_params, "time": total,
        "final_val": final_val, "final_train": history[-1][1] if history else 0,
        "history": history, "mag_cv": ms["global_cv"] if ms else None,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  FREQUENCY BUDGET CURVE — Experiments 3 + 5")
    print(f"  Band counts: {BAND_COUNTS}")
    print(f"  Modes: MLP (ceiling) + two-stage Kerr-ODE (Phase B winner)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text)} chars, {dataset.vocab_size} vocab")

    results = []

    for n_bands in BAND_COUNTS:
        n_embd = n_bands * 2
        print(f"\n{'=' * 70}")
        print(f"  BAND COUNT: {n_bands} ({n_embd}D)")
        print(f"{'=' * 70}")

        # MLP baseline at this dimension
        r_mlp = train_variant(n_bands, "mlp", dataset)
        results.append(r_mlp)

        # Two-stage Kerr-ODE at this dimension
        r_ts = train_variant(n_bands, "two_stage", dataset)
        results.append(r_ts)

    # =========================================================================
    # Results
    # =========================================================================
    print(f"\n\n{'=' * 70}")
    print(f"  RESULTS: Frequency Budget Curve")
    print(f"{'=' * 70}")

    # Reference: MLP at 64 bands
    ref_mlp = next(r for r in results
                   if r["mode"] == "mlp" and r["n_bands"] == 64)
    ref_val = ref_mlp["final_val"]

    print(f"\n  Reference: MLP@64b = {ref_val:.4f}")
    print(f"\n  {'Variant':<18} {'Bands':>5} {'Dim':>4} {'Params':>10}"
          f" {'Val':>8} {'vs MLP@64':>9} {'vs own MLP':>10} {'Time':>5}")
    print(f"  {'-'*18} {'-'*5} {'-'*4} {'-'*10}"
          f" {'-'*8} {'-'*9} {'-'*10} {'-'*5}")

    for n_bands in BAND_COUNTS:
        mlp_r = next(r for r in results
                     if r["mode"] == "mlp" and r["n_bands"] == n_bands)
        ts_r = next(r for r in results
                    if r["mode"] == "two_stage" and r["n_bands"] == n_bands)

        # MLP row
        vs_ref = (mlp_r["final_val"] / ref_val - 1) * 100
        print(f"  {'MLP':<18} {n_bands:>5} {n_bands*2:>4} {mlp_r['n_params']:>10,}"
              f" {mlp_r['final_val']:>8.4f} {vs_ref:>+8.2f}%"
              f" {'---':>10} {mlp_r['time']:>4.0f}s")

        # Two-stage row
        vs_ref = (ts_r["final_val"] / ref_val - 1) * 100
        vs_own = (ts_r["final_val"] / mlp_r["final_val"] - 1) * 100
        mag_str = f" CV={ts_r['mag_cv']:.1f}%" if ts_r["mag_cv"] else ""
        print(f"  {'Two-stage':<18} {n_bands:>5} {n_bands*2:>4} {ts_r['n_params']:>10,}"
              f" {ts_r['final_val']:>8.4f} {vs_ref:>+8.2f}%"
              f" {vs_own:>+9.2f}% {ts_r['time']:>4.0f}s{mag_str}")

    # =========================================================================
    # Budget curve summary
    # =========================================================================
    print(f"\n  BUDGET CURVE (% of MLP@64b performance):")
    print(f"  {'Bands':>5} {'Dim':>4} {'MLP%':>7} {'TwoStg%':>8} {'Gap':>6}")
    print(f"  {'-'*5} {'-'*4} {'-'*7} {'-'*8} {'-'*6}")

    for n_bands in BAND_COUNTS:
        mlp_r = next(r for r in results
                     if r["mode"] == "mlp" and r["n_bands"] == n_bands)
        ts_r = next(r for r in results
                    if r["mode"] == "two_stage" and r["n_bands"] == n_bands)

        # % of MLP@64 = 1 - (val - ref) / ref  (higher is better)
        mlp_pct = (1 - (mlp_r["final_val"] - ref_val) / ref_val) * 100
        ts_pct = (1 - (ts_r["final_val"] - ref_val) / ref_val) * 100
        gap = ts_pct - mlp_pct

        print(f"  {n_bands:>5} {n_bands*2:>4}"
              f" {mlp_pct:>6.1f}% {ts_pct:>7.1f}% {gap:>+5.1f}%")

    # =========================================================================
    # Key questions answered
    # =========================================================================
    print(f"\n  KEY QUESTIONS:")

    # Find minimum bands for 90% of MLP@64
    for threshold in [90, 95]:
        for n_bands in BAND_COUNTS:
            ts_r = next(r for r in results
                        if r["mode"] == "two_stage" and r["n_bands"] == n_bands)
            pct = (1 - (ts_r["final_val"] - ref_val) / ref_val) * 100
            if pct >= threshold:
                print(f"    {threshold}% of MLP@64: {n_bands} bands"
                      f" ({n_bands*2}D) — {pct:.1f}%")
                break
        else:
            print(f"    {threshold}% of MLP@64: not reached")

    # Halving test: 32 bands vs 64 bands
    ts_32 = next((r for r in results
                  if r["mode"] == "two_stage" and r["n_bands"] == 32), None)
    ts_64 = next((r for r in results
                  if r["mode"] == "two_stage" and r["n_bands"] == 64), None)
    if ts_32 and ts_64:
        ratio = ts_32["final_val"] / ts_64["final_val"]
        print(f"    Halving cost: 32b vs 64b = {ratio:.4f}x"
              f" (val {ts_32['final_val']:.4f} vs {ts_64['final_val']:.4f})")

    print(f"\n  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
