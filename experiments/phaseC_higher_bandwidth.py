"""
Higher Bandwidth Test — Experiment 1 follow-up

Does the Kerr-ODE benefit from extra bandwidth as much as MLP?
Flat training (no curriculum) at 96 and 128 bands to complete the
locality penalty curve: 32, 48, 64, 96, 128 bands all flat.

From flat_vs_curriculum.py:
  32 bands: Kerr gap ~3.0%
  48 bands: Kerr gap ~4.9%
From band_count_sweep.py (curriculum):
  64 bands: Kerr gap ~3.4% (curriculum-tuned)
  96 bands: Kerr gap ~4.6% (curriculum)

This adds flat 96 and 128 to see if the gap widens or plateaus.
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

BAND_COUNTS = [96, 128]

N_LAYER = 4
N_HEAD = 4
BLOCK_SIZE = 256
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_ITERS = 2000
EVAL_INTERVAL = 200
EVAL_ITERS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Model components (same as flat_vs_curriculum.py)
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


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


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

        if mode == "mlp":
            layers = [Block(n_embd, N_HEAD, MLP(n_embd))
                      for _ in range(N_LAYER)]
        elif mode == "kerr":
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

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

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
        n_embd = self.n_bands * 2
        tok_emb = F.embedding(idx, self.wte)
        x = tok_emb + self.wpe[:T]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss


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
    name = f"{mode}_{n_bands}b_flat"

    print(f"\n  --- {name} ({n_embd}D) ---")

    torch.manual_seed(42)
    model = GPT(dataset.vocab_size, n_bands, mode=mode).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    history = []
    start_time = time.time()

    for iter_num in range(MAX_ITERS):
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model, dataset)
            elapsed = time.time() - start_time
            print(f"    step {iter_num:>5} | val {losses['val']:.4f}")
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
    print(f"    Done in {total:.1f}s | val={final_val:.4f}")

    return {
        "name": name, "n_bands": n_bands, "mode": mode,
        "n_params": n_params, "final_val": final_val, "time": total,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  HIGHER BANDWIDTH TEST -- Experiment 1")
    print(f"  Band counts: {BAND_COUNTS}")
    print(f"  All flat training (no curriculum)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text)} chars, {dataset.vocab_size} vocab")

    results = []

    for n_bands in BAND_COUNTS:
        print(f"\n{'=' * 70}")
        print(f"  BAND COUNT: {n_bands} ({n_bands * 2}D)")
        print(f"{'=' * 70}")

        r_mlp = train_variant(n_bands, "mlp", dataset)
        results.append(r_mlp)

        r_kerr = train_variant(n_bands, "kerr", dataset)
        results.append(r_kerr)

    # =========================================================================
    # Results + combined curve
    # =========================================================================
    print(f"\n\n{'=' * 70}")
    print(f"  RESULTS: Higher Bandwidth")
    print(f"{'=' * 70}")

    # Previous flat results for complete curve
    prev = [
        (32, "mlp", 2.0141), (32, "kerr", 2.0758),
        (48, "mlp", 1.8263), (48, "kerr", 1.9165),
    ]
    # Sweep results (curriculum) for 64 bands
    prev_64 = [(64, "mlp", 1.6932), (64, "kerr_curriculum", 1.7511)]

    ref_mlp64 = 1.6932

    print(f"\n  COMPLETE LOCALITY PENALTY CURVE (flat training)")
    print(f"  Reference: MLP@64b = {ref_mlp64}")
    print(f"\n  {'Bands':>5} {'Dim':>4} {'MLP val':>8} {'Kerr val':>9}"
          f" {'Kerr gap':>9} {'MLP % of 64':>12}")
    print(f"  {'-'*5} {'-'*4} {'-'*8} {'-'*9} {'-'*9} {'-'*12}")

    # 32 and 48 from flat_vs_curriculum
    for n_bands, mlp_val, kerr_val in [(32, 2.0141, 2.0758), (48, 1.8263, 1.9165)]:
        gap = (kerr_val / mlp_val - 1) * 100
        mlp_pct = (1 - (mlp_val - ref_mlp64) / ref_mlp64) * 100
        print(f"  {n_bands:>5} {n_bands*2:>4} {mlp_val:>8.4f} {kerr_val:>9.4f}"
              f" {gap:>+8.2f}% {mlp_pct:>11.1f}%")

    # 64 from sweep (curriculum -- note this)
    print(f"  {'64*':>5} {'128':>4} {'1.6932':>8} {'1.7511':>9}"
          f" {'+3.42%':>9} {'100.0%':>12}  (* curriculum)")

    # New results
    for n_bands in BAND_COUNTS:
        mlp_r = next(r for r in results
                     if r["mode"] == "mlp" and r["n_bands"] == n_bands)
        kerr_r = next(r for r in results
                      if r["mode"] == "kerr" and r["n_bands"] == n_bands)
        gap = (kerr_r["final_val"] / mlp_r["final_val"] - 1) * 100
        mlp_pct = (1 - (mlp_r["final_val"] - ref_mlp64) / ref_mlp64) * 100
        print(f"  {n_bands:>5} {n_bands*2:>4} {mlp_r['final_val']:>8.4f}"
              f" {kerr_r['final_val']:>9.4f} {gap:>+8.2f}%"
              f" {mlp_pct:>11.1f}%")

    # Key questions
    print(f"\n  KEY QUESTIONS:")
    for n_bands in BAND_COUNTS:
        mlp_r = next(r for r in results
                     if r["mode"] == "mlp" and r["n_bands"] == n_bands)
        kerr_r = next(r for r in results
                      if r["mode"] == "kerr" and r["n_bands"] == n_bands)
        gap = (kerr_r["final_val"] / mlp_r["final_val"] - 1) * 100
        print(f"    {n_bands} bands: Kerr gap = {gap:+.2f}%"
              f" (MLP={mlp_r['final_val']:.4f},"
              f" Kerr={kerr_r['final_val']:.4f})")

    print(f"\n  Does locality penalty grow with bandwidth?")
    print(f"  32b: ~3.1% | 48b: ~4.9% | 64b*: 3.4% | see above for 96/128")

    print(f"\n  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
