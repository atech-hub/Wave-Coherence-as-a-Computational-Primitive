"""
Phase 22c: Wider Clamps -- Is the Ceiling the Clamp or the Dynamics?

Phase 22 found 95% of L3 bands hitting the [-10, 10] clamp. Phase 22b
showed downstream layers adapt to the exact output distribution, including
clamping artifacts. This experiment trains from scratch with different
clamp bounds to separate "architecture forces saturation" from "computation
naturally saturates."

Four modes (all trained from scratch):
  1. MLP baseline
  2. Kerr [-10, 10] -- current architecture (reproduce)
  3. Kerr [-50, 50] -- wider range
  4. Kerr unclamped  -- [-1000, 1000] safety net only

Key diagnostic: actual dynamic range per layer. If the model stays within
[-10, 10] even without the clamp, the dynamics naturally saturate and the
clamp wasn't the constraint. If bands go wider, the clamp was the bottleneck.

Usage:
    python experiments/phase22c_wider_clamps.py
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
# Kerr-ODE Layer with configurable clamp
# =============================================================================

class KerrODELayer(nn.Module):
    def __init__(self, n_bands=N_BANDS, n_embd=N_EMBD, n_steps=8, clamp_bound=10.0):
        super().__init__()
        self.n_bands = n_bands
        self.n_embd = n_embd
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps
        self.clamp_bound = clamp_bound

        self._gamma_raw = nn.Parameter(torch.full((n_bands,), math.log(math.exp(0.1) - 1)))
        omega_init = torch.arange(1, n_bands + 1, dtype=torch.float32) / n_bands
        self.omega = nn.Parameter(omega_init)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('neighbor_kernel',
                             torch.tensor([[[1.0, 1.0, 0.0, 1.0, 1.0]]]))

        # Dynamic range tracking (not parameters, just buffers for monitoring)
        self.register_buffer('max_magnitude', torch.zeros(n_bands))
        self.register_buffer('clamp_hits', torch.zeros(n_bands))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))

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
        cb = self.clamp_bound

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

            # Track dynamic range before clamping (during training)
            if self.training:
                with torch.no_grad():
                    mag = torch.sqrt(r * r + s * s)
                    batch_max = mag.max(dim=0).values  # (n_bands,)
                    self.max_magnitude = torch.max(self.max_magnitude, batch_max)
                    # Count how many bands exceed the old [-10, 10] equivalent
                    self.clamp_hits += (mag.max(dim=0).values > 10.0).float()
                    self.step_count += 1

            r = torch.clamp(r, -cb, cb)
            s = torch.clamp(s, -cb, cb)

        out = torch.stack([r, s], dim=2).reshape(bt, C)
        out = self.out_proj(out)
        return out.view(B, T, C)

    def get_range_stats(self):
        """Return dynamic range statistics."""
        return {
            'max_mag_mean': self.max_magnitude.mean().item(),
            'max_mag_max': self.max_magnitude.max().item(),
            'max_mag_min': self.max_magnitude.min().item(),
            'bands_above_10': (self.max_magnitude > 10.0).sum().item(),
            'bands_above_50': (self.max_magnitude > 50.0).sum().item(),
            'bands_above_100': (self.max_magnitude > 100.0).sum().item(),
        }

    def reset_range_stats(self):
        self.max_magnitude.zero_()
        self.clamp_hits.zero_()
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
    def __init__(self, ffn_type="mlp", n_steps=8, clamp_bound=10.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.ffn_type = ffn_type
        if ffn_type == "mlp":
            self.ffn = MLP()
        elif ffn_type == "kerr":
            self.ffn = KerrODELayer(n_steps=n_steps, clamp_bound=clamp_bound)
        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}")

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, ffn_type="mlp", n_steps=8, clamp_bound=10.0):
        super().__init__()
        self.ffn_type = ffn_type
        self.clamp_bound = clamp_bound
        self.register_buffer("wte", build_harmonic_table(vocab_size, N_EMBD))
        self.register_buffer("wpe", build_positional_table(BLOCK_SIZE, N_EMBD))
        self.blocks = nn.ModuleList(
            [Block(ffn_type=ffn_type, n_steps=n_steps, clamp_bound=clamp_bound)
             for _ in range(N_LAYER)]
        )
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  {ffn_type} (clamp={clamp_bound}) model: {n_params:,} trainable parameters")

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


def train_mode(mode_name, ffn_type, clamp_bound, dataset):
    print(f"\n{'=' * 70}")
    print(f"  Training: {mode_name}")
    if ffn_type == "kerr":
        print(f"  Clamp bounds: [-{clamp_bound}, {clamp_bound}]")
    print(f"{'=' * 70}")

    model = GPT(dataset.vocab_size, ffn_type=ffn_type, n_steps=8,
                clamp_bound=clamp_bound).to(DEVICE)
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

            # Report dynamic range for Kerr modes
            if ffn_type == "kerr":
                for i, block in enumerate(model.blocks):
                    if isinstance(block.ffn, KerrODELayer):
                        s = block.ffn.get_range_stats()
                        gamma = block.ffn.gamma
                        print(f"    L{i}: max_mag=[{s['max_mag_min']:.1f},{s['max_mag_max']:.1f}]"
                              f" avg={s['max_mag_mean']:.1f}"
                              f"  >10: {s['bands_above_10']}/64"
                              f"  >50: {s['bands_above_50']}/64"
                              f"  >100: {s['bands_above_100']}/64"
                              f"  alpha={block.ffn.alpha.item():.4f}"
                              f"  gamma=[{gamma.min().item():.3f},{gamma.max().item():.3f}]")

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

    # Final dynamic range report
    if ffn_type == "kerr":
        print(f"\n  Final dynamic range:")
        for i, block in enumerate(model.blocks):
            if isinstance(block.ffn, KerrODELayer):
                s = block.ffn.get_range_stats()
                print(f"    L{i}: peak magnitude per band"
                      f" [{s['max_mag_min']:.2f}, {s['max_mag_max']:.2f}]"
                      f" mean={s['max_mag_mean']:.2f}")
                print(f"         >10: {s['bands_above_10']}/64"
                      f"  >50: {s['bands_above_50']}/64"
                      f"  >100: {s['bands_above_100']}/64")

    return {
        "mode_name": mode_name,
        "ffn_type": ffn_type,
        "clamp_bound": clamp_bound,
        "history": history,
        "n_params": n_params,
        "model": model,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  Phase 22c: Wider Clamps -- Is the Ceiling the Clamp?")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text)} characters, {dataset.vocab_size} unique")
    print(f"  Train: {len(dataset.train_data)} | Val: {len(dataset.val_data)}")
    print(f"  Model: {N_LAYER} layers, {N_HEAD} heads, {N_EMBD} dim, {N_BANDS} bands")
    print(f"  Framework: PyTorch {torch.__version__}")

    print(f"\n  Phase 22 finding: L3 has 61/64 bands hitting [-10, 10] clamp")
    print(f"  Question: is the clamp constraining or are the dynamics naturally bounded?")

    # Train all modes
    result_mlp = train_mode("MLP baseline", "mlp", 10.0, dataset)
    result_10 = train_mode("Kerr [-10, 10]", "kerr", 10.0, dataset)
    result_50 = train_mode("Kerr [-50, 50]", "kerr", 50.0, dataset)
    result_1000 = train_mode("Kerr unclamped", "kerr", 1000.0, dataset)

    results = [result_mlp, result_10, result_50, result_1000]

    # =========================================================================
    # Comparison
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: Final Validation Loss")
    print(f"{'=' * 70}")

    mlp_val = result_mlp["history"][-1][2]

    print(f"\n  {'Mode':<22} {'Val':>8} {'Train':>8} {'vs MLP':>9} {'vs [-10]':>9}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")

    kerr10_val = result_10["history"][-1][2]

    for r in results:
        if not r["history"]:
            continue
        _, train_l, val_l = r["history"][-1]
        vs_mlp = f"{(val_l/mlp_val - 1)*100:+.2f}%" if r["mode_name"] != "MLP baseline" else "baseline"
        vs_10 = f"{(val_l/kerr10_val - 1)*100:+.2f}%" if r["mode_name"] != "Kerr [-10, 10]" else "baseline"
        print(f"  {r['mode_name']:<22} {val_l:>8.4f} {train_l:>8.4f} {vs_mlp:>9} {vs_10:>9}")

    # =========================================================================
    # Convergence
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  CONVERGENCE")
    print(f"{'=' * 70}")
    print()
    print(f"  {'Step':>6}  {'MLP':>10}  {'[-10,10]':>10}  {'[-50,50]':>10}  {'unclamped':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    min_len = min(len(r["history"]) for r in results if r["history"])
    for i in range(min_len):
        step = result_mlp["history"][i][0]
        vals = [r["history"][i][2] if i < len(r["history"]) else float('nan') for r in results]
        print(f"  {step:>6}  {vals[0]:>10.4f}  {vals[1]:>10.4f}  {vals[2]:>10.4f}  {vals[3]:>10.4f}")

    # =========================================================================
    # Dynamic Range Analysis
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  DYNAMIC RANGE ANALYSIS")
    print(f"{'=' * 70}")
    print()
    print(f"  Peak band magnitude reached during training:")
    print(f"  {'':>10} {'L0':>15} {'L1':>15} {'L2':>15} {'L3':>15}")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")

    for r in [result_10, result_50, result_1000]:
        if r["ffn_type"] != "kerr" or not r["history"]:
            continue
        name = f"[{r['clamp_bound']:.0f}]"
        layer_stats = []
        for block in r["model"].blocks:
            if isinstance(block.ffn, KerrODELayer):
                s = block.ffn.get_range_stats()
                layer_stats.append(f"{s['max_mag_max']:>7.1f}")
            else:
                layer_stats.append("    -  ")
        print(f"  {name:>10} {'  '.join(layer_stats)}")

    # =========================================================================
    # Key Question
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  KEY QUESTION: Was the clamp the bottleneck?")
    print(f"{'=' * 70}")
    print()

    if not result_50["history"] or not result_1000["history"]:
        print("  Some modes failed (NaN). Cannot fully answer.")
    else:
        val_10 = result_10["history"][-1][2]
        val_50 = result_50["history"][-1][2]
        val_1000 = result_1000["history"][-1][2]

        imp_50 = (val_10 - val_50) / val_10 * 100
        imp_1000 = (val_10 - val_1000) / val_10 * 100

        print(f"  [-10, 10] val: {val_10:.4f}")
        print(f"  [-50, 50] val: {val_50:.4f}  ({imp_50:+.2f}% vs [-10])")
        print(f"  unclamped val: {val_1000:.4f}  ({imp_1000:+.2f}% vs [-10])")

        # Check if wider models used the extra range
        for r in [result_50, result_1000]:
            any_above_10 = False
            for block in r["model"].blocks:
                if isinstance(block.ffn, KerrODELayer):
                    if block.ffn.get_range_stats()['bands_above_10'] > 0:
                        any_above_10 = True
            name = f"[{r['clamp_bound']:.0f}]"
            if any_above_10:
                print(f"\n  {name}: model DID use wider range (bands exceeded 10)")
            else:
                print(f"\n  {name}: model stayed within [-10, 10] voluntarily")

        if imp_1000 > 1.0:
            print(f"\n  VERDICT: The clamp WAS a bottleneck.")
            print(f"  Removing it improved loss by {imp_1000:.2f}%.")
            gap_10 = (val_10 / mlp_val - 1) * 100
            gap_1000 = (val_1000 / mlp_val - 1) * 100
            print(f"  MLP gap: {gap_10:.2f}% -> {gap_1000:.2f}% ({gap_10 - gap_1000:.2f}pp closed)")
        elif imp_1000 > 0.1:
            print(f"\n  VERDICT: Minor bottleneck ({imp_1000:.2f}% improvement)")
            print(f"  The clamp was slightly constraining but not the main issue")
        elif imp_1000 < -1.0:
            print(f"\n  VERDICT: Wider range HURT ({imp_1000:.2f}%)")
            print(f"  The clamp was providing useful regularisation")
        else:
            print(f"\n  VERDICT: The clamp was NOT the bottleneck")
            print(f"  The dynamics naturally saturate in the same range")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
