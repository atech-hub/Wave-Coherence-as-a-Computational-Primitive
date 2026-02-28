"""
Phase 20b: Expanded LC Layer — Fair Capacity Test

Phase 20 showed the LC concept works mechanically but was parameter-starved:
148 params vs 131K for standard MLP (890x reduction). PyTorch confirmed
LC params learn meaningful patterns (gain differentiating by band, coupling
activating, phase rotating) but couldn't match MLP capacity — 23.5% worse.

Phase 20b gives the LC architecture a fair parameter budget:
  - Per-band FFN: each harmonic band gets a small 2->16->2 network (GELU inside)
  - Cross-band coupling: linear mixing across bands (cos/sin independently)
  - Total: ~13.4K params per layer vs 131K for MLP (9.8x reduction)

Architecture comparison:
  Standard MLP: all 128 dims interact through 128->512->128 dense layers
  Expanded LC:  per-band local processing (2->16->2 × 64 bands) + global band mixing

This factorization mirrors LC circuit physics:
  - Per-band FFN = multi-stage resonator (amplitude + phase + nonlinear reshaping)
  - Cross-band coupling = mutual inductance (energy transfer between frequency bands)
  - The architecture "knows" about harmonic structure; MLP treats dims as independent

Usage:
    python experiments/phase20b_expanded_lc.py
"""

import math
import os
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration — matches Phase 20 exactly
# =============================================================================

N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
N_BANDS = N_EMBD // 2  # 64
BLOCK_SIZE = 256
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_ITERS = 2000
EVAL_INTERVAL = 250
EVAL_ITERS = 50

# Expanded LC parameters
BAND_HIDDEN = 16  # per-band FFN hidden dimension

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Harmonic Embedding — frozen, deterministic
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
# Expanded LC Circuit Layer — fair capacity version
# =============================================================================

class ExpandedLCLayer(nn.Module):
    """
    Frequency-native transformation with fair parameter budget.

    Instead of scalar gain + phase per band (Phase 20: 148 params/layer),
    each band gets a small FFN (2->16->2) plus cross-band linear mixing.
    Total: ~13.4K params per layer (9.8x fewer than MLP's 131K).

    Architecture:
      1. Per-band FFN: each of 64 bands processes its (cos, sin) pair
         through a small 2->16->2 network with GELU. Independent per band.
      2. Cross-band coupling: linear mixing across bands for cos and sin
         channels independently. Residual (zero-init -> starts as passthrough).

    Physical analog:
      - Per-band FFN = multi-stage resonator (richer than gain + phase)
      - Cross-band coupling = mutual inductance network
    """

    def __init__(self, n_bands=N_BANDS, band_hidden=BAND_HIDDEN):
        super().__init__()
        self.n_bands = n_bands
        self.band_hidden = band_hidden

        # Per-band FFN: 2 -> band_hidden -> 2 (independent per band)
        # Using explicit parameter tensors for einsum-based batched matmul
        self.band_up_w = nn.Parameter(torch.empty(n_bands, 2, band_hidden))
        self.band_up_b = nn.Parameter(torch.zeros(n_bands, band_hidden))
        self.band_down_w = nn.Parameter(torch.empty(n_bands, band_hidden, 2))
        self.band_down_b = nn.Parameter(torch.zeros(n_bands, 2))

        # Cross-band coupling: linear mixing across bands (residual)
        # cos and sin channels mix independently — preserves phase relationships
        # Zero-init: coupling starts as passthrough, learns which bands interact
        self.band_mix_cos = nn.Parameter(torch.zeros(n_bands, n_bands))
        self.band_mix_sin = nn.Parameter(torch.zeros(n_bands, n_bands))

        # Init per-band FFN
        nn.init.normal_(self.band_up_w, std=0.02)
        nn.init.normal_(self.band_down_w, std=0.02 / math.sqrt(2 * N_LAYER))

        # Count params
        self._n_params = (
            n_bands * 2 * band_hidden      # band_up_w
            + n_bands * band_hidden         # band_up_b
            + n_bands * band_hidden * 2     # band_down_w
            + n_bands * 2                   # band_down_b
            + n_bands * n_bands             # band_mix_cos
            + n_bands * n_bands             # band_mix_sin
        )

    def forward(self, x):
        B, T, C = x.size()
        bt = B * T

        # Reshape to bands: (bt, n_bands, 2)
        bands = x.view(bt, self.n_bands, 2)

        # 1. Per-band FFN: 2 -> hidden -> 2
        # For each band n: h[b,n,:] = bands[b,n,:] @ band_up_w[n,:,:] + band_up_b[n,:]
        h = torch.einsum('bni,nih->bnh', bands, self.band_up_w) + self.band_up_b
        h = F.gelu(h)
        out = torch.einsum('bnh,nho->bno', h, self.band_down_w) + self.band_down_b
        # out: (bt, n_bands, 2)

        # 2. Cross-band coupling (residual, zero-init -> starts as passthrough)
        cos_ch = out[:, :, 0]  # (bt, n_bands)
        sin_ch = out[:, :, 1]

        cos_out = cos_ch + cos_ch @ self.band_mix_cos
        sin_out = sin_ch + sin_ch @ self.band_mix_sin

        result = torch.stack([cos_out, sin_out], dim=2)  # (bt, n_bands, 2)
        return result.view(B, T, C)

    def get_param_summary(self):
        """Detailed parameter analysis."""
        with torch.no_grad():
            # Per-band FFN stats
            up_norm = self.band_up_w.pow(2).sum(dim=(1, 2)).sqrt()  # (n_bands,)
            down_norm = self.band_down_w.pow(2).sum(dim=(1, 2)).sqrt()

            # Band mix stats
            mix_cos_norm = self.band_mix_cos.pow(2).sum().sqrt().item()
            mix_sin_norm = self.band_mix_sin.pow(2).sum().sqrt().item()

            # Band mix sparsity: what fraction of entries are near-zero?
            mix_abs = torch.cat([self.band_mix_cos.flatten(), self.band_mix_sin.flatten()])
            threshold = mix_abs.max().item() * 0.01 if mix_abs.max().item() > 0 else 1e-6
            mix_sparsity = (mix_abs.abs() < threshold).float().mean().item()

            # Band mix diagonal dominance: are nearby bands coupled more than distant?
            cos_diag = torch.diagonal(self.band_mix_cos).abs().mean().item()
            cos_off = (self.band_mix_cos.abs().sum().item() - torch.diagonal(self.band_mix_cos).abs().sum().item()) / max(1, self.n_bands * (self.n_bands - 1))

            return {
                "up_norm_avg": up_norm.mean().item(),
                "up_norm_std": up_norm.std().item(),
                "down_norm_avg": down_norm.mean().item(),
                "down_norm_std": down_norm.std().item(),
                "mix_cos_norm": mix_cos_norm,
                "mix_sin_norm": mix_sin_norm,
                "mix_sparsity": mix_sparsity,
                "cos_diag_vs_off": cos_diag / max(cos_off, 1e-8),
            }


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
    def __init__(self, ffn_type="mlp"):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.ffn_type = ffn_type
        if ffn_type == "mlp":
            self.ffn = MLP()
        elif ffn_type == "lc_expanded":
            self.ffn = ExpandedLCLayer()
        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}")

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, ffn_type="mlp"):
        super().__init__()
        self.ffn_type = ffn_type
        self.register_buffer("wte", build_harmonic_table(vocab_size, N_EMBD))
        self.register_buffer("wpe", build_positional_table(BLOCK_SIZE, N_EMBD))
        self.blocks = nn.ModuleList([Block(ffn_type=ffn_type) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  {ffn_type} model: {n_params:,} trainable parameters")

        if ffn_type == "lc_expanded":
            ffn_params = sum(
                p.numel()
                for name, p in self.named_parameters()
                if "ffn." in name
            )
            print(f"  LC FFN params: {ffn_params:,} ({ffn_params // N_LAYER:,} per layer x {N_LAYER} layers)")
            print(f"  Standard MLP would be: {N_LAYER * 131712:,} FFN params")
            print(f"  Reduction: {N_LAYER * 131712 / ffn_params:.1f}x fewer FFN params")

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

    def report_lc_params(self):
        if self.ffn_type != "lc_expanded":
            return
        for i, block in enumerate(self.blocks):
            if isinstance(block.ffn, ExpandedLCLayer):
                s = block.ffn.get_param_summary()
                print(f"    L{i}: up_norm={s['up_norm_avg']:.4f}±{s['up_norm_std']:.4f}"
                      f"  down_norm={s['down_norm_avg']:.4f}±{s['down_norm_std']:.4f}"
                      f"  mix_cos={s['mix_cos_norm']:.4f}  mix_sin={s['mix_sin_norm']:.4f}"
                      f"  sparsity={s['mix_sparsity']:.1%}")


# =============================================================================
# Data
# =============================================================================

def download_shakespeare():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    filepath = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(filepath):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python", "data", "shakespeare.txt")
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


def train_mode(mode_name, ffn_type, dataset):
    print(f"\n{'=' * 70}")
    print(f"  Training: {mode_name.upper()}")
    if ffn_type == "mlp":
        print(f"  FFN: standard MLP (128->512->128, ~131K params/layer)")
    else:
        print(f"  FFN: expanded LC (per-band 2->{BAND_HIDDEN}->2 + band coupling, ~13.4K params/layer)")
    print(f"{'=' * 70}")

    model = GPT(dataset.vocab_size, ffn_type=ffn_type).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ffn_params = sum(
        p.numel() for name, p in model.named_parameters()
        if "ffn." in name or "mlp." in name  # handle both naming conventions
    )

    history = []
    start = time.time()

    for iter_num in range(MAX_ITERS):
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model, dataset)
            elapsed = time.time() - start
            print(f"  step {iter_num:>5} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.1f}s")

            if ffn_type == "lc_expanded" and (iter_num == 0 or iter_num % 500 == 0 or iter_num == MAX_ITERS - 1):
                model.report_lc_params()

            history.append((iter_num, losses["train"], losses["val"]))

        x, y = dataset.get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()

        # Gradient check at first iteration
        if iter_num == 0 and ffn_type == "lc_expanded":
            print("\n  === GRADIENT CHECK (iter 0) ===")
            for i, block in enumerate(model.blocks):
                if isinstance(block.ffn, ExpandedLCLayer):
                    lc = block.ffn
                    parts = [
                        ("band_up_w", lc.band_up_w),
                        ("band_up_b", lc.band_up_b),
                        ("band_down_w", lc.band_down_w),
                        ("band_down_b", lc.band_down_b),
                        ("band_mix_cos", lc.band_mix_cos),
                        ("band_mix_sin", lc.band_mix_sin),
                    ]
                    grad_strs = []
                    for name, param in parts:
                        if param.grad is not None:
                            rms = param.grad.pow(2).mean().sqrt().item()
                            grad_strs.append(f"{name}={rms:.2e}")
                        else:
                            grad_strs.append(f"{name}=NONE")
                    print(f"    L{i}: {', '.join(grad_strs)}")
            print()

        optimizer.step()

    total = time.time() - start
    print(f"  Training complete in {total:.1f}s")

    return {
        "mode_name": mode_name,
        "ffn_type": ffn_type,
        "history": history,
        "n_params": n_params,
        "ffn_params": ffn_params,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  Phase 20b: Expanded LC Layer — Fair Capacity Test")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text)} characters, {dataset.vocab_size} unique")
    print(f"  Train: {len(dataset.train_data)} | Val: {len(dataset.val_data)}")
    print(f"  Model: {N_LAYER} layers, {N_HEAD} heads, {N_EMBD} dim, {N_BANDS} bands")
    print(f"  Framework: PyTorch {torch.__version__}")

    print(f"\n  Phase 20 recap (parameter-starved):")
    print(f"    Tiny LC: 148 FFN params/layer -> 23.5% worse than MLP")
    print(f"    But: params learned meaningful patterns (concept works, capacity doesn't)")

    print(f"\n  Phase 20b expanded LC architecture:")
    print(f"    Per-band FFN: 2->{BAND_HIDDEN}->2 × {N_BANDS} bands = {N_BANDS * (2*BAND_HIDDEN + BAND_HIDDEN + BAND_HIDDEN*2 + 2):,} params/layer")
    print(f"    Cross-band coupling: {N_BANDS}×{N_BANDS} × 2 channels = {2 * N_BANDS * N_BANDS:,} params/layer")
    lc_per_layer = N_BANDS * (2*BAND_HIDDEN + BAND_HIDDEN + BAND_HIDDEN*2 + 2) + 2 * N_BANDS * N_BANDS
    print(f"    Total: {lc_per_layer:,} params/layer vs 131,712 for MLP ({131712/lc_per_layer:.1f}x reduction)")

    # Train both modes
    result_std = train_mode("frozen_standard", "mlp", dataset)
    result_lc = train_mode("lc_expanded", "lc_expanded", dataset)

    results = [result_std, result_lc]

    # =========================================================================
    # Comparison
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: Final Validation Loss")
    print(f"{'=' * 70}")
    print()
    print(f"  {'Mode':<20} {'Val Loss':>10} {'Train Loss':>12} {'vs Std':>10} {'Total':>8} {'FFN':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*10} {'-'*8} {'-'*8}")

    std_val = result_std["history"][-1][2]

    for r in results:
        _, train_l, val_l = r["history"][-1]
        if r["mode_name"] == "frozen_standard":
            diff = "baseline"
        else:
            pct = (1.0 - val_l / std_val) * 100.0
            diff = f"{pct:+.2f}%"
        print(f"  {r['mode_name']:<20} {val_l:>10.4f} {train_l:>12.4f} {diff:>10}"
              f" {r['n_params']//1000:>7}K {r['ffn_params']//1000:>7}K")

    # =========================================================================
    # Parameter Efficiency
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  PARAMETER EFFICIENCY")
    print(f"{'=' * 70}")

    lc_val = result_lc["history"][-1][2]
    lc_ffn = result_lc["ffn_params"]
    std_ffn = result_std["ffn_params"]

    print(f"\n  FFN params:  standard={std_ffn:,}  expanded_LC={lc_ffn:,}  ({std_ffn/lc_ffn:.1f}x reduction)")
    print(f"  Total params: standard={result_std['n_params']:,}  expanded_LC={result_lc['n_params']:,}  ({result_std['n_params']/result_lc['n_params']:.1f}x reduction)")

    loss_ratio = lc_val / std_val
    param_ratio = std_ffn / lc_ffn
    print(f"\n  Loss ratio (LC/std): {loss_ratio:.4f}")
    print(f"  FFN param ratio (std/LC): {param_ratio:.1f}x")
    print(f"  Efficiency: {param_ratio / loss_ratio:.1f}x params saved per unit loss")

    # =========================================================================
    # Convergence
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  CONVERGENCE: Val Loss at Each Checkpoint")
    print(f"{'=' * 70}")
    print()
    print(f"  {'Step':>6}  {'standard':>16}  {'lc_expanded':>16}  {'LC vs std':>12}")
    print(f"  {'-' * 56}")

    for i in range(len(result_std["history"])):
        step, _, std_v = result_std["history"][i]
        _, _, lc_v = result_lc["history"][i]
        gain = (1.0 - lc_v / std_v) * 100.0
        print(f"  {step:>6}  {std_v:>16.4f}  {lc_v:>16.4f}  {gain:>+11.2f}%")

    # =========================================================================
    # Cross-phase comparison
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  CROSS-PHASE COMPARISON")
    print(f"{'=' * 70}")
    print()
    print(f"  Phase 20 (tiny LC):     148 FFN params/layer  ->  23.5% worse than MLP")
    print(f"  Phase 20b (expanded):   {lc_per_layer:,} FFN params/layer  ->  {(1.0 - lc_val/std_val)*100:+.2f}% vs MLP")
    print(f"  Standard MLP:           131,712 FFN params/layer (baseline)")
    print()

    improvement_from_20 = abs(-23.5) - abs((1.0 - lc_val/std_val)*100)
    if lc_val <= std_val * 1.01:
        print(f"  RESULT: Expanded LC MATCHES standard MLP!")
        print(f"  Frequency-native structure compensates for 10x fewer FFN params.")
        print(f"  This confirms Phase 20 was parameter-starved, not concept-broken.")
    elif lc_val < std_val * 1.10:
        pct_worse = (lc_val / std_val - 1.0) * 100
        print(f"  RESULT: Expanded LC within {pct_worse:.1f}% of MLP (down from 23.5%)")
        print(f"  Capacity increase closed {improvement_from_20:.1f} percentage points of the gap.")
        if pct_worse < 5:
            print(f"  The remaining gap is small — more capacity or architectural tuning may close it.")
        else:
            print(f"  Significant improvement but still a gap — frequency structure helps but doesn't fully compensate.")
    else:
        pct_worse = (lc_val / std_val - 1.0) * 100
        print(f"  RESULT: Expanded LC {pct_worse:.1f}% worse than MLP (down from 23.5%)")
        print(f"  More capacity helped but the gap remains large.")
        print(f"  MLP's dense cross-dimension interaction may be fundamentally necessary.")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
