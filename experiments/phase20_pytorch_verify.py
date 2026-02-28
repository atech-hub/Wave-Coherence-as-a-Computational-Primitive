"""
Phase 20 PyTorch Verification — Do LC Parameters Actually Learn?

The Rust/candle implementation showed LC parameters frozen at init values
(gain=1.0, phase=0.0, coupling=0.0) for 2000 iterations. Yet the LC model
OUTPERFORMED standard MLP (3.0793 vs 3.0966, +0.56%). This means the LC
layer was effectively just GELU(x) — and attention + GELU beat attention + MLP.

Two possibilities:
  1. Candle autograd blocked gradients again (same as Phase 19b Finding #7)
  2. The LC parameters genuinely have zero gradient at the identity point

PyTorch will settle it. If LC params learn, we discover the real LC dynamics.
If they don't, GELU alone is genuinely optimal for this model.

Usage:
  python experiments/phase20_pytorch_verify.py
"""

import math
import os
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration — matches Rust exactly
# =============================================================================

N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
N_BANDS = N_EMBD // 2  # 64
COUPLING_K = 2  # 5-wide neighborhood
BLOCK_SIZE = 256
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_ITERS = 2000
EVAL_INTERVAL = 250
EVAL_ITERS = 50

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
# LC Circuit Layer
# =============================================================================

class LCLayer(nn.Module):
    """
    Frequency-native transformation layer. Operates on harmonic bands natively.

    Components:
      - Resonance: per-band amplitude gain and phase rotation
      - Coupling: cross-band interaction (residual, 5-wide convolution)
      - GELU nonlinearity

    At init (gain=1, phase=0, coupling=0): acts as identity + GELU.
    """

    def __init__(self):
        super().__init__()
        # Resonance: per-band gain and phase rotation
        self.gain = nn.Parameter(torch.ones(N_BANDS))
        self.phase = nn.Parameter(torch.zeros(N_BANDS))

        # Coupling: cross-band interaction weights (2k+1, 2, 2)
        # Init to zero — residual design means zero = passthrough
        k = COUPLING_K
        self.coupling_weights = nn.Parameter(torch.zeros(2 * k + 1, 2, 2))

        self.k = k
        self.n_bands = N_BANDS

    def forward(self, x):
        B, T, C = x.size()
        bt = B * T

        # 1. Reshape to harmonic bands: (B*T, N_BANDS, 2)
        bands = x.view(bt, self.n_bands, 2)

        # 2. Resonance: per-band gain and phase rotation
        cos_p = torch.cos(self.phase)  # (N_BANDS,)
        sin_p = torch.sin(self.phase)

        in_cos = bands[:, :, 0]  # (bt, N_BANDS)
        in_sin = bands[:, :, 1]

        # Phase rotation
        rot_cos = in_cos * cos_p - in_sin * sin_p
        rot_sin = in_cos * sin_p + in_sin * cos_p

        # Amplitude scaling
        res_cos = rot_cos * self.gain
        res_sin = rot_sin * self.gain

        # Stack: (bt, N_BANDS, 2)
        resonated = torch.stack([res_cos, res_sin], dim=2)

        # 3. Coupling: residual cross-band interaction
        coupled = self.coupling_residual(resonated)

        # 4. GELU
        activated = F.gelu(coupled)

        # 5. Reshape back
        return activated.view(B, T, C)

    def coupling_residual(self, x):
        """Cross-band interaction via shifted matmuls with zero-padding."""
        bt, n_bands, _ = x.shape
        k = self.k

        # Zero-pad along band dimension
        padded = F.pad(x, (0, 0, k, k))  # pad band dim: (bt, n_bands+2k, 2)

        # Accumulate coupling contributions
        coupling_sum = torch.zeros_like(x)
        for j in range(2 * k + 1):
            s = padded[:, j:j + n_bands, :]  # (bt, n_bands, 2)
            w = self.coupling_weights[j]  # (2, 2)
            coupling_sum = coupling_sum + s @ w

        # Residual
        return x + coupling_sum

    def get_param_summary(self):
        """Return summary of learned parameters."""
        gain = self.gain.detach().cpu()
        phase = self.phase.detach().cpu()
        coupling = self.coupling_weights.detach().cpu()

        return {
            "gain_avg": gain.mean().item(),
            "gain_min": gain.min().item(),
            "gain_max": gain.max().item(),
            "phase_avg": phase.mean().item(),
            "phase_min": phase.min().item(),
            "phase_max": phase.max().item(),
            "coupling_rms": coupling.pow(2).mean().sqrt().item(),
        }

    def get_grad_summary(self):
        """Return gradient magnitudes."""
        result = {}
        if self.gain.grad is not None:
            result["gain_grad_rms"] = self.gain.grad.pow(2).mean().sqrt().item()
        if self.phase.grad is not None:
            result["phase_grad_rms"] = self.phase.grad.pow(2).mean().sqrt().item()
        if self.coupling_weights.grad is not None:
            result["coupling_grad_rms"] = self.coupling_weights.grad.pow(2).mean().sqrt().item()
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
    def __init__(self, use_lc=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.use_lc = use_lc
        if use_lc:
            self.lc = LCLayer()
        else:
            self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        if self.use_lc:
            x = x + self.lc(self.ln_2(x))
        else:
            x = x + self.mlp(self.ln_2(x))
        return x


class LCGPT(nn.Module):
    def __init__(self, vocab_size, use_lc=False):
        super().__init__()
        self.use_lc = use_lc
        self.register_buffer("wte", build_harmonic_table(vocab_size, N_EMBD))
        self.register_buffer("wpe", build_positional_table(BLOCK_SIZE, N_EMBD))
        self.blocks = nn.ModuleList([Block(use_lc=use_lc) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mode = "lc_layer" if use_lc else "standard"
        print(f"  {mode} model: {n_params:,} trainable parameters")

        if use_lc:
            lc_params = sum(
                p.numel()
                for name, p in self.named_parameters()
                if "lc." in name
            )
            print(f"  LC params: {lc_params} ({lc_params // N_LAYER} per layer x {N_LAYER} layers)")

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
        if not self.use_lc:
            return
        print("\n  LC Layer Parameters:")
        for i, block in enumerate(self.blocks):
            if hasattr(block, "lc"):
                s = block.lc.get_param_summary()
                g = block.lc.get_grad_summary()
                print(f"    layer {i}:")
                print(f"      gain:  avg={s['gain_avg']:.4f} [{s['gain_min']:.4f}, {s['gain_max']:.4f}]"
                      + (f"  grad_rms={g['gain_grad_rms']:.8f}" if "gain_grad_rms" in g else ""))
                print(f"      phase: avg={s['phase_avg']:.4f} [{s['phase_min']:.4f}, {s['phase_max']:.4f}]"
                      + (f"  grad_rms={g['phase_grad_rms']:.8f}" if "phase_grad_rms" in g else ""))
                print(f"      coupling_rms={s['coupling_rms']:.6f}"
                      + (f"  grad_rms={g['coupling_grad_rms']:.8f}" if "coupling_grad_rms" in g else ""))


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


def train_mode(mode_name, use_lc, dataset):
    print(f"\n{'=' * 60}")
    print(f"  Training: {mode_name.upper()}")
    print(f"  FFN: {'LC circuit layer' if use_lc else 'standard MLP (128->512->128)'}")
    print(f"{'=' * 60}")

    model = LCGPT(dataset.vocab_size, use_lc=use_lc).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    history = []
    start = time.time()

    for iter_num in range(MAX_ITERS):
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model, dataset)
            elapsed = time.time() - start
            print(f"  step {iter_num:>5} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {elapsed:.1f}s")

            if use_lc:
                model.report_lc_params()

            history.append((iter_num, losses["train"], losses["val"]))

        x, y = dataset.get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()

        # Gradient check at first iteration
        if iter_num == 0 and use_lc:
            print("\n  === GRADIENT CHECK (iter 0) ===")
            for i, block in enumerate(model.blocks):
                if hasattr(block, "lc"):
                    lc = block.lc
                    gain_grad = lc.gain.grad
                    phase_grad = lc.phase.grad
                    coupling_grad = lc.coupling_weights.grad
                    print(f"    layer {i}:")
                    if gain_grad is not None:
                        print(f"      gain grad:     rms={gain_grad.pow(2).mean().sqrt():.8f}  max={gain_grad.abs().max():.8f}")
                    else:
                        print(f"      gain grad:     NONE")
                    if phase_grad is not None:
                        print(f"      phase grad:    rms={phase_grad.pow(2).mean().sqrt():.8f}  max={phase_grad.abs().max():.8f}")
                    else:
                        print(f"      phase grad:    NONE")
                    if coupling_grad is not None:
                        print(f"      coupling grad: rms={coupling_grad.pow(2).mean().sqrt():.8f}  max={coupling_grad.abs().max():.8f}")
                    else:
                        print(f"      coupling grad: NONE")
            print()

        optimizer.step()

    total = time.time() - start
    print(f"  Training complete in {total:.1f}s")

    return {
        "mode_name": mode_name,
        "history": history,
        "n_params": n_params,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  Phase 20 PyTorch Verification")
    print("  Do LC parameters actually learn?")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text)} characters, {dataset.vocab_size} unique")
    print(f"  Train: {len(dataset.train_data)} | Val: {len(dataset.val_data)}")
    print(f"  Model: {N_LAYER} layers, {N_HEAD} heads, {N_EMBD} dim, {N_BANDS} bands")
    print(f"  Framework: PyTorch {torch.__version__}")

    print(f"\n  LC layer: resonance (gain + phase) + coupling (5-wide) + GELU")
    print(f"  Candle result: LC params frozen at init, model still beat MLP by 0.56%")
    print(f"  Question: do LC params have gradient in PyTorch?")

    result_std = train_mode("frozen_standard", False, dataset)
    result_lc = train_mode("lc_layer", True, dataset)

    results = [result_std, result_lc]

    # =========================================================================
    # Comparison
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON: Final Validation Loss")
    print(f"{'=' * 60}")
    print()
    print(f"  {'Mode':<20} {'Val Loss':>10} {'Train Loss':>12} {'vs Std':>10} {'Params':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")

    std_val = result_std["history"][-1][2]
    for result in results:
        _, train_l, val_l = result["history"][-1]
        if result["mode_name"] == "frozen_standard":
            diff = "—"
        else:
            pct = (1.0 - val_l / std_val) * 100.0
            diff = f"{pct:+.1f}%"
        print(f"  {result['mode_name']:<20} {val_l:>10.4f} {train_l:>12.4f} {diff:>10} {result['n_params']//1000:>7}K")

    # =========================================================================
    # Convergence
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"  CONVERGENCE: Val Loss at Each Checkpoint")
    print(f"{'=' * 60}")
    print()
    print(f"  {'Step':>6}  {'standard':>16}  {'lc_layer':>16}  {'LC gain':>10}")
    print(f"  {'-' * 56}")

    for i in range(len(result_std["history"])):
        step, _, std_v = result_std["history"][i]
        _, _, lc_v = result_lc["history"][i]
        gain = (1.0 - lc_v / std_v) * 100.0
        print(f"  {step:>6}  {std_v:>16.4f}  {lc_v:>16.4f}  {gain:>+9.1f}%")

    # =========================================================================
    # Cross-Framework
    # =========================================================================
    lc_val = result_lc["history"][-1][2]
    lc_pct = (1.0 - lc_val / std_val) * 100.0

    print(f"\n{'=' * 60}")
    print(f"  CROSS-FRAMEWORK COMPARISON")
    print(f"{'=' * 60}")
    print()
    print(f"  Candle (Rust):  LC params frozen at init for 2000 iters")
    print(f"                  LC val 3.0793 vs std 3.0966 (+0.56%)")
    print(f"                  Effective LC layer = GELU(x)")
    print()
    print(f"  PyTorch:        LC val {lc_val:.4f} vs std {std_val:.4f} ({lc_pct:+.1f}%)")
    print()

    if abs(lc_pct) < 1.0:
        print(f"  Both frameworks show LC ≈ standard (within 1%).")
    elif lc_pct > 0:
        print(f"  LC outperforms standard in PyTorch too!")
    else:
        print(f"  LC underperforms standard in PyTorch.")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
