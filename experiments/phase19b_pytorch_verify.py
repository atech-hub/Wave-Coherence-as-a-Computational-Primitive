"""
Phase 19b PyTorch Verification — Does Lambda Actually Learn?

The Rust/candle implementation showed lambda stuck at exactly 0.100000
for 2000 iterations. Two possible explanations:
  1. Candle autograd bug — gradients don't flow through frozen embeddings
  2. The maths genuinely produce no useful gradient for lambda

PyTorch's autograd is mature and handles this case correctly. This script
reimplements Phase 19b with identical architecture and hyperparameters:
  - 4 layers, 4 heads, 128 dim, 256 block size
  - Frozen harmonic embeddings (register_buffer)
  - Learnable lambda per head per layer (nn.Parameter, init 0.1)
  - score = Q·K^T/sqrt(d) + λ * interference(i,j)
  - 2000 iterations, eval every 250, batch 64, lr 3e-4

Three possible outcomes:
  A. Lambda learns to zero → model rejects harmonic bias → confirms finding
  B. Lambda learns non-zero + loss improves → candle bug hid a real result
  C. Lambda stays at 0.1 → same as candle → computation graph issue, not framework bug

Usage:
  python experiments/phase19b_pytorch_verify.py
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
    """Deterministic phase encoding: cos(n*theta), sin(n*theta)."""
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
    """Sinusoidal positional encoding (frozen)."""
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
# Model Components
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, harmonic_bias=False):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.n_head = N_HEAD
        self.n_embd = N_EMBD
        self.harmonic_bias = harmonic_bias

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
            .view(1, 1, BLOCK_SIZE, BLOCK_SIZE),
        )

        if harmonic_bias:
            # Learnable lambda per head, shape (1, N_HEAD, 1, 1)
            # init 0.1 — same as Rust version
            self.lambda_param = nn.Parameter(
                torch.full((1, N_HEAD, 1, 1), 0.1)
            )
        else:
            self.lambda_param = None

    def forward(self, x, emb=None):
        B, T, C = x.size()
        head_dim = C // self.n_head

        # Standard Q/K/V — always learned
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Standard attention scores
        scale = 1.0 / math.sqrt(head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        # Add harmonic interference bias if enabled
        if self.lambda_param is not None and emb is not None:
            # Partition embedding by frequency band per head
            emb_heads = emb.view(B, T, self.n_head, head_dim).transpose(1, 2)
            # Interference: dot product of embedding sub-vectors
            interference = (emb_heads @ emb_heads.transpose(-2, -1)) * scale
            # score = Q·K^T/sqrt(d) + lambda * interference
            scores = scores + self.lambda_param * interference

        # Causal mask + softmax
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(scores, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, harmonic_bias=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention(harmonic_bias=harmonic_bias)
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.mlp = MLP()

    def forward(self, x, emb=None):
        x = x + self.attn(self.ln_1(x), emb=emb)
        x = x + self.mlp(self.ln_2(x))
        return x


# =============================================================================
# The Model
# =============================================================================

class BiasedGPT(nn.Module):
    def __init__(self, vocab_size, harmonic_bias=False):
        super().__init__()
        self.harmonic_bias = harmonic_bias

        # Frozen harmonic embeddings — register_buffer, not Parameter
        self.register_buffer("wte", build_harmonic_table(vocab_size, N_EMBD))
        self.register_buffer("wpe", build_positional_table(BLOCK_SIZE, N_EMBD))

        self.blocks = nn.ModuleList([
            Block(harmonic_bias=harmonic_bias) for _ in range(N_LAYER)
        ])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)

        # Standard weight init
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mode_str = "harmonic_bias" if harmonic_bias else "standard"
        print(f"  {mode_str} model: {n_params:,} trainable parameters")
        if harmonic_bias:
            n_lambda = N_HEAD * N_LAYER
            print(f"  (includes {n_lambda} lambda parameters: {N_HEAD} per head x {N_LAYER} layers)")

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
        emb = tok_emb + pos_emb

        emb_ref = emb if self.harmonic_bias else None

        x = emb.clone()
        for block in self.blocks:
            x = block(x, emb=emb_ref)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_lambda_values(self):
        """Return lambda values per layer per head."""
        if not self.harmonic_bias:
            return None
        result = []
        for block in self.blocks:
            lam = block.attn.lambda_param
            result.append(lam.detach().flatten().tolist())
        return result

    def report_lambda(self):
        """Print lambda values and their gradients."""
        if not self.harmonic_bias:
            return
        print("\n  Lambda values (learned harmonic bias strength):")
        for i, block in enumerate(self.blocks):
            lam = block.attn.lambda_param
            vals = lam.detach().flatten().tolist()
            grad_str = ""
            if lam.grad is not None:
                grads = lam.grad.flatten().tolist()
                grad_str = "  grad=[" + ", ".join(f"{g:+.6f}" for g in grads) + "]"
            line = f"    layer {i}:"
            for h, v in enumerate(vals):
                line += f"  h{h}={v:+.6f}"
            avg = sum(vals) / len(vals)
            line += f"  avg={avg:+.6f}{grad_str}"
            print(line)


# =============================================================================
# Data
# =============================================================================

def download_shakespeare():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    filepath = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(filepath):
        # Try shared location
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


@torch.no_grad()
def measure_attention_entropy(model, dataset):
    """Measure per-head attention entropy."""
    model.eval()
    x, _ = dataset.get_batch("val")
    B, T = x.size()
    head_dim = N_EMBD // N_HEAD

    tok_emb = F.embedding(x, model.wte)
    pos_emb = model.wpe[:T]
    emb = tok_emb + pos_emb
    hidden = emb.clone()

    all_entropies = []

    for block in model.blocks:
        normed = block.ln_1(hidden)
        q, k, v = block.attn.c_attn(normed).split(N_EMBD, dim=2)
        q = q.view(B, T, N_HEAD, head_dim).transpose(1, 2)
        k = k.view(B, T, N_HEAD, head_dim).transpose(1, 2)
        v = v.view(B, T, N_HEAD, head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        # Add harmonic bias if present
        if block.attn.lambda_param is not None:
            emb_heads = emb.view(B, T, N_HEAD, head_dim).transpose(1, 2)
            interference = (emb_heads @ emb_heads.transpose(-2, -1)) * scale
            scores = scores + block.attn.lambda_param * interference

        mask = block.attn.mask[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(scores, dim=-1)

        # Entropy: -sum(p * log(p))
        log_att = torch.log(att + 1e-10)
        entropy = -(att * log_att).sum(dim=-1)  # (B, N_HEAD, T)
        head_entropy = entropy.mean(dim=-1).mean(dim=0)  # (N_HEAD,)
        all_entropies.append(head_entropy.tolist())

        # Continue forward pass
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, N_EMBD)
        attn_out = block.attn.c_proj(y)
        hidden = hidden + attn_out
        hidden = hidden + block.mlp(block.ln_2(hidden))

    model.train()
    return all_entropies


def train_mode(mode_name, harmonic_bias, dataset):
    print(f"\n{'=' * 60}")
    print(f"  Training: {mode_name.upper()}")
    print(f"  Attention: {'standard + harmonic bias (lambda per head)' if harmonic_bias else 'standard (learned Q/K/V)'}")
    print(f"{'=' * 60}")

    model = BiasedGPT(dataset.vocab_size, harmonic_bias=harmonic_bias).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    history = []
    lambda_history = []  # Track lambda evolution
    start = time.time()

    for iter_num in range(MAX_ITERS):
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model, dataset)
            elapsed = time.time() - start
            print(f"  step {iter_num:>5} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {elapsed:.1f}s")

            if harmonic_bias:
                model.report_lambda()
                lambda_history.append((iter_num, model.get_lambda_values()))

            history.append((iter_num, losses["train"], losses["val"]))

        x, y = dataset.get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()

        # Check gradient on lambda after first backward pass
        if iter_num == 0 and harmonic_bias:
            print("\n  === GRADIENT CHECK (iter 0) ===")
            for i, block in enumerate(model.blocks):
                lam = block.attn.lambda_param
                if lam.grad is not None:
                    grads = lam.grad.flatten().tolist()
                    print(f"    layer {i} lambda grad: [{', '.join(f'{g:+.8f}' for g in grads)}]")
                    print(f"    lambda requires_grad: {lam.requires_grad}")
                    print(f"    lambda grad_fn: {lam.grad_fn}")
                else:
                    print(f"    layer {i} lambda grad: NONE (no gradient!)")
            print()

        optimizer.step()

    total = time.time() - start
    print(f"  Training complete in {total:.1f}s")

    # Attention entropy
    print("  Measuring attention entropy...")
    entropy = measure_attention_entropy(model, dataset)

    return {
        "mode_name": mode_name,
        "history": history,
        "entropy": entropy,
        "n_params": n_params,
        "lambda_history": lambda_history,
        "final_lambdas": model.get_lambda_values(),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  Phase 19b PyTorch Verification")
    print("  Does lambda actually learn? (candle showed it stuck at 0.1)")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text)} characters, {dataset.vocab_size} unique")
    print(f"  Train: {len(dataset.train_data)} | Val: {len(dataset.val_data)}")
    print(f"  Model: {N_LAYER} layers, {N_HEAD} heads, {N_EMBD} dim")

    print(f"\n  Harmonic bias architecture:")
    print(f"    score = Q·K^T/sqrt(d) + lambda * dot(emb_h, emb_h^T)/sqrt(d)")
    print(f"    lambda: learnable scalar per head per layer (init=0.1)")
    print(f"    Framework: PyTorch {torch.__version__}")

    print(f"\n  Question: Did candle fail to propagate gradients to lambda,")
    print(f"  or is there genuinely no useful gradient?")

    # Train both modes
    result_standard = train_mode("frozen_standard", False, dataset)
    result_biased = train_mode("harmonic_bias", True, dataset)

    results = [result_standard, result_biased]

    # =========================================================================
    # Comparison
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON: Final Validation Loss")
    print(f"{'=' * 60}")
    print()
    print(f"  {'Mode':<20} {'Val Loss':>10} {'Train Loss':>12} {'vs Std':>10} {'Params':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")

    std_val = result_standard["history"][-1][2]

    for result in results:
        _, train_l, val_l = result["history"][-1]
        if result["mode_name"] == "frozen_standard":
            diff = "—"
        else:
            pct = (1.0 - val_l / std_val) * 100.0
            diff = f"{pct:+.1f}%"
        print(f"  {result['mode_name']:<20} {val_l:>10.4f} {train_l:>12.4f} {diff:>10} {result['n_params']//1000:>7}K")

    # =========================================================================
    # Lambda Evolution
    # =========================================================================
    if result_biased["lambda_history"]:
        print(f"\n{'=' * 60}")
        print(f"  LAMBDA EVOLUTION ACROSS TRAINING")
        print(f"  (This is what candle couldn't show — all stayed at 0.100000)")
        print(f"{'=' * 60}")
        print()

        for step, lambdas_at_step in result_biased["lambda_history"]:
            print(f"  Step {step:>5}:")
            for layer_idx, layer_lambdas in enumerate(lambdas_at_step):
                line = f"    layer {layer_idx}:"
                for h, v in enumerate(layer_lambdas):
                    line += f"  h{h}={v:+.6f}"
                print(line)
            print()

    # =========================================================================
    # Final Lambda Analysis
    # =========================================================================
    if result_biased["final_lambdas"]:
        print(f"{'=' * 60}")
        print(f"  FINAL LAMBDA ANALYSIS")
        print(f"{'=' * 60}")
        print()

        all_lambdas = []
        init_val = 0.1
        for layer_idx, layer_lambdas in enumerate(result_biased["final_lambdas"]):
            for h, v in enumerate(layer_lambdas):
                all_lambdas.append(v)

        avg = sum(all_lambdas) / len(all_lambdas)
        max_l = max(all_lambdas)
        min_l = min(all_lambdas)
        moved = [abs(l - init_val) for l in all_lambdas]
        avg_movement = sum(moved) / len(moved)

        print(f"  Initial value: {init_val}")
        print(f"  Final average: {avg:+.6f}")
        print(f"  Final range:   [{min_l:+.6f}, {max_l:+.6f}]")
        print(f"  Avg movement from init: {avg_movement:.6f}")
        print()

        if avg_movement < 0.001:
            print(f"  VERDICT: Lambda barely moved (avg change {avg_movement:.6f})")
            print(f"  This matches candle — the gradient is genuinely near-zero.")
            print(f"  NOT a candle bug. The interference term provides no useful signal.")
        elif avg < 0.01:
            print(f"  VERDICT: Lambda learned to approach ZERO")
            print(f"  The model actively rejects the harmonic bias.")
            print(f"  Candle missed this because gradients didn't flow — but the")
            print(f"  conclusion is the same: harmonic bias doesn't help attention.")
        elif avg > init_val:
            print(f"  VERDICT: Lambda INCREASED from init — model wants more harmonic bias!")
            print(f"  This would mean candle's autograd limitation hid a real result.")
            print(f"  CHECK: Is val loss actually better than standard?")
        else:
            print(f"  VERDICT: Lambda decreased but stayed positive (avg={avg:.4f})")
            print(f"  The model partially uses the harmonic prior.")

    # =========================================================================
    # Convergence Speed
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"  CONVERGENCE: Val Loss at Each Checkpoint")
    print(f"{'=' * 60}")
    print()
    print(f"  {'Step':>6}  {'standard':>16}  {'harmonic_bias':>16}  {'Bias gain':>10}")
    print(f"  {'-' * 56}")

    for i in range(len(result_standard["history"])):
        step, _, std_v = result_standard["history"][i]
        _, _, bias_v = result_biased["history"][i]
        gain = (1.0 - bias_v / std_v) * 100.0
        print(f"  {step:>6}  {std_v:>16.4f}  {bias_v:>16.4f}  {gain:>+9.1f}%")

    # =========================================================================
    # Attention Entropy
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"  ATTENTION HEAD ENTROPY (lower = more specialised)")
    print(f"{'=' * 60}")

    for result in results:
        print(f"\n  {result['mode_name']}:")
        for layer, head_entropies in enumerate(result["entropy"]):
            avg = sum(head_entropies) / len(head_entropies)
            line = f"    layer {layer}:"
            for h, e in enumerate(head_entropies):
                line += f"  h{h}={e:.2f}"
            line += f"  avg={avg:.2f}"
            print(line)

    # =========================================================================
    # Cross-Framework Comparison
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"  CROSS-FRAMEWORK COMPARISON")
    print(f"{'=' * 60}")
    print()
    print(f"  Candle (Rust):  lambda stuck at +0.100000 for 2000 iters")
    print(f"                  val loss 3.1325 (1.1% worse than standard 3.0912)")
    print()

    biased_val = result_biased["history"][-1][2]
    biased_pct = (1.0 - biased_val / std_val) * 100.0
    final_lambdas = result_biased["final_lambdas"]
    if final_lambdas:
        all_l = [v for layer in final_lambdas for v in layer]
        avg_l = sum(all_l) / len(all_l)
        print(f"  PyTorch:        lambda final avg={avg_l:+.6f}")
        print(f"                  val loss {biased_val:.4f} ({biased_pct:+.1f}% vs standard {std_val:.4f})")
        print()

        movement = sum(abs(l - 0.1) for l in all_l) / len(all_l)
        if movement < 0.001:
            print(f"  CONCLUSION: Both frameworks agree. Lambda doesn't learn.")
            print(f"  This is NOT a candle bug — it's a property of the computation.")
            print(f"  Harmonic embedding dot products produce near-uniform scores,")
            print(f"  so the gradient of lambda is near-zero everywhere.")
        else:
            print(f"  CONCLUSION: PyTorch found different lambda values!")
            print(f"  Candle's autograd has a limitation with frozen tensors.")
            if biased_val < std_val:
                print(f"  AND the bias helps — Phase 19b finding needs revision.")
            else:
                print(f"  But the bias still doesn't help — conclusion unchanged.")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
