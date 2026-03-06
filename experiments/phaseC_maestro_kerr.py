"""
Experiment 18: Maestro architecture — squeeze-and-excitation for Kerr-ODE

The bottleneck: Kerr-ODE has local coupling (5-9 bands).
MLP has global coupling (all 64 bands).
Everything tested (wider kernel, FFT, LGM) tried flat global coupling.

Maestro: hierarchical global coordination through a compressed bottleneck.
Like squeeze-and-excitation from computer vision (Hu et al., 2018).

    1. Gather: pool all 64 bands into a small vector (mean/learned, O(N))
    2. Process: small nonlinear transform (tiny MLP, O(bottleneck^2))
    3. Broadcast: modulate each band by the processed global signal (O(N))

Total cost: O(N) regardless of band count. At 64 or 4096 bands,
the maestro is the same size.

Three variants:
1. Maestro-Add: global signal added to Kerr output
2. Maestro-Mult: global signal multiplies Kerr output (SE-style)
3. Maestro-Gate: sigmoid gate from global signal modulates Kerr output
"""

import math, os, time, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BANDS = 64
N_EMBD = 128
N_LAYER = 4
N_HEAD = 4
BLOCK_SIZE = 256
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_ITERS = 2000
EVAL_INTERVAL = 200
EVAL_ITERS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAESTRO_DIM = 16  # bottleneck dimension

def build_harmonic_table(vocab_size, n_embd):
    nh = n_embd // 2
    scale = 1.0 / math.sqrt(nh)
    t = torch.zeros(vocab_size, n_embd)
    for c in range(vocab_size):
        theta = c * 2.0 * math.pi / vocab_size
        for h in range(nh):
            t[c, h*2] = math.cos((h+1)*theta) * scale
            t[c, h*2+1] = math.sin((h+1)*theta) * scale
    return t

def build_positional_table(max_len, n_embd):
    nh = n_embd // 2
    scale = 1.0 / math.sqrt(nh)
    t = torch.zeros(max_len, n_embd)
    for pos in range(max_len):
        for h in range(nh):
            freq = 1.0 / (10000.0 ** (2.0*h/n_embd))
            t[pos, h*2] = math.cos(pos*freq) * scale
            t[pos, h*2+1] = math.sin(pos*freq) * scale
    return t

class PerBandLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_w = nn.Parameter(torch.zeros(N_BANDS, 2, 2))
        with torch.no_grad():
            for k in range(N_BANDS): self.band_w.data[k] = torch.eye(2)
        self.band_b = nn.Parameter(torch.zeros(N_BANDS, 2))
        self.out_proj = nn.Linear(N_EMBD, N_EMBD)
    def forward(self, x):
        B,T,C = x.size()
        bands = x.view(B*T, N_BANDS, 2)
        out = torch.einsum('bni,nij->bnj', bands, self.band_w) + self.band_b
        return self.out_proj(out.reshape(B*T, C)).view(B, T, C)

class KerrODE(nn.Module):
    """Raw Kerr-ODE — no out_proj. Returns (B*T, N_BANDS, 2)."""
    def __init__(self, n_steps=8):
        super().__init__()
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps
        self._gamma_raw = nn.Parameter(torch.full((N_BANDS,), math.log(math.exp(0.1)-1)))
        self.omega = nn.Parameter(torch.arange(1, N_BANDS+1, dtype=torch.float32)/N_BANDS)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.register_buffer('neighbor_kernel', torch.tensor([[[1.,1.,0.,1.,1.]]]))
    @property
    def gamma(self): return F.softplus(self._gamma_raw)
    def _derivative(self, r, s, gamma):
        mag_sq = r*r + s*s
        ns = F.conv1d(mag_sq.unsqueeze(1), self.neighbor_kernel, padding=2).squeeze(1)
        phi = self.omega + self.alpha*mag_sq + self.beta*ns
        return -gamma*r - phi*s, -gamma*s + phi*r
    def _rk4_step(self, r, s, dt, gamma):
        dr1,ds1 = self._derivative(r,s,gamma)
        dr2,ds2 = self._derivative(r+.5*dt*dr1, s+.5*dt*ds1, gamma)
        dr3,ds3 = self._derivative(r+.5*dt*dr2, s+.5*dt*ds2, gamma)
        dr4,ds4 = self._derivative(r+dt*dr3, s+dt*ds3, gamma)
        return r+(dt/6)*(dr1+2*dr2+2*dr3+dr4), s+(dt/6)*(ds1+2*ds2+2*ds3+ds4)
    def forward(self, x_flat):
        r, s = x_flat[:,:,0].contiguous(), x_flat[:,:,1].contiguous()
        dt, gamma = self.dt, self.gamma
        for _ in range(self.n_steps): r,s = self._rk4_step(r,s,dt,gamma)
        return torch.stack([r,s], dim=2)


class Maestro(nn.Module):
    """Squeeze-and-excitation bottleneck for band coordination."""
    def __init__(self, mode="gate"):
        super().__init__()
        self.mode = mode
        # Gather: project N_EMBD -> MAESTRO_DIM
        self.squeeze = nn.Linear(N_EMBD, MAESTRO_DIM)
        # Process: nonlinear transform
        self.process = nn.Sequential(
            nn.GELU(),
            nn.Linear(MAESTRO_DIM, N_EMBD),
        )
        if mode == "gate":
            # Sigmoid for gating
            self.gate_act = nn.Sigmoid()
    def forward(self, x_flat):
        """x_flat: (B*T, N_EMBD). Returns modulation signal (B*T, N_EMBD)."""
        # Gather: global pool is just the input itself (already a band vector)
        h = self.squeeze(x_flat)          # (B*T, MAESTRO_DIM)
        h = self.process(h)               # (B*T, N_EMBD)
        if self.mode == "gate":
            return self.gate_act(h)       # (B*T, N_EMBD) in [0,1]
        return h


class KerrStandard(nn.Module):
    """Standard Kerr-ODE with out_proj (baseline)."""
    def __init__(self):
        super().__init__()
        self.kerr = KerrODE(n_steps=8)
        self.out_proj = nn.Linear(N_EMBD, N_EMBD)
    def forward(self, x):
        B,T,C = x.size()
        bands = x.view(B*T, N_BANDS, 2)
        out = self.kerr(bands)
        return self.out_proj(out.reshape(B*T, C)).view(B, T, C)


class KerrMaestroAdd(nn.Module):
    """Kerr + Maestro additive: output = out_proj(kerr_out) + maestro(input)."""
    def __init__(self):
        super().__init__()
        self.kerr = KerrODE(n_steps=8)
        self.maestro = Maestro(mode="add")
        self.out_proj = nn.Linear(N_EMBD, N_EMBD)
    def forward(self, x):
        B,T,C = x.size()
        x_flat = x.view(B*T, C)
        bands = x_flat.view(B*T, N_BANDS, 2)
        kerr_out = self.kerr(bands).reshape(B*T, C)
        global_signal = self.maestro(x_flat)
        return self.out_proj(kerr_out + global_signal).view(B, T, C)


class KerrMaestroMult(nn.Module):
    """Kerr + Maestro multiplicative: output = out_proj(kerr_out * maestro(input))."""
    def __init__(self):
        super().__init__()
        self.kerr = KerrODE(n_steps=8)
        self.maestro = Maestro(mode="mult")
        self.out_proj = nn.Linear(N_EMBD, N_EMBD)
    def forward(self, x):
        B,T,C = x.size()
        x_flat = x.view(B*T, C)
        bands = x_flat.view(B*T, N_BANDS, 2)
        kerr_out = self.kerr(bands).reshape(B*T, C)
        global_signal = self.maestro(x_flat)
        return self.out_proj(kerr_out * global_signal).view(B, T, C)


class KerrMaestroGate(nn.Module):
    """Kerr + Maestro gated: output = out_proj(kerr_out * sigmoid(maestro(input)))."""
    def __init__(self):
        super().__init__()
        self.kerr = KerrODE(n_steps=8)
        self.maestro = Maestro(mode="gate")
        self.out_proj = nn.Linear(N_EMBD, N_EMBD)
    def forward(self, x):
        B,T,C = x.size()
        x_flat = x.view(B*T, C)
        bands = x_flat.view(B*T, N_BANDS, 2)
        kerr_out = self.kerr(bands).reshape(B*T, C)
        gate = self.maestro(x_flat)       # (B*T, N_EMBD) in [0,1]
        return self.out_proj(kerr_out * gate).view(B, T, C)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4*N_EMBD)
        self.c_proj = nn.Linear(4*N_EMBD, N_EMBD)
    def forward(self, x): return self.c_proj(F.gelu(self.c_fc(x)))

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3*N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE,BLOCK_SIZE)).view(1,1,BLOCK_SIZE,BLOCK_SIZE))
    def forward(self, x):
        B,T,C = x.size(); hd = C//N_HEAD
        q,k,v = self.c_attn(x).split(N_EMBD, dim=2)
        q=q.view(B,T,N_HEAD,hd).transpose(1,2); k=k.view(B,T,N_HEAD,hd).transpose(1,2); v=v.view(B,T,N_HEAD,hd).transpose(1,2)
        att = (q@k.transpose(-2,-1)) * (1.0/math.sqrt(hd))
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float("-inf"))
        return self.c_proj((F.softmax(att,dim=-1)@v).transpose(1,2).contiguous().view(B,T,C))

class Block(nn.Module):
    def __init__(self, ffn):
        super().__init__()
        self.ln_1=nn.LayerNorm(N_EMBD); self.attn=CausalSelfAttention()
        self.ln_2=nn.LayerNorm(N_EMBD); self.ffn=ffn
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)); x = x + self.ffn(self.ln_2(x)); return x

class GPT(nn.Module):
    def __init__(self, vocab_size, mode="mlp"):
        super().__init__()
        self.register_buffer("wte", build_harmonic_table(vocab_size, N_EMBD))
        self.register_buffer("wpe", build_positional_table(BLOCK_SIZE, N_EMBD))
        if mode == "mlp":
            layers = [Block(MLP()) for _ in range(N_LAYER)]
        elif mode == "kerr":
            layers = [Block(PerBandLinear())] + [Block(KerrStandard()) for _ in range(3)]
        elif mode == "maestro_add":
            layers = [Block(PerBandLinear())] + [Block(KerrMaestroAdd()) for _ in range(3)]
        elif mode == "maestro_mult":
            layers = [Block(PerBandLinear())] + [Block(KerrMaestroMult()) for _ in range(3)]
        elif mode == "maestro_gate":
            layers = [Block(PerBandLinear())] + [Block(KerrMaestroGate()) for _ in range(3)]
        self.blocks = nn.ModuleList(layers)
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.apply(self._init_weights)
        for pn,p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*N_LAYER))
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias); nn.init.ones_(m.weight)
    def forward(self, idx, targets=None):
        B,T = idx.size()
        x = F.embedding(idx, self.wte) + self.wpe[:T]
        for block in self.blocks: x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

def download_shakespeare():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "data")
    filepath = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(filepath):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python", "data", "shakespeare.txt")
        if os.path.exists(alt): filepath = alt
        else: os.makedirs(data_dir, exist_ok=True); urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", filepath)
    with open(filepath, "r") as f: return f.read()

class Dataset:
    def __init__(self, text):
        self.chars = sorted(list(set(text))); self.vocab_size = len(self.chars)
        self.stoi = {c:i for i,c in enumerate(self.chars)}
        data = [self.stoi[c] for c in text]; n = int(0.9*len(data))
        self.train_data = torch.tensor(data[:n], dtype=torch.long)
        self.val_data = torch.tensor(data[n:], dtype=torch.long)
    def get_batch(self, split):
        data = self.train_data if split=="train" else self.val_data
        ix = torch.randint(len(data)-BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model, dataset):
    model.eval(); out = {}
    for split in ["train","val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x,y = dataset.get_batch(split); _,loss = model(x,y); losses[k]=loss.item()
        out[split] = losses.mean().item()
    model.train(); return out

def train(mode, dataset, label=""):
    torch.manual_seed(42)
    model = GPT(dataset.vocab_size, mode=mode).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {label}: {n_params:,} params")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    start = time.time()
    for i in range(MAX_ITERS):
        if i % EVAL_INTERVAL == 0 or i == MAX_ITERS-1:
            losses = estimate_loss(model, dataset)
            print(f"    step {i:>5} | val {losses['val']:.4f}")
        x,y = dataset.get_batch("train"); _,loss = model(x,y)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    final = estimate_loss(model, dataset)
    print(f"    Done in {time.time()-start:.1f}s")
    return final["val"]

def main():
    print("="*70)
    print("  Experiment 18: Maestro architecture")
    print(f"  Device: {DEVICE}")
    print(f"  Maestro bottleneck: {MAESTRO_DIM}D")
    print("="*70)
    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"  Dataset: {len(text)} chars, {dataset.vocab_size} vocab\n")

    configs = [
        ("mlp",           "MLP baseline"),
        ("kerr",          "Kerr standard"),
        ("maestro_add",   "Kerr + Maestro (add)"),
        ("maestro_mult",  "Kerr + Maestro (mult)"),
        ("maestro_gate",  "Kerr + Maestro (gate)"),
    ]

    results = {}
    for mode, label in configs:
        print(f"\n  --- {label} ---")
        val = train(mode, dataset, label=label)
        results[mode] = val

    mlp_val = results["mlp"]
    kerr_gap = (results["kerr"] / mlp_val - 1) * 100

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    for mode, label in configs:
        val = results[mode]
        if mode == "mlp":
            print(f"  {label:35s}  {val:.4f}")
        else:
            gap = (val / mlp_val - 1) * 100
            print(f"  {label:35s}  {val:.4f}  gap = {gap:+.2f}%")

    best_maestro = min(results["maestro_add"], results["maestro_mult"], results["maestro_gate"])
    best_gap = (best_maestro / mlp_val - 1) * 100
    closed = kerr_gap - best_gap

    print(f"\n  Reference:")
    print(f"    Kerr standard:       {kerr_gap:+.2f}%")
    print(f"    9-band kernel:       +3.96%  (closed 0.92pp)")
    print(f"    5-band + curriculum: +3.13%  (closed 1.75pp)")
    print(f"    Best Maestro:        {best_gap:+.2f}%  (closed {closed:.2f}pp)")

    if best_gap < 3.13:
        print(f"\n  VERDICT: Maestro beats everything. New ceiling!")
    elif best_gap < 3.96:
        print(f"\n  VERDICT: Maestro competitive with best interventions.")
    elif closed > 0.5:
        print(f"\n  VERDICT: Maestro helps but doesn't beat wider kernel.")
    else:
        print(f"\n  VERDICT: Maestro has negligible effect.")
    print("="*70)

if __name__ == "__main__":
    main()
