"""
Experiment 14: Learnable kernel weights

Current kernels use uniform weights: [1,1,0,1,1] or [1,1,1,1,0,1,1,1,1].
All neighbours contribute equally. But the non-monotonic result (9-band > 13-band)
suggests distant neighbours contribute less.

Let the optimizer learn the coupling profile:
- 5-band learnable: [w1,w2,0,w3,w4] — 4 params per layer
- 9-band learnable: [w1,w2,w3,w4,0,w5,w6,w7,w8] — 8 params per layer

Also test 9-band with distance-weighted init: closer bands start stronger.

If learnable weights close more gap, the uniform kernel was suboptimal.
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

class KerrODELayer(nn.Module):
    """Kerr-ODE with optionally learnable kernel weights."""
    def __init__(self, kernel_width=5, n_steps=8, learnable_kernel=False, distance_decay_init=False):
        super().__init__()
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps
        self.kernel_width = kernel_width
        self.learnable_kernel = learnable_kernel
        self._gamma_raw = nn.Parameter(torch.full((N_BANDS,), math.log(math.exp(0.1)-1)))
        self.omega = nn.Parameter(torch.arange(1, N_BANDS+1, dtype=torch.float32)/N_BANDS)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.out_proj = nn.Linear(N_EMBD, N_EMBD)

        half = kernel_width // 2
        self.padding = half

        if learnable_kernel:
            # Learnable weights, center is always 0 (self-coupling via alpha)
            init_weights = torch.ones(kernel_width)
            if distance_decay_init:
                # Closer bands start stronger: 1/distance
                for i in range(kernel_width):
                    d = abs(i - half)
                    if d > 0:
                        init_weights[i] = 1.0 / d
            init_weights[half] = 0.0
            self.kernel_weights = nn.Parameter(init_weights)
        else:
            k = torch.ones(kernel_width)
            k[half] = 0.0
            self.register_buffer('kernel_weights', k)

    @property
    def gamma(self): return F.softplus(self._gamma_raw)

    def _get_kernel(self):
        w = self.kernel_weights.clone()
        if self.learnable_kernel:
            # Enforce center = 0
            half = self.kernel_width // 2
            w = w * (1 - torch.zeros_like(w).scatter_(0, torch.tensor(half, device=w.device).unsqueeze(0), 1.0))
        return w.view(1, 1, self.kernel_width)

    def _derivative(self, r, s, gamma, kernel):
        mag_sq = r*r + s*s
        ns = F.conv1d(mag_sq.unsqueeze(1), kernel, padding=self.padding).squeeze(1)
        phi = self.omega + self.alpha*mag_sq + self.beta*ns
        return -gamma*r - phi*s, -gamma*s + phi*r

    def _rk4_step(self, r, s, dt, gamma, kernel):
        dr1,ds1 = self._derivative(r,s,gamma,kernel)
        dr2,ds2 = self._derivative(r+.5*dt*dr1, s+.5*dt*ds1, gamma,kernel)
        dr3,ds3 = self._derivative(r+.5*dt*dr2, s+.5*dt*ds2, gamma,kernel)
        dr4,ds4 = self._derivative(r+dt*dr3, s+dt*ds3, gamma,kernel)
        return r+(dt/6)*(dr1+2*dr2+2*dr3+dr4), s+(dt/6)*(ds1+2*ds2+2*ds3+ds4)

    def forward(self, x):
        B,T,C = x.size()
        bands = x.view(B*T, N_BANDS, 2)
        r,s = bands[:,:,0].contiguous(), bands[:,:,1].contiguous()
        dt, gamma = self.dt, self.gamma
        kernel = self._get_kernel()
        for _ in range(self.n_steps): r,s = self._rk4_step(r,s,dt,gamma,kernel)
        return self.out_proj(torch.stack([r,s],dim=2).reshape(B*T,C)).view(B,T,C)

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
    def __init__(self, vocab_size, kernel_width=5, learnable=False, decay_init=False, use_mlp=False):
        super().__init__()
        self.register_buffer("wte", build_harmonic_table(vocab_size, N_EMBD))
        self.register_buffer("wpe", build_positional_table(BLOCK_SIZE, N_EMBD))
        if use_mlp:
            layers = [Block(MLP()) for _ in range(N_LAYER)]
        else:
            layers = [Block(PerBandLinear())] + [
                Block(KerrODELayer(kernel_width=kernel_width, n_steps=8,
                                   learnable_kernel=learnable, distance_decay_init=decay_init))
                for _ in range(3)]
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

def train(model, dataset, label=""):
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

def report_kernels(model, label):
    """Print learned kernel weights."""
    print(f"\n  Learned kernel weights ({label}):")
    for name, p in model.named_parameters():
        if "kernel_weights" in name:
            w = p.detach().cpu()
            half = len(w) // 2
            w_list = [f"{v:.3f}" for v in w.tolist()]
            w_list[half] = "  0  "  # center is always 0
            print(f"    {name}: [{', '.join(w_list)}]")

def main():
    print("="*70)
    print("  Experiment 14: Learnable kernel weights")
    print(f"  Device: {DEVICE}")
    print("="*70)
    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"  Dataset: {len(text)} chars, {dataset.vocab_size} vocab\n")

    configs = [
        ("MLP baseline",        dict(use_mlp=True)),
        ("5-band fixed",        dict(kernel_width=5, learnable=False)),
        ("5-band learnable",    dict(kernel_width=5, learnable=True)),
        ("9-band fixed",        dict(kernel_width=9, learnable=False)),
        ("9-band learnable",    dict(kernel_width=9, learnable=True)),
        ("9-band decay init",   dict(kernel_width=9, learnable=True, decay_init=True)),
    ]

    results = {}
    for label, kwargs in configs:
        print(f"\n  --- {label} ---")
        torch.manual_seed(42)
        model = GPT(dataset.vocab_size, **kwargs).to(DEVICE)
        val = train(model, dataset, label=label)
        results[label] = val
        if "learnable" in label.lower() or "decay" in label.lower():
            report_kernels(model, label)

    mlp_val = results["MLP baseline"]

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    for label, _ in configs:
        val = results[label]
        if "MLP" in label:
            print(f"  {label:30s}  {val:.4f}")
        else:
            gap = (val / mlp_val - 1) * 100
            print(f"  {label:30s}  {val:.4f}  gap = {gap:+.2f}%")

    gap_5f = (results["5-band fixed"] / mlp_val - 1) * 100
    gap_5l = (results["5-band learnable"] / mlp_val - 1) * 100
    gap_9f = (results["9-band fixed"] / mlp_val - 1) * 100
    gap_9l = (results["9-band learnable"] / mlp_val - 1) * 100
    gap_9d = (results["9-band decay init"] / mlp_val - 1) * 100

    print(f"\n  Learnable vs fixed:")
    print(f"    5-band: {gap_5f:+.2f}% -> {gap_5l:+.2f}% ({gap_5f-gap_5l:.2f}pp)")
    print(f"    9-band: {gap_9f:+.2f}% -> {gap_9l:+.2f}% ({gap_9f-gap_9l:.2f}pp)")
    print(f"    9-band decay: {gap_9d:+.2f}% ({gap_9f-gap_9d:.2f}pp vs fixed)")
    print("="*70)

if __name__ == "__main__":
    main()
