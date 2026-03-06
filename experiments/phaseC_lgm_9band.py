"""
Experiment 17: 9-band kernel + LGM combination

Two best interventions combined:
- 9-band kernel: 3.96% gap (closed 0.92pp via wider local reach)
- LGM-before: 4.14% gap (closed 0.74pp via global spectral gating)

These are orthogonal: wider kernel extends local nonlinear reach,
LGM adds global spectral information via multiplicative fusion.
If they stack, sub-3.5% is possible.
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

class SpectralGlobal(nn.Module):
    def __init__(self):
        super().__init__()
        n_freq = N_BANDS // 2 + 1
        self.W_real = nn.Parameter(torch.ones(n_freq))
        self.W_imag = nn.Parameter(torch.zeros(n_freq))
    def forward(self, x_flat):
        r, s = x_flat[:,:,0], x_flat[:,:,1]
        R = torch.fft.rfft(r, dim=1)
        S = torch.fft.rfft(s, dim=1)
        W = torch.complex(self.W_real, self.W_imag)
        R_filt = R * W
        S_filt = S * W
        r_out = torch.fft.irfft(R_filt, n=N_BANDS, dim=1)
        s_out = torch.fft.irfft(S_filt, n=N_BANDS, dim=1)
        return torch.stack([r_out, s_out], dim=2)

class KerrODELayer(nn.Module):
    def __init__(self, kernel_width=5, n_steps=8):
        super().__init__()
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps
        self._gamma_raw = nn.Parameter(torch.full((N_BANDS,), math.log(math.exp(0.1)-1)))
        self.omega = nn.Parameter(torch.arange(1, N_BANDS+1, dtype=torch.float32)/N_BANDS)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        half = kernel_width // 2
        k = torch.ones(kernel_width)
        k[half] = 0.0
        self.register_buffer('neighbor_kernel', k.view(1, 1, kernel_width))
        self.padding = half
    @property
    def gamma(self): return F.softplus(self._gamma_raw)
    def _derivative(self, r, s, gamma):
        mag_sq = r*r + s*s
        ns = F.conv1d(mag_sq.unsqueeze(1), self.neighbor_kernel, padding=self.padding).squeeze(1)
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

class KerrStandard(nn.Module):
    def __init__(self, kernel_width=5, n_steps=8):
        super().__init__()
        self.kerr = KerrODELayer(kernel_width=kernel_width, n_steps=n_steps)
        self.out_proj = nn.Linear(N_EMBD, N_EMBD)
    def forward(self, x):
        B,T,C = x.size()
        bands = x.view(B*T, N_BANDS, 2)
        out = self.kerr(bands)
        return self.out_proj(out.reshape(B*T, C)).view(B, T, C)

class KerrLGMBefore(nn.Module):
    def __init__(self, kernel_width=5, n_steps=8):
        super().__init__()
        self.kerr = KerrODELayer(kernel_width=kernel_width, n_steps=n_steps)
        self.spectral = SpectralGlobal()
        self.out_proj = nn.Linear(N_EMBD, N_EMBD)
    def forward(self, x):
        B,T,C = x.size()
        bands = x.view(B*T, N_BANDS, 2)
        local = self.kerr(bands)
        global_ = self.spectral(bands)
        fused = local * global_ + local
        return self.out_proj(fused.reshape(B*T, C)).view(B, T, C)

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
        elif mode == "kerr_9":
            layers = [Block(PerBandLinear())] + [Block(KerrStandard(kernel_width=9, n_steps=8)) for _ in range(3)]
        elif mode == "kerr_9_lgm":
            layers = [Block(PerBandLinear())] + [Block(KerrLGMBefore(kernel_width=9, n_steps=8)) for _ in range(3)]
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
    print("  Experiment 17: 9-band kernel + LGM combination")
    print(f"  Device: {DEVICE}")
    print("="*70)
    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"  Dataset: {len(text)} chars, {dataset.vocab_size} vocab\n")

    print("  --- MLP baseline ---")
    mlp_val = train("mlp", dataset, label="MLP")

    print(f"\n  --- 9-band Kerr (standard) ---")
    kerr9_val = train("kerr_9", dataset, label="Kerr 9-band")

    print(f"\n  --- 9-band Kerr + LGM ---")
    lgm9_val = train("kerr_9_lgm", dataset, label="Kerr 9-band + LGM")

    gap_9 = (kerr9_val / mlp_val - 1) * 100
    gap_lgm9 = (lgm9_val / mlp_val - 1) * 100

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  MLP baseline:          {mlp_val:.4f}")
    print(f"  9-band Kerr:           {kerr9_val:.4f}  gap = {gap_9:+.2f}%")
    print(f"  9-band Kerr + LGM:     {lgm9_val:.4f}  gap = {gap_lgm9:+.2f}%")
    print(f"\n  Reference points:")
    print(f"    5-band Kerr flat:        +4.88%")
    print(f"    5-band + LGM:            +4.14%  (LGM alone: 0.74pp)")
    print(f"    9-band Kerr flat:        +3.96%  (wider kernel: 0.92pp)")
    print(f"    9-band + LGM:            {gap_lgm9:+.2f}%")
    print(f"    5-band + curriculum:     +3.42%  (Phase B)")

    stacked = gap_9 - gap_lgm9
    print(f"\n  LGM contribution on 9-band: {stacked:.2f}pp")

    if gap_lgm9 < 3.42:
        print(f"\n  VERDICT: Beats Phase B curriculum! New ceiling.")
    elif gap_lgm9 < gap_9 - 0.3:
        print(f"\n  VERDICT: LGM stacks with wider kernel. Interventions are orthogonal.")
    else:
        print(f"\n  VERDICT: LGM doesn't stack — diminishing returns with wider kernel.")
    print("="*70)

if __name__ == "__main__":
    main()
