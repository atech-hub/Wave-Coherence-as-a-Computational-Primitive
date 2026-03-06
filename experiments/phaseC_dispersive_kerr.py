"""
Experiment 11: Dispersive Kerr-ODE

The Lugiato-Lefever equation includes a dispersion term that was dropped
when adapting to the Kerr-ODE. Can adding it back close the locality gap?

Three dispersive variants vs standard Kerr and MLP at 64 bands flat:

1. Quadratic phase: D_k = -beta2*(k/N)^2 added to phi_k
   (per-band modification — optimizer may already have this freedom via omega_k)

2. Band Laplacian: beta2 * (Z_{k+1} - 2*Z_k + Z_{k-1})
   (second derivative in band space — 3-band coupling on complex amplitudes)

3. FFT dispersion: FFT across bands, multiply by dispersion relation, IFFT back
   (global coupling — O(N log N) for full-spectrum reach)

Hypothesis: #1 is null (freedom already exists), #2 is minor (still local),
#3 closes more gap than the 9-band kernel did (0.92pp).
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
    """Standard Kerr-ODE with 5-band kernel (baseline)."""
    def __init__(self, n_steps=8):
        super().__init__()
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps
        self._gamma_raw = nn.Parameter(torch.full((N_BANDS,), math.log(math.exp(0.1)-1)))
        self.omega = nn.Parameter(torch.arange(1, N_BANDS+1, dtype=torch.float32)/N_BANDS)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.out_proj = nn.Linear(N_EMBD, N_EMBD)
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
    def forward(self, x):
        B,T,C = x.size()
        bands = x.view(B*T, N_BANDS, 2)
        r,s = bands[:,:,0].contiguous(), bands[:,:,1].contiguous()
        dt, gamma = self.dt, self.gamma
        for _ in range(self.n_steps): r,s = self._rk4_step(r,s,dt,gamma)
        return self.out_proj(torch.stack([r,s],dim=2).reshape(B*T,C)).view(B,T,C)


class KerrQuadDispersion(nn.Module):
    """Kerr-ODE + per-band quadratic dispersion term."""
    def __init__(self, n_steps=8):
        super().__init__()
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps
        self._gamma_raw = nn.Parameter(torch.full((N_BANDS,), math.log(math.exp(0.1)-1)))
        self.omega = nn.Parameter(torch.arange(1, N_BANDS+1, dtype=torch.float32)/N_BANDS)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.beta2 = nn.Parameter(torch.tensor(0.1))  # dispersion strength
        self.out_proj = nn.Linear(N_EMBD, N_EMBD)
        self.register_buffer('neighbor_kernel', torch.tensor([[[1.,1.,0.,1.,1.]]]))
        # Quadratic dispersion profile: D_k = (k/N)^2
        ks = torch.arange(1, N_BANDS+1, dtype=torch.float32) / N_BANDS
        self.register_buffer('disp_profile', ks * ks)
    @property
    def gamma(self): return F.softplus(self._gamma_raw)
    def _derivative(self, r, s, gamma):
        mag_sq = r*r + s*s
        ns = F.conv1d(mag_sq.unsqueeze(1), self.neighbor_kernel, padding=2).squeeze(1)
        # phi includes quadratic dispersion
        phi = self.omega + self.alpha*mag_sq + self.beta*ns + self.beta2*self.disp_profile
        return -gamma*r - phi*s, -gamma*s + phi*r
    def _rk4_step(self, r, s, dt, gamma):
        dr1,ds1 = self._derivative(r,s,gamma)
        dr2,ds2 = self._derivative(r+.5*dt*dr1, s+.5*dt*ds1, gamma)
        dr3,ds3 = self._derivative(r+.5*dt*dr2, s+.5*dt*ds2, gamma)
        dr4,ds4 = self._derivative(r+dt*dr3, s+dt*ds3, gamma)
        return r+(dt/6)*(dr1+2*dr2+2*dr3+dr4), s+(dt/6)*(ds1+2*ds2+2*ds3+ds4)
    def forward(self, x):
        B,T,C = x.size()
        bands = x.view(B*T, N_BANDS, 2)
        r,s = bands[:,:,0].contiguous(), bands[:,:,1].contiguous()
        dt, gamma = self.dt, self.gamma
        for _ in range(self.n_steps): r,s = self._rk4_step(r,s,dt,gamma)
        return self.out_proj(torch.stack([r,s],dim=2).reshape(B*T,C)).view(B,T,C)


class KerrLaplacian(nn.Module):
    """Kerr-ODE + band-space Laplacian (second derivative coupling on complex amplitudes)."""
    def __init__(self, n_steps=8):
        super().__init__()
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps
        self._gamma_raw = nn.Parameter(torch.full((N_BANDS,), math.log(math.exp(0.1)-1)))
        self.omega = nn.Parameter(torch.arange(1, N_BANDS+1, dtype=torch.float32)/N_BANDS)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.beta2 = nn.Parameter(torch.tensor(0.1))  # Laplacian strength
        self.out_proj = nn.Linear(N_EMBD, N_EMBD)
        self.register_buffer('neighbor_kernel', torch.tensor([[[1.,1.,0.,1.,1.]]]))
        # Laplacian kernel: [1, -2, 1]
        self.register_buffer('laplacian_kernel', torch.tensor([[[1., -2., 1.]]]))
    @property
    def gamma(self): return F.softplus(self._gamma_raw)
    def _derivative(self, r, s, gamma):
        mag_sq = r*r + s*s
        ns = F.conv1d(mag_sq.unsqueeze(1), self.neighbor_kernel, padding=2).squeeze(1)
        phi = self.omega + self.alpha*mag_sq + self.beta*ns
        dr_base = -gamma*r - phi*s
        ds_base = -gamma*s + phi*r
        # Laplacian acts on complex amplitudes directly (dispersive coupling)
        lap_r = F.conv1d(r.unsqueeze(1), self.laplacian_kernel, padding=1).squeeze(1)
        lap_s = F.conv1d(s.unsqueeze(1), self.laplacian_kernel, padding=1).squeeze(1)
        # Dispersion: i * beta2 * laplacian(Z) => adds -beta2*lap_s to dr, +beta2*lap_r to ds
        return dr_base - self.beta2*lap_s, ds_base + self.beta2*lap_r
    def _rk4_step(self, r, s, dt, gamma):
        dr1,ds1 = self._derivative(r,s,gamma)
        dr2,ds2 = self._derivative(r+.5*dt*dr1, s+.5*dt*ds1, gamma)
        dr3,ds3 = self._derivative(r+.5*dt*dr2, s+.5*dt*ds2, gamma)
        dr4,ds4 = self._derivative(r+dt*dr3, s+dt*ds3, gamma)
        return r+(dt/6)*(dr1+2*dr2+2*dr3+dr4), s+(dt/6)*(ds1+2*ds2+2*ds3+ds4)
    def forward(self, x):
        B,T,C = x.size()
        bands = x.view(B*T, N_BANDS, 2)
        r,s = bands[:,:,0].contiguous(), bands[:,:,1].contiguous()
        dt, gamma = self.dt, self.gamma
        for _ in range(self.n_steps): r,s = self._rk4_step(r,s,dt,gamma)
        return self.out_proj(torch.stack([r,s],dim=2).reshape(B*T,C)).view(B,T,C)


class KerrFFTDispersion(nn.Module):
    """Kerr-ODE + FFT-based global dispersion across bands.

    Transform Z across bands via FFT, multiply by learnable dispersion relation,
    IFFT back. This gives every band information about every other band,
    weighted by frequency distance. O(N log N) for global reach.
    """
    def __init__(self, n_steps=8):
        super().__init__()
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps
        self._gamma_raw = nn.Parameter(torch.full((N_BANDS,), math.log(math.exp(0.1)-1)))
        self.omega = nn.Parameter(torch.arange(1, N_BANDS+1, dtype=torch.float32)/N_BANDS)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.beta2 = nn.Parameter(torch.tensor(0.1))  # global dispersion strength
        self.out_proj = nn.Linear(N_EMBD, N_EMBD)
        self.register_buffer('neighbor_kernel', torch.tensor([[[1.,1.,0.,1.,1.]]]))
        # Dispersion relation in FFT space: -k^2 weighting
        # FFT of N_BANDS points gives N_BANDS//2+1 unique frequencies
        n_freq = N_BANDS // 2 + 1
        ks = torch.arange(n_freq, dtype=torch.float32)
        # Normalized quadratic: -(k/N)^2 so it doesn't blow up
        self.register_buffer('disp_relation', -(ks / N_BANDS) ** 2)
    @property
    def gamma(self): return F.softplus(self._gamma_raw)
    def _fft_dispersion(self, r, s):
        """Apply dispersion via FFT across band dimension.
        r, s: (batch, N_BANDS)
        Returns dispersive contribution to dr, ds.
        """
        # FFT across band dimension (real FFT since r,s are real)
        R = torch.fft.rfft(r, dim=1)  # (batch, N_BANDS//2+1) complex
        S = torch.fft.rfft(s, dim=1)
        # Multiply by dispersion relation (real multiplier)
        R_disp = R * self.disp_relation
        S_disp = S * self.disp_relation
        # IFFT back to band space
        r_disp = torch.fft.irfft(R_disp, n=N_BANDS, dim=1)
        s_disp = torch.fft.irfft(S_disp, n=N_BANDS, dim=1)
        return r_disp, s_disp
    def _derivative(self, r, s, gamma):
        mag_sq = r*r + s*s
        ns = F.conv1d(mag_sq.unsqueeze(1), self.neighbor_kernel, padding=2).squeeze(1)
        phi = self.omega + self.alpha*mag_sq + self.beta*ns
        dr_base = -gamma*r - phi*s
        ds_base = -gamma*s + phi*r
        # FFT dispersion: i * beta2 * D(Z) where D is the dispersion operator
        r_disp, s_disp = self._fft_dispersion(r, s)
        # i * beta2 * (r_disp + i*s_disp) = beta2*(-s_disp + i*r_disp)
        return dr_base - self.beta2*s_disp, ds_base + self.beta2*r_disp
    def _rk4_step(self, r, s, dt, gamma):
        dr1,ds1 = self._derivative(r,s,gamma)
        dr2,ds2 = self._derivative(r+.5*dt*dr1, s+.5*dt*ds1, gamma)
        dr3,ds3 = self._derivative(r+.5*dt*dr2, s+.5*dt*ds2, gamma)
        dr4,ds4 = self._derivative(r+dt*dr3, s+dt*ds3, gamma)
        return r+(dt/6)*(dr1+2*dr2+2*dr3+dr4), s+(dt/6)*(ds1+2*ds2+2*ds3+ds4)
    def forward(self, x):
        B,T,C = x.size()
        bands = x.view(B*T, N_BANDS, 2)
        r,s = bands[:,:,0].contiguous(), bands[:,:,1].contiguous()
        dt, gamma = self.dt, self.gamma
        for _ in range(self.n_steps): r,s = self._rk4_step(r,s,dt,gamma)
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
    def __init__(self, vocab_size, mode="mlp"):
        super().__init__()
        self.register_buffer("wte", build_harmonic_table(vocab_size, N_EMBD))
        self.register_buffer("wpe", build_positional_table(BLOCK_SIZE, N_EMBD))
        if mode == "mlp":
            layers = [Block(MLP()) for _ in range(N_LAYER)]
        elif mode == "kerr":
            layers = [Block(PerBandLinear())] + [Block(KerrODELayer(n_steps=8)) for _ in range(3)]
        elif mode == "quad_disp":
            layers = [Block(PerBandLinear())] + [Block(KerrQuadDispersion(n_steps=8)) for _ in range(3)]
        elif mode == "laplacian":
            layers = [Block(PerBandLinear())] + [Block(KerrLaplacian(n_steps=8)) for _ in range(3)]
        elif mode == "fft_disp":
            layers = [Block(PerBandLinear())] + [Block(KerrFFTDispersion(n_steps=8)) for _ in range(3)]
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
    elapsed = time.time()-start
    print(f"    Done in {elapsed:.1f}s")
    return final["val"], elapsed

def main():
    print("="*70)
    print("  Experiment 11: Dispersive Kerr-ODE")
    print(f"  Device: {DEVICE}")
    print("="*70)
    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"  Dataset: {len(text)} chars, {dataset.vocab_size} vocab\n")

    configs = [
        ("mlp",       "MLP baseline"),
        ("kerr",      "Kerr 5-band (standard)"),
        ("quad_disp", "Kerr + quadratic dispersion"),
        ("laplacian", "Kerr + band Laplacian"),
        ("fft_disp",  "Kerr + FFT dispersion"),
    ]

    results = {}
    for mode, label in configs:
        print(f"\n  --- {label} ---")
        val, elapsed = train(mode, dataset, label=label)
        results[mode] = (val, elapsed)

    mlp_val = results["mlp"][0]

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    for mode, label in configs:
        val, elapsed = results[mode]
        if mode == "mlp":
            print(f"  {label:40s}  {val:.4f}  ({elapsed:.0f}s)")
        else:
            gap = (val / mlp_val - 1) * 100
            print(f"  {label:40s}  {val:.4f}  gap = {gap:+.2f}%  ({elapsed:.0f}s)")

    kerr_gap = (results["kerr"][0] / mlp_val - 1) * 100
    fft_gap = (results["fft_disp"][0] / mlp_val - 1) * 100
    closed = kerr_gap - fft_gap

    print(f"\n  Reference: 9-band kernel closed 0.92pp (4.88% -> 3.96%)")
    print(f"  FFT dispersion closed: {closed:.2f}pp ({kerr_gap:.2f}% -> {fft_gap:.2f}%)")

    if fft_gap < kerr_gap - 0.5:
        if fft_gap < 3.96:
            print(f"\n  VERDICT: FFT dispersion beats wider kernel. Global coupling > local widening.")
        else:
            print(f"\n  VERDICT: FFT dispersion helps but doesn't beat 9-band kernel.")
    elif fft_gap > kerr_gap + 0.5:
        print(f"\n  VERDICT: FFT dispersion hurts. The coupling mechanism doesn't help.")
    else:
        print(f"\n  VERDICT: FFT dispersion has negligible effect.")
    print("="*70)

if __name__ == "__main__":
    main()
