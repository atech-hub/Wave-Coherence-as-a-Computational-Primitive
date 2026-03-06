"""
Quick follow-up: Maestro 7L (1+6) and fair-param comparison.

Results so far:
  Kerr 7L (1+6):     -1.90% vs MLP 4L  (591K params)
  Maestro 6L (1+5):  -1.18% vs MLP 4L  (529K params)

Questions:
1. Does Maestro 7L beat Kerr 7L?
2. Fair comparison: MLP 4L has 801K params. Kerr 7L has 591K.
   What does MLP at similar param count (e.g., 6L MLP) achieve?
"""

import math, os, time, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BANDS = 64
N_EMBD = 128
N_HEAD = 4
BLOCK_SIZE = 256
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_ITERS = 2000
EVAL_INTERVAL = 200
EVAL_ITERS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAESTRO_DIM = 16

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
    def __init__(self):
        super().__init__()
        self.squeeze = nn.Linear(N_EMBD, MAESTRO_DIM)
        self.process = nn.Sequential(nn.GELU(), nn.Linear(MAESTRO_DIM, N_EMBD))
    def forward(self, x_flat):
        return self.process(self.squeeze(x_flat))

class KerrStandard(nn.Module):
    def __init__(self):
        super().__init__()
        self.kerr = KerrODE(n_steps=8)
        self.out_proj = nn.Linear(N_EMBD, N_EMBD)
    def forward(self, x):
        B,T,C = x.size()
        out = self.kerr(x.view(B*T, N_BANDS, 2))
        return self.out_proj(out.reshape(B*T, C)).view(B, T, C)

class KerrMaestroAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.kerr = KerrODE(n_steps=8)
        self.maestro = Maestro()
        self.out_proj = nn.Linear(N_EMBD, N_EMBD)
    def forward(self, x):
        B,T,C = x.size()
        x_flat = x.view(B*T, C)
        kerr_out = self.kerr(x_flat.view(B*T, N_BANDS, 2)).reshape(B*T, C)
        return self.out_proj(kerr_out + self.maestro(x_flat)).view(B, T, C)

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
        B,T,C = x.size(); hd = C//4
        q,k,v = self.c_attn(x).split(N_EMBD, dim=2)
        q=q.view(B,T,4,hd).transpose(1,2); k=k.view(B,T,4,hd).transpose(1,2); v=v.view(B,T,4,hd).transpose(1,2)
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
    def __init__(self, vocab_size, n_layers=4, ffn_type="mlp", use_maestro=False):
        super().__init__()
        self.register_buffer("wte", build_harmonic_table(vocab_size, N_EMBD))
        self.register_buffer("wpe", build_positional_table(BLOCK_SIZE, N_EMBD))
        if ffn_type == "mlp":
            layers = [Block(MLP()) for _ in range(n_layers)]
        elif use_maestro:
            layers = [Block(PerBandLinear())] + [Block(KerrMaestroAdd()) for _ in range(n_layers - 1)]
        else:
            layers = [Block(PerBandLinear())] + [Block(KerrStandard()) for _ in range(n_layers - 1)]
        self.blocks = nn.ModuleList(layers)
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.apply(self._init_weights)
        for pn,p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*n_layers))
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

def main():
    print("="*70)
    print("  Follow-up: Maestro 7L + MLP depth comparison")
    print(f"  Device: {DEVICE}")
    print("="*70)
    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"  Dataset: {len(text)} chars, {dataset.vocab_size} vocab\n")

    configs = [
        ("MLP 4L",           dict(n_layers=4, ffn_type="mlp")),
        ("MLP 6L",           dict(n_layers=6, ffn_type="mlp")),
        ("MLP 7L",           dict(n_layers=7, ffn_type="mlp")),
        ("Maestro 7L (1+6)", dict(n_layers=7, ffn_type="kerr", use_maestro=True)),
    ]

    results = {}
    for label, kwargs in configs:
        print(f"\n  --- {label} ---")
        torch.manual_seed(42)
        model = GPT(dataset.vocab_size, **kwargs).to(DEVICE)
        val = train(model, dataset, label=label)
        results[label] = val

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    mlp4 = results["MLP 4L"]
    for label, _ in configs:
        val = results[label]
        n_params = {"MLP 4L": 801664, "MLP 6L": 1063424, "MLP 7L": 1194304, "Maestro 7L (1+6)": 612444}
        gap = (val / mlp4 - 1) * 100
        print(f"  {label:25s}  {val:.4f}  vs MLP4L: {gap:+.2f}%")

    print(f"\n  From Experiment 15:")
    print(f"    Kerr 7L (1+6):    1.6666  vs MLP4L: -1.90%  (591K params)")
    print(f"    Maestro 6L (1+5): 1.6788  vs MLP4L: -1.18%  (529K params)")
    print(f"\n  The question: does MLP also improve with more depth?")
    print(f"  If yes, the Kerr advantage may be about depth, not architecture.")
    print("="*70)

if __name__ == "__main__":
    main()
