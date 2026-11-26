"""Reviewed inference definitions from original models/evaluate/postprocess (CPU)."""

import math

import numpy as np

import torch

from torch import nn

from torch.nn import functional as F

N_CLASSES, WIN, HOP, MERGE_GAP = 4, 2048, 1024, 5

class ConvBlock(nn.Module):

    def __init__(self, cin, cout, k=9):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True), nn.Conv1d(cout, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):

    def __init__(self, ch, k=9):
        super().__init__()
        self.blocks = nn.ModuleList()
        cin = 1
        for c in ch:
            self.blocks.append(ConvBlock(cin, c, k))
            cin = c

    def forward(self, x):
        skips = []
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i < len(self.blocks) - 1:
                skips.append(x)
                x = F.max_pool1d(x, 2)
        return (x, skips)

class Decoder(nn.Module):

    def __init__(self, ch, k=9):
        super().__init__()
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose1d(ch[i], ch[i - 1], 2, stride=2))
            self.blocks.append(ConvBlock(2 * ch[i - 1], ch[i - 1], k))

    def forward(self, x, skips):
        for up, blk, s in zip(self.ups, self.blocks, reversed(skips)):
            x = blk(torch.cat([up(x), s], 1))
        return x

class UNet1D(nn.Module):
    def __init__(self, ch=(8,16,32,64,64), k=9):
        super().__init__()
        self.enc = Encoder(ch,k)
        self.dec = Decoder(ch,k)
        self.head = nn.Conv1d(ch[0],N_CLASSES,1)

    def forward(self,x):
        z, s = self.enc(x)
        return self.head(self.dec(z,s))


class CNNBiLSTM(nn.Module):
    """Conv stem at full rate, BiLSTM at 1/4 rate, upsample + conv refinement."""

    def __init__(self, c=32, hidden=64, layers=2):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(1, c, 9), nn.MaxPool1d(2), ConvBlock(c, c, 9), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(c, hidden, layers, batch_first=True, bidirectional=True)
        self.refine = ConvBlock(2 * hidden + 1, c, 9)
        self.head = nn.Conv1d(c, N_CLASSES, 1)

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)
        h, _ = self.lstm(h)
        h = F.interpolate(h.transpose(1, 2), size=x.shape[-1], mode='linear', align_corners=False)
        return self.head(self.refine(torch.cat([h, x], 1)))

class KANLinear(nn.Module):
    """KAN layer: y_j = sum_i [ w_b[j,i] * silu(x_i) + sum_g c[j,i,g] * B_g(x_i) ].

    basis="bspline": cubic B-splines on a uniform grid over [-r, r] (efficient-KAN formulation, Liu et al. 2024).
    basis="rbf"    : Gaussian RBFs (FastKAN, Li 2024) - cheaper approximation of the B-spline basis.
    Inputs are squashed with a LayerNorm so they stay inside the grid.
    """

    def __init__(self, din, dout, grid=8, order=3, r=2.0, basis='bspline'):
        super().__init__()
        self.din, self.dout, self.grid, self.order, self.basis = (din, dout, grid, order, basis)
        self.norm = nn.LayerNorm(din)
        self.base = nn.Linear(din, dout)
        if basis == 'bspline':
            h = 2 * r / grid
            knots = torch.arange(-order, grid + order + 1, dtype=torch.float32) * h - r
            self.register_buffer('knots', knots)
            nb = grid + order
        else:
            self.register_buffer('centers', torch.linspace(-r, r, grid))
            self.inv_w = (grid - 1) / (2 * r)
            nb = grid
        self.nb = nb
        self.coef = nn.Parameter(torch.randn(dout, din * nb) * (0.1 / math.sqrt(din * nb)))

    def basis_fn(self, x):
        if self.basis == 'rbf':
            return torch.exp(-((x[..., None] - self.centers) * self.inv_w) ** 2)
        t = self.knots
        x = x[..., None]
        b = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)
        for k in range(1, self.order + 1):
            b = (x - t[:-(k + 1)]) / (t[k:-1] - t[:-(k + 1)]) * b[..., :-1] + (t[k + 1:] - x) / (t[k + 1:] - t[1:-k]) * b[..., 1:]
        return b

    def forward(self, x):
        x = self.norm(x)
        sp = self.basis_fn(x).flatten(-2)
        return self.base(F.silu(x)) + F.linear(sp, self.coef)

    @torch.no_grad()
    def univariate(self, i: int, j: int, xs: torch.Tensor) -> torch.Tensor:
        """phi_{j,i}(x) on normalised input values xs (for visualisation)."""
        b = self.basis_fn(xs[:, None].expand(-1, self.din))[:, i, :]
        c = self.coef.view(self.dout, self.din, self.nb)[j, i]
        return self.base.weight[j, i] * F.silu(xs) + b @ c

class PointwiseKAN(nn.Module):
    """Applies a KANLinear over the channel dimension at every time step ([B,C,L] -> [B,C',L])."""

    def __init__(self, din, dout, **kw):
        super().__init__()
        self.kan = KANLinear(din, dout, **kw)

    def forward(self, x):
        return self.kan(x.transpose(1, 2)).transpose(1, 2)

class MLPHead(nn.Module):
    """Point-wise GELU MLP head (LayerNorm -> Linear -> GELU -> Linear); parameter-matched control for the KAN head
    (hidden 29 gives 397 parameters against 404 for the KAN head and 36 for the linear head)."""

    def __init__(self, din, dout, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(din)
        self.fc1 = nn.Linear(din, hidden)
        self.fc2 = nn.Linear(hidden, dout)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.norm(x.transpose(1, 2))))).transpose(1, 2)

class UNetKAN(nn.Module):
    """(a) U-Net with KAN bottleneck (2 pointwise KAN layers, residual) and a KAN classification head.
    where=() with mlp_head>0 gives the U-Net with a GELU MLP head (control)."""

    def __init__(self, ch=(8, 16, 32, 64, 64), k=9, grid=8, basis='bspline', where=('bottleneck', 'head'), order=3, mlp_head=0):
        super().__init__()
        self.where = where
        self.enc = Encoder(ch, k)
        self.dec = Decoder(ch, k)
        if 'bottleneck' in where:
            self.kb = nn.Sequential(PointwiseKAN(ch[-1], ch[-1], grid=grid, basis=basis, order=order), PointwiseKAN(ch[-1], ch[-1], grid=grid, basis=basis, order=order))
        if 'head' in where:
            self.head = PointwiseKAN(ch[0], N_CLASSES, grid=grid, basis=basis, order=order)
        elif mlp_head:
            self.head = MLPHead(ch[0], N_CLASSES, mlp_head)
        else:
            self.head = nn.Conv1d(ch[0], N_CLASSES, 1)

    def forward(self, x):
        z, s = self.enc(x)
        if 'bottleneck' in self.where:
            z = z + self.kb(z)
        return self.head(self.dec(z, s))

class KANConv1d(nn.Module):
    """Convolutional KAN: each output = KANLinear over the flattened receptive field (k * Cin values)."""

    def __init__(self, cin, cout, k=5, stride=1, grid=6, basis='rbf'):
        super().__init__()
        self.k, self.stride = (k, stride)
        self.kan = KANLinear(cin * k, cout, grid=grid, basis=basis)

    def forward(self, x):
        p = F.pad(x, (self.k // 2, self.k // 2))
        u = p.unfold(2, self.k, self.stride)
        u = u.permute(0, 2, 1, 3).flatten(2)
        return self.kan(u).transpose(1, 2)

class ConvKANNet(nn.Module):
    """(b) Fully KAN-based compact encoder-decoder (KAN convolutions, no linear convolutions except resampling)."""

    def __init__(self, ch=(8, 16, 32, 32), k=5, grid=6, basis='rbf'):
        super().__init__()
        self.down = nn.ModuleList()
        cin = 1
        for c in ch:
            self.down.append(KANConv1d(cin, c, k, grid=grid, basis=basis))
            cin = c
        self.up = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.up.append(KANConv1d(ch[i] + ch[i - 1], ch[i - 1], k, grid=grid, basis=basis))
        self.head = KANConv1d(ch[0], N_CLASSES, 1, grid=grid, basis=basis)

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.down):
            x = layer(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = F.avg_pool1d(x, 4 if i == 0 else 2)
        for layer, s in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=s.shape[-1], mode='linear', align_corners=False)
            x = layer(torch.cat([x, s], 1))
        return self.head(x)

def anderson(f, x0, m=5, lam=0.0001, max_iter=30, tol=0.001, beta=1.0):
    """Anderson acceleration for z = f(z). x0: [B, ...]. Returns (z*, info)."""
    bsz = x0.shape[0]
    shape = x0.shape
    d = x0[0].numel()
    X = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
    Fv = torch.zeros_like(X)
    X[:, 0], Fv[:, 0] = (x0.reshape(bsz, -1), f(x0).reshape(bsz, -1))
    X[:, 1], Fv[:, 1] = (Fv[:, 0], f(Fv[:, 0].reshape(shape)).reshape(bsz, -1))
    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1
    res = []
    k = 1
    for k in range(2, max_iter):
        n = min(k, m)
        G = Fv[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]
        X[:, k % m] = beta * (alpha[:, None] @ Fv[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        Fv[:, k % m] = f(X[:, k % m].reshape(shape)).reshape(bsz, -1)
        r = (Fv[:, k % m] - X[:, k % m]).norm(dim=1) / (1e-05 + Fv[:, k % m].norm(dim=1))
        res.append(r)
        if r.max().item() < tol:
            break
    rel = torch.stack(res, 1) if res else torch.zeros(bsz, 1)
    below = rel < tol
    it = torch.where(below.any(1), below.float().argmax(1) + 3, torch.full((bsz,), k + 1, dtype=torch.long, device=x0.device).float())
    return (Fv[:, k % m].reshape(shape), {'rel_res': rel[:, -1].detach(), 'n_iter': k + 1, 'iters_per_sample': it})

class DEQBlock(nn.Module):
    """f(z, u) = GN( z + ReLU( GN( conv2( ReLU( GN( conv1(z) ) ) ) ) + u ) ) with weight normalisation."""

    def __init__(self, c=48, k=9, groups=4):
        super().__init__()
        self.c1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(c, c, k, padding=k // 2))
        self.c2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(c, c, k, padding=k // 2))
        self.n1, self.n2, self.n3 = (nn.GroupNorm(groups, c), nn.GroupNorm(groups, c), nn.GroupNorm(groups, c))

    def forward(self, z, u):
        y = self.n1(F.relu(self.c1(z)))
        y = self.n2(self.c2(y) + u)
        return self.n3(F.relu(z + y))

class DEQSeg(nn.Module):
    """Encoder (to 1/4 rate) -> DEQ core z* = f(z*, u) -> decoder with skip.

    mode = "deq"      : Anderson forward without grad, implicit backward (fixed-point iteration on the adjoint).
    mode = "unrolled" : K plain iterations with full backprop (same parameters; memory grows with K).
    mode = "phantom"  : DEQ forward + 1-step phantom gradient (cheapest backward, Geng et al. 2021).
    """

    def __init__(self, c=48, k=9, mode='deq', max_iter=30, tol=0.001, b_max_iter=30, unroll=8):
        super().__init__()
        self.mode, self.max_iter, self.tol, self.b_max_iter, self.unroll = (mode, max_iter, tol, b_max_iter, unroll)
        self.stem = ConvBlock(1, 16, 9)
        self.down = nn.Sequential(nn.MaxPool1d(2), ConvBlock(16, 32, 9), nn.MaxPool1d(2), ConvBlock(32, c, 9))
        self.inj = nn.Conv1d(c, c, 1)
        self.f = DEQBlock(c, k)
        self.up = nn.Sequential(nn.Upsample(scale_factor=4, mode='linear', align_corners=False), nn.Conv1d(c, 16, 1))
        self.refine = ConvBlock(32, 16, 9)
        self.head = nn.Conv1d(16, N_CLASSES, 1)
        self.last_info: dict = {}

    def core(self, u):
        z0 = torch.zeros_like(u)
        if self.mode == 'unrolled':
            z = z0
            for _ in range(self.unroll):
                z = self.f(z, u)
            with torch.no_grad():
                res = (self.f(z, u) - z).flatten(1).norm(dim=1) / (1e-05 + z.flatten(1).norm(dim=1))
            self.last_info = {'rel_res': res, 'n_iter': self.unroll}
            return z
        with torch.no_grad():
            z, info = anderson(lambda zz: self.f(zz, u), z0, max_iter=self.max_iter, tol=self.tol)
        self.last_info = info
        if not self.training:
            return z
        if self.mode == 'phantom':
            return self.f(z.detach(), u)
        z = self.f(z.detach().requires_grad_(), u)
        z0_ = z.detach().requires_grad_()
        f0 = self.f(z0_, u)

        def backward_hook(grad):
            g = grad
            for _ in range(self.b_max_iter):
                g_new = torch.autograd.grad(f0, z0_, g, retain_graph=True)[0] + grad
                if (g_new - g).norm() / (1e-08 + g_new.norm()) < self.tol:
                    g = g_new
                    break
                g = g_new
            return g
        if z.requires_grad:
            z.register_hook(backward_hook)
        return z

    def forward(self, x):
        s = self.stem(x)
        u = self.inj(self.down(s))
        z = self.core(u)
        return self.head(self.refine(torch.cat([self.up(z), s], 1)))

@torch.no_grad()
def predict_proba(models, x: np.ndarray) -> np.ndarray:
    """models: list of nn.Module (ensemble, probabilities averaged). x: 1-D float32. Returns [4, n]."""
    n = len(x)
    L = max(16, int(np.ceil(min(n, WIN) / 16) * 16))
    if n <= WIN:
        xp = np.zeros(L, np.float32)
        xp[:n] = x
        t = torch.from_numpy(xp)[None, None]
        p = sum((torch.softmax(m(t), 1)[0] for m in models)) / len(models)
        return p[:, :n].numpy()
    acc = np.zeros((4, n), np.float32)
    w = np.zeros(n, np.float32)
    taper = np.hanning(WIN).astype(np.float32) + 0.001
    starts = list(range(0, n - WIN + 1, HOP))
    if starts[-1] != n - WIN:
        starts.append(n - WIN)
    batch = 16
    for b in range(0, len(starts), batch):
        ss = starts[b:b + batch]
        t = torch.from_numpy(np.stack([x[s:s + WIN] for s in ss]))[:, None]
        p = (sum((torch.softmax(m(t), 1) for m in models)) / len(models)).numpy()
        for j, s in enumerate(ss):
            acc[:, s:s + WIN] += p[j] * taper
            w[s:s + WIN] += taper
    return acc / w

def runs(mask: np.ndarray, cls: int) -> list[tuple[int, int]]:
    m = np.concatenate([[False], mask == cls, [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))

def merge(rs, gap=MERGE_GAP):
    out = []
    for s, e in rs:
        if out and s - out[-1][1] - 1 < gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out

def peak_of(x: np.ndarray, s: int, e: int) -> int:
    seg = x[s:e + 1]
    if len(seg) < 3:
        return s + len(seg) // 2
    base = np.linspace(seg[0], seg[-1], len(seg))
    return s + int(np.argmax(np.abs(seg - base)))
