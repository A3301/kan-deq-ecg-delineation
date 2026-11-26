import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
N_CLASSES=4
FS=250
WIN=2048
HOP=1024
MERGE_GAP=5

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

def match(ref: np.ndarray, det: np.ndarray, tol: int):
    ref = np.asarray(ref, dtype=int)
    det = np.asarray(det, dtype=int)
    if len(ref) == 0 or len(det) == 0:
        return (np.zeros(0, int), np.zeros(0, int), len(ref), len(det))
    d = np.abs(det[None, :] - ref[:, None])
    ii, jj = np.nonzero(d <= tol)
    order = np.argsort(d[ii, jj], kind='stable')
    used_r, used_d, pairs = (set(), set(), [])
    for k in order:
        i, j = (ii[k], jj[k])
        if i in used_r or j in used_d:
            continue
        used_r.add(i)
        used_d.add(j)
        pairs.append((i, j))
    if not pairs:
        return (np.zeros(0, int), np.zeros(0, int), len(ref), len(det))
    pr = np.array([p[0] for p in pairs])
    pd_ = np.array([p[1] for p in pairs])
    return (ref[pr], det[pd_], len(ref) - len(pairs), len(det) - len(pairs))