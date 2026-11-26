"""Single-lead 4-class ECG segmentation models (input [B,1,L], output logits [B,4,L]).

Baselines : UNet1D, CNNBiLSTM, PatchTransformer
Proposed  : UNetKAN   (KAN layers in bottleneck + classification head)          [book ch. 14]
            ConvKANNet (fully KAN-based conv encoder-decoder)                    [book ch. 14]
            DEQSeg    (weight-tied conv block solved to a fixed point, implicit gradients) [book ch. 16]
H2 controls: DEQSeg(mode="unrolled")  same block, K unrolled iterations, full backprop (same params)
             ExplicitStack            untied stack of blocks with a similar parameter count
L must be divisible by 16.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 4


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ============================================================================ building blocks
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True))

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
        return x, skips


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


# ============================================================================ baselines
class UNet1D(nn.Module):
    def __init__(self, ch=(8, 16, 32, 64, 64), k=9):
        super().__init__()
        self.enc = Encoder(ch, k)
        self.dec = Decoder(ch, k)
        self.head = nn.Conv1d(ch[0], N_CLASSES, 1)

    def forward(self, x):
        z, s = self.enc(x)
        return self.head(self.dec(z, s))


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
        h = F.interpolate(h.transpose(1, 2), size=x.shape[-1], mode="linear", align_corners=False)
        return self.head(self.refine(torch.cat([h, x], 1)))


class PatchTransformer(nn.Module):
    """Conv patch embedding (patch 8) -> Transformer encoder (relative position via conv PE) -> conv decoder."""

    def __init__(self, d=64, depth=4, heads=4, patch=8, c=16):
        super().__init__()
        self.patch = patch
        self.stem = ConvBlock(1, c, 9)
        self.embed = nn.Conv1d(c, d, patch, stride=patch)
        self.pos = nn.Conv1d(d, d, 7, padding=3, groups=d)  # conditional positional encoding (length-agnostic)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, depth)
        self.up = nn.ConvTranspose1d(d, c, patch, stride=patch)
        self.refine = ConvBlock(2 * c, c, 9)
        self.head = nn.Conv1d(c, N_CLASSES, 1)

    def forward(self, x):
        s = self.stem(x)
        t = self.embed(s)
        t = t + self.pos(t)
        t = self.tr(t.transpose(1, 2)).transpose(1, 2)
        return self.head(self.refine(torch.cat([self.up(t), s], 1)))


# ============================================================================ KAN
class KANLinear(nn.Module):
    """KAN layer: y_j = sum_i [ w_b[j,i] * silu(x_i) + sum_g c[j,i,g] * B_g(x_i) ].

    basis="bspline": cubic B-splines on a uniform grid over [-r, r] (efficient-KAN formulation, Liu et al. 2024).
    basis="rbf"    : Gaussian RBFs (FastKAN, Li 2024) - cheaper approximation of the B-spline basis.
    Inputs are squashed with a LayerNorm so they stay inside the grid.
    """

    def __init__(self, din, dout, grid=8, order=3, r=2.0, basis="bspline"):
        super().__init__()
        self.din, self.dout, self.grid, self.order, self.basis = din, dout, grid, order, basis
        self.norm = nn.LayerNorm(din)
        self.base = nn.Linear(din, dout)
        if basis == "bspline":
            h = 2 * r / grid
            knots = torch.arange(-order, grid + order + 1, dtype=torch.float32) * h - r
            self.register_buffer("knots", knots)
            nb = grid + order
        else:
            self.register_buffer("centers", torch.linspace(-r, r, grid))
            self.inv_w = (grid - 1) / (2 * r)
            nb = grid
        self.nb = nb
        self.coef = nn.Parameter(torch.randn(dout, din * nb) * (0.1 / math.sqrt(din * nb)))

    def basis_fn(self, x):  # x [..., din] -> [..., din, nb]
        if self.basis == "rbf":
            return torch.exp(-((x[..., None] - self.centers) * self.inv_w) ** 2)
        t = self.knots
        x = x[..., None]
        b = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)
        for k in range(1, self.order + 1):
            b = ((x - t[:-(k + 1)]) / (t[k:-1] - t[:-(k + 1)]) * b[..., :-1]
                 + (t[k + 1:] - x) / (t[k + 1:] - t[1:-k]) * b[..., 1:])
        return b

    def forward(self, x):
        x = self.norm(x)
        sp = self.basis_fn(x).flatten(-2)
        return self.base(F.silu(x)) + F.linear(sp, self.coef)

    @torch.no_grad()
    def univariate(self, i: int, j: int, xs: torch.Tensor) -> torch.Tensor:
        """phi_{j,i}(x) on normalised input values xs (for visualisation)."""
        b = self.basis_fn(xs[:, None].expand(-1, self.din))[:, i, :]          # [n, nb]
        c = self.coef.view(self.dout, self.din, self.nb)[j, i]                 # [nb]
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

    def __init__(self, ch=(8, 16, 32, 64, 64), k=9, grid=8, basis="bspline", where=("bottleneck", "head"), order=3,
                 mlp_head=0):
        super().__init__()
        self.where = where
        self.enc = Encoder(ch, k)
        self.dec = Decoder(ch, k)
        if "bottleneck" in where:
            self.kb = nn.Sequential(PointwiseKAN(ch[-1], ch[-1], grid=grid, basis=basis, order=order),
                                    PointwiseKAN(ch[-1], ch[-1], grid=grid, basis=basis, order=order))
        if "head" in where:
            self.head = PointwiseKAN(ch[0], N_CLASSES, grid=grid, basis=basis, order=order)
        elif mlp_head:
            self.head = MLPHead(ch[0], N_CLASSES, mlp_head)
        else:
            self.head = nn.Conv1d(ch[0], N_CLASSES, 1)

    def forward(self, x):
        z, s = self.enc(x)
        if "bottleneck" in self.where:
            z = z + self.kb(z)
        return self.head(self.dec(z, s))


class KANConv1d(nn.Module):
    """Convolutional KAN: each output = KANLinear over the flattened receptive field (k * Cin values)."""

    def __init__(self, cin, cout, k=5, stride=1, grid=6, basis="rbf"):
        super().__init__()
        self.k, self.stride = k, stride
        self.kan = KANLinear(cin * k, cout, grid=grid, basis=basis)

    def forward(self, x):
        p = F.pad(x, (self.k // 2, self.k // 2))
        u = p.unfold(2, self.k, self.stride)                     # [B, Cin, L', k]
        u = u.permute(0, 2, 1, 3).flatten(2)                     # [B, L', Cin*k]
        return self.kan(u).transpose(1, 2)


class ConvKANNet(nn.Module):
    """(b) Fully KAN-based compact encoder-decoder (KAN convolutions, no linear convolutions except resampling)."""

    def __init__(self, ch=(8, 16, 32, 32), k=5, grid=6, basis="rbf"):
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
            x = F.interpolate(x, size=s.shape[-1], mode="linear", align_corners=False)
            x = layer(torch.cat([x, s], 1))
        return self.head(x)


# ============================================================================ DEQ
def anderson(f, x0, m=5, lam=1e-4, max_iter=30, tol=1e-3, beta=1.0):
    """Anderson acceleration for z = f(z). x0: [B, ...]. Returns (z*, info)."""
    bsz = x0.shape[0]
    shape = x0.shape
    d = x0[0].numel()
    X = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
    Fv = torch.zeros_like(X)
    X[:, 0], Fv[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], Fv[:, 1] = Fv[:, 0], f(Fv[:, 0].reshape(shape)).reshape(bsz, -1)
    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1
    res = []
    k = 1
    for k in range(2, max_iter):
        n = min(k, m)
        G = Fv[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype,
                                                                                     device=x0.device)[None]
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]
        X[:, k % m] = beta * (alpha[:, None] @ Fv[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        Fv[:, k % m] = f(X[:, k % m].reshape(shape)).reshape(bsz, -1)
        r = ((Fv[:, k % m] - X[:, k % m]).norm(dim=1) / (1e-5 + Fv[:, k % m].norm(dim=1)))
        res.append(r)
        if r.max().item() < tol:
            break
    rel = torch.stack(res, 1) if res else torch.zeros(bsz, 1)
    # per-sample iteration count: first iteration at which that sample's residual < tol
    below = rel < tol
    it = torch.where(below.any(1), below.float().argmax(1) + 3, torch.full((bsz,), k + 1, dtype=torch.long,
                                                                           device=x0.device).float())
    return Fv[:, k % m].reshape(shape), {"rel_res": rel[:, -1].detach(), "n_iter": k + 1, "iters_per_sample": it}


class DEQBlock(nn.Module):
    """f(z, u) = GN( z + ReLU( GN( conv2( ReLU( GN( conv1(z) ) ) ) ) + u ) ) with weight normalisation."""

    def __init__(self, c=48, k=9, groups=4):
        super().__init__()
        self.c1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(c, c, k, padding=k // 2))
        self.c2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(c, c, k, padding=k // 2))
        self.n1, self.n2, self.n3 = nn.GroupNorm(groups, c), nn.GroupNorm(groups, c), nn.GroupNorm(groups, c)

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

    def __init__(self, c=48, k=9, mode="deq", max_iter=30, tol=1e-3, b_max_iter=30, unroll=8):
        super().__init__()
        self.mode, self.max_iter, self.tol, self.b_max_iter, self.unroll = mode, max_iter, tol, b_max_iter, unroll
        self.stem = ConvBlock(1, 16, 9)
        self.down = nn.Sequential(nn.MaxPool1d(2), ConvBlock(16, 32, 9), nn.MaxPool1d(2), ConvBlock(32, c, 9))
        self.inj = nn.Conv1d(c, c, 1)
        self.f = DEQBlock(c, k)
        self.up = nn.Sequential(nn.Upsample(scale_factor=4, mode="linear", align_corners=False),
                                nn.Conv1d(c, 16, 1))
        self.refine = ConvBlock(32, 16, 9)
        self.head = nn.Conv1d(16, N_CLASSES, 1)
        self.last_info: dict = {}

    def core(self, u):
        z0 = torch.zeros_like(u)
        if self.mode == "unrolled":
            z = z0
            for _ in range(self.unroll):
                z = self.f(z, u)
            with torch.no_grad():
                res = ((self.f(z, u) - z).flatten(1).norm(dim=1) / (1e-5 + z.flatten(1).norm(dim=1)))
            self.last_info = {"rel_res": res, "n_iter": self.unroll}
            return z
        with torch.no_grad():
            z, info = anderson(lambda zz: self.f(zz, u), z0, max_iter=self.max_iter, tol=self.tol)
        self.last_info = info
        if not self.training:
            return z
        if self.mode == "phantom":
            return self.f(z.detach(), u)
        z = self.f(z.detach().requires_grad_(), u)   # one differentiable step
        z0_ = z.detach().requires_grad_()
        f0 = self.f(z0_, u)

        def backward_hook(grad):
            # solve g = J^T g + grad by fixed-point iteration (implicit function theorem)
            g = grad
            for _ in range(self.b_max_iter):
                g_new = torch.autograd.grad(f0, z0_, g, retain_graph=True)[0] + grad
                if (g_new - g).norm() / (1e-8 + g_new.norm()) < self.tol:
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


class ExplicitStack(nn.Module):
    """H2 control: untied stack of `depth` DEQBlock-style residual blocks with a similar parameter count."""

    def __init__(self, c=48, k=9, depth=2):
        super().__init__()
        self.stem = ConvBlock(1, 16, 9)
        self.down = nn.Sequential(nn.MaxPool1d(2), ConvBlock(16, 32, 9), nn.MaxPool1d(2), ConvBlock(32, c, 9))
        self.inj = nn.Conv1d(c, c, 1)
        self.blocks = nn.ModuleList([DEQBlock(c, k) for _ in range(depth)])
        self.up = nn.Sequential(nn.Upsample(scale_factor=4, mode="linear", align_corners=False),
                                nn.Conv1d(c, 16, 1))
        self.refine = ConvBlock(32, 16, 9)
        self.head = nn.Conv1d(16, N_CLASSES, 1)

    def forward(self, x):
        s = self.stem(x)
        u = self.inj(self.down(s))
        z = torch.zeros_like(u)
        for b in self.blocks:
            z = b(z, u)
        return self.head(self.refine(torch.cat([self.up(z), s], 1)))


REGISTRY = {
    "unet": UNet1D, "cnn_bilstm": CNNBiLSTM, "transformer": PatchTransformer,
    "unet_kan": UNetKAN, "convkan": ConvKANNet,
    "deq": lambda **kw: DEQSeg(mode="deq", **kw), "deq_unrolled": lambda **kw: DEQSeg(mode="unrolled", **kw),
    "deq_phantom": lambda **kw: DEQSeg(mode="phantom", **kw), "explicit_stack": ExplicitStack,
}


def build(name: str, **kw) -> nn.Module:
    return REGISTRY[name](**kw)


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    x = torch.randn(8, 1, 1024)
    y = torch.randint(0, 4, (8, 1024))
    for nm in REGISTRY:
        m = build(nm)
        m.train()
        t = time.perf_counter()
        out = m(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        dt = time.perf_counter() - t
        extra = getattr(m, "last_info", {}).get("n_iter", "")
        print(f"{nm:15s} params={n_params(m):8d} out={tuple(out.shape)} fwd+bwd(8x1024)={dt:.3f}s iters={extra}")

