from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
N_CLASSES = 4

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)

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
