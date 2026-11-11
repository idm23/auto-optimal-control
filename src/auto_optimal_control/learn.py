"""Learned proposals for an :class:`~auto_optimal_control.embedding.ExtremalSet` (needs PyTorch).

Install with the ``learn`` extra: ``pip install auto-optimal-control[learn]``.

Every model here is trained on generated *indirect* extremals and only ever
proposes a starting point: the costates ``lam0`` and the time ``t_f`` for a
query ``(x0, xf)``. A proposal is turned into an extremal by
:class:`~auto_optimal_control.generation.ExtremalGenerator` (which finds its
arcs) and into the answer by
:class:`~auto_optimal_control.embedding.Polisher` (which solves exactly and
checks). A wrong proposal costs time, never a wrong answer.

- :class:`ConditionalFlow`: a normalizing flow of affine couplings, the density
  of ``(lam0, log t_f)`` given the endpoints. Several extremals can join the
  same endpoints, so the inverse map is many-valued; a density represents that
  where a regression would average the branches.
- :class:`ExtremalJEPA`: a joint-embedding predictive architecture. A context
  encoder embeds ``(x0, lam0, t_f)``, a target encoder (an exponential moving
  average of an online one) embeds ``xf``, and a predictor maps the first to
  the second. It is a forward model in a learned space; a query inverts it by
  gradient descent on the costates, from any starting proposal.
"""

import numpy as np
import torch
from torch import nn


def _mlp(n_in, n_out, width, depth):
    layers, n = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(n, width), nn.SiLU()]
        n = width
    layers.append(nn.Linear(n, n_out))
    return nn.Sequential(*layers)


class Standardize:
    """Affine maps to zero mean and unit variance per column, and back."""

    def __init__(self, A):
        A = np.asarray(A, dtype=float)
        self.mean = A.mean(axis=0)
        self.std = np.maximum(A.std(axis=0), 1e-12)

    def __call__(self, A):
        return (np.asarray(A, dtype=float) - self.mean) / self.std

    def inverse(self, Z):
        return np.asarray(Z, dtype=float) * self.std + self.mean


def training_data(extremals, index):
    """``(C, Y)``: conditions (endpoint features of ``index``) and targets ``(lam0, log t_f)``.

    ``lam0`` is divided by its norm: on ``H = 0`` its scale follows from its
    direction (:meth:`~auto_optimal_control.generation.ExtremalGenerator.on_level_set`)."""
    C = index.features(extremals.x0, extremals.xf)
    lam = extremals.lam0 / np.linalg.norm(extremals.lam0, axis=1, keepdims=True)
    Y = np.hstack([lam, np.log(extremals.tf - extremals.t0)[:, None]])
    return C, Y


def costates_of(y):
    """``(lam0 direction, t_f - t0)`` from a target row."""
    y = np.asarray(y, dtype=float)
    return y[:-1], float(np.exp(y[-1]))


# =============================================================================
# A conditional normalizing flow
# =============================================================================

class _Coupling(nn.Module):
    def __init__(self, dim, n_cond, mask, width, depth):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = _mlp(dim + n_cond, 2 * dim, width, depth)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, y, c):
        """``y -> z`` and ``log |dz/dy|``."""
        keep = y * self.mask
        s, t = self.net(torch.cat([keep, c], dim=1)).chunk(2, dim=1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        z = keep + (1 - self.mask) * (y * torch.exp(s) + t)
        return z, s.sum(dim=1)

    def inverse(self, z, c):
        keep = z * self.mask
        s, t = self.net(torch.cat([keep, c], dim=1)).chunk(2, dim=1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        return keep + (1 - self.mask) * ((z - t) * torch.exp(-s))


class ConditionalFlow(nn.Module):
    """``p(y | c)``: affine couplings over a standard normal, conditioned on ``c``.

    Between couplings the coordinates are permuted (fixed random permutations),
    so every coordinate is transformed given every other."""

    def __init__(self, dim, n_cond, *, n_layers=8, width=128, depth=2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.dim = dim
        self.layers = nn.ModuleList()
        self.perms = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % 2::2] = 1.0
            self.layers.append(_Coupling(dim, n_cond, mask, width, depth))
            self.perms.append(torch.randperm(dim, generator=g))
        self.cond_net = _mlp(n_cond, n_cond, width, 1)

    def log_prob(self, y, c):
        h = self.cond_net(c)
        logdet = torch.zeros(len(y))
        z = y
        for layer, perm in zip(self.layers, self.perms):
            z, ld = layer(z[:, perm], h)
            logdet = logdet + ld
        base = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.dim * np.log(2 * np.pi)
        return base + logdet

    @torch.no_grad()
    def sample(self, c, n, seed=None):
        """``n`` samples of ``y`` for one condition row ``c``."""
        g = None if seed is None else torch.Generator().manual_seed(seed)
        c = torch.as_tensor(np.asarray(c, dtype=np.float32)).reshape(1, -1).repeat(n, 1)
        h = self.cond_net(c)
        z = torch.randn(n, self.dim, generator=g)
        for layer, perm in zip(reversed(self.layers), reversed(self.perms)):
            y = layer.inverse(z, h)
            inv = torch.argsort(perm)
            z = y[:, inv]
        return z.numpy()


def train_flow(C, Y, *, epochs=300, batch=256, lr=2e-3, seed=0, verbose=False, **kwargs):
    """A :class:`ConditionalFlow` fitted by maximum likelihood; returns ``(flow, sc, sy)``,
    with the standardizations of the conditions and targets."""
    torch.manual_seed(seed)
    sc, sy = Standardize(C), Standardize(Y)
    Ct = torch.as_tensor(sc(C), dtype=torch.float32)
    Yt = torch.as_tensor(sy(Y), dtype=torch.float32)
    flow = ConditionalFlow(Y.shape[1], C.shape[1], seed=seed, **kwargs)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for epoch in range(epochs):
        perm = torch.randperm(len(Yt))
        total = 0.0
        for i in range(0, len(Yt), batch):
            idx = perm[i:i + batch]
            loss = -flow.log_prob(Yt[idx], Ct[idx]).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
        sched.step()
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"flow epoch {epoch}: nll {total / len(Yt):.3f}")
    flow.eval()
    return flow, sc, sy


def flow_proposals(flow, sc, sy, cond, n, seed=None):
    """``n`` proposals ``y`` (unstandardized) for one condition row, most likely first."""
    ys = flow.sample(sc(cond[None])[0], n, seed=seed)
    with torch.no_grad():
        lp = flow.log_prob(torch.as_tensor(ys, dtype=torch.float32),
                           torch.as_tensor(np.repeat(sc(cond[None]), n, axis=0), dtype=torch.float32))
    order = np.argsort(-lp.numpy())
    return sy.inverse(ys[order])


# =============================================================================
# A joint-embedding predictive architecture
# =============================================================================

class ExtremalJEPA(nn.Module):
    """Predict the embedding of where an extremal ends from its start and costates.

    ``context(s, y)`` embeds the start features ``s`` and the costates ``y``;
    ``target(e)`` embeds the end features ``e``; ``predictor`` maps the first
    embedding to the second. The target encoder used in the loss is an
    exponential moving average of the online one, with a stop-gradient, and a
    variance-covariance term (as in VICReg) keeps the embeddings from collapsing.
    """

    def __init__(self, n_start, n_y, n_end, *, dim=64, width=256, depth=3):
        super().__init__()
        self.context = _mlp(n_start + n_y, dim, width, depth)
        self.target = _mlp(n_end, dim, width, depth)
        self.ema = _mlp(n_end, dim, width, depth)
        self.ema.load_state_dict(self.target.state_dict())
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.predictor = _mlp(dim, dim, width, 2)

    def predict(self, s, y):
        return self.predictor(self.context(torch.cat([s, y], dim=1)))

    @torch.no_grad()
    def update_ema(self, decay):
        for pe, pt in zip(self.ema.parameters(), self.target.parameters()):
            pe.mul_(decay).add_(pt, alpha=1 - decay)


def _variance_covariance(z):
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var = torch.relu(1 - std).mean()
    cov = (z.T @ z) / max(len(z) - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return var + (off ** 2).sum() / z.shape[1]


def train_jepa(S, Y, E, *, epochs=300, batch=256, lr=1e-3, decay=0.996, reg=1.0, seed=0,
               verbose=False, **kwargs):
    """An :class:`ExtremalJEPA` on starts ``S``, costates ``Y`` and ends ``E``;
    returns ``(model, ss, sy, se)`` with the standardizations."""
    torch.manual_seed(seed)
    ss, sy, se = Standardize(S), Standardize(Y), Standardize(E)
    St, Yt, Et = (torch.as_tensor(f(A), dtype=torch.float32) for f, A in ((ss, S), (sy, Y), (se, E)))
    model = ExtremalJEPA(S.shape[1], Y.shape[1], E.shape[1], **kwargs)
    params = list(model.context.parameters()) + list(model.target.parameters()) \
        + list(model.predictor.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for epoch in range(epochs):
        perm = torch.randperm(len(Yt))
        total = 0.0
        for i in range(0, len(Yt), batch):
            idx = perm[i:i + batch]
            pred = model.predict(St[idx], Yt[idx])
            with torch.no_grad():
                goal = model.ema(Et[idx])
            online = model.target(Et[idx])
            # The prediction matches the slow target; the online target encoder learns
            # through the regularizer and its pull to the prediction, and feeds the EMA.
            loss = ((pred - goal) ** 2).mean() + ((online - pred.detach()) ** 2).mean() \
                + reg * _variance_covariance(online)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            model.update_ema(decay)
            total += float(loss.detach()) * len(idx)
        sched.step()
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"jepa epoch {epoch}: loss {total / len(Yt):.4f}")
    model.eval()
    return model, ss, sy, se


def jepa_refine(model, ss, sy, se, s, e, y_init, *, steps=200, lr=0.05):
    """Costates ``y`` from ``y_init`` (rows) moved to make the predicted embedding
    match the goal's; returns them unstandardized, best (lowest mismatch) first."""
    st = torch.as_tensor(ss(np.atleast_2d(s)), dtype=torch.float32)
    with torch.no_grad():
        goal = model.ema(torch.as_tensor(se(np.atleast_2d(e)), dtype=torch.float32))
    y = torch.as_tensor(sy(np.atleast_2d(y_init)), dtype=torch.float32).clone().requires_grad_(True)
    opt = torch.optim.Adam([y], lr=lr)
    S = st.repeat(len(y), 1)
    G = goal.repeat(len(y), 1)
    for _ in range(steps):
        loss = ((model.predict(S, y) - G) ** 2).sum(dim=1)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        final = ((model.predict(S, y) - G) ** 2).sum(dim=1).numpy()
    order = np.argsort(final)
    return sy.inverse(y.detach().numpy()[order]), final[order]


# =============================================================================
# A quasimetric of time between states
# =============================================================================

class Quasimetric(nn.Module):
    """``d(x, y)``: a learned estimate of the least time from state ``x`` to state ``y``.

    The least time is a *quasimetric*: ``d(x, x) = 0`` and the triangle
    inequality ``d(x, z) <= d(x, y) + d(y, z)`` hold, with equality exactly
    when ``y`` lies on a fastest path from ``x`` to ``z`` (Bellman), but it is
    not symmetric. Here ``d(x, y) = |f(y) - f(x)| + max_i relu(g_i(y) - g_i(x))``,
    a symmetric norm plus an asymmetric part (the form of metric residual
    networks), which satisfies both for every value of the weights, so they
    need not be learned. States enter as features ``s`` (the rest of the
    state) and ``p`` (coordinates along which the problem is invariant under
    translation, e.g. a joint about gravity's axis); ``p`` enters ``f`` and
    ``g`` linearly, so only ``p(y) - p(x)`` matters.
    """

    def __init__(self, n_s, n_p, *, dim=32, width=128, depth=3):
        super().__init__()
        self.f = _mlp(n_s, dim, width, depth)
        self.g = _mlp(n_s, dim, width, depth)
        self.fp = nn.Linear(n_p, dim, bias=False) if n_p else None
        self.gp = nn.Linear(n_p, dim, bias=False) if n_p else None

    def embed(self, s, p):
        """``(f, g)`` of states; differences of them are what ``d`` and the flow read."""
        f, g = self.f(s), self.g(s)
        if self.fp is not None:
            f, g = f + self.fp(p), g + self.gp(p)
        return f, g

    def forward(self, s0, p0, s1, p1):
        f0, g0 = self.embed(s0, p0)
        f1, g1 = self.embed(s1, p1)
        sym = torch.sqrt(((f1 - f0) ** 2).sum(dim=1) + 1e-12)
        asym = torch.relu(g1 - g0).max(dim=1).values
        return sym + asym


def split_state(X, invariant):
    """``(s, p)``: a state's features apart from, and along, the invariant coordinates."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    keep = np.ones(X.shape[1], dtype=bool)
    keep[list(invariant)] = False
    return X[:, keep], X[:, list(invariant)]


def train_quasimetric(X0, X1, T, *, invariant=(), epochs=300, batch=512, lr=1e-3, over=10.0,
                      seed=0, verbose=False, **kwargs):
    """A :class:`Quasimetric` fitted to times ``T`` between states ``X0`` and ``X1``.

    Each pair comes from a generated extremal, a feasible motion, so its time
    bounds the least time from above: ``d > T`` costs ``over`` times as much
    as ``d < T``. Pairs taken along one extremal are tight when it is a
    fastest path. Returns ``(model, ss, sp)`` with the standardizations."""
    torch.manual_seed(seed)
    S0, P0 = split_state(X0, invariant)
    S1, P1 = split_state(X1, invariant)
    ss = Standardize(np.vstack([S0, S1]))
    sp_scale = np.maximum(np.vstack([P0, P1]).std(axis=0), 1e-12) if len(invariant) else np.ones(0)
    sp = lambda P: np.asarray(P, dtype=float) / sp_scale if len(invariant) else np.zeros((len(P), 0))
    tens = lambda A: torch.as_tensor(A, dtype=torch.float32)
    S0t, S1t, P0t, P1t = tens(ss(S0)), tens(ss(S1)), tens(sp(P0)), tens(sp(P1))
    Tt = tens(T)
    model = Quasimetric(S0.shape[1], P0.shape[1], **kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for epoch in range(epochs):
        perm = torch.randperm(len(Tt))
        total = 0.0
        for i in range(0, len(Tt), batch):
            idx = perm[i:i + batch]
            err = model(S0t[idx], P0t[idx], S1t[idx], P1t[idx]) - Tt[idx]
            loss = (err ** 2 + over * torch.relu(err) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        sched.step()
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"quasimetric epoch {epoch}: loss {total / len(Tt):.5f}")
    model.eval()
    return model, ss, sp


@torch.no_grad()
def distance(model, ss, sp, X0, X1, invariant=()):
    """``d(x0, x1)`` for rows of states (broadcast if one side is a single state)."""
    S0, P0 = split_state(X0, invariant)
    S1, P1 = split_state(X1, invariant)
    n = max(len(S0), len(S1))
    rep = lambda A: np.repeat(A, n, axis=0) if len(A) == 1 else A
    tens = lambda A: torch.as_tensor(A, dtype=torch.float32)
    return model(tens(ss(rep(S0))), tens(sp(rep(P0))), tens(ss(rep(S1))), tens(sp(rep(P1)))).numpy()


@torch.no_grad()
def embedding_features(model, ss, sp, X0, X1, invariant=()):
    """Conditions for a flow: the start's own features, and the embedding's
    differences from start to goal (which add up along a chain of motions)."""
    S0, P0 = split_state(X0, invariant)
    S1, P1 = split_state(X1, invariant)
    tens = lambda A: torch.as_tensor(A, dtype=torch.float32)
    f0, g0 = model.embed(tens(ss(S0)), tens(sp(P0)))
    f1, g1 = model.embed(tens(ss(S1)), tens(sp(P1)))
    return np.hstack([ss(S0), (f1 - f0).numpy(), (g1 - g0).numpy()])


def waypoint_order(model, ss, sp, starts, ends, times, x0, xf, invariant=(), near=None):
    """Stored motions ranked as first legs of the fastest path from ``x0`` to ``xf``.

    A motion from near ``x0`` to ``b`` taking ``t`` is a good first leg when
    ``t + d(b, xf)`` is close to ``d(x0, xf)``: then ``b`` lies (nearly) on a
    fastest path, and by Bellman the costates that start that path are the
    leg's own. ``near`` limits the ranking to these indices (for instance, the
    stored motions whose starts are closest to ``x0``). Returns
    ``(indices best first, the triangle gaps, d(x0, xf))``."""
    idx = np.arange(len(times)) if near is None else np.asarray(near)
    direct = float(distance(model, ss, sp, x0[None], xf[None], invariant)[0])
    rest = distance(model, ss, sp, ends[idx], xf[None], invariant)
    gap = times[idx] + rest - direct
    order = np.argsort(gap)
    return idx[order], gap[order], direct
