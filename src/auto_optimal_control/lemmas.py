"""The objects the structure lemmas are stated in, and oracles that check their conclusions.

The lemmas of ``docs/notes/structure_certificates.md`` are about a few quantities
of an order-1 limit ``g(x) <= 0`` of a problem whose dynamics are affine in the
controls with a constant control matrix ``G`` (certificate C1) and whose running
cost has a constant Hessian ``R`` in the controls (C2). Then

    dg/dt = a(x)^T u + d(x),   a = G^T grad g,   d = grad g . f(x, 0),

and with a box of controls (centre ``m``, half-widths ``r``):

- the holdability margin ``m(x) = sum_k |a_k| r_k - |a^T m + d|`` (Lemma I: an
  arc lies where it is ``>= 0``);
- ``alpha = 1 / (a^T R^-1 a)``, with ``mu_ride = alpha * dg/dt`` under the free
  law when the controls are interior (Lemma E, step 3);
- the rate under the free law, which vanishes at every junction (Lemma J).

:class:`LimitObjects` computes them from the kernels (``grad g``, the control law
and its bounds) and from ``G`` and ``R``, taken from the problem's SymPy
expressions, so a lemma's statement and its code use the same objects.
:func:`check` runs the oracles on a path or exact solution: it returns, per lemma,
how many of its conclusions fail there.
"""

import numpy as np
import sympy as sp


class LimitObjects:
    """``a``, ``d``, the margin, ``alpha`` and the free rate of each limit, at ``(t, z)``.

    ``kernels`` are the ones the shooting uses, ``params`` their parameter vector."""

    def __init__(self, kernels, params):
        self.k, self.p = kernels, np.asarray(params, dtype=float)
        sysm = kernels.system
        prob = sysm.problem
        x, u = list(prob.states), list(prob.controls)
        f = sp.Matrix(prob.dynamics)
        G = f.jacobian(u)
        if any(e.free_symbols & set(x) for e in G):
            raise ValueError("the control matrix is not constant (certificate C1)")
        R = sp.hessian(sp.sympify(prob.running_cost), u)
        if any(e.free_symbols & set(x) | e.free_symbols & set(u) for e in R):
            raise ValueError("the cost's Hessian in the controls is not constant (C2)")
        args = list(sysm.params)
        self.G = np.array(sp.lambdify(args, G, "numpy")(*self.p), dtype=float).reshape(len(x), len(u))
        self.R = np.array(sp.lambdify(args, R, "numpy")(*self.p), dtype=float).reshape(len(u), len(u))
        self._f0 = sp.lambdify([prob.time, x, args], f.subs({v: 0 for v in u}), "numpy")
        self.n = len(x)

    def grad(self, c, t, z):
        _, Nx, _, _ = self.k.tangency_gradient(c, t, z, self.p)
        return np.asarray(Nx[0], dtype=float)[:self.n]

    def rate_parts(self, c, t, z):
        """``(a, d)`` with ``dg_c/dt = a^T u + d``."""
        gx = self.grad(c, t, z)
        f0 = np.asarray(self._f0(t, list(z[:self.n]), list(self.p)), dtype=float).reshape(-1)
        return self.G.T @ gx, float(gx @ f0)

    def margin(self, c, t, z):
        a, d = self.rate_parts(c, t, z)
        lo, hi = (np.asarray(b, dtype=float) for b in self.k.control_bounds(t, z, self.p))
        mid, rad = 0.5 * (lo + hi), 0.5 * (hi - lo)
        return float(np.abs(a) @ rad - abs(a @ mid + d))

    def alpha(self, c, t, z):
        a, _ = self.rate_parts(c, t, z)
        return float(1.0 / (a @ np.linalg.solve(self.R, a)))

    def free_rate(self, c, t, z):
        """``dg_c/dt`` under the free control law at ``z``, and whether that control is
        strictly inside its bounds."""
        a, d = self.rate_parts(c, t, z)
        u = np.asarray(self.k.control(t, z, self.p), dtype=float).reshape(-1)
        lo, hi = (np.asarray(b, dtype=float) for b in self.k.control_bounds(t, z, self.p))
        interior = bool(np.all((u - lo > 1e-9 * (1 + abs(lo))) & (hi - u > 1e-9 * (1 + abs(hi)))))
        return float(a @ u + d), interior


def check(shoot, v, n_points=401):
    """Lemma oracles on a solution of ``shoot`` (a ``MultiArcShooting``): per lemma, how
    many of its conclusions fail. Only single-limit order-1 arcs are checked.

    - ``"H"``: mu >= 0 and does not rise on an arc;
    - ``"I"``: the margin is >= 0 on an arc;
    - ``"J"``: the free law's rate vanishes at each entry (before) and exit (after);
    - ``"E"``: mu = alpha * free rate on an arc where the free law is interior.
    """
    from .shooting import MultiArcShooting

    out = {"H": 0, "I": 0, "J": 0, "E": 0}
    if not isinstance(shoot, MultiArcShooting) or not shoot.arcs:
        return out
    try:
        obj = LimitObjects(shoot.kernels, shoot.params)
    except ValueError:
        return out
    k, p = shoot.kernels, shoot.params
    d = shoot.diagnostics(v)
    order1 = [q == 1 for kind, _, q in shoot.arcs if kind == "boundary"]
    for o, (lo, hi), rise in zip(order1, d["boundary_mu"], d["boundary_mu_rise"]):
        if o and (lo < -1e-7 * max(1.0, abs(hi)) or rise > 1e-6 * max(1.0, abs(hi))):
            out["H"] += 1
    _, _, segs = shoot._evaluate(np.asarray(v, dtype=float), jac=False)
    span = max(segs[-1][3] - shoot.t0, 1e-12)
    for j, (kind, aid, ta, tb, za, _flags, r) in enumerate(segs):
        if kind != "boundary" or aid >= k.N_CONSTRAINTS or k.CONSTRAINT_ORDERS[aid] != 1:
            continue
        c = aid
        scale = max(1.0, np.abs(obj.rate_parts(c, ta, za)[0]).sum())
        # J: tangential at the junctions with free arcs.
        if j > 0 and segs[j - 1][0] is None:
            zb = segs[j - 1][6]["z"]
            if abs(obj.free_rate(c, ta, zb)[0]) > 1e-5 * scale:
                out["J"] += 1
        if j + 1 < len(segs) and segs[j + 1][0] is None:
            ze = r["z"]
            if abs(obj.free_rate(c, tb, ze)[0]) > 1e-5 * scale:
                out["J"] += 1
        # I and E along the arc.
        m = max(int(n_points * (tb - ta) / span), 5)
        z = np.asarray(za, dtype=float)[:2 * obj.n]
        times = np.linspace(ta, tb, m)
        t_prev, bad_i, bad_e = ta, False, False
        for t in times[1:-1]:
            try:
                z = np.asarray(k.propagate_boundary(c, t_prev, z, t, p, **shoot.options)["z"],
                               dtype=float)[:2 * obj.n]
            except RuntimeError:
                break
            t_prev = t
            if obj.margin(c, t, z) < -1e-7 * scale:
                bad_i = True
            rate, interior = obj.free_rate(c, t, z)
            if interior:
                mu = float(np.atleast_1d(k.boundary_multiplier(c, t, z, p))[0])
                expect = obj.alpha(c, t, z) * rate
                if abs(mu - expect) > 1e-6 * max(1.0, abs(mu), abs(expect)):
                    bad_e = True
        out["I"] += bad_i
        out["E"] += bad_e
    return out
