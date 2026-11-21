"""Free arcs of integrator chains in closed form (rule R23, Lemma T of
``docs/notes/lemma_register.md``).

A problem whose state is a set of *integrator chains*, ``s_0' = s_1, ...,
s_{L-1}' = u`` with one control at the top of each chain, a running cost of the
controls alone, and no penalty, has free arcs that need no integration:

- the costates satisfy ``lam_0' = 0`` and ``lam_m' = -lam_{m-1}``, so they are
  polynomials in time, ``lam_m(t0 + s) = sum_{k <= m} lam_k(t0) (-s)^(m-k) / (m-k)!``;
- with ``H`` quadratic in ``u`` (``d2H/du2 = r``, a constant), the control is
  ``u = clip(-(lam_{L-1} + c) / r, lo, hi)``, a clipped polynomial of degree
  ``L - 1``, which reaches its bounds at the real roots of two polynomials;
- the states are the chain's Taylor polynomial plus the integrals
  ``int_0^s (s - sigma)^(L-1-m) / (L-1-m)! u(sigma) dsigma``, polynomial on each
  piece between those roots, so ``L``-point Gauss-Legendre quadrature on each
  piece is exact;
- the state transition matrix is block diagonal, one ``2L x 2L`` block per chain,
  and the derivatives of the integrals are integrals of the same kind: ``u`` is
  continuous where it meets its bounds, so the moving roots add nothing.

A jerk-controlled arm is six chains of length 3 (``q, w, a`` and the jerk); its
torque limits act only on boundary arcs, which are still integrated.
"""

import hashlib
import os
import shutil
import sys
import tempfile
from math import factorial
from pathlib import Path

import numpy as np
import sympy as sp

_SETUP = """\
from Cython.Build import cythonize
from setuptools import Extension, setup

MOD = {mod!r}
flags = ["/O2", "/fp:precise"] if __import__("os").name == "nt" else ["-O3", "-g0", "-ffp-contract=off"]
setup(name=MOD, script_args=["build_ext", "--inplace"],
      ext_modules=cythonize([Extension(MOD, [MOD + ".pyx"], extra_compile_args=flags)],
                            language_level=3, quiet=True))
"""
_MODULE = None


def compiled(cache_dir=None):
    """The compiled flow (``chains_template.pyx``), built once into the kernel cache;
    ``None`` if it cannot be built (the NumPy flow is used then)."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE or None
    from . import build
    try:
        import Cython
        text = Path(__file__).with_name("chains_template.pyx").read_text()
        h = hashlib.sha256()
        for part in (text, _SETUP, Cython.__version__, sys.version):
            h.update(part.encode()); h.update(b"\0")
        mod = "aoc_chains_" + h.hexdigest()[:16]
        root = Path(cache_dir) if cache_dir is not None else build.cache_root()
        root.mkdir(parents=True, exist_ok=True)
        final = root / mod
        path = build._find_extension(final, mod)
        if path is None:
            tmp = tempfile.mkdtemp(prefix=mod + ".", dir=root)
            try:
                build._compile(tmp, mod, {mod + ".pyx": text, "setup.py": _SETUP.format(mod=mod)},
                               False)
                try:
                    os.rename(tmp, final)
                except OSError:          # another process finished first
                    shutil.rmtree(tmp, ignore_errors=True)
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            path = build._find_extension(final, mod)
        _MODULE = build._load(mod, path)
    except Exception:                    # no compiler: the NumPy flow still works
        _MODULE = False
        return None
    return _MODULE


class IntegratorChains:
    """The closed-form free arc of a problem certified by :meth:`certify`."""

    def __init__(self, chains, length, lo, hi, r, c, cost, n, cost_terms=None):
        self.chains = np.asarray(chains, dtype=np.intp)   # (n_chains, L): state indices, bottom first
        self.L = length
        self._lo, self._hi, self._r, self._c, self._cost = lo, hi, r, c, cost
        self._cost_terms = cost_terms     # p -> (L(0), linear, quadratic coefficients per control)
        self.n = n
        self.pinned_of = ()               # Lemma T'': set by ``certify``
        self._key = None
        self._ch_c = np.ascontiguousarray(self.chains, dtype=np.int_)
        self.module = None                # the compiled flow, set by ``use_compiled``
        self._gx, self._gw = np.polynomial.legendre.leggauss(length + 1)
        L = length
        # lam_top(s) = sum_k lam_k(0) * basis_k(s), basis_k(s) = (-s)^(L-1-k) / (L-1-k)!
        self._fact = np.array([factorial(i) for i in range(L + 1)], dtype=float)

    # -- the certificate ----------------------------------------------------

    @classmethod
    def certify(cls, system):
        """``(IntegratorChains, None)`` if the free arcs of ``system`` (a
        :class:`~auto_optimal_control.pontryagin.PontryaginSystem`) have the closed
        form, else ``(None, reason)``."""
        prob = system.problem
        states, controls = list(system.states), list(system.controls)
        params = list(system.params)
        if system.penalty != 0:
            return None, "the problem has a penalty"
        if system.switching_functions:
            return None, "a control enters H linearly (bang-bang)"
        if not system.is_autonomous:
            return None, "H depends on time"
        dyn = list(prob.dynamics)
        top_of = {}
        below = {}
        for i, f in enumerate(dyn):
            if f in controls:
                if f in top_of:
                    return None, f"control {f} drives two states"
                top_of[f] = i
            elif f in states:
                j = states.index(f)
                if j in below:
                    return None, f"state {f} is the rate of two states"
                below[j] = i
            else:
                return None, f"the rate of {states[i]} is not a state or a control"
        if set(top_of) != set(controls):
            return None, "a control drives no state"
        chains = []
        for u in controls:
            chain = [top_of[u]]
            while chain[-1] in below:
                chain.append(below[chain[-1]])
                if len(chain) > len(states):
                    return None, "the chain is a cycle"
            chains.append(chain[::-1])
        if sorted(i for ch in chains for i in ch) != list(range(len(states))):
            return None, "a state is in no chain"
        lengths = {len(ch) for ch in chains}
        if len(lengths) != 1:
            return None, "the chains differ in length"
        L = lengths.pop()
        lam = list(system.costates)
        # Costates: lam_{s_0}' = 0, lam_{s_m}' = -lam_{s_{m-1}}.
        for ch in chains:
            for m, i in enumerate(ch):
                want = sp.S.Zero if m == 0 else -lam[ch[m - 1]]
                if sp.simplify(system.costate_dynamics[i] - want) != 0:
                    return None, f"the costate of {states[i]} is not that of a chain"
        H = system.hamiltonian_full
        rs, cs, los, his = [], [], [], []
        boxes = {}
        for box in prob.control_sets:
            if not hasattr(box, "lower"):
                return None, "a control set is not a box"
            boxes[box.control] = box
        free = set(states) | set(lam) | set(controls) | {system.time}
        for u, ch in zip(controls, chains):
            if u not in boxes:
                return None, f"control {u} has no box"
            if sp.diff(H, u, 3) != 0:
                return None, f"H is not quadratic in {u}"
            r = sp.diff(H, u, 2)
            g1 = sp.expand(sp.diff(H, u).subs({v: 0 for v in controls}))
            c = sp.expand(g1 - lam[ch[-1]])
            for other in controls:
                if other != u and sp.diff(H, u, other) != 0:
                    return None, "H couples two controls"
            lo, hi = boxes[u].lower, boxes[u].upper
            if any(free & sp.sympify(e).free_symbols for e in (r, c, lo, hi)):
                return None, f"the law of {u} depends on more than the parameters"
            rs.append(r), cs.append(c), los.append(lo), his.append(hi)
        Lc = prob.running_cost
        if free - set(controls) & Lc.free_symbols:
            return None, "the running cost depends on the state"
        if sp.Poly(Lc, *controls).total_degree() > 2:
            return None, "the running cost is not quadratic in the controls"
        f = lambda exprs: sp.lambdify([params], sp.Matrix(exprs), "numpy")
        cost = sp.lambdify([controls, params], Lc, "numpy")
        at0 = {v: 0 for v in controls}
        terms = f([Lc.subs(at0)] + [sp.diff(Lc, u).subs(at0) for u in controls]
                  + [sp.diff(Lc, u, 2) / 2 for u in controls])
        out = cls(chains, L, f(los), f(his), f(rs), f(cs), cost, len(states), terms)
        out.pinned_of = cls._pins(prob, states, chains, params, system.time)
        return out, None

    @staticmethod
    def _pins(prob, states, chains, params, time):
        """Lemma T'': per path limit, the chain whose control its riding pins at 0, or
        None. A limit ``a s - b`` on one chain state ``s`` (``a`` a nonzero constant, ``b``
        free of the states and time) is ridden through ``g^(q) = a u`` with no state in
        it, so riding it holds that chain's control at 0 and adds nothing to the
        costates' equations."""
        out = []
        where = {s: j for j, ch in enumerate(chains) for s in (states[i] for i in ch)}
        for pc in prob.path_constraints:
            e = sp.sympify(pc.expr)
            used = [s for s in states if e.has(s)]
            pin = None
            if len(used) == 1 and not e.has(time):
                s = used[0]
                a = sp.diff(e, s)
                if a.free_symbols <= set(params) and sp.diff(e, s, 2) == 0 and a != 0:
                    pin = where[s]
            out.append(pin)
        return tuple(out)

    def use_compiled(self, cache_dir=None):
        """Use the compiled flow where it is available (chains of length 2 or 3)."""
        if 2 <= self.L <= 3:
            self.module = compiled(cache_dir)
        return self.module is not None

    def _constants(self, p):
        """The law's and the cost's constants at ``p``, cached for the last ``p``."""
        key = np.asarray(p, dtype=float).tobytes()
        if key != self._key:
            lo, hi, r, c = self._law(p)
            nc = self.chains.shape[0]
            t = np.asarray(self._cost_terms(list(p)), float).ravel()
            self._const = tuple(np.ascontiguousarray(x, dtype=float) for x in
                                (lo, hi, r, c, t[1:1 + nc], t[1 + nc:1 + 2 * nc])) + (float(t[0]),)
            self._key = key
        return self._const

    # -- the flow -----------------------------------------------------------

    def _law(self, p):
        p = list(p)
        return (np.asarray(self._lo(p), float).ravel(), np.asarray(self._hi(p), float).ravel(),
                np.asarray(self._r(p), float).ravel(), np.asarray(self._c(p), float).ravel())

    def _cuts(self, coef, levels, tau):
        """Per chain, the times in ``(0, tau)`` where ``sum_d coef[:, d] s^d`` equals
        one of ``levels``, as an array padded with ``tau``."""
        nc, L = coef.shape[0], self.L
        out = np.full((nc, 2 * (L - 1)), tau)
        lo_t, hi_t = min(0.0, tau), max(0.0, tau)
        for col, lev in enumerate(levels):
            a = coef.copy()
            a[:, 0] -= lev
            if L == 3:
                A, B, C = a[:, 2], a[:, 1], a[:, 0]
                with np.errstate(divide="ignore", invalid="ignore"):
                    disc = B * B - 4 * A * C
                    sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
                    # Stable roots; a linear (A == 0) or constant polynomial is handled below.
                    qq = -0.5 * (B + np.copysign(sq, B))
                    r1 = np.where(A != 0, qq / A, np.where(B != 0, -C / B, np.nan))
                    r2 = np.where((A != 0) & (qq != 0), C / qq, np.nan)
                roots = np.stack([r1, r2], axis=1)
            elif L == 2:
                with np.errstate(divide="ignore", invalid="ignore"):
                    roots = np.where(a[:, 1:2] != 0, -a[:, 0:1] / a[:, 1:2], np.nan)
            else:
                roots = np.full((nc, L - 1), np.nan)
                for i in range(nc):
                    rr = np.roots(a[i, ::-1]) if np.any(a[i, 1:]) else np.array([])
                    rr = rr[np.abs(rr.imag) <= 1e-12 * (1 + np.abs(rr.real))].real
                    roots[i, :rr.size] = rr
            ok = (roots > lo_t) & (roots < hi_t)
            roots = np.where(ok, roots, tau)
            out[:, col * (L - 1):(col + 1) * (L - 1)] = roots
        return out

    def propagate(self, t0, z0, t1, p, stm=False, cost=False, pinned=()):
        """``z(t1)`` from ``z(t0) = z0`` along the free arc, with ``Phi = dz(t1)/dz0``
        and ``int L dt`` on request: the keys of ``propagate`` that shooting reads.
        ``pinned``: chains whose control is held at 0, an arc riding limits on their
        states (Lemma T'')."""
        if self.module is not None:
            return self._propagate_compiled(t0, z0, t1, p, stm, cost, pinned)
        z0 = np.asarray(z0, dtype=float)
        n, L, ch = self.n, self.L, self.chains
        lo, hi, r, c = self._law(p)
        if pinned:
            lo, hi = lo.copy(), hi.copy()
            lo[list(pinned)] = hi[list(pinned)] = 0.0
        tau = float(t1) - float(t0)
        S = z0[ch]                                   # (nc, L) states, bottom first
        Lam = z0[n + ch]                             # (nc, L) costates
        nc = ch.shape[0]
        fact = self._fact
        # lam_top(s) + c = sum_d coef[:, d] s^d, with coef[:, d] = lam_{L-1-d}(0) (-1)^d / d!
        d = np.arange(L)
        sign_fact = (-1.0) ** d / fact[d]
        coef = Lam[:, L - 1 - d] * sign_fact
        coef[:, 0] += c
        # u = -(lam_top + c) / r reaches hi where lam_top + c = -r hi, lo where = -r lo.
        cuts = self._cuts(coef, (-r * hi, -r * lo), tau) if nc else np.zeros((0, 0))
        edges = np.sort(np.concatenate([np.zeros((nc, 1)), cuts, np.full((nc, 1), tau)], axis=1),
                        axis=1)
        if tau < 0:
            edges = edges[:, ::-1]                   # 0 down to tau: the pieces run backward
        a, b = edges[:, :-1], edges[:, 1:]           # (nc, P) piece ends, signed widths b - a
        half = (b - a)[..., None] / 2
        sig = (a + b)[..., None] / 2 + half * self._gx       # (nc, P, G) nodes
        w = half * self._gw                                    # signed weights: int_0^tau
        powers = sig[..., None] ** d                           # (nc, P, G, L)
        lam_top = np.einsum("cpgd,cd->cpg", powers, coef)
        u_int = -lam_top / r[:, None, None]
        u = np.clip(u_int, lo[:, None, None], hi[:, None, None])
        inside = (u_int > lo[:, None, None]) & (u_int < hi[:, None, None])
        # Kernels K_m(sigma) = (tau - sigma)^(L-1-m) / (L-1-m)!, m = 0..L-1.
        e = L - 1 - d
        K = (tau - sig)[..., None] ** e / fact[e]               # (nc, P, G, L)
        I = np.einsum("cpg,cpgm,cpg->cm", w, K, u)             # (nc, L)
        z = np.empty_like(z0)
        # States: s_m(tau) = sum_{k >= m} s_k(0) tau^(k-m) / (k-m)! + I_m.
        T = np.zeros((L, L))
        for m in range(L):
            for k in range(m, L):
                T[m, k] = tau ** (k - m) / fact[k - m]
        z[ch] = S @ T.T + I
        # Costates: lam_m(tau) = sum_{k <= m} lam_k(0) (-tau)^(m-k) / (m-k)!.
        C = np.zeros((L, L))
        for m in range(L):
            for k in range(m + 1):
                C[m, k] = (-tau) ** (m - k) / fact[m - k]
        z[n + ch] = Lam @ C.T
        out = {"t": float(t1), "z": z, "status": 0, "stm": None, "cost": None}
        if stm:
            # d lam_top(sigma) / d lam_k(0) = (-sigma)^(L-1-k) / (L-1-k)!
            basis = (-sig)[..., None] ** e / fact[e]           # (nc, P, G, L) over k
            du = -np.where(inside, 1.0 / r[:, None, None], 0.0)[..., None] * basis
            dI = np.einsum("cpg,cpgm,cpgk->cmk", w, K, du)     # (nc, L states, L costates)
            Phi = np.zeros((2 * n, 2 * n))
            for i in range(nc):
                rows, lam_rows = ch[i], n + ch[i]
                Phi[np.ix_(rows, rows)] = T
                Phi[np.ix_(rows, lam_rows)] = dI[i]
                Phi[np.ix_(lam_rows, lam_rows)] = C
            out["stm"] = Phi
        if cost:
            # H couples no two controls, so L = L(0) + sum_i (L(u_i e_i) - L(0)); each
            # term is a polynomial of degree <= 2 L - 2 on its own chain's pieces.
            zero = [0.0] * nc
            L0 = float(self._cost(zero, list(p)))
            total = L0 * tau
            for i in range(nc):
                us = [np.zeros_like(u[i]) for _ in range(nc)]
                us[i] = u[i]
                vals = np.asarray(self._cost(us, list(p)), float) - L0
                total += float(np.sum(w[i] * vals))
            out["cost"] = total
        out["dz_dt1"] = self.rhs(z, p, pinned)
        return out

    def _propagate_compiled(self, t0, z0, t1, p, stm, cost, pinned=()):
        lo, hi, r, c, cl, cq, L0 = self._constants(p)
        if pinned:
            lo, hi = lo.copy(), hi.copy()
            lo[list(pinned)] = hi[list(pinned)] = 0.0
        z0 = np.ascontiguousarray(z0, dtype=float)
        tau = float(t1) - float(t0)
        z = np.empty_like(z0)
        Phi = np.zeros((2 * self.n, 2 * self.n)) if stm else np.zeros((1, 1))
        total = self.module.flow(tau, z0, self._ch_c, self.n, lo, hi, r, c, cl, cq,
                                 self._gx, self._gw, z, Phi, bool(stm), bool(cost))
        out = {"t": float(t1), "z": z, "status": 0, "stm": Phi if stm else None,
               "cost": L0 * tau + total if cost else None}
        out["dz_dt1"] = self._rhs(z, lo, hi, r, c)
        return out

    def _rhs(self, z, lo, hi, r, c):
        n, ch = self.n, self.chains
        f = np.zeros_like(z)
        S, Lam = z[ch], z[n + ch]
        f[ch[:, :-1]] = S[:, 1:]
        f[ch[:, -1]] = np.clip(-(Lam[:, -1] + c) / r, lo, hi)
        f[n + ch[:, 1:]] = -Lam[:, :-1]
        return f

    def rhs(self, z, p, pinned=()):
        """``z' = F(z)`` on a free arc (with ``pinned`` chains' controls at 0)."""
        z = np.asarray(z, dtype=float)
        n, L, ch = self.n, self.L, self.chains
        lo, hi, r, c = self._law(p)
        if pinned:
            lo, hi = lo.copy(), hi.copy()
            lo[list(pinned)] = hi[list(pinned)] = 0.0
        f = np.zeros_like(z)
        S, Lam = z[ch], z[n + ch]
        f[ch[:, :-1]] = S[:, 1:]
        f[ch[:, -1]] = np.clip(-(Lam[:, -1] + c) / r, lo, hi)
        f[n + ch[:, 1:]] = -Lam[:, :-1]
        return f
