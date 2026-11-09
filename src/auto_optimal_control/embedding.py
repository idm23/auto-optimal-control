"""Extremals as an embedding: costates for any start and any goal.

Generation (:mod:`~auto_optimal_control.generation`) cannot choose where a
motion ends: it picks a start and costates, and the end is whatever the
extremal reaches. Robotics asks the inverse question, from here to there. This
module answers it with many generated extremals. Each stores its start, its
end, its costates, its arcs and their junction values, so the endpoints are a
map of the space of motions, and the costates are coordinates on it. A query
``(x0, xf)`` looks up stored extremals whose endpoints are close and polishes
one into the exact extremal between the queried states. Polishing is a
boundary-value solve, but one that starts from the right arcs, close by.

:class:`ExtremalSet` holds the extremals, :class:`EndpointIndex` finds near
ones, and :class:`Polisher` solves from them. Learned proposals
(:mod:`~auto_optimal_control.learn`) replace the index with a model of the
costates given the endpoints; the polish and its checks stay the same.
"""

import json
import time
from dataclasses import dataclass, field

import numpy as np

from auto_optimal_control.generation import Extremal
from auto_optimal_control.shooting import MultiArcShooting, ShootingError

_FAILURES = (ShootingError, RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError)


@dataclass
class ExtremalSet:
    """Generated extremals, as arrays, with their arcs.

    ``x0``, ``lam0``, ``xf``, ``lamf`` are ``(N, n)``; ``tf`` and ``cost`` ``(N,)``.
    ``arcs[i]`` and ``values[i]`` are extremal ``i``'s
    :class:`~auto_optimal_control.generation.Extremal` entries.
    """

    t0: float
    x0: np.ndarray
    lam0: np.ndarray
    tf: np.ndarray
    xf: np.ndarray
    lamf: np.ndarray
    cost: np.ndarray
    arcs: list = field(default_factory=list)
    values: list = field(default_factory=list)

    @classmethod
    def from_extremals(cls, extremals):
        ex = list(extremals)
        if not ex:
            raise ValueError("no extremals")
        return cls(t0=float(ex[0].t0), x0=np.array([e.x0 for e in ex]),
                   lam0=np.array([e.lam0 for e in ex]), tf=np.array([e.tf for e in ex]),
                   xf=np.array([e.xf for e in ex]), lamf=np.array([e.lamf for e in ex]),
                   cost=np.array([e.cost for e in ex]), arcs=[tuple(e.arcs) for e in ex],
                   values=[tuple((float(a), float(b), float(p)) for a, b, p in e.values)
                           for e in ex])

    def __len__(self):
        return len(self.tf)

    def __add__(self, other):
        if self.t0 != other.t0:
            raise ValueError("the sets start at different times")
        cat = lambda a, b: np.concatenate([a, b])
        return ExtremalSet(self.t0, cat(self.x0, other.x0), cat(self.lam0, other.lam0),
                           cat(self.tf, other.tf), cat(self.xf, other.xf),
                           cat(self.lamf, other.lamf), cat(self.cost, other.cost),
                           self.arcs + other.arcs, self.values + other.values)

    def extremal(self, i):
        """Extremal ``i`` as an :class:`~auto_optimal_control.generation.Extremal`."""
        return Extremal(x0=self.x0[i], lam0=self.lam0[i], t0=self.t0, tf=float(self.tf[i]),
                        xf=self.xf[i], lamf=self.lamf[i], arcs=self.arcs[i],
                        values=self.values[i], cost=float(self.cost[i]), junction_residual=0.0)

    def structures(self):
        """How many extremals have each structure, most common first."""
        counts = {}
        for a in self.arcs:
            counts[a] = counts.get(a, 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])

    def save(self, path):
        np.savez_compressed(path, t0=self.t0, x0=self.x0, lam0=self.lam0, tf=self.tf,
                            xf=self.xf, lamf=self.lamf, cost=self.cost,
                            arcs=json.dumps([list(map(list, a)) for a in self.arcs]),
                            values=json.dumps([list(map(list, v)) for v in self.values]))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        arcs = [tuple((str(e), int(c)) for e, c in a) for a in json.loads(str(d["arcs"]))]
        values = [tuple(tuple(v) for v in vals) for vals in json.loads(str(d["values"]))]
        return cls(float(d["t0"]), d["x0"], d["lam0"], d["tf"], d["xf"], d["lamf"], d["cost"],
                   arcs, values)


class EndpointIndex:
    """Nearest stored extremals to a query ``(x0, xf)``, by scaled endpoint distance.

    ``invariant`` lists state indices along which the problem is symmetric
    under translation (for an arm whose first joint turns about gravity's
    axis, ``q1``): there only the difference ``xf - x0`` counts.
    """

    def __init__(self, extremals, invariant=()):
        self.set = extremals
        self.invariant = np.asarray(invariant, dtype=int)
        F = self.features(extremals.x0, extremals.xf)
        self.scale = np.maximum(F.std(axis=0), 1e-12)
        self.F = F / self.scale

    def features(self, x0, xf):
        x0, xf = np.atleast_2d(x0).astype(float), np.atleast_2d(xf).astype(float)
        a, b = x0.copy(), xf.copy()
        if len(self.invariant):
            b[:, self.invariant] = xf[:, self.invariant] - x0[:, self.invariant]
            a[:, self.invariant] = 0.0
        keep = np.ones(a.shape[1], dtype=bool)
        keep[self.invariant] = False
        return np.hstack([a[:, keep], b])

    def nearest(self, x0, xf, k=8):
        """Indices of the ``k`` nearest extremals, nearest first, and their distances."""
        q = self.features(x0, xf)[0] / self.scale
        d = np.sqrt(((self.F - q) ** 2).sum(axis=1))
        idx = np.argsort(d)[:k]
        return idx, d[idx]

    def structures(self, x0, xf, k=15):
        """The arc structures likely at ``(x0, xf)``, most likely first: ``[(arcs, p)]``.

        A vote of the ``k`` nearest stored extremals, each weighted by the
        inverse of its distance. The structure is locally constant in the ends
        (Bonnans & Hermant 2008), so near neighbours mostly share it; on the
        six-joint arm the first structure is right about five times in six.
        """
        ids, d = self.nearest(x0, xf, k=k)
        votes = {}
        for i, di in zip(ids, d):
            arcs = tuple(self.set.arcs[i])
            votes[arcs] = votes.get(arcs, 0.0) + 1.0 / (di + 1e-12)
        total = sum(votes.values())
        return sorted(((a, w / total) for a, w in votes.items()), key=lambda item: -item[1])

    def nearest_with(self, x0, xf, arcs, k=1, among=200):
        """The ``k`` nearest stored extremals whose structure is ``arcs`` (indices)."""
        ids, _ = self.nearest(x0, xf, k=min(among, len(self.set)))
        return [i for i in ids if tuple(self.set.arcs[i]) == tuple(arcs)][:k]

    def shifted(self, i, x0):
        """Extremal ``i`` as a guess from ``x0``: its start moved along the invariant directions."""
        e = self.set.extremal(i)
        if len(self.invariant):
            x = e.x0.copy()
            x[self.invariant] = x0[self.invariant]
            e.x0 = x
        return e


@dataclass
class Polished:
    """A query's result: ``accepted`` when the exact solve converged and passed the checks."""

    accepted: bool
    solution: object = None
    shooting: object = None
    arcs: tuple = ()
    tf: float = np.nan
    lam0: np.ndarray = None
    checks: dict = field(default_factory=dict)
    seconds: float = 0.0
    tried: int = 0
    message: str = ""


class Polisher:
    """Exact extremals between two given states, solved from a stored extremal's arcs.

    ``homotopy`` is a :class:`~auto_optimal_control.homotopy.Homotopy` of a
    problem whose ends are parameters (e.g.
    :func:`~auto_optimal_control.robot.jerk_limited_arm` with
    ``moving_ends=True``), and ``params_of(x0, xf)`` gives its parameters for
    a query.
    """

    def __init__(self, homotopy, params_of, *, tol=1e-10, evaluations=60, generator=None):
        self.h = homotopy
        self.params_of = params_of
        self.tol = tol
        self.evaluations = evaluations
        # With a generator, a guess that fails is regenerated from the query's own start
        # (its costates and time kept), which finds the arcs that start leads to.
        self.generator = generator
        # A guess that does not polish directly is continued from its own ends to the
        # query's, halving the step at most this many times.
        self.continuation = True
        self.max_halvings = 64

    def _shooting(self, x0, xf, arcs):
        h = self.h
        h.params = np.asarray(self.params_of(x0, xf), dtype=float)
        kx, px = h.exact_kernels, h.exact_params()
        shoot = MultiArcShooting(kx, px, arcs, x0=x0, open_end=True, **h.options)
        if not kx.N_SWITCHING:
            shoot.tangent_junctions = True
        shoot.max_nfev = h.attempt_nfev
        return shoot

    def guess(self, shoot, e):
        """``v`` from extremal ``e``, with the terminal multipliers fitted (they enter linearly)."""
        k = shoot.kernels
        base = shoot.pack(e.lam0, None, e.tf if k.FREE_FINAL_TIME else None)
        q = {i: shoot.arcs[i][2] for i in range(len(shoot.arcs))}
        v = shoot.pack_arcs(base, [(a, b, [p] * q[i]) for i, (a, b, p) in enumerate(e.values)])
        if k.N_MULTIPLIERS:
            R, J = shoot.residual_jacobian(v)
            cols = slice(shoot.n_free + k.N_STATES, shoot.n_free + k.N_STATES + k.N_MULTIPLIERS)
            v[cols] += np.linalg.lstsq(J[:, cols], -R, rcond=None)[0]
        return v

    def checks(self, shoot, v):
        d = shoot.diagnostics(v)
        return {"violation": float(np.max(d["free_arc_violation"], initial=0.0)),
                "multiplier_min": float(min([lo for lo, _ in d["boundary_mu"]], default=0.0)),
                "pi_min": float(min([float(np.min(p)) for p in d["boundary_pi"]], default=0.0)),
                "bound_violation": float(max(d["boundary_bound_violation"], default=0.0))}

    def solve(self, x0, xf, guesses):
        """The first guess that polishes into an accepted extremal from ``x0`` to ``xf``."""
        t_start = time.perf_counter()
        x0, xf = np.asarray(x0, dtype=float), np.asarray(xf, dtype=float)
        last, tried = "no guesses", 0
        for tried, e in enumerate(guesses, 1):
            attempts = [e]
            if self.generator is not None:
                attempts.append(None)            # regenerated from x0, if the stored one fails
            for e_try in attempts:
                if e_try is None:
                    try:
                        e_try = self.generator.generate(x0, e.lam0, e.tf)
                    except Exception as exc:     # a draw the generator rejects
                        last = f"guess {tried}: regenerating: {exc}"
                        continue
                out, last = self._attempt(x0, xf, e_try, tried)
                if out is not None:
                    out.seconds = time.perf_counter() - t_start
                    return out
        return Polished(False, seconds=time.perf_counter() - t_start, tried=tried, message=last)

    def _solve_at(self, x0, xf, arcs, v):
        shoot = self._shooting(x0, xf, arcs)
        options = {"maxfev": self.evaluations} if not shoot.kernels.N_SWITCHING else None
        return shoot, shoot.solve(v, tol=self.tol, options=options)

    def _continue(self, x0, xf, e, shoot, v):
        """From extremal ``e``'s own ends (where ``v`` is exact) to the query's, in steps.

        The ends move along the segment between them; each solve starts from
        the last. Succeeds when the arcs need not change on the way."""
        s, ds = 0.0, 0.25
        while s < 1.0 and ds >= 1.0 / self.max_halvings:
            f = min(s + ds, 1.0)
            xs, xe = e.x0 + f * (x0 - e.x0), e.xf + f * (xf - e.xf)
            try:
                shoot_t, sol = self._solve_at(xs, xe, e.arcs, v)
            except _FAILURES:
                sol = None
            if sol is None or not sol.success:
                ds /= 2
                continue
            shoot, v, s, ds = shoot_t, sol.v, f, min(2 * ds, 0.5)
        return (shoot, sol) if s >= 1.0 else (None, None)

    def _attempt(self, x0, xf, e, tried):
        last = f"guess {tried}: failed"
        try:
            shoot, sol = self._solve_at(x0, xf, e.arcs, self.guess(self._shooting(x0, xf, e.arcs), e))
        except _FAILURES as exc:
            shoot, sol = None, None
            last = f"guess {tried}: {exc}"
        if (sol is None or not sol.success) and self.continuation:
            try:
                own = self._shooting(e.x0, e.xf, e.arcs)
                shoot_c, sol_c = self._continue(x0, xf, e, own, self.guess(own, e))
            except _FAILURES:
                shoot_c = sol_c = None
            if sol_c is not None:
                shoot, sol = shoot_c, sol_c
        if sol is None:
            return None, last
        if not sol.success:
            return None, f"guess {tried}: residual {sol.residual_norm:.1e}"
        try:
            ch = self.checks(shoot, sol.v)
        except _FAILURES as exc:
            return None, f"guess {tried}: checks failed: {exc}"
        ok = (ch["violation"] < 1e-8 and ch["multiplier_min"] > -1e-8
              and ch["pi_min"] > -1e-8 and ch["bound_violation"] < 1e-8)
        if not ok:
            return None, f"guess {tried}: rejected {ch}"
        return Polished(True, sol, shoot, tuple(e.arcs), sol.tf, sol.lam0, ch, 0.0, tried,
                        "accepted"), "accepted"
