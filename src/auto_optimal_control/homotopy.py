"""From a costate distribution to the exact extremal, with no other input.

:func:`solve` (or :class:`Homotopy`, to keep the compiled kernels for several
solves) takes the problem, its parameter values and a distribution of
initial costates (and of ``t_f`` when it is free). Everything else is found:

1. **Regularize.** :func:`regularize` adds a smoothing term
   ``eps ((u - mid) / half)^2 / 2`` to the running cost for each bang-bang
   control (its law becomes clipped-linear) and an exterior penalty
   ``rho / 2 max(0, g)^2`` for each path constraint, with ``eps`` and ``rho``
   runtime parameters.
2. **Sample.** Single shooting from random costates on the regularized
   problem, at ``eps = 1`` and ``rho = 0``.
3. **Continue, in phases.** When a limit is first violated, ``eps`` is held
   while its penalty weight ramps up, in decades, from ``1e-6 rho_ref`` to
   ``1e4 rho_ref`` (``rho_ref`` makes the penalty of the violation it saw as
   large as the cost); then ``eps`` falls, in decades, with the penalties
   held. A ramp that stalls is held where it got to. The steps adapt: they
   grow after a success and halve after a failure, and a secant predictor
   starts each solve.
4. **Read the structure and jump.** After each step the regularized
   solution is read: the intervals where each limit is violated become
   boundary arcs (or touch points, for order 2), and stretches where a
   bang-bang control stays strictly between its bounds become singular arcs
   (or, when short, the ramps of ordinary switches). Each reading gives
   candidate structures for :class:`~auto_optimal_control.shooting.MultiArcShooting`
   on the *exact* problem, seeded from the regularized costates, junction
   times and fitted jump multipliers, plus a few seeds with jittered times.
5. **Accept.** A candidate is accepted when the exact solve converges and
   passes the necessary conditions it can check (no violation on free arcs,
   controls within bounds on special arcs, ``mu >= 0`` on boundary arcs,
   ``nu >= 0`` at touch points, Legendre-Clebsch, the held bounds agreeing
   with the switching function). Among the candidates accepted at one step,
   the cheapest wins.

Bang-bang switches need no reading at all: on free arcs they are events of
the integration. Without bang-bang controls or path constraints the
regularized problem is the exact one and step 2 is the whole solve.

When no candidate is accepted (a limit of order 3 or more, two limits active
at once, a structure outside what :class:`MultiArcShooting` represents),
:func:`solve` returns the last regularized solution with ``exact = False``.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from auto_optimal_control.build import build_kernels
from auto_optimal_control.pontryagin import derive
from auto_optimal_control.problem import Box, OptimalControlProblem, PathConstraint
from auto_optimal_control.shooting import MultiArcShooting, ShootingError, SingleShooting

EPS = "eps_reg"
RHO = "rho_reg_"
_FAILURES = (ShootingError, RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError)


# =============================================================================
# The regularized and exact problems
# =============================================================================

def bang_bang_controls(problem):
    """The controls the Hamiltonian is linear in, on a :class:`Box` (the bang-bang ones)."""
    return tuple(sw.control for sw in derive(problem).switching_functions)


def _copy(problem, *, name, running_cost, params, path_constraints):
    return OptimalControlProblem(
        name=name, time=problem.time, states=problem.states, controls=problem.controls,
        dynamics=problem.dynamics, running_cost=running_cost,
        terminal_cost=problem.terminal_cost, initial_state=dict(problem.initial_state),
        terminal_constraints=problem.terminal_constraints, initial_time=problem.initial_time,
        final_time=problem.final_time, params=params, control_sets=problem.control_sets,
        path_constraints=path_constraints)


def regularize(problem, bang_bang=None):
    """The problem with a smoothing weight and a penalty weight per path constraint.

    Each bang-bang control ``u`` in ``[lo, hi]`` gets
    ``eps ((u - mid) / half)^2 / 2`` in the running cost (``mid`` and
    ``half`` the middle and half-width of its range), so its law becomes
    ``clip(mid - half^2 S / eps)``. Each path constraint ``c`` gets the
    exterior penalty weight ``rho_reg_<name>``. Both are runtime parameters,
    appended to ``problem.params`` (``eps_reg`` first, when there are
    bang-bang controls). Penalty weights the problem gives itself are
    replaced. ``bang_bang`` defaults to :func:`bang_bang_controls`.

    Returns ``(problem, eps, rhos)``: the regularized problem, the ``eps``
    symbol (``None`` without bang-bang controls) and the ``rho`` symbols.
    """
    if bang_bang is None:
        bang_bang = bang_bang_controls(problem)
    params, cost, eps = list(problem.params), problem.running_cost, None
    if bang_bang:
        eps = sp.Symbol(EPS, positive=True)
        params.append(eps)
        boxes = {cs.control: cs for cs in problem.control_sets if isinstance(cs, Box)}
        for u in bang_bang:
            lo, hi = boxes[u].lower, boxes[u].upper
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            cost = cost + eps * ((u - mid) / half) ** 2 / 2
    rhos = tuple(sp.Symbol(RHO + c.name, nonnegative=True) for c in problem.path_constraints)
    limits = [PathConstraint(c.expr, c.name, penalty=r)
              for c, r in zip(problem.path_constraints, rhos)]
    reg = _copy(problem, name=(problem.name or "problem") + "_regularized", running_cost=cost,
                params=params + list(rhos), path_constraints=limits)
    return reg, eps, rhos


def exact(problem):
    """The problem with its path-constraint penalties removed."""
    return _copy(problem, name=problem.name or "problem", running_cost=problem.running_cost,
                 params=problem.params,
                 path_constraints=[PathConstraint(c.expr, c.name)
                                   for c in problem.path_constraints])


# =============================================================================
# Results
# =============================================================================

@dataclass
class Step:
    """One continuation step: the weights, the residual and what was tried.

    ``s`` is how many decades ``eps`` has fallen from its start (0 without
    bang-bang controls), ``rho`` the penalty weights, ``candidates`` how many
    exact solves were tried from this step's reading, ``accepted`` whether
    one passed.
    """

    s: float
    eps: float
    rho: np.ndarray
    residual: float
    candidates: int = 0
    accepted: bool = False


@dataclass
class HomotopyResult:
    """The result of :meth:`Homotopy.solve`.

    ``exact`` says whether ``solution`` solves the exact problem (with
    ``shooting``, a :class:`SingleShooting` or :class:`MultiArcShooting` on
    the exact kernels) or is the last regularized solution (``shooting`` on
    the regularized kernels, at ``params``). ``arcs`` is the structure found,
    as :class:`MultiArcShooting` entries (empty: free arcs only, bang-bang
    switches included). ``objective`` is the exact cost ``phi + int L`` of
    an exact solution. ``steps`` logs the continuation, ``tries`` the random
    draws of the first solve, and ``message`` says how it ended.
    """

    solution: object
    shooting: object
    arcs: tuple
    exact: bool
    objective: float
    params: np.ndarray
    steps: list = field(default_factory=list)
    tries: int = 0
    message: str = ""

    @property
    def success(self):
        return self.exact


# =============================================================================
# The solver
# =============================================================================

class Homotopy:
    """The regularized and exact kernels of one problem, and the automatic solve.

    The kernels do not depend on the parameter values, so one instance
    serves every :meth:`solve`.

    Parameters
    ----------
    problem : OptimalControlProblem
    cache_dir : path, optional
        Where the compiled kernels are cached.
    rtol, atol : float
        Integration tolerances of the shooting.
    verbose : bool
        Print each step.
    """

    def __init__(self, problem, *, cache_dir=None, rtol=1e-12, atol=1e-12, verbose=False):
        self.problem = problem
        self.params = None
        self.cache_dir = cache_dir
        self.options = dict(rtol=rtol, atol=atol)
        self.verbose = verbose
        self.attempt_evaluations = 60      # MINPACK evaluations per exact attempt
        self.attempt_nfev = 50000          # right-hand-side evaluations per arc
        self.bang_bang = bang_bang_controls(problem)
        self.regularized, self.eps_symbol, self.rho_symbols = regularize(problem, self.bang_bang)
        self.kr = build_kernels(derive(self.regularized), cache_dir=cache_dir)
        self.nc = len(problem.path_constraints)
        self._kx = None
        t, x, p = problem.time, problem.states, problem.params
        self._terminal = sp.lambdify((t, x, p), problem.terminal_cost, "numpy")

    def _say(self, *args):
        if self.verbose:
            print(*args)

    # -- kernels and parameters ---------------------------------------------

    @property
    def exact_kernels(self):
        """Kernels of the exact problem: singular arcs derived when there are
        bang-bang controls; otherwise the regularized kernels (at ``rho = 0``
        they are the exact ones)."""
        if self._kx is None:
            if self.bang_bang:
                self._kx = build_kernels(derive(exact(self.problem), singular=True),
                                         cache_dir=self.cache_dir)
            else:
                self._kx = self.kr
        return self._kx

    def exact_params(self):
        if self.bang_bang:
            return self.params.copy()
        return self.regularized_params(None, np.zeros(self.nc))

    def regularized_params(self, eps, rho):
        p = list(self.params)
        if self.bang_bang:
            p.append(eps)
        return np.array(p + list(rho), dtype=float)

    def objective(self, shooting, v):
        """``phi(t_f, x(t_f)) + int L`` of the exact problem at ``v``."""
        if isinstance(shooting, MultiArcShooting):
            _, _, segs = shooting._evaluate(np.asarray(v, dtype=float), jac=False, record=True)
            running = float(sum(r["cost"] for *_, r in segs))
            tf, xf = segs[-1][3], segs[-1][-1]["z"][:shooting.kernels.N_STATES]
        else:
            x0, lam0, _, tf = shooting.unpack(v)
            r = shooting.kernels.propagate(shooting.t0, np.concatenate([x0, lam0]), tf,
                                           shooting.params, cost=True, **self.options)
            running, xf = float(r["cost"]), r["z"][:shooting.kernels.N_STATES]
        return running + float(self._terminal(tf, list(xf), list(self.params)))

    # -- step 2: sampling -----------------------------------------------------

    def _sampler(self, shoot, costates, final_time):
        n = self.kr.N_STATES

        def draw(spec, rng, what):
            if callable(spec):
                return spec(rng)
            lo, hi = (np.asarray(b, dtype=float) for b in spec)
            return rng.uniform(lo, hi)

        def sample(rng):
            lam0 = np.asarray(draw(costates, rng, "costates"), dtype=float).reshape(-1)
            if lam0.shape != (n,):
                raise ValueError(f"the costate sample must have {n} entries")
            tf = None
            if self.kr.FREE_FINAL_TIME:
                if final_time is None:
                    raise ValueError("the final time is free: give a final_time distribution")
                tf = float(draw(final_time, rng, "final_time"))
            return shoot.guess_multipliers(lam0, tf=tf)

        return sample

    # -- step 4: reading the structure ----------------------------------------

    def _read(self, shoot, v, n_points=2001):
        """Labels of a regularized solution: time, states, violated intervals
        per constraint, and per bang-bang control its bound labels."""
        kr, p = self.kr, shoot.params
        x0, lam0, _, tf = shoot.unpack(v)
        z0 = np.concatenate([x0, lam0])
        tr = kr.trajectory(shoot.t0, z0, tf, p, n_points=n_points, **self.options)
        t, Z, U = tr["t"], tr["z"], tr["u"]
        T = tf - shoot.t0
        out = {"t": t, "z": Z, "u": U, "T": T, "t0": shoot.t0, "tf": tf, "z0": z0,
               "violated": [], "labels": {}}
        # Violated intervals, from the located crossings and g between them.
        for c in range(self.nc):
            cross = np.sort(tr["constraint_times"][tr["constraint_index"] == c])
            ts = np.concatenate([[shoot.t0], cross, [tf]])
            mids = 0.5 * (ts[:-1] + ts[1:])
            gm = np.interp(mids, t, tr["g"][:, c])
            spans = [(a, b) for a, b, g in zip(ts[:-1], ts[1:], gm) if g > 0 and b > a]
            merged = []
            for a, b in spans:
                if merged and a - merged[-1][1] < 0.02 * T:
                    merged[-1] = (merged[-1][0], b)
                else:
                    merged.append((a, b))
            out["violated"].append(merged)
        controls = list(kr.system.controls)
        for u in self.bang_bang:
            j = controls.index(u)
            bounds = np.array([kr.control_bounds(ti, zi, p) for ti, zi in zip(t, Z)])
            lo, hi = bounds[:, 0, j], bounds[:, 1, j]
            tol = 1e-9 * np.maximum(hi - lo, 1e-300)
            lab = np.zeros(len(t), dtype=int)
            lab[U[:, j] >= hi - tol] = 1
            lab[U[:, j] <= lo + tol] = -1
            out["labels"][u] = lab
        return out

    @staticmethod
    def _runs(t, mask):
        """``[(start, end)]`` of the runs where ``mask`` holds, in time."""
        edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
        out = []
        for a, b in zip(edges[0::2], edges[1::2]):
            ta = t[a] if a == 0 else 0.5 * (t[a - 1] + t[a])
            tb = t[b - 1] if b == len(t) else 0.5 * (t[b - 1] + t[b])
            out.append((ta, tb))
        return out

    def _candidates(self, read, max_candidates=24):
        """Candidate structures from a reading: ``(items, held)`` per candidate.

        ``items`` are ``(kind, index, a, b)`` in time order with kind
        ``boundary``, ``touch``, ``singular``; ``held`` gives, per control
        with a singular arc, the bound labels of its free arcs. The second
        return value is whether any candidate could be built (``False``
        while a limit without exact arcs is violated).
        """
        kx = self.exact_kernels
        t, T = read["t"], read["T"]
        options = []                       # per violated interval: kinds to try
        for c, spans in enumerate(read["violated"]):
            for a, b in spans:
                if not kx.EXACT_ARCS[c]:
                    return [], False
                kinds = ["boundary"]
                if kx.CONSTRAINT_ORDERS[c] == 2:
                    kinds = ["touch", "boundary"] if b - a < 0.05 * T else ["boundary", "touch"]
                options.append([(k, c, a, b) for k in kinds])
        limited = [(a, b) for spans in read["violated"] for a, b in spans]

        def on_limit(a, b):
            return sum(max(0.0, min(b, y) - max(a, x)) for x, y in limited)

        names = [sw.control for sw in kx.system.switching_functions]
        singular_choices = []              # per control: list of run subsets
        ramps = {}                         # per control: pieces that count against readiness
        near = 0.02 * T
        for u, lab in read["labels"].items():
            # Stretches strictly between the bounds, cut where a limit is
            # violated (the control there is the one riding the limit).
            pieces = []
            for a, b in self._runs(t, lab == 0):
                cuts = [(a, b)]
                for x, y in limited:
                    cuts = [(p, q) for (p0, q0) in cuts
                            for p, q in ((p0, min(q0, x)), (max(p0, y), q0)) if q > p]
                pieces += cuts
            # A piece that runs onto a limit is the smoothed approach to it;
            # one that leaves a limit may be a joined singular arc.
            onto = [any(abs(b - x) < near for x, _ in limited) for a, b in pieces]
            ramps[u] = [r for r, o in zip(pieces, onto)
                        if not o and not any(abs(r[0] - y) < near for _, y in limited)]
            sw = names.index(u)
            if not kx.SINGULAR_ARCS[sw]:
                singular_choices.append([(u, ())])
                continue
            long = sorted([r for r, o in zip(pieces, onto) if not o and r[1] - r[0] >= 0.02 * T],
                          key=lambda r: r[0] - r[1])[:3]
            subsets = [()]
            for k in range(1, len(long) + 1):
                subsets += [tuple(sorted(c)) for c in itertools.combinations(long, k)]
            singular_choices.append([(u, s) for s in subsets[::-1]])   # most singular first

        out = []
        for limits in itertools.product(*options) if options else [()]:
            for sing in itertools.product(*singular_choices) if singular_choices else [()]:
                # Ready: the ramps not read as singular are short.
                rest = sum(b - a for u, subset in sing for a, b in ramps[u]
                           if (a, b) not in subset)
                if rest > 0.05 * T:
                    continue
                items = sorted(list(limits) + [("singular", u, a, b) for u, subset in sing
                                               for a, b in subset], key=lambda it: it[2])
                out.append(items)
                if len(out) >= max_candidates:
                    return out, True
        return out, True

    def _structure(self, items, read):
        """``(arcs, values)`` for MultiArcShooting from candidate items, or None.

        Adds the joins, the bounds held next to singular arcs and the
        switches of controls with singular arcs on free arcs.
        """
        t, T = read["t"], read["T"]
        special = [(a, b) for kind, _, a, b in items]
        for (a1, b1), (a2, b2) in zip(special[:-1], special[1:]):
            if a2 < b1 - 0.02 * T:
                return None                # two special arcs at once
        frozen = {it[1] for it in items if it[0] == "singular"}
        # Per frozen control: (time, label) of its free-arc samples.
        free = np.ones(len(t), dtype=bool)
        for a, b in special:
            free &= ~((t >= a) & (t <= b))
        events = []                        # (time, entry)
        for i, (kind, c, a, b) in enumerate(items):
            if kind != "singular":
                events.append((a, (kind, c, a, b)))
                continue
            prev = items[i - 1] if i else None
            joined = prev is not None and prev[0] == "boundary" and a - prev[3] < 0.02 * T
            nxt = items[i + 1] if i + 1 < len(items) else None
            if nxt is not None and nxt[2] - b < 0.02 * T:
                return None                # singular -> boundary joins are not supported
            events.append((a, ("singular", c, a, b, joined)))
        # Labels on free arcs for frozen controls: segments and switches.
        held = {}
        for u in frozen:
            lab = read["labels"][u]
            segs = []                      # (start, end, label) of free labeled stretches
            for value in (1, -1):
                for a, b in self._runs(t, free & (lab == value)):
                    segs.append((a, b, value))
            segs.sort()
            held[u] = segs
        timeline = sorted(events, key=lambda e: e[0])
        # Switch events for frozen controls: label flips between consecutive
        # free segments with no singular arc of that control in between.
        for u in frozen:
            segs = held[u]
            sing = [(e[1][2], e[1][3]) for e in timeline if e[1][0] == "singular" and e[1][1] == u]
            for (a1, b1, l1), (a2, b2, l2) in zip(segs[:-1], segs[1:]):
                if l1 != l2 and not any(b1 - 1e-12 <= x <= a2 + 1e-12 or b1 <= y <= a2
                                        for x, y in sing):
                    ts = 0.5 * (b1 + a2)
                    inside = [(x, y) for kind, c, x, y in items if kind != "singular"
                              and x <= ts <= y]
                    if inside:
                        ts = inside[0][1] + 1e-3 * T
                    timeline.append((ts, ("switch", u, ts)))
        timeline.sort(key=lambda e: e[0])

        def label_near(u, time, before):
            segs = held[u]
            if before:
                cand = [s for s in segs if s[1] <= time + 1e-12]
                return cand[-1][2] if cand else None
            cand = [s for s in segs if s[0] >= time - 1e-12]
            return cand[0][2] if cand else None

        sides = []
        for _, ev in timeline:
            if ev[0] == "singular":
                _, u, a, b, joined = ev
                start = a
                if joined:
                    start = [x for kind, c, x, y in items if kind == "boundary" and y <= a + 0.02 * T][-1]
                before = label_near(u, start, True)
                after = label_near(u, b, False)
                sides.append((before, after))
        return timeline, sides

    def _assemble(self, timeline, sides_choice, read, base, jitter_rng=None):
        """Arc entries and their values; ``jitter_rng`` perturbs the times."""
        T, t0, tf = read["T"], read["t0"], read["tf"]
        times = []
        for _, ev in timeline:
            if ev[0] in ("boundary", "touch"):
                times += [ev[2], ev[3]] if ev[0] == "boundary" else [0.5 * (ev[2] + ev[3])]
            elif ev[0] == "singular":
                times += [ev[3]] if ev[4] else [ev[2], ev[3]]
            else:
                times.append(ev[2])
        times = np.array(times, dtype=float)
        lo, hi = t0 + 0.01 * T, tf - 0.01 * T
        times = np.clip(times, lo, hi)
        if jitter_rng is not None and len(times):
            grid = np.concatenate([[t0], times, [tf]])
            gaps = np.minimum(np.diff(grid)[:-1], np.diff(grid)[1:])
            times = times + jitter_rng.uniform(-0.5, 0.5, len(times)) * np.maximum(gaps, 0.02 * T)
            times = np.clip(times, lo, hi)
        if np.any(np.diff(times) <= 0):
            return None
        arcs, values, k = [], [], 0
        si = 0
        for _, ev in timeline:
            kind = ev[0]
            if kind == "boundary":
                a, b = times[k], times[k + 1]
                k += 2
                arcs.append(("boundary", ev[1]))
                values.append([a, b, None])
            elif kind == "touch":
                arcs.append(("touch", ev[1]))
                values.append([times[k], None])
                k += 1
            elif kind == "singular":
                before, after = sides_choice[si]
                si += 1
                name = ev[1].name
                if ev[4]:
                    arcs.append(("singular", name, before, after, "joined"))
                    values.append(times[k])
                    k += 1
                else:
                    arcs.append(("singular", name, before, after))
                    values.append((times[k], times[k + 1]))
                    k += 2
            else:
                arcs.append(("switch", ev[1].name))
                values.append(times[k])
                k += 1
        return arcs, values

    def _fit_jumps(self, shoot, values, read):
        """Fill in boundary ``pi`` and touch ``nu`` from the regularized costates."""
        kx, p, n = shoot.kernels, shoot.params, shoot.kernels.N_STATES
        t, Z = read["t"], read["z"]

        def z_at(ti):
            return np.array([np.interp(ti, t, Z[:, i]) for i in range(Z.shape[1])])

        out = []
        for (kind, c, _), val in zip(shoot.arcs, values):
            if kind == "boundary":
                a, b, _ = val
                q = kx.CONSTRAINT_ORDERS[c]
                pi = np.zeros(q)
                try:
                    za = z_at(a)
                    back = kx.propagate_boundary(c, b, z_at(b), a, p, **self.options)["z"]
                    _, Nx, _, _ = kx.tangency_gradient(c, a, za, p)
                    pi = np.linalg.lstsq(Nx.T, za[n:] - back[n:], rcond=None)[0]
                except _FAILURES:
                    pass
                out.append((a, b, pi))
            elif kind == "touch":
                tc = val[0]
                nu = 0.0
                try:
                    _, Nx, _, _ = kx.tangency_gradient(c, tc, z_at(tc), p)
                    w = 0.02 * read["T"]
                    jump = z_at(tc - w)[n:] - z_at(tc + w)[n:]
                    nu = float(np.linalg.lstsq(Nx[:1].T, jump, rcond=None)[0][0])
                except _FAILURES:
                    pass
                out.append((tc, nu))
            else:
                out.append(val)
        return out

    def _accept(self, shoot, sol, tol):
        """Whether an exact solution passes the checks it can make."""
        if not sol.success or not np.isfinite(sol.residual_norm):
            return False
        if sol.max_violation > 1e-7:
            return False
        if not isinstance(shoot, MultiArcShooting) or not shoot.arcs:
            return True
        try:
            d = shoot.diagnostics(sol.v)
        except _FAILURES:
            return False
        small = 1e-7
        if any(nu < -small for nu in d["touch_nu"]):
            return False
        if any(lo < -small * max(1.0, abs(hi)) for lo, hi in d["boundary_mu"]):
            return False
        if any(lo < -small * max(1.0, abs(hi)) for lo, hi in d["singular_legendre"]):
            return False
        if any(w > 1e-6 for w in d["held_sign_violation"].values()):
            return False
        return True

    def _attempt(self, read, base, n_jitter, rng, tol):
        """Exact solves of every candidate from one reading: ``[(J, shoot, sol)]``."""
        kx, px = self.exact_kernels, self.exact_params()
        candidates, possible = self._candidates(read)
        tried, accepted = 0, []
        seen = set()
        for items in candidates:
            built = self._structure(items, read)
            if built is None:
                continue
            timeline, sides = built
            if any(before is None and after is None for before, after in sides):
                continue                   # the control never reaches a bound: not ready
            side_options = []
            for before, after in sides:
                b_opts = ["upper", "lower"] if before is None else [
                    "upper" if before > 0 else "lower"]
                a_opts = ["lower", "upper"] if after is None else [
                    "upper" if after > 0 else "lower"]
                side_options.append([(b, a) for b in b_opts for a in a_opts])
            for sides_choice in itertools.product(*side_options) if side_options else [()]:
                for j in range(n_jitter + 1):
                    made = self._assemble(timeline, sides_choice, read, base,
                                          None if j == 0 else rng)
                    if made is None:
                        continue
                    arcs, values = made
                    try:
                        if arcs:
                            shoot = MultiArcShooting(kx, px, arcs, max_nfev=self.attempt_nfev,
                                                     **self.options)
                            vals = self._fit_jumps(shoot, values, read)
                            guess = shoot.pack_arcs(base, vals)
                            method = "lm" if kx.N_SWITCHING else "hybr"
                        else:
                            key = ("free",)
                            if key in seen:
                                continue
                            seen.add(key)
                            shoot = SingleShooting(kx, px, **self.options)
                            guess = np.asarray(base, dtype=float)
                            method = "hybr"
                        sol = shoot.solve(guess, method=method, tol=tol,
                                          options=_budget(method, self.attempt_evaluations))
                    except _FAILURES:
                        continue
                    tried += 1
                    if self._accept(shoot, sol, tol):
                        try:
                            J = self.objective(shoot, sol.v)
                        except _FAILURES:
                            continue
                        accepted.append((J, shoot, sol))
                        break              # this structure is solved; next candidate
        return accepted, tried, possible

    # -- the whole solve --------------------------------------------------------

    def solve(self, params, costates, final_time=None, *, n_tries=200, seed=0, eps0=1.0,
              s_max=10.0, max_steps=300, n_jitter=6, tol=1e-10):
        """From a costate distribution to the exact extremal.

        ``params`` are the values of ``problem.params``; penalty weights the
        problem defines are not used (:func:`regularize` replaces them), so
        any value will do for them. ``costates`` is ``(low, high)`` (uniform
        per component) or a function ``rng -> lam0``; ``final_time``
        likewise for ``t_f`` (required when it is free). At most ``n_tries``
        random draws are made, from ``seed`` (an int or a
        ``numpy.random.Generator``). ``eps0`` is the first smoothing weight
        and ``s_max`` how many decades ``eps`` may fall; ``n_jitter`` is the
        number of extra seeds, with perturbed junction times, per candidate
        structure. Returns a :class:`HomotopyResult`.
        """
        params = np.asarray(params, dtype=float).reshape(-1)
        if len(params) != len(self.problem.params):
            raise ValueError(f"expected {len(self.problem.params)} parameters "
                             f"{[str(p) for p in self.problem.params]}, got {len(params)}")
        self.params = params
        if self.kr.FREE_FINAL_TIME and final_time is None:
            raise ValueError("the final time is free: give a final_time distribution")
        rng = np.random.default_rng(seed)
        kr = self.kr
        eps_start = [eps0, 10 * eps0, 0.1 * eps0] if self.bang_bang else [None]
        zero = np.zeros(self.nc)
        sol, tries = None, 0
        for e0 in eps_start:
            shoot = SingleShooting(kr, self.regularized_params(e0, zero), **self.options)
            sol, n = shoot.multistart(self._sampler(shoot, costates, final_time),
                                      n_tries=n_tries, seed=rng, tol=tol)
            tries += n
            if sol is not None and sol.success:
                eps0 = e0
                break
        if sol is None or not sol.success:
            return HomotopyResult(sol, None, (), False, np.nan, None, [], tries,
                                  "no random draw solved the regularized problem")
        self._say(f"regularized solve after {tries} draws, eps = {eps0}")

        # Step 3: continuation in phases. The penalty of a newly violated limit
        # ramps up first, with eps held (log10 rho_c from rho_ref_c 10^m_start
        # to rho_ref_c 10^m_max); then eps falls, with rho held.
        log_eps = np.log10(eps0) if self.bang_bang else 0.0
        log_eps_min = log_eps - s_max
        rho_ref = np.zeros(self.nc)
        m = np.full(self.nc, -np.inf)      # log10(rho / rho_ref); -inf: penalty off
        m_start, m_max = -6.0, 4.0

        def params_at(le, mm):
            eps = 10.0 ** le if self.bang_bang else None
            rho = np.where(np.isfinite(mm), rho_ref * 10.0 ** mm, 0.0)
            return self.regularized_params(eps, rho), eps, rho

        def activate(shoot, v):
            """Switch on the penalty of every newly violated limit (at m_start)."""
            if not self.nc:
                return False
            viol = shoot.constraint_violation(v)
            new = (viol > 1e-9) & ~np.isfinite(state["m"])
            if not new.any():
                return False
            x0, lam0, _, tf = shoot.unpack(v)
            tr = kr.trajectory(shoot.t0, np.concatenate([x0, lam0]), tf, shoot.params,
                               n_points=2001, **self.options)
            J = abs(self._regularized_cost(shoot, v)) or 1.0
            for c in np.flatnonzero(new):
                P = np.trapezoid(np.maximum(tr["g"][:, c], 0.0) ** 2 / 2, tr["t"])
                rho_ref[c] = J / max(P, 1e-300)
                self._say(f"limit {kr.CONSTRAINT_NAMES[c]} violated: rho_ref = {rho_ref[c]:.3g}")
            return new

        steps = []
        state = {"le": log_eps, "m": m.copy(), "v": sol.v, "sol": sol, "prev": None}
        best = None

        def record_and_attempt(le, mm, v, sol):
            p, eps, rho = params_at(le, mm)
            shoot = SingleShooting(kr, p, **self.options)
            step = Step(log_eps - le, eps if eps is not None else np.nan, rho.copy(),
                        sol.residual_norm)
            steps.append(step)
            read = self._read(shoot, v)
            accepted, tried, _ = self._attempt(read, v[:shoot.n_unknowns], n_jitter, rng, tol)
            step.candidates = tried
            self._say(f"eps = {eps}  rho = {rho}  candidates tried: {tried}, "
                      f"accepted: {len(accepted)}")
            if accepted:
                step.accepted = True
            return shoot, accepted

        def advance(move, size):
            """One adaptive step along ``move(state, size) -> (le, m)``: the new
            state, or None when the step falls below its floor."""
            ds = size
            while ds >= 1e-3:
                le, mm = move(ds)
                p, _, _ = params_at(le, mm)
                nxt = SingleShooting(kr, p, **self.options)
                v, prev = state["v"], state["prev"]
                guesses = [v]
                if prev is not None and prev[2] > 0:
                    guesses.insert(0, v + (v - prev[1]) * ds / prev[2])
                for g in guesses:
                    try:
                        r = nxt.solve(g, tol=tol)
                    except _FAILURES:
                        continue
                    if r.success:
                        state.update(le=le, m=mm, prev=(None, v, ds), v=r.v, sol=r)
                        return nxt, ds
                ds *= 0.5
            return None, ds

        if not self.bang_bang and not self.nc:
            # The regularized problem is the exact one.
            J = self.objective(shoot, sol.v)
            return HomotopyResult(sol, shoot, (), True, J, shoot.params,
                                  [Step(0.0, np.nan, np.zeros(0), sol.residual_norm)], tries,
                                  "exact: no bang-bang controls or path constraints")
        new = activate(shoot, sol.v)
        if new is not False:
            state["m"][new] = m_start
        shoot, accepted = record_and_attempt(state["le"], state["m"], state["v"], state["sol"])
        ds_rho, ds_eps = 0.5, 0.25
        done = np.zeros(self.nc, dtype=bool)
        while not accepted and len(steps) < max_steps:
            best = (shoot, state["sol"])
            ramping = np.isfinite(state["m"]) & (state["m"] < m_max) & ~done
            if ramping.any():
                def move(ds):
                    mm = state["m"].copy()
                    mm[ramping] = np.minimum(mm[ramping] + ds, m_max)
                    return state["le"], mm
                nxt, ds = advance(move, ds_rho)
                if nxt is None:
                    # The penalty stalls: hold it where it is and go on lowering eps.
                    self._say("penalty continuation stalled")
                    done |= ramping
                    state["prev"] = None
                    continue
                ds_rho = min(ds * 1.5, 1.0)
            elif self.bang_bang and state["le"] > log_eps_min:
                def move(ds):
                    return max(state["le"] - ds, log_eps_min), state["m"]
                nxt, ds = advance(move, ds_eps)
                if nxt is None:
                    break
                ds_eps = min(ds * 1.5, 1.0)
            else:
                break
            new = activate(nxt, state["v"])
            if new is not False:
                state["m"][new] = m_start
                state["prev"] = None
            shoot, accepted = record_and_attempt(state["le"], state["m"], state["v"],
                                                 state["sol"])
        if accepted:
            J, xs, xsol = min(accepted, key=lambda a: a[0])
            return HomotopyResult(xsol, xs, self._entries(xs), True, J, xs.params, steps,
                                  tries, f"exact after {len(steps)} steps")
        shoot, sol = best if best is not None else (shoot, state["sol"])
        return HomotopyResult(sol, shoot, (), False, np.nan, shoot.params, steps, tries,
                              "no exact structure accepted before the continuation ended")

    def _regularized_cost(self, shoot, v):
        x0, lam0, _, tf = shoot.unpack(v)
        r = shoot.kernels.propagate(shoot.t0, np.concatenate([x0, lam0]), tf, shoot.params,
                                    cost=True, **self.options)
        xf = r["z"][:shoot.kernels.N_STATES]
        return float(r["cost"]) + float(self._terminal(tf, list(xf), list(self.params)))

    @staticmethod
    def _entries(shoot):
        """The structure as MultiArcShooting would be given it."""
        if not isinstance(shoot, MultiArcShooting):
            return ()
        k = shoot.kernels
        out = []
        for kind, c, extra in shoot.arcs:
            if kind in ("boundary", "touch"):
                out.append((kind, k.CONSTRAINT_NAMES[c]))
                continue
            name = k.system.switching_functions[c].control.name
            sides = tuple("upper" if f else "lower" for f in extra)
            if kind == "switch":
                out.append(("switch", name))
            elif kind == "singular_joined":
                out.append(("singular", name, *sides, "joined"))
            else:
                out.append(("singular", name, *sides))
        return tuple(out)


def _budget(method, n):
    """MINPACK options with an evaluation budget."""
    if method == "lm":
        return {"xtol": 1e-14, "ftol": 1e-14, "maxiter": int(n)}
    return {"xtol": 1e-13, "maxfev": int(n)}


def solve(problem, params, costates, final_time=None, *, cache_dir=None, verbose=False,
          **options):
    """:meth:`Homotopy.solve` in one call: see :class:`Homotopy` and the module docs."""
    return Homotopy(problem, cache_dir=cache_dir, verbose=verbose).solve(
        params, costates, final_time, **options)
