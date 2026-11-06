"""Single shooting on the compiled kernels: residuals, exact Jacobians, and a solver.

This module is an optional layer over :class:`~auto_optimal_control.build.Kernels`.
The rest of the package does not depend on it. The forward map,
``Kernels.propagate``, stays the primitive: give it initial costates and it
returns the extremal. :class:`SingleShooting` wraps that map as a square
nonlinear system ``R(v) = 0``.

Any root finder or sampler can drive it. :meth:`SingleShooting.residual` and
:meth:`SingleShooting.residual_jacobian` need only NumPy, and
:meth:`SingleShooting.solve` adds SciPy's MINPACK. Install SciPy with the
``shooting`` extra: ``pip install auto-optimal-control[shooting]``.

The unknowns ``v`` are, in order:

- the free components of the initial state;
- the initial costates ``lam0``;
- the terminal multipliers ``nu``;
- ``t_f``, if the final time is free.

The fixed initial states are eliminated, so their residual rows are dropped
and the system stays square. The Jacobian is exact. It chains the generated
boundary-condition Jacobians with the STM that ``propagate`` integrates,
switching jumps included, and with ``dz/dt_f`` when the final time is free.

Path constraints enter in two ways. A penalty weight is a runtime parameter,
so :func:`parameter_continuation` can raise it from solve to solve. And
``terminate`` makes any propagation that violates a constraint fail with
:class:`ShootingError`, which :meth:`SingleShooting.solve` reports to the
root finder as a large residual.

:class:`MultiArcShooting` solves for exact constrained and singular extremals
instead: given the sequence of boundary arcs, touch points and singular arcs,
its unknowns add the junction times and the costate-jump multipliers, and its
Jacobian chains the STM across the arcs. :meth:`MultiArcShooting.from_penalty`
seeds it from a penalty solution, :meth:`MultiArcShooting.from_smoothed` from
a smoothed problem, given the structure.
:mod:`auto_optimal_control.homotopy` finds the structure itself.
"""

from dataclasses import dataclass

import numpy as np


MAX_NFEV = 1000000                 # right-hand-side evaluations per propagation




class ShootingError(FloatingPointError):
    """The extremal could not be propagated from these unknowns."""


@dataclass
class ShootingSolution:
    """The result of :meth:`SingleShooting.solve`.

    ``success`` means the residual norm reached the requested tolerance. A
    nonzero residual can still be a converged *local* solve; MINPACK's own
    report is in ``result``. ``max_violation`` is the largest ``max(0, g)``
    over the path constraints along the solution (0 without constraints);
    a penalized solution violates its limits slightly.
    """

    v: np.ndarray
    x0: np.ndarray
    lam0: np.ndarray
    nu: np.ndarray
    t0: float
    tf: float
    residual_norm: float
    success: bool
    nfev: int
    result: object = None
    max_violation: float = 0.0

    @property
    def z0(self):
        return np.concatenate([self.x0, self.lam0])


class SingleShooting:
    """The single-shooting system of one problem at fixed parameters.

    Parameters
    ----------
    kernels : Kernels
        From :func:`~auto_optimal_control.build.build_kernels`.
    params : sequence of float
        The runtime parameters, in the order of ``kernels.param_names``.
    x0 : sequence of float, optional
        The fixed initial state. Defaults to the problem's. Entries for free
        initial states are ignored; they are unknowns.
    method, rtol, atol, max_switches, terminate, n_interior
        Passed to ``Kernels.propagate``. With ``terminate``, unknowns whose
        extremal violates one of those path constraints raise
        :class:`ShootingError`.
    """

    def __init__(self, kernels, params=(), *, x0=None, method="DOP853", rtol=1e-12, atol=1e-12,
                 max_switches=10000, terminate=False, n_interior=4):
        self.kernels = k = kernels
        self.params = np.asarray(params, dtype=float).reshape(-1)
        if len(self.params) != k.N_PARAMS:
            raise ValueError(f"expected {k.N_PARAMS} parameters {k.param_names}, "
                             f"got {len(self.params)}")
        problem = k.system.problem
        values = dict(zip(k.system.params, self.params))
        self.t0 = float(problem.initial_time.subs(values))
        self.fixed_tf = None if k.FREE_FINAL_TIME else float(problem.final_time.subs(values))

        n = k.N_STATES
        default = k.initial_state(self.params)
        self.free_states = np.flatnonzero(np.isnan(default))
        x = default if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
        if x.shape != (n,):
            raise ValueError(f"x0 must have {n} entries")
        x[self.free_states] = 0.0
        if np.isnan(x).any():
            raise ValueError("x0 is NaN at a fixed initial state")
        self.x0 = x
        self.options = dict(method=method, rtol=rtol, atol=atol, max_switches=max_switches,
                            terminate=terminate, n_interior=n_interior)

        self.n_free = len(self.free_states)
        self.n_unknowns = self.n_free + n + k.N_MULTIPLIERS + int(k.FREE_FINAL_TIME)
        # Residual rows kept: lambda_i(t0) = 0 for free initial states, then
        # every terminal row (constraints, transversality, H(tf)).
        self._rows = np.concatenate([self.free_states, np.arange(n, k.N_RESIDUALS)])
        # Columns of the z0 derivative that are unknowns.
        self._zcols = np.concatenate([self.free_states, np.arange(n, 2 * n)])
        n_terminal = len(k.system.boundary.terminal)
        self._transversality = slice(n + n_terminal, 2 * n + n_terminal)

    # Tolerance of the state transition matrix, when it may be looser than the state's
    # (None: the same). Steps fall by half or more, but a Jacobian this rough cannot bring
    # the residual below about this level times the unknowns' size.
    jacobian_rtol = None
    max_nfev = MAX_NFEV
    # Residuals apart from the Jacobians. MINPACK asks for far more of the first, and without
    # the matrix they cost a small fraction as much. But their steps are longer, and each
    # saturation corner carries the error of its step, so they reach only about 1e-9.
    fast_residuals = False

    def __repr__(self):
        return (f"<SingleShooting {self.kernels.module.__name__}: {self.n_unknowns} unknowns, "
                f"params={self.params.tolist()}>")

    # -- unknowns -----------------------------------------------------------

    def pack(self, lam0, nu=None, tf=None, x0_free=None):
        """Assemble ``v``. ``nu`` defaults to zeros; ``tf`` is required iff it is free."""
        k = self.kernels
        lam0 = np.asarray(lam0, dtype=float).reshape(-1)
        if lam0.shape != (k.N_STATES,):
            raise ValueError(f"lam0 must have {k.N_STATES} entries")
        nu = np.zeros(k.N_MULTIPLIERS) if nu is None else np.asarray(nu, dtype=float).reshape(-1)
        if nu.shape != (k.N_MULTIPLIERS,):
            raise ValueError(f"nu must have {k.N_MULTIPLIERS} entries")
        xf = np.zeros(self.n_free) if x0_free is None else np.asarray(x0_free, float).reshape(-1)
        if xf.shape != (self.n_free,):
            raise ValueError(f"x0_free must have {self.n_free} entries")
        parts = [xf, lam0, nu]
        if k.FREE_FINAL_TIME:
            if tf is None:
                raise ValueError("the final time is free: pass a guess for tf")
            parts.append([float(tf)])
        elif tf is not None:
            raise ValueError("the final time is fixed by the problem")
        return np.concatenate(parts)

    def unpack(self, v):
        """``(x0, lam0, nu, tf)`` from ``v``; ``x0`` is the full initial state."""
        k = self.kernels
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.shape != (self.n_unknowns,):
            raise ValueError(f"v must have {self.n_unknowns} entries")
        m, n = self.n_free, k.N_STATES
        x0 = self.x0.copy()
        x0[self.free_states] = v[:m]
        lam0 = v[m:m + n]
        nu = v[m + n:m + n + k.N_MULTIPLIERS]
        tf = v[m + n + k.N_MULTIPLIERS] if k.FREE_FINAL_TIME else self.fixed_tf
        return x0, lam0, nu, float(tf)

    # -- the system ---------------------------------------------------------

    def _propagate(self, z0, tf, stm):
        if not tf > self.t0:
            raise ShootingError(f"final time {tf} is not after the initial time {self.t0}")
        # A bound on the work: coincident saturation corners can otherwise never finish.
        tolerance = {"stm_rtol": self.jacobian_rtol} if stm else {}
        r = self.kernels.propagate(self.t0, z0, tf, self.params, stm=stm, **tolerance,
                                   max_nfev=self.max_nfev, **self.options)
        if r["status"] == self.kernels.STATUS_VIOLATED:
            raise ShootingError(f"path constraint violated at t = {r['t']}")
        if r["status"] != 0:
            raise ShootingError(f"propagation failed with status {r['status']}")
        return r

    def residual(self, v):
        """``R(v)``. Raises :class:`ShootingError` if propagation fails."""
        x0, lam0, nu, tf = self.unpack(v)
        z0 = np.concatenate([x0, lam0])
        zf = self._propagate(z0, tf, stm=False)["z"]
        return self.kernels.residual(z0, zf, nu, tf, self.params)[self._rows]

    def residual_jacobian(self, v):
        """``(R(v), dR/dv)``, with the exact Jacobian."""
        k = self.kernels
        x0, lam0, nu, tf = self.unpack(v)
        z0 = np.concatenate([x0, lam0])
        r = self._propagate(z0, tf, stm=True)
        zf = r["z"]
        R = k.residual(z0, zf, nu, tf, self.params)[self._rows]
        J = k.residual_jacobian(z0, zf, nu, tf, self.params)
        Rz0, Rzf = J["z0"][self._rows], J["zf"][self._rows]
        cols = [(Rz0 + Rzf @ r["stm"])[:, self._zcols], J["nu"][self._rows]]
        if k.FREE_FINAL_TIME:
            cols.append((J["tf"][self._rows] + Rzf @ r["dz_dt1"])[:, None])
        return R, np.hstack(cols)

    __call__ = residual_jacobian

    def guess_multipliers(self, lam0, tf=None, x0_free=None):
        """Least-squares ``nu`` for a costate guess.

        The transversality residuals are linear in ``nu``. This propagates the
        guess and fits ``nu`` to them, so only ``lam0`` (and ``tf``) need
        guessing. It returns ``pack(lam0, nu, tf, x0_free)``.
        """
        k = self.kernels
        v = self.pack(lam0, None, tf, x0_free)
        if k.N_MULTIPLIERS == 0:
            return v
        x0, lam0, nu, tf = self.unpack(v)
        z0 = np.concatenate([x0, lam0])
        zf = self._propagate(z0, tf, stm=False)["z"]
        R = k.residual(z0, zf, nu, tf, self.params)[self._transversality]
        A = k.residual_jacobian(z0, zf, nu, tf, self.params)["nu"][self._transversality]
        nu = np.linalg.lstsq(A, -R, rcond=None)[0]
        return self.pack(lam0, nu, None if self.fixed_tf is not None else tf, x0_free)

    # -- solving ------------------------------------------------------------

    def _safe(self, v):
        try:
            return self.residual_jacobian(v)
        except (ShootingError, RuntimeError, ValueError):
            # A large residual makes MINPACK step back without stopping it.
            m = len(v)
            return np.full(m, 1e6), np.eye(m)

    def _safe_residual(self, v):
        try:
            return self.residual(v)
        except (ShootingError, RuntimeError, ValueError):
            return np.full(len(v), 1e6)

    def solve(self, guess, method="hybr", tol=1e-10, options=None):
        """Solve ``R(v) = 0`` from ``guess`` with ``scipy.optimize.root``.

        ``method`` is ``"hybr"`` (Powell hybrid, the default) or ``"lm"``
        (Levenberg-Marquardt), and both use the exact Jacobian. Unknowns whose
        propagation fails are reported to the solver as a large residual,
        so a poor guess fails the solve without raising. ``success`` means
        ``||R|| < tol``.
        """
        try:
            from scipy.optimize import root
        except ImportError as exc:      # pragma: no cover
            raise ImportError("SingleShooting.solve needs SciPy: "
                              "pip install auto-optimal-control[shooting]") from exc
        if options is None:
            options = {"xtol": 1e-13} if method == "hybr" else {"xtol": 1e-14, "ftol": 1e-14}
        if self.fast_residuals:
            sol = root(self._safe_residual, np.asarray(guess, dtype=float),
                       jac=lambda v: self._safe(v)[1], method=method, options=options)
            err = float(np.linalg.norm(self._safe_residual(sol.x)))
        else:
            sol = root(self._safe, np.asarray(guess, dtype=float), jac=True, method=method,
                       options=options)
            err = float(np.linalg.norm(self._safe(sol.x)[0]))
        x0, lam0, nu, tf = self.unpack(sol.x)
        violation = 0.0
        if err < tol:
            try:
                violation = self._max_violation(sol.x)
            except (ShootingError, RuntimeError, ValueError):
                err = violation = float("inf")      # the solution does not propagate again
        return ShootingSolution(v=sol.x, x0=x0, lam0=lam0.copy(), nu=nu.copy(), t0=self.t0, tf=tf,
                                residual_norm=err, success=err < tol,
                                nfev=int(getattr(sol, "nfev", 0)), result=sol,
                                max_violation=violation)

    def multistart(self, sample, n_tries=50, seed=0, **solve_options):
        """Solve from random guesses ``sample(rng) -> v`` until one succeeds.

        ``seed`` is an int or a ``numpy.random.Generator`` (to continue one
        stream across calls). Returns ``(solution, tries)``. The solution is
        the first success, or else the smallest residual found. A ``sample`` that raises
        :class:`ShootingError` (for example in :meth:`guess_multipliers`)
        skips that draw.
        """
        rng = np.random.default_rng(seed)
        best, tries = None, 0
        for tries in range(1, n_tries + 1):
            try:
                guess = sample(rng)
            except (ShootingError, RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
            sol = self.solve(guess, **solve_options)
            if best is None or sol.residual_norm < best.residual_norm:
                best = sol
            if sol.success:
                break
        return best, tries

    def _max_violation(self, v):
        if not self.kernels.N_CONSTRAINTS:
            return 0.0
        return float(np.max(self.constraint_violation(v)))

    def constraint_violation(self, v, n_points=2001):
        """Per path constraint, the largest ``max(0, g)`` along the extremal at ``v``.

        Samples the dense output at ``n_points`` times, plus the midpoints of
        the intervals on which a constraint is violated, between its crossings.
        """
        k = self.kernels
        if not k.N_CONSTRAINTS:
            return np.zeros(0)
        x0, lam0, _, tf = self.unpack(v)
        opts = {key: val for key, val in self.options.items() if key != "terminate"}
        tr = k.trajectory(self.t0, np.concatenate([x0, lam0]), tf, self.params,
                          n_points=n_points, **opts)
        ts = np.concatenate([[self.t0], tr["constraint_times"], [tf]])
        mids = 0.5 * (ts[:-1] + ts[1:])
        extra = k.trajectory(self.t0, np.concatenate([x0, lam0]), tf, self.params,
                             t_eval=mids, **opts)["g"] if len(mids) else np.zeros((0, 1))
        g = np.vstack([tr["g"], extra])
        return np.maximum(g.max(axis=0), 0.0)

    def trajectory(self, v, **kwargs):
        """The extremal at ``v``, sampled; see ``Kernels.trajectory``."""
        x0, lam0, _, tf = self.unpack(v)
        return self.kernels.trajectory(self.t0, np.concatenate([x0, lam0]), tf, self.params,
                                       **kwargs)


class MultiArcShooting(SingleShooting):
    """Exact extremals with constrained and singular arcs: shooting over a fixed sequence of arcs.

    ``arcs`` lists the special arcs and junctions in time order. Free arcs,
    where no limit is active and every control follows its law, lie between
    them and at both ends. Each entry is one of:

    - ``("boundary", c)``: the extremal rides path constraint ``c`` (``g = 0``),
      for a constraint of order 1 or 2. Unknowns: the entry and exit times
      and the jump multipliers ``pi`` (one per order). Conditions:
      ``N = (g, ..., g^(q-1)) = 0`` at the entry, and ``H`` continuous at both
      junctions. The costates jump by ``-dN/dx^T pi`` at the entry and are
      continuous at the exit (indirect adjoining, Bryson, Denham & Dreyfus
      1963; Maurer 1977).
    - ``("touch", c)``: an isolated contact with an order-2 constraint.
      Unknowns: the contact time and the jump ``nu`` (``lambda+ = lambda- -
      nu dg/dx``). Conditions: ``g = g' = 0`` there. ``nu >= 0`` is
      necessary for optimality; see :meth:`diagnostics`.
    - ``("singular", u, before, after)``: switching function ``S`` of the
      bang-bang control ``u`` vanishes on an interval, an order-1 singular
      arc (kernels from ``derive(problem, singular=True)``). ``before`` and
      ``after`` are ``"lower"`` or ``"upper"``: the bound ``u`` holds on the
      free arcs on either side. Unknowns: the entry and exit times.
      Conditions: ``S = S' = 0`` at the entry. The singular control keeps
      them at zero along the arc, and ``H`` is continuous at both junctions
      because ``S = 0`` there, so the exit time is fixed by the conditions
      downstream.
    - ``("singular", u, before, after, "joined")``: a singular arc that
      starts where the boundary arc just before it ends, with no free arc
      between (riding a limit, then singular). Unknown: the exit time.
      Conditions: ``S = S' = 0`` at the junction, which replace the
      boundary arc's ``H`` continuity at its exit (``H`` is linear in ``u``
      and ``S = 0``, so it holds already). ``before`` is still the bound
      held on the last free arc before.
    - ``("switch", u)``: a bang-bang switch of a control that has a
      singular arc. Unknown: the switch time. Condition: ``S = 0`` there.
    - ``("enter", c)`` and ``("exit", c)``: a boundary arc given by its two
      junctions, so that arcs of different limits can overlap or nest:
      ``[("enter", a), ("enter", b), ("exit", a), ("exit", b)]`` rides ``a``,
      then ``a`` and ``b`` together, then ``b``. While several limits are
      ridden the segment follows their set's kernels
      (``derive(problem, simultaneous=...)``; ``kernels.boundary_set``). The
      unknowns are each arc's ``(t_in, t_out, pi)``, in the order the arcs
      are entered. ``("boundary", c)`` is ``("enter", c), ("exit", c)``.

    At the junctions of an order-1 limit, when ``H`` is regular (no
    bang-bang controls) and the free side's controls are strictly inside
    their bounds there, the control and the costates are continuous
    (Jacobson, Lele & Speyer 1971). There ``H`` continuity is replaced by
    conditions with simple roots: ``dg/dt = 0`` just before the entry (the
    arrival is tangential) and ``mu = 0`` at the exit, which gives exact
    junction times where ``H`` continuity converges only linearly.
    ``junctions="hamiltonian"`` keeps ``H`` continuity everywhere.

    ``c`` is a constraint index or name, ``u`` a switching-function index or
    the control's name. ``max_nfev`` caps the right-hand-side evaluations
    of each arc's propagation, so unknowns whose trajectory runs into a
    singularity fail fast (0 for no cap). The unknowns are :class:`SingleShooting`'s,
    followed by each arc's in order. The path constraints are not watched on
    the free arcs (their penalties stay off), so a wrong structure shows up
    as a violation on a free arc, reported by :meth:`diagnostics` and in
    ``max_violation``.

    Bounded controls (:class:`~auto_optimal_control.problem.Box`) are
    supported. On the free arcs they follow their usual law: clipped where
    ``H`` is quadratic in them, bang-bang where it is linear, with the
    switches located (and the STM jumped) as in :class:`SingleShooting`. A
    control with a singular arc in the structure is the exception: on every
    free arc it holds the bound the structure gives it, and its switches are
    the ``"switch"`` entries. Its switching function is not watched there,
    since it touches zero at a singular junction (``S = S' = 0``), where
    event location is ill-conditioned; :meth:`diagnostics` reports whether
    its sign agrees with the bound held.

    On a boundary arc the control is the one that rides the limit, and on a
    singular arc the singular control; either must stay within its bounds,
    and how far it leaves them counts toward ``max_violation`` (see
    :meth:`diagnostics`). Unit-vector control sets are not supported.

    :meth:`from_penalty` builds the structure's initial guess from an
    exterior-penalty solution, and :meth:`from_smoothed` the times of
    singular arcs and switches from a smoothed problem (``eps u^2 / 2``
    added to the running cost) whose control approaches the bang-singular
    one as ``eps`` falls. A bang-bang problem with a path constraint has no
    useful penalty solution: there the penalized optimum rides the limit on
    a singular arc. Smooth it instead, solve that with exact arcs and lower
    ``eps`` by :func:`continuation`. The unknowns of the two problems match,
    so the smoothed ``v`` is a starting point for the bang-bang one.
    """

    _SIZES = {"touch": 2, "singular": 2, "singular_joined": 1, "switch": 1}

    def __init__(self, kernels, params=(), arcs=(), *, x0=None, method="DOP853", rtol=1e-12,
                 atol=1e-12, max_nfev=200000, junctions="auto"):
        super().__init__(kernels, params, x0=x0, method=method, rtol=rtol, atol=atol,
                         n_interior=0)
        if junctions not in ("auto", "hamiltonian"):
            raise ValueError("junctions must be 'auto' or 'hamiltonian'")
        self.junctions = junctions
        k = kernels
        self.arcs = []
        self.events = []                 # ("in" | "out" | "touch" | "legacy", arc index)
        open_arcs = {}                   # constraint -> index of its arc entered, not yet left
        sides = {}                       # singular control -> [(before, after)] in order
        for item in arcs:
            kind, c, *rest = item
            if kind in ("enter", "exit"):
                if rest:
                    raise ValueError(f"an {kind} entry is ({kind!r}, constraint)")
                if isinstance(c, str):
                    c = list(k.CONSTRAINT_NAMES).index(c)
                c = int(c)
                if kind == "enter":
                    if c in open_arcs:
                        raise ValueError(f"{k.CONSTRAINT_NAMES[c]!r} is entered twice without an exit")
                    if not k.EXACT_ARCS[c]:
                        why = k.system.path_constraints[c].unsupported
                        raise ValueError(f"path constraint {k.CONSTRAINT_NAMES[c]!r}: {why}")
                    open_arcs[c] = len(self.arcs)
                    self.events.append(("in", len(self.arcs)))
                    self.arcs.append(("boundary", c, k.CONSTRAINT_ORDERS[c]))
                else:
                    if c not in open_arcs:
                        raise ValueError(f"{k.CONSTRAINT_NAMES[c]!r} is left without being entered")
                    self.events.append(("out", open_arcs.pop(c)))
                continue
            if kind in ("boundary", "touch"):
                if rest:
                    raise ValueError(f"a {kind} entry is ({kind!r}, constraint)")
                if isinstance(c, str):
                    c = list(k.CONSTRAINT_NAMES).index(c)
                if not k.EXACT_ARCS[c]:
                    why = k.system.path_constraints[c].unsupported
                    raise ValueError(f"path constraint {k.CONSTRAINT_NAMES[c]!r}: {why}")
                q = k.CONSTRAINT_ORDERS[c]
                if kind == "touch" and q != 2:
                    raise ValueError("touch points are supported for order-2 constraints")
                if kind == "boundary":
                    self.events += [("in", len(self.arcs)), ("out", len(self.arcs))]
                else:
                    self.events.append(("touch", len(self.arcs)))
                self.arcs.append((kind, int(c), q))
            elif kind in ("singular", "switch"):
                c = self._switching_index(c)
                joined = kind == "singular" and len(rest) == 3 and rest[2] == "joined"
                if joined:
                    rest = rest[:2]
                    if not self.arcs or self.arcs[-1][0] != "boundary":
                        raise ValueError("a joined singular arc must follow a boundary arc")
                    kind = "singular_joined"
                if kind == "singular_joined" or kind == "singular":
                    if len(rest) != 2 or any(r not in ("lower", "upper") for r in rest):
                        raise ValueError("a singular entry is ('singular', control, before, "
                                         "after), with before and after 'lower' or 'upper', "
                                         "optionally followed by 'joined'")
                    if not k.SINGULAR_ARCS[c]:
                        why = k.system.singular_unsupported[c]
                        raise ValueError(f"no singular arc for switching function {c}: {why}")
                    sides.setdefault(c, []).append(tuple(int(r == "upper") for r in rest))
                elif rest:
                    raise ValueError("a switch entry is ('switch', control)")
                self.events.append(("legacy", len(self.arcs)))
                self.arcs.append((kind, c, tuple(int(r == "upper") for r in rest)))
            else:
                raise ValueError("arc kind must be 'boundary', 'touch', 'singular', 'switch', "
                                 f"'enter' or 'exit', got {kind!r}")
        if open_arcs:
            raise ValueError(f"{[k.CONSTRAINT_NAMES[c] for c in open_arcs]} entered but never left")
        self._legacy = any(e[0] == "legacy" for e in self.events)
        # Overlapping: an entry or a touch while another limit is ridden.
        nested, depth = False, 0
        for e in self.events:
            if e[0] != "out" and depth > 0:
                nested = True
            depth += 1 if e[0] == "in" else -1 if e[0] == "out" else 0
        if self._legacy and nested:
            raise ValueError("overlapping boundary arcs are not supported with singular arcs")
        self._nested = nested
        # The bound each control with a singular arc holds on the first free
        # arc: walk back from its first singular arc over its switches.
        self.frozen = {}
        for c, pairs in sides.items():
            flag = pairs[0][0]
            for kind, cc, _ in self.arcs:
                if cc == c and kind in ("singular", "singular_joined"):
                    break
                if cc == c and kind == "switch":
                    flag = 1 - flag
            self.frozen[c] = flag
        flags = dict(self.frozen)
        for kind, c, extra in self.arcs:
            if kind == "switch":
                if c not in flags:
                    raise ValueError(f"switch entries are for controls with a singular arc; "
                                     f"switching function {c} has none")
                flags[c] = 1 - flags[c]
            elif kind in ("singular", "singular_joined"):
                if extra[0] != flags[c]:
                    raise ValueError(f"switching function {c}: a singular arc entered from "
                                     f"{'upper' if extra[0] else 'lower'} follows one at "
                                     f"{'upper' if flags[c] else 'lower'}")
                flags[c] = extra[1]
        self.n_base = self.n_unknowns
        self._arc_slices = []
        off = self.n_base
        for kind, c, q in self.arcs:
            size = 2 + q if kind == "boundary" else self._SIZES[kind]
            self._arc_slices.append(slice(off, off + size))
            off += size
        self.n_unknowns = off
        self.options = dict(method=method, rtol=rtol, atol=atol)
        self.max_nfev = int(max_nfev)

    def _switching_index(self, c):
        k = self.kernels
        if isinstance(c, str):
            names = [sw.control.name for sw in k.system.switching_functions]
            if c not in names:
                raise ValueError(f"{c!r} is not a bang-bang control (those are {names})")
            c = names.index(c)
        c = int(c)
        if not 0 <= c < k.N_SWITCHING:
            raise ValueError(f"switching function index {c} out of range ({k.N_SWITCHING})")
        return c

    def __repr__(self):
        k = self.kernels

        def name(kind, c):
            if kind in ("boundary", "touch"):
                return k.CONSTRAINT_NAMES[c]
            return k.system.switching_functions[c].control.name

        arcs = ", ".join(f"{kind.replace('_', ' ')} {name(kind, c)}"
                         for kind, c, _ in self.arcs)
        return (f"<MultiArcShooting {k.module.__name__}: [{arcs}], "
                f"{self.n_unknowns} unknowns, params={self.params.tolist()}>")

    # -- unknowns -----------------------------------------------------------

    def pack_arcs(self, base, arc_values):
        """``v`` from :class:`SingleShooting` unknowns ``base`` and, per arc,
        ``(t_in, t_out, pi)`` for a boundary arc, ``(t_c, nu)`` for a touch,
        ``(t_in, t_out)`` for a singular arc, ``t_out`` for a joined one or
        ``t_s`` for a switch."""
        parts = [np.asarray(base, dtype=float).reshape(-1)[:self.n_base]]
        for (kind, c, q), val in zip(self.arcs, arc_values, strict=True):
            if kind == "boundary":
                t_in, t_out, pi = val
                pi = np.asarray(pi, dtype=float).reshape(-1)
                if pi.shape != (q,):
                    raise ValueError(f"pi must have {q} entries for an order-{q} constraint")
                parts.append(np.concatenate([[t_in, t_out], pi]))
            else:
                parts.append(np.asarray(val, dtype=float).reshape(self._SIZES[kind]))
        return np.concatenate(parts)

    def unpack_arcs(self, v):
        """Per arc, ``(t_in, t_out, pi)``, ``(t_c, nu)``, ``(t_in, t_out)``, or
        ``t_out`` (joined singular) or ``t_s`` (switch)."""
        v = np.asarray(v, dtype=float).reshape(-1)
        out = []
        for (kind, _, _), sl in zip(self.arcs, self._arc_slices):
            w = v[sl]
            if kind == "boundary":
                out.append((w[0], w[1], w[2:].copy()))
            elif kind in ("switch", "singular_joined"):
                out.append(w[0])
            else:
                out.append((w[0], w[1]))
        return out

    # -- the system ---------------------------------------------------------

    def _free_arc(self, t, z, flags):
        """The mode flags at the start of a free arc: from the signs of the
        mode functions, with the controls that have singular arcs at ``flags``."""
        k = self.kernels
        if not k.N_MODES:
            return None
        arc = np.array(k.initial_arc(t, z, self.params), dtype=np.intc)
        for c, flag in flags.items():
            arc[c] = flag
        return arc

    def _free_hamiltonian(self, t, z, flags):
        """``(H, dH/dt, dH/dz)`` on a free arc, with the held controls at their bounds.

        ``H`` is linear in a bang-bang control ``u`` with coefficient ``S``,
        so ``H(u_held) = H(u*) + S (u_held - u*)``, and the law ``u*`` and
        the bounds are piecewise constant in ``z``."""
        k, p = self.kernels, self.params
        H = k.hamiltonian(t, z, p)
        Ht, Hz = k.hamiltonian_gradient(t, z, p)
        if not flags:
            return H, Ht, Hz
        S = k.switching(t, z, p)
        St, Sz, _ = k.switching_gradient(t, z, p)
        u = k.control(t, z, p)
        lo, hi = k.control_bounds(t, z, p)
        controls = list(k.system.controls)
        for c, flag in flags.items():
            j = controls.index(k.system.switching_functions[c].control)
            du = (hi[j] if flag else lo[j]) - u[j]
            if du:
                H = H + S[c] * du
                Ht = Ht + St[c] * du
                Hz = Hz + Sz[c] * du
        return H, Ht, Hz

    def _segment(self, kind, c, ta, dta, z, D, tb, dtb, jac, cost=False, flags=None):
        """Propagate a free (``kind is None``), boundary or singular arc; chain ``D = dz/dv``."""
        k, p = self.kernels, self.params
        if not tb > ta:
            raise ShootingError(f"arc times out of order: {ta} then {tb}")
        tolerance = {"stm_rtol": self.jacobian_rtol} if jac else {}
        try:
            if kind is None:
                arc = self._free_arc(ta, z, flags)
                r = k.propagate(ta, z, tb, p, stm=jac, cost=cost, constraint_events=False,
                                arc=arc, frozen=sorted(flags), max_nfev=self.max_nfev,
                                **tolerance, **self.options)
                Fa = k.rhs_arc(ta, z, p, r["arc0"] if k.N_MODES else None)
            elif kind == "boundary":
                r = k.propagate_boundary(c, ta, z, tb, p, stm=jac, cost=cost,
                                         max_nfev=self.max_nfev, **tolerance, **self.options)
                Fa = k.boundary_rhs(c, ta, z, p)
            else:
                r = k.propagate_singular(c, ta, z, tb, p, stm=jac, cost=cost,
                                         max_nfev=self.max_nfev, **self.options)
                Fa = k.singular_rhs(c, ta, z, p)
        except RuntimeError as exc:
            raise ShootingError(f"propagation failed on [{ta}, {tb}]: {exc}") from exc
        if r["status"] != 0:
            raise ShootingError(f"propagation failed with status {r['status']}")
        if jac:
            Phi = r["stm"]
            D = Phi @ D + np.outer(r["dz_dt1"], dtb) - np.outer(Phi @ Fa, dta)
        return r, D

    def _arc_id(self, active):
        """The kernels' arc id for the set of constraints ``active`` (None: free)."""
        if not active:
            return None
        if len(active) == 1:
            return next(iter(active))
        return self.kernels.boundary_set(sorted(active))

    def _regular_junction(self, kind, q, side, t, z):
        """Whether a boundary arc's junction drops ``H`` continuity: an order-1
        limit when ``H`` is regular (strictly convex, or convex with clipped
        controls: no bang-bang) and the controls on the free side, ``side``,
        are strictly inside their bounds at ``(t, z)``. There the control and
        the costates are continuous (Jacobson, Lele & Speyer 1971): the state
        arrives tangentially, so ``dg/dt = 0`` at the entry, and ``mu``
        vanishes at the exit. Both have simple roots, where ``H`` continuity
        and ``g = 0`` alone have double ones (Newton then converges only
        linearly). A control still at its bound at the junction jumps there
        instead, and ``H`` continuity is kept. ``junctions="hamiltonian"``
        keeps it everywhere."""
        if not (kind == "boundary" and q == 1 and self.junctions == "auto"
                and not self.kernels.N_SWITCHING):
            return False
        k, p = self.kernels, self.params
        if not k.system.problem.control_sets:
            return True
        u = k.control(t, z, p) if side is None else k.boundary_control(side, t, z, p)
        lo, hi = k.control_bounds(t, z, p)
        u, lo, hi = (np.asarray(a, dtype=float) for a in (u, lo, hi))
        margin = 1e-12 * np.maximum(1.0, np.abs(u))
        return bool(np.all((u - lo > margin) & (hi - u > margin)))

    def _rate_of(self, c, arc_id, t, z, flags):
        """``(dg_c/dt, its t-derivative, its z-gradient)`` along the free arc
        (``arc_id`` None) or a boundary arc, for a time-invariant limit."""
        k, p = self.kernels, self.params
        n = k.N_STATES
        _, Nx, Nxx, _ = k.tangency_gradient(c, t, z, p)

        def flow(tt):
            if arc_id is None:
                arc = self._free_arc(tt, z, flags)
                return k.rhs_arc(tt, z, p, arc) if k.N_MODES else k.rhs(tt, z, p)
            return k.boundary_rhs(arc_id, tt, z, p)

        F = flow(t)
        if arc_id is None:
            A = (k.jacobian(t, z, p, arc=self._free_arc(t, z, flags)) if k.N_MODES
                 else k.jacobian(t, z, p))
        else:
            A = k.boundary_jacobian(arc_id, t, z, p)
        gx = Nx[0]
        val = float(gx @ F[:n])
        gz = gx @ A[:n]
        gz[:n] += Nxx[0] @ F[:n]
        h = 1e-7 * max(1.0, abs(t))
        gt = float(gx @ (flow(t + h)[:n] - flow(t - h)[:n])) / (2 * h)
        return val, gt, gz

    def _multiplier_of(self, arc_id, c, t, z):
        """``(mu_c, dmu_c/dt, dmu_c/dz)`` of constraint ``c`` on arc ``arc_id``."""
        k, p = self.kernels, self.params
        mu = np.atleast_1d(k.boundary_multiplier(arc_id, t, z, p))
        Mt, Mz = k.boundary_multiplier_gradient(arc_id, t, z, p)
        i = list(k.arc_constraints(arc_id)).index(c)
        return float(mu[i]), float(Mt[i]), Mz[i]

    def _set_hamiltonian(self, arc_id, t, z, flags):
        """``(H, dH/dt, dH/dz)`` on the free arc (``arc_id`` None) or a boundary arc."""
        k, p = self.kernels, self.params
        if arc_id is None:
            return self._free_hamiltonian(t, z, flags)
        Ht, Hz = k.hamiltonian_gradient(t, z, p, boundary=arc_id)
        return k.boundary_hamiltonian(arc_id, t, z, p), Ht, Hz

    def _evaluate_events(self, v, jac, record=False):
        """The residual over boundary arcs and touch points that may overlap.

        Between consecutive events the active limits are fixed and the
        segment follows their kernels (free, one limit, or a set ridden at
        once). Entering a limit adds its tangency conditions, the costate
        jump ``-dN/dx^T pi`` and ``H`` continuity; leaving one adds ``H``
        continuity; a touch point adds ``g = g' = 0`` and its jump.
        """
        k, p = self.kernels, self.params
        n, N = k.N_STATES, self.n_unknowns
        x0, lam0, nu, tf = self.unpack(v)
        z0 = np.concatenate([x0, lam0])
        m = self.n_free
        D0 = np.zeros((2 * n, N))
        D0[self.free_states, np.arange(m)] = 1.0
        D0[n + np.arange(n), m + np.arange(n)] = 1.0

        def unit(j):
            e = np.zeros(N)
            e[j] = 1.0
            return e

        rows, grads, segments = [], [], []
        z, D, t, dt = z0, D0, self.t0, np.zeros(N)
        flags = dict(self.frozen)
        active = set()
        for ev, i in self.events:
            kind, c, q = self.arcs[i]
            sl = self._arc_slices[i]
            w = v[sl]
            if ev == "out":
                t_ev, d_ev = w[1], unit(sl.start + 1)
            else:
                t_ev, d_ev = w[0], unit(sl.start)
            aid = self._arc_id(active)
            seg = None if aid is None else "boundary"
            r, D = self._segment(seg, aid, t, dt, z, D, t_ev, d_ev, jac, record, flags)
            segments.append((seg, aid, t, t_ev, z, dict(flags) if aid is None else None, r))
            z = r["z"]
            t, dt = t_ev, d_ev
            if ev == "out":
                if self._regular_junction(kind, q, self._arc_id(active - {c}), t, z):
                    # mu_c vanishes at the exit of an order-1 limit when H is
                    # regular: a simple root, where H continuity has a double one.
                    val, Mt, Mz = self._multiplier_of(aid, c, t, z)
                    rows.append(val)
                    grads.append(Mt * d_ev + Mz @ D if jac else None)
                else:
                    after = self._arc_id(active - {c})
                    H1, H1t, H1z = self._set_hamiltonian(aid, t, z, flags)
                    H2, H2t, H2z = self._set_hamiltonian(after, t, z, flags)
                    rows.append(H1 - H2)
                    grads.append((H1t - H2t) * d_ev + (H1z - H2z) @ D if jac else None)
                active.discard(c)
                continue
            Nv = k.tangency(c, t, z, p)
            Nt, Nx, Nxx, Nxt = k.tangency_gradient(c, t, z, p)
            for r_ in range(2 if kind == "touch" else q):
                rows.append(Nv[r_])
                grads.append(Nt[r_] * d_ev + Nx[r_] @ D[:n] if jac else None)
            used = slice(0, 1) if kind == "touch" else slice(0, q)
            mult = w[1:2] if kind == "touch" else w[2:]
            mult_cols = [sl.start + 1] if kind == "touch" else list(range(sl.start + 2, sl.stop))
            zp = z.copy()
            zp[n:] -= Nx[used].T @ mult
            Dp = None
            if jac:
                Dp = D.copy()
                for r_, mu in enumerate(mult):
                    Dp[n:] -= mu * (Nxx[r_] @ D[:n] + np.outer(Nxt[r_], d_ev))
                Dp[n:, mult_cols] -= Nx[used].T
            if kind == "boundary":
                after = self._arc_id(active | {c})
                if self._regular_junction(kind, q, aid, t, z):
                    # The arrival is tangential: g peaks there along the arc
                    # before, so dg/dt = 0 (a simple root in t_in, where g = 0
                    # alone has a double one).
                    val, gt, gz = self._rate_of(c, aid, t, z, flags)
                    rows.append(val)
                    grads.append(gt * d_ev + gz @ D if jac else None)
                else:
                    # H continuous at the entry: H(before, z-) = H(after, z+).
                    H1, H1t, H1z = self._set_hamiltonian(aid, t, z, flags)
                    H2, H2t, H2z = self._set_hamiltonian(after, t, zp, flags)
                    rows.append(H1 - H2)
                    grads.append((H1t - H2t) * d_ev + H1z @ D - H2z @ Dp if jac else None)
                active.add(c)
            z = zp
            if jac:
                D = Dp
        d_tf = unit(m + n + k.N_MULTIPLIERS) if k.FREE_FINAL_TIME else np.zeros(N)
        r, D = self._segment(None, None, t, dt, z, D, tf, d_tf, jac, record, flags)
        segments.append((None, None, t, tf, z, dict(flags), r))
        zf = r["z"]
        R = np.concatenate([k.residual(z0, zf, nu, tf, p)[self._rows], rows])
        if not jac:
            return R, None, segments
        Jb = k.residual_jacobian(z0, zf, nu, tf, p)
        Jbase = Jb["z0"][self._rows] @ D0 + Jb["zf"][self._rows] @ D
        if k.N_MULTIPLIERS:
            Jbase[:, m + n:m + n + k.N_MULTIPLIERS] += Jb["nu"][self._rows]
        if k.FREE_FINAL_TIME:
            Jbase += np.outer(Jb["tf"][self._rows], d_tf)
        J = np.vstack([Jbase] + [g[None, :] for g in grads]) if grads else Jbase
        return R, J, segments

    def _evaluate(self, v, jac, record=False):
        if not self._legacy:
            return self._evaluate_events(v, jac, record)
        k, p = self.kernels, self.params
        n, N = k.N_STATES, self.n_unknowns
        x0, lam0, nu, tf = self.unpack(v)
        z0 = np.concatenate([x0, lam0])
        m = self.n_free
        D0 = np.zeros((2 * n, N))
        D0[self.free_states, np.arange(m)] = 1.0
        D0[n + np.arange(n), m + np.arange(n)] = 1.0

        def unit(j):
            e = np.zeros(N)
            e[j] = 1.0
            return e

        # segments: (kind, c, t_start, t_end, z_start, frozen flags, propagation)
        rows, grads, segments = [], [], []
        z, D, t, dt = z0, D0, self.t0, np.zeros(N)
        flags = dict(self.frozen)
        arcs = list(zip(self.arcs, self._arc_slices))
        for i_arc, ((kind, c, q), sl) in enumerate(arcs):
            w = v[sl]
            if kind == "singular_joined":
                # Starts at the boundary arc's exit (t, dt, z, D as it left them).
                Nv, Nt, Nz = k.singular_junction(c, t, z, p)
                for i in range(2):
                    rows.append(Nv[i])
                    grads.append(Nt[i] * dt + Nz[i] @ D if jac else None)
                t_out, d_out = w[0], unit(sl.start)
                r, D = self._segment("singular", c, t, dt, z, D, t_out, d_out, jac, record)
                segments.append(("singular", c, t, t_out, z, None, r))
                z = r["z"]
                flags[c] = q[1]
                t, dt = t_out, d_out
                continue
            t_in, d_in = w[0], unit(sl.start)
            r, D = self._segment(None, None, t, dt, z, D, t_in, d_in, jac, record, flags)
            segments.append((None, None, t, t_in, z, dict(flags), r))
            z = r["z"]
            if kind == "switch":
                # z is continuous; the jump in F enters D through dz/dt_s on
                # both sides, since t_s is an unknown.
                St, Sz, _ = k.switching_gradient(t_in, z, p)
                rows.append(k.switching(t_in, z, p)[c])
                grads.append(St[c] * d_in + Sz[c] @ D if jac else None)
                flags[c] = 1 - flags[c]
                t, dt = t_in, d_in
                continue
            if kind == "singular":
                Nv, Nt, Nz = k.singular_junction(c, t_in, z, p)
                for i in range(2):
                    rows.append(Nv[i])
                    grads.append(Nt[i] * d_in + Nz[i] @ D if jac else None)
                t_out, d_out = w[1], unit(sl.start + 1)
                r, D = self._segment("singular", c, t_in, d_in, z, D, t_out, d_out, jac, record)
                segments.append(("singular", c, t_in, t_out, z, None, r))
                z = r["z"]
                flags[c] = q[1]
                t, dt = t_out, d_out
                continue
            Nv = k.tangency(c, t_in, z, p)
            Nt, Nx, Nxx, Nxt = k.tangency_gradient(c, t_in, z, p)
            for i in range(2 if kind == "touch" else q):
                rows.append(Nv[i])
                grads.append(Nt[i] * d_in + Nx[i] @ D[:n] if jac else None)
            # The costate jump at the entry (or the touch).
            used = slice(0, 1) if kind == "touch" else slice(0, q)
            mult = w[1:2] if kind == "touch" else w[2:]
            mult_cols = [sl.start + 1] if kind == "touch" else list(range(sl.start + 2, sl.stop))
            zp = z.copy()
            zp[n:] -= Nx[used].T @ mult
            if jac:
                Dp = D.copy()
                for i, mu in enumerate(mult):
                    Dp[n:] -= mu * (Nxx[i] @ D[:n] + np.outer(Nxt[i], d_in))
                Dp[n:, mult_cols] -= Nx[used].T
            if kind == "touch":
                z, t, dt = zp, t_in, d_in
                if jac:
                    D = Dp
                continue
            # H continuous at the entry: H(u*, z-) = H(u_b, z+).
            Hf, Hf_t, Hf_z = self._free_hamiltonian(t_in, z, flags)
            Hb_t, Hb_z = k.hamiltonian_gradient(t_in, zp, p, boundary=c)
            rows.append(Hf - k.boundary_hamiltonian(c, t_in, zp, p))
            grads.append((Hf_t - Hb_t) * d_in + Hf_z @ D - Hb_z @ Dp if jac else None)
            t_out, d_out = w[1], unit(sl.start + 1)
            r, D = self._segment("boundary", c, t_in, d_in, zp, Dp if jac else None, t_out,
                                 d_out, jac, record)
            segments.append(("boundary", c, t_in, t_out, zp, None, r))
            z = r["z"]
            t, dt = t_out, d_out
            if i_arc + 1 < len(arcs) and arcs[i_arc + 1][0][0] == "singular_joined":
                continue                 # S = S' = 0 there make H continuous
            # H continuous at the exit; the costates are continuous there.
            Hb_t, Hb_z = k.hamiltonian_gradient(t_out, z, p, boundary=c)
            Hf, Hf_t, Hf_z = self._free_hamiltonian(t_out, z, flags)
            rows.append(k.boundary_hamiltonian(c, t_out, z, p) - Hf)
            grads.append((Hb_t - Hf_t) * d_out + (Hb_z - Hf_z) @ D if jac else None)
        d_tf = unit(m + n + k.N_MULTIPLIERS) if k.FREE_FINAL_TIME else np.zeros(N)
        r, D = self._segment(None, None, t, dt, z, D, tf, d_tf, jac, record, flags)
        segments.append((None, None, t, tf, z, dict(flags), r))
        zf = r["z"]

        R = np.concatenate([k.residual(z0, zf, nu, tf, p)[self._rows], rows])
        if not jac:
            return R, None, segments
        Jb = k.residual_jacobian(z0, zf, nu, tf, p)
        Jbase = Jb["z0"][self._rows] @ D0 + Jb["zf"][self._rows] @ D
        if k.N_MULTIPLIERS:
            Jbase[:, m + n:m + n + k.N_MULTIPLIERS] += Jb["nu"][self._rows]
        if k.FREE_FINAL_TIME:
            Jbase += np.outer(Jb["tf"][self._rows], d_tf)
        J = np.vstack([Jbase] + [g[None, :] for g in grads]) if grads else Jbase
        return R, J, segments

    def residual(self, v):
        return self._evaluate(np.asarray(v, dtype=float), jac=False)[0]

    def residual_jacobian(self, v):
        R, J, _ = self._evaluate(np.asarray(v, dtype=float), jac=True)
        return R, J

    __call__ = residual_jacobian

    def guess_multipliers(self, *args, **kwargs):
        raise NotImplementedError("use MultiArcShooting.from_penalty or from_smoothed for a "
                                  "starting point")

    def solve(self, guess, method=None, tol=1e-10, options=None):
        """As :meth:`SingleShooting.solve`. ``method`` defaults to ``"hybr"``,
        or ``"lm"`` with bang-bang controls: there the control jumps at a
        junction, ``H`` continuity reads ``|S| = 0`` (a kink at the root) and
        the Broyden updates of ``"hybr"`` stall on it, while ``"lm"`` takes the
        exact Jacobian at every step. If MINPACK stops close to but short of
        ``tol`` (within 1e-3), plain Newton steps with the exact Jacobian finish
        the solve; with bang-bang controls the Jacobian carries the sign of
        ``S``, so each is Newton's step on the smooth ``S``."""
        if method is None:
            method = "lm" if self.kernels.N_SWITCHING else "hybr"
        sol = super().solve(guess, method=method, tol=tol, options=options)
        if sol.success or not np.isfinite(sol.residual_norm) or sol.residual_norm > 1e-3:
            return sol
        v, best, steps = sol.v, sol.residual_norm, 0
        for steps in range(1, 21):
            try:
                R, J = self.residual_jacobian(v)
                w = v - np.linalg.lstsq(J, R, rcond=None)[0]
                err = float(np.linalg.norm(self.residual(w)))
            except (ShootingError, RuntimeError, ValueError, np.linalg.LinAlgError):
                break
            if not err < best:
                break
            v, best = w, err
            if best < tol:
                break
        if best >= sol.residual_norm:
            return sol
        x0, lam0, nu, tf = self.unpack(v)
        violation = 0.0
        if best < tol:
            try:
                violation = self._max_violation(v)
            except (ShootingError, RuntimeError, ValueError):
                best = violation = float("inf")
        return ShootingSolution(v=v, x0=x0, lam0=lam0.copy(), nu=nu.copy(), t0=self.t0, tf=tf,
                                residual_norm=best, success=best < tol,
                                nfev=sol.nfev + 2 * steps, result=sol.result,
                                max_violation=violation)

    # -- results ------------------------------------------------------------

    def cost(self, v):
        """The running cost integrated over every arc."""
        _, _, segs = self._evaluate(np.asarray(v, dtype=float), jac=False, record=True)
        return float(sum(r["cost"] for *_, r in segs))

    def trajectory(self, v, n_points=401):
        """The extremal at ``v``, sampled on every arc.

        Returns a dict with ``t``, ``z``, ``u``, ``g`` (every path constraint),
        ``arc`` (on boundary arcs the kernels' arc id: the constraint, or
        ``N_CONSTRAINTS + k`` for set ``k`` of limits ridden at once; else -1),
        ``active`` (len, NC): which limits are ridden at each sample,
        ``singular`` (the switching function on singular arcs, else -1) and
        ``held`` (len, NSW): on free arcs, the bound (1 upper, 0 lower) each
        control with a singular arc holds, else -1. Plus ``junctions`` (the
        arc boundaries).
        """
        from fast_numerical_integration.integrators import solve_ivp

        k, p = self.kernels, self.params
        _, _, segs = self._evaluate(np.asarray(v, dtype=float), jac=False)
        span = segs[-1][3] - self.t0
        T, Z, U, A, Sg, Hd = [], [], [], [], [], []
        for kind, c, ta, tb, za, flags, _ in segs:
            npts = max(int(n_points * (tb - ta) / span), 2)
            te = np.linspace(ta, tb, npts)
            held = np.full(k.N_SWITCHING, -1)
            if kind is None:
                # Bang-bang switches are located as in propagate.
                sol = k.trajectory(ta, za, tb, p, t_eval=te, constraint_events=False,
                                   arc=self._free_arc(ta, za, flags), frozen=sorted(flags),
                                   **self.options)
                ts, zs, us = sol["t"], sol["z"], sol["u"]
                for cc, flag in flags.items():
                    held[cc] = flag
                # The law at a frozen control's S may disagree with the bound held.
                if flags:
                    us = np.array(us, dtype=float)
                    for cc, flag in flags.items():
                        sw = k.system.switching_functions[cc]
                        j = list(k.system.controls).index(sw.control)
                        lo, hi = zip(*[k.control_bounds(ti, zi, p) for ti, zi in zip(ts, zs)])
                        us[:, j] = np.array(hi)[:, j] if flag else np.array(lo)[:, j]
            else:
                Dyn = k.BoundaryDynamics if kind == "boundary" else k.SingularDynamics
                ctl = k.boundary_control if kind == "boundary" else k.singular_control
                sol = solve_ivp(Dyn(c, p), (ta, tb), za, method=self.options["method"],
                                t_eval=te, rtol=self.options["rtol"], atol=self.options["atol"])
                ts, zs = sol.t, sol.y.T
                us = [ctl(c, ti, zi, p) for ti, zi in zip(ts, zs)]
            for ti, zi, ui in zip(ts, zs, us):
                T.append(ti)
                Z.append(zi)
                U.append(ui)
                A.append(c if kind == "boundary" else -1)
                Sg.append(c if kind == "singular" else -1)
                Hd.append(held)
        T, Z = np.array(T), np.array(Z)
        G = np.array([k.constraint(ti, zi, p) for ti, zi in zip(T, Z)]).reshape(len(T), -1)
        A = np.array(A)
        active = np.zeros((len(T), k.N_CONSTRAINTS), dtype=bool)
        for aid in set(A.tolist()) - {-1}:
            active[np.ix_(A == aid, list(k.arc_constraints(aid)))] = True
        return {"t": T, "z": Z, "u": np.array(U).reshape(len(T), -1), "g": G,
                "arc": A, "active": active, "singular": np.array(Sg),
                "held": np.array(Hd).reshape(len(T), k.N_SWITCHING),
                "junctions": np.array([s[2] for s in segs[1:]])}

    def constraint_violation(self, v, n_points=2001):
        """Per path constraint, the largest ``max(0, g)`` off its boundary
        arcs, or the largest bound violation of its boundary control on them."""
        tr = self.trajectory(v, n_points)
        return self._violations(tr)[0]

    def _max_violation(self, v, n_points=2001):
        tr = self.trajectory(v, n_points)
        per_constraint, _ = self._violations(tr)
        worst = [float(np.max(per_constraint, initial=0.0))]
        worst += self._singular_bound_violation(tr)
        return max(worst)

    def _bound_violation(self, tr, on):
        k, p = self.kernels, self.params
        worst = 0.0
        for i in np.flatnonzero(on):
            lo, hi = k.control_bounds(tr["t"][i], tr["z"][i], p)
            u = tr["u"][i]
            worst = max(worst, float(np.max(np.maximum(lo - u, u - hi), initial=0.0)))
        return worst

    def _singular_bound_violation(self, tr):
        return [self._bound_violation(tr, tr["singular"] == c)
                for kind, c, _ in self.arcs if kind.startswith("singular")]

    def _violations(self, tr):
        """Per constraint, the largest ``max(0, g)`` where it is not ridden (free
        arcs and other limits' arcs), raised to the bound violation of the
        controls where it is; and per boundary arc that bound violation."""
        k = self.kernels
        out = np.zeros(k.N_CONSTRAINTS)
        for c in range(k.N_CONSTRAINTS):
            off = ~tr["active"][:, c]
            if off.any():
                out[c] = max(float(tr["g"][off, c].max()), 0.0)
        bound = []
        for kind, c, _ in self.arcs:
            if kind != "boundary":
                continue
            worst = self._bound_violation(tr, tr["active"][:, c])
            bound.append(worst)
            out[c] = max(out[c], worst)
        return out, bound

    def diagnostics(self, v, n_points=401):
        """Checks of a solution beyond the residual.

        Returns a dict with ``free_arc_violation`` (per constraint, wherever it
        is not ridden: free arcs and other limits' arcs), ``touch_nu``
        (should be >= 0), ``boundary_pi``, and per boundary arc the range of
        its multiplier ``mu`` (should be >= 0) and of its controls along the
        arc, and ``boundary_bound_violation``: how far those controls leave
        their bounds (0 when they stay inside).

        For singular arcs: per arc the range of the singular control
        (``singular_u``), how far it leaves its bounds
        (``singular_bound_violation``, 0 inside), and the range of the
        Legendre-Clebsch coefficient ``-dS''/du`` (``singular_legendre``,
        should be >= 0). ``held_sign_violation`` is, per switching function
        with a singular arc, the largest ``S`` where the free arcs hold the
        upper bound or ``-S`` where they hold the lower one (should be <= 0:
        the bound held is the one the law would pick).
        """
        k, p = self.kernels, self.params
        tr = self.trajectory(v, n_points)
        arcs = self.unpack_arcs(v)
        free = np.zeros(k.N_CONSTRAINTS)
        for c in range(k.N_CONSTRAINTS):
            off = ~tr["active"][:, c]
            if off.any():
                free[c] = max(float(tr["g"][off, c].max()), 0.0)
        _, bound = self._violations(tr)
        out = {"free_arc_violation": free,
               "boundary_bound_violation": bound,
               "touch_nu": [a[1] for (kind, *_), a in zip(self.arcs, arcs) if kind == "touch"],
               "boundary_pi": [a[2] for (kind, *_), a in zip(self.arcs, arcs)
                               if kind == "boundary"],
               "boundary_mu": [], "boundary_u": [],
               "singular_u": [], "singular_legendre": [],
               "singular_bound_violation": self._singular_bound_violation(tr),
               "held_sign_violation": {}}
        for kind, c, _ in self.arcs:
            if kind == "boundary":
                on = tr["active"][:, c]
                mu = []
                for ti, zi, aid in zip(tr["t"][on], tr["z"][on], tr["arc"][on]):
                    m_ = k.boundary_multiplier(aid, ti, zi, p)
                    mu.append(m_ if aid < k.N_CONSTRAINTS
                              else m_[list(k.arc_constraints(aid)).index(c)])
                out["boundary_mu"].append((min(mu), max(mu)))
                out["boundary_u"].append((tr["u"][on].min(axis=0), tr["u"][on].max(axis=0)))
            elif kind.startswith("singular"):
                on = tr["singular"] == c
                lc = [k.singular_legendre(c, ti, zi, p)
                      for ti, zi in zip(tr["t"][on], tr["z"][on])]
                out["singular_legendre"].append((min(lc), max(lc)))
                out["singular_u"].append((tr["u"][on].min(axis=0), tr["u"][on].max(axis=0)))
        for c in self.frozen:
            on = np.flatnonzero(tr["held"][:, c] >= 0)
            worst = -np.inf
            for i in on:
                S = k.switching(tr["t"][i], tr["z"][i], p)[c]
                worst = max(worst, S if tr["held"][i, c] == 1 else -S)
            out["held_sign_violation"][c] = float(worst)
        return out

    # -- starting points ----------------------------------------------------

    @classmethod
    def from_penalty(cls, kernels, params, arcs, solution, penalty_params, **options):
        """A :class:`MultiArcShooting` and an initial guess, from a penalty solution.

        ``solution`` solves :class:`SingleShooting` at ``penalty_params`` (the
        same problem with a large penalty weight). Its crossings of ``g = 0``
        give the junction times: a boundary arc runs from the first to the
        last crossing of its violated interval (the penalty solution may dip
        back below the limit in between), and a touch sits at its middle. The
        costates come from ``solution``. The entry jump ``pi`` is fitted to
        the costates: integrating the boundary arc back from the penalty
        state at the exit gives ``lambda`` just after the entry, and the jump
        from the penalty ``lambda`` just before it is fitted to
        ``-dN/dx^T pi``. Only boundary arcs and touches are supported.

        Returns ``(shooting, v0)``.
        """
        shoot = cls(kernels, params, arcs, **options)
        if any(kind.startswith("singular") or kind == "switch" for kind, _, _ in shoot.arcs):
            raise ValueError("from_penalty supports boundary arcs and touches; for singular "
                             "arcs use from_smoothed")
        k = kernels
        z0 = solution.z0
        r = k.propagate(solution.t0, z0, solution.tf, penalty_params, **{
            key: val for key, val in options.items() if key in ("method", "rtol", "atol")})
        times, index = r["constraint_times"], r["constraint_index"]
        n = k.N_STATES

        def state_at(t):
            return k.propagate(solution.t0, z0, t, penalty_params)["z"]

        # Per constraint, the violated intervals [entry, exit] of the penalty
        # solution, grouped into as many arcs as the structure has for it: a
        # penalized boundary arc can dip back below the limit, so separate
        # arcs at the longest feasible gaps.
        spans = {}
        for c in {c for _, c, _ in shoot.arcs}:
            mine = times[index == c]
            n_arcs = sum(1 for _, cc, _ in shoot.arcs if cc == c)
            if len(mine) < 2 * n_arcs or len(mine) % 2:
                raise ValueError(f"the penalty solution crosses {k.CONSTRAINT_NAMES[c]!r} "
                                 f"{len(mine)} times; the structure needs {n_arcs} violated "
                                 "intervals")
            entries, exits = mine[0::2], mine[1::2]
            gaps = entries[1:] - exits[:-1]
            cuts = np.sort(np.argsort(gaps)[::-1][:n_arcs - 1])
            bounds = np.concatenate([[-1], cuts, [len(entries) - 1]])
            spans[c] = [(entries[bounds[i] + 1], exits[bounds[i + 1]]) for i in range(n_arcs)]
        used = {c: 0 for c in spans}

        values = []
        for kind, c, q in shoot.arcs:
            a, b = spans[c][used[c]]
            used[c] += 1
            if kind == "touch":
                tc = 0.5 * (a + b)
                _, Nx, _, _ = k.tangency_gradient(c, tc, state_at(tc), params)
                jump = state_at(a)[n:] - state_at(b)[n:]
                nu = np.linalg.lstsq(Nx[:1].T, jump, rcond=None)[0][0]
                values.append((tc, nu))
                continue
            za = state_at(a)
            back = k.propagate_boundary(c, b, state_at(b), a, params)["z"]
            _, Nx, _, _ = k.tangency_gradient(c, a, za, params)
            pi = np.linalg.lstsq(Nx.T, za[n:] - back[n:], rcond=None)[0]
            values.append((a, b, pi))
        return shoot, shoot.pack_arcs(solution.v, values)

    @classmethod
    def from_smoothed(cls, kernels, params, arcs, base, t, u, *, margin=0.02, **options):
        """A :class:`MultiArcShooting` and an initial guess, from a smoothed control history.

        ``t`` and ``u`` (len, NU) sample the control of a smoothed version of
        the problem: ``eps u^2 / 2`` added to the running cost, with the same
        states, bounds and boundary conditions, so its control is clipped
        rather than bang-bang. ``base`` is its solution's
        :class:`SingleShooting` unknowns (``ShootingSolution.v``); the
        unknowns of the two problems match.

        Per control with a singular arc, each sample is at its upper bound,
        at its lower bound (within ``margin`` of the range) or between. The
        longest stretches between bounds, as many as the structure has
        singular arcs of that control, give their entry and exit times. The
        switches are where the control crosses mid-range outside them.
        Only singular arcs and switches are supported.

        Returns ``(shooting, v0)``.
        """
        shoot = cls(kernels, params, arcs, **options)
        if any(kind in ("boundary", "touch") for kind, _, _ in shoot.arcs):
            raise ValueError("from_smoothed supports singular arcs and switches; for boundary "
                             "arcs use from_penalty")
        k = kernels
        t = np.asarray(t, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(len(t), -1)
        x = np.zeros(k.N_Z)
        found = {}
        for c in shoot.frozen:
            sw = k.system.switching_functions[c]
            j = list(k.system.controls).index(sw.control)
            lo, hi = k.control_bounds(t[0], x, shoot.params)
            lo, hi = lo[j], hi[j]
            between = (u[:, j] > lo + margin * (hi - lo)) & (u[:, j] < hi - margin * (hi - lo))
            # Runs of samples between the bounds: [start, stop) index pairs.
            edges = np.flatnonzero(np.diff(np.concatenate([[0], between.astype(int), [0]])))
            runs = list(zip(edges[0::2], edges[1::2]))
            n_sing = sum(1 for kind, cc, _ in shoot.arcs if kind == "singular" and cc == c)
            if len(runs) < n_sing:
                raise ValueError(f"the smoothed control {sw.control} leaves its bounds "
                                 f"{len(runs)} times; the structure has {n_sing} singular arcs")
            longest = sorted(sorted(runs, key=lambda r: t[r[1] - 1] - t[r[0]])[::-1][:n_sing])
            singular = [(t[max(a - 1, 0)], t[min(b, len(t) - 1)]) for a, b in longest]
            mid = 0.5 * (lo + hi)
            side = u[:, j] > mid
            crossings = []
            for i in np.flatnonzero(side[1:] != side[:-1]):
                if any(t[a] <= t[i] <= t[b - 1] for a, b in longest):
                    continue
                s = (mid - u[i, j]) / (u[i + 1, j] - u[i, j])
                crossings.append(t[i] + s * (t[i + 1] - t[i]))
            found[c] = (singular, crossings)
        values = []
        used = {c: [0, 0] for c in found}
        for kind, c, _ in shoot.arcs:
            singular, crossings = found[c]
            if kind == "singular":
                values.append(singular[used[c][0]])
                used[c][0] += 1
            else:
                if used[c][1] >= len(crossings):
                    name = k.system.switching_functions[c].control
                    raise ValueError(f"the smoothed control {name} has {len(crossings)} "
                                     "switches outside its singular arcs; the structure "
                                     "needs more")
                values.append(crossings[used[c][1]])
                used[c][1] += 1
        return shoot, shoot.pack_arcs(base, values)


def parameter_continuation(kernels, params, name, values, v0, *, shooting_options=None,
                           solve_options=None, **continuation_options):
    """:func:`continuation` in one runtime parameter, such as a penalty weight.

    Builds a :class:`SingleShooting` at ``params`` with parameter ``name``
    (or index) set to each value in turn, solving from the previous solution.
    ``v0`` must solve the system at ``params`` as given. Returns the list of
    ``(value, ShootingSolution)``.
    """
    params = np.asarray(params, dtype=float).reshape(-1).copy()
    i = list(kernels.param_names).index(name) if isinstance(name, str) else int(name)
    shooting_options = dict(shooting_options or {})
    solve_options = dict(solve_options or {})

    def solve_at(value, guess):
        p = params.copy()
        p[i] = value
        return SingleShooting(kernels, p, **shooting_options).solve(guess, **solve_options)

    return continuation(solve_at, params[i], values, v0, **continuation_options)


def continuation(solve_at, start, values, v0, *, min_step=1e-7, shrink=0.5, max_solves=2000):
    """Natural-parameter continuation from a solution ``v0`` at ``start``.

    ``solve_at(value, guess)`` returns a :class:`ShootingSolution` (or any
    object with ``v`` and ``success``) and is called for each target in
    ``values``, warm-started from the last solution. When it fails, a target
    partway (``shrink``) from the last solved value is inserted. Raises
    ``RuntimeError`` when that step falls below ``min_step`` (relative), or
    after ``max_solves`` calls (a path that creeps forward in ever smaller
    steps is stalled too). Returns the list of ``(value, solution)``; the last entry is at
    ``values[-1]``.
    """
    path = []
    v, current = np.asarray(v0, dtype=float), start
    targets = list(values)
    i = solves = 0
    while i < len(targets):
        target = targets[i]
        if solves >= max_solves:
            raise RuntimeError(f"continuation stalled at {current} after {solves} solves")
        solves += 1
        sol = solve_at(target, v)
        if sol.success:
            path.append((target, sol))
            v, current = sol.v, target
            i += 1
            continue
        mid = current + shrink * (target - current)
        if abs(mid - current) < min_step * max(1.0, abs(current)):
            raise RuntimeError(f"continuation stalled at {current} "
                               f"(residual {getattr(sol, 'residual_norm', float('nan')):.2e})")
        targets.insert(i, mid)
    return path
