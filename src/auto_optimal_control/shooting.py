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

:class:`MultiArcShooting` solves for exact constrained extremals instead: given
the sequence of boundary arcs and touch points, its unknowns add the junction
times and the costate-jump multipliers, and its Jacobian chains the STM across
the arcs. :meth:`MultiArcShooting.from_penalty` seeds it from a penalty solution.
"""

from dataclasses import dataclass

import numpy as np


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
        r = self.kernels.propagate(self.t0, z0, tf, self.params, stm=stm, **self.options)
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
        sol = root(self._safe, np.asarray(guess, dtype=float), jac=True, method=method,
                   options=options)
        R, _ = self._safe(sol.x)
        err = float(np.linalg.norm(R))
        x0, lam0, nu, tf = self.unpack(sol.x)
        violation = 0.0
        if self.kernels.N_CONSTRAINTS and err < tol:
            violation = float(np.max(self.constraint_violation(sol.x)))
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
    """Exact constrained extremals: shooting over a fixed sequence of arcs.

    ``arcs`` lists the constrained arcs in time order. Free arcs, where no
    limit is active, lie between them and at both ends. Each entry is one of:

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

    ``c`` is a constraint index or name. The unknowns are
    :class:`SingleShooting`'s, followed by each arc's in order. The path
    constraints are not watched on the free arcs (their penalties stay off),
    so a wrong structure shows up as a violation on a free arc, reported by
    :meth:`diagnostics` and in ``max_violation``.

    Bounded controls (:class:`~auto_optimal_control.problem.Box`) are
    supported. On the free arcs they follow their usual law: clipped where
    ``H`` is quadratic in them, bang-bang where it is linear, with the
    switches located (and the STM jumped) as in :class:`SingleShooting`. On a
    boundary arc the control is the one that rides the limit, so it must stay
    within its bounds there; how far it leaves them also counts toward
    ``max_violation`` (see :meth:`diagnostics`). Unit-vector control sets are
    not supported.

    :meth:`from_penalty` builds the structure's initial guess from an
    exterior-penalty solution. A bang-bang problem has no useful penalty
    solution: there the penalized optimum rides the limit on a singular arc,
    which shooting cannot follow. Smooth it instead (add ``eps u^2 / 2`` to
    the running cost, a quadratic control in a box), solve that with exact
    arcs and lower ``eps`` by :func:`continuation`. The unknowns of the two
    problems match, so the smoothed ``v`` is a starting point for the
    bang-bang one.
    """

    def __init__(self, kernels, params=(), arcs=(), *, x0=None, method="DOP853", rtol=1e-12,
                 atol=1e-12):
        super().__init__(kernels, params, x0=x0, method=method, rtol=rtol, atol=atol,
                         n_interior=0)
        k = kernels
        self.arcs = []
        for item in arcs:
            kind, c = item
            if isinstance(c, str):
                c = list(k.CONSTRAINT_NAMES).index(c)
            if kind not in ("boundary", "touch"):
                raise ValueError(f"arc kind must be 'boundary' or 'touch', got {kind!r}")
            if not k.EXACT_ARCS[c]:
                why = k.system.path_constraints[c].unsupported
                raise ValueError(f"path constraint {k.CONSTRAINT_NAMES[c]!r}: {why}")
            q = k.CONSTRAINT_ORDERS[c]
            if kind == "touch" and q != 2:
                raise ValueError("touch points are supported for order-2 constraints")
            self.arcs.append((kind, int(c), q))
        self.n_base = self.n_unknowns
        self._arc_slices = []
        off = self.n_base
        for kind, c, q in self.arcs:
            size = 2 + q if kind == "boundary" else 2
            self._arc_slices.append(slice(off, off + size))
            off += size
        self.n_unknowns = off
        self.options = dict(method=method, rtol=rtol, atol=atol)

    def __repr__(self):
        arcs = ", ".join(f"{kind} {self.kernels.CONSTRAINT_NAMES[c]}" for kind, c, _ in self.arcs)
        return (f"<MultiArcShooting {self.kernels.module.__name__}: [{arcs}], "
                f"{self.n_unknowns} unknowns, params={self.params.tolist()}>")

    # -- unknowns -----------------------------------------------------------

    def pack_arcs(self, base, arc_values):
        """``v`` from :class:`SingleShooting` unknowns ``base`` and, per arc,
        ``(t_in, t_out, pi)`` for a boundary arc or ``(t_c, nu)`` for a touch."""
        parts = [np.asarray(base, dtype=float).reshape(-1)[:self.n_base]]
        for (kind, c, q), val in zip(self.arcs, arc_values, strict=True):
            if kind == "boundary":
                t_in, t_out, pi = val
                pi = np.asarray(pi, dtype=float).reshape(-1)
                if pi.shape != (q,):
                    raise ValueError(f"pi must have {q} entries for an order-{q} constraint")
                parts.append(np.concatenate([[t_in, t_out], pi]))
            else:
                parts.append(np.asarray(val, dtype=float).reshape(2))
        return np.concatenate(parts)

    def unpack_arcs(self, v):
        """Per arc, ``(t_in, t_out, pi)`` or ``(t_c, nu)``."""
        v = np.asarray(v, dtype=float).reshape(-1)
        out = []
        for (kind, _, _), sl in zip(self.arcs, self._arc_slices):
            w = v[sl]
            out.append((w[0], w[1], w[2:].copy()) if kind == "boundary" else (w[0], w[1]))
        return out

    # -- the system ---------------------------------------------------------

    def _segment(self, c, ta, dta, z, D, tb, dtb, jac, cost=False):
        """Propagate a free (``c is None``) or boundary arc; chain ``D = dz/dv``."""
        k, p = self.kernels, self.params
        if not tb > ta:
            raise ShootingError(f"arc times out of order: {ta} then {tb}")
        if c is None:
            r = k.propagate(ta, z, tb, p, stm=jac, cost=cost, constraint_events=False,
                            **self.options)
            Fa = k.rhs_arc(ta, z, p, r["arc0"] if k.N_MODES else None)
        else:
            r = k.propagate_boundary(c, ta, z, tb, p, stm=jac, cost=cost, **self.options)
            Fa = k.boundary_rhs(c, ta, z, p)
        if r["status"] != 0:
            raise ShootingError(f"propagation failed with status {r['status']}")
        if jac:
            Phi = r["stm"]
            D = Phi @ D + np.outer(r["dz_dt1"], dtb) - np.outer(Phi @ Fa, dta)
        return r, D

    def _evaluate(self, v, jac, record=False):
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
        for (kind, c, q), sl in zip(self.arcs, self._arc_slices):
            w = v[sl]
            t_in, d_in = w[0], unit(sl.start)
            r, D = self._segment(None, t, dt, z, D, t_in, d_in, jac, cost=record)
            segments.append((None, t, t_in, z, r))
            z = r["z"]
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
            Hf_t, Hf_z = k.hamiltonian_gradient(t_in, z, p)
            Hb_t, Hb_z = k.hamiltonian_gradient(t_in, zp, p, boundary=c)
            rows.append(k.hamiltonian(t_in, z, p) - k.boundary_hamiltonian(c, t_in, zp, p))
            grads.append((Hf_t - Hb_t) * d_in + Hf_z @ D - Hb_z @ Dp if jac else None)
            t_out, d_out = w[1], unit(sl.start + 1)
            r, D = self._segment(c, t_in, d_in, zp, Dp if jac else None, t_out, d_out, jac,
                                 cost=record)
            segments.append((c, t_in, t_out, zp, r))
            z = r["z"]
            # H continuous at the exit; the costates are continuous there.
            Hb_t, Hb_z = k.hamiltonian_gradient(t_out, z, p, boundary=c)
            Hf_t, Hf_z = k.hamiltonian_gradient(t_out, z, p)
            rows.append(k.boundary_hamiltonian(c, t_out, z, p) - k.hamiltonian(t_out, z, p))
            grads.append((Hb_t - Hf_t) * d_out + (Hb_z - Hf_z) @ D if jac else None)
            t, dt = t_out, d_out
        d_tf = unit(m + n + k.N_MULTIPLIERS) if k.FREE_FINAL_TIME else np.zeros(N)
        r, D = self._segment(None, t, dt, z, D, tf, d_tf, jac, cost=record)
        segments.append((None, t, tf, z, r))
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
        raise NotImplementedError("use MultiArcShooting.from_penalty for a starting point")

    def solve(self, guess, method=None, tol=1e-10, options=None):
        """As :meth:`SingleShooting.solve`. ``method`` defaults to ``"hybr"``,
        or ``"lm"`` with bang-bang controls: there the control jumps at a
        junction, ``H`` continuity reads ``|S| = 0`` (a kink at the root) and
        the Broyden updates of ``"hybr"`` stall on it, while ``"lm"`` takes the
        exact Jacobian at every step."""
        if method is None:
            method = "lm" if self.kernels.N_SWITCHING else "hybr"
        return super().solve(guess, method=method, tol=tol, options=options)

    # -- results ------------------------------------------------------------

    def cost(self, v):
        """The running cost integrated over every arc."""
        _, _, segs = self._evaluate(np.asarray(v, dtype=float), jac=False, record=True)
        return float(sum(r["cost"] for *_, r in segs))

    def trajectory(self, v, n_points=401):
        """The extremal at ``v``, sampled on every arc.

        Returns a dict with ``t``, ``z``, ``u``, ``g`` (every path constraint)
        and ``arc`` (-1 on free arcs, else the active constraint), plus
        ``junctions`` (the arc boundaries).
        """
        from fast_numerical_integration.integrators import solve_ivp

        k, p = self.kernels, self.params
        _, _, segs = self._evaluate(np.asarray(v, dtype=float), jac=False)
        span = segs[-1][2] - self.t0
        T, Z, U, A = [], [], [], []
        for c, ta, tb, za, _ in segs:
            npts = max(int(n_points * (tb - ta) / span), 2)
            te = np.linspace(ta, tb, npts)
            if c is None:
                # Bang-bang switches are located as in propagate.
                sol = k.trajectory(ta, za, tb, p, t_eval=te, constraint_events=False,
                                   **self.options)
                ts, zs, us = sol["t"], sol["z"], sol["u"]
            else:
                sol = solve_ivp(k.BoundaryDynamics(c, p), (ta, tb), za,
                                method=self.options["method"], t_eval=te,
                                rtol=self.options["rtol"], atol=self.options["atol"])
                ts, zs = sol.t, sol.y.T
                us = [k.boundary_control(c, ti, zi, p) for ti, zi in zip(ts, zs)]
            for ti, zi, ui in zip(ts, zs, us):
                T.append(ti)
                Z.append(zi)
                U.append(ui)
                A.append(-1 if c is None else c)
        T, Z = np.array(T), np.array(Z)
        G = np.array([k.constraint(ti, zi, p) for ti, zi in zip(T, Z)]).reshape(len(T), -1)
        return {"t": T, "z": Z, "u": np.array(U).reshape(len(T), -1), "g": G,
                "arc": np.array(A), "junctions": np.array([s[1] for s in segs[1:]])}

    def constraint_violation(self, v, n_points=2001):
        """Per path constraint, the largest ``max(0, g)`` on the free arcs, or
        the largest bound violation of its boundary control on its arcs."""
        tr = self.trajectory(v, n_points)
        return self._violations(tr)[0]

    def _violations(self, tr):
        k, p = self.kernels, self.params
        free = tr["arc"] < 0
        out = np.zeros(k.N_CONSTRAINTS)
        if free.any():
            out = np.maximum(tr["g"][free].max(axis=0), 0.0)
        bound = []
        for kind, c, _ in self.arcs:
            if kind != "boundary":
                continue
            on = np.flatnonzero(tr["arc"] == c)
            worst = 0.0
            for i in on:
                lo, hi = k.control_bounds(tr["t"][i], tr["z"][i], p)
                u = tr["u"][i]
                worst = max(worst, float(np.max(np.maximum(lo - u, u - hi), initial=0.0)))
            bound.append(worst)
            out[c] = max(out[c], worst)
        return out, bound

    def diagnostics(self, v, n_points=401):
        """Checks of a solution beyond the residual.

        Returns a dict with ``free_arc_violation`` (per constraint), ``touch_nu``
        (should be >= 0), ``boundary_pi``, and per boundary arc the range of
        its multiplier ``mu`` (should be >= 0) and of its controls along the
        arc, and ``boundary_bound_violation``: how far those controls leave
        their bounds (0 when they stay inside).
        """
        k, p = self.kernels, self.params
        tr = self.trajectory(v, n_points)
        arcs = self.unpack_arcs(v)
        free = tr["arc"] < 0
        _, bound = self._violations(tr)
        out = {"free_arc_violation": (np.maximum(tr["g"][free].max(axis=0), 0.0) if free.any()
                                      else np.zeros(k.N_CONSTRAINTS)),
               "boundary_bound_violation": bound,
               "touch_nu": [a[1] for (kind, *_), a in zip(self.arcs, arcs) if kind == "touch"],
               "boundary_pi": [a[2] for (kind, *_), a in zip(self.arcs, arcs)
                               if kind == "boundary"],
               "boundary_mu": [], "boundary_u": []}
        for kind, c, _ in self.arcs:
            if kind != "boundary":
                continue
            on = tr["arc"] == c
            mu = [k.boundary_multiplier(c, ti, zi, p) for ti, zi in zip(tr["t"][on], tr["z"][on])]
            out["boundary_mu"].append((min(mu), max(mu)))
            out["boundary_u"].append((tr["u"][on].min(axis=0), tr["u"][on].max(axis=0)))
        return out

    # -- starting points ----------------------------------------------------

    @classmethod
    def from_penalty(cls, kernels, params, arcs, solution, penalty_params, **options):
        """A :class:`MultiArcShooting` and an initial guess, from a penalty solution.

        ``solution`` solves :class:`SingleShooting` at ``penalty_params`` (the
        same problem with a large penalty weight). Its crossings of ``g = 0``
        give the junction times: a boundary arc runs from the first to the
        last crossing of its violated interval (the penalty solution may dip
        back below the limit in between), and a touch sits at its middle. The costates come from ``solution``. The entry jump ``pi`` is
        fitted to the costates: integrating the boundary arc back from the
        penalty state at the exit gives ``lambda`` just after the entry, and
        the jump from the penalty ``lambda`` just before it is fitted to
        ``-dN/dx^T pi``.

        Returns ``(shooting, v0)``.
        """
        shoot = cls(kernels, params, arcs, **options)
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
