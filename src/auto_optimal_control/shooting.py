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
        tf = v[-1] if k.FREE_FINAL_TIME else self.fixed_tf
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


def continuation(solve_at, start, values, v0, *, min_step=1e-7, shrink=0.5):
    """Natural-parameter continuation from a solution ``v0`` at ``start``.

    ``solve_at(value, guess)`` returns a :class:`ShootingSolution` (or any
    object with ``v`` and ``success``) and is called for each target in
    ``values``, warm-started from the last solution. When it fails, a target
    partway (``shrink``) from the last solved value is inserted. Raises
    ``RuntimeError`` when that step falls below ``min_step`` (relative).
    Returns the list of ``(value, solution)``; the last entry is at
    ``values[-1]``.
    """
    path = []
    v, current = np.asarray(v0, dtype=float), start
    targets = list(values)
    i = 0
    while i < len(targets):
        target = targets[i]
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
