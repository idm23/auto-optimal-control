"""A small single-shooting harness for the benchmarks (not part of the package).

The unknowns are the initial costates, the terminal multipliers and, when
the final time is free, ``t_f``; the initial state is fixed. The residual
Jacobian is exact: the generated boundary Jacobians chained with the STM
from ``Kernels.propagate``. Root finding is scipy's MINPACK (``hybr`` or
``lm``).
"""

import numpy as np
from scipy.optimize import root


def unpack(k, v):
    n, nnu = k.N_STATES, k.N_MULTIPLIERS
    lam0, nu = v[:n], v[n:n + nnu]
    tf = v[n + nnu] if k.FREE_FINAL_TIME else None
    return lam0, nu, tf


def shooting_function(k, x0, params, t0=0.0, rtol=1e-12, atol=1e-12):
    """``F(v) -> (R, dR/dv)`` for ``v = [lam0, nu, (tf)]``; initial-state rows dropped."""
    n = k.N_STATES
    x0 = np.asarray(x0, dtype=float)
    p = np.asarray(params, dtype=float)
    fixed_tf = None if k.FREE_FINAL_TIME else float(k.system.problem.final_time.subs(
        dict(zip(k.system.params, p))))

    def F(v):
        lam0, nu, tf = unpack(k, v)
        if tf is not None and not tf > t0:
            raise FloatingPointError("final time must be after the initial time")
        tf = fixed_tf if tf is None else tf
        z0 = np.concatenate([x0, lam0])
        r = k.propagate(t0, z0, tf, p, stm=True, rtol=rtol, atol=atol)
        if r["status"] != 0:
            raise FloatingPointError(f"propagation failed with status {r['status']}")
        R = k.residual(z0, r["z"], nu, tf, p)[n:]
        J = k.residual_jacobian(z0, r["z"], nu, tf, p)
        Rzf = J["zf"][n:]
        cols = [Rzf @ r["stm"][:, n:], J["nu"][n:]]
        if k.FREE_FINAL_TIME:
            cols.append((J["tf"][n:] + Rzf @ r["dz_dt1"])[:, None])
        return R, np.hstack(cols)

    return F


def solve(k, x0, params, guess, method="hybr", tol=1e-11, **kw):
    """Newton-type solve from ``guess``; returns (v, residual norm, scipy result)."""
    F = shooting_function(k, x0, params, **kw)

    def fun(v):
        try:
            return F(v)
        except (FloatingPointError, RuntimeError):
            m = len(v)
            return np.full(m, 1e6), np.eye(m)

    sol = root(fun, np.asarray(guess, dtype=float), jac=True, method=method,
               options={"xtol": 1e-13} if method == "hybr" else {"xtol": 1e-14, "ftol": 1e-14})
    R, _ = fun(sol.x)
    return sol.x, float(np.linalg.norm(R)), sol


def multistart(k, x0, params, sample, n_tries=50, seed=0, method="hybr", tol=1e-9, **kw):
    """Solve from random guesses ``sample(rng)`` until one converges."""
    rng = np.random.default_rng(seed)
    best = None
    for i in range(n_tries):
        v, err, sol = solve(k, x0, params, sample(rng), method=method, **kw)
        if best is None or err < best[1]:
            best = (v, err, i + 1)
        if err < tol:
            break
    return best


def guess_multipliers(k, x0, lam0, params, tf=None, t0=0.0):
    """Least-squares terminal multipliers for a costate guess.

    The transversality residuals are linear in ``nu``; propagate the guess
    and fit ``nu`` to them, so only ``lam0`` (and ``tf``) need guessing.
    """
    n, nnu = k.N_STATES, k.N_MULTIPLIERS
    if nnu == 0:
        return np.zeros(0)
    p = np.asarray(params, dtype=float)
    if tf is None:
        tf = float(k.system.problem.final_time.subs(dict(zip(k.system.params, p))))
    z0 = np.concatenate([x0, lam0])
    zf = k.propagate(t0, z0, tf, p)["z"]
    nu0 = np.zeros(nnu)
    rows = slice(n + len(k.system.boundary.terminal), 2 * n + len(k.system.boundary.terminal))
    R = k.residual(z0, zf, nu0, tf, p)[rows]
    A = k.residual_jacobian(z0, zf, nu0, tf, p)["nu"][rows]
    return np.linalg.lstsq(A, -R, rcond=None)[0]


def continuation(solve_at, start, values, v0, min_step=1e-7, shrink=0.5, tol=1e-9):
    """Natural-parameter continuation from a solution ``v0`` at ``start``.

    ``solve_at(value, guess) -> (v, err)`` is called for each target in
    ``values``; when it fails, the step from the last solved value is halved.
    Returns the list of (value, v, err).
    """
    path = []
    v, current = np.asarray(v0, dtype=float), start
    targets = list(values)
    i = 0
    while i < len(targets):
        target = targets[i]
        v_new, err = solve_at(target, v)
        if err < tol:
            path.append((target, v_new, err))
            v, current = v_new, target
            i += 1
            continue
        mid = current + shrink * (target - current)
        if abs(mid - current) < min_step * max(1.0, abs(current)):
            raise RuntimeError(f"continuation stalled at {current} (residual {err:.2e})")
        targets.insert(i, mid)
    return path
