"""Direct transcription of an :class:`~auto_optimal_control.problem.OptimalControlProblem`,
**for comparison only**.

The indirect method is what this package is for. Nothing in the solver uses this
module: no seed, costate or structure comes from it. It exists to look at an answer
from the other side. Examples are the structure of a case the indirect route cannot
yet reach, and a reference value for a benchmark.

The transcription is multiple shooting:

- the controls are piecewise constant on ``intervals`` equal steps, and each step
  is integrated with ``substeps`` steps of RK4;
- path limits are imposed at every node, controls boxes are variable bounds, and
  the ends are equality constraints;
- the final time is a variable when it is free.

The problem's SymPy expressions are turned into CasADi expressions, so the
solver, IPOPT, gets exact sparse first and second derivatives. CasADi is an
optional dependency: ``pip install "auto-optimal-control[direct]"``.

The multipliers of the path limits at the nodes are returned too. Divided by the
step, they approximate the density of the limit's measure, ``eta`` in
``docs/notes/structure_certificates.md``.
"""

import time as _time
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from auto_optimal_control.problem import Box, UnitVector


def _casadi():
    try:
        import casadi
    except ImportError as exc:      # pragma: no cover - depends on the environment
        raise ImportError("direct transcription needs CasADi: "
                          'pip install "auto-optimal-control[direct]"') from exc
    return casadi


def _to_casadi(args, exprs):
    """A Python function mapping CasADi symbols ``args`` to ``exprs`` (SymPy) as CasADi."""
    ca = _casadi()
    names = {"sin": ca.sin, "cos": ca.cos, "tan": ca.tan, "exp": ca.exp, "log": ca.log,
             "sqrt": ca.sqrt, "Abs": ca.fabs, "Max": ca.fmax, "Min": ca.fmin,
             "atan2": ca.atan2, "asin": ca.asin, "acos": ca.acos, "atan": ca.atan,
             "sinh": ca.sinh, "cosh": ca.cosh, "tanh": ca.tanh, "sign": ca.sign,
             "pi": np.pi, "E": np.e}
    # Piecewise and the like have no CasADi counterpart worth guessing at.
    for e in exprs:
        if sp.sympify(e).has(sp.Piecewise):
            raise ValueError("direct transcription does not support Piecewise expressions")
    return sp.lambdify(args, list(exprs), modules=[names], cse=True)


@dataclass
class DirectSolution:
    """A direct transcription's answer: nodes ``t``, states ``x`` at the nodes, controls
    ``u`` on each interval, the final time and cost, and IPOPT's report.

    ``path_multipliers[k, c]`` is the multiplier of path limit ``c`` at node ``k``.
    """

    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    tf: float
    cost: float
    success: bool
    status: str
    iterations: int
    seconds: float
    path_values: np.ndarray
    path_multipliers: np.ndarray
    names: list = field(default_factory=list)
    control_bounds: tuple = None

    def arcs(self, tol=1e-6):
        """``{name: [(t_start, t_end), ...]}``: runs of nodes where each limit is active
        (``g >= -tol``, relative to the largest ``|g|``)."""
        out = {}
        for c, name in enumerate(self.names):
            g = self.path_values[:, c]
            on = g >= -tol * max(1.0, float(np.abs(g).max()))
            out[name] = _runs(self.t, on)
        return out

    def held(self, tol=1e-6):
        """``{control index: [(t_start, t_end, sign), ...]}``: runs of intervals where each
        control sits at a bound."""
        lo, hi = self.control_bounds
        out = {}
        tm = self.t
        for i in range(self.u.shape[1]):
            u = self.u[:, i]
            span = max(1.0, float(np.nanmax(np.abs(np.r_[lo[i], hi[i]])[np.isfinite(np.r_[lo[i], hi[i]])], initial=1.0)))
            up = u >= hi[i] - tol * span
            down = u <= lo[i] + tol * span
            runs = [(a, b, 1) for a, b in _runs(tm, np.r_[up, False], intervals=True)]
            runs += [(a, b, -1) for a, b in _runs(tm, np.r_[down, False], intervals=True)]
            if runs:
                out[i] = sorted(runs)
        return out


def _runs(t, mask, intervals=False):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out = []
    for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        end = t[min(r[-1] + 1, len(t) - 1)] if intervals else t[r[-1]]
        out.append((float(t[r[0]]), float(end)))
    return out


def solve(problem, params, *, intervals=100, substeps=2, tf_guess=1.0, tf_bounds=(1e-3, 1e3),
          guess=None, regularize=0.0, scale_limits=None, options=None, verbose=False):
    """Solve ``problem`` at ``params`` by direct multiple shooting with IPOPT.

    ``guess``: optional ``(t, x, u)`` arrays to start from (any time grid; they are
    interpolated). Otherwise the states go linearly from the initial to the final
    values, and the controls start at zero. ``regularize`` adds
    ``regularize * sum((u_i / r_i)^2) / 2`` to the running cost, with ``r_i`` the
    half-width of each control's box (the census's secondary cost has this form).
    ``scale_limits`` divides each path limit by a number (a scale of its own), to
    help IPOPT when limits differ by orders of magnitude.
    """
    ca = _casadi()
    P = problem
    p_num = [float(v) for v in params]
    subs = dict(zip(P.params, p_num))
    n, m = len(P.states), len(P.controls)
    if any(isinstance(cs, UnitVector) for cs in P.control_sets):
        raise NotImplementedError("UnitVector control sets are not supported here yet")

    lo, hi = np.full(m, -np.inf), np.full(m, np.inf)
    for cs in P.control_sets:
        if isinstance(cs, Box):
            i = P.controls.index(cs.control)
            lo[i] = float(cs.lower.subs(subs)) if cs.lower.is_finite is not False else -np.inf
            hi[i] = float(cs.upper.subs(subs)) if cs.upper.is_finite is not False else np.inf
    half = np.where(np.isfinite(hi - lo), (hi - lo) / 2, 1.0)

    t_sym = P.time
    x = ca.SX.sym("x", n)
    u = ca.SX.sym("u", m)
    ts = ca.SX.sym("t")
    sym = [t_sym, *P.states, *P.controls]
    run = sp.sympify(P.running_cost).subs(subs)
    exprs = [sp.sympify(e).subs(subs) for e in P.dynamics] + [run]
    f_py = _to_casadi(sym, exprs)
    out = f_py(ts, *[x[i] for i in range(n)], *[u[i] for i in range(m)])
    L = out[n] + (0.5 * regularize * ca.sumsqr(u / ca.DM(half)) if regularize else 0)
    F = ca.Function("f", [ts, x, u], [ca.vertcat(*out[:n]), L])

    gx = [sp.sympify(c.expr).subs(subs) for c in P.path_constraints]
    names = [c.name for c in P.path_constraints]
    scales = np.ones(len(gx)) if scale_limits is None else np.asarray(scale_limits, float)
    if gx:
        g_py = _to_casadi([t_sym, *P.states], gx)
        G = ca.Function("g", [ts, x], [ca.vertcat(*g_py(ts, *[x[i] for i in range(n)])) / ca.DM(scales)])

    t0 = float(sp.sympify(P.initial_time).subs(subs))
    x0 = [P.initial_state.get(s) for s in P.states]     # None: free
    psi = [sp.sympify(c).subs(subs) for c in P.terminal_constraints]
    phi = sp.sympify(P.terminal_cost).subs(subs)
    tfs = ca.SX.sym("tf")
    end_py = _to_casadi([t_sym, *P.states], psi + [phi])
    xf_s = ca.SX.sym("xf", n)
    end = end_py(tfs, *[xf_s[i] for i in range(n)])
    E = ca.Function("end", [tfs, xf_s], [ca.vertcat(*end[:len(psi)]) if psi else ca.SX(0, 1), end[len(psi)]])

    # One shooting interval: RK4 substeps, with the running cost as a quadrature.
    h = ca.SX.sym("h")
    xa, ua, ta = ca.SX.sym("xa", n), ca.SX.sym("ua", m), ca.SX.sym("ta")
    xk, q, tk = xa, 0, ta
    dt = h / substeps
    for _ in range(substeps):
        k1, l1 = F(tk, xk, ua)
        k2, l2 = F(tk + dt / 2, xk + dt / 2 * k1, ua)
        k3, l3 = F(tk + dt / 2, xk + dt / 2 * k2, ua)
        k4, l4 = F(tk + dt, xk + dt * k3, ua)
        xk = xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q = q + dt / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
        tk = tk + dt
    step = ca.Function("step", [ta, xa, ua, h], [xk, q])

    N = intervals
    free_tf = P.final_time is None
    tf_var = ca.MX.sym("tf") if free_tf else None
    tf_e = tf_var if free_tf else float(sp.sympify(P.final_time).subs(subs))
    X = [ca.MX.sym(f"X{k}", n) for k in range(N + 1)]
    U = [ca.MX.sym(f"U{k}", m) for k in range(N)]
    hh = (tf_e - t0) / N
    cons, clo, chi = [], [], []
    J = 0
    for k in range(N):
        xn, qk = step(t0 + k * hh, X[k], U[k], hh)
        cons.append(X[k + 1] - xn); clo += [0.0] * n; chi += [0.0] * n
        J = J + qk
    if gx:
        for k in range(N + 1):
            cons.append(G(t0 + k * hh, X[k])); clo += [-np.inf] * len(gx); chi += [0.0] * len(gx)
    psi_e, phi_e = E(tf_e, X[N])
    if psi:
        cons.append(psi_e); clo += [0.0] * len(psi); chi += [0.0] * len(psi)
    J = J + phi_e

    w = ([tf_var] if free_tf else []) + [v for pair in zip(X, U) for v in pair] + [X[N]]
    wlo, whi, w0 = [], [], []
    if guess is not None:
        tg, xg, ug = (np.asarray(a, float) for a in guess)
        span = tg[-1] - tg[0]
        tf_guess = float(span + t0) if free_tf else tf_guess
    tf0 = tf_guess if free_tf else float(tf_e)
    tn = t0 + (tf0 - t0) * np.arange(N + 1) / N
    if guess is not None:
        sg = (tg - tg[0]) / (tg[-1] - tg[0])
        sn = np.arange(N + 1) / N
        xs = np.array([np.interp(sn, sg, xg[:, i]) for i in range(n)]).T
        sm = (np.arange(N) + 0.5) / N
        us = np.array([np.interp(sm, sg[:len(ug)] if len(ug) == len(sg) else np.linspace(0, 1, len(ug)), ug[:, i])
                       for i in range(m)]).T
    else:
        xa0 = np.array([0.0 if v is None else float(sp.sympify(v).subs(subs)) for v in x0])
        xb0 = xa0.copy()
        for c in psi:       # x_i - value = 0 gives the final value
            syms = [s for s in P.states if c.has(s)]
            if len(syms) == 1 and sp.diff(c, syms[0]) == 1:
                xb0[P.states.index(syms[0])] = float(syms[0] - c)
        s = np.linspace(0, 1, N + 1)[:, None]
        xs = xa0 + s * (xb0 - xa0)
        us = np.zeros((N, m))
    if free_tf:
        wlo.append(tf_bounds[0]); whi.append(tf_bounds[1]); w0.append(tf0)
    for k in range(N + 1):
        if k == 0:
            xl = [(-np.inf if v is None else float(sp.sympify(v).subs(subs))) for v in x0]
            xh = [(np.inf if v is None else float(sp.sympify(v).subs(subs))) for v in x0]
        else:
            xl, xh = [-np.inf] * n, [np.inf] * n
        wlo += xl; whi += xh; w0 += list(xs[k])
        if k < N:
            wlo += list(lo); whi += list(hi); w0 += list(np.clip(us[k], lo, hi))
    nlp = {"x": ca.vertcat(*w), "f": J, "g": ca.vertcat(*cons)}
    opts = {"ipopt.print_level": 5 if verbose else 0, "print_time": False,
            "ipopt.max_iter": 3000, "ipopt.tol": 1e-9, "ipopt.acceptable_tol": 1e-7,
            "ipopt.mu_strategy": "adaptive"}
    opts.update(options or {})
    solver = ca.nlpsol("direct", "ipopt", nlp, opts)
    start = _time.perf_counter()
    res = solver(x0=w0, lbx=wlo, ubx=whi, lbg=clo, ubg=chi)
    seconds = _time.perf_counter() - start
    stats = solver.stats()
    wv = np.asarray(res["x"]).reshape(-1)
    off = 1 if free_tf else 0
    tf = float(wv[0]) if free_tf else float(tf_e)
    block = wv[off:off + N * (n + m)].reshape(N, n + m)
    xs = np.vstack([block[:, :n], wv[off + N * (n + m):][None, :n]])
    us = block[:, n:]
    lam = np.asarray(res["lam_g"]).reshape(-1)
    k0 = N * n
    ng = len(gx)
    Gf = G if gx else None
    tnodes = t0 + (tf - t0) * np.arange(N + 1) / N
    path_values = (np.array([np.asarray(Gf(tnodes[k], xs[k])).reshape(-1) * scales for k in range(N + 1)])
                   if gx else np.zeros((N + 1, 0)))
    path_mult = (lam[k0:k0 + (N + 1) * ng].reshape(N + 1, ng) / scales if gx else np.zeros((N + 1, 0)))
    return DirectSolution(t=tnodes, x=xs, u=us, tf=tf, cost=float(res["f"]),
                          success=bool(stats.get("success")), status=str(stats.get("return_status")),
                          iterations=int(stats.get("iter_count", -1)), seconds=seconds,
                          path_values=path_values, path_multipliers=path_mult, names=names,
                          control_bounds=(lo, hi))
