"""Canonical indirect-method benchmarks with literature reference values.

Each ``run_*`` function builds the problem, finds the extremal from scratch
(random costate guesses, continuation, or costates lifted from a direct
solution), and returns :class:`Result` rows comparing the indirect solution
with a published value.

Spacecraft:
- **orbit_raising**: maximum-radius low-thrust orbit transfer (Bryson & Ho 1975).
- **earth_mars**: fuel-optimal Earth-to-Mars rendezvous with energy-to-fuel
  continuation (benchmark of Taheri et al.; arXiv:2409.01490, Table 1).

Robotics:
- **robot_arm**: minimum-time COPS robot arm, bang-bang in three torques
  (Dolan, Moré & Munson, COPS 3.0).
- **cart_pole**: minimum-effort cart-pole swing-up (Kelly 2017, SIAM Review).

Classical:
- **brachistochrone**: minimum-time descent; closed-form cycloid.
- **bryson_denham**: minimum energy with the order-2 state limit ``x <= l``
  (Bryson, Denham & Dreyfus 1963), solved by exterior-penalty continuation in
  both regimes (a touch point and a boundary arc), then exactly by multi-arc
  shooting seeded from the penalty solution.
- **velocity_limit**: a rest-to-rest move with the order-1 limit ``v <= V``,
  exactly by multi-arc shooting, also with the force clipped to ``|u| <= u_max``.
- **speed_limit**: minimum time over a distance with ``|u| <= u_max`` and
  ``v <= V``: bang-bang with a boundary arc, exactly by multi-arc shooting
  seeded through a smoothing continuation (``1 + eps u^2/2``, ``eps -> 0``).

Rockets:
- **goddard**: Goddard's maximum-altitude climb through drag, full thrust, a
  singular arc and a coast (Graichen & Petit 2008), exactly by multi-arc
  shooting from random guesses; then with the dynamic-pressure limit
  ``q <= 10``: full thrust, the limit ridden, a joined singular arc and a
  coast, seeded from the free solution.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.optimize import brentq

from auto_optimal_control import examples
from auto_optimal_control.build import build_kernels
from auto_optimal_control.pontryagin import derive
from auto_optimal_control.problem import Box, OptimalControlProblem, UnitVector
from auto_optimal_control.shooting import (MultiArcShooting, ShootingError, SingleShooting,
                                           continuation, parameter_continuation)


@dataclass
class Result:
    benchmark: str
    quantity: str
    value: float
    reference: float
    source: str
    tolerance: float          # relative tolerance the check uses
    seconds: float
    details: dict = field(default_factory=dict)

    @property
    def rel_error(self):
        return abs(self.value - self.reference) / abs(self.reference)

    @property
    def ok(self):
        return self.rel_error <= self.tolerance


def _kernels(problem, cache_dir):
    return build_kernels(derive(problem), cache_dir=cache_dir)


# =============================================================================
# Brachistochrone
# =============================================================================

def brachistochrone_problem():
    """Minimum time to slide from rest at the origin to (x_f, y_f); y points down."""
    t = sp.Symbol("t", real=True)
    x, y, v, ax, ay = sp.symbols("x y v a_x a_y", real=True)
    g, xf, yf = sp.symbols("g x_f y_f", positive=True)
    return OptimalControlProblem(
        name="brachistochrone", time=t, states=[x, y, v], controls=[ax, ay], params=[g, xf, yf],
        dynamics=[v * ax, v * ay, g * ay], running_cost=1, initial_state=[0, 0, 0],
        final_state={x: xf, y: yf}, control_sets=[UnitVector([ax, ay])])


def brachistochrone_exact(g, xf, yf):
    """Cycloid x = R (phi - sin phi), y = R (1 - cos phi); t_f = phi_f sqrt(R / g)."""
    phi = brentq(lambda f: (f - np.sin(f)) / (1 - np.cos(f)) - xf / yf, 1e-6, 2 * np.pi - 1e-6)
    R = yf / (1 - np.cos(phi))
    return phi * np.sqrt(R / g)


def run_brachistochrone(cache_dir=None, seed=0):
    start = time.perf_counter()
    k = _kernels(brachistochrone_problem(), cache_dir)
    rows = []
    rng = np.random.default_rng(seed)
    for g, xf, yf in [(9.81, 1.0, 1.0), (9.81, 2.0, 0.5), (1.0, 1.0, 2.0)]:
        shoot = SingleShooting(k, [g, xf, yf])
        sol, _ = shoot.multistart(
            lambda rng: shoot.guess_multipliers(rng.uniform(-1, 1, 3), tf=rng.uniform(0.2, 3.0)),
            n_tries=100, seed=rng)
        rows.append(Result("brachistochrone", f"t_f (g={g}, x_f={xf}, y_f={yf})", sol.tf,
                           brachistochrone_exact(g, xf, yf), "closed form (cycloid)", 1e-10,
                           time.perf_counter() - start, {"residual": sol.residual_norm}))
    return rows


# =============================================================================
# Maximum-radius orbit transfer (Bryson & Ho)
# =============================================================================

ORBIT_PARAMS = [1.0, 0.1405, 1.0, 0.0749, 3.32]      # mu, T, m0, mdot, t_f


def orbit_raising_problem():
    """Polar coordinates; thrust direction (u_1, u_2) is a unit vector; mass m0 - mdot t."""
    t = sp.Symbol("t", real=True)
    r = sp.Symbol("r", positive=True)
    th, vr, vt, u1, u2 = sp.symbols("theta v_r v_theta u_1 u_2", real=True)
    mu, T, m0, mdot, tf = sp.symbols("mu T m_0 mdot t_f", positive=True)
    a = T / (m0 - mdot * t)
    return OptimalControlProblem(
        name="orbit_raising", time=t, states=[r, th, vr, vt], controls=[u1, u2],
        params=[mu, T, m0, mdot, tf],
        dynamics=[vr, vt / r, vt ** 2 / r - mu / r ** 2 + a * u1, -vr * vt / r + a * u2],
        terminal_cost=-r, terminal_constraints=[vr, vt - sp.sqrt(mu / r)], final_time=tf,
        initial_state=[1, 0, 0, sp.sqrt(mu)], control_sets=[UnitVector([u1, u2])])


def run_orbit_raising(cache_dir=None, seed=0):
    start = time.perf_counter()
    k = _kernels(orbit_raising_problem(), cache_dir)
    shoot = SingleShooting(k, ORBIT_PARAMS)
    sol, tries = shoot.multistart(lambda rng: shoot.guess_multipliers(rng.uniform(-2, 2, 4)),
                                  n_tries=200, seed=seed)
    r = k.propagate(sol.t0, sol.z0, sol.tf, ORBIT_PARAMS)
    return [Result("orbit_raising", "r(t_f)", r["z"][0], 1.5252775945,
                   "YAPSS orbit-raising example (IPOPT); Bryson & Ho 1975 pp. 66-69 give 1.525",
                   1e-6, time.perf_counter() - start,
                   {"residual": sol.residual_norm, "random guesses": tries})]


# =============================================================================
# Earth-to-Mars fuel-optimal rendezvous
# =============================================================================

AU, TU = 1.496e8, 3.1536e7                 # km, s (as in arXiv:2409.01490)
MARS = dict(
    mu=132712440018.0 * TU ** 2 / AU ** 3,
    m0=1000.0,
    T=0.5 / 1000.0 / 1000.0 * TU ** 2 / AU,              # 0.5 N on 1000 kg, in AU/TU^2
    c=2000.0 * 9.80665 / 1000.0 * TU / AU,               # Isp 2000 s
    rE=np.array([-140699693.0, -51614428.0, 980.0]) / AU,
    vE=np.array([9.774596, -28.07828, 4.337725e-4]) * TU / AU,
    rM=np.array([-172682023.0, 176959469.0, 7948912.0]) / AU,
    vM=np.array([-16.427384, -14.860506, 9.21486e-2]) * TU / AU,
    tf=348.795 * 86400.0 / TU,
)


def earth_mars_problem(homotopy):
    """Cartesian, mass normalized by m0. ``homotopy`` adds the energy-to-fuel term

    ``L = (T/c) (delta - eps delta (1 - delta))`` (Bertrand & Epenoy): eps = 1
    is energy-optimal with a smooth saturated throttle, eps -> 0 fuel-optimal.
    """
    t = sp.Symbol("t", real=True)
    r = sp.symbols("r_x r_y r_z", real=True)
    v = sp.symbols("v_x v_y v_z", real=True)
    m = sp.Symbol("m", positive=True)
    a = sp.symbols("a_x a_y a_z", real=True)
    d = sp.Symbol("delta", real=True)
    mu, T, c, tf = sp.symbols("mu T_max c t_f", positive=True)
    x0 = sp.symbols("x0_1:8", real=True)
    xf = sp.symbols("xf_1:7", real=True)
    eps = sp.Symbol("epsilon", positive=True)
    rn3 = (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** sp.Rational(3, 2)
    dynamics = list(v) + [-mu * r[i] / rn3 + T * d * a[i] / m for i in range(3)] + [-T * d / c]
    L = T / c * (d - eps * d * (1 - d)) if homotopy else T / c * d
    return OptimalControlProblem(
        name="earth_mars_" + ("homotopy" if homotopy else "fuel"), time=t,
        states=[*r, *v, m], controls=[*a, d],
        params=[mu, T, c, tf, *x0, *xf] + ([eps] if homotopy else []),
        dynamics=dynamics, running_cost=L, initial_state=list(x0),
        final_state=dict(zip([*r, *v], xf)), final_time=tf,
        control_sets=[UnitVector(a), Box(d, 0, 1)])


def run_earth_mars(cache_dir=None, seed=0):
    start = time.perf_counter()
    kh = _kernels(earth_mars_problem(True), cache_dir)
    kf = _kernels(earth_mars_problem(False), cache_dir)
    M = MARS
    x0 = np.concatenate([M["rE"], M["vE"], [1.0]])
    base = [M["mu"], M["T"], M["c"], M["tf"], *x0, *M["rM"], *M["vM"]]
    energy = SingleShooting(kh, base + [1.0])
    sol, tries = energy.multistart(        # costates in [0, 1], as in the paper
        lambda rng: energy.guess_multipliers(rng.uniform(0, 1, 7)), n_tries=200, seed=seed)

    def at(eps, guess):
        return SingleShooting(kh, base + [eps]).solve(guess, tol=1e-9)

    path = continuation(at, 1.0, [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 1e-3, 1e-4], sol.v)
    sol = SingleShooting(kf, base).solve(path[-1][1].v)
    err = sol.residual_norm
    r = kf.propagate(sol.t0, sol.z0, sol.tf, base, cost=True)
    days = r["switch_times"] * TU / 86400.0
    return [Result("earth_mars", "m(t_f) [kg]", r["z"][6] * M["m0"], 603.935,
                   "arXiv:2409.01490 Table 2 (HTS/L2 smoothing, rho = 1e-5)", 2e-5,
                   time.perf_counter() - start,
                   {"residual": err, "random guesses": tries, "continuation steps": len(path),
                    "switches (days)": np.round(days, 3).tolist()})]


# =============================================================================
# COPS robot arm, minimum time
# =============================================================================

def robot_arm_problem(homotopy):
    """COPS #8: L rho'' = u_rho, I_theta theta'' = u_theta, I_phi phi'' = u_phi, |u| <= 1.

    ``homotopy`` uses ``L = 1 + eps sum(u^2)`` (saturated controls); eps -> 0
    gives the bang-bang minimum-time problem.
    """
    t = sp.Symbol("t", real=True)
    rho, th, ph, wr, wt, wp = sp.symbols("rho theta phi rho_d theta_d phi_d", real=True)
    ur, ut, up = sp.symbols("u_rho u_theta u_phi", real=True)
    L, eps = sp.symbols("L epsilon", positive=True)
    I_phi = ((L - rho) ** 3 + rho ** 3) / 3
    I_theta = I_phi * sp.sin(ph) ** 2
    cost = 1 + eps * (ur ** 2 + ut ** 2 + up ** 2) if homotopy else 1
    return OptimalControlProblem(
        name="robot_arm_" + ("homotopy" if homotopy else "time"), time=t,
        states=[rho, th, ph, wr, wt, wp], controls=[ur, ut, up],
        params=[L] + ([eps] if homotopy else []),
        dynamics=[wr, wt, wp, ur / L, ut / I_theta, up / I_phi], running_cost=cost,
        initial_state=[sp.Rational(9, 2), 0, sp.pi / 4, 0, 0, 0],
        final_state={rho: sp.Rational(9, 2), th: 2 * sp.pi / 3, ph: sp.pi / 4,
                     wr: 0, wt: 0, wp: 0},
        control_sets=[Box(ur, -1, 1), Box(ut, -1, 1), Box(up, -1, 1)])


def run_robot_arm(cache_dir=None, seed=0):
    start = time.perf_counter()
    kh = _kernels(robot_arm_problem(True), cache_dir)
    kb = _kernels(robot_arm_problem(False), cache_dir)
    smooth = SingleShooting(kh, [5.0, 1.0])
    sol, tries = smooth.multistart(
        lambda rng: smooth.guess_multipliers(rng.uniform(-1, 1, 6) * 10 ** rng.uniform(-1, 1, 6),
                                             tf=rng.uniform(5, 15)),
        n_tries=300, seed=seed)

    def at(eps, guess):
        return SingleShooting(kh, [5.0, eps]).solve(guess, tol=1e-9)

    path = continuation(at, 1.0, [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 3e-3, 1e-3, 3e-4, 1e-4],
                        sol.v)
    sol = SingleShooting(kb, [5.0]).solve(path[-1][1].v)
    err = sol.residual_norm
    r = kb.propagate(sol.t0, sol.z0, sol.tf, [5.0])
    return [Result("robot_arm", "t_f", sol.tf, 9.1409,
                   "COPS robot arm; GAMS NOA / PROPT (nh = 800 discretization)", 2e-5,
                   time.perf_counter() - start,
                   {"residual": err, "random guesses": tries, "continuation steps": len(path),
                    "switches": np.round(r["switch_times"], 6).tolist(),
                    "switching control": [("u_rho", "u_theta", "u_phi")[i]
                                          for i in r["switch_index"]]})]


# =============================================================================
# Cart-pole swing-up (Kelly 2017)
# =============================================================================

CART_POLE_PARAMS = [1.0, 0.3, 0.5, 9.81, 20.0, 1.0, 2.0]   # m1, m2, l, g, u_max, d, T


def cart_pole_problem():
    """Kelly (2017) eqs. (6.1)-(6.3): min int u^2, |u| <= 20 N, T = 2 s, d = 1 m."""
    t = sp.Symbol("t", real=True)
    q1, q2, w1, w2, u = sp.symbols("q_1 q_2 qd_1 qd_2 u", real=True)
    m1, m2, l, g, umax, d, T = sp.symbols("m_1 m_2 l g u_max d T", positive=True)
    den = m1 + m2 * (1 - sp.cos(q2) ** 2)
    q1dd = (l * m2 * sp.sin(q2) * w2 ** 2 + u + m2 * g * sp.cos(q2) * sp.sin(q2)) / den
    q2dd = -(l * m2 * sp.cos(q2) * sp.sin(q2) * w2 ** 2 + u * sp.cos(q2)
             + (m1 + m2) * g * sp.sin(q2)) / (l * den)
    return OptimalControlProblem(
        name="cart_pole", time=t, states=[q1, q2, w1, w2], controls=[u],
        params=[m1, m2, l, g, umax, d, T], dynamics=[w1, w2, q1dd, q2dd], running_cost=u ** 2,
        initial_state=[0, 0, 0, 0], final_state={q1: d, q2: sp.pi, w1: 0, w2: 0}, final_time=T,
        control_sets=[Box(u, -umax, umax)])


def _cart_pole_plant(x, u, p):
    m1, m2, l, g = p[:4]
    q2, d2 = x[1], x[3]
    den = m1 + m2 * (1 - np.cos(q2) ** 2)
    return np.array([x[2], d2,
                     (l * m2 * np.sin(q2) * d2 ** 2 + u + m2 * g * np.cos(q2) * np.sin(q2)) / den,
                     -(l * m2 * np.cos(q2) * np.sin(q2) * d2 ** 2 + u * np.cos(q2)
                       + (m1 + m2) * g * np.sin(q2)) / (l * den)])


def cart_pole_direct(p=CART_POLE_PARAMS, N=40):
    """Kelly's direct method: Hermite-Simpson collocation from a straight-line guess (SLSQP).

    Returns (t, X, U, J). This is the method behind the published value.
    """
    from scipy.optimize import minimize

    T, umax, d = p[6], p[4], p[5]
    h = T / N

    def split(w):
        return w[:4 * (N + 1)].reshape(N + 1, 4), w[4 * (N + 1):]

    def defects(w):
        X, U = split(w)
        out = []
        for i in range(N):
            fk, fk1 = _cart_pole_plant(X[i], U[i], p), _cart_pole_plant(X[i + 1], U[i + 1], p)
            um = 0.5 * (U[i] + U[i + 1])
            xm = 0.5 * (X[i] + X[i + 1]) + h / 8 * (fk - fk1)
            out.append(X[i + 1] - X[i] - h / 6 * (fk + 4 * _cart_pole_plant(xm, um, p) + fk1))
        out += [X[0], X[-1] - [d, np.pi, 0, 0]]
        return np.concatenate(out)

    def cost(w):
        _, U = split(w)
        um = 0.5 * (U[:-1] + U[1:])
        return h / 6 * np.sum(U[:-1] ** 2 + 4 * um ** 2 + U[1:] ** 2)

    t = np.linspace(0, T, N + 1)
    w0 = np.concatenate([np.stack([d * t / T, np.pi * t / T, 0 * t, 0 * t], 1).ravel(),
                         np.zeros(N + 1)])
    bounds = [(-2, 2), (None, None), (None, None), (None, None)] * (N + 1) + [(-umax, umax)] * (N + 1)
    res = minimize(cost, w0, method="SLSQP", constraints={"type": "eq", "fun": defects},
                   bounds=bounds, options={"maxiter": 500, "ftol": 1e-12})
    X, U = split(res.x)
    return t, X, U, cost(res.x)


def costates_from_direct(t, X, U, p):
    """Initial costates consistent with a direct solution.

    Along the given trajectory the costate equation ``lam' = -(df/dx)^T lam``
    is linear, so ``lam(t) = Psi(t) lam0``. The control law ``u = -(df/du . lam)/2``
    then fixes ``lam0`` by least squares.
    """
    from scipy.integrate import solve_ivp as scipy_ivp
    from scipy.interpolate import CubicSpline

    Xs, Us = CubicSpline(t, X), CubicSpline(t, U)

    def jac(x, u, e=1e-7):
        A = np.column_stack([(_cart_pole_plant(x + dx, u, p) - _cart_pole_plant(x - dx, u, p)) / (2 * e)
                             for dx in np.eye(4) * e])
        b = (_cart_pole_plant(x, u + e, p) - _cart_pole_plant(x, u - e, p)) / (2 * e)
        return A, b

    def adjoint(s, Y):
        A, _ = jac(Xs(s), float(Us(s)))
        return (-A.T @ Y.reshape(4, 4)).ravel()

    sol = scipy_ivp(adjoint, (t[0], t[-1]), np.eye(4).ravel(), dense_output=True,
                    rtol=1e-10, atol=1e-12)
    samples = np.linspace(t[0], t[-1], 202)[1:-1]
    rows = [-0.5 * jac(Xs(s), float(Us(s)))[1] @ sol.sol(s).reshape(4, 4) for s in samples]
    return np.linalg.lstsq(np.array(rows), Us(samples), rcond=None)[0]


def run_cart_pole(cache_dir=None, seed=0, n_random=400):
    start = time.perf_counter()
    k = _kernels(cart_pole_problem(), cache_dir)
    p = CART_POLE_PARAMS
    shoot = SingleShooting(k, p)

    def polish(lam0):
        sol = shoot.solve(shoot.guess_multipliers(lam0))
        r = k.propagate(sol.t0, sol.z0, sol.tf, p, cost=True)
        tr = shoot.trajectory(sol.v, n_points=2001)
        return (sol.v, sol.residual_norm, r["cost"], float(np.abs(tr["z"][:, 0]).max()),
                float(np.abs(tr["u"]).max()))

    # 1) The published extremal: Kelly's direct method, then the indirect polish.
    t, X, U, J_direct = cart_pole_direct(p)
    v, err, J, xmax, umax = polish(costates_from_direct(t, X, U, p))
    rows = [Result("cart_pole", "J (published extremal)", J, 58.80819,
                   "ASSET reproduction of Kelly 2017 (LGL5, 64 segments)", 2e-5,
                   time.perf_counter() - start,
                   {"residual": err, "direct HS N=40 J": float(J_direct), "max |x|": xmax,
                    "max |u|": umax})]

    # 2) Random costate search: finds a different, cheaper extremal. It is not in
    # the literature, so it is checked independently: Kelly's equations are
    # simulated with SciPy under this extremal's control history.
    start2 = time.perf_counter()
    rng = np.random.default_rng(seed)
    found = {}
    for _ in range(n_random):
        lam0 = rng.uniform(-60, 60, 4)
        try:
            v, err, J, xmax, umax = polish(lam0)
        except (ShootingError, RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        if err < 1e-10:
            key = round(J, 6)
            hits = found[key][1] + 1 if key in found else 1
            found[key] = (v, hits, xmax, J)
    best = min(found)
    J_sim, x_end = simulate_cart_pole(k, found[best][0][:4], p)
    rows.append(Result("cart_pole", "J (best of random search)", found[best][3], float(J_sim),
                       "independent SciPy simulation of Kelly's equations under this u(t)",
                       1e-8, time.perf_counter() - start2,
                       {"extremals found (J: hits)": {j: found[j][1] for j in sorted(found)},
                        "max |x|": found[best][2],
                        "simulated end-state error": float(np.abs(x_end - [p[5], np.pi, 0, 0]).max()),
                        "published J": 58.80819}))
    return rows


def simulate_cart_pole(k, lam0, p, n=4001):
    """Cost and end state of the plant driven by the extremal's control, via SciPy."""
    from scipy.integrate import solve_ivp as scipy_ivp
    from scipy.interpolate import CubicSpline

    T = p[6]
    tr = k.trajectory(0.0, np.concatenate([np.zeros(4), lam0]), T, p, n_points=n)
    u = CubicSpline(tr["t"], tr["u"][:, 0])

    def f(t, y):
        ut = float(u(t))
        return np.append(_cart_pole_plant(y[:4], ut, p), ut ** 2)

    sol = scipy_ivp(f, (0.0, T), np.zeros(5), method="DOP853", rtol=1e-12, atol=1e-12)
    return sol.y[4, -1], sol.y[:4, -1]


# =============================================================================
# Bryson-Denham: an order-2 state limit
# =============================================================================

def bryson_denham_exact(ell):
    """Closed-form cost: touch at t = 1/2 for 1/6 <= l < 1/4, boundary arc below."""
    if ell < 1 / 6:
        return 4 / (9 * ell)
    a, b = -4 + 12 * ell, 4 - 16 * ell        # x = t + a t^2 + b t^3 on [0, 1/2]
    c0, c1, T = 2 * a, 6 * b, 0.5              # u = c0 + c1 t
    return c0 ** 2 * T + c0 * c1 * T ** 2 + c1 ** 2 * T ** 3 / 3


def run_bryson_denham(cache_dir=None, seed=0):
    """Raise the penalty weight rho from 1 by continuation. The penalty
    solution approaches the exact cost from below; the gap shrinks as rho
    grows, so the tolerance reflects the final rho, not the integration."""
    k = _kernels(examples.bryson_denham(), cache_dir)
    rows = []
    for ell, rho_max, tol, rtol in [(0.2, 1e8, 1e-10, 1e-4), (0.1, 1e7, 1e-8, 1e-3)]:
        start = time.perf_counter()
        v = SingleShooting(k, [ell, 1.0]).solve([0, 2.0, 0, 0]).v
        path = parameter_continuation(k, [ell, 1.0], "rho",
                                      np.logspace(0, np.log10(rho_max), 33), v,
                                      solve_options=dict(tol=tol))
        rho, sol = path[-1]
        r = k.propagate(0, sol.z0, 1.0, [ell, rho], cost=True)
        regime = "touch" if ell >= 1 / 6 else "boundary arc"
        rows.append(Result("bryson_denham", f"J (l={ell}, {regime}, rho={rho:.0e})", r["cost"],
                           bryson_denham_exact(ell), "closed form (Bryson, Denham & Dreyfus 1963)",
                           rtol, time.perf_counter() - start,
                           {"max_violation": sol.max_violation, "solves": len(path)}))
        # Exact: the penalty solution seeds multi-arc shooting with the arc structure.
        start = time.perf_counter()
        arcs = [("touch" if ell >= 1 / 6 else "boundary", "x_max")]
        shoot, v = MultiArcShooting.from_penalty(k, [ell, 0.0], arcs, sol, [ell, rho])
        ex = shoot.solve(v)
        rows.append(Result("bryson_denham", f"J (l={ell}, {regime}, exact arcs)", shoot.cost(ex.v),
                           bryson_denham_exact(ell), "closed form (Bryson, Denham & Dreyfus 1963)",
                           1e-10, time.perf_counter() - start,
                           {"residual": ex.residual_norm, "max_violation": ex.max_violation}))
    return rows


def run_velocity_limit(cache_dir=None, seed=0):
    """Order-1 limit v <= V on a rest-to-rest move: penalty continuation to
    rho = 100, then exact multi-arc shooting."""
    k = _kernels(examples.velocity_limited_move(), cache_dir)
    rows = []
    for V in (1.1, 1.2, 1.4):
        start = time.perf_counter()
        v = SingleShooting(k, [V, 1.0]).solve([12.0, 6.0, 0, 0]).v
        rho, sol = parameter_continuation(k, [V, 1.0], "rho", np.logspace(0, 2, 9), v)[-1]
        shoot, v = MultiArcShooting.from_penalty(k, [V, 0.0], [("boundary", "v_max")], sol,
                                                 [V, rho])
        ex = shoot.solve(v)
        t1 = 3 * (V - 1) / (2 * V)
        rows.append(Result("velocity_limit", f"J (V={V}, exact arcs)", shoot.cost(ex.v),
                           4 * V ** 2 / (3 * t1), "closed form (order-1 boundary arc)", 1e-10,
                           time.perf_counter() - start,
                           {"residual": ex.residual_norm, "t_in": shoot.unpack_arcs(ex.v)[0][0]}))
    # The force clipped as well: full force, a ramp onto the limit, the limit.
    k = _kernels(examples.velocity_limited_move(bounded=True), cache_dir)
    for V, umax in ((1.2, 8.0), (1.25, 7.0)):
        start = time.perf_counter()
        v = SingleShooting(k, [V, 1.0, umax]).solve([12.0, 6.0, 0, 0]).v
        rho, sol = parameter_continuation(k, [V, 1.0, umax], "rho", np.logspace(0, 2, 9), v)[-1]
        shoot, v = MultiArcShooting.from_penalty(k, [V, 0.0, umax], [("boundary", "v_max")], sol,
                                                 [V, rho, umax])
        ex = shoot.solve(v)
        J, _, s_full = examples.velocity_limited_move_bounded_exact(V, umax)
        rows.append(Result("velocity_limit", f"J (V={V}, |u|<={umax}, exact arcs)",
                           shoot.cost(ex.v), J, "closed form (saturated, then a boundary arc)",
                           1e-10, time.perf_counter() - start,
                           {"residual": ex.residual_norm, "full_force": s_full,
                            "bound_violation": shoot.diagnostics(ex.v)["boundary_bound_violation"]}))
    return rows


def speed_limit_record(V, D, umax, cache_dir=None, eps_min=1e-4, record=None):
    """Smoothed min time, penalty -> exact arcs -> eps down -> bang-bang exact arcs.

    Returns ``(bang-bang shooting, its solution, eps path, smoothed kernels)``.
    ``record`` wraps the shooting classes (the animation records evaluations).
    """
    wrap = record or (lambda cls: cls)
    ks = _kernels(examples.speed_limited_min_time(smoothed=True), cache_dir)
    kb = _kernels(examples.speed_limited_min_time(), cache_dir)
    base = [V, D, umax]
    single = SingleShooting(ks, base + [1.0, 1.0])
    # Costates on the scale of the unconstrained run (full power for half of
    # t_guess, peaking at v_peak): lam_x ~ -1/v_peak, lam_v(0) ~ lam_x t_guess/2.
    t_guess = 2 * np.sqrt(D / umax)
    v_peak = umax * t_guess / 2

    def sample(r):
        lx = -r.uniform(0.3, 3) / v_peak
        return single.pack([lx, lx * r.uniform(0.2, 1.5) * t_guess / 2],
                           tf=r.uniform(0.8, 1.5) * t_guess)

    sol, _ = single.multistart(sample, n_tries=200, seed=0)
    path = parameter_continuation(ks, base + [1.0, 1.0], "rho", np.logspace(0, 1.5, 7), sol.v,
                                  solve_options=dict(tol=1e-9))
    rho, pen = path[-1]
    arcs = [("boundary", "v_max")]
    shoot, v = wrap(MultiArcShooting).from_penalty(ks, base + [1.0, 0.0], arcs, pen,
                                                   base + [1.0, rho])
    first = shoot.solve(v)

    def solve_at(eps, guess):
        return wrap(MultiArcShooting)(ks, base + [eps, 0.0], arcs).solve(guess)

    n = int(round(4 * np.log10(1 / eps_min))) + 1
    eps_path = [(1.0, first)] + continuation(solve_at, 1.0, np.logspace(0, np.log10(eps_min), n)[1:],
                                             first.v)
    bang = wrap(MultiArcShooting)(kb, base, arcs)
    return bang, bang.solve(eps_path[-1][1].v), {"penalty": path, "seed": (shoot, v),
                                                  "eps": eps_path}, ks


def run_speed_limit(cache_dir=None, seed=0):
    """Minimum time with |u| <= u_max and v <= V: t_f = D/V + V/u_max."""
    rows = []
    for V, D, umax in ((1.0, 3.0, 1.0), (2.0, 10.0, 0.5)):
        start = time.perf_counter()
        bang, sol, _, _ = speed_limit_record(V, D, umax, cache_dir)
        t_in, t_out, _ = bang.unpack_arcs(sol.v)[0]
        rows.append(Result("speed_limit", f"t_f (V={V}, D={D}, u_max={umax}, bang-bang exact arcs)",
                           sol.tf, D / V + V / umax, "closed form (full power, cruise, brake)",
                           1e-10, time.perf_counter() - start,
                           {"residual": sol.residual_norm, "t_in": t_in, "t_out": t_out}))
    return rows


GODDARD_REFERENCE = {"free": (1.01283692, 0.19885626), "q": (1.01271727, 0.20407108)}
GODDARD_SOURCE = "Graichen & Petit (2008), collocation with saturation functions"


def goddard_kernels(cache_dir=None):
    """Kernels of Goddard's rocket without and with the dynamic-pressure limit."""
    return tuple(build_kernels(derive(examples.goddard(q), singular=True), cache_dir=cache_dir)
                 for q in (False, True))


def goddard_free(k, guess=None, record=None):
    """Full thrust, singular, coast, from a rough guess (or ``guess``)."""
    cls = record(MultiArcShooting) if record else MultiArcShooting
    shoot = cls(k, list(examples.GODDARD_PARAMS), [("singular", "u", "upper", "lower")])
    if guess is None:
        guess = shoot.pack_arcs(shoot.pack([-2.5, -0.18, 0.005], [-0.03], 0.2), [(0.03, 0.1)])
    return shoot, shoot.solve(guess)


def goddard_q(kq, free, free_sol, record=None):
    """Full thrust, ride q = q_max, singular, coast: seeded from the free solution.

    The free extremal's q rises above the limit and falls back: the crossings
    give the boundary arc, the jump ``pi`` starts at 0 and the singular arc
    ends where the free one did.
    """
    params = list(examples.GODDARD_Q_PARAMS)
    cls = record(MultiArcShooting) if record else MultiArcShooting
    shoot = cls(kq, params, [("boundary", "q_max"), ("singular", "u", "upper", "lower", "joined")])
    tr = free.trajectory(free_sol.v, n_points=4001)
    h, v = tr["z"][:, 0], tr["z"][:, 1]
    q = params[5] * v ** 2 * np.exp(-params[1] * (h - 1)) / 2
    over = np.flatnonzero(q > params[6])
    t_in, t_out = tr["t"][over[0]], tr["t"][over[-1]]
    t_sing = free.unpack_arcs(free_sol.v)[0][1]
    guess = shoot.pack_arcs(free_sol.v, [(t_in, t_out, [0.0]), max(t_sing, t_out + 1e-3)])
    return shoot, shoot.solve(guess), guess


def run_goddard(cache_dir=None, seed=0):
    """Full thrust, singular, coast: the exact structure from random guesses.

    Then the dynamic-pressure limit, from that solution."""
    start = time.perf_counter()
    k, kq = goddard_kernels(cache_dir)
    shoot = MultiArcShooting(k, list(examples.GODDARD_PARAMS),
                             [("singular", "u", "upper", "lower")])

    def sample(rng):
        lam0 = [-rng.uniform(1, 6), -rng.uniform(0, 0.5), rng.uniform(-0.1, 0.1)]
        t_in = rng.uniform(0.005, 0.05)
        return shoot.pack_arcs(shoot.pack(lam0, [rng.uniform(-0.2, 0.2)], rng.uniform(0.15, 0.3)),
                               [(t_in, t_in + rng.uniform(0.02, 0.1))])

    sol, tries = shoot.multistart(sample, n_tries=200, seed=seed)
    t_in, t_out = shoot.unpack_arcs(sol.v)[0]
    h_f = shoot.trajectory(sol.v)["z"][-1, 0]
    d = shoot.diagnostics(sol.v)
    details = {"residual": sol.residual_norm, "random guesses": tries, "t_in": t_in,
               "t_out": t_out, "singular thrust": [float(d["singular_u"][0][0][0]),
                                                   float(d["singular_u"][0][1][0])],
               "Legendre-Clebsch min": d["singular_legendre"][0][0]}
    seconds = time.perf_counter() - start
    (h_ref, tf_ref), (hq_ref, tfq_ref) = GODDARD_REFERENCE["free"], GODDARD_REFERENCE["q"]
    rows = [Result("goddard", "h(t_f)", h_f, h_ref, GODDARD_SOURCE, 1e-8, seconds, details),
            Result("goddard", "t_f", sol.tf, tf_ref, GODDARD_SOURCE, 1e-7, seconds, details)]

    start = time.perf_counter()
    qshoot, qsol, _ = goddard_q(kq, shoot, sol)
    (b_in, b_out, pi), s_out = qshoot.unpack_arcs(qsol.v)
    hq_f = qshoot.trajectory(qsol.v)["z"][-1, 0]
    d = qshoot.diagnostics(qsol.v)
    details = {"residual": qsol.residual_norm, "max q - q_max": qsol.max_violation,
               "boundary arc": [b_in, b_out], "singular exit": s_out, "pi": float(pi[0]),
               "mu min": d["boundary_mu"][0][0], "Legendre-Clebsch min":
               d["singular_legendre"][0][0]}
    seconds = time.perf_counter() - start
    # The reference keeps q strictly inside the limit (saturation functions
    # with a vanishing penalty); the exact extremal climbs 4e-7 higher.
    rows += [Result("goddard", "h(t_f), q <= 10", hq_f, hq_ref, GODDARD_SOURCE, 1e-6, seconds,
                    details),
             Result("goddard", "t_f, q <= 10", qsol.tf, tfq_ref, GODDARD_SOURCE, 2e-4, seconds,
                    details)]
    return rows


ALL = {
    "brachistochrone": run_brachistochrone,
    "bryson_denham": run_bryson_denham,
    "velocity_limit": run_velocity_limit,
    "speed_limit": run_speed_limit,
    "orbit_raising": run_orbit_raising,
    "earth_mars": run_earth_mars,
    "robot_arm": run_robot_arm,
    "cart_pole": run_cart_pole,
    "goddard": run_goddard,
}
