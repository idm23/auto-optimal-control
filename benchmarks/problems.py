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
"""

import time
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.optimize import brentq

from auto_optimal_control.build import build_kernels
from auto_optimal_control.pontryagin import derive
from auto_optimal_control.problem import Box, OptimalControlProblem, UnitVector
from shooting import continuation, guess_multipliers, solve


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
        p = [g, xf, yf]
        x0 = np.zeros(3)
        for _ in range(100):
            lam0 = rng.uniform(-1, 1, 3)
            tf = rng.uniform(0.2, 3.0)
            try:
                nu = guess_multipliers(k, x0, lam0, p, tf=tf)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
            v, err, _ = solve(k, x0, p, np.concatenate([lam0, nu, [tf]]))
            if err < 1e-10:
                break
        rows.append(Result("brachistochrone", f"t_f (g={g}, x_f={xf}, y_f={yf})", v[-1],
                           brachistochrone_exact(g, xf, yf), "closed form (cycloid)", 1e-10,
                           time.perf_counter() - start, {"residual": err}))
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
    p, x0 = ORBIT_PARAMS, np.array([1.0, 0.0, 0.0, 1.0])
    rng = np.random.default_rng(seed)
    for tries in range(1, 201):
        lam0 = rng.uniform(-2, 2, 4)
        v, err, _ = solve(k, x0, p, np.concatenate([lam0, guess_multipliers(k, x0, lam0, p)]))
        if err < 1e-10:
            break
    r = k.propagate(0.0, np.concatenate([x0, v[:4]]), p[-1], p)
    return [Result("orbit_raising", "r(t_f)", r["z"][0], 1.5252775945,
                   "YAPSS orbit-raising example (IPOPT); Bryson & Ho 1975 pp. 66-69 give 1.525",
                   1e-6, time.perf_counter() - start, {"residual": err, "random guesses": tries})]


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
    rng = np.random.default_rng(seed)
    for tries in range(1, 201):                     # costates in [0, 1], as in the paper
        lam0 = rng.uniform(0, 1, 7)
        v, err, _ = solve(kh, x0, base + [1.0], np.concatenate(
            [lam0, guess_multipliers(kh, x0, lam0, base + [1.0])]))
        if err < 1e-10:
            break

    def at(eps, guess):
        v, err, _ = solve(kh, x0, base + [eps], guess)
        return v, err

    path = continuation(at, 1.0, [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 1e-3, 1e-4], v)
    v, err, _ = solve(kf, x0, base, path[-1][1])
    r = kf.propagate(0.0, np.concatenate([x0, v[:7]]), M["tf"], base, cost=True)
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
    x0 = np.array([4.5, 0.0, np.pi / 4, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(seed)
    for tries in range(1, 301):
        lam0 = rng.uniform(-1, 1, 6) * 10 ** rng.uniform(-1, 1, 6)
        tf = rng.uniform(5, 15)
        try:
            nu = guess_multipliers(kh, x0, lam0, [5.0, 1.0], tf=tf)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        v, err, _ = solve(kh, x0, [5.0, 1.0], np.concatenate([lam0, nu, [tf]]))
        if err < 1e-10:
            break

    def at(eps, guess):
        v, err, _ = solve(kh, x0, [5.0, eps], guess)
        return v, err

    path = continuation(at, 1.0, [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 3e-3, 1e-3, 3e-4, 1e-4], v)
    v, err, _ = solve(kb, x0, [5.0], path[-1][1])
    r = kb.propagate(0.0, np.concatenate([x0, v[:6]]), v[-1], [5.0])
    return [Result("robot_arm", "t_f", v[-1], 9.1409,
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
    p, x0 = CART_POLE_PARAMS, np.zeros(4)

    def polish(lam0):
        v, err, _ = solve(k, x0, p, np.concatenate([lam0, guess_multipliers(k, x0, lam0, p)]))
        z0 = np.concatenate([x0, v[:4]])
        r = k.propagate(0.0, z0, p[6], p, cost=True)
        tr = k.trajectory(0.0, z0, p[6], p, n_points=2001)
        return v, err, r["cost"], float(np.abs(tr["z"][:, 0]).max()), float(np.abs(tr["u"]).max())

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
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
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


ALL = {
    "brachistochrone": run_brachistochrone,
    "orbit_raising": run_orbit_raising,
    "earth_mars": run_earth_mars,
    "robot_arm": run_robot_arm,
    "cart_pole": run_cart_pole,
}
