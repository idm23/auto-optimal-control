"""Canonical indirect-method benchmarks with literature reference values.

Each ``run_*`` function gives :class:`~auto_optimal_control.homotopy.Homotopy`
the problem, its parameter values and a box (or a function) to draw initial
costates and ``t_f`` from, and nothing else: smoothing, penalties,
continuation and the arc structure are found automatically. It returns
:class:`Result` rows comparing the indirect solution with a published value.

Spacecraft:
- **orbit_raising**: maximum-radius low-thrust orbit transfer (Bryson & Ho 1975).
- **earth_mars**: fuel-optimal Earth-to-Mars rendezvous, bang-bang throttle
  (benchmark of Taheri et al.; arXiv:2409.01490, Table 1).

Robotics:
- **robot_arm**: minimum-time COPS robot arm, bang-bang in three torques
  (Dolan, Moré & Munson, COPS 3.0).
- **two_link_arm** / **two_link_arm_min_time**: a planar two-link arm with
  joint torque limits, minimum jerk; then minimum time with bounded jerks
  (degenerate as stated: the default policy adds a small jerk cost).
- **six_joint_arm** / **six_joint_arm_min_time**: six joints under gravity with torque
  limits, minimum jerk; then minimum time with bounded jerks.
- **cart_pole**: minimum-effort cart-pole swing-up (Kelly 2017, SIAM Review).
  Kelly's direct method is run for comparison only.

Classical:
- **brachistochrone**: minimum-time descent; closed-form cycloid.
- **bryson_denham**: minimum energy with the order-2 state limit ``x <= l``
  (Bryson, Denham & Dreyfus 1963): a touch point and a boundary arc.
- **velocity_limit**: a rest-to-rest move with the order-1 limit ``v <= V``,
  also with the force clipped to ``|u| <= u_max``.
- **speed_limit**: minimum time over a distance with ``|u| <= u_max`` and
  ``v <= V``: bang-bang with a boundary arc.

Rockets:
- **goddard**: Goddard's maximum-altitude climb through drag, full thrust, a
  singular arc and a coast (Graichen & Petit 2008); then with the
  dynamic-pressure limit ``q <= 10``: full thrust, the limit ridden, a joined
  singular arc and a coast.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.optimize import brentq

from auto_optimal_control import examples
from auto_optimal_control.build import build_kernels
from auto_optimal_control.pontryagin import derive
from auto_optimal_control.homotopy import Homotopy
from auto_optimal_control.problem import Box, OptimalControlProblem, UnitVector
from auto_optimal_control.robot import GRAVITY, SIX_JOINT_LINKS, joint_torques
# Re-exported for the animations in visuals/, which import them from here.
from auto_optimal_control.shooting import (  # noqa: F401
    MultiArcShooting, ShootingError, SingleShooting, continuation, parameter_continuation)


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


def _details(r):
    """What a :class:`~auto_optimal_control.homotopy.HomotopyResult` found, for the report."""
    return {"residual": r.solution.residual_norm,
            "unknowns scale": float(max(1.0, np.abs(r.solution.v).max())),
            "random draws": r.tries, "continuation steps": len(r.steps),
            "structure": list(r.arcs) or "free arcs"}


def run_brachistochrone(cache_dir=None, seed=0):
    rows = []
    rng = np.random.default_rng(seed)
    h = Homotopy(brachistochrone_problem(), cache_dir=cache_dir)
    for g, xf, yf in [(9.81, 1.0, 1.0), (9.81, 2.0, 0.5), (1.0, 1.0, 2.0)]:
        start = time.perf_counter()
        r = h.solve(
            [g, xf, yf], ([-1] * 3, [1] * 3), (0.2, 3.0), n_tries=100, seed=rng)
        rows.append(Result("brachistochrone", f"t_f (g={g}, x_f={xf}, y_f={yf})", r.solution.tf,
                           brachistochrone_exact(g, xf, yf), "closed form (cycloid)", 1e-10,
                           time.perf_counter() - start, _details(r)))
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
    r = Homotopy(orbit_raising_problem(), cache_dir=cache_dir).solve(
        ORBIT_PARAMS, ([-2] * 4, [2] * 4), seed=seed)
    k, sol = r.shooting.kernels, r.solution
    rf = k.propagate(sol.t0, sol.z0, sol.tf, r.params)
    return [Result("orbit_raising", "r(t_f)", rf["z"][0], 1.5252775945,
                   "YAPSS orbit-raising example (IPOPT); Bryson & Ho 1975 pp. 66-69 give 1.525",
                   1e-6, time.perf_counter() - start, _details(r))]


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
    """Fuel-optimal: bang-bang throttle, reached from the costate box of the paper."""
    start = time.perf_counter()
    M = MARS
    x0 = np.concatenate([M["rE"], M["vE"], [1.0]])
    base = [M["mu"], M["T"], M["c"], M["tf"], *x0, *M["rM"], *M["vM"]]
    r = Homotopy(earth_mars_problem(False), cache_dir=cache_dir).solve(
        base, ([0] * 7, [1] * 7), seed=seed)
    k, sol = r.shooting.kernels, r.solution
    rf = k.propagate(sol.t0, sol.z0, sol.tf, r.params)
    days = rf["switch_times"] * TU / 86400.0
    return [Result("earth_mars", "m(t_f) [kg]", rf["z"][6] * M["m0"], 603.935,
                   "arXiv:2409.01490 Table 2 (HTS/L2 smoothing, rho = 1e-5)", 2e-5,
                   time.perf_counter() - start,
                   {**_details(r), "switches (days)": np.round(days, 3).tolist()})]


# =============================================================================
# COPS robot arm, minimum time
# =============================================================================

def robot_arm_problem():
    """COPS #8: L rho'' = u_rho, I_theta theta'' = u_theta, I_phi phi'' = u_phi, |u| <= 1."""
    t = sp.Symbol("t", real=True)
    rho, th, ph, wr, wt, wp = sp.symbols("rho theta phi rho_d theta_d phi_d", real=True)
    ur, ut, up = sp.symbols("u_rho u_theta u_phi", real=True)
    L = sp.Symbol("L", positive=True)
    I_phi = ((L - rho) ** 3 + rho ** 3) / 3
    I_theta = I_phi * sp.sin(ph) ** 2
    return OptimalControlProblem(
        name="robot_arm", time=t, states=[rho, th, ph, wr, wt, wp], controls=[ur, ut, up],
        params=[L], dynamics=[wr, wt, wp, ur / L, ut / I_theta, up / I_phi], running_cost=1,
        initial_state=[sp.Rational(9, 2), 0, sp.pi / 4, 0, 0, 0],
        final_state={rho: sp.Rational(9, 2), th: 2 * sp.pi / 3, ph: sp.pi / 4,
                     wr: 0, wt: 0, wp: 0},
        control_sets=[Box(ur, -1, 1), Box(ut, -1, 1), Box(up, -1, 1)])


def run_robot_arm(cache_dir=None, seed=0):
    start = time.perf_counter()
    r = Homotopy(robot_arm_problem(), cache_dir=cache_dir).solve(
        [5.0], lambda rng: rng.uniform(-1, 1, 6) * 10 ** rng.uniform(-1, 1, 6), (5.0, 15.0),
        n_tries=300, seed=seed)
    k, sol = r.shooting.kernels, r.solution
    rf = k.propagate(sol.t0, sol.z0, sol.tf, r.params)
    return [Result("robot_arm", "t_f", sol.tf, 9.1409,
                   "COPS robot arm; GAMS NOA / PROPT (nh = 800 discretization)", 2e-5,
                   time.perf_counter() - start,
                   {**_details(r), "switches": np.round(rf["switch_times"], 6).tolist(),
                    "switching control": [("u_rho", "u_theta", "u_phi")[i]
                                          for i in rf["switch_index"]]})]


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





def run_cart_pole(cache_dir=None, seed=0, n_random=400):
    """Random costate draws, each solved to its extremal; the cheapest is checked by an
    independent simulation. Kelly's direct method is run only for comparison: from his
    straight-line guess it converges to the published, more expensive local minimum."""
    start = time.perf_counter()
    p = CART_POLE_PARAMS
    h = Homotopy(cart_pole_problem(), cache_dir=cache_dir)
    rng = np.random.default_rng(seed)
    found = {}
    for _ in range(n_random):
        r = h.solve(p, ([-60] * 4, [60] * 4), n_tries=1, seed=rng)
        if r.exact:
            key = round(r.objective, 6)
            hits = found[key][1] + 1 if key in found else 1
            found[key] = (r, hits)
    best = min(found)
    r = found[best][0]
    J_sim, x_end = simulate_cart_pole(r.shooting.kernels, r.solution.lam0, p)
    tr = r.shooting.trajectory(r.solution.v, n_points=2001)
    seconds = time.perf_counter() - start
    _, _, _, J_direct = cart_pole_direct(p)
    return [Result("cart_pole", "J (cheapest extremal found)", r.objective, float(J_sim),
                   "independent SciPy simulation of Kelly's equations under this u(t)", 1e-8,
                   seconds,
                   {"residual": r.solution.residual_norm, "random draws": n_random,
                    "extremals found (J: hits)": {j: found[j][1] for j in sorted(found)},
                    "max |x|": float(np.abs(tr["z"][:, 0]).max()),
                    "simulated end-state error": float(np.abs(x_end - [p[5], np.pi, 0, 0]).max()),
                    "published J (Kelly 2017, direct)": 58.80819,
                    "Kelly's direct method here (HS, N = 40), comparison only": float(J_direct)})]


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
    """A touch point (l = 0.2) and a boundary arc (l = 0.1), each found from a costate box."""
    rows = []
    h = Homotopy(examples.bryson_denham(), cache_dir=cache_dir)
    for ell in (0.2, 0.1):
        start = time.perf_counter()
        r = h.solve(
            [ell, 0.0], ([-10, -10], [10, 10]), seed=seed)
        regime = "touch" if ell >= 1 / 6 else "boundary arc"
        rows.append(Result("bryson_denham", f"J (l={ell}, {regime})", r.objective,
                           bryson_denham_exact(ell), "closed form (Bryson, Denham & Dreyfus 1963)",
                           1e-10, time.perf_counter() - start,
                           {**_details(r), "max_violation": r.solution.max_violation}))
    return rows


def run_velocity_limit(cache_dir=None, seed=0):
    """Order-1 limit v <= V on a rest-to-rest move, and with the force clipped as well."""
    rows = []
    h = Homotopy(examples.velocity_limited_move(), cache_dir=cache_dir)
    for V in (1.1, 1.2, 1.4):
        start = time.perf_counter()
        r = h.solve(
            [V, 0.0], ([-20, -20], [20, 20]), seed=seed)
        t1 = 3 * (V - 1) / (2 * V)
        rows.append(Result("velocity_limit", f"J (V={V})", r.objective, 4 * V ** 2 / (3 * t1),
                           "closed form (order-1 boundary arc)", 1e-10,
                           time.perf_counter() - start, _details(r)))
    h = Homotopy(examples.velocity_limited_move(bounded=True), cache_dir=cache_dir)
    for V, umax in ((1.2, 8.0), (1.25, 7.0)):
        start = time.perf_counter()
        r = h.solve([V, 0.0, umax], ([-20, -20], [20, 20]), seed=seed)
        J, _, s_full = examples.velocity_limited_move_bounded_exact(V, umax)
        # The cost carries the integration error at the saturation corner (the
        # restart state comes from the dense output of the step that found it):
        # about 1e-9, though the unknowns match the closed form to 1e-13.
        rows.append(Result("velocity_limit", f"J (V={V}, |u|<={umax})", r.objective, J,
                           "closed form (saturated, then a boundary arc)", 2e-9,
                           time.perf_counter() - start, {**_details(r), "full_force": s_full}))
    return rows


def run_speed_limit(cache_dir=None, seed=0):
    """Minimum time with |u| <= u_max and v <= V: t_f = D/V + V/u_max."""
    rows = []
    h = Homotopy(examples.speed_limited_min_time(), cache_dir=cache_dir)
    for V, D, umax in ((1.0, 3.0, 1.0), (2.0, 10.0, 0.5)):
        start = time.perf_counter()
        r = h.solve(
            [V, D, umax], ([-3, -3], [0, 3]), (1.0, 20.0), seed=seed)
        rows.append(Result("speed_limit", f"t_f (V={V}, D={D}, u_max={umax})", r.solution.tf,
                           D / V + V / umax, "closed form (full power, cruise, brake)", 1e-10,
                           time.perf_counter() - start, _details(r)))
    return rows


JERK_CASES = ((1.0, 1.0, 2.0, 10.0), (1.0, 1.0, 2.0, 4.0), (1.0, 1.0, 0.5, 10.0),
              (1.0, 1.0, 2.0, 1.0))


def run_jerk_limited(cache_dir=None, seed=0):
    """The S-curve: minimum time with |j| <= J, |a| <= A and v <= V, in each regime."""
    rows = []
    h = Homotopy(examples.jerk_limited_move(), cache_dir=cache_dir)
    for J, A, V, D in JERK_CASES:
        start = time.perf_counter()
        r = h.solve([J, A, V, D], ([-3, -3, -3], [3, 3, 3]), (0.5, 30.0), seed=seed)
        arcs = ", ".join(name for _, name in r.arcs) or "no arcs"
        rows.append(Result("jerk_limited", f"t_f (J={J}, A={A}, V={V}, D={D}; {arcs})",
                           r.solution.tf, examples.jerk_limited_min_time(J, A, V, D)[0],
                           "closed form (S-curve)", 1e-9, time.perf_counter() - start,
                           _details(r)))
    return rows


ARM_CASES = ((20.0, 7.0, "one after another"), (17.0, 6.5, "overlapping"),
             (15.0, 7.0, "nested"))


def simulate_two_link_arm(r, params):
    """The cost and the final state of the arm under the extremal's jerk, simulated
    with SciPy apart from the package; and the peak torques along it."""
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d

    tr = r.shooting.trajectory(r.solution.v, n_points=20001)
    t, u = tr["t"], tr["u"]
    keep = np.concatenate([[True], np.diff(t) > 0])
    jerk = interp1d(t[keep], u[keep], axis=0)
    T = params[4]
    sol = solve_ivp(lambda s, x: np.concatenate([x[2:6], jerk(min(s, t[-1]))]), (0, T),
                    np.zeros(6), rtol=1e-10, atol=1e-12, max_step=1e-3)
    tau1, tau2 = examples.two_link_arm_torques(*params[:4], *tr["z"][:, 1:6].T)
    J = float(np.trapezoid(0.5 * (u[keep] ** 2).sum(axis=1), t[keep]))
    return J, sol.y[:, -1], float(np.abs(tau1).max()), float(np.abs(tau2).max())


def run_two_link_arm(cache_dir=None, seed=0):
    """Minimum jerk on a two-link arm with torque limits: arcs in sequence, overlapping and nested."""
    rows = []
    h = Homotopy(examples.two_link_arm(), cache_dir=cache_dir)
    base = list(examples.TWO_LINK_ARM_PARAMS)
    for t1, t2, how in ARM_CASES:
        start = time.perf_counter()
        p = base + [t1, t2]
        r = h.solve(p, ([-200] * 6, [200] * 6), seed=seed)
        seconds = time.perf_counter() - start
        J, xf, tau1, tau2 = simulate_two_link_arm(r, p)
        details = {**_details(r), "max |tau1|": tau1, "max |tau2|": tau2,
                   "final state error": float(np.abs(xf - [p[5], p[6], 0, 0, 0, 0]).max())}
        rows.append(Result("two_link_arm", f"J (tau <= {t1}/{t2}, {how})", r.objective, J,
                           "independent SciPy simulation of the arm under this jerk", 1e-5,
                           seconds, details))
    return rows


ARM_MIN_TIME_CASES = ((2.2, 2.0, "joint 1 limited"), (2.5, 0.6, "joint 2 limited"),
                      (1.5, 0.4, "both, overlapping"))
ARM_MIN_TIME_BASE = [1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0]   # m1, m2, l1, l2, q1f, q2f, j_max
ARM_MIN_TIME_WEIGHT = 1e-2                               # the default SecondaryCost weight


def simulate_two_link_arm_min_time(r, params, weight=ARM_MIN_TIME_WEIGHT):
    """``t_f + w int (j / j_max)^2 / 2`` and the final state under the extremal's jerk,
    simulated with SciPy apart from the package; the peak torques and jerk along it."""
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d

    tr = r.shooting.trajectory(r.solution.v, n_points=20001)
    t, u = tr["t"], tr["u"]
    keep = np.concatenate([[True], np.diff(t) > 0])
    jerk = interp1d(t[keep], u[keep], axis=0)
    tf, jmax = t[-1], params[6]
    sol = solve_ivp(lambda s, x: np.concatenate([x[2:6], jerk(min(s, tf))]), (0, tf),
                    np.zeros(6), rtol=1e-10, atol=1e-12, max_step=1e-3)
    tau1, tau2 = examples.two_link_arm_torques(*params[:4], *tr["z"][:, 1:6].T)
    cost = tf + weight * float(np.trapezoid(0.5 * ((u[keep] / jmax) ** 2).sum(axis=1), t[keep]))
    return (cost, sol.y[:, -1], float(np.abs(tau1).max()), float(np.abs(tau2).max()),
            float(np.abs(u).max()))


def run_two_link_arm_min_time(cache_dir=None, seed=0):
    """The arm in minimum time with bounded jerks and torque limits: degenerate as
    stated, so the default policy adds the secondary cost ``w int (j / j_max)^2 / 2``."""
    rows = []
    h = Homotopy(examples.two_link_arm_min_time(), cache_dir=cache_dir)
    for t1, t2, how in ARM_MIN_TIME_CASES:
        start = time.perf_counter()
        p = ARM_MIN_TIME_BASE + [t1, t2]
        r = h.solve(p, ([-3] * 6, [3] * 6), (1.0, 20.0), seed=seed)
        seconds = time.perf_counter() - start
        cost, xf, tau1, tau2, jerk = simulate_two_link_arm_min_time(r, p)
        details = {**_details(r), "t_f": r.solution.tf, "max |tau1|": tau1,
                   "max |tau2|": tau2, "max |j|": jerk,
                   "final state error": float(np.abs(xf - [p[4], p[5], 0, 0, 0, 0]).max())}
        rows.append(Result("two_link_arm_min_time", f"t_f + w int j^2 (tau <= {t1}/{t2}, {how})",
                           r.objective, cost,
                           "independent SciPy simulation of the arm under this jerk", 1e-6,
                           seconds, details))
    return rows


SIX_JOINT_MOVE = (1.0, [1.0, -1.0, 0.8, 1.0, -0.8, 1.0])          # T, qf (from q = 0)
SIX_JOINT_CASES = ((0.8, 0.8, 0.8, 1.5, 1.5, 1.5, "three joints limited"),
                   (0.9, 0.9, 0.9, 0.9, 0.9, 0.9, "all six limited"))


def _six_joint_torques():
    q, w, a = (sp.symbols(f"{s}1:7", real=True) for s in "qwa")
    f = sp.lambdify((q, w, a), joint_torques(SIX_JOINT_LINKS, q, w, a, GRAVITY), "numpy")

    def torques(Q, W, A):
        return np.array([np.broadcast_to(np.asarray(v, dtype=float), Q.shape[:1])
                         for v in f(Q.T, W.T, A.T)]).T
    return torques


def run_six_joint_arm(cache_dir=None, seed=0):
    """Six joints under gravity, minimum jerk. With loose limits the joints follow
    independent quintics, in closed form. With tight ones, each limit a fraction of the way
    from the larger end-point torque to the peak of that motion, the reference is an
    independent replay of the jerk: integrated with SciPy, torques from the recursion."""
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d

    T, qf = SIX_JOINT_MOVE
    qf = np.array(qf)
    torques = _six_joint_torques()
    s = np.linspace(0, 1, 801)[:, None]
    tau = torques(qf * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5),
                  qf * (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T,
                  qf * (60 * s - 180 * s ** 2 + 120 * s ** 3) / T ** 2)
    peak = np.abs(tau).max(axis=0)
    ends = np.maximum(np.abs(tau[0]), np.abs(tau[-1]))
    h = Homotopy(examples.six_joint_arm(), cache_dir=cache_dir)
    box = ([-50] * 18, [50] * 18)
    rows = []

    start = time.perf_counter()
    r = h.solve([T, *np.zeros(6), *qf, *(1.5 * peak)], box, seed=seed)
    rows.append(Result("six_joint_arm", "J (limits loose: six quintics)", r.objective,
                       360.0 * float(qf @ qf) / T ** 5, "closed form (minimum-jerk quintics)",
                       1e-10, time.perf_counter() - start, _details(r)))

    for *fractions, how in SIX_JOINT_CASES:
        start = time.perf_counter()
        limits = ends + (peak - ends) * np.array(fractions)
        r = h.solve([T, *np.zeros(6), *qf, *limits], box, seed=seed)
        seconds = time.perf_counter() - start
        tr = r.shooting.trajectory(r.solution.v, n_points=40001)
        t, u = tr["t"], tr["u"]
        keep = np.concatenate([[True], np.diff(t) > 0])
        jerk = interp1d(t[keep], u[keep], axis=0)
        sol = solve_ivp(lambda ti, x: np.concatenate([x[6:], jerk(min(ti, t[-1]))]), (0, t[-1]),
                        np.zeros(18), rtol=1e-10, atol=1e-12, max_step=1e-3,
                        t_eval=np.linspace(0, t[-1], 4001))
        over = np.abs(torques(sol.y[:6].T, sol.y[6:12].T, sol.y[12:].T)) - limits
        cost = float(np.trapezoid(0.5 * (u[keep] ** 2).sum(axis=1), t[keep]))
        details = {**_details(r), "final state error":
                   float(np.abs(sol.y[:, -1] - np.concatenate([qf, np.zeros(12)])).max()),
                   "largest torque over its limit": float(over.max())}
        rows.append(Result("six_joint_arm", f"J ({how})", r.objective, cost,
                           "independent SciPy replay of the arm under this jerk", 1e-6, seconds,
                           details))
    return rows


SIX_JOINT_MIN_TIME_GOAL = [1.0, -0.9, 0.8, 1.1, -0.7, 0.6]
SIX_JOINT_JERK = 30.0
SIX_JOINT_WEIGHT = 1e-2
SIX_JOINT_MIN_TIME_FRACTIONS = (0.95, 0.9)


def run_six_joint_arm_min_time(cache_dir=None, seed=0):
    """Six joints in minimum time with bounded jerks: degenerate as stated, so the default
    policy adds a small jerk cost. Without active limits the slowest joint's time is close to
    the closed form. With tight ones the reference is an independent replay of the jerk."""
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d

    qf = np.array(SIX_JOINT_MIN_TIME_GOAL)
    J = SIX_JOINT_JERK
    torques = _six_joint_torques()
    h = Homotopy(examples.six_joint_arm_min_time(), cache_dir=cache_dir)
    base = [*np.zeros(6), *qf, *([J] * 6)]
    box, times = ([-0.5] * 18, [0.5] * 18), (0.3, 5.0)
    rows = []

    start = time.perf_counter()
    r = h.solve([*base, *([1e4] * 6)], box, times, seed=seed)
    rows.append(Result("six_joint_arm_min_time", "t_f (limits loose: the slowest joint alone)",
                       r.solution.tf, (32 * float(np.abs(qf).max()) / J) ** (1 / 3),
                       "closed form, without the small jerk cost", 1e-4,
                       time.perf_counter() - start, _details(r)))
    tr = r.shooting.trajectory(r.solution.v, n_points=4001)
    tau = torques(tr["z"][:, :6], tr["z"][:, 6:12], tr["z"][:, 12:18])
    peak = np.abs(tau).max(axis=0)
    ends = 1.001 * np.maximum(np.abs(tau[0]), np.abs(tau[-1]))

    for fraction in SIX_JOINT_MIN_TIME_FRACTIONS:
        start = time.perf_counter()
        limits = ends + fraction * (peak - ends).clip(0)
        r = h.solve([*base, *limits], box, times, seed=seed)
        seconds = time.perf_counter() - start
        tr = r.shooting.trajectory(r.solution.v, n_points=40001)
        t, u = tr["t"], tr["u"]
        keep = np.concatenate([[True], np.diff(t) > 0])
        jerk = interp1d(t[keep], u[keep], axis=0)
        sol = solve_ivp(lambda ti, x: np.concatenate([x[6:], jerk(min(ti, t[-1]))]), (0, t[-1]),
                        np.zeros(18), rtol=1e-10, atol=1e-12, max_step=1e-3,
                        t_eval=np.linspace(0, t[-1], 4001))
        over = np.abs(torques(sol.y[:6].T, sol.y[6:12].T, sol.y[12:].T)) - limits
        cost = t[-1] + SIX_JOINT_WEIGHT * float(np.trapezoid(
            0.5 * ((u[keep] / J) ** 2).sum(axis=1), t[keep]))
        details = {**_details(r), "t_f": r.solution.tf, "max |j|": float(np.abs(u).max()),
                   "final state error":
                   float(np.abs(sol.y[:, -1] - np.concatenate([qf, np.zeros(12)])).max()),
                   "largest torque over its limit": float(over.max())}
        rows.append(Result("six_joint_arm_min_time",
                           f"t_f + w int j^2 (all six limits at {fraction:.0%})", r.objective, cost,
                           "independent SciPy replay of the arm under this jerk", 1e-6, seconds,
                           details))
    return rows


GODDARD_REFERENCE = {"free": (1.01283692, 0.19885626), "q": (1.01271727, 0.20407108)}
GODDARD_SOURCE = "Graichen & Petit (2008), collocation with saturation functions"


GODDARD_COSTATES = ([-6, -0.5, -0.1], [-1, 0, 0.1])
GODDARD_FINAL_TIME = (0.15, 0.3)


def run_goddard(cache_dir=None, seed=0):
    """Full thrust, singular, coast; then with q <= 10: full thrust, the limit ridden, a
    joined singular arc and a coast. Both structures are found, from one costate box."""
    rows = []
    (h_ref, tf_ref), (hq_ref, tfq_ref) = GODDARD_REFERENCE["free"], GODDARD_REFERENCE["q"]
    for q, params, tols in ((False, examples.GODDARD_PARAMS, (1e-8, 1e-7)),
                            (True, examples.GODDARD_Q_PARAMS, (1e-6, 2e-4))):
        start = time.perf_counter()
        r = Homotopy(examples.goddard(q), cache_dir=cache_dir).solve(
            list(params), GODDARD_COSTATES, GODDARD_FINAL_TIME, seed=seed)
        d = r.shooting.diagnostics(r.solution.v)
        details = {**_details(r), "junctions": [np.round(np.atleast_1d(a[:2] if isinstance(
                       a, tuple) else a), 6).tolist() for a in r.shooting.unpack_arcs(r.solution.v)],
                   "Legendre-Clebsch min": d["singular_legendre"][0][0]}
        if q:
            details["max q - q_max"] = r.solution.max_violation
            details["mu min"] = d["boundary_mu"][0][0]
        seconds = time.perf_counter() - start
        label = ", q <= 10" if q else ""
        refs = (hq_ref, tfq_ref) if q else (h_ref, tf_ref)
        # The reference keeps q strictly inside the limit (saturation functions
        # with a vanishing penalty); the exact extremal climbs 4e-7 higher.
        rows += [Result("goddard", f"h(t_f){label}", -r.objective, refs[0], GODDARD_SOURCE,
                        tols[0], seconds, details),
                 Result("goddard", f"t_f{label}", r.solution.tf, refs[1], GODDARD_SOURCE,
                        tols[1], seconds, details)]
    return rows


ALL = {
    "brachistochrone": run_brachistochrone,
    "bryson_denham": run_bryson_denham,
    "velocity_limit": run_velocity_limit,
    "speed_limit": run_speed_limit,
    "jerk_limited": run_jerk_limited,
    "two_link_arm": run_two_link_arm,
    "two_link_arm_min_time": run_two_link_arm_min_time,
    "six_joint_arm": run_six_joint_arm,
    "six_joint_arm_min_time": run_six_joint_arm_min_time,
    "orbit_raising": run_orbit_raising,
    "earth_mars": run_earth_mars,
    "robot_arm": run_robot_arm,
    "cart_pole": run_cart_pole,
    "goddard": run_goddard,
}
