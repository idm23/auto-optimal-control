"""Integrate derived extremals with fast-numerical-integration.

The derived necessary conditions are lambdified (``NumericSystem``) and
integrated with FNI's ``solve_ivp`` from known optimal costates. Each
trajectory must reproduce the textbook solution, satisfy every derived
boundary condition, and keep the Hamiltonian constant.
"""

import numpy as np
import pytest

from auto_optimal_control import examples
from auto_optimal_control.numeric import NumericSystem
from auto_optimal_control.pontryagin import derive
from fast_numerical_integration.integrators import solve_ivp

TOL = dict(method="DOP853", rtol=1e-12, atol=1e-12)


def numeric(problem):
    return NumericSystem(derive(problem))


def hamiltonian_along(num, sol, t, p):
    return np.array([num.hamiltonian(ti, sol(ti), p) for ti in t])


def test_zermelo_heading_law_and_free_time_condition():
    """tan(theta) = tan(theta_0) - k t and y = (V/k)(sec theta_0 - sec theta)."""
    num = numeric(examples.zermelo())
    V, k, theta0, tf = 1.3, 0.8, 1.1, 1.5
    lam0 = -np.array([np.cos(theta0), np.sin(theta0)]) / V      # H(t_0) = 1 - V |lam| = 0
    p = [V, k, 0.0, 0.0]
    r = solve_ivp(num.ode(p), (0.0, tf), [0.0, 0.0, *lam0], dense_output=True, **TOL)
    assert r.success

    t = np.linspace(0.0, tf, 31)
    for ti in t:
        z = r.sol(ti)
        ux, uy = num.control(ti, z, p)
        assert np.hypot(ux, uy) == pytest.approx(1.0, abs=1e-14)
        theta = np.arctan2(uy, ux)
        assert np.tan(theta) == pytest.approx(np.tan(theta0) - k * ti, abs=1e-9)
        assert z[1] == pytest.approx(V / k * (1 / np.cos(theta0) - 1 / np.cos(theta)), abs=1e-9)
    np.testing.assert_allclose(hamiltonian_along(num, r.sol, t, p), 0.0, atol=1e-10)

    # The arrival point of this extremal is a target it reaches optimally:
    # every boundary condition holds with nu = lam(t_f).
    zf = r.y[:, -1]
    p_target = [V, k, zf[0], zf[1]]
    res = num.residual(r.y[:, 0], zf, zf[2:], tf, p_target)
    np.testing.assert_allclose(res, 0.0, atol=1e-10)


@pytest.mark.parametrize("a,umax", [(1.0, 1.0), (2.5, 0.5), (0.3, 4.0)])
def test_min_time_double_integrator(a, umax):
    """From rest at x = a: switch at sqrt(a/u_max), arrive at 2 sqrt(a/u_max)."""
    num = numeric(examples.double_integrator_min_time())
    p = [umax, a, 0.0]
    ts = np.sqrt(a / umax)
    lam_v0 = 1.0 / umax                       # H(0) = 1 - u_max lam_v0 = 0
    lam_x = lam_v0 / ts                       # lam_v(t) = lam_v0 - lam_x t vanishes at ts
    z0 = [a, 0.0, lam_x, lam_v0]
    r = solve_ivp(num.ode(p), (0.0, 2 * ts), z0, events=num.switching_events(p),
                  dense_output=True, **TOL)
    assert r.success
    np.testing.assert_allclose(r.t_events[0], [ts], rtol=1e-12)
    assert num.control(0.0, z0, p)[0] == -umax
    assert num.control(1.9 * ts, r.sol(1.9 * ts), p)[0] == umax

    zf = r.y[:, -1]
    np.testing.assert_allclose(zf[:2], 0.0, atol=1e-9)
    res = num.residual(r.y[:, 0], zf, zf[2:], 2 * ts, p)
    np.testing.assert_allclose(res, 0.0, atol=1e-9)
    t = np.linspace(0.0, 2 * ts, 41)
    np.testing.assert_allclose(hamiltonian_along(num, r.sol, t, p), 0.0, atol=1e-9)


def _min_energy_costates(a, b, T):
    """lam_x = c1, lam_v = c2 - c1 t, u = -lam_v; solve x(T) = v(T) = 0."""
    A = np.array([[T ** 3 / 6, -T ** 2 / 2], [T ** 2 / 2, -T]])
    return np.linalg.solve(A, [-(a + b * T), -b])


def test_min_energy_double_integrator_and_its_cost():
    num = numeric(examples.double_integrator_min_energy())
    a, b, T = 1.0, 0.5, 2.0
    p = [T, a, b]
    assert num.final_time(p) == T and num.initial_time(p) == 0.0
    c1, c2 = _min_energy_costates(a, b, T)

    def with_cost(t, y):
        return np.append(num.rhs(t, y[:4], p), num.cost_rate(t, y[:4], p))

    r = solve_ivp(with_cost, (0.0, T), [a, b, c1, c2, 0.0], **TOL)
    zf = r.y[:4, -1]
    np.testing.assert_allclose(num.residual(r.y[:4, 0], zf, zf[2:], p=p), 0.0, atol=1e-10)
    # J = int (c2 - c1 t)^2 / 2 dt
    exact = (c2 ** 2 * T - c1 * c2 * T ** 2 + c1 ** 2 * T ** 3 / 3) / 2
    assert r.y[4, -1] == pytest.approx(exact, rel=1e-11)


def test_saturated_control_clips_and_conserves_H():
    num = numeric(examples.double_integrator_min_energy(bounded=True))
    a, b, T = 1.0, 0.5, 2.0
    c1, c2 = _min_energy_costates(a, b, T)
    umax = 0.6 * abs(c2)                      # |u| peaks at t = 0 at |c2|
    p = [T, a, b, umax]
    r = solve_ivp(num.ode(p), (0.0, T), [a, b, c1, c2], dense_output=True, **TOL)
    t = np.linspace(0.0, T, 81)
    u = np.array([num.control(ti, r.sol(ti), p)[0] for ti in t])
    free = np.array([-r.sol(ti)[3] for ti in t])
    np.testing.assert_array_equal(u, np.clip(free, -umax, umax))
    assert np.any(np.abs(u) == umax) and np.any(np.abs(u) < umax)
    H = hamiltonian_along(num, r.sol, t, p)
    np.testing.assert_allclose(H, H[0], atol=1e-10)


def test_soft_terminal_cost_transversality():
    num = numeric(examples.soft_terminal_lq())
    T, w, a = 2.0, 3.0, 1.5
    lam = w * a / (1 + w * T)
    r = solve_ivp(num.ode([T, w, a]), (0.0, T), [a, lam], **TOL)
    zf = r.y[:, -1]
    assert zf[0] == pytest.approx(a / (1 + w * T), rel=1e-12)
    np.testing.assert_allclose(num.residual(r.y[:, 0], zf, p=[T, w, a]), 0.0, atol=1e-12)
    # Any other initial costate violates transversality.
    r2 = solve_ivp(num.ode([T, w, a]), (0.0, T), [a, 1.1 * lam], **TOL)
    assert abs(num.residual(r2.y[:, 0], r2.y[:, -1], p=[T, w, a])[1]) > 1e-3


def test_low_thrust_extremal_switches_and_conserves_H():
    """A bang-bang low-thrust extremal: H is constant across every switch."""
    num = numeric(examples.planar_low_thrust_fuel())
    p = [1.0, 0.05, 1.0, 1.0, 1.0, 1.5, 6.0]   # mu, T_max, c, m_0, r_0, r_f, t_f
    z0 = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.2, 0.1, 0.1, 0.95, 0.0])
    np.testing.assert_allclose(num.residual(z0, z0, [0, 0, 0], p=p)[:5], 0.0, atol=1e-15)
    r = solve_ivp(num.ode(p), (0.0, 6.0), z0, events=num.switching_events(p),
                  dense_output=True, **TOL)
    assert r.success
    switches = r.t_events[0]
    assert len(switches) == 3
    for ts, zs in zip(switches, r.y_events[0]):
        assert abs(num.switching(ts, zs, p)[0]) < 1e-12

    t = np.linspace(0.0, 6.0, 121)
    H = hamiltonian_along(num, r.sol, t, p)
    np.testing.assert_allclose(H, H[0], atol=1e-10)
    # Mass drops only while thrusting, at T_max / c.
    burn = [ti for ti in t if num.control(ti, r.sol(ti), p)[2] == 1.0]
    assert 0 < len(burn) < len(t)
    for ti in t:
        z = r.sol(ti)
        ax, ay, delta = num.control(ti, z, p)
        np.testing.assert_allclose([ax, ay], -z[7:9] / np.linalg.norm(z[7:9]), rtol=1e-14)
        np.testing.assert_allclose(num.rhs(ti, z, p)[4], -0.05 * delta, rtol=1e-14)
