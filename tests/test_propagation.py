"""Arc-by-arc propagation of extremals: switches, cost, STM and parameter sensitivities.

``propagate`` is the forward map from an initial state and costate to the
extremal. Its state transition matrix must equal finite differences of that
map, including across bang-bang switches (where it jumps).
"""

import threading

import numpy as np
import pytest

from fast_numerical_integration.integrators import solve_ivp


def fd_columns(f, x, h):
    x = np.asarray(x, dtype=float)
    cols = []
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = h
        cols.append((f(x + e) - f(x - e)) / (2 * h))
    return np.array(cols).T


KW = dict(rtol=1e-12, atol=1e-13)

CASES = {
    # name: (params, z0, t1)
    "zermelo": ([1.3, 0.8, 0.0, 0.0], [0.0, 0.0, -0.35, -0.68], 1.5),
    "min_time": ([2.0, 1.0, 0.0], [1.0, 0.0, 0.5 / np.sqrt(0.5), 0.5], 2 * np.sqrt(0.5)),
    "min_energy_bounded": ([2.0, 1.0, 0.5, 0.8], [1.0, 0.5, 1.5, 1.6], 2.0),
    "low_thrust": ([1.0, 0.05, 1.0, 1.0, 1.0, 1.5, 6.0],
                   [1.0, 0.0, 0.0, 1.0, 1.0, 0.2, 0.1, 0.1, 0.95, 0.0], 6.0),
}


def test_min_time_switch_cost_and_endpoint(compiled):
    _, _, k = compiled["min_time"]
    for a, umax in [(1.0, 1.0), (2.5, 0.5)]:
        ts = np.sqrt(a / umax)
        r = k.propagate(0.0, [a, 0.0, 1.0 / (umax * ts), 1.0 / umax], 2 * ts, [umax, a, 0.0],
                        cost=True, **KW)
        assert r["status"] == 0
        np.testing.assert_allclose(r["switch_times"], [ts], rtol=1e-13)
        np.testing.assert_array_equal(r["arc0"], [0])     # u = -u_max first
        np.testing.assert_array_equal(r["arc"], [1])
        np.testing.assert_allclose(r["z"][:2], 0.0, atol=1e-12)
        assert r["cost"] == pytest.approx(2 * ts, rel=1e-13)  # minimum time: J = t_f


def test_min_energy_cost(compiled):
    _, _, k = compiled["min_energy"]
    a, b, T = 1.0, 0.5, 2.0
    c1, c2 = np.linalg.solve([[T ** 3 / 6, -T ** 2 / 2], [T ** 2 / 2, -T]], [-(a + b * T), -b])
    r = k.propagate(0.0, [a, b, c1, c2], T, [T, a, b], cost=True, **KW)
    assert r["cost"] == pytest.approx((c2 ** 2 * T - c1 * c2 * T ** 2 + c1 ** 2 * T ** 3 / 3) / 2,
                                      rel=1e-12)
    np.testing.assert_allclose(r["z"][:2], 0.0, atol=1e-12)


@pytest.mark.parametrize("name", list(CASES))
def test_stm_matches_finite_differences_across_switches(compiled, name):
    _, _, k = compiled[name]
    p, z0, t1 = CASES[name]
    r = k.propagate(0.0, z0, t1, p, stm=True, param_sensitivity=True, **KW)
    assert r["status"] == 0
    # Perturbations must not change the switching structure.
    n_sw = len(r["switch_times"])

    def end(z):
        out = k.propagate(0.0, z, t1, p, **KW)
        assert len(out["switch_times"]) == n_sw
        return out["z"]

    def end_p(pp):
        return k.propagate(0.0, z0, t1, pp, **KW)["z"]

    # Saturation corners are not located (only bang-bang switches are), so
    # steps across them carry ~1e-9 errors: difference over a larger step.
    h, tol = (1e-4, 1e-4) if name == "min_energy_bounded" else (1e-6, 1e-6)
    np.testing.assert_allclose(r["stm"], fd_columns(end, z0, h), rtol=tol, atol=tol)
    np.testing.assert_allclose(r["dz_dp"], fd_columns(end_p, p, h), rtol=tol, atol=tol)
    # Carrying Psi changes the step sizes (it enters the error norm), not the STM.
    r2 = k.propagate(0.0, z0, t1, p, stm=True, **KW)
    np.testing.assert_allclose(r2["stm"], r["stm"], rtol=1e-9, atol=1e-10)
    # Sensitivity to the final time is the dynamics at the end.
    later = k.propagate(0.0, z0, t1 + 1e-6, p, **KW)["z"]
    earlier = k.propagate(0.0, z0, t1 - 1e-6, p, **KW)["z"]
    np.testing.assert_allclose(r["dz_dt1"], (later - earlier) / 2e-6, rtol=1e-6, atol=1e-6)


def test_jump_is_needed(compiled):
    """Integrating Phi' = A Phi across a switch without the jump gives the wrong STM."""
    _, _, k = compiled["min_time"]
    p, z0, t1 = CASES["min_time"]
    r = k.propagate(0.0, z0, t1, p, stm=True, **KW)
    (ts,) = r["switch_times"]
    eye = np.eye(4).reshape(-1)
    # Fixed arcs, no events: plain variational equations on each side.
    before = solve_ivp(k.ArcDynamics(p, [0], stm=True), (0.0, ts), np.concatenate([z0, eye]),
                       method="DOP853", **KW).y[:, -1]
    after = solve_ivp(k.ArcDynamics(p, [1], stm=True), (ts, t1), np.concatenate([before[:4], eye]),
                      method="DOP853", **KW).y[4:, -1].reshape(4, 4)
    without_jump = after @ before[4:].reshape(4, 4)
    assert not np.allclose(without_jump, r["stm"], atol=1e-3)
    jumped = k.stm_jump(0, ts, before, p, [0])[4:].reshape(4, 4)
    np.testing.assert_allclose(after @ jumped, r["stm"], rtol=1e-10, atol=1e-12)


def test_low_thrust_switches_match_dense_trajectory(compiled):
    _, num, k = compiled["low_thrust"]
    p, z0, t1 = CASES["low_thrust"]
    r = k.propagate(0.0, z0, t1, p, cost=True, **KW)
    np.testing.assert_allclose(r["switch_times"], [1.50833477, 4.66741994, 5.55846603], rtol=1e-7)
    tr = k.trajectory(0.0, z0, t1, p, n_points=301, **KW)
    np.testing.assert_allclose(tr["switch_times"], r["switch_times"], rtol=1e-10)
    np.testing.assert_allclose(tr["z"][-1], r["z"], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(tr["H"], tr["H"][0], atol=1e-10)
    # Coast, burn, coast, burn. Propellant used = T_max/c * burn time = mass lost.
    np.testing.assert_array_equal(r["arc0"], [0])
    t_s = r["switch_times"]
    burn = (t_s[1] - t_s[0]) + (t1 - t_s[2])
    assert r["cost"] == pytest.approx(0.05 * burn, rel=1e-9)
    assert z0[4] - r["z"][4] == pytest.approx(r["cost"], rel=1e-9)
    on = tr["u"][:, 2] == 1.0
    np.testing.assert_array_equal(on, ((tr["t"] > t_s[0]) & (tr["t"] < t_s[1])) | (tr["t"] > t_s[2]))


def test_batch_equals_single_and_threads_agree(compiled):
    _, _, k = compiled["low_thrust"]
    p, z0, t1 = CASES["low_thrust"]
    rng = np.random.default_rng(5)
    Z0 = np.tile(z0, (16, 1))
    Z0[:, 5:9] += 0.05 * rng.normal(size=(16, 4))
    serial = k.propagate_batch(0.0, Z0, t1, p, stm=True, cost=True, **KW)
    threaded = k.propagate_batch(0.0, Z0, t1, p, stm=True, cost=True, n_threads=4, **KW)
    for key in ("z", "stm", "cost", "n_switches", "nfev", "status", "arc"):
        np.testing.assert_array_equal(serial[key], threaded[key])
    for i in (0, 7, 15):
        one = k.propagate(0.0, Z0[i], t1, p, stm=True, cost=True, **KW)
        np.testing.assert_array_equal(serial["z"][i], one["z"])
        np.testing.assert_array_equal(serial["stm"][i], one["stm"])
        assert serial["cost"][i] == one["cost"]
        n = serial["n_switches"][i]
        np.testing.assert_array_equal(serial["switch_times"][i, :n], one["switch_times"])
        assert np.isnan(serial["switch_times"][i, n:]).all()
    # Per-row time spans.
    t1s = np.linspace(1.0, 6.0, 16)
    var = k.propagate_batch(0.0, Z0, t1s, p, **KW)
    np.testing.assert_array_equal(var["t"], t1s)


def test_sampling_costates_from_the_fixed_initial_state(compiled):
    """The forward map lam0 -> extremal, as a sampler would use it."""
    _, _, k = compiled["min_time"]
    p = [1.0, 1.0, 0.0]
    np.testing.assert_array_equal(k.initial_state(p), [1.0, 0.0])
    lam = np.random.default_rng(6).normal(size=(200, 2))
    Z0 = np.array([k.z0(l, p) for l in lam])
    out = k.propagate_batch(0.0, Z0, 2.0, p, cost=True, n_threads=2, **KW)
    assert (out["status"] == 0).all()
    assert out["n_switches"].max() <= 1                    # the double integrator switches at most once
    np.testing.assert_allclose(out["cost"], 2.0, rtol=1e-12)
    best = np.argmin(np.linalg.norm(out["z"][:, :2], axis=1))
    assert np.linalg.norm(out["z"][best, :2]) < 1.0


def test_gil_is_released(compiled):
    _, _, k = compiled["low_thrust"]
    p, z0, t1 = CASES["low_thrust"]
    Z0 = np.tile(z0, (64, 1))
    results = []

    def work():
        results.append(k.propagate_batch(0.0, Z0, t1, p, stm=True, **KW)["z"])

    threads = [threading.Thread(target=work) for _ in range(3)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for z in results[1:]:
        np.testing.assert_array_equal(z, results[0])


def test_saturation_kinks_restart_the_step(compiled):
    """u = clip(-lam_v, -1, 1) with lam_v = 1.5 - 2t: saturated at -1, a ramp from
    t = 0.25 to 1.25, saturated at 1. The integration restarts at both kinks
    (they do not count as switches) and matches the exact piecewise
    polynomial to the integration tolerance."""
    k = compiled["min_energy_bounded"][2]
    p = [2.0, 0.0, 0.0, 1.0]                       # T, a, b, u_max
    z0 = [0.0, 0.0, 2.0, 1.5]
    assert k.N_KINKS == 2
    r = k.propagate(0.0, z0, 2.0, p, cost=True, max_switches=0, **KW)
    assert r["status"] == 0 and len(r["switch_times"]) == 0

    # Exact: integrate u piece by piece.
    def pieces():
        yield 0.0, 0.25, np.poly1d([-1.0])
        yield 0.25, 1.25, np.poly1d([2.0, -1.5])
        yield 1.25, 2.0, np.poly1d([1.0])

    x = v = J = 0.0
    for a, b, u in pieces():
        V = u.integ()
        X = (V - V(a) + v).integ()
        x, v = x + X(b) - X(a), v + V(b) - V(a)
        J += (u * u / 2).integ()(b) - (u * u / 2).integ()(a)
    np.testing.assert_allclose(r["z"][:2], [x, v], rtol=0, atol=1e-11)
    assert r["cost"] == pytest.approx(J, abs=1e-11)
    tr = k.trajectory(0.0, z0, 2.0, p, t_eval=[2.0], **KW)
    np.testing.assert_allclose(tr["z"][0, :2], [x, v], atol=1e-11)
    assert len(tr["switch_times"]) == 0
