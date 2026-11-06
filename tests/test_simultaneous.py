"""Several limits ridden at once: derivation, kernels, overlapping arcs and the automatic solve.

The two-link arm (``examples.two_link_arm``) has a torque limit on each joint,
and each involves both joint jerks.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from auto_optimal_control import examples
from auto_optimal_control.homotopy import Homotopy
from auto_optimal_control.pontryagin import derive
from auto_optimal_control.shooting import MultiArcShooting

P = list(examples.TWO_LINK_ARM_PARAMS)


@pytest.fixture(scope="module")
def arm(kernel_cache):
    return Homotopy(examples.two_link_arm(), cache_dir=kernel_cache)


def test_sets_pair_the_joints_and_skip_exclusive_bounds():
    pmp = derive(examples.two_link_arm())
    names = [c.name for c in pmp.path_constraints]
    pairs = {tuple(names[i] for i in idx) for idx, _ in pmp.boundary_sets}
    assert pairs == {("tau1_max", "tau2_max"), ("tau1_max", "tau2_min"),
                     ("tau1_min", "tau2_max"), ("tau1_min", "tau2_min")}
    # One control: two limits can never be ridden at once.
    assert derive(examples.jerk_limited_move()).boundary_sets == ()
    assert derive(examples.two_link_arm(), simultaneous=1).boundary_sets == ()


def test_set_control_rides_both_limits(arm):
    """At the set's control both torques are stationary: g1' = g2' = 0."""
    k = arm.kr
    pmp = k.system
    sid = k.boundary_set(["tau1_max", "tau2_max"])
    assert k.arc_constraints(sid) == (0, 2)
    p = np.concatenate([P, [17.0, 6.5], np.zeros(4), np.zeros(4)])   # + rho, offsets
    z = np.random.default_rng(0).normal(size=12)
    u = k.boundary_control(sid, 0.1, z, p)
    values = dict(zip(pmp.z, z)) | dict(zip(pmp.params, p)) | dict(zip(pmp.controls, u))
    for c in (0, 2):
        assert abs(float(pmp.path_constraints[c].derivatives[-1].subs(values))) < 1e-12
    mu = k.boundary_multiplier(sid, 0.1, z, p)
    assert mu.shape == (2,)
    Mt, Mz = k.boundary_multiplier_gradient(sid, 0.1, z, p)
    h = 1e-6
    for j in range(3):
        e = np.zeros(12)
        e[j] = h
        fd = (k.boundary_multiplier(sid, 0.1, z + e, p) - k.boundary_multiplier(sid, 0.1, z - e, p)) / (2 * h)
        np.testing.assert_allclose(Mz[:, j], fd, rtol=1e-6, atol=1e-6)


def test_structure_entries_are_checked(arm):
    k = arm.kr
    p = np.concatenate([P, [17.0, 6.5], np.zeros(4), np.zeros(4)])   # + rho, offsets
    with pytest.raises(ValueError, match="entered twice"):
        MultiArcShooting(k, p, [("enter", "tau1_max"), ("enter", "tau1_max")])
    with pytest.raises(ValueError, match="without being entered"):
        MultiArcShooting(k, p, [("exit", "tau1_max")])
    with pytest.raises(ValueError, match="never left"):
        MultiArcShooting(k, p, [("enter", "tau1_max")])


def _independent(r, T):
    """Simulate the arm's chains of integrators under the jerk, apart from the package."""
    tr = r.shooting.trajectory(r.solution.v, n_points=20001)
    t, u = tr["t"], tr["u"]
    keep = np.concatenate([[True], np.diff(t) > 0])
    jerk = interp1d(t[keep], u[keep], axis=0)
    sol = solve_ivp(lambda s, x: np.concatenate([x[2:6], jerk(min(s, t[-1]))]), (0, T),
                    np.zeros(6), rtol=1e-10, atol=1e-12, max_step=1e-3)
    tau1, tau2 = examples.two_link_arm_torques(*P[:4], *tr["z"][:, 1:6].T)
    J = np.trapezoid(0.5 * (u[keep] ** 2).sum(axis=1), t[keep])
    return sol.y[:, -1], np.abs(tau1).max(), np.abs(tau2).max(), J


@pytest.mark.parametrize("limits, events", [
    pytest.param((17.0, 6.5), ("enter tau1_max", "enter tau2_max", "exit tau1_max",
                               "exit tau2_max", "enter tau1_min", "exit tau1_min"), id="staggered"),
    pytest.param((15.0, 7.0), ("enter tau1_max", "enter tau2_max", "exit tau2_max",
                               "exit tau1_max", "enter tau1_min", "exit tau1_min"), id="nested",
                 marks=pytest.mark.slow),
])
def test_arm_rides_both_joints_limits(arm, limits, events):
    r = arm.solve(P + list(limits), ([-200] * 6, [200] * 6))
    assert r.exact, r.message
    assert tuple(f"{a} {b}" for a, b in r.arcs) == events
    xf, t1, t2, J = _independent(r, P[4])
    np.testing.assert_allclose(xf, [P[5], P[6], 0, 0, 0, 0], atol=1e-5)
    # Exactly at the limits, never past them.
    assert t1 == pytest.approx(limits[0], abs=1e-6) and t2 == pytest.approx(limits[1], abs=1e-6)
    assert J == pytest.approx(r.objective, rel=1e-5)
    d = r.shooting.diagnostics(r.solution.v)
    assert all(lo > -1e-6 * max(1, hi) for lo, hi in d["boundary_mu"])


def test_arm_without_active_limits_is_the_quintic(arm):
    """Loose limits: each joint follows the minimum-jerk quintic, int j^2 / 2 = 360 q_f^2 / T^5."""
    r = arm.solve(P + [100.0, 100.0], ([-200] * 6, [200] * 6))
    assert r.exact and r.arcs == ()
    assert r.objective == pytest.approx(360 * (P[5] ** 2 + P[6] ** 2) / P[4] ** 5, rel=1e-10)
