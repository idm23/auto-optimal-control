"""Path constraints: order detection, penalties, crossings, termination.

The Bryson-Denham problem (``examples.bryson_denham``) has closed-form
solutions in both regimes of its order-2 limit ``x <= l``: a touch at
``t = 1/2`` for ``1/6 <= l < 1/4``, a boundary arc for ``l < 1/6``. Its
exterior-penalty solutions approach them from below as the weight ``rho``
grows.
"""

import warnings

import numpy as np
import pytest
import sympy as sp

from auto_optimal_control.pontryagin import constraint_derivatives, derive
from auto_optimal_control.problem import OptimalControlProblem, PathConstraint
from auto_optimal_control.shooting import (ShootingError, SingleShooting,
                                           parameter_continuation)

# An extremal near rho = 2e4 at l = 0.2 (touch regime), found by continuation.
# With the penalty off the flow overshoots, so g has a peak and a trough
# (dg/dt = v has two zeros) inside a single long step of the STM run. Before
# interior detection that run missed the violation the residual run found.
TOUCH_P = np.array([0.2, 19952.62314968879])
TOUCH_Z0 = np.array([0.0, 1.0, 4.499634128106173, 3.1182084032267263])


def touch_cost(ell):
    """Exact cost for 1/6 <= l <= 1/4: the cubic from (0, 1) to (l, 0) at t = 1/2, mirrored."""
    a, b = -4 + 12 * ell, 4 - 16 * ell        # x = t + a t^2 + b t^3 on [0, 1/2]
    c0, c1, T = 2 * a, 6 * b, 0.5              # u = c0 + c1 t
    return 2 * 0.5 * (c0 ** 2 * T + c0 * c1 * T ** 2 + c1 ** 2 * T ** 3 / 3)


@pytest.fixture(scope="module")
def bd(compiled):
    return compiled["bryson_denham"][2]


@pytest.fixture(scope="module")
def bd_start(bd):
    """The rho = 1 solution at l = 0.2."""
    return SingleShooting(bd, [0.2, 1.0]).solve([0, 2.0, 0, 0])


# --- order ---------------------------------------------------------------------------------------

def jerk_chain(*constraints):
    t, p, v, a, j = sp.symbols("t p v a j", real=True)
    exprs = [c(p, v, a) for c in constraints]
    return OptimalControlProblem(
        name="jerk", time=t, states=[p, v, a], controls=[j], dynamics=[v, a, j],
        running_cost=j ** 2 / 2, initial_state=[0, 0, 0], final_state={p: 1, v: 0, a: 0},
        final_time=5,
        path_constraints=exprs)


def test_order_is_the_number_of_derivatives_until_a_control_appears():
    prob = jerk_chain(lambda p, v, a: a - 1, lambda p, v, a: v - 2, lambda p, v, a: p - 3)
    orders = [len(constraint_derivatives(prob, c.expr)) - 1 for c in prob.path_constraints]
    assert orders == [1, 2, 3]
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # order 3 is supported now: no warning
        pmp = derive(prob)
    assert [c.order for c in pmp.path_constraints] == [1, 2, 3]
    assert [c.name for c in pmp.path_constraints] == ["g0", "g1", "g2"]


def test_planet_avoidance_is_order_two():
    t, x, y, vx, vy, ux, uy = sp.symbols("t x y vx vy ux uy", real=True)
    R = sp.Symbol("R", positive=True)
    prob = OptimalControlProblem(
        name="flyby", time=t, states=[x, y, vx, vy], controls=[ux, uy], params=[R],
        dynamics=[vx, vy, ux, uy], running_cost=(ux ** 2 + uy ** 2) / 2,
        initial_state=[-3, 0.5, 1, 0], final_state={x: 3, y: 0.5, vx: 1, vy: 0}, final_time=6,
        path_constraints=[PathConstraint(R ** 2 - x ** 2 - y ** 2, "surface")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pmp = derive(prob)
    assert pmp.path_constraints[0].order == 2


def test_bryson_denham_is_order_two(compiled):
    c = compiled["bryson_denham"][0].path_constraints[0]
    assert (c.name, c.order) == ("x_max", 2)


def test_constraint_no_control_can_reach_raises():
    t, x, y, u = sp.symbols("t x y u", real=True)
    prob = OptimalControlProblem(
        name="drift", time=t, states=[x, y], controls=[u], dynamics=[1, u],
        running_cost=u ** 2, initial_state=[0, 0], final_time=1, path_constraints=[x - 2])
    with pytest.raises(ValueError, match="control"):
        derive(prob)


def test_validation():
    t, x, u = sp.symbols("t x u", real=True)
    k = sp.Symbol("k", positive=True)
    base = dict(name="v", time=t, states=[x], controls=[u], dynamics=[u], running_cost=u ** 2,
                initial_state=[0], final_time=1, params=[k])
    with pytest.raises(ValueError, match="unique"):
        OptimalControlProblem(**base, path_constraints=[PathConstraint(x - 1, "a"),
                                                         PathConstraint(-x - 1, "a")])
    with pytest.raises(ValueError, match="does not depend on the states"):
        OptimalControlProblem(**base, path_constraints=[t - k])
    with pytest.raises(ValueError, match="penalty"):
        OptimalControlProblem(**base, path_constraints=[PathConstraint(x - 1, penalty=x)])


# --- penalty and crossings ---------------------------------------------------------------------

def test_monitor_constraint_leaves_the_dynamics_alone(compiled):
    """Without a penalty, a constraint is only watched: crossings are reported
    and the extremal is the unconstrained one (here, the l = 0.2 overshoot)."""
    t, x, v, u = sp.symbols("t x v u", real=True)
    prob = OptimalControlProblem(
        name="bd_monitor", time=t, states=[x, v], controls=[u], dynamics=[v, u],
        running_cost=u ** 2 / 2, initial_state=[0, 1], final_state={x: 0, v: -1},
        final_time=1, path_constraints=[PathConstraint(x - sp.Rational(1, 5), "x_max")])
    from auto_optimal_control.build import build_kernels
    k = build_kernels(derive(prob), cache_dir=None)
    sol = SingleShooting(k).solve([0, 2.0, 0, 0])     # x = t - t^2, u = -2
    np.testing.assert_allclose(sol.lam0, [0.0, 2.0], atol=1e-9)
    r = k.propagate(0, sol.z0, 1.0, cost=True)
    assert r["cost"] == pytest.approx(2.0, rel=1e-12)
    roots = np.roots([1, -1, 0.2])                # t - t^2 = 0.2
    np.testing.assert_allclose(r["constraint_times"], np.sort(roots), rtol=1e-9)
    assert sol.max_violation == pytest.approx(0.05, rel=1e-6)


def test_every_run_sees_the_same_crossings(bd):
    """The residual, cost, STM and fine-step runs of one extremal agree on the
    crossings, so Newton's residual and Jacobian describe the same dynamics."""
    ref = bd.propagate(0, TOUCH_Z0, 1.0, TOUCH_P, max_step=1e-3)["constraint_times"]
    assert ref.size == 2
    for kw in (dict(), dict(cost=True), dict(stm=True), dict(stm=True, cost=True),
               dict(stm=True, param_sensitivity=True)):
        r = bd.propagate(0, TOUCH_Z0, 1.0, TOUCH_P, **kw)
        np.testing.assert_allclose(r["constraint_times"], ref, rtol=1e-9, err_msg=str(kw))


def test_stm_matches_finite_differences_across_crossings(bd):
    r = bd.propagate(0, TOUCH_Z0, 1.0, TOUCH_P, stm=True)
    h = 1e-7
    fd = np.empty((4, 4))
    for j in range(4):
        dz = np.zeros(4)
        dz[j] = h
        fd[:, j] = (bd.propagate(0, TOUCH_Z0 + dz, 1.0, TOUCH_P)["z"]
                    - bd.propagate(0, TOUCH_Z0 - dz, 1.0, TOUCH_P)["z"]) / (2 * h)
    np.testing.assert_allclose(r["stm"], fd, rtol=1e-5, atol=1e-6 * np.abs(fd).max())


def test_endpoint_only_detection_misses_the_overshoot(bd):
    """Why interior detection is on by default with path constraints: the STM
    run's long step holds both zeros of dg/dt, so with only endpoint tests
    neither the peak guard nor the constraint fires."""
    r = bd.propagate(0, TOUCH_Z0, 1.0, TOUCH_P, stm=True, n_interior=0)
    assert r["constraint_times"].size == 0


def test_touch_point_continuation_converges(bd, bd_start):
    """l = 0.2: the penalty solution tends to the touch at t = 1/2, cost 2.24.
    This continuation used to stall at rho ~ 2.4e4."""
    path = parameter_continuation(bd, [0.2, 1.0], "rho", np.logspace(0, 7, 29), bd_start.v)
    rho, sol = path[-1]
    assert rho == pytest.approx(1e7)
    r = bd.propagate(0, sol.z0, 1.0, [0.2, rho], cost=True)
    exact = touch_cost(0.2)
    assert exact == pytest.approx(2.24)
    assert exact - 1e-3 < r["cost"] < exact
    np.testing.assert_allclose(r["constraint_times"], [0.5, 0.5], atol=0.02)
    assert sol.max_violation < 1e-4


def test_boundary_arc_continuation_approaches_closed_form(bd):
    """l = 0.1: the limit is ridden on [0.3, 0.7] with cost 4 / (9 l)."""
    ell = 0.1
    v = SingleShooting(bd, [ell, 1.0]).solve([0, 2.0, 0, 0]).v
    path = parameter_continuation(bd, [ell, 1.0], "rho", np.logspace(0, 5, 21), v,
                                  solve_options=dict(tol=1e-8))
    rho, sol = path[-1]
    r = bd.propagate(0, sol.z0, 1.0, [ell, rho], cost=True)
    exact = 4 / (9 * ell)
    assert exact * 0.99 < r["cost"] < exact
    c = r["constraint_times"]
    assert c.size >= 2 and c[0] < 0.3 and c[-1] > 0.7     # the soft limit spreads the arc


def test_trajectory_reports_g_and_crossings(bd):
    tr = bd.trajectory(0, TOUCH_Z0, 1.0, TOUCH_P, n_points=2001)
    r = bd.propagate(0, TOUCH_Z0, 1.0, TOUCH_P)
    np.testing.assert_allclose(tr["constraint_times"], r["constraint_times"], rtol=1e-9)
    np.testing.assert_allclose(tr["g"][:, 0], tr["z"][:, 0] - 0.2, atol=1e-15)
    inside = (tr["t"] > r["constraint_times"][0]) & (tr["t"] < r["constraint_times"][1])
    assert np.all(tr["g"][inside, 0] > 0) and np.all(tr["g"][~inside, 0] <= 1e-12)


# --- termination -------------------------------------------------------------------------------

def test_terminate_stops_at_the_first_violation(bd, bd_start):
    z0 = bd_start.z0
    p = [0.2, 1.0]
    full = bd.propagate(0, z0, 1.0, p)
    entry = full["constraint_times"][0]
    for terminate in (True, 0, "x_max", ["x_max"]):
        r = bd.propagate(0, z0, 1.0, p, terminate=terminate)
        assert r["status"] == bd.module.STATUS_VIOLATED
        assert r["t"] == pytest.approx(entry, rel=1e-12)
        np.testing.assert_allclose(r["constraint_times"], [entry])
        assert r["z"][0] == pytest.approx(0.2, abs=1e-12)
    tr = bd.trajectory(0, z0, 1.0, p, terminate=True)
    assert tr["status"] == bd.module.STATUS_VIOLATED
    assert tr["t"][-1] == pytest.approx(entry, rel=1e-9)
    batch = bd.propagate_batch(0, np.tile(z0, (3, 1)), 1.0, p, terminate=True)
    np.testing.assert_array_equal(batch["status"], bd.module.STATUS_VIOLATED)
    np.testing.assert_allclose(batch["t"], entry, rtol=1e-12)


def test_terminate_on_an_already_violated_start(bd):
    z0 = np.array([0.3, 1.0, 0.0, 0.0])          # x = 0.3 > l = 0.2
    r = bd.propagate(0, z0, 1.0, [0.2, 1.0], terminate=True)
    assert r["status"] == bd.module.STATUS_VIOLATED and r["t"] == 0.0
    np.testing.assert_array_equal(r["z"], z0)


def test_shooting_with_terminate_rejects_violating_guesses(bd, bd_start):
    shoot = SingleShooting(bd, [0.2, 1.0], terminate=True)
    with pytest.raises(ShootingError):
        shoot.residual(bd_start.v)
    with pytest.raises(ValueError, match="out of range"):
        bd.propagate(0, np.zeros(4), 1.0, [0.2, 1.0], terminate=3)
