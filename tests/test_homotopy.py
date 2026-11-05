"""The automatic homotopy: regularization, structure reading and exact solves.

Every solve here is given only the problem, its parameters and a box (or a
function) to draw initial costates (and ``t_f``) from.
"""

import pytest
import sympy as sp

from auto_optimal_control import examples
from auto_optimal_control.homotopy import EPS, OFF, RHO, Homotopy, exact, regularize
from auto_optimal_control.problem import OptimalControlProblem, PathConstraint


def bryson_denham_cost(ell):
    if ell < 1 / 6:
        return 4 / (9 * ell)
    a, b = -4 + 12 * ell, 4 - 16 * ell
    c0, c1, T = 2 * a, 6 * b, 0.5
    return c0 ** 2 * T + c0 * c1 * T ** 2 + c1 ** 2 * T ** 3 / 3


def test_regularize_adds_smoothing_and_penalties():
    problem = examples.goddard(dynamic_pressure=True)
    reg, eps, rhos = regularize(problem)
    names = [str(p) for p in reg.params]
    assert names[:len(problem.params)] == [str(p) for p in problem.params]
    assert names[len(problem.params):] == [EPS, RHO + "q_max", OFF + "q_max"]
    u = problem.controls[0]
    # eps ((u - 1.75) / 1.75)^2 / 2 for 0 <= u <= u_max.
    umax = [p for p in problem.params if str(p) == "u_max"][0]
    extra = sp.simplify(reg.running_cost - problem.running_cost
                        - eps * ((u - umax / 2) / (umax / 2)) ** 2 / 2)
    assert extra == 0
    assert [c.penalty for c in reg.path_constraints] == list(rhos)
    # Each limit carries an offset: g - off <= 0.
    off = sp.Symbol(OFF + "q_max", real=True)
    assert sp.simplify(reg.path_constraints[0].expr - problem.path_constraints[0].expr + off) == 0


def test_regularize_without_bang_bang_controls():
    reg, eps, rhos = regularize(examples.bryson_denham())
    assert eps is None
    assert [str(r) for r in rhos] == [RHO + "x_max"]
    assert reg.running_cost == examples.bryson_denham().running_cost


def test_exact_drops_penalties():
    problem = exact(examples.bryson_denham())
    assert all(c.penalty is None for c in problem.path_constraints)


@pytest.mark.slow
def test_free_final_time_needs_a_distribution(kernel_cache):
    h = Homotopy(examples.speed_limited_min_time(), cache_dir=kernel_cache)
    with pytest.raises(ValueError, match="final_time"):
        h.solve([1.0, 3.0, 1.0], ([-3, -3], [0, 3]))


@pytest.mark.parametrize("ell, kind", [(0.2, "touch"), (0.1, "boundary")])
def test_bryson_denham_finds_its_structure(kernel_cache, ell, kind):
    h = Homotopy(examples.bryson_denham(), cache_dir=kernel_cache)
    r = h.solve([ell, 0.0], ([-10, -10], [10, 10]))
    assert r.exact, r.message
    assert r.arcs == ((kind, "x_max"),)
    assert abs(r.objective - bryson_denham_cost(ell)) < 1e-9


@pytest.mark.slow
def test_velocity_limit_with_force_bound(kernel_cache):
    V, umax = 1.2, 8.0
    h = Homotopy(examples.velocity_limited_move(bounded=True), cache_dir=kernel_cache)
    r = h.solve([V, 0.0, umax], ([-20, -20], [20, 20]))
    assert r.exact and r.arcs == (("boundary", "v_max"),)
    J, _, _ = examples.velocity_limited_move_bounded_exact(V, umax)
    assert abs(r.objective - J) < 1e-9


@pytest.mark.slow
@pytest.mark.parametrize("V, D, umax", [(1.0, 3.0, 1.0), (2.0, 10.0, 0.5)])
def test_speed_limit_bang_bang_with_boundary_arc(kernel_cache, V, D, umax):
    h = Homotopy(examples.speed_limited_min_time(), cache_dir=kernel_cache)
    r = h.solve([V, D, umax], ([-3, -3], [0, 3]), (1.0, 20.0))
    assert r.exact and r.arcs == (("boundary", "v_max"),), r.message
    assert abs(r.solution.tf - (D / V + V / umax)) < 1e-9


@pytest.mark.slow
def test_singular_regulator(kernel_cache):
    a = 1.0
    h = Homotopy(examples.singular_regulator(), cache_dir=kernel_cache)
    r = h.solve([4.0, a], ([-3], [3]))
    assert r.exact and r.arcs == (("singular", "u", "lower", "upper"),)
    assert abs(r.objective - 2 * a ** 3 / 3) < 1e-10


GODDARD_BOX = (([-6, -0.5, -0.1], [-1, 0, 0.1]), (0.15, 0.3))


@pytest.mark.slow
def test_goddard_finds_its_singular_arc(kernel_cache):
    h = Homotopy(examples.goddard(), cache_dir=kernel_cache)
    r = h.solve(examples.GODDARD_PARAMS, *GODDARD_BOX)
    assert r.exact and r.arcs == (("singular", "u", "upper", "lower"),)
    assert abs(-r.objective - 1.012836918) < 1e-9
    assert abs(r.solution.tf - 0.1988562621) < 1e-9


@pytest.mark.slow
def test_goddard_with_dynamic_pressure_limit(kernel_cache):
    h = Homotopy(examples.goddard(True), cache_dir=kernel_cache)
    r = h.solve(examples.GODDARD_Q_PARAMS, *GODDARD_BOX)
    assert r.exact, r.message
    assert r.arcs == (("boundary", "q_max"), ("singular", "u", "upper", "lower", "joined"))
    assert abs(-r.objective - 1.012717683) < 1e-9
    assert r.solution.max_violation < 1e-9


@pytest.mark.slow
def test_order_three_limit_is_reported_inexact(kernel_cache):
    """x''' = u with x <= l: order 3, so no exact structure; the penalty
    solution comes back, flagged."""
    t, x, v, a, u = sp.symbols("t x v a u", real=True)
    ell = sp.Symbol("l", positive=True)
    problem = OptimalControlProblem(
        name="triple_integrator_limit", time=t, states=[x, v, a], controls=[u], params=[ell],
        dynamics=[v, a, u], running_cost=u ** 2 / 2, initial_state=[0, 1, 0],
        final_state={x: 0, v: -1, a: 0}, final_time=1,
        path_constraints=[PathConstraint(x - ell, "x_max")])
    h = Homotopy(problem, cache_dir=kernel_cache)
    r = h.solve([0.1], ([-50, -50, -50], [50, 50, 50]), max_steps=40)
    assert not r.exact
    assert r.solution.success          # the last penalty solution
    assert "no exact structure" in r.message


@pytest.mark.parametrize("J, A, V, D, arcs", [
    (1.0, 1.0, 2.0, 10.0, ("a_max", "v_max", "a_min")),    # every limit reached
    pytest.param(1.0, 1.0, 2.0, 4.0, ("a_max", "a_min"), marks=pytest.mark.slow),   # no cruise
    pytest.param(1.0, 1.0, 0.5, 10.0, ("v_max",), marks=pytest.mark.slow),          # A never reached
    pytest.param(1.0, 1.0, 2.0, 1.0, (), marks=pytest.mark.slow),                   # bang-bang only
])
def test_jerk_limited_move_matches_the_s_curve(kernel_cache, J, A, V, D, arcs):
    """The S-curve of motion control, found by tightening the limits."""
    h = Homotopy(examples.jerk_limited_move(), cache_dir=kernel_cache)
    r = h.solve([J, A, V, D], ([-3, -3, -3], [3, 3, 3]), (0.5, 30.0))
    assert r.exact, r.message
    tf, _, _ = examples.jerk_limited_min_time(J, A, V, D)
    assert r.solution.tf == pytest.approx(tf, abs=1e-9)
    assert tuple(name for _, name in r.arcs) == arcs
    assert all(kind == "boundary" for kind, _ in r.arcs)
