"""Exact boundary arcs and touch points: MultiArcShooting against closed forms.

- Bryson-Denham (order 2), ``l = 0.1``: the limit ``x = l`` is ridden on
  ``[3l, 1 - 3l]``, ``lam0 = (2/(9 l^2), 2/(3 l))``, cost ``4/(9 l)``.
- Bryson-Denham, ``l = 0.2``: a touch at ``t = 1/2``, ``lam0 = (4.8, 3.2)``,
  jump ``nu = 9.6``, cost 2.24.
- Velocity-limited move (order 1), ``V = 1.2``: the limit ``v = V`` is ridden
  on ``[1/4, 3/4]``, cost ``4 V^2 / (3 t1) = 7.68``.
- The same move with the force clipped to ``|u| <= 8``: full force first,
  then a ramp onto the limit (``velocity_limited_move_bounded_exact``).
- Minimum time over ``D = 3`` with ``|u| <= 1`` and ``v <= 1``: bang-bang,
  full power to ``t = 1``, coast to ``t = 3``, brake to ``t_f = 4``.

The smooth ones start from an exterior-penalty solution via
``MultiArcShooting.from_penalty``. The bang-bang one starts from its smoothed
variant (``1 + eps u^2/2``), solved with exact arcs and ``eps`` lowered.
"""

import numpy as np
import pytest
import sympy as sp

from auto_optimal_control.examples import velocity_limited_move_bounded_exact
from auto_optimal_control.pontryagin import derive
from auto_optimal_control.problem import OptimalControlProblem, PathConstraint, UnitVector
from auto_optimal_control.shooting import (MultiArcShooting, ShootingError, SingleShooting,
                                           continuation, parameter_continuation)


def penalty_solution(k, first, rho_max, guess=(0, 2.0, 0, 0)):
    v0 = SingleShooting(k, [first, 1.0]).solve(list(guess)).v
    path = parameter_continuation(k, [first, 1.0], "rho", np.logspace(0, np.log10(rho_max), 13),
                                  v0, solve_options=dict(tol=1e-9))
    return path[-1]


def exact(k, first, arcs, rho_max, guess=(0, 2.0, 0, 0)):
    rho, pen = penalty_solution(k, first, rho_max, guess)
    shoot, v = MultiArcShooting.from_penalty(k, [first, 0.0], arcs, pen, [first, rho])
    return shoot, v, shoot.solve(v)


def fd_jacobian(f, v, h=1e-7):
    return np.array([(f(v + h * e) - f(v - h * e)) / (2 * h) for e in np.eye(len(v))]).T


@pytest.fixture(scope="module")
def bd(compiled):
    return compiled["bryson_denham"][2]


@pytest.fixture(scope="module")
def vel(compiled):
    return compiled["velocity_limited_move"][2]


@pytest.fixture(scope="module")
def bd_boundary(bd):
    return exact(bd, 0.1, [("boundary", "x_max")], 1e4)


@pytest.fixture(scope="module")
def bd_touch(bd):
    return exact(bd, 0.2, [("touch", "x_max")], 1e4)


@pytest.fixture(scope="module")
def vel_boundary(vel):
    return exact(vel, 1.2, [("boundary", "v_max")], 1e2, guess=(12.0, 6.0, 0, 0))


@pytest.fixture(scope="module")
def vel_bounded(compiled):
    k = compiled["velocity_limited_move_bounded"][2]
    v0 = SingleShooting(k, [1.2, 1.0, 8.0]).solve([12.0, 6.0, 0, 0]).v
    rho, pen = parameter_continuation(k, [1.2, 1.0, 8.0], "rho", np.logspace(0, 2, 9), v0)[-1]
    shoot, v = MultiArcShooting.from_penalty(k, [1.2, 0.0, 8.0], [("boundary", "v_max")], pen,
                                             [1.2, rho, 8.0])
    return shoot, v, shoot.solve(v)


TRAIN = [1.0, 3.0, 1.0]          # V, D, u_max: t_f = D/V + V/u_max = 4


@pytest.fixture(scope="module")
def train_smoothed(compiled):
    """The smoothed run with exact arcs, from eps = 1 down to 1e-4."""
    k = compiled["speed_limited_min_time_smoothed"][2]
    single = SingleShooting(k, TRAIN + [1.0, 1.0])
    v0 = single.solve(single.pack([-0.85, -1.5], tf=3.8)).v
    rho, pen = parameter_continuation(k, TRAIN + [1.0, 1.0], "rho", np.logspace(0, 1.5, 7), v0,
                                      solve_options=dict(tol=1e-9))[-1]
    arcs = [("boundary", "v_max")]
    shoot, v = MultiArcShooting.from_penalty(k, TRAIN + [1.0, 0.0], arcs, pen, TRAIN + [1.0, rho])
    v = shoot.solve(v).v

    def solve_at(eps, guess):
        return MultiArcShooting(k, TRAIN + [eps, 0.0], arcs).solve(guess)

    return continuation(solve_at, 1.0, np.logspace(0, -4, 17), v)


@pytest.fixture(scope="module")
def train(compiled, train_smoothed):
    shoot = MultiArcShooting(compiled["speed_limited_min_time"][2], TRAIN, [("boundary", "v_max")])
    v = train_smoothed[-1][1].v
    return shoot, v, shoot.solve(v)


# --- derivation and kernels --------------------------------------------------------------------

def test_boundary_arc_derivation(compiled):
    arc = compiled["bryson_denham"][0].path_constraints[0].boundary
    u = compiled["bryson_denham"][0].controls[0]
    lam_x, lam_v = compiled["bryson_denham"][0].costates
    assert arc.control == {u: 0}
    assert arc.multiplier == -lam_v
    assert arc.dynamics[2:] == (0, -lam_x)
    sc = compiled["velocity_limited_move"][0].path_constraints[0]
    assert sc.order == 1 and len(sc.tangency) == 1


def test_bounded_controls_get_exact_arcs(compiled):
    arc = compiled["velocity_limited_move_bounded"][0].path_constraints[0].boundary
    u = compiled["velocity_limited_move_bounded"][0].controls[0]
    umax = compiled["velocity_limited_move_bounded"][0].params[2]
    assert arc.control == {u: 0} and arc.bounds == {u: (-umax, umax)}
    pmp = compiled["speed_limited_min_time"][0]
    assert len(pmp.switching_functions) == 1         # bang-bang, and still an exact arc
    assert pmp.path_constraints[0].boundary.control == {u: 0}


def test_saturation_guards(compiled):
    """A clipped control gets one guard per bound, zero where it saturates."""
    k = compiled["velocity_limited_move_bounded"][2]
    assert k.N_KINKS == 2 and compiled["velocity_limited_move"][2].N_KINKS == 0
    z, p = np.array([0.0, 0.0, -3.0, -8.0]), [1.2, 0.0, 8.0]     # u_free = -lam_v = 8
    np.testing.assert_allclose(sorted(np.abs(k.kinks(0.0, z, p))), [0.0, 16.0], atol=1e-14)
    lo, hi = k.control_bounds(0.0, z, p)
    assert lo[0] == -8.0 and hi[0] == 8.0
    lo, hi = compiled["velocity_limited_move"][2].control_bounds(0.0, z, [1.2, 0.0])
    assert lo[0] == -np.inf and hi[0] == np.inf


def test_unsupported_constraints_say_why():
    t, p, v, a, j, s = sp.symbols("t p v a j s", real=True)
    # A position limit with jerk control has order 3: supported (touched, not ridden).
    prob = OptimalControlProblem(
        name="jerk", time=t, states=[p, v, a], controls=[j], dynamics=[v, a, j],
        running_cost=j ** 2 / 2, initial_state=[0, 0, 0], final_state={p: 1, v: 0, a: 0},
        final_time=5, path_constraints=[PathConstraint(p - 3, "p_max")])
    sc = derive(prob).path_constraints[0]
    assert sc.order == 3 and sc.boundary is not None
    # With snap control it has order 4: not supported, and it says so.
    prob = OptimalControlProblem(
        name="snap", time=t, states=[p, v, a, j], controls=[s], dynamics=[v, a, j, s],
        running_cost=s ** 2 / 2, initial_state=[0, 0, 0, 0],
        final_state={p: 1, v: 0, a: 0, j: 0}, final_time=5,
        path_constraints=[PathConstraint(p - 3, "p_max")])
    with pytest.warns(UserWarning, match="order 4"):
        sc = derive(prob).path_constraints[0]
    assert sc.boundary is None and "order 4" in sc.unsupported
    ux, uy = sp.symbols("u_x u_y", real=True)
    x, y = sp.symbols("x y", real=True)
    prob = OptimalControlProblem(
        name="steer", time=t, states=[x, y], controls=[ux, uy], dynamics=[ux, uy],
        running_cost=1, initial_state=[0, 0], final_state={x: 1, y: 1},
        control_sets=[UnitVector([ux, uy])], path_constraints=[PathConstraint(y - 0.5, "y_max")])
    assert "unit-vector" in derive(prob).path_constraints[0].unsupported
    prob = OptimalControlProblem(
        name="rising", time=t, states=[p, v], controls=[a], dynamics=[v, a],
        running_cost=a ** 2 / 2, initial_state=[0, 0], final_state={p: 1, v: 0}, final_time=1,
        path_constraints=[PathConstraint(v - 1 - t / 4, "v_max")])
    assert "time-invariant" in derive(prob).path_constraints[0].unsupported


@pytest.mark.parametrize("name, point", [
    ("bryson_denham", ([0.13, -0.4, 3.0, 1.5], [0.1, 0.0])),
    ("velocity_limited_move", ([0.4, 1.1, -5.0, 2.0], [1.2, 0.0])),
])
def test_exact_arc_kernels_match_finite_differences(compiled, name, point):
    k = compiled[name][2]
    z, p = np.array(point[0]), point[1]
    t, n = 0.37, k.N_STATES
    np.testing.assert_allclose(k.boundary_jacobian(0, t, z, p),
                               fd_jacobian(lambda w: k.boundary_rhs(0, t, w, p), z), atol=1e-7)
    for b in (None, 0):
        Ht, Hz = k.hamiltonian_gradient(t, z, p, boundary=b)
        H = ((lambda s, w: k.hamiltonian(s, w, p)) if b is None
             else (lambda s, w: k.boundary_hamiltonian(0, s, w, p)))
        np.testing.assert_allclose(Hz, fd_jacobian(lambda w: np.atleast_1d(H(t, w)), z)[0],
                                   atol=1e-6)
        assert Ht == pytest.approx((H(t + 1e-7, z) - H(t - 1e-7, z)) / 2e-7, abs=1e-6)
    Nt, Nx, Nxx, _ = k.tangency_gradient(0, t, z, p)
    N = lambda w: k.tangency(0, t, np.r_[w, z[n:]], p)          # noqa: E731
    np.testing.assert_allclose(Nx, fd_jacobian(N, z[:n]), atol=1e-7)
    for i in range(len(Nx)):
        np.testing.assert_allclose(
            Nxx[i], fd_jacobian(lambda w: k.tangency_gradient(0, t, np.r_[w, z[n:]], p)[1][i],
                                z[:n]), atol=1e-6)


def test_boundary_arc_holds_the_limit(bd):
    """Along the boundary dynamics from N = 0, g stays at zero."""
    z = np.array([0.1, 0.0, -22.2, -8.9])
    r = bd.propagate_boundary(0, 0.3, z, 0.7, [0.1, 0.0])
    assert r["status"] == 0
    np.testing.assert_allclose(bd.tangency(0, 0.7, r["z"], [0.1, 0.0]), 0.0, atol=1e-12)


# --- closed forms --------------------------------------------------------------------------------

def test_bryson_denham_boundary_arc_is_exact(bd_boundary):
    shoot, _, sol = bd_boundary
    ell = 0.1
    assert sol.success
    np.testing.assert_allclose(sol.lam0, [2 / (9 * ell ** 2), 2 / (3 * ell)], rtol=1e-8)
    (t_in, t_out, pi), = shoot.unpack_arcs(sol.v)
    assert t_in == pytest.approx(3 * ell, abs=1e-6)
    assert t_out == pytest.approx(1 - 3 * ell, abs=1e-6)
    np.testing.assert_allclose(pi, [4 / (9 * ell ** 2), 8 / (9 * ell)], rtol=1e-6)
    assert shoot.cost(sol.v) == pytest.approx(4 / (9 * ell), rel=1e-10)
    d = shoot.diagnostics(sol.v)
    assert d["free_arc_violation"][0] < 1e-12
    mu_lo, mu_hi = d["boundary_mu"][0]
    assert mu_lo > -1e-6                     # mu >= 0 along the arc
    np.testing.assert_allclose(d["boundary_u"][0], 0.0, atol=1e-12)


def test_bryson_denham_touch_point_is_exact(bd_touch):
    shoot, _, sol = bd_touch
    assert sol.success
    np.testing.assert_allclose(sol.lam0, [4.8, 3.2], rtol=1e-10)
    (tc, nu), = shoot.unpack_arcs(sol.v)
    assert tc == pytest.approx(0.5, abs=1e-10)
    assert nu == pytest.approx(9.6, rel=1e-10)
    assert shoot.cost(sol.v) == pytest.approx(2.24, rel=1e-12)
    d = shoot.diagnostics(sol.v)
    assert d["touch_nu"][0] > 0 and d["free_arc_violation"][0] < 1e-12


def test_order_one_boundary_arc_is_exact(vel_boundary):
    shoot, _, sol = vel_boundary
    V = 1.2
    t1 = 3 * (V - 1) / (2 * V)
    assert sol.success
    (t_in, t_out, _), = shoot.unpack_arcs(sol.v)
    # dg/dt = 0 at the entry and mu = 0 at the exit have simple roots, so the
    # junction times are exact (H continuity alone gave them to about 1e-8).
    assert t_in == pytest.approx(t1, abs=1e-12) and t_out == pytest.approx(1 - t1, abs=1e-12)
    assert shoot.cost(sol.v) == pytest.approx(4 * V ** 2 / (3 * t1), rel=1e-10)
    tr = shoot.trajectory(sol.v)
    on = tr["arc"] == 0
    np.testing.assert_allclose(tr["z"][on, 1], V, atol=1e-10)
    assert tr["g"][~on].max() < 1e-10


def test_bounded_order_one_arc_is_exact(vel_bounded):
    """Full force for s, a ramp to u = 0 at t1, the limit, then the mirror image."""
    shoot, _, sol = vel_bounded
    J, t1, s = velocity_limited_move_bounded_exact(1.2, 8.0)
    assert sol.success
    (t_in, t_out, _), = shoot.unpack_arcs(sol.v)
    assert t_in == pytest.approx(t1, abs=1e-8) and t_out == pytest.approx(1 - t1, abs=1e-8)
    assert shoot.cost(sol.v) == pytest.approx(J, rel=1e-11)
    tr = shoot.trajectory(sol.v, n_points=2001)
    early = tr["t"] < s - 1e-3
    np.testing.assert_allclose(tr["u"][early, 0], 8.0, atol=1e-12)    # saturated
    assert np.abs(tr["u"][:, 0]).max() <= 8.0 + 1e-12
    d = shoot.diagnostics(sol.v)
    assert d["boundary_bound_violation"] == [0.0] and sol.max_violation < 1e-10


def test_bang_bang_boundary_arc_is_exact(train, train_smoothed):
    shoot, _, sol = train
    assert sol.success and sol.nfev < 30
    assert sol.tf == pytest.approx(4.0, abs=1e-12)
    (t_in, t_out, pi), = shoot.unpack_arcs(sol.v)
    assert t_in == pytest.approx(1.0, abs=1e-12) and t_out == pytest.approx(3.0, abs=1e-12)
    assert pi[0] == pytest.approx(2.0, abs=1e-10)
    assert shoot.cost(sol.v) == pytest.approx(4.0, abs=1e-12)        # min time: J = t_f
    tr = shoot.trajectory(sol.v, n_points=801)
    u, t = tr["u"][:, 0], tr["t"]
    np.testing.assert_array_equal(u[t < 1 - 1e-9], 1.0)
    np.testing.assert_array_equal(u[(t > 1 + 1e-9) & (t < 3 - 1e-9)], 0.0)
    np.testing.assert_array_equal(u[t > 3 + 1e-9], -1.0)
    d = shoot.diagnostics(sol.v)
    assert d["boundary_mu"][0][0] > -1e-12                            # mu = -lam_v >= 0
    # The smoothed runs approach it: t_f = 4 + O(eps).
    eps, smooth = train_smoothed[-1]
    assert eps == pytest.approx(1e-4) and smooth.tf - 4.0 == pytest.approx(0, abs=1e-8)


def test_bang_bang_solve_finishes_from_nearby_starts(train):
    """MINPACK can stop short of the root on the |S| kink at a junction;
    Newton polishing finishes it from every start."""
    shoot, _, sol = train
    rng = np.random.default_rng(0)
    for _ in range(40):
        s = shoot.solve(sol.v + rng.normal(0, 1e-4, sol.v.size))
        assert s.success and s.tf == pytest.approx(4.0, abs=1e-11)


def test_bang_bang_jacobian_matches_finite_differences(train):
    """Away from the solution (where switches sit on the junctions), with a
    switch inside each free arc."""
    shoot, v, _ = train
    for dw in ([0.05, -0.1, 0.03, 0.02, -0.03, 0.1, -0.1, 0.2],
               [0.0, 0.3, 0.0, 0.1, 0.1, 0, 0.1, -0.3]):
        w = v + np.array(dw)
        _, J = shoot.residual_jacobian(w)
        fd = fd_jacobian(shoot.residual, w)
        np.testing.assert_allclose(J, fd, atol=1e-6 * np.abs(fd).max())


@pytest.mark.parametrize("case", ["bd_boundary", "bd_touch", "vel_boundary", "vel_bounded"])
def test_jacobian_matches_finite_differences(request, case):
    shoot, v, sol = request.getfixturevalue(case)
    for w in (v, sol.v):                     # at the penalty guess and at the solution
        R, J = shoot.residual_jacobian(w)
        np.testing.assert_allclose(R, shoot.residual(w), atol=1e-11)   # STM vs plain steps
        # Restarts at the saturation kinks move with w, which adds noise near
        # the integration tolerance to R: a wider step keeps it out of the
        # difference quotient.
        fd = fd_jacobian(shoot.residual, w, h=1e-5 if case == "vel_bounded" else 1e-7)
        np.testing.assert_allclose(J, fd, atol=1e-5 * max(1.0, np.abs(fd).max()))


# --- wrong structures and validation ----------------------------------------------------------

def test_wrong_structure_shows_as_a_free_arc_violation(bd):
    """At l = 0.1 a touch-point extremal exists, but it crosses the limit
    before t = 1/2 (x peaks at about 0.113)."""
    shoot = MultiArcShooting(bd, [0.1, 0.0], [("touch", "x_max")])
    sol = shoot.solve(shoot.pack_arcs([4.0, 3.0, 0.0, 0.0], [(0.5, 5.0)]))
    assert sol.success
    assert sol.max_violation == pytest.approx(0.0134, abs=1e-3)


def test_validation(bd, compiled):
    with pytest.raises(ValueError, match="touch points"):
        MultiArcShooting(compiled["velocity_limited_move"][2], [1.2, 0.0], [("touch", 0)])
    with pytest.raises(ValueError, match="arc kind"):
        MultiArcShooting(bd, [0.1, 0.0], [("ride", 0)])
    shoot = MultiArcShooting(bd, [0.1, 0.0], [("boundary", "x_max")])
    with pytest.raises(ShootingError, match="out of order"):
        shoot.residual(shoot.pack_arcs([20, 6, 0, 0], [(0.7, 0.3, [0, 0])]))
    with pytest.raises(ValueError, match="pi must have 2"):
        shoot.pack_arcs([20, 6, 0, 0], [(0.3, 0.7, [0])])
