"""Exact boundary arcs and touch points: MultiArcShooting against closed forms.

- Bryson-Denham (order 2), ``l = 0.1``: the limit ``x = l`` is ridden on
  ``[3l, 1 - 3l]``, ``lam0 = (2/(9 l^2), 2/(3 l))``, cost ``4/(9 l)``.
- Bryson-Denham, ``l = 0.2``: a touch at ``t = 1/2``, ``lam0 = (4.8, 3.2)``,
  jump ``nu = 9.6``, cost 2.24.
- Velocity-limited move (order 1), ``V = 1.2``: the limit ``v = V`` is ridden
  on ``[1/4, 3/4]``, cost ``4 V^2 / (3 t1) = 7.68``.

Each starts from an exterior-penalty solution via ``MultiArcShooting.from_penalty``.
"""

import numpy as np
import pytest
import sympy as sp

from auto_optimal_control.pontryagin import derive
from auto_optimal_control.problem import OptimalControlProblem, PathConstraint
from auto_optimal_control.shooting import (MultiArcShooting, ShootingError, SingleShooting,
                                           parameter_continuation)


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


def test_unsupported_constraints_say_why():
    t, p, v, a, j = sp.symbols("t p v a j", real=True)
    prob = OptimalControlProblem(
        name="jerk", time=t, states=[p, v, a], controls=[j], dynamics=[v, a, j],
        running_cost=j ** 2 / 2, initial_state=[0, 0, 0], final_state={p: 1, v: 0, a: 0},
        final_time=5, path_constraints=[PathConstraint(p - 3, "p_max")])
    with pytest.warns(UserWarning):
        sc = derive(prob).path_constraints[0]
    assert sc.boundary is None and "order 3" in sc.unsupported


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
    assert t_in == pytest.approx(t1, abs=1e-8) and t_out == pytest.approx(1 - t1, abs=1e-8)
    assert shoot.cost(sol.v) == pytest.approx(4 * V ** 2 / (3 * t1), rel=1e-10)
    tr = shoot.trajectory(sol.v)
    on = tr["arc"] == 0
    np.testing.assert_allclose(tr["z"][on, 1], V, atol=1e-10)
    assert tr["g"][~on].max() < 1e-10


@pytest.mark.parametrize("case", ["bd_boundary", "bd_touch", "vel_boundary"])
def test_jacobian_matches_finite_differences(request, case):
    shoot, v, sol = request.getfixturevalue(case)
    for w in (v, sol.v):                     # at the penalty guess and at the solution
        R, J = shoot.residual_jacobian(w)
        np.testing.assert_allclose(R, shoot.residual(w), atol=1e-12)   # STM vs plain steps
        fd = fd_jacobian(shoot.residual, w)
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
    with pytest.raises(NotImplementedError, match="bang-bang"):
        MultiArcShooting(compiled["min_time"][2], [1.0, 1.0, 1.0], [])
    shoot = MultiArcShooting(bd, [0.1, 0.0], [("boundary", "x_max")])
    with pytest.raises(ShootingError, match="out of order"):
        shoot.residual(shoot.pack_arcs([20, 6, 0, 0], [(0.7, 0.3, [0, 0])]))
    with pytest.raises(ValueError, match="pi must have 2"):
        shoot.pack_arcs([20, 6, 0, 0], [(0.3, 0.7, [0])])
