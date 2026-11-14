"""SingleShooting: unknown layout, exact Jacobian, solving, continuation."""

import numpy as np
import pytest
import sympy as sp
from scipy.linalg import expm

from auto_optimal_control.build import build_kernels
from auto_optimal_control.pontryagin import derive
from auto_optimal_control.problem import OptimalControlProblem
from auto_optimal_control.shooting import (ShootingError, ShootingSolution, SingleShooting,
                                           continuation)

MIN_TIME = [2.0, 1.0, 0.0]             # u_max, x(0), v(0)


def fd_jacobian(shoot, v, h=1e-6):
    cols = []
    for i in range(len(v)):
        e = np.zeros(len(v))
        e[i] = h * max(1.0, abs(v[i]))
        cols.append((shoot.residual(v + e) - shoot.residual(v - e)) / (2 * e[i]))
    return np.column_stack(cols)


@pytest.fixture(scope="module")
def free_start(kernel_cache):
    """x' = -x + u, L = (x^2 + u^2)/2, x(0) free, x(T) = 1: a free initial state."""
    t, x, u = sp.symbols("t x u", real=True)
    T = sp.Symbol("T", positive=True)
    problem = OptimalControlProblem(
        name="free_start", time=t, states=[x], controls=[u], params=[T], dynamics=[-x + u],
        running_cost=(x ** 2 + u ** 2) / 2, final_state={x: 1}, final_time=T)
    return build_kernels(derive(problem), cache_dir=kernel_cache)


def test_layout_min_time(compiled):
    k = compiled["min_time"][2]
    shoot = SingleShooting(k, MIN_TIME)
    assert shoot.n_unknowns == 2 + 2 + 1           # lam0, nu, tf
    np.testing.assert_array_equal(shoot.x0, [1.0, 0.0])
    v = shoot.pack([1.0, 2.0], [3.0, 4.0], tf=5.0)
    x0, lam0, nu, tf = shoot.unpack(v)
    np.testing.assert_array_equal(x0, [1.0, 0.0])
    np.testing.assert_array_equal(lam0, [1.0, 2.0])
    np.testing.assert_array_equal(nu, [3.0, 4.0])
    assert tf == 5.0
    with pytest.raises(ValueError, match="free"):
        shoot.pack([1.0, 2.0])
    with pytest.raises(ValueError, match="parameters"):
        SingleShooting(k, [1.0])


def test_min_time_solve_matches_closed_form(compiled):
    k = compiled["min_time"][2]
    shoot = SingleShooting(k, MIN_TIME)
    umax, a = MIN_TIME[0], MIN_TIME[1]
    ts = np.sqrt(a / umax)
    lam = [1.0 / (umax * ts), 1.0 / umax]
    v = shoot.guess_multipliers([1.15 * lam[0], 0.9 * lam[1]], tf=1.1 * 2 * ts)
    np.testing.assert_allclose(shoot.residual_jacobian(v)[1], fd_jacobian(shoot, v),
                               rtol=1e-6, atol=1e-6)
    sol = shoot.solve(v)
    assert isinstance(sol, ShootingSolution) and sol.success
    np.testing.assert_allclose(sol.lam0, lam, rtol=1e-10)
    assert abs(sol.tf - 2 * ts) < 1e-10
    np.testing.assert_allclose(sol.z0, np.concatenate([[a, 0.0], lam]), rtol=1e-10)
    tr = shoot.trajectory(sol.v, n_points=11)
    np.testing.assert_allclose(tr["z"][-1, :2], 0.0, atol=1e-10)
    np.testing.assert_allclose(tr["switch_times"], [ts], rtol=1e-10)


def test_newton_polishes_a_near_guess_past_the_tolerance(compiled):
    """solve_newton: from a guess close to the answer it converges quadratically, takes
    one step past ``tol`` (``polish``), and stops at the first step that does not help."""
    k = compiled["min_time"][2]
    shoot = SingleShooting(k, MIN_TIME)
    umax, a = MIN_TIME[0], MIN_TIME[1]
    ts = np.sqrt(a / umax)
    exact = shoot.solve(shoot.guess_multipliers([1.0 / (umax * ts), 1.0 / umax], tf=2 * ts))
    assert exact.success
    near = exact.v * (1 + 1e-5)
    sol = shoot.solve_newton(near, tol=1e-8)
    assert sol.success and sol.residual_norm < 1e-8
    np.testing.assert_allclose(sol.v, exact.v, rtol=1e-10)
    loose = shoot.solve_newton(near, tol=1e-8, polish=0)
    assert loose.success and loose.nfev <= sol.nfev
    assert sol.residual_norm <= loose.residual_norm
    # A guess too far for Newton fails cleanly and leaves MINPACK to the caller.
    far = shoot.solve_newton(exact.v * 3.0, tol=1e-12, max_iterations=1, polish=0)
    assert not far.success


def test_zermelo_solve(compiled):
    k = compiled["zermelo"][2]
    p = [1.0, 1.0, 1.0, 0.5]
    shoot = SingleShooting(k, p)
    sol = shoot.solve(shoot.pack([-0.7, -0.7], [-0.7, -0.7], tf=1.0))
    assert sol.success
    r = k.propagate(sol.t0, sol.z0, sol.tf, p, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(r["z"][:2], [1.0, 0.5], atol=1e-10)
    assert abs(k.hamiltonian(0.0, sol.z0, p)) < 1e-10


def test_free_initial_state(free_start):
    shoot = SingleShooting(free_start, [1.0])
    assert shoot.n_free == 1 and shoot.n_unknowns == 1 + 1 + 1     # x0, lam0, nu
    v = shoot.pack([0.3], [0.1], x0_free=[0.5])
    np.testing.assert_allclose(shoot.residual_jacobian(v)[1], fd_jacobian(shoot, v),
                               rtol=1e-6, atol=1e-7)
    sol = shoot.solve(v)
    assert sol.success
    # Natural condition at the free start, and the target reached.
    assert abs(sol.lam0[0]) < 1e-10
    tr = shoot.trajectory(sol.v, n_points=3)
    assert abs(tr["z"][-1, 0] - 1.0) < 1e-10
    # Closed form: u = -lam, so (x, lam)' = A (x, lam) with lam(0) = 0.
    E = expm(np.array([[-1.0, -1.0], [-1.0, 1.0]]))
    assert abs(sol.x0[0] - 1.0 / E[0, 0]) < 1e-10


def test_failed_propagation(compiled):
    k = compiled["min_time"][2]
    shoot = SingleShooting(k, MIN_TIME)
    with pytest.raises(ShootingError, match="not after"):
        shoot.residual(shoot.pack([1.0, 1.0], tf=-1.0))
    # solve() reports failure instead of raising.
    sol = shoot.solve(shoot.pack([1.0, 1.0], tf=-1.0), options={"maxfev": 5})
    assert not sol.success


def test_multistart_and_continuation(compiled):
    k = compiled["soft_terminal"][2]                # params T, w, a
    T, a = 1.0, 2.0

    def exact(w):                                   # lam = w x(T), x(T) = a - lam T
        return w * a / (1 + w * T)

    shoot = SingleShooting(k, [T, 1.0, a])
    assert shoot.n_unknowns == 1
    sol, tries = shoot.multistart(lambda rng: shoot.pack(rng.uniform(-5, 5, 1)), n_tries=5)
    assert sol.success and tries == 1
    assert abs(sol.lam0[0] - exact(1.0)) < 1e-10

    ws = [10.0, 100.0, 1000.0]
    path = continuation(lambda w, g: SingleShooting(k, [T, w, a]).solve(g), 1.0, ws, sol.v)
    assert [w for w, _ in path][-1] == ws[-1]
    for w, s in path:
        assert abs(s.lam0[0] - exact(w)) < 1e-9 * max(1.0, exact(w))

    with pytest.raises(RuntimeError, match="stalled"):
        continuation(lambda w, g: ShootingSolution(g, None, None, None, 0, 1, 1.0, False, 0),
                     1.0, [2.0], sol.v)
