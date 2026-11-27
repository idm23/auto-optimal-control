"""SingleShooting: unknown layout, exact Jacobian, solving, continuation."""

import time

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


def test_line_newton_reaches_the_answer_and_fails_cleanly(compiled):
    """solve_line_newton (Rule R90): from a guess near the answer it converges to it with
    fresh Jacobians, and from one it cannot improve along Newton's direction it fails
    cleanly, leaving MINPACK to the caller, at least as low as solve_newton gets."""
    k = compiled["min_time"][2]
    shoot = SingleShooting(k, MIN_TIME)
    umax, a = MIN_TIME[0], MIN_TIME[1]
    ts = np.sqrt(a / umax)
    exact = shoot.solve(shoot.guess_multipliers([1.0 / (umax * ts), 1.0 / umax], tf=2 * ts))
    assert exact.success
    sol = shoot.solve_line_newton(exact.v * (1 + 1e-5), tol=1e-9)
    assert sol.success and sol.residual_norm < 1e-9
    np.testing.assert_allclose(sol.v, exact.v, rtol=1e-8)
    guess = shoot.guess_multipliers([1.15 / (umax * ts), 0.9 / umax], tf=1.1 * 2 * ts)
    plain = shoot.solve_newton(guess, tol=1e-9, max_iterations=12, polish=0)
    searched = shoot.solve_line_newton(guess, tol=1e-9)
    # Both may reach the answer; below the tolerance their residuals are rounding.
    assert searched.residual_norm <= max(plain.residual_norm, 1e-12)
    stuck = shoot.solve_line_newton(exact.v * 3.0, tol=1e-12, max_iterations=1, max_halvings=1)
    assert not stuck.success


def test_chord_and_newton_stop_where_the_start_cannot_be_propagated(compiled):
    """A start whose propagation fails has no Jacobian. _safe's stand-in (a large
    residual, the identity) read as one gives the step -R, moving every unknown by
    1e6, so both iterations stop there instead, with _safe's residual."""
    from auto_optimal_control.shooting import ShootingError
    k = compiled["min_time"][2]
    shoot = SingleShooting(k, MIN_TIME)
    v = shoot.guess_multipliers([1.0, 1.0], tf=1.0)
    calls = []

    def fails(x, *a, **kw):
        calls.append(np.array(x))
        raise ShootingError("propagation failed", "free_failed")

    shoot.residual_jacobian = fails
    shoot.residual = fails
    for solve in (shoot.solve_chord, shoot.solve_newton, shoot.solve_line_newton):
        calls.clear()
        sol = solve(v)
        assert not sol.success and sol.nfev == 1 and len(calls) == 1
        np.testing.assert_array_equal(sol.v, v)
        assert sol.residual_norm == np.linalg.norm(np.full(len(v), 1e6))


def test_zermelo_solve(compiled):
    k = compiled["zermelo"][2]
    p = [1.0, 1.0, 1.0, 0.5]
    shoot = SingleShooting(k, p)
    sol = shoot.solve(shoot.pack([-0.7, -0.7], [-0.7, -0.7], tf=1.0))
    assert sol.success
    r = k.propagate(sol.t0, sol.z0, sol.tf, p, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(r["z"][:2], [1.0, 0.5], atol=1e-10)
    assert abs(k.hamiltonian(0.0, sol.z0, p)) < 1e-10


def test_trajectory_integrates_with_the_shootings_options(compiled):
    """trajectory() is the shooting's own integration, at its tolerances, not the
    kernels' defaults: replayed at the defaults, a minimum-fuel orbit raise solved at
    rtol 1e-12 gained a switch and missed its target by 0.018."""
    k = compiled["zermelo"][2]
    p = [1.0, 1.0, 1.0, 0.5]
    sol = SingleShooting(k, p).solve(SingleShooting(k, p).pack([-0.7, -0.7], [-0.7, -0.7], tf=1.0))
    loose = SingleShooting(k, p, rtol=1e-4, atol=1e-4)
    x0, lam0, _, tf = loose.unpack(sol.v)
    own = loose._propagate(np.concatenate([x0, lam0]), tf, stm=False)["z"]
    np.testing.assert_allclose(loose.trajectory(sol.v, n_points=5)["z"][-1], own, rtol=0, atol=1e-13)
    # The loose tolerances matter here, so the check above could fail; an explicit
    # argument still wins.
    tight = loose.trajectory(sol.v, n_points=5, rtol=1e-12, atol=1e-12)["z"][-1]
    assert np.abs(tight - own).max() > 1e-9


def test_past_the_deadline_every_propagation_fails_and_a_solve_stops(compiled):
    """SingleShooting.deadline (set by Homotopy.solve from solve_seconds) fails every
    propagation once passed, so a root-find stops instead of carrying a solve past its
    cap; before it, nothing changes."""
    k = compiled["zermelo"][2]
    p = [1.0, 1.0, 1.0, 0.5]
    shoot = SingleShooting(k, p)
    guess = shoot.pack([-0.7, -0.7], [-0.7, -0.7], tf=1.0)
    saved = SingleShooting.deadline
    try:
        SingleShooting.deadline = time.perf_counter() + 3600
        assert shoot.solve(guess).success
        SingleShooting.deadline = 0.0
        with pytest.raises(ShootingError) as err:
            shoot.residual(guess)
        assert err.value.failure == "deadline"
        late = SingleShooting(k, p)
        sol = late.solve(guess)
        assert not sol.success and late.failures.get("deadline", 0) > 0
    finally:
        SingleShooting.deadline = saved


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
    # Past its deadline it makes one draw and stops (a minimum-time orbit transfer of
    # a hundred revolutions spent hours on 200 failing draws under a 300 s cap).
    def no_guess(rng):
        raise ShootingError("no guess")

    assert shoot.multistart(no_guess, n_tries=50) == (None, 50)
    assert shoot.multistart(no_guess, n_tries=50, deadline=0.0) == (None, 1)

    ws = [10.0, 100.0, 1000.0]
    path = continuation(lambda w, g: SingleShooting(k, [T, w, a]).solve(g), 1.0, ws, sol.v)
    assert [w for w, _ in path][-1] == ws[-1]
    for w, s in path:
        assert abs(s.lam0[0] - exact(w)) < 1e-9 * max(1.0, exact(w))

    with pytest.raises(RuntimeError, match="stalled"):
        continuation(lambda w, g: ShootingSolution(g, None, None, None, 0, 1, 1.0, False, 0),
                     1.0, [2.0], sol.v)


def _chord_stub(fun, jac):
    """A SingleShooting whose residual is ``fun`` (Jacobian ``jac``): only what
    solve_chord reads."""
    shoot = SingleShooting.__new__(SingleShooting)
    shoot.t0 = 0.0
    shoot.unpack = lambda v: (v, np.zeros(1), np.zeros(0), 1.0)
    calls = {"jac": 0, "res": 0}

    def try_(v):
        calls["jac"] += 1
        return fun(v), jac(v)

    def res(v):
        calls["res"] += 1
        return fun(v)

    shoot._try, shoot._safe_residual = try_, res
    return shoot, calls


def test_chord_contraction_reads_the_distance_to_the_root():
    """Rule R37's measurement: Newton's first contraction from a guess at distance d
    of a root of a smooth map is omega d / 2 to first order (Kantorovich), so it halves
    with d, and a guess too far for Newton is turned back after one Jacobian and one
    residual, with the guess unchanged."""
    a = np.array([[2.0, 0.3], [0.1, 1.0]])
    fun = lambda v: a @ v + np.array([v[0] ** 2, v[0] * v[1]])
    jac = lambda v: a + np.array([[2 * v[0], 0.0], [v[1], v[0]]])
    thetas = []
    for d in (1e-2, 5e-3):
        shoot, _ = _chord_stub(fun, jac)
        sol = shoot.solve_chord(np.array([d, -d]), tol=1e-12)
        assert sol.success and not sol.too_far
        thetas.append(sol.contraction)
    assert 0 < thetas[1] < thetas[0] < 0.05
    assert thetas[0] / thetas[1] == pytest.approx(2.0, rel=0.05)
    shoot, calls = _chord_stub(fun, jac)
    guess = np.array([-0.9, 0.5])
    sol = shoot.solve_chord(guess, tol=1e-12, reject_theta=0.5)
    assert sol.too_far and not sol.success and sol.contraction > 0.5
    np.testing.assert_array_equal(sol.v, guess)
    assert calls == {"jac": 1, "res": 1}
    # Without the threshold the same guess is iterated as before.
    shoot, _ = _chord_stub(fun, jac)
    assert not shoot.solve_chord(guess, tol=1e-12).too_far


def test_a_segment_of_rounding_width_samples_without_raising():
    """Two junctions at one instant leave a segment one float wide (or of zero width):
    linspace repeats a time there, which an integrator refuses. The segment is then its
    ends, or its one point."""
    from auto_optimal_control.shooting import sample_segment

    def decay(t, y):
        return -y

    a = 0.636959565379545
    for b in (np.nextafter(a, 1.0), np.nextafter(a, 0.0)):
        t, z = sample_segment(decay, a, b, [1.0], 5, method="DOP853", rtol=1e-12, atol=1e-12)
        assert list(t) == [a, b] and z.shape == (2, 1)
    t, z = sample_segment(decay, a, a, [1.0], 5, method="DOP853", rtol=1e-12, atol=1e-12)
    assert list(t) == [a] and z.tolist() == [[1.0]]
    t, z = sample_segment(decay, 0.0, 1.0, [1.0], 5, method="DOP853", rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(z[:, 0], np.exp(-t), rtol=1e-9)


def test_rule_R55_accepts_a_step_that_contracts_newton_but_not_the_residual():
    # Lemma AC: on R = D G(v) with G(x) = (x0 + x1^2, x1) and D = diag(1e3, 1), Newton's
    # first step from (0, 0.01) contracts its correction a hundredfold, but the residual
    # stays at 0.1: the residual test backtracks where the natural one goes on.
    from types import SimpleNamespace

    from auto_optimal_control.shooting import SingleShooting

    D = np.diag([1e3, 1.0])

    class Stub(SingleShooting):
        def __init__(self, natural):
            self.natural_monotonicity = natural

        def _safe_residual(self, v):
            return D @ np.array([v[0] + v[1] ** 2, v[1]])

        def _try(self, v):
            return self._safe_residual(v), D @ np.array([[1.0, 2 * v[1]], [0.0, 1.0]])

        def _chord_result(self, v, err, nfev, tol=0.0):
            return SimpleNamespace(v=v, residual_norm=err, success=err < tol, nfev=nfev)

    natural = Stub(True).solve_chord(np.array([0.0, 0.01]), tol=1e-12)
    residual = Stub(False).solve_chord(np.array([0.0, 0.01]), tol=1e-12)
    assert natural.success and np.allclose(natural.v, 0.0, atol=1e-12)
    assert natural.nfev < residual.nfev
