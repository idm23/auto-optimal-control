"""Tapes, the run-time riding kernels, and a six-joint arm."""

import functools

import numpy as np
import pytest
import sympy as sp

from auto_optimal_control import examples, homotopy
from auto_optimal_control.build import build_kernels
from auto_optimal_control.homotopy import Homotopy
from auto_optimal_control.numeric import NumericSystem
from auto_optimal_control.policy import SecondaryCost, analyze
from auto_optimal_control.pontryagin import derive
from auto_optimal_control.problem import Box, OptimalControlProblem, PathConstraint
from auto_optimal_control.robot import (GRAVITY, SIX_JOINT_LINKS, jerk_controlled_arm,
                                        joint_torques, link_frames)
from auto_optimal_control.tape import Outlined, Tape, Unsupported


# -- tapes ------------------------------------------------------------------------

x, y = sp.symbols("x y", real=True)
EXPRESSIONS = [
    x * y + sp.sin(x) ** 2 * sp.cos(y) - 3,
    sp.exp(x * y) / (1 + x ** 2),
    sp.log(1 + x ** 2) * sp.tan(y / 4) - x ** -3,
    (1 + x ** 2) ** sp.Rational(3, 2) * y,
    -(x + y) * (x - y) + (y - x) ** 3,
]


@pytest.mark.parametrize("expr", EXPRESSIONS)
def test_tape_values_and_derivatives_match_sympy(expr):
    tape = Tape()
    step = tape.from_sympy(expr)
    point = {"x": 0.7, "y": -1.3}
    subs = {x: 0.7, y: -1.3}
    assert tape.evaluate([step], point)[0] == pytest.approx(float(expr.subs(subs)), rel=1e-13)
    forward = [tape.diff(step, n) for n in "xy"]
    reverse = tape.gradient(step, ["x", "y"])
    along = tape.directional(step, {"x": tape.const(2.0), "y": tape.const(-0.5)})
    exact = [float(sp.diff(expr, s).subs(subs)) for s in (x, y)]
    assert tape.evaluate(forward, point) == pytest.approx(exact, rel=1e-12)
    assert tape.evaluate(reverse, point) == pytest.approx(exact, rel=1e-12)
    assert tape.evaluate([along], point)[0] == pytest.approx(2 * exact[0] - 0.5 * exact[1], rel=1e-12)
    second = tape.diff(tape.diff(step, "x"), "y")
    assert tape.evaluate([second], point)[0] == pytest.approx(
        float(sp.diff(expr, x, y).subs(subs)), rel=1e-11)


def test_tape_merges_what_a_tree_writes_out():
    # Doubling a sum again and again: 2^40 terms written out, 40 steps recorded.
    tape, e = Tape(), x + y
    for _ in range(40):
        e = sp.Mul(e, x + 1, evaluate=False) + sp.Mul(e, y + 1, evaluate=False)
    step = tape.from_sympy(e)
    assert len(tape.reachable([step])) < 200
    assert np.isfinite(tape.evaluate([tape.diff(step, "x")], {"x": 0.01, "y": 0.02})[0])


def test_tape_refuses_what_it_cannot_differentiate():
    with pytest.raises(Unsupported):
        Tape().from_sympy(sp.Abs(x))


def test_outlined_differentiates_through_sympy():
    tape = Tape()
    expr = sp.sin(x) * y ** 2
    o = tape.outlined(tape.from_sympy(expr))
    assert isinstance(o, Outlined)
    assert sp.diff(o, x).has(Outlined) and sp.diff(sp.diff(o, x), x, y).has(Outlined)
    assert sp.diff(o, sp.Symbol("unrelated")) == 0
    # d^3 / dx^2 dy, carried by the tape behind ordinary sp.diff calls.
    d = sp.diff(o, x, x, y)
    node = next(iter(d.atoms(Outlined)))
    value = float(d / node) * tape.evaluate([int(node.args[0])], {"x": 0.3, "y": 2.0})[0]
    assert value == pytest.approx(float(sp.diff(expr, x, x, y).subs({x: 0.3, y: 2.0})))


def test_outlined_second_derivative_after_a_substitution():
    # Code generation substitutes C names first and differentiates after.
    tape = Tape()
    expr = sp.sin(x) * y ** 2
    a, b = sp.Symbol("z[0]"), sp.Symbol("z[1]")
    o = tape.outlined(tape.from_sympy(expr)).xreplace({x: a, y: b})
    d = sp.diff(o, a, b)
    assert d != 0 and d.free_symbols == {a, b}
    assert tape.evaluate([int(d.args[0])], {"x": 0.3, "y": 2.0})[0] == pytest.approx(
        float(sp.diff(expr, x, y).subs({x: 0.3, y: 2.0})))


# -- the torques ------------------------------------------------------------------

def _energy(links, q, w, gravity):
    """Kinetic plus potential energy, from the frames alone (no Newton-Euler)."""
    def com_and_axes(angles):
        return [(R, p + R @ np.array(link.com, dtype=float))
                for link, (R, p) in zip(links, link_frames(links, angles))]

    h = 1e-6
    here = com_and_axes(q)
    ahead, behind = com_and_axes(q + h * w), com_and_axes(q - h * w)
    E = 0.0
    for link, (R, c), (Ra, ca), (Rb, cb) in zip(links, here, ahead, behind):
        v = (ca - cb) / (2 * h)
        S = (Ra - Rb) / (2 * h) @ R.T          # the skew matrix of the angular rate
        omega = R.T @ np.array([S[2, 1], S[0, 2], S[1, 0]])
        E += 0.5 * float(link.mass) * v @ v
        E += 0.5 * omega @ (np.array(link.inertia, dtype=float) * omega)
        E += float(link.mass) * float(gravity) * c[2]
    return E


def test_torques_do_the_work_the_energy_gains():
    # sum tau_i w_i = dE/dt, with E from the frames: a check independent of the recursion.
    n = 6
    q, w, a = (sp.symbols(f"{s}1:{n + 1}", real=True) for s in "qwa")
    f = sp.lambdify((q, w, a), joint_torques(SIX_JOINT_LINKS, q, w, a, GRAVITY), "numpy")
    rng = np.random.default_rng(3)
    for _ in range(5):
        Q, W, A = rng.uniform(-2, 2, (3, n))
        h = 1e-4
        rate = (_energy(SIX_JOINT_LINKS, Q + h * W + h ** 2 / 2 * A, W + h * A, GRAVITY)
                - _energy(SIX_JOINT_LINKS, Q - h * W + h ** 2 / 2 * A, W - h * A, GRAVITY)) / (2 * h)
        power = float(np.dot(np.array(f(Q, W, A), dtype=float), W))
        assert power == pytest.approx(rate, rel=1e-5, abs=1e-5)


def test_mass_matrix_is_symmetric_positive_definite():
    n = 6
    q, w, a = (sp.symbols(f"{s}1:{n + 1}", real=True) for s in "qwa")
    f = sp.lambdify((q, w, a), joint_torques(SIX_JOINT_LINKS, q, w, a, 0), "numpy")
    Q = np.random.default_rng(1).uniform(-2, 2, n)
    M = np.column_stack([np.array(f(Q, np.zeros(n), e), dtype=float) for e in np.eye(n)])
    assert np.max(np.abs(M - M.T)) < 1e-13
    assert np.min(np.linalg.eigvalsh(M)) > 0


def test_a_twist_must_be_a_quarter_turn():
    bad = [SIX_JOINT_LINKS[0].__class__(0, 0, sp.pi / 3, 1, (0, 0, 0), (1, 1, 1))]
    with pytest.raises(ValueError, match="multiple of pi / 2"):
        joint_torques(bad, [sp.Symbol("q")], [sp.Symbol("w")], [sp.Symbol("a")])


# -- kernels ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_ways(kernel_cache):
    """The two-link arm's kernels both ways: symbolic arcs and arcs solved at run time."""
    problem = examples.two_link_arm()
    symbolic = derive(problem)
    numeric = derive(problem, outline_above=0, simultaneous=2)
    assert symbolic.riding is None and numeric.riding is not None
    return (build_kernels(symbolic, name="arm_symbolic", cache_dir=kernel_cache),
            build_kernels(numeric, name="arm_numeric", cache_dir=kernel_cache))


PARAMS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 0.5, 0.5]


def _close(a, b, rel=1e-11):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert np.max(np.abs(a - b)) <= rel * (1 + np.max(np.abs(a)))


@pytest.mark.slow
def test_run_time_arcs_agree_with_symbolic_ones(two_ways):
    ks, kn = two_ways
    assert ks.BOUNDARY_SETS == kn.BOUNDARY_SETS
    rng = np.random.default_rng(1)
    for arc in range(4 + len(ks.BOUNDARY_SETS)):
        for _ in range(10):
            z = rng.uniform(-2, 2, ks.N_Z)
            for name in ("boundary_rhs", "boundary_jacobian", "boundary_control",
                         "boundary_multiplier", "boundary_hamiltonian"):
                _close(getattr(ks, name)(arc, 0.3, z, PARAMS), getattr(kn, name)(arc, 0.3, z, PARAMS))
            _close(ks.boundary_multiplier_gradient(arc, 0.3, z, PARAMS)[1],
                   kn.boundary_multiplier_gradient(arc, 0.3, z, PARAMS)[1])
            _close(ks.hamiltonian_gradient(0.3, z, PARAMS, boundary=arc)[1],
                   kn.hamiltonian_gradient(0.3, z, PARAMS, boundary=arc)[1])


@pytest.mark.slow
def test_outlined_free_arc_kernels_agree_with_symbolic_ones(two_ways):
    ks, kn = two_ways
    rng = np.random.default_rng(2)
    for _ in range(10):
        z = rng.uniform(-2, 2, ks.N_Z)
        _close(ks.constraint(0.3, z, PARAMS), kn.constraint(0.3, z, PARAMS))
        _close(ks.rhs(0.3, z, PARAMS), kn.rhs(0.3, z, PARAMS))
        for c in range(4):
            for a, b in zip(ks.tangency_gradient(c, 0.3, z, PARAMS),
                            kn.tangency_gradient(c, 0.3, z, PARAMS)):
                _close(a, b)


def test_large_limits_share_roots_and_list_every_set():
    pmp = derive(jerk_controlled_arm(SIX_JOINT_LINKS))
    assert pmp.riding is not None
    assert pmp.riding.root_of == (0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5)
    assert pmp.riding.sign_of[:2] == (1, -1)
    assert len(pmp.boundary_sets) + 12 == 3 ** 6 - 1     # one side of each joint, or neither


def test_what_run_time_arcs_cannot_take_is_reported():
    problem = jerk_controlled_arm(SIX_JOINT_LINKS)
    q1 = problem.states[0]
    bounded = OptimalControlProblem(
        name="state_bound", time=problem.time, states=problem.states, controls=problem.controls,
        params=problem.params, dynamics=problem.dynamics, running_cost=problem.running_cost,
        initial_state=dict(problem.initial_state), terminal_constraints=problem.terminal_constraints,
        final_time=problem.final_time, path_constraints=problem.path_constraints,
        control_sets=[Box(problem.controls[0], -1 - q1 ** 2, 1 + q1 ** 2)])
    pmp = derive(bounded)
    assert pmp.riding is None
    assert all(c.boundary is None and "free of the states" in c.unsupported
               for c in pmp.path_constraints)


@pytest.fixture(scope="module")
def clipped_two_ways(kernel_cache):
    """The minimum-time two-link arm with its jerk cost, both ways: jerks clipped on a line."""
    stated = examples.two_link_arm_min_time()
    problem = SecondaryCost(0.01).apply(stated, analyze(stated))
    return (build_kernels(derive(problem), name="clipped_symbolic", cache_dir=kernel_cache),
            build_kernels(derive(problem, outline_above=0, simultaneous=2), name="clipped_numeric",
                          cache_dir=kernel_cache))


@pytest.mark.slow
def test_bounded_jerks_at_run_time_agree_with_the_clipped_law(clipped_two_ways):
    ks, kn = clipped_two_ways
    p = np.array([1, 1, 1, 1, 1.5, 1, 1.0, 2.2, 2.0])
    rng = np.random.default_rng(3)
    at_a_bound = compared = 0
    for arc in range(4):
        for _ in range(60):
            z = rng.uniform(-1.5, 1.5, ks.N_Z)
            z[ks.N_STATES:] *= rng.choice([0.01, 0.05, 0.3])
            try:
                kn.boundary_control(arc, 0.3, z, p)
            except RuntimeError:
                continue                   # no jerks within their bounds ride this limit here
            u = ks.boundary_control(arc, 0.3, z, p)
            compared += 1
            at_a_bound += np.any(np.abs(np.abs(u) - 1.0) < 1e-12)
            _close(u, kn.boundary_control(arc, 0.3, z, p))
            _close(ks.boundary_rhs(arc, 0.3, z, p), kn.boundary_rhs(arc, 0.3, z, p))
            _close(ks.boundary_jacobian(arc, 0.3, z, p), kn.boundary_jacobian(arc, 0.3, z, p))
            _close(ks.boundary_multiplier(arc, 0.3, z, p), kn.boundary_multiplier(arc, 0.3, z, p))
    assert compared > 100 and at_a_bound > 50


@pytest.mark.slow
@pytest.mark.parametrize("taus, t_f", [((2.2, 2.0), 3.656661), ((1.5, 0.4), 4.164857)])
def test_minimum_time_through_run_time_arcs(monkeypatch, kernel_cache, taus, t_f):
    # The same answers as the symbolic route: the benchmark's times, seeded with no draws.
    monkeypatch.setattr(homotopy, "derive", functools.partial(derive, outline_above=0))
    with pytest.warns(UserWarning, match="not unique"):
        h = Homotopy(examples.two_link_arm_min_time(), cache_dir=kernel_cache)
    assert h.kr.system.riding is not None
    r = h.solve([1, 1, 1, 1, 1.5, 1.0, 1.0, *taus], ([-3] * 6, [3] * 6), (1.0, 20.0))
    assert r.exact and r.tries == 0
    assert r.solution.tf == pytest.approx(t_f, abs=2e-6)


def test_axes_with_exactly_tied_moves_solve(kernel_cache):
    # Their saturation corners coincide: integration used to run for ever there. The time
    # of one axis alone is (32 D / J)^(1/3), a little longer with the jerk cost shared.
    with pytest.warns(UserWarning, match="not unique"):
        h = Homotopy(examples.two_axes_min_time(), cache_dir=kernel_cache)
    r = h.solve([1.0, 1.0, 1.0], ([-3] * 6, [3] * 6), (1.0, 20.0))
    assert r.exact and r.tries == 0
    assert r.solution.tf == pytest.approx(32 ** (1 / 3), abs=1e-3) and r.solution.tf > 32 ** (1 / 3)


def test_merging_a_limits_arcs_keeps_another_limits_arc_between_them():
    h = Homotopy.__new__(Homotopy)
    h.kr = type("K", (), {"CONSTRAINT_ORDERS": (1, 1)})()
    arcs = [("boundary", 0), ("boundary", 1), ("boundary", 0)]
    values = [(0.2, 0.3, np.zeros(1)), (0.31, 0.6, np.zeros(1)), (0.5, 0.7, np.ones(1))]
    h._arcs_of = lambda shoot, v: (arcs, values)

    class Shoot:
        @staticmethod
        def unpack(v):
            return None, None, None, 1.0
        t0 = 0.0

    (merged, spans), = h._activate(Shoot, None, [(0, 0.25, 0.55, 1.0)])
    assert merged == [("boundary", 0), ("boundary", 1)]
    assert spans[0][:2] == (0.2, 0.7) and spans[0][2] == pytest.approx([1.0])
    assert spans[1][:2] == (0.31, 0.6)


def test_newton_for_a_guess_accepts_a_stall_near_the_root():
    # A residual that stops falling at 3e-6, the noise of a loose integration: the seed
    # depended on which BLAS kernels rounding happened to favour.
    def noisy(v, with_jacobian):
        R = np.maximum(np.abs(v), 3e-6) * np.sign(v + 1e-300)
        return (R, np.eye(1)) if with_jacobian else R

    assert homotopy._newton(noisy, [1.0], 1e-6) is None
    assert abs(homotopy._newton(noisy, [1.0], 1e-6, accept=1e-4)[0]) < 1e-4
    assert homotopy._newton(noisy, [1.0], 1e-6, accept=1e-6) is None


def test_the_common_time_shares_the_cost_of_time():
    # Each path: (T, lam, H), with H = 1 + (the component's part). Alone a component would
    # stop where H = 0; two of them at the same time share it, so they stop later.
    path = [(3.0, None, 0.9), (2.0, None, 0.5), (1.0, None, 0.0)]
    assert Homotopy._common_time([path]) == pytest.approx(1.0)
    assert Homotopy._common_time([path, path]) == pytest.approx(2.0)


@pytest.fixture(scope="module")
def arm_kernels(kernel_cache):
    pmp = derive(jerk_controlled_arm(SIX_JOINT_LINKS, name="six_joint_arm"))
    return pmp, build_kernels(pmp, cache_dir=kernel_cache)


def _arm_params(k, limit=50.0):
    p = np.zeros(len(k.param_names))
    p[0] = 1.0
    p[-6:] = limit
    return p


@pytest.mark.slow
def test_every_arc_of_the_arm_holds_exactly_its_limits(arm_kernels):
    # Along F_b, the rate of each limit ridden is zero, and mu is what the KKT system says.
    _, k = arm_kernels
    p = _arm_params(k)
    rng = np.random.default_rng(4)
    arcs = list(range(12)) + list(rng.choice(np.arange(12, 12 + len(k.BOUNDARY_SETS)), 40))
    h = 1e-6
    for arc in arcs:
        z = rng.uniform(-1, 1, k.N_Z)
        dz = k.boundary_rhs(arc, 0.3, z, p)
        rate = (k.constraint(0.3, z + h * dz, p) - k.constraint(0.3, z - h * dz, p)) / (2 * h)
        for c in k.arc_constraints(arc):
            assert abs(rate[c]) < 1e-5


@pytest.mark.slow
def test_arc_jacobians_of_the_arm_match_differences(arm_kernels):
    _, k = arm_kernels
    p = _arm_params(k)
    rng = np.random.default_rng(5)
    h = 1e-6
    for arc in (0, 7, k.boundary_set([0, 3]), k.boundary_set([1, 2, 5, 7, 8, 11])):
        z = rng.uniform(-1, 1, k.N_Z)
        A = k.boundary_jacobian(arc, 0.3, z, p)
        for j in range(k.N_Z):
            e = np.zeros(k.N_Z)
            e[j] = h
            fd = (k.boundary_rhs(arc, 0.3, z + e, p) - k.boundary_rhs(arc, 0.3, z - e, p)) / (2 * h)
            assert np.max(np.abs(A[:, j] - fd)) < 2e-5 * (1 + np.max(np.abs(fd)))


def test_numeric_reference_refuses_outlined_limits():
    with pytest.raises(ValueError, match="on a tape"):
        NumericSystem(derive(examples.two_link_arm(), outline_above=0, simultaneous=1))


def test_the_arm_integrates_only_the_entries_of_phi_a_free_arc_can_hold(arm_kernels):
    # On a free arc each joint is a triple integrator, with 21 entries of its 36.
    _, k = arm_kernels
    n = k.N_Z
    assert k.STM_IS_SPARSE and k.N_PHI == 21 * 6
    p = _arm_params(k)
    z0 = np.linspace(-0.4, 0.5, n)
    dense = k.propagate(0.0, z0, 0.8, p, stm=True, sparse_stm=False, rtol=1e-11, atol=1e-11)
    sparse = k.propagate(0.0, z0, 0.8, p, stm=True, rtol=1e-11, atol=1e-11)
    assert sparse["z"] == pytest.approx(dense["z"], abs=1e-9)
    assert sparse["stm"] == pytest.approx(dense["stm"], abs=1e-8)
    outside = np.ones(n * n, dtype=bool)
    outside[k.PHI_ENTRIES] = False
    assert (dense["stm"].reshape(-1)[outside] == 0.0).all()


def test_a_penalty_that_comes_on_mid_flight_gets_the_whole_phi(kernel_cache):
    # Off, the two axes are separate and Phi is sparse. On, the penalty ties them together.
    t = sp.Symbol("t", real=True)
    x1, v1, x2, v2, u1, u2 = sp.symbols("x1 v1 x2 v2 u1 u2", real=True)
    problem = OptimalControlProblem(
        name="two_axes_penalized", time=t, states=[x1, v1, x2, v2], controls=[u1, u2],
        dynamics=[v1, u1, v2, u2], running_cost=(u1 ** 2 + u2 ** 2) / 2,
        initial_state=[0, 0, 0, 0], final_state={x1: 1, v1: 0, x2: 1, v2: 0}, final_time=1,
        path_constraints=[PathConstraint(x1 + x2 - 1, "both", penalty=10)])
    k = build_kernels(derive(problem), cache_dir=kernel_cache)
    assert k.STM_IS_SPARSE and k.N_PHI == 20
    quiet = [0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1]
    across = [0, 1, 0, 1, 0, 0, 0, 0]
    for z0, crosses in ((quiet, False), (across, True)):
        dense = k.propagate(0.0, z0, 1.0, stm=True, sparse_stm=False, rtol=1e-11, atol=1e-11)
        sparse = k.propagate(0.0, z0, 1.0, stm=True, rtol=1e-11, atol=1e-11)
        assert len(dense["constraint_times"]) == int(crosses)
        assert sparse["z"] == pytest.approx(dense["z"], abs=1e-12)
        assert sparse["stm"] == pytest.approx(dense["stm"], abs=1e-12)
        outside = np.ones(64, dtype=bool)
        outside[k.PHI_ENTRIES] = False
        assert np.abs(dense["stm"].reshape(-1)[outside]).max() > 0.0 if crosses else True
    # Starting on the penalty, and integrating sensitivities to the parameters, are dense too.
    on = k.propagate(0.0, [2, 0, 0, 0, 0, 0, 0, 0], 0.5, stm=True, rtol=1e-11, atol=1e-11)
    assert np.abs(on["stm"].reshape(-1)[outside]).max() > 0.0


def test_a_violation_across_a_junction_is_one_violation():
    h = Homotopy.__new__(Homotopy)
    h.nc = 1
    t1, t2 = np.linspace(0.0, 1.0, 11), np.linspace(1.0, 2.0, 11)
    g1, g2 = (t1 - 0.8)[:, None], (1.2 - t2)[:, None]    # positive from 0.8 to 1.2
    h._free_samples = lambda shoot, v: [(t1, g1, ()), (t2, g2, (5,))]
    found = h._free_violations(None, None, np.ones(1))
    assert len(found) == 1
    assert found[0][0] == 0 and found[0][1] <= 0.8 and found[0][2] >= 1.2


def test_a_limit_broken_at_the_end_state_is_reported_at_once(kernel_cache):
    t, x, u = sp.symbols("t x u", real=True)
    L = sp.Symbol("L", positive=True)
    problem = OptimalControlProblem(
        name="limit_below_goal", time=t, states=[x], controls=[u], params=[L], dynamics=[u],
        running_cost=u ** 2 / 2, initial_state=[0], final_state={x: 1}, final_time=1,
        path_constraints=[PathConstraint(x - L, "x_max")])
    h = Homotopy(problem, cache_dir=kernel_cache)
    r = h.solve([0.5], ([-5], [5]))
    assert not r.exact and "the final state violates x_max by 0.5" in r.message
    assert h.solve([1.5], ([-5], [5])).exact


@pytest.mark.slow
def test_the_arm_solves_with_overlapping_limits(arm_kernels, kernel_cache):
    # Tightened on three joints at once: arcs overlap, and the torques hold at their limits.
    h = Homotopy(examples.six_joint_arm(), cache_dir=kernel_cache)
    T, qf = 1.0, np.array([1.0, -1.0, 0.8, 1.0, -0.8, 1.0])
    limits = [18.8, 66.5, 20.4, 2.0, 0.2, 0.01]
    r = h.solve([T, *np.zeros(6), *qf, *limits], ([-50] * 18, [50] * 18))
    assert r.exact and any(kind == "enter" for kind, _ in r.arcs)
    tr = r.shooting.trajectory(r.solution.v, n_points=2001)
    assert np.max(tr["g"]) < 1e-7
    assert r.objective > 360 * qf @ qf / T ** 5
