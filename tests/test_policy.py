"""Degenerate problems found from the statement, and the policies that make them unique."""

import warnings

import numpy as np
import pytest
import sympy as sp

from auto_optimal_control import examples
from auto_optimal_control.homotopy import Homotopy
from auto_optimal_control.policy import (IllPosedProblem, Lexicographic, SecondaryCost,
                                         analyze, resolve)
from auto_optimal_control.problem import Box, OptimalControlProblem

DEGENERATE = {"two_link_arm_min_time", "two_axes_min_time"}


@pytest.mark.parametrize("name", list(examples.all_examples()))
def test_only_the_degenerate_examples_are_flagged(name):
    a = analyze(examples.all_examples()[name])
    assert a.degenerate == (name in DEGENERATE), str(a)


def test_two_axes_are_separable_and_both_slack():
    a = analyze(examples.two_axes_min_time())
    assert a.degenerate and a.separable and a.pure_minimum_time
    assert [c.controls for c in a.controlled] == [(sp.Symbol("j1", real=True),),
                                                  (sp.Symbol("j2", real=True),)]
    assert len(a.slack) == 2 and not a.linked_by


def test_torque_limits_link_the_arm_joints_but_do_not_remove_the_freedom():
    a = analyze(examples.two_link_arm_min_time())
    assert a.degenerate and not a.separable
    assert set(a.linked_by) == {"tau1_max", "tau1_min", "tau2_max", "tau2_min"}
    assert "only while they are active" in str(a)


def test_one_axis_in_minimum_time_is_fine():
    a = analyze(examples.jerk_limited_move())
    assert not a.degenerate and len(a.controlled) == 1 and a.time_priced


def _axes(cost, final_time=None, coupled=False):
    t = sp.Symbol("t", real=True)
    x1, v1, x2, v2, u1, u2 = sp.symbols("x1 v1 x2 v2 u1 u2", real=True)
    return OptimalControlProblem(
        time=t, states=[x1, v1, x2, v2], controls=[u1, u2],
        dynamics=[v1, u1, v2, u2 + (x1 if coupled else 0)], running_cost=cost(u1, u2, x1, x2),
        initial_state=[0] * 4, final_state={x1: 1, v1: 0, x2: 1, v2: 0},
        final_time=final_time, control_sets=[Box(u1, -1, 1), Box(u2, -1, 1)])


def test_rules():
    # Coupled through the dynamics: one component, so minimum time is fine.
    assert not analyze(_axes(lambda u1, u2, x1, x2: 1, coupled=True)).degenerate
    # Each axis priced: unique.
    assert not analyze(_axes(lambda u1, u2, x1, x2: 1 + u1 ** 2 + u2 ** 2)).degenerate
    # Fixed time, axis 2 in no cost term: any feasible motion of it is optimal.
    a = analyze(_axes(lambda u1, u2, x1, x2: u1 ** 2, final_time=3))
    assert a.degenerate and [c.controls for c in a.slack] == [(sp.Symbol("u2", real=True),)]
    assert "no cost term involves the controls u2" in str(a)
    # A cost term on both axes couples and prices them.
    assert not analyze(_axes(lambda u1, u2, x1, x2: (x1 - x2) ** 2, final_time=3)).degenerate


def test_unbounded_linear_control_is_flagged():
    t, x, v, u = sp.symbols("t x v u", real=True)
    p = OptimalControlProblem(time=t, states=[x, v], controls=[u], dynamics=[v, u],
                              running_cost=1, initial_state=[0, 0], final_state={x: 1, v: 0})
    a = analyze(p)
    assert a.unbounded == (u,) and a.degenerate
    # The secondary cost makes the Hamiltonian convex in it.
    q = SecondaryCost(0.1).apply(p, a)
    assert sp.simplify(q.running_cost - (1 + 0.1 * u ** 2 / 2)) == 0
    with pytest.raises(ValueError, match="unbounded"):
        Lexicographic().check(a)


def test_secondary_cost_scales_each_control_by_its_bound():
    p = examples.two_link_arm_min_time()
    q = SecondaryCost(0.01).apply(p, analyze(p))
    j1, j2 = p.controls
    jmax = [s for s in p.params if s.name == "j_max"][0]
    assert sp.simplify(q.running_cost - 1 - 0.01 * (j1 ** 2 + j2 ** 2) / jmax ** 2 / 2) == 0


def test_resolve():
    a = analyze(examples.two_axes_min_time())
    assert isinstance(resolve("auto", a), SecondaryCost)
    assert resolve("as_stated", a) is None
    with pytest.raises(IllPosedProblem, match="Choose a policy"):
        resolve("strict", a)
    with pytest.raises(ValueError, match="policy must be"):
        resolve("fastest", a)
    fine = analyze(examples.jerk_limited_move())
    assert resolve("auto", fine) is None and resolve("strict", fine) is None


def test_lexicographic_stages_split_the_axes():
    p = examples.two_axes_min_time()
    parts, final = Lexicographic(0.05).stages(p, analyze(p))
    assert [idx for _, idx in parts] == [[0, 1, 2], [3, 4, 5]]
    for sub, idx in parts:
        assert sub.free_final_time and len(sub.states) == 3 and len(sub.terminal_constraints) == 3
        assert not analyze(sub).degenerate
    assert not final.free_final_time and final.final_time == final.params[-1]
    with pytest.raises(ValueError, match="link the components"):
        Lexicographic().stages(examples.two_link_arm_min_time(),
                               analyze(examples.two_link_arm_min_time()))


def test_homotopy_warns_when_it_applies_the_default():
    with pytest.warns(UserWarning, match="Applied the default policy"):
        h = Homotopy(examples.two_axes_min_time())
    assert isinstance(h.policy, SecondaryCost) and h.problem is not h.stated
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h = Homotopy(examples.two_axes_min_time(), policy=SecondaryCost(0.1))
        assert "0.1" in h.policy_text
        h = Homotopy(examples.jerk_limited_move())
        assert h.policy is None and h.problem is h.stated
    with pytest.raises(IllPosedProblem):
        Homotopy(examples.two_axes_min_time(), policy="strict")


T1 = (32 * 10.0) ** (1 / 3)        # axis 1 alone: (32 D1 / J)^(1/3)


@pytest.mark.slow
def test_two_axes_secondary_cost_approaches_the_minimum_time():
    h = Homotopy(examples.two_axes_min_time(), policy=SecondaryCost(1e-3))
    r = h.solve([1.0, 10.0, 2.0], ([-3] * 6, [3] * 6), (2.0, 20.0))
    assert r.exact and "secondary cost" in r.policy
    assert 0 < r.solution.tf - T1 < 1e-5


@pytest.mark.slow
def test_two_axes_lexicographic_is_bang_bang_then_quintic():
    h = Homotopy(examples.two_axes_min_time(), policy=Lexicographic(0.01))
    r = h.solve([1.0, 10.0, 2.0], ([-3] * 6, [3] * 6), (2.0, 20.0))
    assert r.exact and len(r.stages) == 2
    assert r.stages[0].solution.tf == pytest.approx(T1, rel=1e-10)
    assert r.stages[1].solution.tf == pytest.approx(4.0, rel=1e-10)     # (32 * 2)^(1/3)
    assert r.solution.tf == pytest.approx(1.01 * T1, rel=1e-12)
    tr = r.shooting.trajectory(r.solution.v, n_points=2001)
    s = tr["t"] / tr["t"][-1]
    # The slack axis, minimizing its jerk energy in the time given, is the quintic.
    assert np.abs(tr["z"][:, 3] - 2.0 * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)).max() < 1e-9
    assert np.abs(tr["u"][:, 0]).max() <= 1.0 + 1e-12


def test_riding_a_torque_limit_keeps_the_jerks_within_their_bounds():
    """One limit ridden with two bounded controls leaves a line of controls; the
    law is the best point on it within the box, and mu the multiplier of the
    rows of the controls off their bounds (Karush-Kuhn-Tucker)."""
    p0 = examples.two_link_arm_min_time()
    p = SecondaryCost().apply(p0, analyze(p0))
    from auto_optimal_control.pontryagin import derive
    pmp = derive(p)
    P = [1, 1, 1, 1, 1.5, 1.0, 1.0, 2.5, 0.9]
    rng = np.random.default_rng(1)
    mu_s = sp.Symbol("mu_check")
    for sc in (pmp.path_constraints[0], pmp.path_constraints[2]):
        arc, gq = sc.boundary, sc.derivatives[-1]
        args = list(pmp.z) + list(p.params)
        law = sp.lambdify(args, [arc.control[c] for c in p.controls])
        mu = sp.lambdify(args, arc.multiplier)
        L = (p.running_cost + sum(lam * f for lam, f in zip(pmp.costates, p.dynamics))
             + mu_s * gq)
        rate = sp.lambdify(args + list(p.controls), gq)
        grad = sp.lambdify(args + list(p.controls) + [mu_s],
                           [sp.diff(L, c) for c in p.controls])
        clipped = infeasible = 0
        for _ in range(500):
            z = rng.uniform(-1, 1, 12) * np.r_[[1.0] * 6, [0.05] * 6]
            u = np.array(law(*z, *P))
            if np.abs(u).max() > 1 + 1e-12:
                infeasible += 1        # no point of the line within the box
                continue
            at = np.abs(np.abs(u) - 1) < 1e-12
            clipped += at.any()
            assert abs(rate(*z, *P, *u)) < 1e-9
            gr = np.array(grad(*z, *P, *u, mu(*z, *P)))
            assert np.all(np.abs(gr[~at]) < 1e-9)
            # At a bound, H + mu g^(q) decreases outward: the bound is what holds u.
            assert np.all(gr[at] * np.sign(u[at]) <= 1e-9)
        assert clipped > 100 and infeasible < 10


ARM = [1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0]     # m1, m2, l1, l2, q1f, q2f, j_max


@pytest.mark.slow
@pytest.mark.parametrize("tau, arcs", [((100.0, 100.0), ()),
                                       ((2.2, 2.0), ("tau1_max", "tau1_min")),
                                       ((1.5, 0.4), ("tau2_max", "tau1_min", "tau2_min"))])
def test_two_link_arm_in_minimum_time(tau, arcs, kernel_cache):
    h = Homotopy(examples.two_link_arm_min_time(), cache_dir=kernel_cache)
    r = h.solve(ARM + list(tau), ([-3] * 6, [3] * 6), (1.0, 20.0))
    assert r.exact and "secondary cost" in r.policy
    assert tuple(dict.fromkeys(name for _, name in r.arcs)) == arcs
    tr = r.shooting.trajectory(r.solution.v, n_points=4001)
    tau1, tau2 = examples.two_link_arm_torques(*ARM[:4], *tr["z"][:, 1:6].T)
    assert np.abs(tr["u"]).max() <= 1 + 1e-9
    assert np.abs(tau1).max() <= tau[0] + 1e-8 and np.abs(tau2).max() <= tau[1] + 1e-8
    # Jerk alone limits joint 1 to (32 q1f / j_max)^(1/3); torque limits only slow it.
    assert r.solution.tf > (32 * 1.5) ** (1 / 3)
    if not arcs:
        assert r.solution.tf == pytest.approx((32 * 1.5) ** (1 / 3), rel=1e-3)
