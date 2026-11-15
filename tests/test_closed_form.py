"""Free arcs of integrator chains in closed form (rule R23, Lemma T)."""

import warnings

import numpy as np
import pytest

from auto_optimal_control import examples
from auto_optimal_control.build import build_kernels
from auto_optimal_control.closed_form import IntegratorChains
from auto_optimal_control.homotopy import Homotopy
from auto_optimal_control.policy import SecondaryCost, analyze
from auto_optimal_control.pontryagin import derive
from auto_optimal_control.shooting import MultiArcShooting


@pytest.fixture(scope="module")
def s_curve(kernel_cache):
    stated = examples.jerk_limited_move()
    problem = SecondaryCost(0.01).apply(stated, analyze(stated))
    return build_kernels(derive(problem), name="s_curve_clipped", cache_dir=kernel_cache)


@pytest.fixture(scope="module")
def two_link(kernel_cache):
    stated = examples.two_link_arm_min_time()
    problem = SecondaryCost(0.01).apply(stated, analyze(stated))
    return build_kernels(derive(problem, outline_above=0, simultaneous=2),
                         name="clipped_numeric", cache_dir=kernel_cache)


def _agree(k, p, scales, n_trials, seed):
    chains, why = IntegratorChains.certify(k.system)
    assert chains is not None, why
    rng = np.random.default_rng(seed)
    worst = np.zeros(4)
    for _ in range(n_trials):
        z0 = rng.uniform(-1, 1, len(scales)) * scales
        tau = rng.choice([-1, 1]) * rng.uniform(0.05, 1.0)
        r = k.propagate(0.0, z0, tau, p, stm=True, cost=True, rtol=1e-13, atol=1e-13,
                        constraint_events=False)
        c = chains.propagate(0.0, z0, tau, p, stm=True, cost=True)
        worst = np.maximum(worst, [np.abs(c["z"] - r["z"]).max(), np.abs(c["stm"] - r["stm"]).max(),
                                   np.abs(c["dz_dt1"] - r["dz_dt1"]).max(), abs(c["cost"] - r["cost"])])
    return worst


def test_one_axis_agrees_with_the_integration_both_ways(s_curve):
    """The S-curve axis: one chain of three. The closed form and the integrated arc
    agree to the integration's own error, forward and backward, saturated or not."""
    p = np.array([1.0, 0.3, 1.0, 2.0] + [0.0] * (len(s_curve.system.params) - 4))
    worst = _agree(s_curve, p, np.array([1, 1, 1, 1, 2, 0.05]), 200, 0)
    assert worst[0] < 1e-12 and worst[1] < 1e-9 and worst[2] < 1e-11 and worst[3] < 1e-12


def test_two_link_arm_agrees_with_the_integration(two_link):
    """Two chains with their torque limits beside: the free arc ignores the limits."""
    k = two_link
    chains, _ = IntegratorChains.certify(k.system)
    assert sorted(map(sorted, chains.chains.tolist())) == [[0, 2, 4], [1, 3, 5]]
    p = np.array([1, 1, 1, 1, 1.5, 1, 1.0, 2.2, 2.0] + [0.0] * (len(k.system.params) - 9))
    worst = _agree(k, p, np.array([1] * 6 + [1, 1, 2, 2, 0.05, 0.05]), 100, 1)
    assert worst[0] < 1e-11 and worst[1] < 1e-8 and worst[2] < 1e-10


def test_the_compiled_flow_is_the_numpy_flow(two_link, kernel_cache):
    """The C flow and the NumPy flow are the same formulas: they agree to round-off."""
    chains, _ = IntegratorChains.certify(two_link.system)
    ref, _ = IntegratorChains.certify(two_link.system)
    assert chains.use_compiled(kernel_cache)
    p = np.array([1, 1, 1, 1, 1.5, 1, 1.0, 2.2, 2.0] + [0.0] * (len(two_link.system.params) - 9))
    rng = np.random.default_rng(5)
    for _ in range(100):
        z0 = rng.uniform(-1, 1, 12) * np.array([1] * 6 + [1, 1, 2, 2, 0.05, 0.05])
        tau = rng.choice([-1, 1]) * rng.uniform(0.01, 1.0)
        a = chains.propagate(0.0, z0, tau, p, stm=True, cost=True)
        b = ref.propagate(0.0, z0, tau, p, stm=True, cost=True)
        for key in ("z", "stm", "dz_dt1"):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-13)
        assert abs(a["cost"] - b["cost"]) < 1e-12


def test_the_certificate_refuses_what_it_does_not_cover(compiled):
    """Bang-bang jerks (H linear in u), and dynamics that are not chains."""
    assert IntegratorChains.certify(compiled["jerk_limited_move"][0])[0] is None
    for name in ("min_time", "zermelo"):
        assert IntegratorChains.certify(compiled[name][0])[0] is None


def test_a_solve_is_the_same_with_closed_free_arcs(kernel_cache):
    """The S-curve with both acceleration limits, solved both ways."""
    out = []
    for on in (False, True):
        before = MultiArcShooting.closed_free_default
        MultiArcShooting.closed_free_default = on
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                h = Homotopy(examples.jerk_limited_move(), cache_dir=kernel_cache,
                             policy=SecondaryCost(), outline_above=0)
                r = h.solve([1.0, 0.3, 1.0, 2.0], ([-1.0] * 3, [1.0] * 3), (0.5, 20.0))
        finally:
            MultiArcShooting.closed_free_default = before
        assert r.exact and (r.shooting._chains is not None) == on
        out.append((float(r.solution.tf), r.arcs))
    assert out[0][1] == out[1][1]
    assert abs(out[0][0] - out[1][0]) < 1e-10
