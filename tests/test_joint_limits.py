"""Rate and angle limits on jerk-controlled joints: orders 2 and 3, at run time too.

With jerk control a rate limit has order 2 (ridden through ``g'' = j``) and an
angle limit order 3. With a smooth control law (the jerk cost of the default
policy) an order-3 limit is only ever touched, never ridden (Jacobson, Lele &
Speyer 1971), so its arcs are touch points.
"""

import warnings

import numpy as np
import pytest

from auto_optimal_control import examples
from auto_optimal_control.homotopy import Homotopy
from auto_optimal_control.policy import SecondaryCost


@pytest.mark.parametrize("D", [2.0, 8.0])
def test_a_rate_limit_ridden_at_run_time_matches_its_symbolic_arc(kernel_cache, D):
    # The jerk-limited axis with the jerk cost: at D = 8 it rides v <= V (order 2).
    hs = [Homotopy(examples.jerk_limited_move(), cache_dir=kernel_cache, policy=SecondaryCost(),
                   **kw) for kw in ({}, {"outline_above": 0})]
    assert hs[0].kr.system.riding is None and hs[1].kr.system.riding is not None
    symbolic, run_time = (h.solve([1.0, 1.0, 1.0, D], ([-1.0] * 3, [1.0] * 3), (0.5, 20.0))
                          for h in hs)
    assert symbolic.exact and run_time.exact
    assert symbolic.arcs == run_time.arcs
    assert (("boundary", "v_max") in run_time.arcs) == (D > 4)
    assert abs(symbolic.solution.tf - run_time.solution.tf) < 1e-10
    assert np.max(np.abs(symbolic.solution.lam0 - run_time.solution.lam0)) < 1e-8


ARM = [1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.5, 0.4]      # m, l, q_f, j_max, tau_max
LOOSE_RATES, LOOSE_ANGLES = [100.0, 100.0], [-100.0, -100.0, 100.0, 100.0]


@pytest.fixture(scope="module")
def arm(kernel_cache):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # the default policy's notice
        return Homotopy(examples.two_link_arm_min_time(rate_limits=True, position_limits=True),
                        cache_dir=kernel_cache)


def solve(h, base, rates, angles):
    r = h.solve([*base, *rates, *angles], ([-3] * 6, [3] * 6), (1.0, 20.0), seed=0)
    assert r.exact, r.message
    return r, r.shooting.trajectory(r.solution.v, 4000)["z"]


def test_loose_joint_limits_change_nothing(arm):
    r, _ = solve(arm, ARM, LOOSE_RATES, LOOSE_ANGLES)
    assert abs(r.solution.tf - 4.164857446) < 1e-8
    assert all(c.startswith("tau") for _, c in r.arcs)


def test_an_arm_rides_its_rate_limit(arm):
    # The elbow's free peak rate is 0.5131; at 70% of it the elbow rides its limit twice.
    w_max = 0.7 * 0.5131
    r, z = solve(arm, ARM, [100.0, w_max], LOOSE_ANGLES)
    assert [a for a in r.arcs if a[1].startswith("w")] == [("enter", "w2_max"), ("exit", "w2_max")] * 2
    assert abs(np.abs(z[:, 3]).max() - w_max) < 1e-8
    assert r.solution.tf > 4.164857446


def test_an_overlap_needing_too_much_jerk_is_shortened(arm):
    # At 90% of the shoulder's peak rate the route reaches a stretch riding both torques
    # that needs jerk 1.51 against a bound of 1: the overlap must give way.
    w_max = 0.9 * 0.6979
    r, z = solve(arm, ARM, [w_max, 100.0], LOOSE_ANGLES)
    assert abs(r.solution.tf - 4.192579166) < 1e-6
    assert abs(np.abs(z[:, 2]).max() - w_max) < 1e-8


def test_an_arm_touches_its_angle_limit(arm):
    # Rest to rest with the elbow back at 0: it swings out to +-0.049 on the way.
    base = ARM[:5] + [0.0] + ARM[6:]
    r, z = solve(arm, base, LOOSE_RATES, LOOSE_ANGLES)
    swing = z[:, 1].max()
    assert 0.04 < swing < 0.06
    q_max = 0.5 * swing
    r2, z2 = solve(arm, base, LOOSE_RATES, [-100.0, -100.0, 100.0, q_max])
    assert [a for a in r2.arcs if a[1].startswith("q")] == [("touch", "q2_max")]
    assert abs(z2[:, 1].max() - q_max) < 1e-8
    assert r2.solution.tf > r.solution.tf
