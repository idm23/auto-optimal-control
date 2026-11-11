"""The lemma objects match their closed forms, and the lemma oracles hold on solutions."""

import warnings

import numpy as np
import pytest

from auto_optimal_control import examples
from auto_optimal_control.homotopy import Homotopy
from auto_optimal_control.lemmas import LimitObjects, check
from auto_optimal_control.policy import SecondaryCost


@pytest.fixture(scope="module")
def axis(kernel_cache):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = Homotopy(examples.jerk_limited_move(), cache_dir=kernel_cache, policy=SecondaryCost(),
                     outline_above=0)
    h.check_lemmas = True
    r = h.solve([1.0, 0.3, 1.0, 2.0], ([-1.0] * 3, [1.0] * 3), (0.5, 20.0))
    return h, r


def test_the_objects_of_an_acceleration_limit_have_their_closed_forms(axis):
    # g = a - A: dg/dt = j, so a = 1 and d = 0; the box is |j| <= J = 1, so the
    # margin is J; the jerk cost 0.005 j^2 / J^2 has R = 0.01, so alpha = 0.01.
    _, r = axis
    sh = r.shooting
    obj = LimitObjects(sh.kernels, sh.params)
    c = list(sh.kernels.CONSTRAINT_NAMES).index("a_max")
    z = np.array([0.1, 0.2, 0.25, 0.3, -0.1, 0.05])
    a, d = obj.rate_parts(c, 0.0, z)
    assert np.allclose(a, [1.0]) and abs(d) < 1e-14
    assert abs(obj.margin(c, 0.0, z) - 1.0) < 1e-12
    assert abs(obj.alpha(c, 0.0, z) - 0.01) < 1e-12


def test_the_lemmas_hold_on_the_path_and_the_answer(axis):
    _, r = axis
    assert r.exact and ("boundary", "a_max") in r.arcs
    assert r.diagnostics["oracles"] == {}
    assert check(r.shooting, r.solution.v) == {"H": 0, "I": 0, "J": 0, "E": 0}


def test_the_oracles_catch_a_wrong_jump(axis):
    # A jump 20% off is no extremal: the arc's multiplier no longer leaves at zero,
    # so the exit is not tangential (Lemma J).
    _, r = axis
    sh, v = r.shooting, r.solution.v.copy()
    sl = sh._arc_slices[0]
    v[sl.stop - 1] *= 1.2
    bad = check(sh, v)
    assert bad["J"] + bad["E"] + bad["H"] > 0


@pytest.fixture(scope="module")
def drift(kernel_cache):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = Homotopy(examples.drift_limited_axis(), cache_dir=kernel_cache, policy=SecondaryCost(),
                     outline_above=0)
    h.check_lemmas = True
    return h


def test_the_drift_axis_has_a_velocity_dependent_margin(drift):
    # tau = a + B v^2: d tau/dt = j + 2 B v a, so a = 1, d = 2 B v a and the margin
    # is J - |2 B v a| (Lemma I): it closes at speed.
    r = drift.solve([1.0, 0.5, 0.5, 2.0], ([-1.0] * 3, [1.0] * 3), (0.5, 40.0))
    assert r.exact and ("boundary", "tau_max") in r.arcs
    sh = r.shooting
    obj = LimitObjects(sh.kernels, sh.params)
    c = list(sh.kernels.CONSTRAINT_NAMES).index("tau_max")
    z = np.array([0.3, 0.8, 0.4, 0.1, 0.2, -0.3])
    a, d = obj.rate_parts(c, 0.0, z)
    assert np.allclose(a, [1.0]) and abs(d - 2 * 0.5 * 0.8 * 0.4) < 1e-12
    assert abs(obj.margin(c, 0.0, z) - (1.0 - 0.32)) < 1e-12
    assert r.diagnostics["oracles"] == {}
    assert check(sh, r.solution.v) == {"H": 0, "I": 0, "J": 0, "E": 0}


def test_the_flow_through_an_arc_is_smooth_through_zero_width(drift):
    # Rule R10 (Lemma E, step 1): riding for a width w then the free law to a fixed
    # time is the flows composed, backward for w < 0, so it is smooth through w = 0.
    from auto_optimal_control.shooting import MultiArcShooting
    r = drift.solve([1.0, 0.5, 0.5, 2.0], ([-1.0] * 3, [1.0] * 3), (0.5, 40.0))
    sh = r.shooting
    assert isinstance(sh, MultiArcShooting) and sh.backward_segments
    _, _, segs = sh._evaluate(np.asarray(r.solution.v), jac=False)
    _, c, t_in, _, z_in, _, _ = next(sg for sg in segs if sg[0] == "boundary")
    t1 = t_in + 0.05
    zero = np.zeros(1)

    def through(w):
        ride, _ = sh._segment("boundary", c, t_in, zero, z_in, None, t_in + w, zero, False)
        free, _ = sh._segment(None, None, t_in + w, zero, ride["z"], None, t1, zero, False,
                              flags={})
        return free["z"]

    h = 1e-5
    zs = {k: through(k * h) for k in (-2, -1, 1, 2)}
    left = (zs[-1] - zs[-2]) / h
    right = (zs[2] - zs[1]) / h
    middle = (zs[1] - zs[-1]) / (2 * h)
    assert np.allclose(middle, right, rtol=1e-3, atol=1e-7)
    assert np.allclose(middle, left, rtol=1e-3, atol=1e-7)


def test_an_answer_out_of_order_is_rejected(drift):
    r = drift.solve([1.0, 0.5, 0.5, 2.0], ([-1.0] * 3, [1.0] * 3), (0.5, 40.0))
    sh, sol = r.shooting, r.solution
    assert drift._verdict(sh, sol) is None
    assert sh.in_order(sol.v)
    bad = sol.__class__(**{**sol.__dict__, "v": sol.v.copy()})
    sl = sh._arc_slices[0]
    bad.v[sl.start + 1] = bad.v[sl.start] - 1e-3
    assert drift._verdict(sh, bad) == "junctions out of order"
