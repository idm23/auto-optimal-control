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
