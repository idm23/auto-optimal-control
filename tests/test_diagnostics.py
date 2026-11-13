"""The solver's diagnostics: failures by class, step outcomes, and snapshots that replay."""

import warnings

import pytest

from auto_optimal_control import examples
from auto_optimal_control.diagnostics import Diagnostics, load_snapshot, replay
from auto_optimal_control.homotopy import Homotopy
from auto_optimal_control.policy import SecondaryCost

CLASSES = {"negative_width", "riding_failed", "free_failed", "singular_failed",
           "bad_final_time", "violated", "other"}


@pytest.fixture(scope="module")
def axis(kernel_cache, tmp_path_factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = Homotopy(examples.jerk_limited_move(), cache_dir=kernel_cache, policy=SecondaryCost(),
                     outline_above=0)
    h.snapshot_dir = str(tmp_path_factory.mktemp("snapshots"))
    h.snapshot_tag = "axis"
    r = h.solve([1.0, 1.0, 1.0, 8.0], ([-1.0] * 3, [1.0] * 3), (0.5, 20.0))
    return h, r


def test_a_solve_reports_what_it_ran_into(axis):
    _, r = axis
    assert r.exact
    d = r.diagnostics
    assert set(d) == {"evaluations", "steps", "rejected", "rules", "oracles", "snapshots"}
    assert d["steps"].get("converged", 0) > 0
    assert set(d["evaluations"]) <= CLASSES


def test_a_snapshot_rebuilds_its_step(axis):
    h, r = axis
    paths = r.diagnostics["snapshots"]
    assert paths
    for path in paths:
        snap = load_snapshot(path)
        assert snap["failure"] and snap["tag"] == "axis"
        sh, sol, failures = replay(h, snap)
        assert set(failures) <= CLASSES
        if snap["failure"] == "settle failed":
            # The step's solve converged; it was the structure that had to change.
            assert sol.success


def test_snapshots_are_capped_per_class(tmp_path):
    d = Diagnostics()
    paths = [d.snapshot(str(tmp_path), "t", "seed failed: riding_failed", {"x": i}, per_class=2)
             for i in range(4)]
    assert sum(p is not None for p in paths) == 2
    assert len(d.snapshots) == 2


def test_every_rule_can_be_switched(axis):
    from auto_optimal_control.rules import RULES, enabled, set_rule
    h, _ = axis
    before = enabled(h)
    try:
        for r in RULES:
            if r.switch == "always":
                assert enabled(h)[r.name]
                continue
            set_rule(h, r.name, True)
            assert enabled(h)[r.name]
            set_rule(h, r.name, False)
            assert not enabled(h)[r.name]
    finally:
        for name, on in before.items():
            set_rule(h, name, on)
    assert enabled(h) == before


def test_birth_fraction_is_the_secant_root():
    """Rule R16: the step to the birth is the secant root of the peak excess (Lemma N)."""
    import numpy as np

    h = Homotopy.__new__(Homotopy)
    g = np.array([[-0.5, -1.0], [-0.1, -2.0], [-0.3, -1.0]])
    h._free_samples = lambda shoot, v: [(np.arange(3.0), g, ())]
    scale = np.array([1.0, 1.0])
    # Limit 0 goes from peak -0.1 to +0.9 over the step: the target 2e-4 is reached
    # at (2e-4 + 0.1) / 1.0 of it.
    assert h._birth_fraction(None, None, {0: 0.9}, scale) == pytest.approx(0.1002)
    # The smallest fraction over the limits that overshot, clipped to [0.05, 0.9].
    assert h._birth_fraction(None, None, {0: 0.9, 1: 1e6}, scale) == pytest.approx(0.05)
    assert h._birth_fraction(None, None, {}, scale) is None


def _stub_structure(arcs, values, orders, T=2.0):
    """A Homotopy and a stand-in multi-arc shooting whose structure is ``(arcs, values)``."""
    import types

    import numpy as np

    from auto_optimal_control.homotopy import MultiArcShooting

    h = Homotopy.__new__(Homotopy)
    h.kr = types.SimpleNamespace(CONSTRAINT_ORDERS=np.asarray(orders))
    h.diagnostics = Diagnostics()
    h._arcs_of = lambda shoot, v: (list(arcs), list(values))
    shoot = MultiArcShooting.__new__(MultiArcShooting)
    shoot.t0 = 0.0
    shoot.unpack = lambda v: (None, None, None, T)
    shoot.in_order = lambda v: True
    return h, shoot


def test_collapsed_order1_arc_becomes_a_touch_only_under_r20():
    """Rule R20: an order-1 arc of zero width that still carries a jump is kept as a
    touch point; without the rule, or with no jump, it goes."""
    import numpy as np

    arcs = [("boundary", 0), ("boundary", 1)]
    values = [(0.1, 0.4, np.array([0.015])), (0.6312, 0.6312, np.array([2.7e-6]))]
    h, shoot = _stub_structure(arcs, values, [1, 1])
    h.collapsed_touches = False
    assert h._retire(shoot, None) == ([("boundary", 0)], [values[0]])
    h.collapsed_touches = True
    out_a, out_v = h._retire(shoot, None)
    assert out_a == [("boundary", 0), ("touch", 1)]
    assert out_v[1] == pytest.approx((0.6312, 2.7e-6))
    assert h.diagnostics.rules.get("R20 collapse") == 1
    # A collapsed arc whose jump is zero is gone, as before.
    values[1] = (0.6312, 0.6312, np.array([0.0]))
    h, shoot = _stub_structure(arcs, values, [1, 1])
    h.collapsed_touches = True
    assert h._retire(shoot, None) == ([("boundary", 0)], [values[0]])


def test_order1_touches_become_short_arcs_for_the_exact_solve():
    """Rule R20: the exact problem gets each order-1 touch as an arc centred on it,
    carrying its jump; touches of higher order stay touches."""
    import numpy as np

    arcs = [("boundary", 0), ("touch", 1), ("touch", 2)]
    values = [(0.1, 0.4, np.array([0.015])), (0.63, 3e-6), (1.2, 0.5)]
    h, shoot = _stub_structure(arcs, values, [1, 1, 2], T=1.5)
    out = h._touch_arcs(shoot, None, widths=(1e-3, 1e-2))
    assert len(out) == 2
    for (new_a, new_v), rel in zip(out, (1e-3, 1e-2)):
        assert new_a == [("boundary", 0), ("boundary", 1), ("touch", 2)]
        lo, hi, pi = new_v[1]
        assert (lo + hi) / 2 == pytest.approx(0.63)
        assert hi - lo == pytest.approx(rel * 1.5)
        assert pi.tolist() == [3e-6]
        assert new_v[2] == (1.2, 0.5)
    # Nothing to do without an order-1 touch.
    h, shoot = _stub_structure(arcs[:1] + arcs[2:], values[:1] + values[2:], [1, 1, 2])
    assert h._touch_arcs(shoot, None) == []
