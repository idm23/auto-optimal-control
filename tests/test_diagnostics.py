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
