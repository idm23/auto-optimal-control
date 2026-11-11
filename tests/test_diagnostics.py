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
