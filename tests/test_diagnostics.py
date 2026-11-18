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


def test_split_fraction_is_the_secant_root():
    """Rule R25: the step to a split is the secant root of the density at the time of
    its minimum (V2), aimed a tenth of the way past it."""
    import numpy as np
    from types import SimpleNamespace

    h = Homotopy.__new__(Homotopy)
    h.kr = SimpleNamespace(CONSTRAINT_NAMES=["g0", "g1"])
    h._detail = lambda *a: None
    t = np.linspace(0.0, 1.0, 11)
    start = [(1, t, np.full(11, 2.0), 0.0, 5.0)]
    # The density at t = 0.5 goes from 2 to -1 over the step; mu rises by 0.1 there.
    dip = np.full(11, 2.0)
    dip[5] = -1.0
    end = [(1, t, dip, 0.1, 5.0)]
    h._densities = lambda shoot, v: start if shoot == "start" else end
    # eta(0.5) = 2 + theta * (-1 - 2) = 0.1 * (-1): theta = 2.1 / 3.
    assert h._split_fraction("start", None, "end", None) == pytest.approx(0.7)
    # A rise below the threshold, or a step that starts past the zero, is left to R5.
    end[0] = (1, t, dip, 1e-4, 5.0)
    assert h._split_fraction("start", None, "end", None) is None
    end[0] = (1, t, dip, 0.1, 5.0)
    start[0] = (1, t, np.full(11, -0.5), 0.0, 5.0)
    assert h._split_fraction("start", None, "end", None) is None
    # A rise from a jump of mu (no negative density) is not a split.
    start[0] = (1, t, np.full(11, 2.0), 0.0, 5.0)
    end[0] = (1, t, np.full(11, 2.0), 0.1, 5.0)
    assert h._split_fraction("start", None, "end", None) is None


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


def test_rejected_exact_solution_is_kept_for_r21():
    """Rule R21 needs the last exact solution rejected and why; an accepted one or an
    unconverged one leaves nothing behind."""
    import types

    h = Homotopy.__new__(Homotopy)
    h.diagnostics = Diagnostics()
    h._last_rejected = None
    sol = types.SimpleNamespace(success=True, residual_norm=1e-12)
    h._verdict = lambda shoot, sol: "mu rises (Lemma H)"
    assert not h._accept("shoot", sol, 1e-10)
    assert h._last_rejected == ("shoot", sol, "mu rises (Lemma H)")
    h._last_rejected = None
    h._verdict = lambda shoot, sol: "not converged"
    assert not h._accept("shoot", types.SimpleNamespace(success=False, residual_norm=1.0), 1e-10)
    assert h._last_rejected is None
    h._verdict = lambda shoot, sol: None
    assert h._accept("shoot", sol, 1e-10)
    assert h._last_rejected is None


def test_split_gap_is_the_normal_form():
    """Rule R27: the split's gap is centred on the density's minimum, with half-width
    h, h^2 = -6 eta_min / eta'' (Lemma E3), sqrt(3) times where eta < 0."""
    import numpy as np

    t = np.linspace(0.0, 1.0, 2001)
    eta_min, curv, tm = -0.02, 8.0, 0.4
    eta = eta_min + 0.5 * curv * (t - tm) ** 2
    # mu(t) = int_t^1 eta, so that eta = -dmu/dt.
    mu = np.concatenate([[0.0], np.cumsum(0.5 * (eta[1:] + eta[:-1]) * np.diff(t))])
    mu = mu[-1] - mu
    a, b, level = Homotopy._split_gap(t, mu)
    assert np.interp(a, t, mu) == pytest.approx(level) and np.interp(b, t, mu) == pytest.approx(level)
    h = np.sqrt(-6 * eta_min / curv)
    assert (a + b) / 2 == pytest.approx(tm, abs=1e-6)
    assert (b - a) / 2 == pytest.approx(h, rel=1e-4)
    assert (b - a) / 2 == pytest.approx(np.sqrt(3) * np.sqrt(-2 * eta_min / curv), rel=1e-4)
    # A negative window between two jumps of eta (a held set, Lemma P): the gap is the
    # window, a little widened on each side.
    eta = np.where((t > 0.5) & (t < 0.52), -0.5, 4.0)
    mu = np.concatenate([[0.0], np.cumsum(0.5 * (eta[1:] + eta[:-1]) * np.diff(t))])
    mu = mu[-1] - mu
    a, b, _ = Homotopy._split_gap(t, mu)
    assert 0.497 < a < 0.5 and 0.52 < b < 0.523
    # The density nowhere negative: no split.
    assert Homotopy._split_gap(t, -np.cumsum(np.full_like(t, 1e-3))) is None


def test_a_split_is_taken_only_if_it_resolves_the_rise():
    """Rule R28: a split solution counts only with no rise left on the limit's arcs and
    no piece or gap of zero width (V4)."""
    import numpy as np

    h, shoot = _stub_structure([("boundary", 0), ("boundary", 0)],
                               [(0.2, 0.4, np.zeros(1)), (0.5, 0.7, np.zeros(1))], [1])
    h._split_limit = 0
    h._densities = lambda shoot, v: [(0, None, None, 0.0, 1.0), (0, None, None, 0.0, 1.0)]
    assert h._split_resolved(shoot, None)
    h._densities = lambda shoot, v: [(0, None, None, 1e-3, 1.0), (0, None, None, 0.0, 1.0)]
    assert not h._split_resolved(shoot, None)
    h._densities = lambda shoot, v: [(0, None, None, 0.0, 1.0)]
    h._arcs_of = lambda shoot, v: ([("boundary", 0), ("boundary", 0)],
                                   [(0.2, 0.4, np.zeros(1)), (0.4, 0.7, np.zeros(1))])
    assert not h._split_resolved(shoot, None)       # the gap closed


def _two_rising_arcs():
    """Two order-1 arcs whose multipliers both rise over the window [0.5, 0.52] (another
    limit held there, Lemma P): limit 0 with mu ~ 3 and a small relative rise, limit 1
    with mu ~ 1e-4 that dips below zero before the window."""
    import types

    import numpy as np

    def mu0(t):
        return 3 * (0.9 - t) / 0.8 + 0.1 * np.clip((t - 0.5) / 0.02, 0, 1) - 0.1

    def mu1(t):
        return 1e-4 * (0.8 - t) / 0.6 + 2e-4 * np.clip((t - 0.5) / 0.02, 0, 1) - 2e-4

    arcs = [("boundary", 0), ("boundary", 1)]
    values = [(0.1, 0.9, np.array([mu0(0.1)])), (0.2, 0.8, np.array([mu1(0.2)]))]
    h, shoot = _stub_structure(arcs, values, [1, 1], T=1.0)
    h.kr.CONSTRAINT_NAMES = ["a_min", "b_min"]
    h._detail = lambda *a: None
    h.trim_rising = h.exit_before_saturation = h.split_normal_form = False
    t = np.linspace(0.0, 1.0, 801)
    shoot.params = None
    shoot.kernels = types.SimpleNamespace(
        N_CONSTRAINTS=2, arc_constraints=lambda aid: (0, 1),
        boundary_multiplier=lambda aid, t_, z, p: np.array([mu0(t_), mu1(t_)]))
    shoot.trajectory = lambda v, n: {
        "t": t, "z": t, "arc": np.full(len(t), 2),
        "active": np.stack([(t >= 0.1) & (t <= 0.9), (t >= 0.2) & (t <= 0.8)], axis=1)}
    return h, shoot, values


def test_r5_splits_the_largest_relative_rise_under_r33():
    """Rule R33: scaling a limit by k scales its multiplier by 1/k, so R5 compares rises
    relative to each arc's largest multiplier. Without the rule the large multiplier's
    small rise is split; with it, the arc whose multiplier dips below zero."""
    h, shoot, _ = _two_rising_arcs()
    h.relative_rise = False
    h._split_rising(shoot, None)
    assert h._split_limit == 0
    h.relative_rise = True
    out = h._split_rising(shoot, None)
    assert h._split_limit == 1
    assert out[0][0] == [("boundary", 0), ("boundary", 1), ("boundary", 1)]
    assert h.diagnostics.rules.get("R33") == 1


def test_wider_gaps_seeded_by_continuity_under_r34():
    """Rule R34: after R5's gaps, the split is offered with the first piece keeping the
    arc's entry jump, and the gap wider by half and twice the rise's stretch."""
    import numpy as np

    h, shoot, values = _two_rising_arcs()
    h.relative_rise = True
    h.wide_gaps = False
    plain = h._split_rising(shoot, None)
    h.wide_gaps = True
    out = h._split_rising(shoot, None)
    assert len(out) == len(plain) + 2
    gaps = []
    for arcs, vals in out[len(plain):]:
        assert arcs == [("boundary", 0), ("boundary", 1), ("boundary", 1)]
        (a1, b1, p1), (a2, b2, _) = vals[1], vals[2]
        assert a1 == values[1][0] and b2 == values[1][1]
        assert p1.tolist() == values[1][2].tolist()      # continuity: the entry jump kept
        gaps.append((b1, a2))
    # Widened by half and by twice the rise's stretch w on each side: 2w and 5w wide,
    # about the same stretch, which lies in the window.
    (lo1, hi1), (lo2, hi2) = gaps
    assert (hi2 - lo2) / (hi1 - lo1) == pytest.approx(2.5, rel=1e-9)
    assert (lo1 + hi1) / 2 == pytest.approx((lo2 + hi2) / 2)
    assert lo2 < lo1 < 0.5 < 0.52 < hi1 < hi2
    assert h.diagnostics.rules.get("R34") == 2
    assert np.all(np.isfinite([lo1, hi1, lo2, hi2]))


def test_an_order1_arc_with_a_negative_entry_multiplier_goes_under_r31():
    """Rule R31: 0 <= mu <= pi on an order-1 arc (Lemma H), so an arc whose entry
    multiplier pi is negative is retired; without the rule it stays."""
    import numpy as np

    arcs = [("boundary", 0), ("boundary", 1)]
    values = [(0.1, 0.4, np.array([0.015])), (0.2, 0.3, np.array([-5e-4]))]
    h, shoot = _stub_structure(arcs, values, [1, 1])
    h.retire_negative = False
    assert h._retire(shoot, None) is None
    h.retire_negative = True
    assert h._retire(shoot, None) == ([("boundary", 0)], [values[0]])
    assert h.diagnostics.rules.get("R31") == 1
    # A jump at rounding level is not a death.
    values[1] = (0.2, 0.3, np.array([-1e-12]))
    h, shoot = _stub_structure(arcs, values, [1, 1])
    h.retire_negative = True
    assert h._retire(shoot, None) is None


def test_the_order_monitor_measures_how_far_junctions_crossed():
    """Rule R30: the largest step back between consecutive junctions, relative to T;
    zero when they are in order."""
    import numpy as np

    h, shoot = _stub_structure([("boundary", 0), ("boundary", 1)], [], [1, 1], T=2.0)
    shoot.arcs = [("boundary", 0, None), ("boundary", 1, None)]
    shoot.event_times = lambda v: np.array([0.2, 0.5, 0.9, 1.1])
    assert Homotopy._order_crossing(shoot, None) == 0.0
    shoot.event_times = lambda v: np.array([0.2, 0.9, 0.86, 1.1])
    assert Homotopy._order_crossing(shoot, None) == pytest.approx(0.04 / 2.0)


def test_the_exact_check_sees_violations_below_the_route_threshold():
    """Rule R32: the route counts a violation from 1e-6 of the limit's scale; the
    exact check from 1e-7. ``floor`` lowers the route's threshold to the exact one."""
    import numpy as np

    h, shoot = _stub_structure([], [], [1, 1])
    h.nc = 2
    t = np.linspace(0.0, 1.0, 11)
    g = np.full((11, 2), -1.0)
    g[5, 1] = 3e-7                       # above the exact check, below the route's
    h._free_samples = lambda shoot, v: [(t, g, set())]
    scale = np.ones(2)
    assert h._free_violations(shoot, None, scale) == []
    (c, a, b, peak), = h._free_violations(shoot, None, scale, floor=1e-7)
    assert (c, a, b) == (1, t[4], t[6]) and peak == pytest.approx(3e-7)


def test_touches_become_arcs_from_a_structure():
    """Rule R20's conversion, from a structure rather than a shooting (R32 uses it on
    the births it offers)."""
    import numpy as np

    h, _ = _stub_structure([], [], [1, 2])
    arcs = [("boundary", 0), ("touch", 0), ("touch", 1)]
    values = [(0.1, 0.3, np.array([0.2])), (0.5, 1e-5), (0.7, 0.1)]
    (a1, v1), = h._touches_as_arcs(arcs, values, 2.0, widths=(1e-2,))
    assert a1 == [("boundary", 0), ("boundary", 0), ("touch", 1)]
    assert v1[1][0] == pytest.approx(0.49) and v1[1][1] == pytest.approx(0.51)
    assert v1[1][2].tolist() == [1e-5] and v1[2] == (0.7, 0.1)
    assert h._touches_as_arcs(arcs[:1] + arcs[2:], values[:1] + values[2:], 2.0) == []


def test_the_solve_budget_covers_the_reversal():
    """``Homotopy.solve_seconds`` bounds a whole solve: the route gets at most the
    budget, the reversed problem (R4) only what is left of it, and none once it is
    spent; the route's own budget is restored afterwards."""
    import time
    import types

    h = Homotopy.__new__(Homotopy)
    h.stated = types.SimpleNamespace(params=[None])
    h._stages, h.reversal, h.policy_text = None, True, ""
    h.active_set_seconds, h.solve_seconds, h._solve_end = 300.0, 0.2, None
    seen = []

    def slow(*a, **k):
        seen.append(h.active_set_seconds)
        time.sleep(0.25)
        return types.SimpleNamespace(exact=False)

    h._solve = slow
    h._through_reversal = lambda *a, **k: seen.append(("reversal", h.active_set_seconds))
    h.solve([1.0], None)
    assert seen == [0.2]
    assert h.active_set_seconds == 300.0 and h._solve_end is None

    seen.clear()
    h.solve_seconds = 5.0
    h._solve = lambda *a, **k: (seen.append(h.active_set_seconds), types.SimpleNamespace(exact=False))[1]
    h.solve([1.0], None)
    assert seen[0] == 5.0 and seen[1][0] == "reversal" and 4.0 < seen[1][1] <= 5.0
    assert h.active_set_seconds == 300.0


def test_a_convex_order2_touch_becomes_an_arc_under_r35(monkeypatch):
    """Rule R35: a touch is a local maximum of its limit, so g'' <= 0 there. A touch
    where g' rises through zero (g'' > 0) becomes a short arc about it, clear of its
    neighbours' events, the touch's jump as the arc's entry jump; a concave one stays."""
    import types

    import numpy as np

    from auto_optimal_control import lemmas

    t = np.linspace(0.0, 1.0, 4001)

    def setup(curv):
        arcs = [("touch", 0), ("touch", 1)]
        values = [(0.5, 0.2), (0.5004, 0.1)]
        h, shoot = _stub_structure(arcs, values, [2, 2], T=1.0)
        h.touch_curvature = True
        shoot.kernels, shoot.params = None, None
        shoot.trajectory = lambda v, n_points=4001: {
            "t": t, "z": t[:, None], "u": np.zeros((len(t), 1))}

        class Objects:                     # g_c' = curv_c (t - t_touch)
            def __init__(self, k, p): pass

            def rate_parts(self, c, t_, z):
                return np.zeros(1), curv[c] * (t_ - values[c][0])

        monkeypatch.setattr(lemmas, "LimitObjects", Objects)
        return h, shoot, values

    h, shoot, values = setup([2.0, -2.0])
    assert set(h._touch_curvatures(shoot, None)) == {0}
    out_a, out_v = h._retire(shoot, None)
    assert out_a == [("boundary", 0), ("touch", 1)]
    lo, hi, pi = out_v[0]
    assert (lo + hi) / 2 == pytest.approx(0.5) and hi - lo == pytest.approx(2 * 0.25 * 0.0004)
    assert pi.tolist() == [0.2, 0.0]
    assert h.diagnostics.rules.get("R35") == 1
    h, shoot, _ = setup([-2.0, -2.0])
    assert h._touch_curvatures(shoot, None) == {} and h._retire(shoot, None) is None
