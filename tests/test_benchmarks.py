"""Literature benchmarks (``benchmarks/``): each must match its published value.

Slow (about two minutes, mostly compiling and the cart-pole searches); run
with ``--runslow``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from problems import ALL  # noqa: E402


@pytest.mark.slow
@pytest.mark.parametrize("name", list(ALL))
def test_benchmark_matches_literature(name, kernel_cache):
    for r in ALL[name](cache_dir=kernel_cache):
        assert r.ok, f"{r.benchmark} {r.quantity}: {r.value!r} vs {r.reference!r} ({r.source})"
        if "residual" in r.details:
            # Relative to the size of the unknowns (costates in the thousands on the arm).
            assert r.details["residual"] < 1e-9 * r.details.get("unknowns scale", 1.0)


def test_position_limits_cap_only_the_overshoot():
    """``position_cases.limits``: a joint that swings past its start-to-goal range is capped
    at its pct of the way from the range to its peak; a side that does not overshoot gets
    a bound far away, and the start and goal always lie inside the limits."""
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from position_cases import LOOSE, limits

    t = np.linspace(0, 1, 201)[:, None]
    # joint 0 overshoots its goal 1.0 by 0.2; joint 1 moves monotonically from 0 to -1.
    q = np.hstack([t + 0.8 * np.sin(np.pi * t) ** 2 * t * (1 - t) * 4,
                   -t])
    over = q[:, 0].max() - 1.0
    q_min, q_max, binding = limits(q, 70)
    assert binding.tolist() == [True, False]
    assert np.isclose(q_max[0], 1.0 + 0.7 * over)
    assert np.isclose(q_min[0], 0.0 - LOOSE) and np.isclose(q_max[1], 0.0 + LOOSE)
    assert np.all(q_min <= np.minimum(q[0], q[-1])) and np.all(q_max >= np.maximum(q[0], q[-1]))


def test_position_cases_repeat_overshooting_answers_with_limits():
    """The census's position-limit cases (benchmarks/position_cases.json): each repeats a
    census case (its params unchanged) with q_min, q_max appended, rate limits kept for
    a rate case, and at least one binding joint."""
    import json

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "position_cases.json"
    if not path.exists():
        pytest.skip("no position cases generated")
    rates = json.loads((path.parent / "rate_cases.json").read_text())
    cases = json.loads(path.read_text())
    assert cases
    for name, case in cases.items():
        p = np.asarray(case["params"])
        assert name == "p" + case["base"] and case["joints"]
        n_base = 30 if case["base"] in rates else 24
        assert p.shape == (n_base + 12,)
        if case["base"] in rates:
            assert np.allclose(p[:30], rates[case["base"]]["params"])
        q0, qf = p[:6], p[6:12]
        q_min, q_max = p[n_base:n_base + 6], p[n_base + 6:]
        assert np.all(q_min < np.minimum(q0, qf)) and np.all(q_max > np.maximum(q0, qf))


def test_rate_cases_repeat_the_census_with_rates_capped():
    """The census's rate-limit cases (benchmarks/rate_cases.json): each census case
    once, its params (q0, qf, j_max, tau_max) unchanged and every joint's rate capped
    at its pct of the jerk-limited motion's peak rate."""
    import json

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from rate_cases import names

    cases = json.loads((Path(__file__).resolve().parents[1] / "benchmarks"
                        / "rate_cases.json").read_text())
    assert list(cases) == names() and len(cases) == 45
    for name, case in cases.items():
        p = np.asarray(case["params"])
        assert p.shape == (30,) and case["base"] == name[1:]
        assert case["pct"] in (90, 80, 70)
        if "@" in name:
            assert case["pct"] == int(name.split("@")[1])
        assert np.allclose(p[24:], case["pct"] / 100 * np.asarray(case["peak"]))
        assert np.all(p[24:] > 0)


def test_obstacle_cases_block_a_rest_answer_but_clear_its_ends():
    """The census's obstacle cases (benchmarks/obstacle_cases.json): each repeats a rest
    case (its params unchanged) with an obstacle's appended; the obstacle keeps every body
    point of the start and goal poses clear and is entered by the rest answer."""
    import json

    import numpy as np

    from auto_optimal_control.robot import SIX_JOINT_LINKS, obstacle_values

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "obstacle_cases.json"
    if not path.exists():
        pytest.skip("no obstacle cases generated")
    from obstacle_cases import CLEAR
    cases = json.loads(path.read_text())
    assert cases
    for name, case in cases.items():
        p = np.asarray(case["params"])
        kind, n = case["obstacle"], (4 if case["obstacle"] == "sphere" else 7)
        assert name in {k + case["base"][4:] for k in ("ograze", "oblock", "owall")}
        assert kind == ("box" if name.startswith("owall") else "sphere")
        assert p.shape == (24 + n,) and case["g_base"] > 0
        for q in (p[:6], p[6:12]):
            assert obstacle_values(SIX_JOINT_LINKS, q, kind, p[24:]).max() < CLEAR
