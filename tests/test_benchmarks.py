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


def test_orbit_census_cases_ladder_down_in_thrust_between_fixed_orbits():
    """The orbit census (benchmarks/orbit_census.py): each family's thrust falls down its
    ladder, a ``j`` case is its two-body case with the Earth's J2, the parameters start on
    the family's orbit and end on its target, and the search boxes hold the estimate."""
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from orbit_census import FAMILIES, case, case_names
    from orbitprop import EARTH, equinoctial_to_keplerian

    names = case_names()
    assert len(names) == len(set(names)) == 2 * sum(len(t) for _, t in FAMILIES.values())
    for fam, ((start, target), ladder) in FAMILIES.items():
        assert list(ladder) == sorted(ladder, reverse=True)
        for i in range(len(ladder)):
            two, j2 = case(f"{fam}{i}"), case(f"j{fam}{i}")
            assert two["family"] == j2["family"] == fam and two["T"] == ladder[i]
            assert two["params"][2] == 0 and j2["params"][2] == EARTH.J[2]
            assert two["params"][3:] == j2["params"][3:]
            a, e, inc, *_ = equinoctial_to_keplerian(two["params"][3:9])
            assert np.allclose([a, e, np.degrees(inc)], start)
            a, e, inc, *_ = equinoctial_to_keplerian([*two["params"][9:14], 0.0])
            assert np.allclose([a, e, np.degrees(inc)], target, atol=1e-12)
            lo, hi = two["final_time"]
            assert lo < two["tf_estimate"] < hi
    with pytest.raises(KeyError):
        case("jump3")


def test_orbit_export_draws_the_orbits_in_earth_radii():
    """The census page's geometry (benchmarks/orbit_export.py) agrees with orbitprop's:
    positions, the thrust frame, and the start and target orbits."""
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from orbit_export import frames, keplerian, orbit_curve, positions
    from orbitprop import equinoctial_to_cartesian, keplerian_to_equinoctial

    rng = np.random.default_rng(3)
    z = np.array([keplerian_to_equinoctial(rng.uniform(1.1, 7), rng.uniform(0, 0.7),
                                           rng.uniform(0, 1.2), *rng.uniform(0, 6, 3))
                  for _ in range(20)])
    r = positions(z)
    _, _, wh, ir, it = frames(z)
    for zi, ri, w, a, b in zip(z, r, wh, ir, it):
        rc, vc = equinoctial_to_cartesian(zi, 1.0)
        assert np.allclose(ri, rc)
        assert np.allclose(a, rc / np.linalg.norm(rc))
        assert np.allclose(w, np.cross(rc, vc) / np.linalg.norm(np.cross(rc, vc)))
        assert np.allclose(b, np.cross(w, a))
    a, e, _ = keplerian(z)
    assert np.allclose(a * (1 - e ** 2), z[:, 0])
    ring = orbit_curve(keplerian_to_equinoctial(2.0, 0, 0.3, 0, 0, 0)[:5])
    assert np.allclose(np.linalg.norm(ring, axis=1), 2.0)
