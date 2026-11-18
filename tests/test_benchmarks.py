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
