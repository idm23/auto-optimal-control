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
            assert r.details["residual"] < 1e-9
