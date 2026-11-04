"""benchmarks/animate.py: recording a solve and rendering it (needs the viz group).

Slow (the direct method alone takes about 40 s); run with ``--runslow``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))


@pytest.mark.slow
def test_cart_pole_recording_and_animation(kernel_cache, tmp_path):
    pytest.importorskip("matplotlib")
    from visuals import cart_pole

    data = cart_pole.record(cache_dir=kernel_cache, n_random=6, verbose=False)
    # The recorded stages end where the benchmark does.
    assert data["direct_defect"][-1] < 1e-8
    assert data["polish_R"][-1] < 1e-10
    assert data["published_J"] == pytest.approx(58.80819, rel=2e-5)
    assert np.isfinite(data["polish_z"]).all()
    np.testing.assert_allclose(data["published_z"][-1], [1.0, np.pi, 0, 0], atol=1e-6)
    cart_pole.save(data, tmp_path / "rec.npz")
    again = cart_pole.load(tmp_path / "rec.npz")
    assert again.keys() == data.keys()
    out = cart_pole.animate(again, tmp_path / "cart_pole.gif", fps=4, dpi=40,
                            stills_dir=tmp_path)
    assert out.stat().st_size > 0
    for stage in ("direct", "polish", "search", "motion"):
        assert (tmp_path / f"cart_pole_{stage}.png").exists()
