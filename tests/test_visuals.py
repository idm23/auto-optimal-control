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
    for stage in ("problem", "direct", "polish", "search", "motion"):
        assert (tmp_path / f"cart_pole_{stage}.png").exists()


@pytest.mark.slow
def test_earth_mars_recording_and_animation(kernel_cache, tmp_path):
    pytest.importorskip("matplotlib")
    from visuals import earth_mars

    data = earth_mars.record(cache_dir=kernel_cache, verbose=False)
    assert data["shoot_R"][-1] < 1e-9
    assert data["cont_eps"][0] == 1.0 and data["cont_eps"][-1] == pytest.approx(1e-4)
    assert np.all(np.diff(data["cont_m"][:, -1]) > -1e-6)       # mass rises along the path
    assert data["fuel_m"][-1] == pytest.approx(603.935, rel=2e-5)
    np.testing.assert_allclose(data["switch_t"] * earth_mars.TU / 86400,
                               [46.6, 68.0, 142.7, 290.3], atol=0.1)
    np.testing.assert_allclose(data["fuel_r"][-1], data["mars"][-1], atol=1e-8)
    out = earth_mars.animate(data, tmp_path / "earth_mars.gif", fps=3, dpi=40,
                             stills_dir=tmp_path)
    assert out.stat().st_size > 0
    for stage in ("problem", "shoot", "cont", "flight"):
        assert (tmp_path / f"earth_mars_{stage}.png").exists()


@pytest.mark.slow
def test_bryson_denham_recording_and_animation(kernel_cache, tmp_path):
    pytest.importorskip("matplotlib")
    from visuals import bryson_denham

    data = bryson_denham.record(cache_dir=kernel_cache, verbose=False)
    assert data["touch_cost"] == pytest.approx(2.24, rel=1e-12)
    assert data["arc_cost"] == pytest.approx(4 / 0.9, rel=1e-12)
    np.testing.assert_allclose(data["arc_junctions"], [0.3, 0.7], atol=1e-6)
    assert np.all(np.diff(data["touch_pen_J"]) > -1e-9)     # the penalty cost climbs
    assert data["touch_exact_R"][-1] < 1e-10 and data["arc_exact_R"][-1] < 1e-10
    out = bryson_denham.animate(data, tmp_path / "bd.gif", fps=3, dpi=40, stills_dir=tmp_path)
    assert out.stat().st_size > 0
    for stage in ("problem", "touch_pen", "touch_exact", "arc_pen", "arc_exact", "motion"):
        assert (tmp_path / f"bryson_denham_{stage}.png").exists()


@pytest.mark.slow
def test_velocity_limit_recording_and_animation(kernel_cache, tmp_path):
    pytest.importorskip("matplotlib")
    from visuals import velocity_limit

    data = velocity_limit.record(cache_dir=kernel_cache, verbose=False)
    assert data["cost"] == pytest.approx(data["J_exact"], rel=1e-10)
    assert data["exact_R"][-1] < 1e-10
    assert np.abs(data["exact_u"][-1]).max() <= 8.0 + 1e-9          # the force limit holds
    np.testing.assert_allclose(data["exact_junctions"][-1], [data["t1"], 1 - data["t1"]],
                               atol=1e-7)
    out = velocity_limit.animate(data, tmp_path / "vl.gif", fps=3, dpi=40, stills_dir=tmp_path)
    assert out.stat().st_size > 0
    for stage in ("problem", "pen", "exact", "motion"):
        assert (tmp_path / f"velocity_limit_{stage}.png").exists()


@pytest.mark.slow
def test_speed_limit_recording_and_animation(kernel_cache, tmp_path):
    pytest.importorskip("matplotlib")
    from visuals import speed_limit

    data = speed_limit.record(cache_dir=kernel_cache, verbose=False)
    assert data["tf"] == pytest.approx(4.0, abs=1e-12)
    np.testing.assert_allclose(data["junctions"], [1.0, 3.0], atol=1e-12)
    assert data["eps"][0] == 1.0 and data["eps"][-1] == pytest.approx(1e-4)
    assert np.all(np.diff(data["eps_tf"]) < 0)                      # t_f falls toward 4
    assert data["bang_R"][-1] < 1e-12
    out = speed_limit.animate(data, tmp_path / "sl.gif", fps=3, dpi=40, stills_dir=tmp_path)
    assert out.stat().st_size > 0
    for stage in ("problem", "pen", "eps", "bang", "motion"):
        assert (tmp_path / f"speed_limit_{stage}.png").exists()


def test_every_suite_states_its_problem():
    pytest.importorskip("matplotlib")
    from animate import SUITES

    for name, suite in SUITES.items():
        problem = suite.PROBLEM
        assert problem.name and problem.story and problem.summary, name
        assert problem.formulation and problem.solved_by, name
