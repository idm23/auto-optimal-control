"""The direct transcription (comparison only) on a problem with a known answer."""

import numpy as np
import pytest
import sympy as sp

pytest.importorskip("casadi")

from auto_optimal_control import direct  # noqa: E402
from auto_optimal_control.problem import Box, OptimalControlProblem, PathConstraint  # noqa: E402


def double_integrator(speed_limit=None):
    t, x, v, u = sp.symbols("t x v u", real=True)
    limits = [PathConstraint(v - speed_limit, "v_max")] if speed_limit else []
    return OptimalControlProblem(time=t, states=[x, v], controls=[u], dynamics=[v, u],
                                 running_cost=1, initial_state=[0, 0], final_state={x: 1, v: 0},
                                 control_sets=[Box(u, -1, 1)], path_constraints=limits)


def test_minimum_time_double_integrator_is_bang_bang():
    sol = direct.solve(double_integrator(), [], intervals=40, tf_guess=3.0)
    assert sol.success
    assert sol.tf == pytest.approx(2.0, abs=1e-6)
    held = sol.held(1e-6)[0]
    assert [s for _, _, s in held] == [1, -1]
    assert held[0][1] == pytest.approx(1.0, abs=1e-6)


def test_speed_limit_gives_an_arc_and_a_nonnegative_multiplier():
    # Accelerate to v = 0.5, cruise, brake: tf = 0.5 + 1.5 + 0.5 = 2.5 for distance 1.
    sol = direct.solve(double_integrator(0.5), [], intervals=50, tf_guess=3.0)
    assert sol.success
    assert sol.tf == pytest.approx(2.5, abs=1e-6)
    (a, b), = sol.arcs(1e-6)["v_max"]
    assert a == pytest.approx(0.5, abs=0.06) and b == pytest.approx(2.0, abs=0.06)
    assert np.all(sol.path_multipliers >= -1e-8)
