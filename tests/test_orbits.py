"""Low-thrust orbit transfers (auto_optimal_control.orbits) and derive() on large dynamics."""

import numpy as np
import pytest
import sympy as sp

from auto_optimal_control.orbits import low_thrust_transfer
from auto_optimal_control.pontryagin import SIMPLIFY_LIMIT_OPS, _simplified, derive


def test_simplify_is_bounded_by_the_expression_size():
    x = sp.Symbol("x", real=True)
    assert _simplified(sp.sin(x) ** 2 + sp.cos(x) ** 2) == 1
    big = sp.Add(*[sp.sin(x) ** 2 * sp.cos(k * x) + sp.cos(x) ** 2 * sp.cos(k * x) for k in range(1, 60)])
    assert sp.count_ops(big) > SIMPLIFY_LIMIT_OPS
    assert _simplified(big) is big


def test_a_transfer_in_equinoctial_elements_derives_and_thrusts_along_minus_b():
    """The two-body minimum-time transfer derives in seconds (simplify() on its rates took
    many minutes), and its thrust direction is the unit vector that minimizes H."""
    P = low_thrust_transfer(objective="time", zonal_degree=1)
    D = derive(P)
    u = list(P.controls)
    rng = np.random.default_rng(1)
    x = dict(zip(P.states, [1.2, 0.05, -0.02, 0.1, 0.03, 0.7, 0.9]))
    for _ in range(3):
        point = {**x, **dict(zip(D.costates, rng.normal(size=7))),
                 **dict(zip(P.params, [0.01, 3.7] + [0.0] * 11))}
        star = np.array([float(D.control_law[c].subs(point)) for c in u])
        assert np.linalg.norm(star) == pytest.approx(1, rel=1e-12)
        b = np.array([float(sp.diff(D.hamiltonian_full, c).subs(point)) for c in u])
        assert star == pytest.approx(-b / np.linalg.norm(b), rel=1e-10)


def test_the_fuel_transfers_direction_does_not_depend_on_the_throttle():
    """With a throttle the thrust's direction is still -b/|b| with the throttle, thrust and
    mass cancelled: the throttle's law (with c and lam_m in it) is not substituted into
    it, and the regularized problem (a Max and a Min in the throttle's law) derives."""
    from auto_optimal_control.homotopy import regularize

    P = low_thrust_transfer(objective="fuel", zonal_degree=1)
    for problem in (P, regularize(P)[0]):
        D = derive(problem)
        names = {str(s) for c in list(P.controls)[:3] for s in D.control_law[c].free_symbols}
        assert not names & {"delta", "c", "lam_m", "T_max", "eps_reg"}
        assert "lam_p" in names
        # and the throttle's own law is free of every control
        assert not D.control_law[P.controls[3]].free_symbols & set(P.controls)
