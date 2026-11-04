"""Singular arcs: derivation, kernels, and exact shooting against closed forms.

- The singular regulator, ``min int x^2``, ``x' = u``, ``|u| <= 1``, from
  ``x(0) = a`` back to ``x(T) = a``: ``u = -1`` to ``t = a``, the singular
  arc ``x = 0`` to ``T - a``, then ``u = +1``. ``lambda(0) = a^2``,
  ``nu = -a^2``, cost ``2 a^3 / 3``.
- Goddard's rocket: full thrust, singular, coast. ``h(t_f) = 1.01283692``
  at ``t_f = 0.19885626`` (Graichen & Petit 2008, collocation with a
  vanishing penalty).
"""

import numpy as np
import pytest
import sympy as sp

from auto_optimal_control import examples
from auto_optimal_control.pontryagin import derive
from auto_optimal_control.problem import Box, OptimalControlProblem
from auto_optimal_control.shooting import (MultiArcShooting, SingleShooting,
                                           parameter_continuation)

GODDARD = list(examples.GODDARD_PARAMS)


def fd_jacobian(f, v, h=1e-7):
    return np.array([(f(v + h * e) - f(v - h * e)) / (2 * h) for e in np.eye(len(v))]).T


@pytest.fixture(scope="module")
def reg(compiled):
    return compiled["singular_regulator"][2]


@pytest.fixture(scope="module")
def god(compiled):
    return compiled["goddard"][2]


# -- derivation ---------------------------------------------------------------

def test_regulator_singular_arc():
    s = derive(examples.singular_regulator(), singular=True)
    (arc,) = s.singular_arcs
    x, lam, u = s.states[0], s.costates[0], s.controls[0]
    assert arc.derivatives == (lam, -2 * x, -2 * u)
    assert arc.law == {u: 0}
    assert arc.legendre_clebsch == 2
    assert arc.dynamics == (0, -2 * x)
    assert arc.junction == (lam, -2 * x)


def test_goddard_singular_control_keeps_the_junction_conditions():
    """Along F(u_s), d/dt (S, S') = (S', S''(u_s)) = (S', 0) at any point."""
    s = derive(examples.goddard(), singular=True)
    (arc,) = s.singular_arcs
    z = s.z
    rng = np.random.default_rng(1)
    params = dict(zip(s.params, GODDARD))
    for _ in range(3):
        point = dict(zip(z, [1.01, 0.1, 0.8, -1.0, -0.3, -0.5] + rng.normal(0, 0.05, 6)))
        point.update(params)
        S, S1 = arc.junction
        F = [e.subs(point) for e in arc.dynamics]
        d0 = sum(sp.diff(S, zi).subs(point) * fi for zi, fi in zip(z, F))
        d1 = sum(sp.diff(S1, zi).subs(point) * fi for zi, fi in zip(z, F))
        assert float(d0) == pytest.approx(float(S1.subs(point)), rel=1e-12)
        assert abs(float(d1)) < 1e-9


def test_singular_arcs_are_opt_in():
    s = derive(examples.singular_regulator())
    assert s.singular_arcs == (None,)
    assert "singular=True" in s.singular_unsupported[0]


def test_order_two_is_unsupported():
    # Minimum-time double integrator: S = lam_v, S' = -lam_x, S'' = 0.
    s = derive(examples.double_integrator_min_time(), singular=True)
    assert s.singular_arcs == (None,)
    assert "order 2" in s.singular_unsupported[0]


def test_unit_vector_is_unsupported():
    s = derive(examples.planar_low_thrust_fuel(), singular=True)
    assert s.singular_arcs == (None,)
    assert "unit-vector" in s.singular_unsupported[0]


def test_other_bounded_controls_are_unsupported():
    t, x, y, u, w = sp.symbols("t x y u w", real=True)
    prob = OptimalControlProblem(
        time=t, states=[x, y], controls=[u, w], dynamics=[u, w], running_cost=x ** 2 + w ** 2,
        initial_state=[1, 0], final_time=1, control_sets=[Box(u, -1, 1), Box(w, -1, 1)])
    s = derive(prob, singular=True)
    assert s.singular_arcs == (None,)
    assert "smooth laws" in s.singular_unsupported[0]


# -- kernels ------------------------------------------------------------------

def test_goddard_singular_kernels(god):
    rng = np.random.default_rng(2)
    z = np.array([1.01, 0.1, 0.8, -1.0, -0.3, -0.5]) + rng.normal(0, 0.05, 6)
    F = god.singular_rhs(0, 0.0, z, GODDARD)
    N, Nt, Nz = god.singular_junction(0, 0.0, z, GODDARD)
    assert Nt[0] + Nz[0] @ F == pytest.approx(N[1], rel=1e-10)
    assert abs(Nt[1] + Nz[1] @ F) < 1e-8 * np.abs(Nz[1] * F).max()
    A = god.singular_jacobian(0, 0.0, z, GODDARD)
    fd = fd_jacobian(lambda w: god.singular_rhs(0, 0.0, w, GODDARD), z)
    assert np.allclose(A, fd, rtol=1e-6, atol=1e-6 * np.abs(A).max())
    # The thrust is S'' = 0 solved for u: on the arc H does not depend on it.
    u_s = god.singular_control(0, 0.0, z, GODDARD)
    assert god.singular_hamiltonian(0, 0.0, z, GODDARD) == pytest.approx(
        god.hamiltonian(0, z, GODDARD) + N[0] * (u_s[0] - god.control(0, z, GODDARD)[0]),
        rel=1e-12)
    with pytest.raises(ValueError, match="out of range"):
        god.singular_rhs(1, 0.0, z, GODDARD)


def test_frozen_propagation(reg):
    # From x = 1 with lam = 1 (S > 0, the law picks u = -1), held at +1.
    r = reg.propagate(0.0, [1.0, 1.0], 1.0, [4.0, 1.0], arc=[1], frozen=[0])
    assert r["z"][0] == pytest.approx(2.0, abs=1e-12)
    assert len(r["switch_times"]) == 0
    with pytest.raises(ValueError, match="out of range"):
        reg.propagate(0.0, [1.0, 1.0], 1.0, [4.0, 1.0], frozen=[1])


def test_singular_kernels_need_derivation(compiled):
    k = compiled["min_time"][2]
    assert k.SINGULAR_ARCS == (False,)
    with pytest.raises(ValueError, match="no singular-arc kernels"):
        k.singular_rhs(0, 0.0, np.zeros(4), [1.0, 1.0, 0.0])


# -- shooting -----------------------------------------------------------------

@pytest.mark.parametrize("T, a", [(4.0, 1.0), (3.0, 0.5)])
def test_regulator_exact(reg, T, a):
    sh = MultiArcShooting(reg, [T, a], [("singular", "u", "lower", "upper")])
    v = sh.pack_arcs(sh.pack([0.8 * a * a], [-0.5]), [(1.2 * a, T - 0.7 * a)])
    R, J = sh.residual_jacobian(v)
    assert np.allclose(J, fd_jacobian(sh.residual, v), atol=1e-6)
    sol = sh.solve(v)
    assert sol.success and sol.max_violation == 0.0
    assert sol.v == pytest.approx([a * a, -a * a, a, T - a], abs=1e-10)
    assert sh.cost(sol.v) == pytest.approx(2 * a ** 3 / 3, abs=1e-12)
    d = sh.diagnostics(sol.v)
    assert d["singular_legendre"] == [(2.0, 2.0)]
    assert d["singular_bound_violation"] == [0.0]
    assert np.allclose(d["singular_u"][0], 0.0)
    assert d["held_sign_violation"][0] < 1e-12
    tr = sh.trajectory(sol.v)
    on = tr["singular"] == 0
    assert np.abs(tr["z"][on, 0]).max() < 1e-12
    assert set(tr["u"][~on, 0]) == {-1.0, 1.0}
    assert tr["junctions"] == pytest.approx([a, T - a], abs=1e-10)


def test_switch_jacobian(reg):
    """A switch time is an unknown: the Jacobian chains the jump in F through dz/dt_s."""
    sh = MultiArcShooting(reg, [4.0, 1.0], [("switch", "u"), ("singular", "u", "upper", "lower")])
    assert sh.frozen == {0: 0}
    v = sh.pack_arcs(sh.pack([0.5], [0.2]), [0.4, (0.9, 2.5)])
    R, J = sh.residual_jacobian(v)
    assert len(R) == len(v) == 5
    assert np.allclose(J, fd_jacobian(sh.residual, v), atol=1e-6)


def test_structure_errors(reg, compiled):
    with pytest.raises(ValueError, match="before, after"):
        MultiArcShooting(reg, [4.0, 1.0], [("singular", "u")])
    with pytest.raises(ValueError, match="entered from"):
        MultiArcShooting(reg, [4.0, 1.0], [("singular", "u", "lower", "upper"),
                                           ("singular", "u", "lower", "upper")])
    with pytest.raises(ValueError, match="not a bang-bang control"):
        MultiArcShooting(reg, [4.0, 1.0], [("singular", "x", "lower", "upper")])
    with pytest.raises(ValueError, match="no singular arc"):
        MultiArcShooting(compiled["min_time"][2], [1.0, 1.0, 0.0],
                         [("singular", "u", "lower", "upper")])
    with pytest.raises(ValueError, match="arc kind"):
        MultiArcShooting(reg, [4.0, 1.0], [("bang", "u")])


def test_regulator_from_smoothed(reg, compiled):
    """The smoothed regulator at eps = 0.1 seeds the exact solve.

    Smoothed single shooting does not get much lower: between the ramps
    ``x' = -lambda / eps``, ``lambda' = -2 x`` grows like ``exp(sqrt(2/eps) t)``,
    so it fails below about ``eps = 0.07``. The junctions are then off by
    about ``sqrt(eps)``, close enough for the exact structure.
    """
    ks = compiled["singular_regulator_smoothed"][2]
    path = parameter_continuation(ks, [4.0, 1.0, 1.0], "eps", np.logspace(0, -1, 5),
                                  SingleShooting(ks, [4.0, 1.0, 1.0]).solve([1.0, -1.0]).v)
    eps, smooth = path[-1]
    tr = SingleShooting(ks, [4.0, 1.0, eps]).trajectory(smooth.v, n_points=4001)
    sh, v = MultiArcShooting.from_smoothed(reg, [4.0, 1.0], [("singular", "u", "lower", "upper")],
                                           smooth.v, tr["t"], tr["u"])
    assert sh.unpack_arcs(v)[0] == pytest.approx((1.0, 3.0), abs=0.3)
    sol = sh.solve(v)
    assert sol.success
    assert sol.v == pytest.approx([1.0, -1.0, 1.0, 3.0], abs=1e-10)
    with pytest.raises(ValueError, match="leaves its bounds"):
        MultiArcShooting.from_smoothed(reg, [4.0, 1.0], [("singular", "u", "lower", "upper")],
                                       smooth.v, tr["t"], np.ones_like(tr["u"]))


# -- Goddard ------------------------------------------------------------------

def test_goddard_exact(god):
    """Full thrust, singular, coast. With the thrust held at its bounds on the
    free arcs, the exact structure converges from a rough guess."""
    sh = MultiArcShooting(god, GODDARD, [("singular", "u", "upper", "lower")])
    v = sh.pack_arcs(sh.pack([-2.5, -0.18, 0.005], [-0.03], 0.2), [(0.03, 0.1)])
    exact = sh.solve(v)
    assert exact.success and exact.max_violation == 0.0
    R, J = sh.residual_jacobian(exact.v)
    assert np.allclose(J, fd_jacobian(sh.residual, exact.v, h=1e-8), rtol=1e-4,
                       atol=1e-5 * np.abs(J).max())
    t_in, t_out = sh.unpack_arcs(exact.v)[0]
    traj = sh.trajectory(exact.v)
    assert traj["z"][-1, 0] == pytest.approx(1.01283692, abs=1e-8)
    assert exact.tf == pytest.approx(0.19885626, abs=1e-8)
    assert traj["z"][-1, 2] == pytest.approx(GODDARD[4], abs=1e-10)
    assert 0 < t_in < t_out < exact.tf
    d = sh.diagnostics(exact.v)
    lo, hi = d["singular_u"][0]
    assert 0 < lo[0] < hi[0] < GODDARD[3]
    assert d["singular_legendre"][0][0] > 0
    assert d["held_sign_violation"][0] < 1e-10
