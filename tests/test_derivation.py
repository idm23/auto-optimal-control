"""Symbolic checks of the Pontryagin derivation against hand-derived results.

Each example's Hamiltonian, control law, switching function, costate
equations and boundary conditions are compared with the textbook forms.
Two generic properties are then checked numerically on every example at
random points: ``u*`` minimizes ``H`` over the admissible set, and the
augmented dynamics are Hamilton's equations of ``H*`` (envelope theorem).
"""

import numpy as np
import pytest
import sympy as sp

from auto_optimal_control import examples
from auto_optimal_control.pontryagin import derive
from auto_optimal_control.problem import Box, OptimalControlProblem, UnitVector


def same(a, b):
    return sp.simplify(sp.sympify(a) - sp.sympify(b)) == 0


def syms(system, *names):
    table = {s.name: s for s in (*system.states, *system.costates, *system.controls,
                                 *system.params, system.time)}
    return [table[n] for n in names]


def test_zermelo():
    s = derive(examples.zermelo())
    y, lx, ly, ux, uy, V, k = syms(s, "y", "lam_x", "lam_y", "u_x", "u_y", "V", "k")
    norm = sp.sqrt(lx ** 2 + ly ** 2)
    assert same(s.control_law[ux], -lx / norm)
    assert same(s.control_law[uy], -ly / norm)
    assert same(s.hamiltonian, 1 - V * norm + k * y * lx)
    assert s.costate_dynamics == (0, -k * lx)          # lam_x' = 0, lam_y' = -k lam_x
    assert s.switching_functions == ()
    assert s.is_autonomous
    bc = s.boundary
    assert len(bc.unknowns) == len(bc.residuals) == 7    # x0, lam0, nu (2), t_f
    assert bc.unknowns[-1].name == "t_f"
    # transversality: lam(t_f) = nu; free final time: H(t_f) = 0
    assert [same(r, l - n) for r, l, n in zip(bc.transversality, bc.lamf, bc.nu)] == [True, True]
    assert len(bc.hamiltonian) == 1


def test_double_integrator_min_time_is_bang_bang():
    s = derive(examples.double_integrator_min_time())
    lx, lv, u, umax, v = syms(s, "lam_x", "lam_v", "u", "u_max", "v")
    (sw,) = s.switching_functions
    assert sw.control == u and same(sw.expr, lv)
    assert (sw.lower, sw.upper) == (-umax, umax)
    law = s.control_law[u]
    assert law.subs(lv, -2) == umax and law.subs(lv, 3) == -umax
    assert s.costate_dynamics == (0, -lx)
    assert same(s.hamiltonian.subs(lv, 2), 1 + lx * v - 2 * umax)


def test_min_energy_unconstrained_and_saturated():
    s = derive(examples.double_integrator_min_energy())
    lx, lv, u, v = syms(s, "lam_x", "lam_v", "u", "v")
    assert s.control_law[u] == -lv
    assert same(s.hamiltonian, lx * v - lv ** 2 / 2)
    assert s.problem.final_time is not None and len(s.boundary.unknowns) == 6
    assert s.boundary.hamiltonian == ()

    b = derive(examples.double_integrator_min_energy(bounded=True))
    lv, u, umax = syms(b, "lam_v", "u", "u_max")
    law = b.control_law[u]
    assert law.subs({lv: 0.5, umax: 1}) == -0.5
    assert law.subs({lv: -3, umax: 1}) == 1
    assert law.subs({lv: 3, umax: 1}) == -1
    assert b.switching_functions == ()


def test_soft_terminal_transversality():
    s = derive(examples.soft_terminal_lq())
    w = syms(s, "w")[0]
    bc = s.boundary
    (tr,) = bc.transversality
    assert same(tr, bc.lamf[0] - w * bc.xf[0])
    assert bc.nu == () and bc.terminal == ()


def test_low_thrust_primer_vector_and_switching_function():
    s = derive(examples.planar_low_thrust_fuel())
    lvx, lvy, lm, m, T, c, ax, ay, delta = syms(
        s, "lam_v_x", "lam_v_y", "lam_m", "m", "T_max", "c", "a_x", "a_y", "delta")
    norm = sp.sqrt(lvx ** 2 + lvy ** 2)
    assert same(s.control_law[ax], -lvx / norm)
    assert same(s.control_law[ay], -lvy / norm)
    (sw,) = s.switching_functions
    assert sw.control == delta and (sw.lower, sw.upper) == (0, 1)
    assert same(sw.expr, T / c * (1 - c * norm / m - lm))
    # The direction does not depend on the throttle.
    assert delta not in s.control_law[ax].free_symbols
    # Mass costate: lam_m' = -dH/dm = -T delta ||lam_v|| / m^2 (thrust on).
    on = {lm: 5}                                        # S < 0 everywhere
    assert same(s.costate_dynamics[4].subs(on), -T * norm / m ** 2)
    bc = s.boundary
    assert len(bc.nu) == 3 and len(bc.unknowns) == len(bc.residuals) == 13
    assert bc.transversality[4] == bc.lamf[4]           # final mass free: lam_m(t_f) = 0


def test_free_initial_state_gives_zero_costate():
    t, x, u = sp.symbols("t x u", real=True)
    s = derive(OptimalControlProblem(time=t, states=[x], controls=[u], dynamics=[u],
                                     running_cost=(x - 1) ** 2 + u ** 2, final_time=1))
    assert s.boundary.initial == (s.boundary.lam0[0],)


def test_time_varying_problem():
    t, x, u = sp.symbols("t x u", real=True)
    s = derive(OptimalControlProblem(time=t, states=[x], controls=[u], dynamics=[t * u],
                                     running_cost=u ** 2 / 2, initial_state=[0],
                                     final_state={x: 1}, final_time=2))
    assert not s.is_autonomous
    assert same(s.control_law[u], -t * s.costates[0])


def test_multiple_unconstrained_controls_are_solved_jointly():
    t, x, y, u1, u2 = sp.symbols("t x y u1 u2", real=True)
    s = derive(OptimalControlProblem(
        time=t, states=[x, y], controls=[u1, u2], dynamics=[u1 + u2, u1 - u2],
        running_cost=u1 ** 2 + u1 * u2 + u2 ** 2, initial_state=[0, 0], final_time=1))
    lx, ly = s.costates
    # Solve 2u1 + u2 + lx + ly = 0, u1 + 2u2 + lx - ly = 0.
    assert same(s.control_law[u1], -(lx + 3 * ly) / 3)
    assert same(s.control_law[u2], -(lx - 3 * ly) / 3)


def test_derivation_errors():
    t, x, u, ux, uy = sp.symbols("t x u u_x u_y", real=True)
    base = dict(time=t, states=[x], initial_state=[0], final_time=1)
    with pytest.raises(ValueError, match="positive"):          # concave in u: no minimum
        derive(OptimalControlProblem(controls=[u], dynamics=[u], running_cost=-u ** 2,
                                     control_sets=[Box(u, -1, 1)], **base))
    with pytest.raises(ValueError, match="finite"):
        derive(OptimalControlProblem(controls=[u], dynamics=[u], running_cost=0,
                                     control_sets=[Box(u, 0)], **base))
    with pytest.raises(NotImplementedError):
        derive(OptimalControlProblem(controls=[u], dynamics=[u], running_cost=u ** 4,
                                     control_sets=[Box(u, -1, 1)], **base))
    with pytest.raises(ValueError, match="linear in the unit vector"):
        derive(OptimalControlProblem(controls=[ux, uy], dynamics=[ux ** 2], running_cost=0,
                                     control_sets=[UnitVector([ux, uy])], **base))
    with pytest.raises(ValueError, match="minimizer|closed-form"):
        derive(OptimalControlProblem(controls=[u], dynamics=[sp.sin(u)], running_cost=0, **base))
    lam_x = sp.Symbol("lam_x")
    with pytest.raises(ValueError, match="lam_x"):
        derive(OptimalControlProblem(controls=[u], dynamics=[u], running_cost=u ** 2,
                                     params=[lam_x], **base))


# ---------------------------------------------------------------------------
# Generic properties on every example, at random points
# ---------------------------------------------------------------------------

_all_examples = examples.all_examples


@pytest.fixture(scope="module")
def derived():
    return {name: derive(p) for name, p in _all_examples().items()}


def _random_point(system, rng):
    point = {system.time: rng.uniform(0.1, 2.0)}
    for s in (*system.states, *system.costates, *system.params):
        lo = 0.2 if s.is_positive else -2.0
        point[s] = rng.uniform(lo, 2.0)
    return point


def _random_admissible_control(problem, point, rng):
    u = {}
    for cs in problem.control_sets:
        if isinstance(cs, Box):
            lo, hi = float(cs.lower.subs(point)), float(cs.upper.subs(point))
            u[cs.control] = rng.uniform(max(lo, -5), min(hi, 5))
        else:
            d = rng.normal(size=len(cs.controls))
            u.update(zip(cs.controls, d / np.linalg.norm(d)))
    for c in problem.controls:
        u.setdefault(c, rng.uniform(-5, 5))
    return u


@pytest.mark.parametrize("name", list(_all_examples()))
def test_control_law_minimizes_hamiltonian(derived, name):
    s = derived[name]
    rng = np.random.default_rng(1)
    args = sorted(s.hamiltonian_full.free_symbols, key=str)
    H = sp.lambdify(args, s.hamiltonian_full)
    for _ in range(20):
        point = _random_point(s, rng)
        u_star = {u: float(law.subs(point)) for u, law in s.control_law.items()}
        h_star = H(*[{**point, **u_star}.get(a) for a in args])
        assert np.isclose(h_star, float(s.hamiltonian.subs(point)), rtol=1e-12, atol=1e-12)
        for _ in range(20):
            u = _random_admissible_control(s.problem, point, rng)
            assert H(*[{**point, **u}.get(a) for a in args]) >= h_star - 1e-12


@pytest.mark.parametrize("name", list(_all_examples()))
def test_dynamics_are_hamiltons_equations_of_H_star(derived, name):
    """x' = dH*/dlam and lam' = -dH*/dx away from switching surfaces."""
    s = derived[name]
    rng = np.random.default_rng(2)
    grads = ([sp.diff(s.hamiltonian, l) for l in s.costates]
             + [-sp.diff(s.hamiltonian, x) for x in s.states])
    for _ in range(10):
        point = _random_point(s, rng)
        for f, g in zip(s.dynamics, grads):
            assert np.isclose(float(f.subs(point)), float(g.subs(point)), rtol=1e-10, atol=1e-12)
