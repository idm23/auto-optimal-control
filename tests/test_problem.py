"""Validation of OptimalControlProblem and its control sets."""

import pytest
import sympy as sp

from auto_optimal_control.problem import Box, OptimalControlProblem, UnitVector

t, x, v, u, w = sp.symbols("t x v u w", real=True)
k = sp.Symbol("k")


def make(**overrides):
    kw = dict(time=t, states=[x, v], controls=[u], dynamics=[v, u], running_cost=1,
              initial_state=[0, 0], final_state={x: 1, v: 0})
    kw.update(overrides)
    return OptimalControlProblem(**kw)


def test_basic_attributes():
    p = make(params=[k], dynamics=[v, k * u], control_sets=[Box(u, -1, 1)])
    assert p.states == (x, v) and p.controls == (u,) and p.params == (k,)
    assert p.free_final_time
    assert p.initial_state == {x: 0, v: 0}
    assert p.terminal_constraints == (x - 1, v)
    assert "free final time" in repr(p)


def test_initial_state_forms():
    assert make(initial_state={v: 2}).initial_state == {v: 2}
    assert make(initial_state=[None, 3]).initial_state == {v: 3}
    assert make(initial_state=None).initial_state == {}
    with pytest.raises(ValueError, match="entries"):
        make(initial_state=[0])
    with pytest.raises(ValueError, match="not a state"):
        make(initial_state={u: 0})


@pytest.mark.parametrize("overrides,match", [
    (dict(dynamics=[v]), "1 dynamics for 2 states"),
    (dict(dynamics=[v, u + w]), "undeclared symbols"),
    (dict(running_cost=w), "undeclared symbols"),
    (dict(terminal_cost=u), "undeclared symbols"),
    (dict(initial_state=[x, 0]), "undeclared symbols"),
    (dict(states=[x, x]), "duplicates"),
    (dict(controls=[x], dynamics=[v, x]), "both states and controls"),
    (dict(control_sets=[Box(w, 0, 1)]), "not a control"),
    (dict(control_sets=[Box(u, 0, 1), Box(u, -1, 1)]), "more than one control set"),
    (dict(final_state={x: 1, v: 0}, terminal_constraints=[x + v, x - v]), "more terminal constraints"),
    (dict(states=[]), "at least one state"),
])
def test_invalid_problems(overrides, match):
    with pytest.raises(ValueError, match=match):
        make(**overrides)


def test_type_errors():
    with pytest.raises(TypeError):
        make(time="t")
    with pytest.raises(TypeError):
        make(states=[x, 2 * v])
    with pytest.raises(TypeError):
        make(control_sets=[(u, 0, 1)])
    with pytest.raises(ValueError, match="at least two"):
        UnitVector([u])


def test_box_defaults_are_infinite():
    b = Box(u)
    assert b.lower == -sp.oo and b.upper == sp.oo and b.controls == (u,)
