"""Symbolic definition of an optimal control problem.

The problem is::

    minimize    J = phi(t_f, x(t_f)) + integral_{t_0}^{t_f} L(t, x, u) dt
    subject to  x' = f(t, x, u)
                x_i(t_0) = x0_i              for every state with a fixed initial value
                psi(t_f, x(t_f)) = 0         (terminal constraints)
                u(t) in U                    (control sets, see below)

where every expression may also depend on symbolic parameters ``p``. The
parameters stay symbolic through the derivation, so one compiled kernel
serves a whole continuation sweep.

Controls are unconstrained unless they belong to a control set:

- :class:`Box` -- bounds ``lower <= u <= upper`` on one control. A Hamiltonian
  linear in ``u`` gives bang-bang control with a switching function; a
  convex quadratic one gives a saturated (clipped) control.
- :class:`UnitVector` -- a direction ``||(u_1, ..., u_k)|| = 1``, such as a
  thrust direction or a heading. The Hamiltonian must be linear in it.

State limits are :class:`PathConstraint` objects, ``g(t, x, p) <= 0`` along
the whole trajectory. The derivation finds each one's order, the number of
time derivatives of ``g`` it takes for a control to appear. Propagation
locates every crossing of ``g = 0``; it can stop there (see
``Kernels.propagate``), and a constraint with a ``penalty`` weight adds the
exterior penalty ``penalty / 2 * max(0, g)**2`` to the running cost.
"""

from dataclasses import dataclass

import sympy as sp


def _as_symbols(items, what):
    symbols = tuple(items)
    for s in symbols:
        if not isinstance(s, sp.Symbol):
            raise TypeError(f"{what} must be SymPy Symbols, got {s!r}")
    if len(set(symbols)) != len(symbols):
        raise ValueError(f"{what} contain duplicates: {symbols}")
    return symbols


@dataclass(frozen=True)
class Box:
    """Bounds ``lower <= control <= upper``; either bound may be infinite."""

    control: sp.Symbol
    lower: sp.Expr
    upper: sp.Expr

    def __init__(self, control, lower=-sp.oo, upper=sp.oo):
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "lower", sp.sympify(lower))
        object.__setattr__(self, "upper", sp.sympify(upper))

    @property
    def controls(self):
        return (self.control,)


@dataclass(frozen=True)
class UnitVector:
    """The controls form a unit vector: ``sum(u_i**2) == 1``."""

    controls: tuple

    def __init__(self, controls):
        object.__setattr__(self, "controls", _as_symbols(controls, "UnitVector controls"))
        if len(self.controls) < 2:
            raise ValueError("a UnitVector needs at least two components")


@dataclass(frozen=True)
class PathConstraint:
    """A state limit ``expr <= 0``, enforced along the whole trajectory.

    ``expr`` may depend on time, states and parameters, but not on the
    controls. ``penalty`` is an optional weight (a number or an expression in
    the parameters, usually a parameter so a continuation can raise it). When
    given, ``penalty / 2 * max(0, expr)**2`` is added to the running cost.
    Without it the constraint is only detected: its crossings are reported
    and propagation can stop at a violation.
    """

    expr: sp.Expr
    name: str
    penalty: sp.Expr

    def __init__(self, expr, name="", penalty=None):
        object.__setattr__(self, "expr", sp.sympify(expr))
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "penalty", None if penalty is None else sp.sympify(penalty))


class OptimalControlProblem:
    """An optimal control problem in Bolza form, expressed with SymPy.

    Parameters
    ----------
    time : Symbol
        The independent variable ``t``.
    states, controls : sequence of Symbol
    dynamics : sequence of Expr
        ``f(t, x, u, p)``, one entry per state.
    running_cost : Expr
        ``L(t, x, u, p)``. ``1`` gives a minimum-time problem.
    terminal_cost : Expr
        ``phi(t, x, p)``, evaluated at the final time (``time`` stands for
        ``t_f`` and each state for its final value).
    initial_state : dict or sequence
        Fixed initial values. A dict maps states to values; a sequence gives
        one value per state, ``None`` meaning free. States not given are free,
        and get the natural condition ``lambda_i(t_0) = 0``.
    final_state : dict
        Shorthand for terminal constraints ``x_i - value = 0``.
    terminal_constraints : sequence of Expr
        ``psi(t, x, p) = 0`` at the final time, in addition to ``final_state``.
    initial_time : Expr
        Fixed ``t_0`` (a number or an expression in the parameters).
    final_time : Expr or None
        Fixed ``t_f``, or ``None`` for a free final time.
    params : sequence of Symbol
        Runtime parameters, in the order of the parameter vector.
    control_sets : sequence of Box or UnitVector
        Constraints on the controls; each control is in at most one set.
    path_constraints : sequence of PathConstraint or Expr
        State limits ``g(t, x, p) <= 0``. A bare expression is a constraint
        without a penalty.
    name : str
    """

    def __init__(self, *, time, states, controls, dynamics, running_cost=0, terminal_cost=0,
                 initial_state=None, final_state=None, terminal_constraints=(),
                 initial_time=0, final_time=None, params=(), control_sets=(), path_constraints=(),
                 name=""):
        if not isinstance(time, sp.Symbol):
            raise TypeError("time must be a SymPy Symbol")
        self.time = time
        self.states = _as_symbols(states, "states")
        self.controls = _as_symbols(controls, "controls")
        self.params = _as_symbols(params, "params")
        self.name = name
        if not self.states:
            raise ValueError("at least one state is required")

        groups = {"time": (time,), "states": self.states, "controls": self.controls,
                  "params": self.params}
        seen = {}
        for group, syms in groups.items():
            for s in syms:
                if s in seen:
                    raise ValueError(f"{s} appears in both {seen[s]} and {group}")
                seen[s] = group

        self.dynamics = tuple(sp.sympify(e) for e in dynamics)
        if len(self.dynamics) != len(self.states):
            raise ValueError(f"{len(self.dynamics)} dynamics for {len(self.states)} states")
        self.running_cost = sp.sympify(running_cost)
        self.terminal_cost = sp.sympify(terminal_cost)

        self.initial_state = self._initial_values(initial_state)
        constraints = [sp.sympify(c) for c in terminal_constraints]
        for s, value in self._as_state_dict(final_state or {}, "final_state").items():
            constraints.append(s - sp.sympify(value))
        self.terminal_constraints = tuple(constraints)
        if len(self.terminal_constraints) > len(self.states) + (final_time is None):
            raise ValueError("more terminal constraints than the problem has freedom for")

        self.initial_time = sp.sympify(initial_time)
        self.final_time = None if final_time is None else sp.sympify(final_time)

        self.control_sets = tuple(control_sets)
        owner = {}
        for cs in self.control_sets:
            if not isinstance(cs, (Box, UnitVector)):
                raise TypeError(f"unknown control set {cs!r}")
            for u in cs.controls:
                if u not in self.controls:
                    raise ValueError(f"control set refers to {u}, which is not a control")
                if u in owner:
                    raise ValueError(f"control {u} is in more than one control set")
                owner[u] = cs

        constraints = []
        for k, c in enumerate(path_constraints):
            if not isinstance(c, PathConstraint):
                c = PathConstraint(c)
            if not c.name:
                c = PathConstraint(c.expr, f"g{k}", c.penalty)
            constraints.append(c)
        names = [c.name for c in constraints]
        if len(set(names)) != len(names):
            raise ValueError(f"path constraint names must be unique: {names}")
        self.path_constraints = tuple(constraints)

        self._check_symbols()

    @property
    def free_final_time(self):
        return self.final_time is None

    def _as_state_dict(self, values, what):
        out = {}
        for s, v in dict(values).items():
            if s not in self.states:
                raise ValueError(f"{what} refers to {s}, which is not a state")
            out[s] = sp.sympify(v)
        return out

    def _initial_values(self, initial_state):
        if initial_state is None:
            return {}
        if isinstance(initial_state, dict):
            return self._as_state_dict(initial_state, "initial_state")
        values = list(initial_state)
        if len(values) != len(self.states):
            raise ValueError(f"initial_state has {len(values)} entries for {len(self.states)} states")
        return {s: sp.sympify(v) for s, v in zip(self.states, values) if v is not None}

    def _check_symbols(self):
        t, x, u, p = {self.time}, set(self.states), set(self.controls), set(self.params)

        def check(expr, allowed, what):
            extra = expr.free_symbols - allowed
            if extra:
                raise ValueError(f"{what} depends on undeclared symbols {sorted(map(str, extra))}")

        for e in self.dynamics:
            check(e, t | x | u | p, "dynamics")
        check(self.running_cost, t | x | u | p, "running_cost")
        check(self.terminal_cost, t | x | p, "terminal_cost")
        for c in self.terminal_constraints:
            check(c, t | x | p, "terminal constraint")
        for v in self.initial_state.values():
            check(v, p, "initial_state")
        for c in self.path_constraints:
            check(c.expr, t | x | p, f"path constraint {c.name!r}")
            if not c.expr.free_symbols & x:
                raise ValueError(f"path constraint {c.name!r} does not depend on the states")
            if c.penalty is not None:
                check(c.penalty, p, f"penalty of path constraint {c.name!r}")
        check(self.initial_time, p, "initial_time")
        if self.final_time is not None:
            check(self.final_time, p, "final_time")

    def __repr__(self):
        label = f" {self.name!r}" if self.name else ""
        limits = (f"{len(self.path_constraints)} path constraints, "
                  if self.path_constraints else "")
        return (f"<OptimalControlProblem{label}: {len(self.states)} states, "
                f"{len(self.controls)} controls, {len(self.params)} params, {limits}"
                f"{'free' if self.free_final_time else 'fixed'} final time>")
