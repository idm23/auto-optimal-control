"""Pontryagin's minimum principle: from an optimal control problem to a TPBVP.

:func:`derive` turns an :class:`~auto_optimal_control.problem.OptimalControlProblem`
into a :class:`PontryaginSystem`, which holds every expression an indirect
(shooting) solver needs:

- the Hamiltonian ``H = L + lambda . f`` (normal case, ``lambda_0 = 1``);
- the optimal control law ``u*(t, x, lambda, p)``, which minimizes ``H``;
- switching functions for bang-bang controls;
- the augmented dynamics of ``z = [x, lambda]``:
  ``x' = dH/dlambda``, ``lambda' = -dH/dx``, evaluated at ``u*``;
- the boundary-condition residuals: fixed initial states (or
  ``lambda_i(t_0) = 0`` for free ones), terminal constraints, transversality
  ``lambda(t_f) = dPhi/dx`` with ``Phi = phi + nu . psi``, and, for a free
  final time, ``H(t_f) + dPhi/dt = 0``.

The controls are eliminated one control set at a time, in this order:

1. :class:`~auto_optimal_control.problem.UnitVector` sets. ``H`` must be
   linear in the vector, with coefficient ``b``; then ``u* = -b / ||b||``.
2. :class:`~auto_optimal_control.problem.Box` controls. If ``H`` is linear in
   ``u`` with coefficient ``S`` (the switching function), ``u* = upper`` where
   ``S < 0`` and ``lower`` where ``S > 0`` (bang-bang; singular arcs, where
   ``S`` vanishes on an interval, are not handled). If ``H`` is a convex
   quadratic in ``u``, ``u*`` is the unconstrained minimizer clipped to the
   bounds.
3. Unconstrained controls, from ``dH/du = 0`` solved jointly, keeping the
   solution where the Hessian is positive definite.

While a direction is eliminated, a bounded control with ``lower >= 0`` is
treated as nonnegative. This is what turns ``||T delta lambda_v / m||`` into
``T delta ||lambda_v|| / m`` in a thrust-magnitude-and-direction problem, so
the Hamiltonian stays linear in the throttle ``delta``. Declare positive
quantities (masses, thrust, exhaust speed) with ``positive=True`` so SymPy can
simplify the same way.
"""

from dataclasses import dataclass

import sympy as sp

from auto_optimal_control.problem import Box, OptimalControlProblem, UnitVector


@dataclass(frozen=True)
class SwitchingFunction:
    """A bang-bang control: ``u = upper`` where ``expr < 0``, ``lower`` where ``expr > 0``."""

    control: sp.Symbol
    expr: sp.Expr
    lower: sp.Expr
    upper: sp.Expr


@dataclass(frozen=True)
class BoundaryConditions:
    """The shooting function of the TPBVP.

    ``residuals`` vanish at a solution. They are expressions in the boundary
    values ``z0 = (x0, lam0)`` at ``t0`` and ``zf = (xf, lamf)`` at ``tf`` (symbols
    named ``x(t_0)``, ``lam_x(t_f)``, ...), the
    terminal multipliers ``nu``, ``tf`` (a symbol when the final time is free,
    otherwise its fixed value) and the parameters.

    ``unknowns`` are ``x0 + lam0 + nu`` plus ``tf`` if it is free. ``zf``
    follows from them by integration, and ``len(residuals) == len(unknowns)``.
    Fixed initial states are among the unknowns, pinned by their ``initial``
    residuals; a shooting method may eliminate them.
    """

    t0: sp.Expr
    tf: sp.Expr
    x0: tuple
    lam0: tuple
    xf: tuple
    lamf: tuple
    nu: tuple
    unknowns: tuple
    residuals: tuple
    initial: tuple
    terminal: tuple
    transversality: tuple
    hamiltonian: tuple


@dataclass(frozen=True)
class PontryaginSystem:
    """The necessary conditions of an optimal control problem.

    ``z = states + costates`` is the order of the augmented state vector,
    and ``params`` the order of the parameter vector.
    """

    problem: OptimalControlProblem
    time: sp.Symbol
    states: tuple
    costates: tuple
    controls: tuple
    params: tuple
    hamiltonian_full: sp.Expr      # H(t, x, lambda, u, p), controls left symbolic
    control_law: dict              # control -> u*(t, x, lambda, p)
    switching_functions: tuple     # SwitchingFunction per bang-bang control
    hamiltonian: sp.Expr           # H at u*
    running_cost: sp.Expr          # L at u*
    state_dynamics: tuple          # x' at u*
    costate_dynamics: tuple        # lambda' at u*
    boundary: BoundaryConditions

    @property
    def z(self):
        return self.states + self.costates

    @property
    def dynamics(self):
        """``z' = F(t, z, p)``."""
        return self.state_dynamics + self.costate_dynamics

    @property
    def is_autonomous(self):
        """True if ``H`` does not depend on time, so it is constant along extremals."""
        return self.time not in self.hamiltonian.free_symbols

    def __repr__(self):
        return (f"<PontryaginSystem for {self.problem!r}: "
                f"{len(self.switching_functions)} switching functions, "
                f"{len(self.boundary.unknowns)} shooting unknowns>")


def _new_symbol(name, taken, **assumptions):
    if name in taken:
        raise ValueError(f"cannot create symbol {name!r}: a problem symbol already uses that name")
    taken.add(name)
    return sp.Symbol(name, **assumptions)


def _is_zero(expr):
    return sp.simplify(expr) == 0


def _eliminate_unit_vector(H, cs):
    b = [sp.diff(H, u) for u in cs.controls]
    for bi in b:
        for u in cs.controls:
            if not _is_zero(sp.diff(bi, u)):
                raise ValueError(f"H must be linear in the unit vector {cs.controls}")
    # factor() exposes common factors (thrust, throttle, mass) that the
    # square root can then pull out by their sign assumptions.
    norm = sp.sqrt(sp.factor(sp.Add(*[bi ** 2 for bi in b])))
    law = {u: sp.simplify(-bi / norm) for u, bi in zip(cs.controls, b)}
    # H is linear in u, so H(u*) = H(0) + b . u* = H(0) - ||b||.
    H_star = H.subs({u: 0 for u in cs.controls}) - norm
    return law, H_star


def _eliminate_box(H, cs):
    u = cs.control
    d1 = sp.diff(H, u)
    d2 = sp.simplify(sp.diff(d1, u))
    if d2 == 0:
        if not (cs.lower.is_finite and cs.upper.is_finite):
            raise ValueError(f"H is linear in {u}: both of its bounds must be finite")
        S = sp.simplify(d1)
        law = sp.Piecewise((cs.upper, S < 0), (cs.lower, True))
        return {u: law}, SwitchingFunction(u, S, cs.lower, cs.upper)
    if not _is_zero(sp.diff(d2, u)):
        raise NotImplementedError(f"H is neither linear nor quadratic in the bounded control {u}")
    if d2.is_positive is not True:
        raise ValueError(
            f"cannot show d2H/d{u}2 = {d2} is positive. Declare the symbols it depends on "
            f"with positive=True if it is (a negative value means H has no minimum in {u})")
    u_free = sp.simplify(u - d1 / d2)
    law = u_free
    if cs.lower.is_finite is not False:
        law = sp.Max(law, cs.lower)
    if cs.upper.is_finite is not False:
        law = sp.Min(law, cs.upper)
    return {u: law}, None


def _positive_definite(M):
    if M.shape == (1, 1):
        return M[0, 0].is_positive
    return M.is_positive_definite


def _eliminate_unconstrained(H, controls):
    eqs = [sp.diff(H, u) for u in controls]
    solutions = sp.solve(eqs, controls, dict=True)
    if not solutions:
        raise ValueError(f"dH/du = 0 has no closed-form solution for {controls}")
    hessian = sp.hessian(H, controls)
    graded = []
    for sol in solutions:
        if set(sol) != set(controls):
            continue                     # not an isolated stationary point
        pd = _positive_definite(sp.simplify(hessian.subs(sol)))
        if pd is not False:
            graded.append((pd, sol))
    certain = [sol for pd, sol in graded if pd]
    if len(certain) == 1:
        return certain[0]
    if len(graded) == 1:
        return graded[0][1]
    raise ValueError(
        f"dH/du = 0 does not give a unique minimizer for {controls}: {solutions}. "
        "For a direction, use a UnitVector control set; for bounds, a Box.")


def _resolve(law, controls):
    """Substitute laws into each other until no control symbol remains."""
    resolved = dict(law)
    for _ in range(len(controls) + 1):
        changed = False
        for u, expr in resolved.items():
            new = expr.subs(resolved)
            if new != expr:
                resolved[u] = new
                changed = True
        if not changed:
            break
    for u, expr in resolved.items():
        left = expr.free_symbols & set(controls)
        if left:
            raise ValueError(f"control law for {u} still depends on {sorted(map(str, left))}")
    return resolved


def _control_law(problem, H):
    in_set = set()
    boxes, vectors = [], []
    for cs in problem.control_sets:
        in_set.update(cs.controls)
        (vectors if isinstance(cs, UnitVector) else boxes).append(cs)
    free = [u for u in problem.controls if u not in in_set]

    # Treat bounded controls with a nonnegative (nonpositive) range as such
    # while directions are eliminated, so norms factor them out.
    signed = {}
    for cs in boxes:
        if cs.lower.is_nonnegative:
            signed[cs.control] = sp.Dummy(cs.control.name, nonnegative=True)
        elif cs.upper.is_nonpositive:
            signed[cs.control] = sp.Dummy(cs.control.name, nonpositive=True)
    unsigned = {d: u for u, d in signed.items()}

    law, switching = {}, []
    Hs = H.subs(signed)
    for cs in vectors:
        part, Hs = _eliminate_unit_vector(Hs, cs)
        law.update({u: e.subs(unsigned) for u, e in part.items()})
    Hs = Hs.subs(unsigned)
    for cs in boxes:
        part, sw = _eliminate_box(Hs, cs)
        law.update(part)
        if sw is not None:
            switching.append(sw)
        Hs = Hs.subs(part)
    if free:
        part = _eliminate_unconstrained(Hs, free)
        law.update(part)
        Hs = Hs.subs(part)

    law = _resolve(law, problem.controls)
    # Switching functions may still contain controls of other sets.
    switching = [SwitchingFunction(sw.control, sp.simplify(sw.expr.subs(law)), sw.lower, sw.upper)
                 for sw in switching]
    return {u: law[u] for u in problem.controls}, tuple(switching), sp.simplify(Hs)


def _boundary_conditions(problem, H_star, costates, taken):
    x = problem.states
    t = problem.time
    x0 = tuple(_new_symbol(f"{s.name}(t_0)", taken, real=True) for s in x)
    xf = tuple(_new_symbol(f"{s.name}(t_f)", taken, real=True) for s in x)
    lam0 = tuple(_new_symbol(f"{l.name}(t_0)", taken, real=True) for l in costates)
    lamf = tuple(_new_symbol(f"{l.name}(t_f)", taken, real=True) for l in costates)
    nu = tuple(_new_symbol(f"nu_{k}", taken, real=True)
               for k in range(len(problem.terminal_constraints)))
    t0 = problem.initial_time
    tf = _new_symbol("t_f", taken, real=True) if problem.free_final_time else problem.final_time

    at_final = dict(zip(x, xf))
    phi = problem.terminal_cost + sp.Add(*[n * c for n, c in zip(nu, problem.terminal_constraints)])

    initial = []
    for s, s0, l0 in zip(x, x0, lam0):
        if s in problem.initial_state:
            initial.append(s0 - problem.initial_state[s].subs(t, t0))
        else:
            initial.append(l0)
    terminal = [c.subs(at_final).subs(t, tf) for c in problem.terminal_constraints]
    transversality = [lf - sp.diff(phi, s).subs(at_final).subs(t, tf) for s, lf in zip(x, lamf)]
    hamiltonian = []
    if problem.free_final_time:
        H_f = H_star.subs({**at_final, **dict(zip(costates, lamf))}).subs(t, tf)
        hamiltonian.append(H_f + sp.diff(phi, t).subs(at_final).subs(t, tf))

    unknowns = x0 + lam0 + nu + ((tf,) if problem.free_final_time else ())
    residuals = tuple(initial + terminal + transversality + hamiltonian)
    assert len(residuals) == len(unknowns)
    return BoundaryConditions(
        t0=t0, tf=tf, x0=x0, lam0=lam0, xf=xf, lamf=lamf, nu=nu,
        unknowns=unknowns, residuals=residuals, initial=tuple(initial),
        terminal=tuple(terminal), transversality=tuple(transversality),
        hamiltonian=tuple(hamiltonian))


def derive(problem):
    """Derive the first-order necessary conditions of ``problem``.

    Returns a :class:`PontryaginSystem`.
    """
    taken = {s.name for s in (problem.time, *problem.states, *problem.controls, *problem.params)}
    costates = tuple(_new_symbol(f"lam_{s.name}", taken, real=True) for s in problem.states)

    H = problem.running_cost + sp.Add(*[l * f for l, f in zip(costates, problem.dynamics)])
    # H_star comes out of the elimination, where each direction contributed
    # -||b||; substituting u* into H gives the same value in a messier form.
    law, switching, H_star = _control_law(problem, H)

    # dH/dx with the controls still symbolic, then u* substituted: correct
    # for interior, saturated and bang-bang controls alike.
    state_dynamics = tuple(sp.simplify(f.subs(law)) for f in problem.dynamics)
    costate_dynamics = tuple(sp.simplify(-sp.diff(H, s).subs(law)) for s in problem.states)

    boundary = _boundary_conditions(problem, H_star, costates, taken)
    return PontryaginSystem(
        problem=problem, time=problem.time, states=problem.states, costates=costates,
        controls=problem.controls, params=problem.params, hamiltonian_full=H,
        control_law=law, switching_functions=switching, hamiltonian=H_star,
        running_cost=sp.simplify(problem.running_cost.subs(law)),
        state_dynamics=state_dynamics, costate_dynamics=costate_dynamics, boundary=boundary)
