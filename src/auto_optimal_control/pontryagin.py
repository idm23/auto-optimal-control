"""Pontryagin's minimum principle: from an optimal control problem to a TPBVP.

:func:`derive` turns an :class:`~auto_optimal_control.problem.OptimalControlProblem`
into a :class:`PontryaginSystem`, which holds every expression an indirect
(shooting) solver needs:

- the Hamiltonian ``H = L + lambda . f`` (normal case, ``lambda_0 = 1``);
- the optimal control law ``u*(t, x, lambda, p)``, which minimizes ``H``;
- switching functions for bang-bang controls;
- the augmented dynamics of ``z = [x, lambda]``:
  ``x' = dH/dlambda``, ``lambda' = -dH/dx``, evaluated at ``u*``;
- for each path constraint ``g <= 0``, its order and time derivatives, and
  the exterior penalty it adds to the running cost, if it has a weight;
- the boundary arcs that ride one limit, or several at once
  (``boundary_sets``: up to ``simultaneous`` limits, at most one per
  control), with their controls, multipliers and dynamics;
- the boundary-condition residuals: fixed initial states (or
  ``lambda_i(t_0) = 0`` for free ones), terminal constraints, transversality
  ``lambda(t_f) = dPhi/dx`` with ``Phi = phi + nu . psi``, and, for a free
  final time, ``H(t_f) + dPhi/dt = 0``.

The controls are eliminated one control set at a time, in this order:

1. :class:`~auto_optimal_control.problem.UnitVector` sets. ``H`` must be
   linear in the vector, with coefficient ``b``; then ``u* = -b / ||b||``.
2. :class:`~auto_optimal_control.problem.Box` controls. If ``H`` is linear in
   ``u`` with coefficient ``S`` (the switching function), ``u* = upper`` where
   ``S < 0`` and ``lower`` where ``S > 0`` (bang-bang). Where ``S`` vanishes
   on an interval the arc is singular; ``derive(problem, singular=True)``
   derives the singular control (see :class:`SingularArc`), which
   :class:`~auto_optimal_control.shooting.MultiArcShooting` follows on arcs
   the structure names. If ``H`` is a convex
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

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import sympy as sp

from auto_optimal_control.functions import greater, lesser
from auto_optimal_control.problem import Box, OptimalControlProblem, UnitVector
from auto_optimal_control.tape import Outlined, Tape, Unsupported


@dataclass(frozen=True)
class SwitchingFunction:
    """A bang-bang control: ``u = upper`` where ``expr < 0``, ``lower`` where ``expr > 0``."""

    control: sp.Symbol
    expr: sp.Expr
    lower: sp.Expr
    upper: sp.Expr


@dataclass(frozen=True)
class SingularArc:
    """The extremal while switching function ``S`` of a bang-bang control vanishes.

    ``derivatives`` holds ``S, S', S''``, taken along the state-costate
    dynamics with the control left symbolic. ``S'`` does not involve the
    control; ``S'' = a + b u`` does (an order-1 arc, ``b != 0``), and
    ``S'' = 0`` gives the singular control ``u_s = -a / b`` (``control``,
    with the other controls at their laws). Along ``dynamics``, ``z' = F(u_s)``,
    ``S`` and ``S'`` stay constant, so an arc that enters with ``S = S' = 0``
    keeps them at zero; ``H`` is continuous at both junctions because
    ``S = 0`` there. ``legendre_clebsch`` is ``-dS''/du = -b``, which is
    ``>= 0`` along a minimizing singular arc (the generalized
    Legendre-Clebsch condition of Kelley, Kopp & Moyer, 1967). The control
    must also stay within its bounds; shooting reports both.
    """

    control: sp.Symbol
    derivatives: tuple
    law: dict
    dynamics: tuple
    hamiltonian: sp.Expr
    running_cost: sp.Expr
    legendre_clebsch: sp.Expr
    lower: sp.Expr
    upper: sp.Expr

    @property
    def junction(self):
        """``(S, S')``: zero at the entry of the arc."""
        return self.derivatives[:2]


@dataclass(frozen=True)
class StateConstraint:
    """A path constraint ``g(t, x, p) <= 0`` after derivation.

    ``order`` is the number of time derivatives of ``g``, taken along the
    dynamics with the controls left symbolic, before a control appears.
    ``derivatives`` holds ``g, g', ..., g^(order)``; only the last depends on
    the controls. ``penalty`` is the weight of ``penalty / 2 * max(0, g)**2``
    in the running cost, or ``None``.
    """

    name: str
    expr: sp.Expr
    order: int
    derivatives: tuple
    penalty: sp.Expr
    boundary: object = None        # BoundaryArc, or None if exact arcs are not supported
    unsupported: str = ""          # why not, when boundary is None

    @property
    def tangency(self):
        """``N = (g, ..., g^(order-1))``: zero at the entry of a boundary arc."""
        return self.derivatives[:self.order]


@dataclass(frozen=True)
class BoundaryArc:
    """The extremal while a path constraint ``g = 0`` is active (riding the limit).

    ``g^(q) = 0`` along the arc fixes the control: ``control`` minimizes ``H``
    subject to it, and ``multiplier`` is the multiplier ``mu`` of
    ``H + mu g^(q)`` (indirect adjoining). ``dynamics`` is ``z' = F_b(t, z, p)``:
    ``x' = f(u_b)`` and ``lambda' = -d(H + mu g^(q))/dx`` at ``(u_b, mu)``.
    ``hamiltonian`` is ``H`` at ``u_b`` and ``running_cost`` is ``L`` there.
    At the entry, the costates jump by ``-dN/dx^T pi``; at the exit they are
    continuous, and ``H`` is continuous at both.

    The same equations hold for bounded controls, smooth (``H`` quadratic) or
    bang-bang (``H`` linear, so ``dH/du`` is the switching function): ``u_b``
    is then the control that rides the limit, which must lie within its
    bounds along the arc. ``bounds`` maps each bounded control to its
    ``(lower, upper)``; shooting reports how far ``u_b`` leaves them. When
    riding leaves a line of controls (one more control than limits) and
    ``H`` is quadratic, the bounds are kept instead: ``u_b`` is the best
    point of the line within them (:func:`_clipped_on_line`), so a jerk can
    sit at its bound while the other rides the torque limit.

    An arc that rides several limits at once (:attr:`PontryaginSystem.boundary_sets`)
    adjoins each with its own multiplier: the controls and ``mu_i`` solve
    ``dH/du + sum mu_i dg_i^(q)/du = 0`` and every ``g_i^(q) = 0``;
    ``constraints`` lists the limits and ``multipliers`` their ``mu_i``.
    """

    control: dict
    multiplier: sp.Expr
    dynamics: tuple
    hamiltonian: sp.Expr
    running_cost: sp.Expr
    bounds: dict = None
    constraints: tuple = ()        # for an arc riding several limits: their indices
    multipliers: tuple = ()        # and one mu per limit (``multiplier`` is the first)


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
class RidingSystem:
    """Boundary arcs of large order-1 limits, solved at run time (:mod:`~auto_optimal_control.riding`).

    Each limit is ``sign_of[c] * root + (terms free of the states)``, with
    ``roots[root_of[c]]`` a step of ``tape``: both limits on one torque share
    its root. The set of limits ridden is chosen when the kernels are called,
    so there is one kernel for all of them.
    """

    tape: Tape
    roots: tuple
    root_of: tuple
    sign_of: tuple


class _RiddenAtRunTime:
    """Stands in :attr:`StateConstraint.boundary` for a :class:`RidingSystem` arc."""


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
    path_constraints: tuple = ()   # StateConstraint per path constraint
    penalty: sp.Expr = sp.S.Zero   # sum of the exterior penalties, part of L and H above
    singular_arcs: tuple = ()      # per switching function: SingularArc, or None
    singular_unsupported: tuple = ()   # per switching function: why not, when None
    boundary_sets: tuple = ()      # ((constraint indices, BoundaryArc), ...): limits ridden at once
    riding: object = None          # RidingSystem: boundary arcs solved at run time, or None

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
                f"{len(self.path_constraints)} path constraints, "
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


MAX_CONSTRAINT_ORDER = 8


def constraint_derivatives(problem, expr):
    """``(g, g', ..., g^(q))`` along the dynamics, stopping at the first that involves a control.

    Each derivative is ``dg/dt + dg/dx . f(t, x, u, p)`` with the controls
    symbolic, so ``q`` is the order of the constraint ``g <= 0``. Raises
    ``ValueError`` if no control appears within ``MAX_CONSTRAINT_ORDER``
    derivatives.
    """
    t, controls = problem.time, problem.controls
    g = sp.sympify(expr)
    derivatives = [g]
    for _ in range(MAX_CONSTRAINT_ORDER):
        g = sp.diff(g, t) + sp.Add(*[sp.diff(g, x) * f for x, f in
                                     zip(problem.states, problem.dynamics)])
        if not g.has(Outlined):
            g = sp.simplify(g)
        derivatives.append(g)
        if any(not _is_zero(sp.diff(g, u)) for u in controls):
            return tuple(derivatives)
    raise ValueError(f"no control appears in the first {MAX_CONSTRAINT_ORDER} time derivatives "
                     f"of the path constraint {expr} <= 0: the controls cannot influence it")


# A path constraint longer than this written out goes on a tape.
OUTLINE_ABOVE_OPS = 200

# simplify() is kept for small expressions, where it reads best; on larger
# ones it can take minutes (trigsimp on a two-link arm's boundary arcs), and
# code generation shares their common subexpressions anyway.
SIMPLIFY_MAX_OPS = 40


def _tidy(expr):
    """``simplify(expr)`` when it is small, else ``expr`` as it is."""
    if sp.count_ops(expr) <= SIMPLIFY_MAX_OPS:
        return sp.simplify(expr)
    return expr


def _solve_linear(eqs, unknowns):
    """The unique solution of equations linear in ``unknowns``, or None.

    None when they are not linear (the caller falls back to ``sp.solve``) or
    have no unique solution.
    """
    try:
        A, b = sp.linear_eq_to_matrix(eqs, unknowns)
    except (sp.polys.polyerrors.PolyNonlinearError, ValueError):
        return None
    if A.shape[0] != A.shape[1]:
        return None
    det = sp.cancel(A.det())
    if det == 0:
        return None
    x = A.LUsolve(b)
    return dict(zip(unknowns, x))


def _boundary_arc(problem, H, costates, derivatives):
    """The :class:`BoundaryArc` of a constraint with ``g^(q) = derivatives[-1]``,
    or ``(None, reason)``."""
    order = len(derivatives) - 1
    if order > 2:
        return None, f"order {order}: exact arcs are supported for orders 1 and 2"
    if any(isinstance(cs, UnitVector) for cs in problem.control_sets):
        return None, "unit-vector control sets are not supported with exact arcs"
    if any(problem.time in e.free_symbols for e in derivatives[:order]):
        # At the entry H then jumps by pi . dN/dt, which shooting does not model.
        return None, "g, ..., g^(q-1) depend on time: exact arcs need a time-invariant limit"
    arc = _riding(problem, H, [derivatives[-1]])
    if arc is None:
        return None, "g^(q) = 0 and dH/du + mu dg^(q)/du = 0 have no unique solution for u, mu"
    return arc, ""


def _riding(problem, H, gqs, constraints=()):
    """The :class:`BoundaryArc` that rides every limit whose top derivative is in
    ``gqs`` at once, or None when the controls and multipliers are not unique.

    The controls and the multipliers ``mu`` solve ``dH/du + sum mu_i dg_i^(q)/du = 0``
    and ``g_i^(q) = 0``; the costates follow ``-d(H + sum mu_i g_i^(q))/dx``.
    """
    mus = [sp.Symbol(f"mu_b{i}" if len(gqs) > 1 else "mu_b") for i in range(len(gqs))]
    lagrangian = H + sp.Add(*[m * g for m, g in zip(mus, gqs)])
    unknowns = [*problem.controls, *mus]
    eqs = [sp.diff(lagrangian, u) for u in problem.controls] + list(gqs)
    sol = _solve_linear(eqs, unknowns)
    if sol is None:
        if len(gqs) > 1:
            return None
        try:
            sols = sp.solve(eqs, unknowns, dict=True)
        except NotImplementedError:
            sols = []
        if len(sols) != 1 or any(u not in sols[0] for u in unknowns):
            return None
        sol = sols[0]
    sol = {k: _tidy(v) for k, v in sol.items()}
    sol = _clipped_on_line(problem, lagrangian, gqs, mus, sol)
    law = {u: sol[u] for u in problem.controls}
    xdot = tuple(_tidy(f.subs(law)) for f in problem.dynamics)
    lamdot = tuple(_tidy(-sp.diff(lagrangian, x).subs(sol)) for x in problem.states)
    bounds = {cs.control: (cs.lower, cs.upper) for cs in problem.control_sets}
    multipliers = tuple(sol[m] for m in mus)
    return BoundaryArc(control=law, multiplier=multipliers[0], dynamics=xdot + lamdot,
                       hamiltonian=_tidy(H.subs(law)),
                       running_cost=_tidy(problem.running_cost.subs(law)),
                       bounds=bounds, constraints=tuple(constraints), multipliers=multipliers)


def _sample_values(symbols, n, seed=0):
    """Random values of ``symbols``: positive ones in [0.5, 2], others in [-2, 2]."""
    rng = np.random.default_rng(seed)
    return [rng.uniform(0.5, 2.0, n) if s.is_positive else rng.uniform(-2.0, 2.0, n)
            for s in symbols]


def _pivot(gqs, controls):
    """The control to keep as the free parameter of the line ``g^(q) = 0``: the one
    whose complement's coefficient matrix stays farthest from singular at random
    points (so the others can be solved for along the whole arc)."""
    best, best_score = None, -1.0
    symbols = sorted(set().union(*(g.free_symbols for g in gqs)) - set(controls), key=str)
    values = _sample_values(symbols, 64)
    for p in controls:
        others = [u for u in controls if u != p]
        A = sp.Matrix([[sp.diff(g, u) for u in others] for g in gqs])
        det = sp.lambdify(symbols, A.det(), "numpy")(*values)
        score = float(np.min(np.abs(np.broadcast_to(det, (64,)))))
        if score > best_score:
            best, best_score = p, score
    return best


def _clipped_on_line(problem, lagrangian, gqs, mus, sol):
    """``sol`` with the controls kept within their :class:`Box` bounds, when riding
    leaves a line of controls (one more control than limits ridden).

    ``g^(q) = 0`` then leaves one free parameter ``s``: the control kept free
    (:func:`_pivot`), with the others affine in it. Each bound on a control
    is an interval of ``s``, and with ``H`` convex the best point within all
    of them is the unconstrained ``s*`` clipped to their intersection. The
    multipliers come from the stationarity rows of the controls not at a
    bound, so they change branch where the clip does. Otherwise ``sol`` is
    returned as it is (and shooting reports how far the controls leave their
    bounds).
    """
    controls = list(problem.controls)
    boxes = {cs.control: cs for cs in problem.control_sets
             if isinstance(cs, Box) and cs.lower.is_finite and cs.upper.is_finite}
    if len(controls) != len(gqs) + 1 or not boxes:
        return sol
    p = _pivot(gqs, controls)
    others = [u for u in controls if u != p]
    along = _solve_linear(list(gqs), others)
    if along is None:
        return sol
    # Candidate ends of the interval of s, with the control each one saturates.
    lows, highs = [], []
    if p in boxes:
        lows.append((boxes[p].lower, p))
        highs.append((boxes[p].upper, p))
    for u in others:
        if u not in boxes:
            continue
        beta = sp.diff(along[u], p)
        if _is_zero(beta):
            continue                     # fixed by the limits alone: only checked
        alpha = along[u].subs(p, 0)
        a, b = (boxes[u].lower - alpha) / beta, (boxes[u].upper - alpha) / beta
        lows.append((lesser(a, b), u))
        highs.append((greater(a, b), u))
    if not lows:
        return sol
    s_star = sol[p]
    low = greater(*[e for e, _ in lows])
    high = lesser(*[e for e, _ in highs])
    s = lesser(greater(s_star, low), high)
    clipped = {p: s, **{u: along[u].subs(p, s) for u in others}}

    def at_end(ends, is_low):
        """Per multiplier, its value when the tightest of ``ends`` binds."""
        values = []
        for e, u in ends:
            # The rows of every control but the saturated one give the multipliers.
            rows = [sp.diff(lagrangian, v).subs(clipped) for v in controls if v != u]
            m = _solve_linear(rows, mus)
            if m is None:
                return None
            values.append(m)
        out = {}
        for mu in mus:
            # The tightest end: the largest low, or the smallest high. Each test is a
            # single comparison (SymPy's Piecewise can recurse on compound ones).
            pick = values[-1][mu]
            for k in range(len(ends) - 2, -1, -1):
                e = ends[k][0]
                # End k binds if it is at least as tight as every later one (an
                # earlier one that is tighter was picked before it).
                tests = [(e - f >= 0) if is_low else (f - e >= 0) for f, _ in ends[k + 1:]]
                pick = sp.Piecewise((values[k][mu], sp.And(*tests)), (pick, True))
            out[mu] = pick
        return out

    below, above = at_end(lows, True), at_end(highs, False)
    if below is None or above is None:
        return sol
    out = dict(clipped)
    for mu in mus:
        out[mu] = sp.Piecewise((below[mu], s_star - low < 0), (above[mu], s_star - high > 0),
                               (sol[mu], True))
    return out


def _exclusive(problem, a, b):
    """Whether limits ``a`` and ``b`` can never be active together: ``g_a + g_b``
    does not depend on the states (a lower and an upper bound on one quantity)."""
    total = sp.simplify(a.expr + b.expr)
    return not (total.free_symbols & set(problem.states)) and total != 0


def _boundary_sets(problem, H, constraints, simultaneous):
    """Boundary arcs riding several limits at once: ``((indices, BoundaryArc), ...)``.

    Every set of up to ``simultaneous`` limits with exact arcs (and at most one
    per control) whose riding controls and multipliers are unique, leaving
    out pairs that exclude each other.
    """
    exact = [i for i, c in enumerate(constraints) if c.boundary is not None]
    largest = min(int(simultaneous), len(problem.controls), len(exact))
    out = []
    for size in range(2, largest + 1):
        for idx in itertools.combinations(exact, size):
            if any(_exclusive(problem, constraints[a], constraints[b])
                   for a, b in itertools.combinations(idx, 2)):
                continue
            arc = _riding(problem, H, [constraints[i].derivatives[-1] for i in idx], idx)
            if arc is not None:
                out.append((idx, arc))
    return tuple(out)


def _outline(problem, above):
    """Put large path constraints on a tape: ``(exprs, tape, roots, root_of, sign_of)``, or None.

    None when every constraint has at most ``above`` operations written out,
    or one uses something a tape does not record; the constraints then stay
    ordinary SymPy expressions.
    """
    limits = problem.path_constraints
    if above is None or not limits or max(sp.count_ops(c.expr) for c in limits) <= above:
        return None
    variables = {problem.time, *problem.states}
    tape, rng = Tape(), np.random.default_rng(0)
    points = [{}, {}]

    def values(i):
        for point in points:
            for name in tape.input_names(i):
                point.setdefault(name, float(rng.uniform(0.5, 1.5)))
        return np.array([tape.evaluate([i], point)[0] for point in points])

    exprs, roots, seen, root_of, sign_of = [], [], [], [], []
    for c in limits:
        terms = sp.Add.make_args(c.expr)
        state = sp.Add(*[e for e in terms if e.free_symbols & variables])
        rest = sp.Add(*[e for e in terms if not e.free_symbols & variables])
        try:
            step = tape.from_sympy(state)
        except Unsupported:
            return None
        v = values(step)
        for k, other in enumerate(seen):
            scale = 1e-9 * max(1.0, float(np.max(np.abs(v))))
            if np.all(np.abs(v - other) < scale):
                sign = 1
            elif np.all(np.abs(v + other) < scale):
                sign = -1
            else:
                continue
            break
        else:
            k, sign = len(roots), 1
            roots.append(step)
            seen.append(v)
        root_of.append(k)
        sign_of.append(sign)
        exprs.append(sign * tape.outlined(roots[k]) + rest)
    return exprs, tape, tuple(roots), tuple(root_of), tuple(sign_of)


def _rides_at_run_time(problem, derivatives_of):
    """Whether boundary arcs of these limits can be solved at run time, or why not."""
    if any(not isinstance(cs, Box) for cs in problem.control_sets):
        return "unit-vector control sets are not supported with exact arcs"
    z_free_bounds = {problem.time, *problem.states}
    if any((cs.lower.free_symbols | cs.upper.free_symbols) & z_free_bounds
           for cs in problem.control_sets):
        return "limits this large need control bounds free of the states and time"
    if any(len(d) != 2 for d in derivatives_of):
        return "limits this large are supported at order 1"
    t, u = problem.time, problem.controls
    z_free = {problem.time, *problem.states}
    H_like = [problem.running_cost, *problem.dynamics]
    if any(t in e.free_symbols for e in H_like):
        return "limits this large need autonomous dynamics and cost"
    if any(sp.diff(f, a, b) != 0 for f in problem.dynamics for a in u for b in u):
        return "limits this large need dynamics affine in the controls"
    second = [sp.diff(problem.running_cost, a, b) for a in u for b in u]
    if any(e.free_symbols & z_free for e in second):
        return "limits this large need a running cost quadratic in the controls"
    if sp.Matrix(len(u), len(u), second).is_zero_matrix:
        return "limits this large need a running cost strictly convex in the controls"
    return ""


def _sets_at_run_time(root_of, n_controls, simultaneous):
    """Each set of up to ``simultaneous`` limits, at most one per root and per control."""
    by_root = {}
    for c, k in enumerate(root_of):
        by_root.setdefault(k, []).append(c)
    largest = min(simultaneous, n_controls, len(by_root))
    out = []
    for size in range(2, largest + 1):
        for chosen in itertools.combinations(sorted(by_root), size):
            for limits in itertools.product(*[by_root[k] for k in chosen]):
                out.append((tuple(sorted(limits)), None))
    return tuple(out)


def _state_constraints(problem, H=None, costates=(), outlined=None):
    out = []
    exprs = [c.expr for c in problem.path_constraints] if outlined is None else outlined[0]
    derivatives_of = [constraint_derivatives(problem, e) for e in exprs]
    why_not = _rides_at_run_time(problem, derivatives_of) if outlined is not None else ""
    for c, expr, derivatives in zip(problem.path_constraints, exprs, derivatives_of):
        order = len(derivatives) - 1
        if order >= 3:
            warnings.warn(
                f"path constraint {c.name!r} has order {order}. Optimal solutions that ride a "
                "constraint of order 3 or more generally chatter (infinitely many touch points), "
                "so only its penalty and termination are meaningful", stacklevel=3)
        if outlined is not None:
            arc, why = (None, why_not) if why_not else (_RiddenAtRunTime(), "")
        else:
            arc, why = (None, "") if H is None else _boundary_arc(problem, H, costates, derivatives)
        out.append(StateConstraint(name=c.name, expr=expr, order=order,
                                   derivatives=derivatives, penalty=c.penalty, boundary=arc,
                                   unsupported=why))
    return tuple(out)


def _singular_arc(problem, H, law, costates, sw):
    """The :class:`SingularArc` of switching function ``sw``, or ``(None, reason)``.

    The singular expressions are not simplified: ``simplify`` on the
    singular control of Goddard's rocket takes minutes, and code
    generation shares the common subexpressions anyway.
    """
    if any(isinstance(cs, UnitVector) for cs in problem.control_sets):
        return None, "unit-vector control sets are not supported with singular arcs"
    u = sw.control
    others = {w: e for w, e in law.items() if w != u}
    if any(e.has(sp.Piecewise, sp.Min, sp.Max) for e in others.values()):
        return None, "the other controls must have smooth laws (no bounds) on a singular arc"
    t, z = problem.time, problem.states + costates
    # The flow with u symbolic: H is linear in u, so u enters only there.
    F = [f.subs(others) for f in problem.dynamics] + [
        -sp.diff(H, x).subs(others) for x in problem.states]

    def rate(e):
        return sp.diff(e, t) + sp.Add(*[sp.diff(e, zi) * fi for zi, fi in zip(z, F)])

    S = sw.expr
    S1 = rate(S)
    if not _is_zero(sp.diff(S1, u)):         # cannot happen for one control (Kelley et al.)
        return None, f"S' depends on {u}"
    S1 = sp.factor_terms(S1.subs(u, 0))
    S2 = rate(S1)
    b = sp.diff(S2, u)
    if not _is_zero(sp.diff(b, u)):
        return None, f"S'' is not linear in {u}"
    if _is_zero(b):
        return None, (f"{u} does not appear in S'': the arc has order 2 or more, where "
                      "bang-singular junctions chatter (Fuller's phenomenon)")
    a = S2.subs(u, 0)
    u_s = -a / b
    sing = {u: u_s, **{w: e.subs(u, u_s) for w, e in others.items()}}
    xdot = tuple(f.subs(sing) for f in problem.dynamics)
    lamdot = tuple(-sp.diff(H, x).subs(sing) for x in problem.states)
    return SingularArc(control=u, derivatives=(S, S1, S2), law=sing, dynamics=xdot + lamdot,
                       hamiltonian=H.subs(sing), running_cost=problem.running_cost.subs(sing),
                       legendre_clebsch=-b, lower=sw.lower,
                       upper=sw.upper), ""


def derive(problem, singular=False, simultaneous=None, outline_above=OUTLINE_ABOVE_OPS):
    """Derive the first-order necessary conditions of ``problem``.

    With ``singular=True``, also derive the singular arc of each bang-bang
    control (:class:`SingularArc`), for exact shooting over structures with
    singular arcs. ``simultaneous`` is the largest number of path
    constraints an exact arc may ride at once (:attr:`PontryaginSystem.boundary_sets`);
    1 derives single limits only; by default it is 2, or every set when
    the arcs are solved at run time. A path constraint with more than
    ``outline_above`` operations written out goes on a tape
    (:mod:`~auto_optimal_control.tape`), and its arcs, when they qualify, are
    solved at run time (:mod:`~auto_optimal_control.riding`); ``None`` keeps
    everything symbolic. Returns a :class:`PontryaginSystem`.
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

    # Path-constraint penalties depend on the state only, so they leave the
    # control law alone and add w max(0, g) dg/dx to -lambda'.
    outlined = _outline(problem, outline_above)
    constraints = _state_constraints(problem, H, costates, outlined)
    penalized = [c for c in constraints if c.penalty is not None]
    penalty = sp.Add(*[c.penalty / 2 * sp.Max(0, c.expr) ** 2 for c in penalized])
    costate_dynamics = tuple(
        e - sp.Add(*[c.penalty * sp.Max(0, c.expr) * sp.diff(c.expr, s) for c in penalized])
        for e, s in zip(costate_dynamics, problem.states))

    singular_arcs, singular_unsupported = [], []
    for sw in switching:
        arc, why = (_singular_arc(problem, H, law, costates, sw) if singular
                    else (None, "not derived: use derive(problem, singular=True)"))
        singular_arcs.append(arc)
        singular_unsupported.append(why)

    at_run_time = outlined is not None and all(isinstance(c.boundary, _RiddenAtRunTime)
                                               for c in constraints)
    riding = None
    if at_run_time:
        riding = RidingSystem(*outlined[1:])
        sets = _sets_at_run_time(riding.root_of, len(problem.controls),
                                 len(problem.controls) if simultaneous is None else simultaneous)
    elif outlined is not None:
        sets = ()
    else:
        simultaneous = 2 if simultaneous is None else simultaneous
        sets = _boundary_sets(problem, H, constraints, simultaneous) if simultaneous > 1 else ()
    boundary = _boundary_conditions(problem, H_star + penalty, costates, taken)
    return PontryaginSystem(
        problem=problem, time=problem.time, states=problem.states, costates=costates,
        controls=problem.controls, params=problem.params, hamiltonian_full=H + penalty,
        control_law=law, switching_functions=switching, hamiltonian=H_star + penalty,
        running_cost=sp.simplify(problem.running_cost.subs(law)) + penalty,
        state_dynamics=state_dynamics, costate_dynamics=costate_dynamics, boundary=boundary,
        path_constraints=constraints, penalty=penalty, singular_arcs=tuple(singular_arcs),
        singular_unsupported=tuple(singular_unsupported), boundary_sets=sets, riding=riding)
