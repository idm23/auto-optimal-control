"""Problems whose optimal motion is not unique, found from the statement, and what to do about them.

Minimum time on several axes fixes the time but not the motion (Lecture 18).
One axis sets the time. Every motion of the others that finishes within it
is optimal, their costates vanish, and shooting has a singular Jacobian. The
same happens whenever some controlled part of the problem appears in no cost
term. Both can be read off the problem statement, before anything is solved.

:func:`analyze` splits the states and controls into **components**. Two
variables are in the same component when one appears in the other's
dynamics, or both appear in one cost term, terminal constraint or
:class:`~auto_optimal_control.problem.UnitVector`. A component is **priced**
when some term of the cost involves it. The problem is **degenerate** when a
controlled component is unpriced, unless it is the only controlled component
and the cost is the (free) final time. One axis on its own, in minimum time,
is fine. Two axes in minimum time are not: only the slower one sets the time.

Path constraints are not counted as coupling. They couple their states only
while they are active, and on the free arcs between them the argument above
applies unchanged. A two-link arm whose joints are coupled only through
torque limits is degenerate in minimum time. :class:`Analysis` reports which
limits link the components.

A control that enters the Hamiltonian linearly and has no bound on one side
is also flagged. The Hamiltonian has no minimum in it unless its switching
function vanishes.

A degenerate problem needs a **policy**, a rule for which of the equally
good motions to return:

- :class:`SecondaryCost` (the default): add ``w int sum (u / s)^2 / 2`` over
  the controls of the unpriced components and the unbounded linear ones,
  with ``s`` each control's bound (1 when unbounded). It is well posed for
  any ``w > 0``. It selects the smoothest of the fastest motions, at the
  price of a final time slightly above the minimum. The costates of the
  slack parts scale like ``w`` and the conditioning like ``1 / w``, so ``w``
  cannot be taken to zero.
- :class:`Lexicographic`: first the minimum time ``T*``, from each component
  solved on its own (exact when the components are fully separate), then
  ``t_f = (1 + delta) T*`` fixed and the effort minimized. Only for pure
  minimum-time problems whose components no path constraint links.
- ``"as_stated"``: solve the problem as given (for a flag known to be
  harmless).
- ``"strict"``: raise :class:`IllPosedProblem` instead.

``"auto"`` picks :class:`SecondaryCost` for a degenerate problem and nothing
otherwise. :class:`~auto_optimal_control.homotopy.Homotopy` applies it
automatically and records it in the result.
"""

from dataclasses import dataclass, field

import sympy as sp

from auto_optimal_control.problem import Box, OptimalControlProblem, UnitVector


class IllPosedProblem(ValueError):
    """The problem's optimal motion is not unique, and the policy is ``"strict"``."""


# =============================================================================
# Analysis
# =============================================================================

def _terms(expr):
    """The additive terms of ``expr``, expanded when that is cheap."""
    expr = sp.sympify(expr)
    if expr == 0:
        return ()
    if sp.count_ops(expr) <= 200:
        expr = sp.expand(expr)
    return tuple(t for t in sp.Add.make_args(expr) if t != 0)


class _Partition:
    def __init__(self, items):
        self.parent = {s: s for s in items}

    def find(self, s):
        while self.parent[s] != s:
            self.parent[s] = self.parent[self.parent[s]]
            s = self.parent[s]
        return s

    def join(self, symbols):
        symbols = [s for s in symbols if s in self.parent]
        for a, b in zip(symbols, symbols[1:]):
            self.parent[self.find(a)] = self.find(b)

    def groups(self, order):
        out = {}
        for s in order:
            out.setdefault(self.find(s), []).append(s)
        return list(out.values())


@dataclass(frozen=True)
class Component:
    """States and controls that no dynamics, cost term or end condition connects to the rest."""

    states: tuple
    controls: tuple
    priced: bool


@dataclass
class Analysis:
    """What :func:`analyze` found. ``degenerate`` says whether a policy is needed.

    ``components`` lists every component; ``slack`` the controlled ones no cost
    term involves; ``linked_by`` the names of the path constraints that span
    several components (they couple them only while active); ``unbounded``
    the controls the Hamiltonian is linear in with no bound on some side;
    ``time_priced`` whether the cost has a final-time part and ``t_f`` is
    free; ``reasons`` the findings in words.
    """

    components: tuple
    slack: tuple
    linked_by: tuple
    unbounded: tuple
    time_priced: bool
    free_final_time: bool
    reasons: list = field(default_factory=list)

    @property
    def degenerate(self):
        return bool(self.reasons)

    @property
    def controlled(self):
        return tuple(c for c in self.components if c.controls)

    @property
    def separable(self):
        """Several controlled components and no path constraint linking them."""
        return len(self.controlled) > 1 and not self.linked_by

    @property
    def pure_minimum_time(self):
        """The cost is only the final time (no component is priced)."""
        return self.time_priced and not any(c.priced for c in self.components)

    def __str__(self):
        if not self.degenerate:
            return "well posed: every controlled part of the problem is priced"
        return "not unique as stated:\n" + "\n".join(f"- {r}" for r in self.reasons)


def _names(symbols):
    return ", ".join(str(s) for s in symbols)


def _unbounded_linear(problem):
    """Controls ``H`` is affine in, with an infinite bound (or none) on some side."""
    bounded = set()
    for cs in problem.control_sets:
        if isinstance(cs, UnitVector):
            bounded.update(cs.controls)
        elif isinstance(cs, Box) and cs.lower.is_finite and cs.upper.is_finite:
            bounded.add(cs.control)
    out = []
    for u in problem.controls:
        if u in bounded:
            continue
        exprs = (problem.running_cost,) + problem.dynamics
        if all(sp.simplify(sp.diff(e, u, 2)) == 0 for e in exprs if e.has(u)):
            out.append(u)
    return tuple(out)


def analyze(problem):
    """Find, from the statement alone, whether ``problem`` has a unique optimal motion.

    Returns an :class:`Analysis`; see the module docs for the rules.
    """
    x, u = problem.states, problem.controls
    variables = set(x) | set(u)
    part = _Partition(x + u)
    for s, f in zip(x, problem.dynamics):
        part.join([s] + sorted(f.free_symbols & variables, key=str))
    for c in problem.terminal_constraints:
        part.join(sorted(c.free_symbols & variables, key=str))
    for cs in problem.control_sets:
        if isinstance(cs, UnitVector):
            part.join(list(cs.controls))
    cost_terms = _terms(problem.running_cost) + _terms(problem.terminal_cost)
    priced_vars = set()
    for term in cost_terms:
        used = sorted(term.free_symbols & variables, key=str)
        part.join(used)
        priced_vars.update(used)

    time_part = any(not (term.free_symbols & variables)
                    for term in _terms(problem.running_cost))
    time_part = time_part or any(term.has(problem.time) and not (term.free_symbols & variables)
                                 for term in _terms(problem.terminal_cost))
    time_priced = time_part and problem.free_final_time

    components = []
    for group in part.groups(x + u):
        states = tuple(s for s in group if s in x)
        controls = tuple(s for s in group if s in u)
        components.append(Component(states, controls, bool(set(group) & priced_vars)))
    components = tuple(components)
    controlled = [c for c in components if c.controls]
    slack = tuple(c for c in controlled if not c.priced)

    where = {s: k for k, c in enumerate(components) for s in c.states + c.controls}
    linked = tuple(pc.name for pc in problem.path_constraints
                   if len({where[s] for s in pc.expr.free_symbols & variables}) > 1)
    unbounded = _unbounded_linear(problem)

    reasons = []
    if slack and not (len(controlled) == 1 and time_priced):
        if time_priced:
            reasons.append(
                f"{len(controlled)} separate controlled parts share only the final time "
                f"({'; '.join(_names(c.controls) for c in controlled)}): only the slowest "
                "sets it, and every motion of the others that finishes in time is optimal")
        for c in slack:
            if not time_priced:
                reasons.append(f"no cost term involves the controls {_names(c.controls)} "
                               "or their states: any feasible motion of them is optimal")
        if linked:
            reasons.append(f"the path constraints {', '.join(linked)} link these parts, but "
                           "only while they are active; between them the freedom remains")
    if unbounded:
        reasons.append(f"the Hamiltonian is linear in {_names(unbounded)}, which "
                       f"{'is' if len(unbounded) == 1 else 'are'} not bounded on both sides: "
                       "it has no minimum unless the switching function vanishes")
    return Analysis(components, slack, linked, unbounded, time_priced,
                    problem.free_final_time, reasons)


# =============================================================================
# Policies
# =============================================================================

def _scale(problem, u):
    for cs in problem.control_sets:
        if isinstance(cs, Box) and cs.control == u:
            bounds = [abs(b) for b in (cs.lower, cs.upper) if b.is_finite and b != 0]
            if bounds:
                return sp.Max(*bounds) if len(bounds) > 1 else bounds[0]
    return sp.S.One


def _effort(problem, controls):
    return sum(((u / _scale(problem, u)) ** 2 for u in controls), sp.S.Zero) / 2


def _rebuilt(problem, *, name, running_cost, params=None, final_time="keep", states=None,
             controls=None, dynamics=None, initial_state=None, terminal_constraints=None,
             control_sets=None, path_constraints=None, terminal_cost=None):
    return OptimalControlProblem(
        name=name, time=problem.time,
        states=problem.states if states is None else states,
        controls=problem.controls if controls is None else controls,
        dynamics=problem.dynamics if dynamics is None else dynamics,
        running_cost=running_cost,
        terminal_cost=problem.terminal_cost if terminal_cost is None else terminal_cost,
        initial_state=dict(problem.initial_state) if initial_state is None else initial_state,
        terminal_constraints=(problem.terminal_constraints if terminal_constraints is None
                              else terminal_constraints),
        initial_time=problem.initial_time,
        final_time=problem.final_time if final_time == "keep" else final_time,
        params=problem.params if params is None else params,
        control_sets=problem.control_sets if control_sets is None else control_sets,
        path_constraints=problem.path_constraints if path_constraints is None
        else path_constraints)


def _penalized(analysis):
    # Every unpriced component: which one sets the time is not known in advance.
    controls = [u for c in analysis.slack for u in c.controls]
    controls += [u for u in analysis.unbounded if u not in controls]
    return tuple(controls)


@dataclass(frozen=True)
class SecondaryCost:
    """Add ``weight * int sum (u / s)^2 / 2`` over the controls that need it.

    ``s`` is each control's largest finite bound (1 when it has none), so the
    weight is relative to a cost of order one per unit time. ``weight`` is a
    number or an expression in the problem's parameters (to sweep it at
    runtime). ``controls`` overrides which controls are penalized.
    """

    weight: object = 1e-2
    controls: tuple = None

    def apply(self, problem, analysis):
        controls = _penalized(analysis) if self.controls is None else tuple(self.controls)
        cost = problem.running_cost + sp.sympify(self.weight) * _effort(problem, controls)
        return _rebuilt(problem, name=(problem.name or "problem") + "_secondary",
                        running_cost=cost)

    def describe(self, analysis):
        controls = _penalized(analysis) if self.controls is None else self.controls
        return f"secondary cost {self.weight} * int (u / s)^2 / 2 on {_names(controls)}"


@dataclass(frozen=True)
class Lexicographic:
    """Minimum time first, then minimum effort at ``t_f = (1 + delta) T*``.

    ``T*`` is the largest of the components' own minimum times, each solved on
    its own (well posed, and exact). Then the final time is fixed and the cost
    becomes ``int sum (u / s)^2 / 2`` over every control. Each stage has one
    answer; the second has room to choose only because ``delta > 0``.
    Needs a pure minimum-time problem whose components no path constraint
    links, and bounded controls.
    """

    delta: float = 1e-2

    def check(self, analysis):
        if not analysis.pure_minimum_time:
            raise ValueError("the lexicographic policy needs a pure minimum-time problem")
        if analysis.linked_by:
            raise ValueError(f"the path constraints {', '.join(analysis.linked_by)} link the "
                             "components, so their minimum times cannot be found apart; "
                             "use SecondaryCost")
        if analysis.unbounded:
            raise ValueError(f"the controls {_names(analysis.unbounded)} are unbounded: "
                             "the minimum time is not attained; use SecondaryCost")
        if self.delta < 0:
            raise ValueError("delta must be >= 0")

    def stages(self, problem, analysis):
        """``(components, final)``: one minimum-time problem per controlled component, with
        its state indices, and the fixed-time effort problem (its last parameter is
        ``t_f``)."""
        self.check(analysis)
        index = {s: i for i, s in enumerate(problem.states)}
        own = []
        for k, c in enumerate(analysis.controlled):
            states = set(c.states)
            idx = [index[s] for s in c.states]
            within = lambda e: (e.free_symbols & set(problem.states)) <= states
            sub = _rebuilt(
                problem, name=f"{problem.name or 'problem'}_part{k}",
                running_cost=problem.running_cost, states=c.states, controls=c.controls,
                dynamics=[problem.dynamics[i] for i in idx],
                initial_state={s: v for s, v in problem.initial_state.items() if s in states},
                terminal_constraints=[e for e in problem.terminal_constraints
                                      if e.free_symbols & states and within(e)],
                control_sets=[cs for cs in problem.control_sets
                              if set(cs.controls) <= set(c.controls)],
                path_constraints=[pc for pc in problem.path_constraints
                                  if pc.expr.free_symbols & states])
            own.append((sub, idx))
        tf = sp.Symbol("t_f_lex", positive=True)
        final = _rebuilt(problem, name=(problem.name or "problem") + "_lexicographic",
                         running_cost=_effort(problem, problem.controls),
                         terminal_cost=sp.S.Zero, params=problem.params + (tf,),
                         final_time=tf)
        return own, final

    def describe(self, analysis):
        return (f"lexicographic: minimum time T* per component, then minimum effort at "
                f"t_f = (1 + {self.delta}) T*")


POLICIES = ("auto", "as_stated", "strict")


def resolve(policy, analysis):
    """The policy to apply: ``None`` (solve as stated), or a policy object.

    ``"auto"``: :class:`SecondaryCost` when ``analysis.degenerate``, else
    ``None``. ``"as_stated"``: ``None``. ``"strict"``: ``None``, or raise
    :class:`IllPosedProblem` when degenerate. A policy object is returned as
    it is (when the problem is not degenerate too: the user asked for it).
    """
    if isinstance(policy, (SecondaryCost, Lexicographic)):
        return policy
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, SecondaryCost or Lexicographic, "
                         f"got {policy!r}")
    if policy == "as_stated" or not analysis.degenerate:
        return None
    if policy == "strict":
        raise IllPosedProblem(f"{analysis}\nChoose a policy: SecondaryCost(weight), "
                              "Lexicographic(delta), or 'as_stated' to solve it anyway.")
    return SecondaryCost()
