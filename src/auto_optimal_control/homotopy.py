"""From a costate distribution to the exact extremal, with no other input.

:func:`solve` (or :class:`Homotopy`, to keep the compiled kernels for several
solves) takes the problem, its parameter values and a distribution of
initial costates (and of ``t_f`` when it is free). Everything else is found:

0. **Check the statement.** :func:`~auto_optimal_control.policy.analyze`
   reads whether the optimal motion is unique (it is not when some
   controlled part of the problem is in no cost term, as in minimum time
   on several axes). If it is not, a policy chooses among the equally good
   motions: by default a small secondary cost, with a warning (see
   :mod:`~auto_optimal_control.policy` and ``Homotopy(policy=...)``).

1. **Regularize.** :func:`regularize` adds a smoothing term
   ``eps ((u - mid) / half)^2 / 2`` to the running cost for each bang-bang
   control (its law becomes clipped-linear), and for each path constraint
   an exterior penalty ``rho / 2 max(0, g)^2`` and an offset (the limit
   becomes ``g - off <= 0``), all runtime parameters.
2. **Sample.** Single shooting from random costates on the regularized
   problem, at ``eps = 1``, with every limit off.
   A free final time with separate components needs no draws: each follows its
   costates from a long time down to its own minimum time, and they share ``t_f``.
3. **The active-set route** (problems with path constraints). Exact arcs on
   the smoothed problem, built up as the limits are tightened:

   - each limit's offset starts just above the unconstrained solution and
     falls to zero in adaptive steps (a secant predictor starts each solve);
   - when a free arc starts to violate a limit, the structure changes
     there: a newborn touch point (order 2 or more) or short boundary arc
     (order 1) with zero jump; a touch point violated beside it grows into a
     boundary arc or splits into two (order 3 or more: another touch point,
     since such limits have no arcs with a smooth control law); neighbouring arcs of one limit merge;
     an arc that shrinks to nothing, or a touch point whose jump turns
     negative, is retired;
   - then ``eps`` falls, in decades, with the structure following, and the
     *exact* problem is tried with the current structure at every step.

   Arcs are born where a limit first comes into play, small and with small
   jumps, which is where shooting with exact arcs is well conditioned. The
   route ends at the first exact solution accepted (step 6). It goes first
   unless the exact problem may have singular arcs (which only the penalty
   route reads); each route is the other's fallback, and the active-set
   route has a time budget, ``Homotopy.active_set_seconds``.
4. **The penalty route.** When a limit is first violated, ``eps`` is held
   while its penalty weight ramps up, in decades, from ``1e-6 rho_ref`` to
   ``1e4 rho_ref`` (``rho_ref`` makes the penalty of the violation it saw as
   large as the cost); then ``eps`` falls, in decades, with the penalties
   held. A ramp that stalls is held where it got to. The steps adapt: they
   grow after a success and halve after a failure, and a secant predictor
   starts each solve.
5. **Read the structure and jump** (penalty route). After each step the
   regularized solution is read: the intervals where each limit is violated
   become boundary arcs (or touch points, for order 2 or more), and stretches where
   a bang-bang control stays strictly between its bounds become singular
   arcs (or, when short, the ramps of ordinary switches). Each reading
   gives candidate structures for
   :class:`~auto_optimal_control.shooting.MultiArcShooting` on the *exact*
   problem, seeded from the regularized costates, junction times and
   fitted jump multipliers, plus a few seeds with jittered times. This is
   the route that finds singular arcs.
6. **Accept.** A candidate is accepted when the exact solve converges and
   passes the necessary conditions it can check (no violation on free arcs,
   controls within bounds on special arcs, ``mu >= 0`` on boundary arcs,
   ``nu >= 0`` at touch points, Legendre-Clebsch, the held bounds agreeing
   with the switching function). Among the candidates accepted at one step,
   the cheapest wins.

Bang-bang switches need no reading at all: on free arcs they are events of
the integration. Without bang-bang controls or path constraints the
regularized problem is the exact one and step 2 is the whole solve.

When no candidate is accepted (a limit of order 4 or more, a structure outside
what :class:`MultiArcShooting` represents, a search out of time),
:func:`solve` returns the last regularized solution with ``exact = False``. A
fixed end state that already violates a limit is reported the same way, at once.
"""

import itertools
import time
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import sympy as sp

from auto_optimal_control.build import build_kernels
from auto_optimal_control.policy import (Lexicographic, SecondaryCost, _effort, _penalized,
                                         analyze, resolve)
from auto_optimal_control.pontryagin import OUTLINE_ABOVE_OPS, bang_bang_controls, derive
from auto_optimal_control.problem import Box, OptimalControlProblem, PathConstraint
from auto_optimal_control.diagnostics import Diagnostics
from auto_optimal_control.pipeline import (PREDICT, SETTLE, SETTLE_START, STEP_ACCEPTED_SIZE,
                                           STEP_FAILED_RESCUE, STEP_FAILED_SIZE,
                                           STEP_FAILED_SPLIT, Accept, Reject, Rescued, Retry,
                                           SettleContext, StepContext, Turn, default_pipelines,
                                           failure_pipelines, settle_pipelines, EXACT_REJECTED,
                                           ExactContext, exact_pipelines)
from auto_optimal_control.shooting import MultiArcShooting, ShootingError, SingleShooting, sample_segment

# The path of the continuation is followed to 1e-6: a rough Jacobian steers Newton as well.
PATH_JACOBIAN_RTOL = 1e-7
SEED_ACCEPT = 100.0                # a seed is a guess: how far above tol a stalled Newton may stop

EPS = "eps_reg"
RHO = "rho_reg_"
OFF = "off_reg_"
WEIGHT = "w_secondary"
_FAILURES = (ShootingError, RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError)


# =============================================================================
# The regularized and exact problems
# =============================================================================

def _copy(problem, *, name, running_cost, params, path_constraints):
    return OptimalControlProblem(
        name=name, time=problem.time, states=problem.states, controls=problem.controls,
        dynamics=problem.dynamics, running_cost=running_cost,
        terminal_cost=problem.terminal_cost, initial_state=dict(problem.initial_state),
        terminal_constraints=problem.terminal_constraints, initial_time=problem.initial_time,
        final_time=problem.final_time, params=params, control_sets=problem.control_sets,
        path_constraints=path_constraints, certified=problem.certified)


def regularize(problem, bang_bang=None):
    """The problem with a smoothing weight, and a penalty weight and an offset per path constraint.

    Each bang-bang control ``u`` in ``[lo, hi]`` gets
    ``eps ((u - mid) / half)^2 / 2`` in the running cost (``mid`` and
    ``half`` the middle and half-width of its range), so its law becomes
    ``clip(mid - half^2 S / eps)``. Each path constraint ``g <= 0`` gets the
    exterior penalty weight ``rho_reg_<name>`` and becomes
    ``g - off_reg_<name> <= 0``: an offset loosens the limit, and tightening
    it back to zero is how :class:`Homotopy` brings a limit into play. All
    are runtime parameters, appended to ``problem.params`` in that order
    (``eps_reg`` first, when there are bang-bang controls, then the weights,
    then the offsets). Penalty weights the problem gives itself are replaced.
    ``bang_bang`` defaults to :func:`bang_bang_controls`.

    Returns ``(problem, eps, rhos)``: the regularized problem, the ``eps``
    symbol (``None`` without bang-bang controls) and the ``rho`` symbols.
    """
    if bang_bang is None:
        bang_bang = bang_bang_controls(problem)
    params, cost, eps = list(problem.params), problem.running_cost, None
    if bang_bang:
        eps = sp.Symbol(EPS, positive=True)
        params.append(eps)
        boxes = {cs.control: cs for cs in problem.control_sets if isinstance(cs, Box)}
        for u in bang_bang:
            lo, hi = boxes[u].lower, boxes[u].upper
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            cost = cost + eps * ((u - mid) / half) ** 2 / 2
    rhos = tuple(sp.Symbol(RHO + c.name, nonnegative=True) for c in problem.path_constraints)
    offs = tuple(sp.Symbol(OFF + c.name, real=True) for c in problem.path_constraints)
    limits = [PathConstraint(c.expr - o, c.name, penalty=r)
              for c, r, o in zip(problem.path_constraints, rhos, offs)]
    reg = _copy(problem, name=(problem.name or "problem") + "_regularized", running_cost=cost,
                params=params + list(rhos) + list(offs), path_constraints=limits)
    return reg, eps, rhos


def exact(problem):
    """The problem with its path-constraint penalties removed."""
    return _copy(problem, name=problem.name or "problem", running_cost=problem.running_cost,
                 params=problem.params,
                 path_constraints=[PathConstraint(c.expr, c.name)
                                   for c in problem.path_constraints])


# =============================================================================
# Results
# =============================================================================

@dataclass
class Step:
    """One continuation step: the weights, the residual and what was tried.

    ``s`` is how many decades ``eps`` has fallen from its start (0 without
    bang-bang controls), ``rho`` the penalty weights, ``candidates`` how many
    exact solves were tried from this step's reading, ``accepted`` whether
    one passed. On the active-set route ``structure`` is the set of arcs
    the smoothed solution has at that step.
    """

    s: float
    eps: float
    rho: np.ndarray
    residual: float
    candidates: int = 0
    accepted: bool = False
    structure: tuple = ()


@dataclass
class HomotopyResult:
    """The result of :meth:`Homotopy.solve`.

    ``exact`` says whether ``solution`` solves the exact problem (with
    ``shooting``, a :class:`SingleShooting` or :class:`MultiArcShooting` on
    the exact kernels) or is the last regularized solution (``shooting`` on
    the regularized kernels, at ``params``). ``arcs`` is the structure found,
    as :class:`MultiArcShooting` entries (empty: free arcs only, bang-bang
    switches included). ``objective`` is the exact cost ``phi + int L`` of
    an exact solution. ``steps`` logs the continuation, ``tries`` the random
    draws of the first solve, and ``message`` says how it ended. ``policy``
    describes the policy applied to a degenerate problem (empty: solved as
    stated); with :class:`~auto_optimal_control.policy.Lexicographic`,
    ``stages`` holds the minimum-time result of each component.
    """

    solution: object
    shooting: object
    arcs: tuple
    exact: bool
    objective: float
    params: np.ndarray
    steps: list = field(default_factory=list)
    tries: int = 0
    message: str = ""
    policy: str = ""
    stages: list = field(default_factory=list)
    #: What the solve ran into (:class:`~auto_optimal_control.diagnostics.Diagnostics`).
    diagnostics: dict = field(default_factory=dict)

    @property
    def success(self):
        return self.exact


# =============================================================================
# The solver
# =============================================================================

class Homotopy:
    """The regularized and exact kernels of one problem, and the automatic solve.

    The kernels do not depend on the parameter values, so one instance
    serves every :meth:`solve`.

    Parameters
    ----------
    problem : OptimalControlProblem
    cache_dir : path, optional
        Where the compiled kernels are cached.
    rtol, atol : float
        Integration tolerances of the shooting.
    verbose : bool or int
        Print each step; 2 also prints every solve of the active-set route.
    policy : str or policy object
        What to do when the problem's optimal motion is not unique as stated
        (:mod:`~auto_optimal_control.policy`): ``"auto"`` (a
        :class:`~auto_optimal_control.policy.SecondaryCost` when needed),
        ``"as_stated"``, ``"strict"`` (raise), or a ``SecondaryCost`` or
        ``Lexicographic``. ``analysis`` holds what was found and
        ``problem`` the problem actually solved.
    outline_above : int or None
        Path constraints with more operations than this go on a tape and
        their arcs are solved at run time (:func:`~auto_optimal_control.pontryagin.derive`).
    """

    def __init__(self, problem, *, cache_dir=None, rtol=1e-12, atol=1e-12, verbose=False,
                 policy="auto", outline_above=OUTLINE_ABOVE_OPS):
        self.stated = problem
        self.analysis = analyze(problem)
        self.policy = resolve(policy, self.analysis)
        self.policy_text = "" if self.policy is None else self.policy.describe(self.analysis)
        self._stages = None
        if isinstance(self.policy, SecondaryCost):
            problem = self.policy.apply(problem, self.analysis)
        elif isinstance(self.policy, Lexicographic):
            parts, problem = self.policy.stages(problem, self.analysis)
            self._stages = [(Homotopy(sub, cache_dir=cache_dir, rtol=rtol, atol=atol,
                                      verbose=verbose, policy="as_stated",
                                      outline_above=outline_above), idx)
                            for sub, idx in parts]
        if self.policy is not None and policy == "auto":
            name = self.stated.name or "problem"
            warnings.warn(f"{name}: {self.analysis}\nApplied the default policy, "
                          f"{self.policy_text}. Pass policy= to choose another.", stacklevel=2)
        self.problem = problem
        self.params = None
        self.cache_dir = cache_dir
        self.options = dict(rtol=rtol, atol=atol)
        self.verbose = verbose
        self.attempt_evaluations = 60      # MINPACK evaluations per exact attempt
        self.attempt_nfev = 50000          # right-hand-side evaluations per arc
        #: Right-hand-side evaluations per boundary arc solved at run time. A riding that
        #: can be held needs a few dozen to a few thousand (the census's arm, measured on
        #: five hard cases: medians 14 to 42, at most 610 with the STM and 3914 without);
        #: one that cannot runs away with ever shorter steps until a cap stops it, which
        #: at ``attempt_nfev`` cost 4 to 5 s per failed seed with the STM.
        self.riding_nfev = 10000
        #: The same with the STM, where each evaluation is dearer: holdable ridings needed
        #: at most 610 there, and a runaway costs about 0.1 ms per evaluation.
        self.riding_stm_nfev = 2000
        self.active_set_seconds = 60.0     # time budget of the active-set route (see below)
        #: Wall-clock budget of a whole solve, the reversed problem (rule R4) and the
        #: parallel routes' grace included; None leaves each route its own budget.
        self.solve_seconds = None
        self._solve_end = None
        self.active_set_backtracks = 8     # structures the active-set route may back out of
        # Where the limit offsets start: "together" (just above every peak) or "staggered"
        # (rule R1: a random head start per limit, drawn with offset_seed).
        self.offset_start = "together"
        self.offset_seed = 0
        # Rule R4: when a solve fails and the problem is certified time-reversible, solve
        # the reversed problem and map its extremal back (Lemma F).
        self.reversal = True
        self.split_rising = True
        self.holdable_births = True
        self.width_steps = False
        self.merge_touching = True
        self.offset_order = "joint"
        #: Rule R15: routes to try after the first fails, as attribute settings (each
        #: gets an equal share of the time left); empty for one route.
        self.portfolio = []
        self.stall_seconds = 60.0
        #: The path step's rules at each hook (``pipeline.py``): a hook's behaviour is
        #: the composition of its pipeline's rules that are on.
        self.pipelines = {**default_pipelines(), **settle_pipelines(), **failure_pipelines(),
                          **exact_pipelines()}
        #: Run the portfolio's routes at once in forked processes (one core each).
        self.parallel_routes = False
        self.corner_touches = False         # rule R12: refuted by Lemma L (see the note)
        self.corner_rel = 1e-2              # R12: how near the edge a peak must be
        #: Rule R16: after a step that overshoots a birth, place the next one at the
        #: birth by a secant on the peak excess (Lemma N), not by halving.
        self.locate_births = False
        #: Rule R25: after a step that crosses a split (an order-1 arc's density turns
        #: negative inside it), place the next one at the split by a secant on the
        #: density (V2), not by settling the split far from it.
        self.locate_splits = False
        #: How large an event inside a step may be and still be repaired where the step
        #: landed (relative: a birth's peak excess to the limit's scale, a split's rise
        #: to max(1, |mu|)); a larger one is located first (R16, R25).
        self.locate_tol = 1e-3
        #: Rule R26: a violation near an arc of its limit is born as a new arc when the
        #: limit is strictly below its level somewhere between them (event E1, Lemma E),
        #: rather than widening the arc over it.
        self.separate_births = False
        #: Rule R27: R5's split is offered first with the gap of Lemma E3's moment
        #: conditions: mu equal at its ends, and D (mu - mu(a)) integrating to zero.
        self.split_normal_form = False
        #: Rule R28: a split R5 or R27 offers is taken only if the limit's multiplier no
        #: longer rises on the solution and no piece or gap has zero width (V4).
        self.split_must_resolve = False
        #: Rule R29 (Lemma U): when an arc's multiplier starts to rise where the limit's
        #: own control joins the held set, the piece is offered first leaving just
        #: before that moment.
        self.exit_before_saturation = False
        #: Rule R30 (V1's order monitor on the route): a solution whose junctions have
        #: crossed is not accepted. A crossing larger than ``locate_tol`` (of T) in one
        #: step means a shorter step; a smaller one is the reorder (E5), and a reorder
        #: that crosses back again has not continued the event.
        self.watch_order = False
        #: Rule R38: junctions of different limits that cross by less than
        #: ``coincide_tol`` of T are coincident, not crossed: the order monitor (R30), the
        #: retiring of arcs and acceptance read them as in order. Symmetry makes events
        #: coincide exactly (rest-to-rest rate peaks at T/2), and Newton then leaves
        #: them crossed by rounding, which R30 chased step after step.
        self.coincident_junctions = False
        self.coincide_tol = 1e-8
        #: Rule R39 (a hand-over, Lemma K's forced exit): when R30's reorder of two
        #: junctions of different limits crosses back, each order's solution pushes
        #: them past each other, so they are one junction. They are tied: the earlier
        #: event's own row gives way to its time equal to the next one's, and the tie
        #: stays wherever the two events are adjacent. Acceptance judges the result.
        self.handover_ties = False
        self._ties = set()
        #: Rule R31 (V1's jump monitor on the route): an order-1 arc whose entry
        #: multiplier pi is negative goes. By Lemma H, 0 <= mu <= pi on the arc, so at
        #: pi = 0 the multiplier vanishes on it and the arc dies without moving the motion.
        self.retire_negative = False
        #: Rule R33: R5 splits the arc whose multiplier rises most relative to its own
        #: largest value on the arc. Scaling a limit's function by k scales its
        #: multiplier by 1/k, so only the relative rise compares limits; a rise one arc
        #: induces in another (Lemma P: eta jumps where the held set changes) is small
        #: relative to that arc's multiplier.
        self.relative_rise = False
        #: Rule R34: R5's split is also offered seeded by continuity (the first piece
        #: keeps the arc's entry jump) with the gap wider, by half and twice the rise's
        #: stretch on each side: Lemma E3's gap contains the held-set window, and the
        #: arc held there widens once the split relieves it (a seed, not a structure).
        self.wide_gaps = False
        #: Rule R35 (Bonnans & Hermant's touch that becomes an arc): at a touch point of
        #: an order-2 limit the limit has a local maximum, so ``g'' <= 0`` there (the
        #: control is continuous). A touch with ``g'' > 0`` is a local minimum, with the
        #: limit violated on both sides: it has become an arc, which takes its place,
        #: short and centred on it, the touch's jump as its entry jump.
        self.touch_curvature = False
        #: Rule R36 (Lemma H2): on an order-2 arc the multiplier of ``g''`` is convex and
        #: falls to zero at the exit, so it must not rise either. Acceptance checks it at
        #: order 2 too, and R5 splits an order-2 arc where it rises, the second piece
        #: entering with the jump that continues the arc's costate there.
        self.order2_rise = False
        #: Rule R37 (Deuflhard's adaptive pathfollowing): Newton's first contraction
        #: theta from a step's predictor sets the step. A theta above
        #: ``contraction_reject`` ends that solve at once (the step is too long for
        #: Newton, and no fallback solver is spent on it), and the next step, after a
        #: success or such a rejection, is ``h * (contraction_target / theta)^(1/p)``:
        #: theta grows like ``h^p`` (p = 2 from the secant predictor, 1 from the last
        #: solution). Steers only; acceptance is unchanged. Below ``contraction_floor``
        #: (in the leg's fraction) a step is solved as without the rule.
        self.contraction_steps = False
        self.contraction_target = 0.25
        self.contraction_reject = 0.5
        self.contraction_floor = 1e-3
        #: Rule R44: a fold of Lemma X (both orders of a crossed pair crossed, at a regular
        #: point with alpha * beta < 0) is passed by arclength: the route turns back on the
        #: other order's branch until the structure changes, then goes on to its target.
        self.fold_turns = False
        self.fold_condition = 1e8          # the largest equilibrated condition of J taken as regular
        self.fold_turn_limit = 4           # turn-backs per route
        #: Rule R45: "corner" tightens the limits at R24's larger weight and then lowers the
        #: weight; "diagonal" lowers it together with the offsets.
        self.weight_path = "corner"
        #: Rule R47: when R24's walk of the weight stalls within ``walk_exact_gap`` decades
        #: of the policy's weight, the exact problem is solved from where it stalled.
        self.walk_exact = False
        self.walk_exact_gap = 0.3
        #: Rule R48: a quadratic predictor through the last three solutions of one
        #: structure, tried before the secant.
        self.quadratic_predictor = False
        #: Rule R46 (Lemma S): after a failed step, the steps grow to at most 0.9 of it
        #: for three successes before probing larger again.
        self.step_memory = False
        #: Rule R41: an exact solution rejected because junctions of different limits
        #: crossed is solved again with the events in the order of their times (E5).
        self.exact_reorders = False
        #: Rule R17: when a step fails while an arc shrinks, extrapolate w^2 (linear
        #: in s by Lemma E) to the arc's death and step past it without the arc.
        self.locate_deaths = False
        #: Rule R54: R25's location is not repeated at a density that jumps (Lemma P).
        self.density_jumps = False
        #: Rule R55 (Lemma AC): path solves judge Newton's steps by the natural
        #: monotonicity test, not by the residual's norm.
        self.natural_monotonicity = False
        #: The free-violation checks read the densities' trajectory instead of sampling
        #: the solution again: one integration of each accepted solution for both
        #: (Lemma V1's monitors are functionals of one sampled extremal).
        self.one_sampling = True
        #: Rule R50: an order-2 arc whose width crosses zero inside a step dies there
        #: into a touch (Lemmas E'' and Z): the death is located by the secant and the
        #: touch solved just past it.
        self.order2_deaths = False
        #: Rule R18: an arc whose multiplier rises from its entry (to its exit) is
        #: trimmed to where the rise ends (starts), before any split is tried.
        self.trim_rising = False
        #: Rule R20: an order-1 arc that collapses to zero width with a positive jump is
        #: carried on as an order-1 touch point, and an order-1 birth is offered as a
        #: touch too (after the arc). By Jacobson, Lele & Speyer an order-1 touch is not
        #: an extremal with a strictly convex Hamiltonian; it stands in for an arc too
        #: short to solve, and grows back into one when a violation appears beside it.
        self.collapsed_touches = False
        #: The exact solve starts with Newton steps (fresh Jacobians) from the path
        #: solution, and falls back to MINPACK only if they do not reach the tolerance.
        self.exact_newton = True
        #: The separate seed traces all moving components at once (one integration per
        #: step) rather than one at a time.
        self.joint_seed = True
        #: Rule R21: an exact solution rejected because an order-1 arc's multiplier
        #: rises (Lemma H) is split there, settled on the route at the end of the
        #: offsets (new violations brought in), and solved exactly again.
        self.exact_splits = False
        #: Rule R32: an exact solution rejected because a limit is violated off its
        #: arcs, by less than the route sees (its threshold is 1e-6 of the limit's
        #: scale, the exact check's 1e-7), gets the birth there (E1, Lemma E), settled
        #: on the route at the end of the offsets and solved exactly again, as R21.
        self.exact_births = False
        self._last_rejected = None
        #: Diagnostics of the last solve; snapshots of failed steps go to ``snapshot_dir``
        #: (None: none), named by ``snapshot_tag``; ``check_lemmas`` runs the lemma
        #: oracles on every path solution (slow: for development).
        self.diagnostics = Diagnostics()
        self.snapshot_dir = None
        self.snapshot_tag = None
        self.check_lemmas = False
        #: Path solves the active-set route may make (a budget that does not depend on
        #: the machine; ``active_set_seconds`` is the wall-clock one).
        self.max_path_solves = 1500        # rule R2: neutral on the census (see the note)
        self.edge_exits = False         # rule R8: tried, no gain (see the note)
        self._certificates = None
        self.penalty_seconds = 300.0       # time budget of the penalty route
        self.path_options = dict(rtol=rtol, atol=atol)    # along the active-set route
        self.path_evaluations = 240        # MINPACK evaluations per solve on that route
        self.path_fallback = 1e-2          # a chord step that ends below this earns MINPACK
        # Picking a new structure: with parallel > 1, each candidate and `variants` jittered
        # copies of it are solved at once in forked processes, each stopped after
        # candidate_seconds, and the first candidate (in order) that converged is taken.
        self.parallel = 0
        self.variants = 3
        self.candidate_seconds = 2.0
        self.bang_bang = bang_bang_controls(problem)
        # A numeric secondary weight is a runtime parameter of the kernels, after the
        # problem's own: rule R24 walks it down to its value along the route.
        self.weight_symbol, self.weight_value = None, None
        kernel_problem = problem
        if isinstance(self.policy, SecondaryCost) and sp.sympify(self.policy.weight).is_number:
            w = sp.Symbol(WEIGHT, positive=True)
            controls = (_penalized(self.analysis) if self.policy.controls is None
                        else tuple(self.policy.controls))
            kernel_problem = _copy(self.stated, name=problem.name,
                                   running_cost=self.stated.running_cost
                                   + w * _effort(self.stated, controls),
                                   params=[*self.stated.params, w],
                                   path_constraints=self.stated.path_constraints)
            self.weight_symbol, self.weight_value = w, float(self.policy.weight)
        #: Rule R24: the active-set route brings the limits in with the secondary weight
        #: this many times its value, then walks it down to its value (Lemma W); None
        #: or 1: at its value throughout.
        self.weight_start = None
        self.regularized, self.eps_symbol, self.rho_symbols = regularize(kernel_problem,
                                                                        self.bang_bang)
        # Limits ridden at run time have no penalty route, and a penalty's tape in every
        # kernel takes most of the compile.
        self.outline_above = outline_above
        self.kr = build_kernels(derive(self.regularized, penalties_when_riding=False,
                                       outline_above=outline_above), cache_dir=cache_dir)
        if self.kr.system.riding is not None:
            self.active_set_seconds = 300.0     # many arcs, and no penalty route to fall back on
        self.nc = len(problem.path_constraints)
        self._kx = None
        self._ends = None
        self._seed_key = self._seed = None
        t, x, p = problem.time, problem.states, problem.params
        self._terminal = sp.lambdify((t, x, p), problem.terminal_cost, "numpy")

    def _say(self, *args):
        if self.verbose:
            print(*args)

    def _detail(self, *args):
        if self.verbose and self.verbose > 1:
            print(*args)

    # -- kernels and parameters ---------------------------------------------

    @property
    def exact_kernels(self):
        """Kernels of the exact problem: singular arcs derived when there are
        bang-bang controls; otherwise the regularized kernels (at ``rho = 0``
        they are the exact ones)."""
        if self._kx is None:
            if self.bang_bang:
                self._kx = build_kernels(derive(exact(self.problem), singular=True,
                                                outline_above=self.outline_above),
                                         cache_dir=self.cache_dir)
            else:
                self._kx = self.kr
        return self._kx

    def exact_params(self):
        if self.bang_bang:
            return self.params.copy()
        return self.regularized_params(None, np.zeros(self.nc))

    def regularized_params(self, eps, rho, off=None):
        """The regularized problem's parameter vector at smoothing ``eps``, penalty
        weights ``rho`` and limit offsets ``off`` (default zero: the exact limits)."""
        p = list(self.params)
        if self.weight_symbol is not None:
            # On a problem without bang-bang controls ``eps`` is the secondary weight
            # (rule R24); None is its value.
            p.append(self.weight_value if self.bang_bang or eps is None else eps)
        if self.bang_bang:
            p.append(eps)
        off = np.zeros(self.nc) if off is None else off
        return np.array(p + list(rho) + list(off), dtype=float)

    def _weight_route(self):
        """Rule R24 applies: a numeric secondary weight and no bang-bang smoothing (with
        smoothing the route walks eps, and a weight route would repeat the first)."""
        return self.weight_symbol is not None and not self.bang_bang

    def _eps_at(self, le):
        """The ``eps`` argument of :meth:`regularized_params` at route coordinate ``le``:
        the smoothing weight, the secondary weight while rule R24 has it away from its
        value, else None."""
        if self.bang_bang:
            return 10.0 ** le
        if self.weight_symbol is not None and abs(le - np.log10(self.weight_value)) > 1e-12:
            return 10.0 ** le
        return None

    def _weight_moved(self, sol, le_from, le_to, tol=1e-10):
        """Rule R24: the unconstrained solution ``sol`` at secondary weight ``10^le_from``
        carried to ``10^le_to`` by Newton steps along ``log w`` (halved on failure):
        ``(sol, le_to)``, or None. Off the limits the path is smooth in ``w``."""
        le, v, ds, r = le_from, np.asarray(sol.v, dtype=float), le_to - le_from, sol
        while abs(le_to - le) > 1e-12:
            if abs(ds) < 1e-3:
                return None
            nxt = le_to if abs(ds) >= abs(le_to - le) else le + ds
            shoot = self._smoothed([], self._eps_at(nxt), np.zeros(self.nc))
            try:
                r = shoot.solve_newton(v, tol=tol, max_iterations=12)
            except _FAILURES:
                r = None
            if r is None or not r.success:
                ds *= 0.5
                continue
            le, v = nxt, r.v
        return r, le_to

    def objective(self, shooting, v):
        """``phi(t_f, x(t_f)) + int L`` of the exact problem at ``v``."""
        if isinstance(shooting, MultiArcShooting):
            _, _, segs = shooting._evaluate(np.asarray(v, dtype=float), jac=False, record=True)
            running = float(sum(r["cost"] for *_, r in segs))
            tf, xf = segs[-1][3], segs[-1][-1]["z"][:shooting.kernels.N_STATES]
        else:
            x0, lam0, _, tf = shooting.unpack(v)
            r = shooting.kernels.propagate(shooting.t0, np.concatenate([x0, lam0]), tf,
                                           shooting.params, cost=True, **self.options)
            running, xf = float(r["cost"]), r["z"][:shooting.kernels.N_STATES]
        return running + float(self._terminal(tf, list(xf), list(self.params)))

    def _limits_broken_at_the_ends(self):
        """Path constraints the fixed initial or final state already violates.

        No motion can satisfy them, whatever the search does; finding that
        out by searching takes minutes on a large problem. Only ends with
        every state fixed are checked, and only time-invariant limits.
        """
        problem, kr = self.problem, self.kr
        if not self.nc:
            return []
        if self._ends is None:
            t, x, p = problem.time, problem.states, problem.params
            final = {}
            for psi in problem.terminal_constraints:
                involved = [s for s in x if s in psi.free_symbols]
                if len(involved) == 1 and sp.diff(psi, involved[0]) == 1:
                    final[involved[0]] = involved[0] - psi
            at_end = None
            if len(final) == len(x) and not any(
                    t in e.free_symbols for e in final.values()):
                at_end = sp.lambdify([p], [final[s] for s in x], "numpy")
            start = None
            if len(problem.initial_state) == len(x):
                start = sp.lambdify([p], [problem.initial_state[s] for s in x], "numpy")
            ok = not any(t in c.expr.free_symbols for c in problem.path_constraints)
            self._ends = (start, at_end) if ok else (None, None)
        start, at_end = self._ends
        params = self.regularized_params(1.0 if self.bang_bang else None, np.zeros(self.nc))
        broken = []
        for label, end in (("the initial state", start), ("the final state", at_end)):
            if end is None:
                continue
            x = np.asarray(end(list(self.params)), dtype=float)
            g = np.asarray(kr.constraint(0.0, np.concatenate([x, np.zeros(len(x))]), params))
            for c in np.flatnonzero(g > 1e-9 * np.maximum(1.0, np.abs(g))):
                broken.append(f"{label} violates {kr.CONSTRAINT_NAMES[c]} by {g[c]:.6g}")
        return broken

    # -- step 2: sampling -----------------------------------------------------

    def _free_arc_params(self):
        """Indices of the parameters that matter off the path constraints."""
        problem = self.problem
        exprs = [*problem.dynamics, problem.running_cost, problem.terminal_cost,
                 *problem.terminal_constraints, *problem.initial_state.values(),
                 problem.initial_time]
        for cs in problem.control_sets:
            exprs += [cs.lower, cs.upper] if isinstance(cs, Box) else []
        used = set().union(*(sp.sympify(e).free_symbols for e in exprs))
        return [i for i, s_ in enumerate(problem.params) if s_ in used]

    def _separate_blocks(self, shoot):
        """Per controlled component, its state indices and its terminal rows, or None.

        Only for a free final time with no terminal cost, each component fixing its
        own final state: then, with the penalties off, a component with the other
        costates at zero is its own minimum-time problem inside the whole one.
        """
        k, problem = shoot.kernels, self.problem
        parts = self.analysis.controlled
        if (len(parts) < 2 or not k.FREE_FINAL_TIME or problem.terminal_cost != 0
                or shoot.n_free or k.N_SWITCHING):
            return None
        index = {s: i for i, s in enumerate(problem.states)}
        terminal = [{index[s] for s in psi.free_symbols if s in index}
                    for psi in problem.terminal_constraints]
        blocks, covered = [], set()
        for part in parts:
            states = sorted(index[s] for s in part.states)
            rows = [i for i, involved in enumerate(terminal) if involved & set(states)]
            if len(rows) != len(states) or any(not terminal[i] <= set(states) for i in rows):
                return None
            covered |= set(states)
            blocks.append((np.array(states), np.array(rows)))
        return blocks if len(covered) == len(index) else None

    def _block_system(self, shoot, states, rows):
        """One component's equations ``(R, J)`` in its costates, and ``t_f`` when it is free."""
        k = shoot.kernels
        n = k.N_STATES
        nu = np.zeros(k.N_MULTIPLIERS)

        def system(lam, tf, free, with_jacobian=True):
            lam0 = np.zeros(n)
            lam0[states] = lam
            z0 = np.concatenate([shoot.x0, lam0])
            r = shoot._propagate(z0, tf, stm=with_jacobian)
            zf, phi = r["z"], r["stm"]
            psi = k.residual(z0, zf, nu, tf, shoot.params)[n:][rows]
            if not with_jacobian:
                H = np.atleast_1d(k.hamiltonian(tf, zf, shoot.params))
                return np.concatenate([psi, H]) if free else psi
            jac = k.residual_jacobian(z0, zf, nu, tf, shoot.params)
            Rzf = jac["zf"][n:][rows]
            columns = n + states
            J = Rzf @ phi[:, columns]
            if not free:
                return psi, J
            Ht, Hz = k.hamiltonian_gradient(tf, zf, shoot.params)
            H = np.atleast_1d(k.hamiltonian(tf, zf, shoot.params))
            top = np.hstack([J, (Rzf @ r["dz_dt1"])[:, None]])
            bottom = np.concatenate([Hz @ phi[:, columns], [float(Ht) + Hz @ r["dz_dt1"]]])
            return np.concatenate([psi, H]), np.vstack([top, bottom])
        return system

    def _separate_seed(self, shoot, final_time, tol):
        """A first guess from the components solved one at a time: ``v``, or None.

        Random draws must land every component at once, and a nearly bang-bang
        control gives Newton little to work with. But given plenty of time a
        component's cost is nearly quadratic and its bounds are idle, so Newton needs no
        guess there. From that end each component follows its costates as its time
        shrinks, until ``H`` reaches zero: its own minimum time. The slowest sets
        ``t_f``, and the others take their costates from their paths at that time.
        """
        blocks = self._separate_blocks(shoot)
        if blocks is None:
            return None
        # A guess, not the answer: the whole system's solve polishes it afterwards.
        whole = shoot
        shoot = SingleShooting(shoot.kernels, shoot.params, rtol=1e-8, atol=1e-10, n_interior=0)
        # It does not depend on the path constraints (_free_arc_params), so it ignores their guards.
        shoot.options["constraint_events"] = False
        shoot.max_nfev = 20000             # a bad guess can chatter on a saturation corner
        shoot.jacobian_rtol = PATH_JACOBIAN_RTOL
        tol = max(tol, 1e-6)
        k = shoot.kernels
        n = k.N_STATES
        rng = np.random.default_rng(0)
        start = max(float(_draw(final_time, rng)) for _ in range(8)) if callable(final_time) \
            else float(np.max(final_time[1]))
        if self.joint_seed:
            # The components are independent, so one trace of all the moving ones at once
            # ends where the whole system's H is zero: the same common time the
            # one-at-a-time traces interpolate to, for one integration per step instead
            # of one per component.
            moving = []
            for states, rows in blocks:
                system = self._block_system(shoot, states, rows)
                if np.max(np.abs(system(np.zeros(len(states)), start, False)[0])) >= tol:
                    moving.append((states, rows))
            if not moving:
                return None
            states = np.concatenate([np.atleast_1d(st) for st, _ in moving])
            rows = np.concatenate([np.atleast_1d(rw) for _, rw in moving])
            system = self._block_system(shoot, states, rows)
            path = self._shrink(shoot, system, states, start, tol)
            if path is not None and path[-1][2] == 0.0:
                T, lam = path[-1][0], path[-1][1]
                self._say(f"separate seed: t_f = {T:.6f}")
                lam0 = np.zeros(n)
                lam0[states] = lam
                try:
                    r = whole._propagate(np.concatenate([whole.x0, lam0]), T, stm=False)
                except _FAILURES:
                    r = None
                if r is not None:
                    return whole.pack(lam0, r["z"][n:n + k.N_MULTIPLIERS], T)
            # Otherwise the one-at-a-time traces below.
        paths = []
        for states, rows in blocks:
            system = self._block_system(shoot, states, rows)
            if np.max(np.abs(system(np.zeros(len(states)), start, False)[0])) < tol:
                paths.append(None)         # already there: it never moves
                continue
            # A component only has to reach the slowest time found so far.
            floor = max((p[-1][0] for p in paths if p is not None), default=0.0)
            path = self._shrink(shoot, system, states, start, tol, floor)
            if path is None:
                self._say("separate seed: a component's time did not shrink to its minimum")
                return None
            paths.append(path)
        moving = [p for p in paths if p is not None]
        if not moving:
            return None
        T = self._common_time(moving)
        self._say(f"separate seed: t_f = {T:.6f}")
        lam0 = np.zeros(n)
        for (states, rows), path in zip(blocks, paths):
            if path is None:
                continue
            system = self._block_system(shoot, states, rows)
            at = [point[1] for point in path if point[0] >= T][-1]
            lam = _newton(lambda w, jac, tf=T: system(w, tf, False, jac), at, tol,
                          accept=SEED_ACCEPT * tol)
            if lam is None:
                self._say("separate seed: a component did not reach the common time")
                return None
            lam0[states] = lam
        try:
            r = whole._propagate(np.concatenate([whole.x0, lam0]), T, stm=False)
        except _FAILURES:
            return None
        return whole.pack(lam0, r["z"][n:n + k.N_MULTIPLIERS], T)

    @staticmethod
    def _common_time(paths):
        """The ``t_f`` where the components' Hamiltonians add up to zero.

        Each path's ``H`` is one (the cost of time) plus that component's own part, so the
        sum is one plus the parts. It is positive with time to spare, and each part
        falls to minus one at the component's own minimum time. Several components that
        need nearly the same time share the one between them, so ``t_f`` sits above
        them all, and Newton does not have to find that from a long way off.
        """
        grid = np.unique(np.concatenate([[p[0] for p in path] for path in paths]))
        low = max(path[-1][0] for path in paths)
        grid = grid[grid >= low]

        def total(time_):
            value = 1.0
            for path in paths:
                times, values = np.array([p[0] for p in path])[::-1], np.array([p[2] for p in path])[::-1]
                value += float(np.interp(time_, times, values)) - 1.0
            return value

        values = [total(g) for g in grid]
        for k in range(len(grid) - 1):
            if values[k] <= 0.0 < values[k + 1]:
                share = -values[k] / (values[k + 1] - values[k])
                return float(grid[k] + share * (grid[k + 1] - grid[k]))
        return float(low)

    def _shrink(self, shoot, system, states, start, tol, floor=0.0):
        """``[(T, lam, H)]`` from a long time down to the component's minimum time, or to
        ``floor`` if that comes first; or None."""
        lam = _newton(lambda w, jac: system(w, start, False, jac), np.zeros(len(states)), tol)
        if lam is None:
            return None
        path, T, previous = [], start, None
        step = 0.15
        k = shoot.kernels

        def hamiltonian(lam_, tf):
            lam0 = np.zeros(k.N_STATES)
            lam0[states] = lam_
            zf = shoot._propagate(np.concatenate([shoot.x0, lam0]), tf, stm=False)["z"]
            return float(np.atleast_1d(k.hamiltonian(tf, zf, shoot.params))[0])

        H0 = hamiltonian(lam, T)
        if H0 <= 0.0:                      # already below its minimum time: nothing to follow
            return None
        path.append((T, lam, H0))
        while step > 1e-5:
            to = max(T * (1.0 - step), floor)
            guess = lam
            if previous is not None:
                # A secant in logarithms: costates grow like powers of 1 / T.
                ratio = np.log(to / T) / np.log(T / previous[0])
                same = lam * previous[1] > 0.0
                guess = np.where(same, lam * np.abs(lam / np.where(same, previous[1], 1.0)) ** ratio,
                                 lam + (lam - previous[1]) * (to - T) / (T - previous[0]))
            found = _newton(lambda w, jac, tf=to: system(w, tf, False, jac), guess, tol)
            if found is None and previous is not None:
                found = _newton(lambda w, jac, tf=to: system(w, tf, False, jac), lam, tol)
            if found is None:
                step *= 0.5
                continue
            try:
                H = hamiltonian(found, to)
            except _FAILURES:
                step *= 0.5
                continue
            if H < 0.0:
                # Between T and to: solve for the time as well.
                w = np.append(found, to)
                root = _newton(lambda v, jac: system(v[:-1], v[-1], True, jac), w, tol,
                               accept=SEED_ACCEPT * tol)
                if root is not None and root[-1] > 0.0:
                    path.append((float(root[-1]), root[:-1], 0.0))
                    return path
                step *= 0.5
                continue
            previous = (T, lam)
            T, lam = to, found
            path.append((T, lam, H))
            if T <= floor:
                return path
            step = min(step * 1.3, 0.15)
        return None

    def _sampler(self, shoot, costates, final_time):
        n = self.kr.N_STATES

        def sample(rng):
            lam0 = np.asarray(_draw(costates, rng), dtype=float).reshape(-1)
            if lam0.shape != (n,):
                raise ValueError(f"the costate sample must have {n} entries")
            tf = None
            if self.kr.FREE_FINAL_TIME:
                if final_time is None:
                    raise ValueError("the final time is free: give a final_time distribution")
                tf = float(_draw(final_time, rng))
            return shoot.guess_multipliers(lam0, tf=tf)

        return sample

    # -- step 4: reading the structure ----------------------------------------

    def _read(self, shoot, v, n_points=2001):
        """Labels of a regularized solution: time, states, violated intervals
        per constraint, and per bang-bang control its bound labels."""
        kr, p = self.kr, shoot.params
        x0, lam0, _, tf = shoot.unpack(v)
        z0 = np.concatenate([x0, lam0])
        tr = kr.trajectory(shoot.t0, z0, tf, p, n_points=n_points, **self.options)
        t, Z, U = tr["t"], tr["z"], tr["u"]
        T = tf - shoot.t0
        out = {"t": t, "z": Z, "u": U, "T": T, "t0": shoot.t0, "tf": tf, "z0": z0,
               "violated": [], "labels": {}}
        # Violated intervals, from the located crossings and g between them.
        for c in range(self.nc):
            cross = np.sort(tr["constraint_times"][tr["constraint_index"] == c])
            ts = np.concatenate([[shoot.t0], cross, [tf]])
            mids = 0.5 * (ts[:-1] + ts[1:])
            gm = np.interp(mids, t, tr["g"][:, c])
            spans = [(a, b) for a, b, g in zip(ts[:-1], ts[1:], gm) if g > 0 and b > a]
            merged = []
            for a, b in spans:
                if merged and a - merged[-1][1] < 0.02 * T:
                    merged[-1] = (merged[-1][0], b)
                else:
                    merged.append((a, b))
            out["violated"].append(merged)
        controls = list(kr.system.controls)
        for u in self.bang_bang:
            j = controls.index(u)
            bounds = np.array([kr.control_bounds(ti, zi, p) for ti, zi in zip(t, Z)])
            lo, hi = bounds[:, 0, j], bounds[:, 1, j]
            tol = 1e-9 * np.maximum(hi - lo, 1e-300)
            lab = np.zeros(len(t), dtype=int)
            lab[U[:, j] >= hi - tol] = 1
            lab[U[:, j] <= lo + tol] = -1
            out["labels"][u] = lab
        return out

    @staticmethod
    def _runs(t, mask):
        """``[(start, end)]`` of the runs where ``mask`` holds, in time."""
        edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
        out = []
        for a, b in zip(edges[0::2], edges[1::2]):
            ta = t[a] if a == 0 else 0.5 * (t[a - 1] + t[a])
            tb = t[b - 1] if b == len(t) else 0.5 * (t[b - 1] + t[b])
            out.append((ta, tb))
        return out

    def _candidates(self, read, max_candidates=24):
        """Candidate structures from a reading: ``(items, held)`` per candidate.

        ``items`` are ``(kind, index, a, b)`` in time order with kind
        ``boundary``, ``touch``, ``singular``; ``held`` gives, per control
        with a singular arc, the bound labels of its free arcs. The second
        return value is whether any candidate could be built (``False``
        while a limit without exact arcs is violated).
        """
        kx = self.exact_kernels
        t, T = read["t"], read["T"]
        options = []                       # per violated interval: kinds to try
        for c, spans in enumerate(read["violated"]):
            for a, b in spans:
                if not kx.EXACT_ARCS[c]:
                    return [], False
                kinds = ["boundary"]
                if kx.CONSTRAINT_ORDERS[c] == 2:
                    kinds = ["touch", "boundary"] if b - a < 0.05 * T else ["boundary", "touch"]
                elif kx.CONSTRAINT_ORDERS[c] >= 3:
                    kinds = ["touch"]          # no arcs at order 3 or more
                options.append([(k, c, a, b) for k in kinds])
        limited = [(a, b) for spans in read["violated"] for a, b in spans]

        def on_limit(a, b):
            return sum(max(0.0, min(b, y) - max(a, x)) for x, y in limited)

        names = [sw.control for sw in kx.system.switching_functions]
        singular_choices = []              # per control: list of run subsets
        ramps = {}                         # per control: pieces that count against readiness
        near = 0.02 * T
        for u, lab in read["labels"].items():
            # Stretches strictly between the bounds, cut where a limit is
            # violated (the control there is the one riding the limit).
            pieces = []
            for a, b in self._runs(t, lab == 0):
                cuts = [(a, b)]
                for x, y in limited:
                    cuts = [(p, q) for (p0, q0) in cuts
                            for p, q in ((p0, min(q0, x)), (max(p0, y), q0)) if q > p]
                pieces += cuts
            # A piece that runs onto a limit is the smoothed approach to it;
            # one that leaves a limit may be a joined singular arc.
            onto = [any(abs(b - x) < near for x, _ in limited) for a, b in pieces]
            ramps[u] = [r for r, o in zip(pieces, onto)
                        if not o and not any(abs(r[0] - y) < near for _, y in limited)]
            sw = names.index(u)
            if not kx.SINGULAR_ARCS[sw]:
                singular_choices.append([(u, ())])
                continue
            long = sorted([r for r, o in zip(pieces, onto) if not o and r[1] - r[0] >= 0.02 * T],
                          key=lambda r: r[0] - r[1])[:3]
            subsets = [()]
            for k in range(1, len(long) + 1):
                subsets += [tuple(sorted(c)) for c in itertools.combinations(long, k)]
            singular_choices.append([(u, s) for s in subsets[::-1]])   # most singular first

        out = []
        for limits in itertools.product(*options) if options else [()]:
            for sing in itertools.product(*singular_choices) if singular_choices else [()]:
                # Ready: the ramps not read as singular are short.
                rest = sum(b - a for u, subset in sing for a, b in ramps[u]
                           if (a, b) not in subset)
                if rest > 0.05 * T:
                    continue
                items = sorted(list(limits) + [("singular", u, a, b) for u, subset in sing
                                               for a, b in subset], key=lambda it: it[2])
                out.append(items)
                if len(out) >= max_candidates:
                    return out, True
        return out, True

    def _structure(self, items, read):
        """``(arcs, values)`` for MultiArcShooting from candidate items, or None.

        Adds the joins, the bounds held next to singular arcs and the
        switches of controls with singular arcs on free arcs.
        """
        t, T = read["t"], read["T"]
        special = [(a, b) for kind, _, a, b in items]
        for (a1, b1), (a2, b2) in zip(special[:-1], special[1:]):
            if a2 < b1 - 0.02 * T:
                return None                # two special arcs at once
        frozen = {it[1] for it in items if it[0] == "singular"}
        # Per frozen control: (time, label) of its free-arc samples.
        free = np.ones(len(t), dtype=bool)
        for a, b in special:
            free &= ~((t >= a) & (t <= b))
        events = []                        # (time, entry)
        for i, (kind, c, a, b) in enumerate(items):
            if kind != "singular":
                events.append((a, (kind, c, a, b)))
                continue
            prev = items[i - 1] if i else None
            joined = prev is not None and prev[0] == "boundary" and a - prev[3] < 0.02 * T
            nxt = items[i + 1] if i + 1 < len(items) else None
            if nxt is not None and nxt[2] - b < 0.02 * T:
                return None                # singular -> boundary joins are not supported
            events.append((a, ("singular", c, a, b, joined)))
        # Labels on free arcs for frozen controls: segments and switches.
        held = {}
        for u in frozen:
            lab = read["labels"][u]
            segs = []                      # (start, end, label) of free labeled stretches
            for value in (1, -1):
                for a, b in self._runs(t, free & (lab == value)):
                    segs.append((a, b, value))
            segs.sort()
            held[u] = segs
        timeline = sorted(events, key=lambda e: e[0])
        # Switch events for frozen controls: label flips between consecutive
        # free segments with no singular arc of that control in between.
        for u in frozen:
            segs = held[u]
            sing = [(e[1][2], e[1][3]) for e in timeline if e[1][0] == "singular" and e[1][1] == u]
            for (a1, b1, l1), (a2, b2, l2) in zip(segs[:-1], segs[1:]):
                if l1 != l2 and not any(b1 - 1e-12 <= x <= a2 + 1e-12 or b1 <= y <= a2
                                        for x, y in sing):
                    ts = 0.5 * (b1 + a2)
                    inside = [(x, y) for kind, c, x, y in items if kind != "singular"
                              and x <= ts <= y]
                    if inside:
                        ts = inside[0][1] + 1e-3 * T
                    timeline.append((ts, ("switch", u, ts)))
        timeline.sort(key=lambda e: e[0])

        def label_near(u, time, before):
            segs = held[u]
            if before:
                cand = [s for s in segs if s[1] <= time + 1e-12]
                return cand[-1][2] if cand else None
            cand = [s for s in segs if s[0] >= time - 1e-12]
            return cand[0][2] if cand else None

        sides = []
        for _, ev in timeline:
            if ev[0] == "singular":
                _, u, a, b, joined = ev
                start = a
                if joined:
                    start = [x for kind, c, x, y in items if kind == "boundary" and y <= a + 0.02 * T][-1]
                before = label_near(u, start, True)
                after = label_near(u, b, False)
                sides.append((before, after))
        return timeline, sides

    def _assemble(self, timeline, sides_choice, read, base, jitter_rng=None):
        """Arc entries and their values; ``jitter_rng`` perturbs the times."""
        T, t0, tf = read["T"], read["t0"], read["tf"]
        times = []
        for _, ev in timeline:
            if ev[0] in ("boundary", "touch"):
                times += [ev[2], ev[3]] if ev[0] == "boundary" else [0.5 * (ev[2] + ev[3])]
            elif ev[0] == "singular":
                times += [ev[3]] if ev[4] else [ev[2], ev[3]]
            else:
                times.append(ev[2])
        times = np.array(times, dtype=float)
        lo, hi = t0 + 0.01 * T, tf - 0.01 * T
        times = np.clip(times, lo, hi)
        if jitter_rng is not None and len(times):
            grid = np.concatenate([[t0], times, [tf]])
            gaps = np.minimum(np.diff(grid)[:-1], np.diff(grid)[1:])
            times = times + jitter_rng.uniform(-0.5, 0.5, len(times)) * np.maximum(gaps, 0.02 * T)
            times = np.clip(times, lo, hi)
        if np.any(np.diff(times) <= 0):
            return None
        arcs, values, k = [], [], 0
        si = 0
        for _, ev in timeline:
            kind = ev[0]
            if kind == "boundary":
                a, b = times[k], times[k + 1]
                k += 2
                arcs.append(("boundary", ev[1]))
                values.append([a, b, None])
            elif kind == "touch":
                arcs.append(("touch", ev[1]))
                values.append([times[k], None])
                k += 1
            elif kind == "singular":
                before, after = sides_choice[si]
                si += 1
                name = ev[1].name
                if ev[4]:
                    arcs.append(("singular", name, before, after, "joined"))
                    values.append(times[k])
                    k += 1
                else:
                    arcs.append(("singular", name, before, after))
                    values.append((times[k], times[k + 1]))
                    k += 2
            else:
                arcs.append(("switch", ev[1].name))
                values.append(times[k])
                k += 1
        return arcs, values

    def _fit_jumps(self, shoot, values, read):
        """Fill in boundary ``pi`` and touch ``nu`` from the regularized costates."""
        kx, p, n = shoot.kernels, shoot.params, shoot.kernels.N_STATES
        t, Z = read["t"], read["z"]

        def z_at(ti):
            return np.array([np.interp(ti, t, Z[:, i]) for i in range(Z.shape[1])])

        out = []
        for (kind, c, _), val in zip(shoot.arcs, values):
            if kind == "boundary":
                a, b, _ = val
                q = kx.CONSTRAINT_ORDERS[c]
                pi = np.zeros(q)
                try:
                    za = z_at(a)
                    back = kx.propagate_boundary(c, b, z_at(b), a, p, **self.options)["z"]
                    _, Nx, _, _ = kx.tangency_gradient(c, a, za, p)
                    pi = np.linalg.lstsq(Nx.T, za[n:] - back[n:], rcond=None)[0]
                except _FAILURES:
                    pass
                out.append((a, b, pi))
            elif kind == "touch":
                tc = val[0]
                nu = 0.0
                try:
                    _, Nx, _, _ = kx.tangency_gradient(c, tc, z_at(tc), p)
                    w = 0.02 * read["T"]
                    jump = z_at(tc - w)[n:] - z_at(tc + w)[n:]
                    nu = float(np.linalg.lstsq(Nx[:1].T, jump, rcond=None)[0][0])
                except _FAILURES:
                    pass
                out.append((tc, nu))
            else:
                out.append(val)
        return out

    def _accept(self, shoot, sol, tol):
        """Whether an exact solution passes the checks it can make. A converged one
        that does not is counted in ``diagnostics.rejected`` by the check it failed."""
        reason = self._verdict(shoot, sol)
        if reason is not None and sol.success and np.isfinite(sol.residual_norm):
            self.diagnostics.reject(reason)
            self._last_rejected = (shoot, sol, reason)
        return reason is None

    def _verdict(self, shoot, sol):
        """None if an exact solution passes every check, else the check it fails."""
        if not sol.success or not np.isfinite(sol.residual_norm):
            return "not converged"
        if sol.max_violation > 1e-7:
            return "limit violated off its arcs"
        if not isinstance(shoot, MultiArcShooting) or not shoot.arcs:
            return None
        if not shoot.in_order(sol.v, self._coincide()):
            return "junctions out of order"
        try:
            d = shoot.diagnostics(sol.v)
        except _FAILURES:
            return "diagnostics failed"
        small = 1e-7
        if any(nu < -small for nu in d["touch_nu"]):
            return "touch jump negative"
        if any(lo < -small * max(1.0, abs(hi)) for lo, hi in d["boundary_mu"]):
            return "mu negative (Lemma H)"
        # Order 1: mu = int_t^{t_out} eta with eta >= 0, so mu must not rise along an arc
        # (docs/notes/structure_certificates.md, Lemma H).
        # Order 2 (R36): mu is convex and ends at 0 with slope <= 0 (Lemma H2): the same.
        rising = (1, 2) if getattr(self, "order2_rise", False) else (1,)
        order1 = [q in rising for kind, _, q in shoot.arcs if kind == "boundary"]
        if any(o and rise > 1e-6 * max(1.0, abs(hi))
               for o, rise, (_, hi) in zip(order1, d["boundary_mu_rise"], d["boundary_mu"])):
            return "mu rises (Lemma H)"
        if any(lo < -small * max(1.0, abs(hi)) for lo, hi in d["singular_legendre"]):
            return "Legendre condition"
        if any(w > 1e-6 for w in d["held_sign_violation"].values()):
            return "held control sign"
        return None

    def _attempt(self, read, base, n_jitter, rng, tol):
        """Exact solves of every candidate from one reading: ``[(J, shoot, sol)]``."""
        kx, px = self.exact_kernels, self.exact_params()
        candidates, possible = self._candidates(read)
        tried, accepted = 0, []
        seen = set()
        for items in candidates:
            built = self._structure(items, read)
            if built is None:
                continue
            timeline, sides = built
            if any(before is None and after is None for before, after in sides):
                continue                   # the control never reaches a bound: not ready
            side_options = []
            for before, after in sides:
                b_opts = ["upper", "lower"] if before is None else [
                    "upper" if before > 0 else "lower"]
                a_opts = ["lower", "upper"] if after is None else [
                    "upper" if after > 0 else "lower"]
                side_options.append([(b, a) for b in b_opts for a in a_opts])
            for sides_choice in itertools.product(*side_options) if side_options else [()]:
                for j in range(n_jitter + 1):
                    made = self._assemble(timeline, sides_choice, read, base,
                                          None if j == 0 else rng)
                    if made is None:
                        continue
                    arcs, values = made
                    try:
                        if arcs:
                            shoot = self._multi_arc(kx, px, arcs, max_nfev=self.attempt_nfev,
                                                     **self.options)
                            vals = self._fit_jumps(shoot, values, read)
                            guess = shoot.pack_arcs(base, vals)
                            method = "lm" if kx.N_SWITCHING else "hybr"
                        else:
                            key = ("free",)
                            if key in seen:
                                continue
                            seen.add(key)
                            shoot = SingleShooting(kx, px, **self.options)
                            guess = np.asarray(base, dtype=float)
                            method = "hybr"
                        sol = shoot.solve(guess, method=method, tol=_scaled(tol, guess),
                                          options=_budget(method, self.attempt_evaluations))
                    except _FAILURES:
                        continue
                    tried += 1
                    if self._accept(shoot, sol, _scaled(tol, guess)):
                        try:
                            J = self.objective(shoot, sol.v)
                        except _FAILURES:
                            continue
                        accepted.append((J, shoot, sol))
                        break              # this structure is solved; next candidate
        return accepted, tried, possible

    # -- the active-set route ---------------------------------------------------

    def _multi_arc(self, kernels, params, arcs, **options):
        """A :class:`MultiArcShooting`, with :attr:`riding_nfev` on its boundary arcs when
        they are solved at run time."""
        shoot = MultiArcShooting(kernels, params, arcs, **options)
        if getattr(self, "handover_ties", False) and self._ties:
            shoot.tied = self._tied_events(shoot)
        system = getattr(kernels, "system", None)
        if getattr(system, "riding", None) is not None and self.riding_nfev:
            shoot.boundary_max_nfev = int(self.riding_nfev)
            if self.riding_stm_nfev:
                shoot.boundary_stm_max_nfev = int(self.riding_stm_nfev)
        return shoot

    def _smoothed(self, arcs, eps, off):
        """Shooting on the regularized kernels with exact arcs ``arcs`` and penalties off."""
        p = self.regularized_params(eps, np.zeros(self.nc), off)
        shoot = (self._multi_arc(self.kr, p, arcs, max_nfev=self.attempt_nfev, open_end=True,
                                  order1_touches=True if self.collapsed_touches else None,
                                  **self.path_options)
                 if arcs else SingleShooting(self.kr, p, **self.path_options))
        shoot.jacobian_rtol = PATH_JACOBIAN_RTOL
        shoot.fast_residuals = True
        shoot.tangent_junctions = self.kr.system.riding is not None
        shoot.natural_monotonicity = getattr(self, "natural_monotonicity", False)
        return shoot

    def _free_samples(self, shoot, v, n_points=801):
        """``[(t, g, active)]`` sampled on every segment of ``shoot`` at ``v``:
        free arcs (``active`` empty) and boundary arcs (the limits ridden)."""
        from fast_numerical_integration.integrators import solve_ivp

        k = shoot.kernels
        if not isinstance(shoot, MultiArcShooting):
            x0, lam0, _, tf = shoot.unpack(v)
            tr = k.trajectory(shoot.t0, np.concatenate([x0, lam0]), tf, shoot.params,
                              n_points=n_points, constraint_events=False, **self.options)
            return [(tr["t"], tr["g"], ())]
        if getattr(self, "one_sampling", False):
            # The trajectory that R25's densities sample (memoized), split where the arc
            # changes: one integration of the solution serves both checks.
            return self._samples_of(shoot.trajectory(v, n_points))
        key = shoot.memo_key(v, n_points, tuple(sorted(self.options.items())))
        return list(shoot.memoized("free_samples", key,
                                   lambda: self._sample_segments(shoot, v, n_points)))

    @staticmethod
    def _samples_of(tr):
        """``_free_samples`` from a trajectory: its runs of one arc, each with the limits
        it rides (none on a free arc)."""
        A, act = tr["arc"], tr["active"]
        cuts = [0] + [i for i in range(1, len(A)) if A[i] != A[i - 1]] + [len(A)]
        out = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            active = tuple(int(c) for c in np.flatnonzero(act[a])) if A[a] >= 0 else ()
            out.append((tr["t"][a:b], tr["g"][a:b], active))
        return out

    def _sample_segments(self, shoot, v, n_points):
        """``_free_samples`` for a multi-arc structure, without reuse."""
        k = shoot.kernels
        _, _, segs = shoot._evaluate(np.asarray(v, dtype=float), jac=False)
        span = segs[-1][3] - shoot.t0
        out = []
        for kind, aid, ta, tb, za, flags, _ in segs:
            m = max(int(n_points * (tb - ta) / span), 3)
            if kind is None:
                tr = k.trajectory(ta, za, tb, shoot.params, n_points=m, constraint_events=False,
                                  arc=shoot._free_arc(ta, za, flags), frozen=sorted(flags),
                                  **self.options)
                out.append((tr["t"], tr["g"], ()))
            elif kind == "boundary":
                ts, zs = sample_segment(k.BoundaryDynamics(aid, shoot.params), ta, tb, za, m,
                                        method="DOP853", rtol=self.options["rtol"],
                                        atol=self.options["atol"])
                g = np.array([k.constraint(ti, zi, shoot.params)
                              for ti, zi in zip(ts, zs)]).reshape(len(ts), -1)
                out.append((ts, g, tuple(k.arc_constraints(aid))))
        return out

    def _birth_fraction(self, shoot, v, overshoot, scale, target=2e-4):
        """Rule R16: the fraction of an overshooting step at which the first limit's
        peak excess reaches ``target`` (relative to ``scale``), by a secant between the
        consistent solution ``v`` and the overshooting one, or None.

        Before a birth the structure is fixed, and the peak excess is a C^1 function of
        the route parameter with derivative the explicit one at the peak (Lemma N), so
        the secant puts the step within O(ds^2) of the birth. Only the step length
        changes; the step is solved and judged as any other."""
        if not overshoot:
            return None
        try:
            samples = self._free_samples(shoot, v)
        except _FAILURES:
            return None
        best = None
        for c, p1 in overshoot.items():
            vals = [g[:, c] for t, g, active in samples if c not in active]
            if not vals:
                continue
            p0 = float(max(np.max(x) for x in vals)) / scale[c]
            if not p1 > target > p0:
                continue
            theta = (target - p0) / (p1 - p0)
            best = theta if best is None else min(best, theta)
        return None if best is None else float(np.clip(best, 0.05, 0.9))

    def _densities(self, shoot, v, n_points=801):
        """Per order-1 boundary arc, its measure's density along it:
        ``[(c, t, eta, rise, mu_max)]``, with ``eta = -dmu/dt`` at the midpoints ``t``
        of the samples of ``mu`` on the arc (Lemma H: ``mu(t) = int_t^{t_out} eta``),
        and ``rise`` the largest rise of ``mu`` along it (0 when ``eta >= 0``)."""
        if not isinstance(shoot, MultiArcShooting):
            return []
        k, p = shoot.kernels, shoot.params
        arcs, values = self._arcs_of(shoot, v)
        tr = shoot.trajectory(v, n_points)
        out = []
        for (kind, c), val in zip(arcs, values):
            if kind != "boundary" or self.kr.CONSTRAINT_ORDERS[c] != 1:
                continue
            on = np.flatnonzero(tr["active"][:, c] & (tr["t"] >= val[0]) & (tr["t"] <= val[1]))
            if len(on) < 3:
                continue
            mu = []
            for j in on:
                aid = tr["arc"][j]
                m_ = k.boundary_multiplier(aid, tr["t"][j], tr["z"][j], p)
                mu.append(float(np.atleast_1d(m_)[0]) if aid < k.N_CONSTRAINTS
                          else float(np.atleast_1d(m_)[list(k.arc_constraints(aid)).index(c)]))
            t, mu = tr["t"][on], np.asarray(mu)
            dt = np.diff(t)
            keep = dt > 0
            eta = -np.diff(mu)[keep] / dt[keep]
            # The rise of the continuous part of mu: a jump between two samples at one
            # time (another limit's junction) is not the density's doing.
            cont = np.concatenate([[0.0], np.cumsum(np.diff(mu)[keep])])
            rise = float(np.max(cont - np.minimum.accumulate(cont)))
            out.append((c, 0.5 * (t[1:] + t[:-1])[keep], eta, rise, float(np.abs(mu).max())))
        return out

    def _split_fraction(self, shoot0, v0, shoot1, v1, too_far=1e-3, target=0.1):
        """Rule R25: the fraction of a step at which an order-1 arc's density first
        reaches zero inside the arc, or None.

        The step ends at ``(shoot1, v1)``, where the density ``eta`` of some arc is
        negative enough that ``mu`` rises by more than ``too_far`` (relative to
        ``max(1, |mu|)``, the scale of ``_verdict``'s check) along the arc; it started
        at ``(shoot0, v0)``, where an arc of the same limit covered that time with
        ``eta > 0`` there. So the event (V4's E3, a split) is inside the step.

        At a fixed time ``t*`` the density is a C^1 function of the route parameter
        while the structure is fixed (V2), so the secant on ``eta(t*)`` between the
        two ends places the step where ``eta(t*) = target * eta_1(t*)``: just past
        the zero, to O(ds^2), with a small rise for Rule R5 to split. Only the
        step length changes; the step is solved and judged as any other."""
        try:
            new = self._densities(shoot1, v1)
        except _FAILURES:
            return None
        worst = None
        for c, t, eta, rise, mu_max in new:
            if len(eta) == 0 or rise <= too_far * max(1.0, mu_max):
                continue
            j = int(np.argmin(eta))
            if worst is None or rise > worst[0]:
                worst = (rise, c, float(t[j]), float(eta[j]))
        if worst is None:
            return None
        _, c, t_star, eta1 = worst
        if not eta1 < 0.0:
            return None
        try:
            old = self._densities(shoot0, v0)
        except _FAILURES:
            return None
        for c0, t, eta, _, _ in old:
            if c0 != c or len(t) < 2 or not (t[0] <= t_star <= t[-1]):
                continue
            eta0 = float(np.interp(t_star, t, eta))
            if not eta0 > 0.0:
                return None                # the step starts past the event: R5 splits
            memo = getattr(self, "_split_memo", None)
            self._split_memo = (c, t_star, eta1)
            if (getattr(self, "density_jumps", False) and memo is not None and memo[0] == c
                    and abs(memo[1] - t_star) < 1e-3 * (shoot1.unpack(v1)[3] - shoot1.t0)
                    and eta1 < 0.5 * memo[2]):
                # Rule R54: the last location put the step where eta(t*) is a tenth of its
                # end value, if eta(t*) is continuous in s. It is not smaller by half: eta
                # jumps at t* where a held-set change sweeps through it (Lemma P), at the
                # base. No shorter step reaches the zero; R5 splits from here.
                self.diagnostics.fired("R54")
                self._split_memo = None
                self._detail(f"    {self.kr.CONSTRAINT_NAMES[c]}: density at t = {t_star:.5f} "
                             f"jumps at the base ({memo[2]:.2e}, then {eta1:.2e}): no location")
                return None
            theta = (eta0 - target * eta1) / (eta0 - eta1)
            self._detail(f"    {self.kr.CONSTRAINT_NAMES[c]}: density at t = {t_star:.5f} "
                         f"falls from {eta0:.2e} to {eta1:.2e} over the step")
            return float(np.clip(theta, 0.05, 0.9))
        return None

    def _free_violations(self, shoot, v, scale, floor=None):
        """``[(c, start, end, peak)]``: where each limit is violated while it is not ridden
        (by more than ``floor``, by default 1e-6 of its scale)."""
        out = []
        for t, g, active in self._free_samples(shoot, v):
            for c in range(self.nc):
                if c in active:
                    continue
                # Above rounding at tangential junctions.
                bad = g[:, c] > (1e-6 * scale[c] if floor is None else floor)
                if not bad.any():
                    continue
                idx = np.flatnonzero(bad)
                for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                    a, b = run[0], run[-1]
                    # Widen to the neighbouring samples, where g was still <= 0.
                    ta, tb = t[max(a - 1, 0)], t[min(b + 1, len(t) - 1)]
                    out.append((c, ta, tb, float(g[run, c].max())))
        # One violation can run across a junction of other limits: it is one, not two.
        merged = []
        for c, ta, tb, peak in sorted(out, key=lambda v: (v[0], v[1])):
            if merged and merged[-1][0] == c and ta <= merged[-1][2]:
                merged[-1] = (c, merged[-1][1], max(tb, merged[-1][2]), max(peak, merged[-1][3]))
            else:
                merged.append((c, ta, tb, peak))
        return merged

    def _merge_touching(self, arcs, values):
        """Rule R11: boundary arcs of one limit that touch or overlap are one arc.

        When the gap between two arcs of a limit closes (a split whose stretch off the
        limit dies, or a predictor carrying one arc into the next), their union is the
        set of times on the limit, so they are one arc. Its entry jump carries both,
        since mu = int d eta adds over the pieces (the convention of ``_activate``)."""
        arcs, values = list(arcs), list(values)
        merged = self.merge_touching
        while merged:
            merged = False
            # Consecutive arcs of the same limit, whatever other limits' arcs start between
            # them (Lemma E4: a gap of width <= 0 is the merged arc).
            order = sorted((i for i, (kind, _) in enumerate(arcs) if kind == "boundary"),
                           key=lambda i: (arcs[i][1], float(values[i][0])))
            for i, j in zip(order, order[1:]):
                if arcs[i][1] != arcs[j][1]:
                    continue
                (a0, a1, pa), (b0, b1, pb) = values[i], values[j]
                if b0 <= a1:
                    keep = (a0, max(a1, b1), np.asarray(pa, dtype=float) + np.asarray(pb, dtype=float))
                    values[i] = keep
                    del arcs[j], values[j]
                    self.diagnostics.fired("R11")
                    merged = True
                    break
        return arcs, values

    def _mas_entries(self, arcs, values, names=False):
        """MultiArcShooting entries for arcs that may overlap: ``(entries, values)``.

        The events (entries, exits, touches) are ordered by their times, so a
        limit that comes into play while another is ridden becomes an arc
        inside or across that one; the values follow the order of entry.
        """
        arcs, values = self._merge_touching(arcs, values)
        # A predictor can carry a short arc past zero length: it stays an arc, just
        # barely, and the solve or the retirement rule decides what becomes of it.
        values = [(v[0], max(v[1], v[0] + 1e-10 * max(1.0, abs(v[0]))), *v[2:])
                  if kind == "boundary" and np.isfinite(v[0]) else v
                  for (kind, _), v in zip(arcs, values)]
        events = []
        for i, ((kind, c), val) in enumerate(zip(arcs, values)):
            if kind == "touch":
                events.append((float(val[0]), 1, "touch", i))
            else:
                events.append((float(val[0]), 1, "enter" if np.isfinite(val[0]) else "start", i))
                if np.isfinite(val[1]):
                    events.append((float(val[1]), 0, "exit", i))
        events.sort(key=lambda e: (e[0], e[1]))
        label = (lambda c: self.kr.CONSTRAINT_NAMES[c]) if names else (lambda c: c)
        depth, overlap = 0, False
        for _, _, ev, _ in events:
            if ev != "exit" and depth > 0:
                overlap = True             # an entry or touch while an arc is ridden
            depth += 1 if ev in ("enter", "start") else -1 if ev == "exit" else 0
        if any(ev == "start" for _, _, ev, _ in events) or depth > 0:
            overlap = True                 # only the event list can say "start" or leave one open
        order = [i for _, _, ev, i in events if ev != "exit"]
        if not overlap:
            # Plain sequence: the compact entries.
            entries = [("boundary" if arcs[i][0] == "boundary" else "touch", label(arcs[i][1]))
                       for i in order]
        else:
            entries = [(ev, label(arcs[i][1])) for _, _, ev, i in events]
        return entries, [values[i] for i in order]

    @staticmethod
    def _arcs_of(shoot, v):
        """The structure and its values, as MultiArcShooting entries (``[]`` for none)."""
        if not isinstance(shoot, MultiArcShooting):
            return [], []
        # An arc ridden from the start has t_in = -inf, one ridden to the end t_out = inf.
        values = []
        for i, val in enumerate(shoot.unpack_arcs(v)):
            if shoot.arcs[i][0] == "boundary":
                val = (-np.inf if i in shoot._started else val[0],
                       np.inf if np.isnan(val[1]) else val[1], val[2])
            values.append(val)
        return [(kind, c) for kind, c, _ in shoot.arcs], values

    def _activate(self, shoot, v, violations):
        """Structures that bring the violated limits in, most likely first.

        A violation away from the limit's arcs becomes a newborn touch point
        (order 2; always for order 3 or more, which has no arcs) or a short
        boundary arc (order 1), with zero jump. A
        violation beside arcs of the same limit widens them into one
        boundary arc that covers it: a boundary arc widens, a touch point
        grows into a boundary arc (its jump kept as the first ``pi``), and
        neighbouring arcs with the violation between them merge. A touch
        point violated on both sides may instead split into two touch points
        that share its jump; that structure is offered second. Returns a list
        of ``(arcs, values)``.
        """
        kr = self.kr
        arcs, values = self._arcs_of(shoot, v)
        T = shoot.unpack(v)[3] - shoot.t0
        near = 0.05 * T
        t_start = shoot.t0

        def on_limit(c, which):
            return self._on_limit(shoot, v, c, which)

        def span(i):
            kind, val = arcs[i][0], values[i]
            return (val[0], val[0]) if kind == "touch" else (val[0], val[1])

        groups, born = [], []              # groups: [set of arc indices, [(a, b), ...]]
        touch_born = {}                    # R20: index in born -> the birth as a touch
        split_off = getattr(self, "separate_births", False)
        samples = self._free_samples(shoot, v) if split_off else []

        def separated(c, lo, hi):
            """Rule R26: the limit is strictly below its level somewhere in the free
            time between ``lo`` and ``hi``, so a violation there is its own peak."""
            if not hi > lo:
                return False
            gs = [g[(t > lo) & (t < hi), c] for t, g, active in samples if c not in active]
            gs = [x for x in gs if len(x)]
            if not gs:
                return False
            reach = max(float(np.abs(g[:, c]).max()) for t, g, active in samples if c not in active)
            return float(min(x.min() for x in gs)) < -1e-6 * max(reach, 1e-300)

        for c, a, b, _ in violations:
            hits = {i for i, (_, cc) in enumerate(arcs)
                    if cc == c and a - near <= span(i)[1] and b + near >= span(i)[0]}
            if split_off:
                # Lemma E (event E1): a violation is a birth unless the limit stays at its
                # level between it and the arc; only then does the arc widen over it.
                kept = {i for i in hits
                        if not (separated(c, span(i)[1], a) or separated(c, b, span(i)[0]))}
                if kept != hits:
                    self.diagnostics.fired("R26")
                hits = kept
            q = kr.CONSTRAINT_ORDERS[c]
            if q == 1 and not hits:
                # On the limit at an end of the motion: the arc reaches that end.
                if a <= t_start + near and on_limit(c, 0):
                    born.append((("boundary", c), (-np.inf, b, np.zeros(1))))
                    continue
                if b >= t_start + T - near and on_limit(c, -1):
                    born.append((("boundary", c), (a, np.inf, np.zeros(1))))
                    continue
            if q >= 3:
                # Order 3 or more: touch points only (no boundary arcs with a smooth
                # control law; Jacobson, Lele & Speyer 1971), so each violation is one more.
                born.append((("touch", c), (0.5 * (a + b), 0.0)))
                continue
            if not hits:
                if q == 2:
                    born.append((("touch", c), (0.5 * (a + b), 0.0)))
                else:
                    w = max(b - a, 1e-3 * T) / 2
                    mid = 0.5 * (a + b)
                    lo, hi = mid - w, mid + w
                    t_edge = (self._corner_peak(shoot, v, c, lo, hi, rel=self.corner_rel)
                              if self.corner_touches else None)
                    if t_edge is not None:
                        # Lemma L: a peak on the edge is born as a touch point, not an arc.
                        self.diagnostics.fired("R12")
                        born.append((("touch", c), (t_edge, 0.0)))
                        continue
                    if self.holdable_births:
                        lo, hi = self._holdable(shoot, v, c, lo, hi)
                    born.append((("boundary", c), (lo, hi, np.zeros(q))))
                    if self.collapsed_touches:
                        touch_born[len(born) - 1] = (("touch", c), (self._peak_time(shoot, v, c, a, b), 0.0))
                continue
            for g in groups:
                if g[0] & hits:
                    g[0] |= hits
                    g[1].append((a, b))
                    break
            else:
                groups.append([set(hits), [(a, b)]])

        def overlap(lo, hi):
            """The span of the arc this one overlaps most, or None."""
            spans = [span(i) for i, (kind, _) in enumerate(arcs) if kind == "boundary"]
            best = max(spans, key=lambda y: min(hi, y[1]) - max(lo, y[0]), default=None)
            return best if best and min(hi, best[1]) > max(lo, best[0]) else None

        def placed(entry_in, exit_in):
            """The newborn arcs with their ends moved inside the arc they overlap."""
            out = []
            for entry, val in born:
                other = overlap(val[0], val[1]) if entry[0] == "boundary" else None
                if other is None:
                    out.append((entry, val))
                    continue
                inset = 1e-3 * min(other[1] - other[0], T)
                lo = max(val[0], other[0] + inset) if entry_in else val[0]
                hi = min(val[1], other[1] - inset) if exit_in else val[1]
                out.append((entry, (lo, hi, val[2])))
            return out

        def build(split, newborn=None, narrow=False):
            replaced = {}                  # arc index -> what stands in its place
            for idx, spans in groups:
                lo_i = min(idx)
                (kind, c), val = arcs[lo_i], values[lo_i]
                for i in idx:
                    replaced[i] = []
                if split and len(idx) == 1 and kind == "touch":
                    tc, nu = val
                    left = [s_ for s_ in spans if s_[1] <= tc + 1e-12 * T]
                    right = [s_ for s_ in spans if s_[0] >= tc - 1e-12 * T]
                    if left and right:
                        ta = 0.5 * (min(x for x, _ in left) + tc)
                        tb = 0.5 * (tc + max(y for _, y in right))
                        replaced[lo_i] = [(("touch", c), (ta, 0.5 * nu)),
                                          (("touch", c), (tb, 0.5 * nu))]
                        continue
                # The merged arc's entry jump carries the jumps of all it absorbs. Only
                # this limit's arcs go: another limit's arc may lie between them.
                pi = np.zeros(kr.CONSTRAINT_ORDERS[c])
                for i in idx:
                    if arcs[i][0] == "touch":
                        pi[0] += values[i][1]
                    else:
                        pi += np.asarray(values[i][2], dtype=float)
                lo = min([span(i)[0] for i in idx] + [x for x, _ in spans])
                hi = max([span(i)[1] for i in idx] + [y for _, y in spans])
                if narrow and len(idx) == 1 and kind == "touch":
                    # A touch point becomes a short arc about its time: at order 2 the
                    # arc born is much shorter than the stretch first violated.
                    tc = val[0]
                    w = 0.1 * max(min(tc - lo, hi - tc), 1e-6 * T)
                    lo, hi = tc - w, tc + w
                replaced[lo_i] = [(("boundary", c), (lo, hi, pi))]
            out = []
            for i, item in enumerate(zip(arcs, values)):
                out += replaced.get(i, [item])
            out += list(born if newborn is None else newborn)
            out.sort(key=lambda w: float(w[1][0]))
            return [a for a, _ in out], [w for _, w in out]

        # A limit that comes into play while another is ridden may enter and
        # leave during that arc, or cross either end. The events' order is
        # fixed in a solve, and a guess in the wrong order does not converge;
        # nested is the commonest, as two limits' peaks close in on each other.
        options = []
        if any(entry[0] == "boundary" and overlap(val[0], val[1]) for entry, val in born):
            options += [build(False, placed(True, True)), build(False),
                        build(False, placed(True, False)), build(False, placed(False, True))]
        else:
            options = [build(False)]
        if any(len(g[0]) == 1 and arcs[next(iter(g[0]))][0] == "touch" for g in groups):
            alternative = build(True)
            if alternative[0] != options[0][0]:
                options.append(alternative)
            narrow = build(False, narrow=True)
            convex = (self._touch_curvatures(shoot, v)
                      if getattr(self, "touch_curvature", False) else {})
            if any(len(g[0]) == 1 and next(iter(g[0])) in convex for g in groups):
                # Rule R35: the violated touch is convex, so it has become an arc: the
                # short arc about it first, before the split into two touches.
                self.diagnostics.fired("R35")
                options.insert(0, narrow)
            else:
                options.append(narrow)
        if touch_born:
            # Rule R20: the order-1 births as touch points, offered after the arcs.
            options.append(build(False, [touch_born.get(i, b) for i, b in enumerate(born)]))
        return options

    def _touch_arcs(self, shoot, v, widths=(1e-4, 1e-3, 1e-2)):
        """Rule R20: the structure with each order-1 touch point made a boundary arc
        centred on it, carrying its jump, for each relative width in ``widths``."""
        if not isinstance(shoot, MultiArcShooting):
            return []
        arcs, values = self._arcs_of(shoot, v)
        return self._touches_as_arcs(arcs, values, shoot.unpack(v)[3] - shoot.t0, widths)

    def _touches_as_arcs(self, arcs, values, T, widths=(1e-4, 1e-3, 1e-2)):
        """Rule R20's conversion of the structure ``(arcs, values)`` (of duration ``T``)."""
        if not any(k_ == "touch" and self.kr.CONSTRAINT_ORDERS[c] == 1 for k_, c in arcs):
            return []
        out = []
        for rel in widths:
            new_v = []
            for (kind, c), val in zip(arcs, values):
                if kind == "touch" and self.kr.CONSTRAINT_ORDERS[c] == 1:
                    w = 0.5 * rel * T
                    new_v.append((val[0] - w, val[0] + w, np.array([float(val[1])])))
                else:
                    new_v.append(val)
            new_a = [("boundary", c) if kind == "touch" and self.kr.CONSTRAINT_ORDERS[c] == 1
                     else (kind, c) for kind, c in arcs]
            out.append((new_a, new_v))
        return out

    def _touch_curvatures(self, shoot, v, rel=1e-3):
        """Rule R35: the order-2 touch points of ``v`` whose limit is convex there,
        ``{arc index: g''}``. At order 2 the control does not enter ``g'``, so
        ``g'' = grad(g') . (f0 + G u)`` is exact at a state: it is taken at the samples
        on either side of the touch (the control is continuous there) and averaged. A
        curvature counts when it exceeds ``rel`` times the largest ``|g'|`` over ``T``
        (``g''``'s own scale). The trajectory is R5's (801 points), so a route step
        samples ``v`` once for both."""
        if not isinstance(shoot, MultiArcShooting):
            return {}
        arcs, values = self._arcs_of(shoot, v)
        touches = [i for i, (kind, c) in enumerate(arcs)
                   if kind == "touch" and self.kr.CONSTRAINT_ORDERS[c] == 2]
        if not touches:
            return {}
        from auto_optimal_control.lemmas import LimitObjects
        try:
            obj = LimitObjects(shoot.kernels, shoot.params)
            tr = shoot.trajectory(v, n_points=801)
        except Exception:
            return {}
        T = shoot.unpack(v)[3] - shoot.t0
        ts, zs, us = np.asarray(tr["t"]), tr["z"], np.asarray(tr["u"], dtype=float)
        out = {}
        for i in touches:
            c, tc = arcs[i][1], float(values[i][0])
            before = np.flatnonzero(ts <= tc)
            after = np.flatnonzero(ts >= tc)
            if not len(before) or not len(after):
                continue
            m0, m1 = before[-1], after[0]
            g2 = 0.5 * (obj.curvature(c, ts[m0], zs[m0], us[m0])
                        + obj.curvature(c, ts[m1], zs[m1], us[m1]))
            reach = max(abs(obj.rate_parts(c, ts[m], zs[m])[1])
                        for m in range(0, len(ts), max(1, len(ts) // 200)))
            if g2 > rel * max(reach, 1e-300) / T:
                out[i] = g2
        return out

    def _peak_time(self, shoot, v, c, a, b, n_points=2001):
        """The time of the largest value of limit ``c`` on ``[a, b]`` at ``v``."""
        best_t, best_g = 0.5 * (a + b), -np.inf
        try:
            samples = self._free_samples(shoot, v, n_points)
        except _FAILURES:
            return best_t
        for t, g, active in samples:
            inside = (t >= a) & (t <= b)
            if c in active or not inside.any():
                continue
            j = np.flatnonzero(inside)[int(np.argmax(g[inside, c]))]
            if g[j, c] > best_g:
                best_t, best_g = float(t[j]), float(g[j, c])
        return best_t

    def _on_limit(self, shoot, v, c, which):
        """Whether the true limit ``c`` holds at the start (0) or the end (-1) of the motion."""
        p = np.array(shoot.params, dtype=float)
        p[len(p) - self.nc:] = 0.0                 # the offsets come last
        _, _, segs = shoot._evaluate(np.asarray(v, dtype=float), jac=False)
        t, z = (shoot.t0, segs[0][4]) if which == 0 else (segs[-1][3], segs[-1][-1]["z"])
        return abs(float(shoot.kernels.constraint(t, z, p)[c])) < 1e-5

    def _end_structure(self, shoot, v):
        """The structure with arcs next to an end of the motion running out to it.

        As the offsets reach zero with an end state on a limit, the free stretch
        between that arc and the end shrinks to nothing, which no step can
        cross: the arc must become one ridden from the start or to the end.
        Returns ``(arcs, values)``, or None if no arc qualifies."""
        if not isinstance(shoot, MultiArcShooting) or shoot.kernels.N_SWITCHING:
            return None
        arcs, values = self._arcs_of(shoot, v)
        tf = shoot.unpack(v)[3]
        near = 0.1 * (tf - shoot.t0)
        out, changed = [], False
        for (kind, c), val in zip(arcs, values):
            if kind == "boundary" and shoot.kernels.CONSTRAINT_ORDERS[c] == 1:
                if np.isfinite(val[1]) and tf - val[1] < near and self._on_limit(shoot, v, c, -1):
                    val, changed = (val[0], np.inf, val[2]), True
                if np.isfinite(val[0]) and val[0] - shoot.t0 < near and self._on_limit(shoot, v, c, 0):
                    val, changed = (-np.inf, val[1], val[2]), True
            out.append(val)
        return (arcs, out) if changed else None

    def _bound_options(self, shoot, v):
        """Structures that undo an overlap of arcs whose riding control leaves its bounds.

        Riding as many limits as controls fixes every control, so a stretch that
        needs more than the bound allows cannot be ridden. One arc must then end
        before that stretch, or start after it; the junction there has a control
        at its bound, so ``H`` continuity holds at it. Returns a list of
        ``(arcs, values)``, or ``None`` when every riding control is in bounds."""
        if not isinstance(shoot, MultiArcShooting) or len(shoot.arcs) < 2:
            return None
        arcs, values = self._arcs_of(shoot, v)
        tr = shoot.trajectory(v, 801)
        k, p = shoot.kernels, shoot.params
        lo, hi = (np.array(b, dtype=float) for b in zip(*[k.control_bounds(t, z, p)
                                                          for t, z in zip(tr["t"], tr["z"])]))
        over = np.maximum(tr["u"] - hi, lo - tr["u"]).max(axis=1)
        bad = (over > 1e-6 * np.maximum(hi - lo, 1e-12).max(axis=1)) & (tr["active"].sum(axis=1) >= 2)
        if not bad.any():
            return None
        t_bad = tr["t"][bad]
        ta, tb = float(t_bad[0]), float(t_bad[-1])
        margin = 1e-3 * (tr["t"][-1] - tr["t"][0])
        hit = [i for i, ((kind, _), val) in enumerate(zip(arcs, values))
               if kind == "boundary" and val[0] <= tb and val[1] >= ta]
        options = []
        for i in hit:
            a, b, pi = values[i]
            for new in ((a, ta - margin, pi), (tb + margin, b, pi)):
                if new[1] - new[0] > margin:
                    options.append((arcs, [new if j == i else w for j, w in enumerate(values)]))
        return options or None

    def _jittered(self, arcs, values, rng):
        """A copy of a structure's values with junction times and jumps perturbed."""
        out = []
        for (kind, _), val in zip(arcs, values):
            if kind == "boundary":
                a, b, pi = val
                width = (b - a) if np.isfinite(a) and np.isfinite(b) else 0.05
                a2 = a + rng.uniform(-0.25, 0.25) * width if np.isfinite(a) else a
                b2 = b + rng.uniform(-0.25, 0.25) * width if np.isfinite(b) else b
                out.append((a2, b2, np.asarray(pi, dtype=float) * rng.uniform(0.5, 1.5)))
            elif kind == "touch":
                t, nu = val
                out.append((t * (1 + rng.uniform(-0.01, 0.01)), nu * rng.uniform(0.5, 1.5)))
            else:
                out.append(val)
        return out

    def _routes_in_parallel(self, members, first, eps0, s_max, steps, tol):
        """Rule R15 in parallel: each route of the portfolio in a forked process with the
        whole time budget; the first exact answer is solved again here (and accepted or
        not by the same checks). ``(J, shoot, sol)`` or None."""
        import multiprocessing as mp
        ctx = mp.get_context("fork")

        def child(conn, member):
            try:
                for k_, val_ in member.items():
                    setattr(self, k_, val_)
                self.portfolio = []
                hit = self._active_set(first, eps0, s_max, steps, tol,
                                       max_solves=self.max_path_solves)
                out = None
                if hit is not None:
                    _, xs, xsol = hit
                    out = (self._entries(xs), np.asarray(xsol.v, dtype=float))
                conn.send((out, self.diagnostics.as_dict()))
            except Exception:
                conn.send((None, {}))
            conn.close()

        running = []
        started = time.perf_counter()
        for n_, member in enumerate(members):
            parent, kid = ctx.Pipe(duplex=False)
            p = ctx.Process(target=child, args=(kid, member), daemon=True)
            p.start()
            kid.close()
            running.append((n_, p, parent))
        deadline = time.perf_counter() + self.active_set_seconds + 30.0
        if self._solve_end is not None:
            deadline = min(deadline, self._solve_end)
        found = None
        while running and found is None and time.perf_counter() < deadline:
            time.sleep(0.05)
            for item in list(running):
                n_, p, conn = item
                if not conn.poll():
                    if not p.is_alive():
                        running.remove(item)
                    continue
                try:
                    out, diag = conn.recv()
                except EOFError:
                    out, diag = None, {}
                running.remove(item)
                p.join(timeout=1)
                self.diagnostics.routes.append(
                    [n_ + 1, round(time.perf_counter() - started, 2),
                     "exact" if out is not None else "ended"])
                for part in ("evaluations", "steps", "rejected", "rules", "oracles"):
                    for key_, cnt in diag.get(part, {}).items():
                        table = getattr(self.diagnostics, part)
                        table[key_] = table.get(key_, 0) + cnt
                if out is not None:
                    entries, v = out
                    hit = self._exact_jump(list(entries), v, tol)
                    if hit is not None:
                        if n_:
                            self.diagnostics.fired("R15")
                        self._say(f"active set: route {n_ + 1} of {len(members)} found it")
                        found = hit
                        break
        for n_, p, conn in running:
            self.diagnostics.routes.append(
                [n_ + 1, round(time.perf_counter() - started, 2), "stopped"])
            p.kill()
            p.join(timeout=1)
            conn.close()
        return found

    def _speculate(self, options, base, le, off, solve_structure, state):
        """Solve candidate structures at once in forked processes; the first in order
        that converges within ``candidate_seconds``, as ``((arcs, values), shoot, sol)``."""
        import multiprocessing as mp
        rng = np.random.default_rng(len(options) + int(1e6 * float(np.sum(np.abs(off)))) % 2 ** 31)
        tasks = []
        for arcs, values in options:
            tasks.append((arcs, values))
            tasks += [(arcs, self._jittered(arcs, values, rng)) for _ in range(self.variants)]
        ctx = mp.get_context("fork")

        def child(conn, arcs, values):
            try:
                _, r = solve_structure(arcs, values, base, le, off)
                conn.send((bool(r.success), float(r.residual_norm), r.v))
            except Exception:
                conn.send((False, float("inf"), None))
            conn.close()

        results = {}
        pending = list(enumerate(tasks))
        running = {}
        while pending or running:
            while pending and len(running) < self.parallel:
                i, (arcs, values) = pending.pop(0)
                parent, kid = ctx.Pipe(duplex=False)
                p = ctx.Process(target=child, args=(kid, arcs, values), daemon=True)
                p.start()
                kid.close()
                running[i] = (p, parent, time.perf_counter())
            time.sleep(0.01)
            for i, (p, conn, started) in list(running.items()):
                if conn.poll():
                    try:
                        results[i] = conn.recv()
                    except EOFError:
                        results[i] = (False, float("inf"), None)
                elif p.is_alive() and time.perf_counter() - started < self.candidate_seconds:
                    continue
                elif p.is_alive():
                    results[i] = (False, float("inf"), None)
                else:
                    results.setdefault(i, (False, float("inf"), None))
                if p.is_alive():
                    p.kill()
                p.join(timeout=1)
                conn.close()
                del running[i]
            # Stop early once the best-ranked candidate still possible has converged.
            done = [i for i in sorted(results) if results[i][0]]
            if done and all(j in results for j in range(done[0])):
                for p, conn, _ in running.values():
                    p.kill()
                    p.join(timeout=1)
                    conn.close()
                running.clear()
                pending.clear()
        state["solves"] += len(results)
        per = self.variants + 1
        converged = [i for i in sorted(results) if results[i][0]]
        for i in sorted(results):
            ok, _, v = results[i]
            if not ok:
                continue
            arcs, values = tasks[i]
            original = tasks[(i // per) * per][1]
            moved = [(round(float(w[0] - o[0]), 6), round(float(w[1] - o[1]), 6) if np.isfinite(w[1]) and np.isfinite(o[1]) else None,
                      round(float(np.atleast_1d(w[2])[0] / np.atleast_1d(o[2])[0]), 3) if len(w) > 2 and np.atleast_1d(o[2])[0] else None)
                     if kind == "boundary" else None for (kind, _), w, o in zip(arcs, values, original)]
            self._detail(f"    parallel pick: option {i // per} of {len(options)}, "
                         f"{'original' if i % per == 0 else f'jittered copy {i % per}'}; "
                         f"converged {[(j // per, j % per) for j in converged]} of {len(tasks)}; "
                         f"residuals of the originals {[round(results[j][1], 9) for j in range(0, len(tasks), per) if j in results]}; "
                         f"moved (d t_in, d t_out, pi ratio) {moved}")
            entries, _ = self._mas_entries(arcs, values)
            sh = self._smoothed(entries, self._eps_at(le), off)
            r = sh.solve_chord(v, tol=max(1e-6, 0.0)) if self.kr.system.riding is not None else \
                sh.solve(v, tol=1e-6, options=_budget("hybr", self.path_evaluations))
            if r.success:
                return (arcs, values), sh, r
        return None

    @staticmethod
    def _event_labels(shoot):
        """Each event of ``shoot`` as ``(kind, limit)``: ``("out", c)``, ``("in", c)``..."""
        return [(ev, shoot.arcs[i][1]) for ev, i in shoot.events]

    def _tied_events(self, shoot):
        """Rule R39: the indices of ``shoot``'s events tied to the next one."""
        labels = self._event_labels(shoot)
        return {j for j in range(len(labels) - 1)
                if (labels[j], labels[j + 1]) in self._ties and labels[j][0] != "touch"}

    def _tie_locked(self, shoot, v, tol=1e-5):
        """Rule R39: tie the closest adjacent pair of junctions of different limits
        (neither a touch) that lie within ``tol`` of T, untied so far. True if one was."""
        if not isinstance(shoot, MultiArcShooting) or not shoot.arcs:
            return False
        times = shoot.event_times(v)
        labels = self._event_labels(shoot)
        T = shoot.unpack(v)[3] - shoot.t0
        best = None
        for j in range(len(times) - 1):
            a, b = labels[j], labels[j + 1]
            gap = abs(times[j + 1] - times[j])
            if (a[1] != b[1] and "touch" not in (a[0], b[0]) and (a, b) not in self._ties
                    and gap <= tol * T and (best is None or gap < best[0])):
                best = (gap, a, b)
        if best is None:
            return False
        _, a, b = best
        self._ties.add((a, b))
        self.diagnostics.fired("R39")
        N = self.kr.CONSTRAINT_NAMES
        self._detail(f"    {a[0]}:{N[a[1]]} and {b[0]}:{N[b[1]]} locked together at a "
                     f"failed step: one junction")
        return True

    def _tie_crossing(self, shoot, v):
        """Rule R39: tie the two junctions that cross most in ``v`` (of different limits,
        neither a touch). True if a new tie was made."""
        if not isinstance(shoot, MultiArcShooting) or not shoot.arcs:
            return False
        times = shoot.event_times(v)
        labels = self._event_labels(shoot)
        if len(times) < 2:
            return False
        j = int(np.argmax([a - b for a, b in zip(times, times[1:])]))
        a, b = labels[j], labels[j + 1]
        if a[1] == b[1] or "touch" in (a[0], b[0]) or (a, b) in self._ties:
            return False
        self._ties.add((a, b))
        self.diagnostics.fired("R39")
        N = self.kr.CONSTRAINT_NAMES
        self._detail(f"    {a[0]}:{N[a[1]]} and {b[0]}:{N[b[1]]} hand over: one junction")
        return True

    def _coincide(self):
        """The tolerance (of T) below which junctions of different limits coincide, in
        either order: Rule R38's ``coincide_tol`` when it is on, and never less than the
        integration's ``rtol``. Two junctions crossed by ``d`` leave a segment of width
        ``-d`` integrated backward, which moves the motion by ``O(d |f|)``: below
        ``rtol T`` that is inside the integration's own error, so no order of the two
        can be told from the other (by symmetry, rate arcs of a rest-to-rest motion
        enter and leave at one instant, and their computed times differ by rounding)."""
        floor = float(getattr(self, "options", {}).get("rtol", 0.0))
        if getattr(self, "coincident_junctions", False):
            return max(float(self.coincide_tol), floor)
        return floor

    def _turn_back(self, fold, frac, f, le0, off_start, target_le, target_off,
                   solve_structure, settle, state):
        """Rule R44: put the route on the other order's branch of a fold, at a point the
        route has already passed (the side of Lemma X's fold where both orders are
        valid). The seed moves along that branch's tangent from where it was solved.
        True if a solution of that order, in order, is settled there."""
        sh_b, v_b, tangent = fold["shoot"], fold["v"], fold["tangent"]
        for f_back in (frac, frac + 0.5 * (f - frac), frac + 0.75 * (f - frac)):
            le_b = le0 + f_back * (target_le - le0)
            off_b = off_start + f_back * (np.asarray(target_off, dtype=float) - off_start)
            seed = v_b + (f_back - f) * tangent
            arcs_b, vals_b = self._arcs_of(sh_b, seed)
            try:
                sh, r = solve_structure(arcs_b, vals_b, seed[:sh_b.n_base], le_b, off_b)
            except _FAILURES:
                continue
            if not (r.success and isinstance(sh, MultiArcShooting)
                    and sh.in_order(r.v, self._coincide())):
                continue
            state["backward"] = True       # no second fold test inside this settle
            try:
                done = settle(sh, r, le_b, off_b, changed=True)
            finally:
                state["backward"] = False
            if done is None:
                continue
            self._detail(f"    R44: the other order's branch at {f_back:.6f} of the leg "
                         f"(the fold at {f:.6f}); following it back")
            state.update(shoot=done[0], v=done[1].v, le=le_b, off=off_b)
            return True
        self._detail("    R44: the other order's branch could not be taken")
        return False

    @staticmethod
    def _event_column(shoot, j):
        """The column of ``v`` that holds event ``j``'s time."""
        ev, i = shoot.events[j]
        sl = shoot._arc_slices[i]
        return sl.start + ((0 if i in shoot._started else 1) if ev == "out" else 0)

    def _fold_rates(self, shoot, v, le, off, direction, first, second, h=1e-6):
        """Lemma X's quantities for one order: the gap ``t(second) - t(first)`` between two
        events (given as indices in ``shoot.events``), its rate along the route's
        ``direction`` = (d le, d off) per unit step, from the branch's tangent
        ``-J^-1 dR/ds``, the tangent itself, and J's condition after row and column
        equilibration."""
        v = np.asarray(v, dtype=float)
        dle, doff = direction
        p0 = shoot.params

        def residual_at(sign):
            shoot.params = np.asarray(self.regularized_params(
                self._eps_at(le + sign * h * dle), np.zeros(self.nc), off + sign * h * doff),
                dtype=float)
            try:
                return shoot.residual(v)
            finally:
                shoot.params = p0

        Rs = (residual_at(1.0) - residual_at(-1.0)) / (2 * h)
        _, J = shoot.residual_jacobian(v)
        rows = 1.0 / np.maximum(np.abs(J).max(axis=1), 1e-300)
        A = J * rows[:, None]
        cols = 1.0 / np.maximum(np.abs(A).max(axis=0), 1e-300)
        sv = np.linalg.svd(A * cols[None, :], compute_uv=False)
        tangent = -np.linalg.solve(J, Rs)
        times = shoot.event_times(v)
        gap = times[second] - times[first]
        rate = tangent[self._event_column(shoot, second)] - tangent[self._event_column(shoot, first)]
        return gap, rate, tangent, float(sv[0] / max(sv[-1], 1e-300))

    def _fold_hypotheses(self, shoot, v, t_star, window):
        """Whether Lemma X's hypotheses hold at ``t_star``: every arc within ``window`` of it
        rides with its control interior to the box, and the control gradients of the
        limits active there are independent (Lemma B's rank). ``(ok, why)``."""
        k, p = shoot.kernels, shoot.params
        G = shoot._control_matrix()
        if G is None:
            return False, "no constant control matrix"
        _, _, segs = shoot._evaluate(np.asarray(v, dtype=float), jac=False, record=True)
        active = set()
        for kind, aid, ta, tb, za, flags, rr in segs:
            if not (min(ta, tb) <= t_star + window and max(ta, tb) >= t_star - window):
                continue
            for tt, zz in ((ta, za), (tb, rr["z"])):
                u = np.asarray(k.control(tt, zz, p) if kind is None
                               else k.boundary_control(aid, tt, zz, p), dtype=float)
                lo, hi = (np.asarray(x, dtype=float) for x in k.control_bounds(tt, zz, p))
                if np.any(np.minimum(u - lo, hi - u) <= 1e-9 * np.maximum(hi - lo, 1.0)):
                    return False, "a control at its bound at the junction (a corner)"
            if kind == "boundary":
                active |= set(k.arc_constraints(aid)) if aid >= k.N_CONSTRAINTS else {aid}
        z_star = None
        for kind, aid, ta, tb, za, flags, rr in segs:
            if min(ta, tb) <= t_star <= max(ta, tb):
                z_star = za
                break
        if z_star is None or not active:
            return True, ""
        grads = []
        for c in sorted(active):
            q = k.CONSTRAINT_ORDERS[c]
            Nx = np.asarray(k.tangency_gradient(c, t_star, z_star, p)[1])
            g_u = G.T @ Nx[q - 1][:k.N_STATES]
            grads.append(g_u / max(np.linalg.norm(g_u), 1e-300))
        sv = np.linalg.svd(np.array(grads), compute_uv=False)
        if len(grads) > G.shape[1] or sv[-1] < 1e-6:
            return False, "the active limits' control gradients are dependent (Lemma B)"
        return True, ""

    def _fold_check(self, history, shoot, sol, le, off, direction):
        """Rule R44: whether the crossed solution ``sol`` and an earlier one in ``history``
        (this settle's solutions, same point) are the two orders of one pair of events,
        both crossed, at a fold of Lemma X. Returns the other order's branch to turn back
        on, or None."""
        if direction is None or not isinstance(shoot, MultiArcShooting):
            return None
        mine = self._event_labels(shoot)
        times = shoot.event_times(sol.v)
        by_time = [mine[j] for j in sorted(range(len(mine)), key=lambda j_: times[j_])]
        for sh_o, sol_o in reversed(history):
            theirs = self._event_labels(sh_o)
            if theirs != by_time:
                continue                   # not the order this solution's times ask for
            diff = [j for j in range(len(mine)) if mine[j] != theirs[j]]
            if not (len(diff) == 2 and diff[1] == diff[0] + 1
                    and mine[diff[0]] == theirs[diff[1]] and mine[diff[0]][1] != mine[diff[1]][1]):
                return None
            j = diff[0]
            if sh_o.in_order(sol_o.v, self._coincide()):
                return None                # the earlier order was not crossed: E5's reorder
            try:
                gA, aA, tA, cA = self._fold_rates(sh_o, sol_o.v, le, off, direction, j, j + 1)
                gB, aB, tB, cB = self._fold_rates(shoot, sol.v, le, off, direction, j + 1, j)
            except _FAILURES:
                return None
            # Common coordinate: the gap t(theirs[j+1]) - t(theirs[j]); valid >= 0 for the
            # earlier order (A), <= 0 for this one (B).
            if max(cA, cB) > self.fold_condition:
                self._detail(f"    fold test: Jacobian condition {max(cA, cB):.1e} (not regular)")
                return None
            t_star = 0.5 * (shoot.event_times(sol.v)[j] + sh_o.event_times(sol_o.v)[j])
            T = shoot.unpack(sol.v)[3] - shoot.t0
            ok, why = self._fold_hypotheses(shoot, sol.v, t_star, abs(gA) + abs(gB) + 1e-6 * T)
            if ok:
                ok, why = self._fold_hypotheses(sh_o, sol_o.v, t_star, abs(gA) + abs(gB) + 1e-6 * T)
            names = [self.kr.CONSTRAINT_NAMES[c] for _, c in theirs[j:j + 2]]
            if not ok:
                self._detail(f"    fold test at {names}: {why}")
                return None
            self._detail(f"    fold test at {[e for e, _ in theirs[j:j + 2]]} {names}: gap rates "
                         f"{aA:+.2e} and {aB:+.2e}" + (": a fold (Lemma X)" if aA * aB < 0 else ""))
            if aA * aB >= 0:
                return None
            return {"shoot": shoot, "v": np.asarray(sol.v, dtype=float), "tangent": tB,
                    "gap": gB, "rate": aB, "key": (tuple(names), round(t_star, 3))}
        return None

    def _retire(self, shoot, v):
        """The structure without arcs that have died: a boundary arc shrunk to
        nothing becomes a touch point (order 2 or more) or goes (order 1), and a touch
        point whose jump has the wrong sign goes. ``None`` if nothing changes."""
        arcs, values = self._arcs_of(shoot, v)
        if not arcs:
            return None
        T = shoot.unpack(v)[3] - shoot.t0
        # Junctions that crossed (rule R10 lets the residual through them): the same arcs,
        # with their events sorted again, is a structure change.
        crossed = isinstance(shoot, MultiArcShooting) and not shoot.in_order(v, self._coincide())
        out_a, out_v, changed = [], [], crossed
        convex = self._touch_curvatures(shoot, v) if getattr(self, "touch_curvature", False) else {}
        for i, ((kind, c), val) in enumerate(zip(arcs, values)):
            if i in convex:
                # Rule R35: the touch has become an arc. Short, and inside its
                # neighbours' events, so that no two junctions coincide (Lemma D).
                self.diagnostics.fired("R35")
                changed = True
                others = [t_ for j, (k_, v_) in enumerate(zip(arcs, values)) if j != i
                          for t_ in ((v_[0],) if k_[0] == "touch" else (v_[0], v_[1]))
                          if np.isfinite(t_)]
                # Events coincident with the touch (R38: other limits' touches at the
                # same instant, as by symmetry) bound no width: their arcs start and end
                # together with this one.
                same = self._coincide() * T
                gap = min((abs(t_ - val[0]) for t_ in others if abs(t_ - val[0]) > same),
                          default=T)
                w = min(1e-3 * T, 0.25 * gap)
                out_a.append(("boundary", c))
                out_v.append((val[0] - w, val[0] + w, np.array([float(val[1]), 0.0])))
                continue
            if kind == "boundary" and val[1] - val[0] < 1e-6 * T:
                changed = True
                if self.kr.CONSTRAINT_ORDERS[c] >= 2 or (
                        self.collapsed_touches and float(val[2][0]) > 1e-12):
                    # Rule R20 at order 1: a collapsed arc that still carries a jump.
                    if self.kr.CONSTRAINT_ORDERS[c] == 1:
                        self.diagnostics.fired("R20 collapse")
                    out_a.append(("touch", c))
                    out_v.append((0.5 * (val[0] + val[1]), float(val[2][0])))
                continue
            if kind == "touch" and val[1] < -1e-9:
                changed = True
                continue
            if (getattr(self, "retire_negative", False) and kind == "boundary"
                    and self.kr.CONSTRAINT_ORDERS[c] == 1
                    and float(np.atleast_1d(val[2])[0]) < -1e-7):
                # Rule R31: the jump monitor of an order-1 arc is negative.
                self.diagnostics.fired("R31")
                changed = True
                continue
            out_a.append((kind, c))
            out_v.append(val)
        return (out_a, out_v) if changed else None

    def _corner_peak(self, shoot, v, c, a, b, rel=1e-2):
        """The time of the peak of limit ``c`` on ``[a, b]`` if it lies on the edge of the
        holdable set (margin within ``rel`` of the jerks' reach there: the equality case
        of Lemma I'), else None."""
        from auto_optimal_control.lemmas import LimitObjects
        try:
            obj = LimitObjects(shoot.kernels, shoot.params)
            tr = shoot.trajectory(v, n_points=4001)
        except Exception:
            return None
        inside = np.flatnonzero((tr["t"] >= a) & (tr["t"] <= b))
        if len(inside) < 3:
            return None
        j0 = inside[int(np.argmax(tr["g"][inside, c]))]
        t0, z0 = float(tr["t"][j0]), tr["z"][j0]
        a_, d_ = obj.rate_parts(c, t0, z0)
        lo_, hi_ = (np.asarray(x, dtype=float) for x in shoot.kernels.control_bounds(t0, z0, shoot.params))
        reach = float(np.abs(a_) @ (0.5 * (hi_ - lo_)))
        m = obj.margin(c, t0, z0)
        return t0 if abs(m) <= rel * max(reach, 1e-300) else None

    def _holdable(self, shoot, v, c, a, b):
        """The stretch of ``[a, b]`` about the peak of limit ``c`` on which it can be held.

        Where a limit of order 1 is reached at an interior time, its one-sided rates
        have opposite signs, so (the control set being convex) some admissible control
        makes the rate zero: every contact lies in the set where the riding problem is
        feasible (Lemma I of ``docs/notes/structure_certificates.md``). In particular the
        peak of a violation does, up to the sampling. A newborn arc seeded where the
        limit cannot be held fails to integrate, so its ends are pulled in to the
        stretch about the peak where the riding control exists (the kernel's riding
        solve is exact: it fails only when no admissible control holds the limit)."""
        k, p = shoot.kernels, shoot.params
        try:
            tr = shoot.trajectory(v, n_points=4001)
        except Exception:
            return a, b
        ts, zs = tr["t"], tr["z"]
        inside = np.flatnonzero((ts >= a) & (ts <= b))
        if len(inside) < 3:
            return a, b
        g = tr["g"][inside, c]
        j0 = int(np.argmax(g))

        def ok(j):
            try:
                k.boundary_control(c, ts[inside[j]], zs[inside[j]][:2 * k.N_STATES], p)
                return True
            except Exception:
                return False

        if not ok(j0):
            # The peak is on the edge (equality in Lemma I's corollary: the free control
            # is at the corner there), so the arc is born one-sided, on the side where
            # the limit becomes holdable: start from the nearest holdable sample.
            near = [j for j in sorted(range(len(inside)), key=lambda j: abs(j - j0)) if ok(j)]
            if not near:
                return a, b
            j0 = near[0]
            self.diagnostics.fired("R6 one-sided")
        lo = j0
        while lo > 0 and ok(lo - 1):
            lo -= 1
        hi = j0
        while hi < len(inside) - 1 and ok(hi + 1):
            hi += 1
        if lo == 0 and hi == len(inside) - 1:
            return a, b
        T = shoot.unpack(v)[3] - shoot.t0
        lo_t, hi_t = float(ts[inside[lo]]), float(ts[inside[hi]])
        if hi_t - lo_t < 1e-4 * T:
            mid = float(ts[inside[j0]])
            lo_t, hi_t = mid - 5e-5 * T, mid + 5e-5 * T
        self.diagnostics.fired("R6")
        self._detail(f"    {self.kr.CONSTRAINT_NAMES[c]}: held only on [{lo_t:.5f}, {hi_t:.5f}] "
                     f"of [{a:.5f}, {b:.5f}]: the newborn arc is placed there")
        return lo_t, hi_t

    def _solve_structure(self, arcs, values, base, le, off, path_tol, state=None):
        """The path problem at ``(le, off)`` with structure ``(arcs, values)``, solved
        from ``base``: ``(shoot, sol)``. ``state["solves"]`` counts the solves."""
        if state is None:
            state = {"solves": 0}
        entries, values = self._mas_entries(arcs, values)
        sh = self._smoothed(entries, self._eps_at(le), off)
        guess = sh.pack_arcs(base, values) if arcs else np.asarray(base, dtype=float)
        state["solves"] += 1
        if isinstance(sh, MultiArcShooting) and (sh._open or sh._started):
            # On an arc that reaches a fixed end on the limit, the terminal row and the
            # riding say the same thing: rank-deficient, which needs the damped solver.
            return sh, sh.solve_damped(guess, tol=path_tol, max_steps=40, max_seconds=30.0)
        options = _budget("hybr", self.path_evaluations)
        if self.kr.system.riding is not None:
            # Limits ridden at run time make every evaluation dear: a cheap attempt
            # first, and only a step that gets close earns the slower solve.
            reject = getattr(self, "_reject_theta", None)
            r = sh.solve_chord(guess, tol=path_tol, reject_theta=reject)
            if getattr(r, "too_far", False):
                return sh, r               # Rule R37: a shorter step, not a slower solve
            if (not r.success and r.residual_norm >= 1e5 and self.edge_exits and arcs
                    and isinstance(sh, MultiArcShooting)):
                # The seed did not propagate. If an arc rides past the edge of the
                # holdable set, its exit is pulled in to the edge (Lemma I).
                pulled = sh.pull_in_exits(guess)
                if pulled is not None:
                    self.diagnostics.fired("R8")
                    self._detail("    exits pulled in to the edge of the holdable set")
                    guess = pulled
                    state["solves"] += 1
                    r = sh.solve_chord(guess, tol=path_tol)
            if r.success or not r.residual_norm < self.path_fallback:
                return sh, r
        r = sh.solve(guess, tol=path_tol, options=options)
        if (not r.success and r.residual_norm < 1e-2 and arcs and sh.tangent_junctions):
            # A control at its bound at a junction jumps there: H continuity, not tangency.
            state["solves"] += 1
            sh.tangent_junctions = False
            alt = sh.solve(guess, tol=path_tol, options=options)
            if alt.success:
                return sh, alt
            sh.tangent_junctions = True
        return sh, r

    def _width_step(self, shoot, v, frac, point, factor, tol, iterations=10, guess=None,
                    arc=None):
        """Rule R2: a step in a young arc's width, with the route's parameter solved for.

        Near a birth (or death) of an order-1 arc its width moves like the square root
        of the route's parameter (Lemma E), so steps in the parameter stall while steps
        in the width are regular. The narrowest finite order-1 arc's width is set to
        ``factor`` times its current value and the system ``[Phi(v; s), w - w_target]``
        is solved for ``(v, s)``, with ``point(s) = (le, off)`` the route's line.
        ``guess`` is a predicted ``(v, s)`` to start from, ``arc`` the arc's index.
        Returns ``(v, s, arc, w, w_target)`` or None."""
        if not isinstance(shoot, MultiArcShooting):
            return None
        T = shoot.unpack(v)[3] - shoot.t0
        best = None
        for i, ((kind, c, q), sl) in enumerate(zip(shoot.arcs, shoot._arc_slices)):
            if kind != "boundary" or q != 1 or i in shoot._started or i in shoot._open:
                continue
            if arc is not None and i != arc:
                continue
            w = v[sl.start + 1] - v[sl.start]
            if 0 < w < 0.05 * T and (best is None or w < best[1]):
                best = (i, w, sl)
        if best is None:
            return None
        i, w, sl = best
        target = factor * w
        p_saved = shoot.params.copy()

        def params(sv):
            le, off = point(sv)
            return np.asarray(self.regularized_params(self._eps_at(le), np.zeros(self.nc), off),
                              dtype=float)

        def resid(x, jac):
            shoot.params = params(x[-1])
            if jac:
                R, J = shoot.residual_jacobian(x[:-1])
            else:
                R, J = shoot.residual(x[:-1]), None
            row = x[sl.start + 1] - x[sl.start] - target
            return np.append(R, row), J

        if guess is not None:
            x = np.append(np.asarray(guess[0], dtype=float), float(guess[1]))
        else:
            x = np.append(np.asarray(v, dtype=float), frac)
        mid = 0.5 * (x[sl.start] + x[sl.start + 1])
        x[sl.start], x[sl.start + 1] = mid - 0.5 * target, mid + 0.5 * target
        try:
            for _ in range(iterations):
                R, J = resid(x, True)
                err = float(np.linalg.norm(R))
                if not np.isfinite(err):
                    return None
                if err < tol:
                    break
                h = 1e-7 * max(1.0, abs(x[-1]))
                xs = x.copy()
                xs[-1] += h
                Rs, _ = resid(xs, False)
                col = (Rs[:-1] - R[:-1]) / h
                n = len(x) - 1
                A = np.zeros((n + 1, n + 1))
                A[:n, :n] = J
                A[:n, n] = col
                A[n, sl.start + 1], A[n, sl.start] = 1.0, -1.0
                dx = np.linalg.lstsq(A, -R, rcond=None)[0]
                step = 1.0
                for _ in range(6):
                    try:
                        Rt, _ = resid(x + step * dx, False)
                        if np.linalg.norm(Rt) < (1 - 0.25 * step) * err:
                            break
                    except _FAILURES:
                        pass
                    step *= 0.5
                else:
                    return None
                x = x + step * dx
            else:
                return None
        except _FAILURES:
            return None
        finally:
            shoot.params = p_saved
        self.diagnostics.fired("R2")
        self._detail(f"    width step: arc {i} from {w:.3e} to {target:.3e}, route parameter "
                     f"{frac:.6f} -> {x[-1]:.6f}")
        return x[:-1], float(x[-1]), i, w, target

    def _order2_death(self, shoot0, v0, shoot1, v1):
        """Rule R50: ``(arc index, theta)`` for the order-2 arc of ``v0``'s structure whose
        width is positive at ``v0`` and has crossed zero at ``v1`` (the same structure, one
        step on), earliest first; ``theta`` is the secant's zero in the step (Lemma E'':
        the width is linear in the route parameter). None if no arc died."""
        if not (isinstance(shoot0, MultiArcShooting) and isinstance(shoot1, MultiArcShooting)):
            return None
        a0, val0 = self._arcs_of(shoot0, v0)
        a1, val1 = self._arcs_of(shoot1, v1)
        if a0 != a1:
            return None
        T = shoot1.unpack(v1)[3] - shoot1.t0
        best = None
        for i, ((kind, c), x0, x1) in enumerate(zip(a0, val0, val1)):
            if kind != "boundary" or self.kr.CONSTRAINT_ORDERS[c] < 2:
                continue
            if i in shoot0._open or i in shoot0._started:
                continue
            w0, w1 = float(x0[1] - x0[0]), float(x1[1] - x1[0])
            if not (w0 > 1e-6 * T and w1 < 1e-6 * T):
                continue
            th = w0 / (w0 - w1)
            if 0.0 < th < 1.0 and (best is None or th < best[1]):
                best = (i, th)
        return best

    def _death_points(self, shoot, v1, v0, s1, s0, s_fail):
        """Rule R17: ``[(arc index, s*)]`` for finite order-1 arcs that shrink from ``v1``
        (at ``s1``) to ``v0`` (at ``s0``) and, by a secant in ``w^2``, die before
        ``s_fail``, the step that failed; earliest death first.

        By Lemma E an arc's width obeys ``s - s* = kappa w^2 + O(w^3)`` near its birth
        or death, with the branch smooth in ``w``: so ``w^2`` is C^1 in ``s`` with a
        nonzero slope, and the secant through two points of the path finds ``s*`` to
        ``O(ds^2)``. Past ``s*`` the arc is gone; the step without it is solved and
        judged as any other."""
        if not isinstance(shoot, MultiArcShooting) or not s0 > s1:
            return []
        a0, val0 = self._arcs_of(shoot, v0)
        a1, val1 = self._arcs_of(shoot, v1)
        if a0 != a1:
            return []
        T = shoot.unpack(v0)[3] - shoot.t0
        out = []
        for i, ((kind, c), x0, x1) in enumerate(zip(a0, val0, val1)):
            if kind != "boundary" or self.kr.CONSTRAINT_ORDERS[c] != 1 or len(x0) < 2:
                continue
            if i in shoot._open or i in shoot._started:
                continue                   # an arc run out to an end has no death
            w0, w1 = float(x0[1] - x0[0]), float(x1[1] - x1[0])
            if not (w1 > w0 > 0 and w0 < 0.05 * T):
                continue
            s_star = s0 + (s0 - s1) * w0 ** 2 / (w1 ** 2 - w0 ** 2)
            if s0 < s_star < s_fail:
                out.append((i, s_star))
        return sorted(out, key=lambda e: e[1])

    def _width_moves(self, shoot, v, frac, point, tol):
        """Rule R2's moves at a stall, from Lemma E's normal form ``s = s* + kappa w^2``.

        A step to half the young arc's width tells which side of the event the route is
        on. If ``s`` moves forward the arc is dying: ``w^2`` is linear in ``s``, so the
        death is at ``s* = s_half + (s_half - s) / 3``; the moves are the structure
        without the arc just past ``s*``, then the half step. If ``s`` moves back the
        arc is being born: ``kappa`` and ``s*`` from the two points predict ``s`` at
        ``1.5 w``, and ``v`` is predicted linearly in ``w`` (the branch is smooth in
        ``w``). Yields ``(v, s, dropped arc or None)``, best first."""
        half = self._width_step(shoot, v, frac, point, 0.5, tol)
        if half is None:
            return
        v_h, s_h, i, w0, w_h = half
        if s_h > frac:
            s_star = s_h + (s_h - frac) * w_h ** 2 / (w0 ** 2 - w_h ** 2)
            beyond = s_star + 0.25 * (s_star - frac)
            if s_star < 1.0:
                self.diagnostics.fired("R2 death")
                self._detail(f"    width steps: arc {i} dies at s = {s_star:.6f}; past it without")
                yield v, min(beyond, 1.0), i
            yield v_h, s_h, None
            return
        kappa = (frac - s_h) / (w0 ** 2 - w_h ** 2)
        if not kappa > 0:
            return
        s_star = frac - kappa * w0 ** 2
        w_t = 1.5 * w0
        s_t = s_star + kappa * w_t ** 2
        v_t = v + (v - v_h) * (w_t - w0) / (w0 - w_h)
        grown = self._width_step(shoot, v, frac, point, 1.5, tol, guess=(v_t, s_t), arc=i)
        if grown is not None and frac < grown[1] <= 1.0:
            self.diagnostics.fired("R2 birth")
            yield grown[0], grown[1], None

    def _split_rising_order2(self, arcs, values, i, t_lo, t_hi, mu_lo, mu_hi, at_entry,
                             at_exit, t_on, mu_on, T):
        """Rule R36: R5's split of an order-2 arc where its multiplier rises (Lemma H2).

        The entry jump of an order-2 arc is ``(pi_1, pi_2)`` with ``pi_2 = mu(t_in+)`` and
        ``pi_1 = nu_in - mu'(t_in+)``. The first piece keeps the arc's jump; the second
        enters with ``(-mu'(t), mu(t))`` at its entry ``t``, the jump that continues the
        arc's costate there (no atom), which Newton then corrects. A rise from the entry
        (to the exit) trims the arc instead, as R18; the gaps are the plain ones, a little
        wider and narrower than the rise, and R34's, wider by half and twice."""
        a, b, pi = values[i]
        self.diagnostics.fired("R36")
        self._detail(f"    {self.kr.CONSTRAINT_NAMES[arcs[i][1]]}: multiplier rises on "
                     f"[{t_lo:.5f}, {t_hi:.5f}] (Lemma H2): split the order-2 arc")
        # The trajectory samples each junction from both sides, so a time can repeat:
        # mu' is taken on the distinct times (the last value at each), never across a
        # zero step, which would make the jump seed NaN.
        t_u, last = np.unique(np.asarray(t_on, float)[::-1], return_index=True)
        mu_u = np.asarray(mu_on, float)[::-1][last]
        dmu = np.gradient(mu_u, t_u) if len(t_u) > 1 else np.zeros_like(mu_u)
        pi_first = np.atleast_1d(np.asarray(pi, float)).copy()

        def entering(t):
            return np.array([-float(np.interp(t, t_u, dmu)), float(np.interp(t, t_u, mu_u))])

        out_a = arcs[:i] + [arcs[i], arcs[i]] + arcs[i + 1:]
        out = []
        w = t_hi - t_lo
        # A rise that starts (ends) within a tenth of its stretch of the entry (exit) is
        # at it: the piece it would leave is no arc. The rise is often bounded by another
        # limit's junction, so the new junction is offered a little past it, then half
        # and twice the stretch past (as R34), never on it (Lemma D).
        piece = 1e-4 * T
        at_entry = at_entry or t_lo - a < max(0.1 * w, piece)
        at_exit = at_exit or b - t_hi < max(0.1 * w, piece)
        for f in (0.02, 0.5, 2.0):
            if at_entry and t_hi + f * w < b - piece:
                t_e = t_hi + f * w
                out.append((arcs, values[:i] + [(t_e, b, entering(t_e))] + values[i + 1:]))
            if at_exit and t_lo - f * w > a + piece:
                out.append((arcs, values[:i] + [(a, t_lo - f * w, pi_first.copy())]
                            + values[i + 1:]))
        if not (at_entry or at_exit):
            for f in (0.02, -0.02, 0.5, 2.0):
                lo, hi = t_lo - f * w, t_hi + f * w
                if lo > a + piece and hi < b - piece and hi - lo > 1e-6 * T:
                    mu_hi_ = float(np.interp(hi, t_on, mu_on))
                    for p2 in (entering(hi), np.array([0.0, mu_hi_]),
                               np.array([float(pi_first[0]), mu_hi_])):
                        out.append((out_a, values[:i] + [(a, lo, pi_first.copy()), (hi, b, p2)]
                                    + values[i + 1:]))
            # The arc may not come back after the rise (or not have been there before
            # it): the piece before the rise alone, then the piece after it alone.
            for f in (0.02, 0.5, 2.0):
                if t_lo - f * w > a + piece:
                    out.append((arcs, values[:i] + [(a, t_lo - f * w, pi_first.copy())]
                                + values[i + 1:]))
            for f in (0.02, 0.5, 2.0):
                if t_hi + f * w < b - piece:
                    t_e = t_hi + f * w
                    out.append((arcs, values[:i] + [(t_e, b, entering(t_e))] + values[i + 1:]))
        return out or None

    def _split_rising(self, shoot, v, n_points=801):
        """Structures with an order-1 boundary arc split where its multiplier rises.

        On an order-1 arc ``mu(t) = int_t^{t_out} d eta`` with ``eta >= 0`` (Lemma H of
        ``docs/notes/structure_certificates.md``, which holds for each limit separately
        while others are active too). Where ``mu`` rises, ``eta < 0``: the limit must
        leave there. The arc becomes two, with the gap over the stretch of the largest
        rise (running minimum to maximum); the second arc's entry jump is ``mu`` where it
        starts. A list of ``(arcs, values)``, or ``None`` if no arc rises."""
        if not isinstance(shoot, MultiArcShooting):
            return None
        k, p = shoot.kernels, shoot.params
        arcs, values = self._arcs_of(shoot, v)
        tr = shoot.trajectory(v, n_points)
        T = shoot.unpack(v)[3] - shoot.t0
        best = None
        best_abs = None
        orders = (1, 2) if getattr(self, "order2_rise", False) else (1,)
        for i, ((kind, c), val) in enumerate(zip(arcs, values)):
            if kind != "boundary" or self.kr.CONSTRAINT_ORDERS[c] not in orders:
                continue
            on = np.flatnonzero(tr["active"][:, c] & (tr["t"] >= val[0]) & (tr["t"] <= val[1]))
            if len(on) < 3:
                continue
            mu = []
            for j in on:
                aid = tr["arc"][j]
                m_ = k.boundary_multiplier(aid, tr["t"][j], tr["z"][j], p)
                mu.append(float(np.atleast_1d(m_)[0]) if aid < k.N_CONSTRAINTS
                          else float(np.atleast_1d(m_)[list(k.arc_constraints(aid)).index(c)]))
            mu = np.asarray(mu)
            run_min = np.minimum.accumulate(mu)
            rise = mu - run_min
            j_hi = int(np.argmax(rise))
            if rise[j_hi] <= 1e-6 * max(1.0, float(np.abs(mu).max())):
                continue
            j_lo = int(np.flatnonzero(mu[:j_hi + 1] == run_min[j_hi])[-1])
            t_lo, t_hi = float(tr["t"][on[j_lo]]), float(tr["t"][on[j_hi]])
            score = rise[j_hi]
            if getattr(self, "relative_rise", False):
                score = rise[j_hi] / max(float(np.abs(mu).max()), 1e-300)
            if t_hi - t_lo < 1e-6 * T:
                continue
            if best_abs is None or rise[j_hi] > best_abs[0]:
                best_abs = (rise[j_hi], i)
            if best is not None and score <= best[0]:
                continue
            best = (score, i, t_lo, t_hi, mu[j_lo], mu[j_hi], j_lo == 0, j_hi == len(on) - 1,
                    tr["t"][on], mu, on)
        if best is None:
            return None
        _, i, t_lo, t_hi, mu_lo, mu_hi, at_entry, at_exit, t_on, mu_on, on_idx = best
        if best_abs[1] != i:
            # Rule R33 chose another arc than the largest absolute rise.
            self.diagnostics.fired("R33")
        self._split_limit = arcs[i][1]
        a, b, pi = values[i]
        if self.kr.CONSTRAINT_ORDERS[arcs[i][1]] == 2:
            return self._split_rising_order2(arcs, values, i, t_lo, t_hi, mu_lo, mu_hi,
                                             at_entry, at_exit, t_on, mu_on, T)
        self.diagnostics.fired("R5")
        self._detail(f"    {self.kr.CONSTRAINT_NAMES[arcs[i][1]]}: multiplier rises on "
                     f"[{t_lo:.5f}, {t_hi:.5f}] (eta < 0): split the arc")
        out_a = arcs[:i] + [arcs[i], arcs[i]] + arcs[i + 1:]
        # The first piece must leave with mu = 0: mu_new = mu - mu(t_lo) on it, so its
        # entry jump falls by mu(t_lo). The second enters with mu(t_hi).
        pi1 = np.atleast_1d(np.asarray(pi, float)).copy()
        pi1[0] -= mu_lo
        pi2 = np.atleast_1d(np.asarray(mu_hi, float))
        # The rise is often exactly another limit's arc (its junctions bound it): a gap
        # on those junctions would make events coincide (Lemma D), so the gap is
        # offered a little wider, then a little narrower.
        d = 0.02 * (t_hi - t_lo)
        out = []
        if self.trim_rising and (at_entry or at_exit):
            # Rule R18: mu on an arc is at most its entry value (Lemma H), so a rise that
            # starts at the entry (ends at the exit) is a stretch the arc must not cover;
            # the split would leave a piece of zero width there, and the trimmed arc is
            # the structure it tends to. It is offered first.
            self.diagnostics.fired("R18")
            lo_t, hi_t = (t_hi, b) if at_entry else (a, t_lo)
            pi_t = pi2 if at_entry else pi1
            if hi_t - lo_t > 1e-6 * T:
                out.append((arcs, values[:i] + [(lo_t, hi_t, pi_t)] + values[i + 1:]))
        if self.exit_before_saturation:
            t_s = self._own_saturation(shoot, tr, on_idx, arcs[i][1], t_lo)
            if t_s is not None and t_s - a > 1e-6 * T:
                # Rule R29 (Lemma U): the rise starts where the limit's own control joins
                # the held set, so the piece leaves just before that moment: a gap from
                # there to the rise's end, or, for a rise to the exit, a trim.
                self.diagnostics.fired("R29")
                t_e = t_s - 0.02 * (t_s - a)
                p1 = np.atleast_1d(np.asarray(pi, float)).copy()
                p1[0] -= float(np.interp(t_e, t_on, mu_on))
                if at_exit or not t_hi + d < b:
                    out.insert(0, (arcs, values[:i] + [(a, t_e, p1)] + values[i + 1:]))
                else:
                    out.insert(0, (out_a, values[:i] + [(a, t_e, p1), (t_hi + d, b, pi2)]
                                   + values[i + 1:]))
        if self.split_normal_form:
            c_split = arcs[i][1]
            gap = self._split_gap(t_on, mu_on, self._split_weights(shoot, tr, on_idx, c_split))
            if gap is not None and a < gap[0] < gap[1] < b:
                # Rule R27: the gap of Lemma E3's moment conditions, offered first.
                self.diagnostics.fired("R27")
                lo, hi, level = gap
                p1 = np.atleast_1d(np.asarray(pi, float)).copy()
                p1[0] -= level
                p2 = np.atleast_1d(level)
                out.append((out_a, values[:i] + [(a, lo, p1), (hi, b, p2)] + values[i + 1:]))
        for lo, hi in ((t_lo - d, t_hi + d), (t_lo + d, t_hi - d)):
            if not (lo > a and hi < b):
                continue
            out.append((out_a, values[:i] + [(a, lo, pi1), (hi, b, pi2)] + values[i + 1:]))
        if getattr(self, "wide_gaps", False):
            # Rule R34: the same split seeded by continuity (the first piece keeps the
            # arc's entry jump, so the seed is the current solution up to the gap),
            # with the gap wider than the rise.
            w = t_hi - t_lo
            for f in (0.5, 2.0):
                lo, hi = t_lo - f * w, t_hi + f * w
                if lo > a and hi < b:
                    self.diagnostics.fired("R34")
                    out.append((out_a, values[:i] + [(a, lo, np.atleast_1d(np.asarray(pi, float)).copy()),
                                                     (hi, b, pi2)] + values[i + 1:]))
        return out or None

    def _own_saturation(self, shoot, tr, idx, c, t_rise, samples=3):
        """Lemma U: the time near ``t_rise`` (within ``samples`` samples of ``tr``) at
        which the held set on the arc changes in a control that moves limit ``c``'s
        rate (``a_k != 0``), joining or leaving, or None. ``idx`` are the arc's samples
        in ``tr``. (Lemma U holds at any such change after which the density is
        negative, not only at the limit's own control joining.)"""
        from .lemmas import LimitObjects
        try:
            obj = LimitObjects(shoot.kernels, shoot.params)
        except ValueError:
            return None
        k, p = shoot.kernels, shoot.params
        idx = np.asarray(idx)
        j = int(np.argmin(np.abs(tr["t"][idx] - t_rise)))
        a_vec = np.abs(np.asarray(obj.rate_parts(c, tr["t"][idx[j]], tr["z"][idx[j]])[0], float))
        moving = [kk for kk in range(len(a_vec)) if a_vec[kk] > 1e-6 * max(a_vec.max(), 1e-300)]

        def held(q):
            lo, hi = (np.asarray(x, float) for x in k.control_bounds(tr["t"][q], tr["z"][q], p))
            u = np.asarray(tr["u"][q], float)
            return tuple(kk for kk in moving
                         if min(u[kk] - lo[kk], hi[kk] - u[kk]) <= 1e-6 * (hi[kk] - lo[kk]))

        for q in range(max(j - samples, 1), min(j + samples + 1, len(idx))):
            if held(idx[q]) != held(idx[q - 1]):
                return float(tr["t"][idx[q]])
        return None

    @staticmethod
    def _order_crossing(shoot, v):
        """Rule R30: how far ``v``'s junctions have crossed, as a fraction of T: the
        largest step back between consecutive event times (the order monitor of V1,
        negated); 0 when they are in order."""
        if not isinstance(shoot, MultiArcShooting) or not shoot.arcs:
            return 0.0
        times = [shoot.t0, *shoot.event_times(v), shoot.unpack(v)[3]]
        back = max(a - b for a, b in zip(times, times[1:]))
        return max(back, 0.0) / (times[-1] - times[0])

    def _split_resolved(self, shoot, v):
        """Rule R28: whether a split (or trim) solution has the density monitor of the
        limit it split back to zero: no rise of its multiplier beyond the final
        check's tolerance (V1), and no arc of it of zero width."""
        c = getattr(self, "_split_limit", None)
        if c is None or not isinstance(shoot, MultiArcShooting):
            return True
        try:
            dens = self._densities(shoot, v)
        except _FAILURES:
            return False
        T = shoot.unpack(v)[3] - shoot.t0
        arcs, values = self._arcs_of(shoot, v)
        spans = sorted((val[0], val[1]) for (kind, cc), val in zip(arcs, values)
                       if kind == "boundary" and cc == c)
        if any(b - a <= 1e-6 * T for a, b in spans) or any(
                n[0] - p[1] <= 1e-6 * T for p, n in zip(spans, spans[1:])):
            return False
        return all(rise <= 1e-6 * max(1.0, mu_max) for cc, _, _, rise, mu_max in dens if cc == c)

    def _split_weights(self, shoot, tr, idx, c):
        """``D = 1 / [(A_F R_F^{-1} A_F^T)^{-1}]_{cc}`` at the samples ``idx`` of
        ``tr``: how fast limit ``c`` moves per unit of its released multiplier, with
        the other limits ridden there still held and the controls at their bounds
        fixed (Lemma P's ``a_F^T R_F^{-1} a_F`` for one limit). Ones when the
        problem lacks the certificates (C1, C2) that make it computable."""
        from .lemmas import LimitObjects
        try:
            obj = LimitObjects(shoot.kernels, shoot.params)
        except ValueError:
            return np.ones(len(idx))
        k, p = shoot.kernels, shoot.params
        out = np.ones(len(idx))
        for n, j in enumerate(idx):
            t, z, u = tr["t"][j], tr["z"][j], np.asarray(tr["u"][j], float)
            lo, hi = (np.asarray(b, float) for b in k.control_bounds(t, z, p))
            F = np.flatnonzero(np.minimum(u - lo, hi - u) > 1e-6 * (hi - lo))
            rows = [c] + [cc for cc in np.flatnonzero(tr["active"][j]) if cc != c]
            try:
                A = np.array([obj.rate_parts(cc, t, z)[0] for cc in rows])[:, F]
                M = A @ np.linalg.solve(obj.R[np.ix_(F, F)], A.T)
                out[n] = 1.0 / np.linalg.inv(M)[0, 0]
            except (np.linalg.LinAlgError, ValueError, IndexError):
                out[n] = np.nan
        if not np.all(np.isfinite(out) & (out > 0)):
            return np.ones(len(idx))
        return out

    @staticmethod
    def _split_gap(t, mu, D=None):
        """Rule R27: the free gap of a split (Lemma E3), ``(a, b, level)`` or None.

        ``mu`` is the arc's multiplier at the times ``t``. Released on ``[a, b]``,
        the limit moves at ``dg/dt = D (mu - mu(a))`` to leading order, and it must
        return to its level tangentially. So the gap's ends satisfy:
        - ``mu(a) = mu(b)``, the level;
        - ``int_a^b D (mu - mu(a)) = 0``.
        The gap straddles the stretch where ``mu`` rises (``eta < 0``). The level is
        found by bisection between ``mu``'s values at the stretch's ends.
        - For a quadratic dip of ``eta`` and constant ``D``, the gap is centred on
          the minimum, with ``h^2 = -6 eta_min / eta''``.
        - Where ``eta`` is negative on one window of a held set (Lemma P), the gap
          is that window, slightly widened."""
        t, mu = np.asarray(t, float), np.asarray(mu, float)
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, mu = t[keep], mu[keep]
        D = np.ones_like(t) if D is None else np.asarray(D, float)[keep]
        if len(t) < 5:
            return None
        eta = -np.gradient(mu, t)
        j = int(np.argmin(eta))
        if not eta[j] < 0:
            return None
        r1, r2 = j, j
        while r1 > 0 and eta[r1 - 1] < 0:
            r1 -= 1
        while r2 < len(t) - 1 and eta[r2 + 1] < 0:
            r2 += 1

        def ends(level):
            a = r1
            while a > 0 and mu[a] < level:
                a -= 1
            b = r2
            while b < len(t) - 1 and mu[b] > level:
                b += 1
            if mu[a] < level or mu[b] > level or a == 0 or b == len(t) - 1:
                return None        # the gap would reach the arc's end: R18's trim
            ta = float(np.interp(level, [mu[a + 1], mu[a]], [t[a + 1], t[a]]))
            tb = float(np.interp(level, [mu[b], mu[b - 1]], [t[b], t[b - 1]]))
            return a, b, ta, tb

        def moment(level):
            e = ends(level)
            if e is None:
                return None
            a, b, ta, tb = e
            tt = np.concatenate([[ta], t[a + 1:b], [tb]])
            ff = np.concatenate([[0.0], D[a + 1:b] * (mu[a + 1:b] - level), [0.0]])
            return float(np.sum(0.5 * (ff[1:] + ff[:-1]) * np.diff(tt)))

        lo, hi = mu[r1], mu[r2]
        span = hi - lo
        if not span > 0:
            return None
        lo, hi = lo + 1e-9 * span, hi - 1e-9 * span
        f_lo, f_hi = moment(lo), moment(hi)
        if f_lo is None or f_hi is None or not f_lo > 0 > f_hi:
            return None
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f = moment(mid)
            if f is None:
                return None
            lo, hi = (mid, hi) if f > 0 else (lo, mid)
        level = 0.5 * (lo + hi)
        _, _, ta, tb = ends(level)
        return ta, tb, float(level)

    def _exact_jump(self, arcs, v, tol, reorders=0):
        """The exact problem's solve of structure ``arcs`` from ``v``: ``(J, shoot, sol)`` or None."""
        kx, px = self.exact_kernels, self.exact_params()
        tol_in, tol = tol, _scaled(tol, v)
        # A large problem's search has run on tangential junctions; if a control really does
        # jump at one, the rows that allow it are the ones to try next.
        for tangent in ((True, False) if kx.system.riding is not None else (False,)):
            try:
                if arcs:
                    shoot = self._multi_arc(kx, px, arcs, max_nfev=self.attempt_nfev, open_end=True,
                                             **self.options)
                    method = "lm" if kx.N_SWITCHING else "hybr"
                else:
                    shoot = SingleShooting(kx, px, **self.options)
                    method = "hybr"
                shoot.tangent_junctions = tangent
                if arcs and (shoot._open or shoot._started):
                    sol = shoot.solve_damped(v, tol=tol, max_seconds=60.0)
                else:
                    # A good path solution is one or two Newton steps away; MINPACK can
                    # throw it far off, and takes many more evaluations to get there.
                    sol = shoot.solve_newton(v, tol=tol) if self.exact_newton else None
                    if sol is None or not sol.success:
                        sol = shoot.solve(v, method=method, tol=tol,
                                          options=_budget(method, 4 * self.attempt_evaluations))
                    if not sol.success and method == "hybr" and kx.system.riding is not None:
                        other = shoot.solve_chord(v, tol=tol)
                        sol = other if other.residual_norm < sol.residual_norm else sol
                if not sol.success and sol.residual_norm < SEED_ACCEPT * tol:
                    polished = shoot.solve_damped(sol.v, tol=tol, max_steps=10, max_seconds=30.0)
                    if polished.residual_norm < sol.residual_norm:
                        sol = polished
                    if not sol.success:
                        sol = replace(sol, success=True)   # stalled at the integrator's rounding
                if sol.success:
                    sol = replace(sol, max_violation=shoot._max_violation(sol.v))
                if self._accept(shoot, sol, tol):
                    return self.objective(shoot, sol.v), shoot, sol
                self._detail(f"  exact solve: residual {sol.residual_norm:.1e} (max |v| "
                             f"{np.abs(sol.v).max():.1e}), "
                             f"max violation {sol.max_violation:.1e}; {self._rejection(shoot, sol)}")
                if (getattr(self, "exact_reorders", False) and arcs and reorders < 2
                        and sol.success and isinstance(shoot, MultiArcShooting)
                        and not shoot.in_order(sol.v, self._coincide())):
                    # Rule R41: junctions of the exact solution crossed. Two events of
                    # different limits pass each other transversally (E5), so the
                    # structure with the events in the order of their times is the one
                    # beyond the crossing; it is solved from this solution and judged as
                    # any other.
                    arcs_x, vals_x = self._arcs_of(shoot, sol.v)
                    entries_x, ev_x = self._mas_entries(arcs_x, vals_x)
                    if list(entries_x) != list(arcs):
                        self.diagnostics.fired("R41")
                        self._detail("  exact solve: junctions crossed; the events in the "
                                     "order of their times")
                        seed = self._multi_arc(kx, px, entries_x, max_nfev=self.attempt_nfev,
                                               open_end=True, **self.options).pack_arcs(
                                                   sol.v[:shoot.n_base], ev_x)
                        hit = self._exact_jump(entries_x, seed, tol_in, reorders + 1)
                        if hit is not None:
                            return hit
            except _FAILURES as exc:
                self._detail(f"  exact solve failed: {exc}")
        return None

    def _rejection(self, shoot, sol):
        """Why ``_accept`` turned a converged solution down (for the log)."""
        if not sol.success or not isinstance(shoot, MultiArcShooting) or not shoot.arcs:
            return "not converged" if not sol.success else "rejected"
        try:
            d = shoot.diagnostics(sol.v)
        except _FAILURES as exc:
            return f"diagnostics failed: {exc}"
        names = shoot.kernels.CONSTRAINT_NAMES
        violated = {names[c]: round(float(g), 6) for c, g in enumerate(d["free_arc_violation"])
                    if g > 1e-8}
        return (f"violated off their arcs {violated}, controls past their bounds on arcs "
                f"{np.round(d['boundary_bound_violation'], 6).tolist()}, touch nu "
                f"{np.round(d['touch_nu'], 6).tolist()}, mu rises "
                f"{[float(f'{x:.3g}') for x in d.get('boundary_mu_rise', [])]}, mu ranges "
                f"{[(round(a, 6), round(b, 6)) for a, b in d['boundary_mu']]}")

    def _active_set(self, sol, eps0, s_max, steps, tol, max_solves=1500):
        """The active-set route: from the unconstrained regularized solution ``sol``
        at ``eps0``, tighten every limit from just above the solution to its
        true value, bringing in arcs as limits come into play, then lower
        ``eps`` with the structure following, trying the exact problem at each
        step. Returns ``(J, shoot, sol)`` or None.

        Each limit has its own offset, so the offsets form a vector annealed to
        zero. They first fall together; when that stalls, typically because
        several limits come into play at once, each moves on alone, which
        separates the events. When none can move, the route returns to the
        last change of structure and rules out the structure chosen there."""
        deadline = time.perf_counter() + self.active_set_seconds
        banned, checkpoints, ended = set(), [], set()
        tried_splits = {}                  # split structure -> times tried (at most 3)

        def key(arcs, values):
            return tuple(self._mas_entries(arcs, values, True)[0])

        le_start = np.log10(eps0) if self.bang_bang else 0.0
        le_target = None
        if self._weight_route():
            le_start = le_target = np.log10(self.weight_value)
            factor = float(self.weight_start) if self.weight_start else 1.0
            if factor != 1.0:
                # Rule R24: the unconstrained solution at the larger weight, by Newton
                # along log w from the one at its value (no arcs: a smooth path).
                moved = self._weight_moved(sol, le_target, le_target + np.log10(factor))
                if moved is not None:
                    self.diagnostics.fired("R24")
                    sol, le_start = moved
        shoot = self._smoothed([], self._eps_at(le_start) if not self.bang_bang else eps0,
                               np.zeros(self.nc))
        (_, g, _), = self._free_samples(shoot, sol.v)
        # Violations are measured against how much each limit's g moves along the
        # unconstrained solution, or, when it hardly moves there (a joint the motion
        # barely uses), how far that solution is from the limit.
        scale = np.maximum(np.maximum(g.max(axis=0) - g.min(axis=0), np.abs(g.max(axis=0))), 1e-12)
        off0 = np.maximum(g.max(axis=0), 0.0) + 1e-3 * scale
        if self.offset_start == "staggered":
            # Rule R1 (docs/notes/structure_certificates.md, Lemmas C and D): offsets that
            # all start just above their peaks bring every limit in at once; a random head
            # start per limit separates the births, with probability one.
            peaks = np.maximum(g.max(axis=0), 0.0)
            draw = np.random.default_rng(self.offset_seed).uniform(0.5, 1.5, self.nc)
            off0 = peaks * (1.0 + draw) + 1e-3 * scale
        peaks0 = g.max(axis=0)
        state = {"shoot": shoot, "v": sol.v, "le": le_start, "off": off0.copy(), "solves": 0}
        log_eps_min = state["le"] - s_max
        state["best_tight"], state["progress_at"] = np.inf, time.perf_counter()

        def stalled():
            """Rule R15: no new low of the offsets for ``stall_seconds`` (with a portfolio,
            the next route gets the time that is left)."""
            if not self.portfolio or state["best_tight"] <= 1e-12:
                return False               # one route, or offsets at zero: the last phase
            return time.perf_counter() - state["progress_at"] > self.stall_seconds

        # The path only has to be followed closely, not solved to the last digit:
        # at an order-1 entry with a smooth control H continuity has a double
        # root, and Newton converges only linearly there. The exact jumps use ``tol``.
        path_tol = max(tol, 1e-6)

        def solve_structure(arcs, values, base, le, off):
            return self._solve_structure(arcs, values, base, le, off, path_tol, state)

        def settle(sh, r, le, off, changed=False):
            """Bring in violated limits and retire dead arcs until the solution at
            (le, off) is consistent; ``(shoot, sol)`` or None. ``changed``: the
            structure was just changed, so a large violation is that change's doing."""
            # hook: settle (pipeline.py): the step's monitors, each passing, rejecting the
            # step with a reason, accepting it, or proposing structures to repair it.
            ctx = SettleContext(self, state, sh, r, le, off, changed, scale, tried_splits, key)
            verdict = self.pipelines[SETTLE_START].first(ctx)
            if isinstance(verdict, Reject):
                state["why"] = verdict.why
                return None
            for _ in range(2 * self.nc + 4):
                ctx.sh, ctx.r, ctx.changed = sh, r, changed
                verdict = self.pipelines[SETTLE].first(ctx)
                if isinstance(verdict, Reject):
                    state["why"] = verdict.why
                    return None
                if isinstance(verdict, Accept):
                    return sh, r
                options, optional = verdict.options, verdict.optional
                base = r.v[:sh.n_base] if isinstance(sh, MultiArcShooting) else r.v
                options = [o for o in options if key(*o) not in banned]
                if self.parallel > 1 and options and not optional:
                    hit = self._speculate(options, base, le, off, solve_structure, state)
                    if hit is None:
                        state["why"] = "no structure converged"
                        return None
                    (arcs, values), nsh, nr = hit
                    self._detail(f"    try in parallel: {self._mas_entries(arcs, values, True)[0]}")
                    sh, r = nsh, nr
                    changed = True
                    self._say(f"  structure: {self._names(arcs)}")
                    continue
                for arcs, values in options:
                    if key(arcs, values) in banned:
                        continue
                    try:
                        nsh, nr = solve_structure(arcs, values, base, le, off)
                    except _FAILURES as exc:
                        self._detail(f"    try {self._mas_entries(arcs, values, True)[0]}: {exc}")
                        continue
                    self._detail(f"    try {self._mas_entries(arcs, values, True)[0]}: "
                                 f"residual {nr.residual_norm:.1e}")
                    if nr.success and optional and self.split_must_resolve \
                            and not self._split_resolved(nsh, nr.v):
                        # Rule R28: a split that leaves the rise (or closes its gap) has
                        # not continued the event (V4); the next option is tried.
                        self.diagnostics.fired("R28")
                        self._detail("    the split leaves the multiplier rising: not taken")
                        continue
                    if nr.success:
                        break
                else:
                    if optional:
                        return sh, r
                    state["why"] = "no structure converged"
                    return None
                sh, r = nsh, nr
                changed = True
                self._say(f"  structure: {self._names(arcs)}")
            state["why"] = "structure changes did not settle"
            return None

        def advance(target_le, target_off, on_step=None, ds_min=1e-6, turn_to=None,
                    backward=False):
            """Towards the target along a straight line; False, at the last solution
            reached, if the step falls below ``ds_min``. Rule R44 turns at a fold towards
            ``turn_to`` (the line's other end; the start by default), following the
            solution curve by arclength; ``backward``: this run goes against the
            route, so reaching its end means the curve does not lead on (False)."""
            le0, off_start = state["le"], state["off"].copy()
            frac, ds = 0.0, 0.25
            prev = None                    # (frac, v, structure) of the solution before
            prev2 = None                   # Rule R48: the solution before prev
            memory = {}                    # the leg's own state for path rules (R46)
            leg_dir = (target_le - le0, np.asarray(target_off, dtype=float) - off_start)
            state["leg_dir"] = leg_dir
            while frac < 1.0:
                if (state["solves"] > max_solves or ds < ds_min
                        or time.perf_counter() > deadline or stalled()):
                    self.diagnostics.step("stalled" if ds < ds_min else "out of budget")
                    return False
                f = min(frac + ds, 1.0)
                le = le0 + f * (target_le - le0)
                off = off_start + f * (target_off - off_start)
                sh0, v0 = state["shoot"], state["v"]
                arcs, values = self._arcs_of(sh0, v0)
                ctx = StepContext(self, frac, f, ds, ds_min, v0, arcs, prev, prev2,
                                  moves_weight=target_le != le0, memory=memory)
                guesses = self.pipelines[PREDICT].collect(ctx)    # hook: predict
                done = None
                outcome, attempt = None, None
                # Below the floor, and within three halvings of the leg's ds_min, a
                # step is solved as without the rule (the fallback solver included).
                floor = max(self.contraction_floor, 8 * ds_min) if getattr(
                    self, "contraction_steps", False) else np.inf
                steered = f - frac > floor
                theta_far, theta_done = None, None
                for gi, g in enumerate(guesses):
                    order = 2 if len(guesses) > 1 and gi == 0 else 1
                    base = g[:sh0.n_base] if isinstance(sh0, MultiArcShooting) else g
                    vals = self._arcs_of(sh0, g)[1] if arcs else values
                    attempt = (arcs, vals, base)
                    before = dict(self.diagnostics.evaluations)
                    state["why"] = None
                    try:
                        self._reject_theta = self.contraction_reject if steered else None
                        try:
                            sh, r = solve_structure(arcs, vals, base, le, off)
                        finally:
                            self._reject_theta = None
                        theta = getattr(r, "contraction", None)
                        if getattr(r, "too_far", False):
                            theta_far = (theta, order)
                            self.diagnostics.fired("R37")
                            self._detail(f"  step to {f:.6f} (ds {ds:.1e}): too long for "
                                         f"Newton (theta {theta:.2g})")
                            outcome = "too far (R37)"
                            break          # the shorter step takes the predictor again
                        self._detail(f"  step to {f:.6f} (ds {ds:.1e}): residual "
                                     f"{r.residual_norm:.1e}; junctions "
                                     f"{self._junction_text(sh, r.v)}")
                        death = (self._order2_death(sh0, v0, sh, r.v)
                                 if r.success and getattr(self, "order2_deaths", False)
                                 and arcs else None)
                        if death is not None:
                            # Rule R50: an order-2 arc's width crossed zero inside the step.
                            # It is linear in s (Lemma E''), so the death is at the secant's
                            # zero; there the arc is the touch with jump pi_1 (Lemma Z).
                            # The touch is solved just past the death, from the solution
                            # interpolated to it.
                            i_d, th_d = death
                            fd = frac + th_d * (f - frac)
                            fd = min(fd + 0.25 * (fd - frac), f)
                            le_d = le0 + fd * (target_le - le0)
                            off_d = off_start + fd * (target_off - off_start)
                            v_d = np.asarray(v0) + th_d * (np.asarray(r.v) - np.asarray(v0))
                            arcs_d, vals_d = self._arcs_of(sh, v_d)
                            val_i = vals_d[i_d]
                            arcs_d = list(arcs_d)
                            vals_d = list(vals_d)
                            arcs_d[i_d] = ("touch", arcs_d[i_d][1])
                            vals_d[i_d] = (0.5 * (val_i[0] + val_i[1]),
                                           float(np.atleast_1d(val_i[2])[0]))
                            try:
                                sh_d, r_d = solve_structure(arcs_d, vals_d,
                                                            v_d[:sh0.n_base], le_d, off_d)
                                done_d = settle(sh_d, r_d, le_d, off_d) if r_d.success else None
                            except _FAILURES:
                                done_d = None
                            self._detail(f"    arc {i_d} dies at {frac + th_d * (f - frac):.6f}"
                                         f" (R50): touch at {fd:.6f} "
                                         + ("solved" if done_d is not None else "not solved"))
                            if done_d is not None:
                                self.diagnostics.fired("R50")
                                done, f, le, off = done_d, fd, le_d, off_d
                                ds = f - frac
                                outcome = "converged"
                                theta_done = (theta, order)
                                break
                        done = settle(sh, r, le, off) if r.success else None
                        if done is not None:
                            theta_done = (theta, order)
                        if r.success:
                            outcome = ("converged" if done is not None
                                       else f"settle failed: {state.get('why') or 'other'}")
                        elif r.residual_norm >= 1e5:
                            new = {k_: n_ - before.get(k_, 0)
                                   for k_, n_ in self.diagnostics.evaluations.items()}
                            worst = max(new, key=new.get) if any(new.values()) else "other"
                            outcome = f"seed failed: {worst}"
                        else:
                            outcome = "not converged"
                    except _FAILURES as exc:
                        self._detail(f"  step to {f:.6f} (ds {ds:.1e}): {exc}")
                        done = None
                        outcome = f"raised: {getattr(exc, 'failure', type(exc).__name__)}"
                    if done is not None or state["why"] == "density negative":
                        break              # solved, or the split is inside the step
                self.diagnostics.step(outcome)
                located = None
                if (done is None and self.locate_births
                        and outcome == "settle failed: violation too large"):
                    located = self._birth_fraction(sh0, v0, state.get("overshoot"), scale,
                                                   target=0.2 * self.locate_tol)
                if done is None:
                    ctx.f, ctx.ds, ctx.outcome, ctx.located = f, ds, outcome, located
                    ctx.theta_far, ctx.sh0, ctx.state = theta_far, sh0, state
                    ctx.leg = (le0, off_start, target_le, target_off)
                    ctx.path_tol, ctx.ended = path_tol, ended
                    ctx.solve_structure, ctx.settle = solve_structure, settle
                    split = self.pipelines[STEP_FAILED_SPLIT].first(ctx)   # hook: R25
                    if split is not None:
                        ds = split.ds
                        continue
                if done is None and attempt is not None and self.snapshot_dir is not None:
                    cur_arcs, cur_vals = self._arcs_of(sh0, v0)
                    self.diagnostics.snapshot(self.snapshot_dir, self.snapshot_tag, outcome, {
                        "params": np.array(self.params, dtype=float), "path_tol": path_tol,
                        "frac": frac, "f": f, "ds": ds,
                        "before": {"arcs": cur_arcs, "values": cur_vals,
                                   "base": v0[:sh0.n_base] if isinstance(sh0, MultiArcShooting)
                                   else v0, "le": state["le"], "off": state["off"].copy()},
                        "attempt": {"arcs": attempt[0], "values": attempt[1], "base": attempt[2],
                                    "le": le, "off": np.array(off, dtype=float)}})
                if done is None:
                    # hook: step failed: rescue (ends, R17, R2, R39, R44)
                    rescue = self.pipelines[STEP_FAILED_RESCUE].first(ctx)
                    if isinstance(rescue, Retry):
                        ds = rescue.ds
                        continue
                    if isinstance(rescue, Turn):
                        # The curve goes on the other way: towards this line's other end,
                        # turning again at the next fold (arclength continuation).
                        other_le, other_off = turn_to if turn_to is not None else (le0, off_start)
                        return advance(other_le, other_off, on_step, ds_min,
                                       turn_to=(target_le, np.asarray(target_off, dtype=float)),
                                       backward=not backward)
                    if isinstance(rescue, Rescued):
                        state.pop("fold", None)
                        done, f, le, off = rescue.done, rescue.f, rescue.le, rescue.off
                if done is None:
                    # hook: step failed: size (R16, R37, R46, halving)
                    ctx.f, ctx.ds, ctx.located, ctx.theta_far = f, ds, located, theta_far
                    ds = self.pipelines[STEP_FAILED_SIZE].first(ctx)
                    continue
                prev2 = prev if prev is not None and prev[2] == arcs else None
                prev = (frac, v0, arcs)
                changed = self._arcs_of(done[0], done[1].v)[0] != arcs
                if changed:
                    # What to return to if this new structure leads nowhere.
                    new_arcs, new_values = self._arcs_of(done[0], done[1].v)
                    checkpoints.append((dict(state), key(new_arcs, new_values)))
                state.update(shoot=done[0], v=done[1].v, le=le, off=off)
                tight = float(np.sum(np.asarray(off, dtype=float) / scale))
                if tight < state["best_tight"] - 1e-9:
                    state["best_tight"], state["progress_at"] = tight, time.perf_counter()
                if self.check_lemmas:
                    from auto_optimal_control.lemmas import check
                    try:
                        for lemma, n_bad in check(done[0], done[1].v).items():
                            if n_bad:
                                self.diagnostics.oracle(lemma)
                    except _FAILURES:
                        self.diagnostics.oracle("check failed")
                if changed:
                    prev = None            # the structure changed: no secant across it
                    prev2 = None
                    state["progress_at"] = time.perf_counter()     # progress, for R15
                # hook: step accepted: size (R46, R37, growth)
                ctx.f, ctx.ds, ctx.changed, ctx.theta_done = f, ds, changed, theta_done
                ctx.steered = f - frac > floor
                frac, ds = f, self.pipelines[STEP_ACCEPTED_SIZE].first(ctx)
                if on_step is not None and on_step():
                    return True
            if backward:
                self._detail("    R44: the curve led back to the start of the leg")
                return False
            return True

        found = []

        def try_exact(depth=0):
            sh, v = state["shoot"], state["v"]
            self._last_rejected = None
            arcs, values = self._arcs_of(sh, v)
            entries, _ = self._mas_entries(arcs, values, names=True)
            step = Step(log_eps_min + s_max - state["le"] if self.bang_bang else 0.0,
                        10.0 ** state["le"] if self.bang_bang else np.nan, np.zeros(self.nc),
                        0.0, candidates=1, structure=tuple(entries))
            steps.append(step)
            hit = self._exact_jump(entries, v, tol)
            ectx = ExactContext(self, state, sh, v, tol, depth, split_exact)
            if hit is None:
                # hook: exact rejected, its first part (R20); R21 and R32 come after the
                # verdict is said, as they always have.
                got = self.pipelines[EXACT_REJECTED].rules[0].act(ectx)
                if got is not None:
                    hit, arcs = got
            self._say(f"active set at eps = {step.eps:.3g}: {self._names(arcs)}, "
                      f"exact {'accepted' if hit else 'not accepted'}")
            if hit is None:
                for rule in self.pipelines[EXACT_REJECTED].rules[1:]:
                    got = rule.act(ectx)
                    if got is not None:
                        hit = got[0]
                        break
            if hit is not None:
                step.accepted = True
                found.append(hit)
                return True
            return False

        def split_exact(depth):
            """Rule R21: the last exact solution rejected for a rising multiplier, split
            where it rises (Lemma H, as R5), settled on the route and tried again."""
            rejected = self._last_rejected
            if rejected is None or time.perf_counter() > deadline:
                return None
            xs, xsol, why = rejected
            try:
                if why == "mu rises (Lemma H)" and self.exact_splits and self.split_rising:
                    options = self._split_rising(xs, xsol.v) or []
                elif why == "limit violated off its arcs" and self.exact_births:
                    # Rule R32: the violation the exact check sees, brought in as the
                    # route brings in any violation (births, widenings, R26). It is
                    # below what the route resolves, so the options go to the exact
                    # solve directly (order-1 touches as R20's short arcs).
                    viol = self._free_violations(xs, xsol.v, scale, floor=1e-7)
                    options = self._activate(xs, xsol.v, viol) if viol else []
                    for arcs_b, vals_b in options:
                        self.diagnostics.fired("R32")
                        tries = [(arcs_b, vals_b)]
                        if any(kind == "touch" and self.kr.CONSTRAINT_ORDERS[c] == 1
                               for kind, c in arcs_b):
                            tries = self._touches_as_arcs(
                                arcs_b, vals_b, xs.unpack(xsol.v)[3] - xs.t0)
                        for arcs_x, vals_x in tries:
                            entries_x, ev_x = self._mas_entries(arcs_x, vals_x)
                            v_x = self._smoothed(entries_x, None, state["off"]).pack_arcs(
                                xsol.v[:xs.n_base], ev_x)
                            hit = self._exact_jump(entries_x, v_x, tol)
                            if hit is not None:
                                return hit
                    return None
                else:
                    return None
            except _FAILURES:
                return None
            saved = dict(state)
            for arcs_s, vals_s in options:
                try:
                    sh, r = solve_structure(arcs_s, vals_s, xsol.v[:xs.n_base], state["le"],
                                            state["off"])
                except _FAILURES as exc:
                    self._detail(f"  exact split: {exc}")
                    continue
                self._detail(f"  exact split {self._mas_entries(arcs_s, vals_s, True)[0]}: "
                             f"residual {r.residual_norm:.1e}")
                # The split moves the motion: a violation it causes is brought in as it
                # is, as after any structure change (there is no step to shorten).
                done = settle(sh, r, state["le"], state["off"], changed=True) if r.success else None
                if done is None:
                    continue
                self.diagnostics.fired("R21")
                state.update(shoot=done[0], v=done[1].v)
                if try_exact(depth + 1):
                    return found.pop()     # the caller records it
                solves = state["solves"]
                state.clear()
                state.update(saved, solves=solves)
            return None

        def out_of_budget():
            return state["solves"] > max_solves or time.perf_counter() > deadline or stalled()

        def anneal():
            """The vector of offsets to zero: together, then one by one, then back up."""
            backtracks = 0
            if self.offset_order == "sequential":
                # Rule R13: the limits come in one at a time, the one the free motion
                # breaks most first. A limit that slows the motion is then tight before
                # the ones whose holdable set shrinks with speed are born (Lemma I'), so
                # their births happen off the edge, as on the true path.
                excess = np.maximum(peaks0, 0.0) / scale
                for c in np.argsort(-excess):
                    if excess[c] <= 0 or out_of_budget():
                        continue
                    target = state["off"].copy()
                    target[c] = 0.0
                    self.diagnostics.fired("R13")
                    advance(state["le"], target, ds_min=1e-4)
            while True:
                # Rule R45: with a larger starting weight (R24), the weight comes down
                # together with the offsets, on the diagonal of the (log w, offsets)
                # plane, instead of after them (each point a real problem: Lemma W).
                le_leg = (le_target if getattr(self, "weight_path", "corner") == "diagonal"
                          and le_target is not None else state["le"])
                if advance(le_leg, np.zeros(self.nc)):
                    return True
                if out_of_budget():
                    return False
                before = state["off"].copy()
                order = np.argsort(-before / scale)
                for c in order:
                    if before[c] <= 0 or out_of_budget():
                        continue
                    target = state["off"].copy()
                    target[c] = 0.0
                    advance(state["le"], target, ds_min=1e-3)
                moved = np.max(np.abs(state["off"] - before) / scale)
                if moved > 1e-3:
                    self._detail(f"    offsets moved one by one, by up to {moved:.1e} of their scale")
                    continue
                if not checkpoints or backtracks >= self.active_set_backtracks or out_of_budget():
                    return False
                saved, structure = checkpoints.pop()
                banned.add(structure)
                backtracks += 1
                solves = state["solves"]
                state.clear()
                state.update(saved, solves=solves)
                self._detail(f"    backing up: ruling out {structure}")

        self._say(f"active set: tightening the limits from offsets {np.round(off0, 6)}")
        if not anneal():
            self._say("active set: tightening stalled")
            return None
        if le_target is not None and state["le"] != le_target:
            # Rule R24: the limits are in at the larger weight; walk it down to its value
            # with the structure following, and solve the exact problem there.
            self._say(f"active set: secondary weight from {10.0 ** state['le']:.3g} "
                      f"to {self.weight_value:.3g}")
            if not advance(le_target, np.zeros(self.nc), ds_min=1e-4):
                gap = state["le"] - le_target
                if not (getattr(self, "walk_exact", False) and gap < self.walk_exact_gap):
                    self._say("active set: lowering the secondary weight stalled")
                    return None
                # Rule R47 (Lemma W (1)): the walk stalled close to the policy's weight;
                # the exact problem there is the same family's, so its Newton solve is
                # tried from the walk's last solution, and acceptance judges it.
                self.diagnostics.fired("R47")
                self._say(f"active set: the walk stalled {gap:.2f} decades above the weight; "
                          f"the exact solve from there")
            state["le"] = le_target
        if try_exact():
            return found[0]
        converted = self._end_structure(state["shoot"], state["v"])
        if converted is not None:
            try:
                sh, r = solve_structure(*converted, state["v"][:state["shoot"].n_base], state["le"],
                                        state["off"])
            except _FAILURES as exc:
                sh, r = None, None
                self._detail(f"  arcs run out to the ends: {exc}")
            if r is not None:
                self._detail(f"  arcs run out to the ends: residual {r.residual_norm:.1e}")
                if r.success:
                    state.update(shoot=sh, v=r.v)
                    if try_exact():
                        return found[0]
        if not self.bang_bang:
            return None
        if advance(log_eps_min, np.zeros(self.nc), on_step=try_exact) and found:
            return found[0]
        return found[0] if found else None

    def _junction_text(self, shoot, v):
        if isinstance(shoot, MultiArcShooting) and shoot._nested:
            return " ".join(f"{ev}:{self.kr.CONSTRAINT_NAMES[shoot.arcs[i][1]]}"
                            for ev, i in shoot.events) + " | " + self._junction_text_plain(shoot, v)
        return self._junction_text_plain(shoot, v)

    def _junction_text_plain(self, shoot, v):
        arcs, values = self._arcs_of(shoot, v)
        return " ".join(f"{self.kr.CONSTRAINT_NAMES[c]}@" + (
            f"{val[0]:.5f}" if kind == "touch" else f"[{val[0]:.5f},{val[1]:.5f}]")
            for (kind, c), val in zip(arcs, values)) or "-"

    def _names(self, arcs):
        return ", ".join(f"{kind} {self.kr.CONSTRAINT_NAMES[c]}" for kind, c in arcs) or "free"

    # -- the whole solve --------------------------------------------------------

    def solve(self, params, costates, final_time=None, **options):
        """From a costate distribution to the exact extremal.

        ``params`` are the values of the stated problem's parameters; the
        policy, if one applies, is handled here (see :class:`Homotopy`).
        The options are those of :meth:`_solve`. Returns a
        :class:`HomotopyResult`.
        """
        params = np.asarray(params, dtype=float).reshape(-1)
        if len(params) != len(self.stated.params):
            raise ValueError(f"expected {len(self.stated.params)} parameters "
                             f"{[str(p) for p in self.stated.params]}, got {len(params)}")
        self.diagnostics = Diagnostics()
        self._ties = set()                 # R39's hand-overs belong to one solve
        sink, SingleShooting.sink = SingleShooting.sink, self.diagnostics.evaluations
        budget = self.active_set_seconds
        if self.solve_seconds is not None:
            self._solve_end = time.perf_counter() + self.solve_seconds
            self.active_set_seconds = min(budget, self.solve_seconds)
        try:
            if self._stages is None:
                r = self._solve(params, costates, final_time, **options)
                left = None if self._solve_end is None else self._solve_end - time.perf_counter()
                if not r.exact and self.reversal and (left is None or left > 0):
                    if left is not None:
                        self.active_set_seconds = min(budget, left)
                    back = self._through_reversal(params, costates, final_time, **options)
                    if back is not None:
                        self.diagnostics.fired("R4")
                        r = back
            else:
                r = self._lexicographic(params, costates, final_time, **options)
        finally:
            SingleShooting.sink = sink
            self.active_set_seconds = budget
            self._solve_end = None
        r.policy = self.policy_text
        r.diagnostics = self.diagnostics.as_dict()
        return r

    @property
    def certificates(self):
        """The certificates of the problem solved (:func:`~auto_optimal_control.certificates.certify`)."""
        if self._certificates is None:
            from auto_optimal_control.certificates import certify
            self._certificates = certify(self.problem)
        return self._certificates

    def _reversed_params(self, params):
        """The parameters of the reversed problem (rule R4), or None if the ends do not
        map to parameters: each end value must be a number or a parameter symbol."""
        c = self.certificates["time_reversible"]
        if not c:
            return None
        p = self.problem
        signs = dict(zip(p.states, c.detail["state_signs"]))
        index = {s: i for i, s in enumerate(p.params)}
        final = {}
        for e in p.terminal_constraints:
            for s in p.states:
                rest = sp.expand(s - e)
                if not (rest.free_symbols & set(p.states)):
                    final[s] = rest
        out = np.array(params, dtype=float)
        for s in p.states:
            if s not in p.initial_state or s not in final:
                return None
            e0, ef = sp.sympify(p.initial_state[s]), sp.sympify(final[s])
            r = signs[s]
            if e0.is_number and ef.is_number:
                if not (float(r * ef) == float(e0) and float(r * e0) == float(ef)):
                    return None
            elif e0 in index and ef in index:
                out[index[e0]], out[index[ef]] = r * params[index[ef]], r * params[index[e0]]
            else:
                return None
        return out

    def _through_reversal(self, params, costates, final_time, **options):
        """Rule R4 (Lemma F): solve the reversed problem and map its extremal back exactly.

        The reversed extremal's junction times reflect (t -> T - t), each arc keeps
        its jump and moves to the limit reversal maps it to, and the costates map
        as lambda(0) = -R lambda~(T); the exact problem is then solved from there."""
        if self.bang_bang:
            return None
        rev = self._reversed_params(params)
        if rev is None:
            return None
        self._say("reversal: solving the reversed problem (rule R4)")
        rr = self._solve(rev, costates, final_time, **options)
        if not rr.exact or not isinstance(rr.shooting, (MultiArcShooting, SingleShooting)):
            self._say("reversal: the reversed problem was not solved either")
            return None
        c = self.certificates["time_reversible"]
        R = np.array(c.detail["state_signs"], dtype=float)
        limit_map = c.detail["limit_map"]
        sh, v = rr.shooting, rr.solution.v
        n = sh.kernels.N_STATES
        segs = sh._evaluate(np.asarray(v, dtype=float), jac=False)[2] if isinstance(sh, MultiArcShooting) \
            else None
        if segs is not None:
            zT = segs[-1][-1]["z"]
        else:
            x0, lam0, _, tf_ = sh.unpack(v)
            zT = sh.kernels.propagate(sh.t0, np.concatenate([x0, lam0]), tf_, sh.params,
                                      **self.options)["z"]
        T = rr.solution.tf
        lam0 = -R * zT[n:]
        arcs, values = self._arcs_of(sh, v)
        f_arcs, f_vals = [], []
        for (kind, cc), val in zip(arcs, values):
            f_arcs.append((kind, limit_map[cc]))
            if kind == "boundary":
                a, b, pi = val
                f_vals.append((T - b, T - a, pi))
            else:
                f_vals.append((T - val[0], *val[1:]))
        self.params = np.asarray(params, dtype=float)
        kx, px = self.exact_kernels, self.exact_params()
        entries, f_vals = self._mas_entries(f_arcs, f_vals)
        if entries:
            shoot = self._multi_arc(kx, px, entries, max_nfev=self.attempt_nfev, open_end=True,
                                     **self.options)
            base = shoot.pack(lam0, None, T if kx.FREE_FINAL_TIME else None)
            guess = shoot.pack_arcs(base, f_vals)
        else:
            shoot = SingleShooting(kx, px, **self.options)
            guess = shoot.pack(lam0, None, T if kx.FREE_FINAL_TIME else None)
        if kx.N_MULTIPLIERS:
            # The terminal multipliers enter linearly: fit them. A mapped extremal that does
            # not propagate forward has nothing to fit; the exact solve judges it as it is.
            try:
                Rv, Jv = shoot.residual_jacobian(guess)
                cols = slice(shoot.n_free + n, shoot.n_free + n + kx.N_MULTIPLIERS)
                guess[cols] += np.linalg.lstsq(Jv[:, cols], -Rv, rcond=None)[0]
            except _FAILURES as exc:
                self._say(f"reversal: the mapped extremal does not propagate ({exc})")
        hit = self._exact_jump(entries, guess, _scaled(options.get("tol", 1e-10), guess))
        if hit is None:
            self._say("reversal: the mapped extremal was not accepted")
            return None
        J, xs, xsol = hit
        return HomotopyResult(xsol, xs, self._entries(xs), True, J, xs.params, rr.steps, rr.tries,
                              f"exact through the reversed problem (rule R4); {rr.message}")

    def _lexicographic(self, params, costates, final_time, **options):
        """Each component's minimum time, then minimum effort at ``(1 + delta) T*``."""
        stages, times = [], []
        for h, idx in self._stages:
            if callable(costates):
                part = lambda rng, f=costates, idx=idx: np.asarray(f(rng))[idx]
            else:
                part = tuple(np.asarray(b, dtype=float)[idx] for b in costates)
            r = h.solve(params, part, final_time, **options)
            stages.append(r)
            if not r.exact:
                r0 = HomotopyResult(r.solution, r.shooting, (), False, np.nan, None,
                                    r.steps, r.tries,
                                    f"the minimum time of component {len(stages)} was not "
                                    f"found: {r.message}", stages=stages)
                return r0
            times.append(r.solution.tf)
        tf = (1.0 + self.policy.delta) * max(times)
        self._say(f"lexicographic: component minimum times {times}, t_f = {tf}")
        r = self._solve(np.append(params, tf), costates, None, **options)
        r.stages = stages
        r.message = (f"t_f = (1 + {self.policy.delta}) * {max(times):.12g}; " + r.message)
        return r

    def _solve(self, params, costates, final_time=None, *, n_tries=200, seed=0, eps0=1.0,
               s_max=10.0, max_steps=300, n_jitter=6, tol=1e-10):
        """The automatic solve of ``self.problem`` (the stated problem with its policy).

        ``params`` are the values of ``problem.params``; penalty weights the
        problem defines are not used (:func:`regularize` replaces them), so
        any value will do for them. ``costates`` is ``(low, high)`` (uniform
        per component) or a function ``rng -> lam0``; ``final_time``
        likewise for ``t_f`` (required when it is free). At most ``n_tries``
        random draws are made, from ``seed`` (an int or a
        ``numpy.random.Generator``). ``eps0`` is the first smoothing weight
        and ``s_max`` how many decades ``eps`` may fall; ``n_jitter`` is the
        number of extra seeds, with perturbed junction times, per candidate
        structure. Returns a :class:`HomotopyResult`.
        """
        params = np.asarray(params, dtype=float).reshape(-1)
        if len(params) != len(self.problem.params):
            raise ValueError(f"expected {len(self.problem.params)} parameters "
                             f"{[str(p) for p in self.problem.params]}, got {len(params)}")
        self.params = params
        if self.kr.FREE_FINAL_TIME and final_time is None:
            raise ValueError("the final time is free: give a final_time distribution")
        broken = self._limits_broken_at_the_ends()
        if broken:
            return HomotopyResult(None, None, (), False, np.nan, None, [], 0,
                                  f"infeasible: {'; '.join(broken)}")
        rng = np.random.default_rng(seed)
        kr = self.kr
        eps_start = [eps0, 10 * eps0, 0.1 * eps0] if self.bang_bang else [None]
        zero = np.zeros(self.nc)
        sol, tries = None, 0
        if kr.FREE_FINAL_TIME and not self.bang_bang:
            shoot = SingleShooting(kr, self.regularized_params(None, zero), **self.options)
            # With the penalties off the limits do not enter, so their values can change.
            key = tuple(params[self._free_arc_params()])
            if key != self._seed_key:
                self._seed_key = key
                self._seed = self._separate_seed(shoot, final_time, tol)
            seed_v = self._seed
            if seed_v is not None:
                # To the path's accuracy: the exact solve at the end polishes it.
                shoot = self._smoothed([], None, zero)
                shoot.max_nfev = self.attempt_nfev
                candidate = shoot.solve(seed_v, tol=max(tol, 1e-6), options=_budget("hybr", 80))
                if candidate.success:
                    sol = candidate
        for e0 in eps_start if sol is None else ():
            shoot = SingleShooting(kr, self.regularized_params(e0, zero), **self.options)
            sol, n = shoot.multistart(self._sampler(shoot, costates, final_time),
                                      n_tries=n_tries, seed=rng, tol=tol)
            tries += n
            if sol is not None and sol.success:
                eps0 = e0
                break
        if sol is None or not sol.success:
            return HomotopyResult(sol, None, (), False, np.nan, None, [], tries,
                                  "no random draw solved the regularized problem")
        self._say(f"regularized solve after {tries} draws, eps = {eps0}")

        # Step 3: continuation in phases. The penalty of a newly violated limit
        # ramps up first, with eps held (log10 rho_c from rho_ref_c 10^m_start
        # to rho_ref_c 10^m_max); then eps falls, with rho held.
        log_eps = np.log10(eps0) if self.bang_bang else 0.0
        log_eps_min = log_eps - s_max
        rho_ref = np.zeros(self.nc)
        m = np.full(self.nc, -np.inf)      # log10(rho / rho_ref); -inf: penalty off
        m_start, m_max = -6.0, 4.0

        def params_at(le, mm):
            eps = 10.0 ** le if self.bang_bang else None
            rho = np.where(np.isfinite(mm), rho_ref * 10.0 ** mm, 0.0)
            return self.regularized_params(eps, rho), eps, rho

        def activate(shoot, v):
            """Switch on the penalty of every newly violated limit (at m_start)."""
            if not self.nc:
                return False
            viol = shoot.constraint_violation(v)
            new = (viol > 1e-9) & ~np.isfinite(state["m"])
            if not new.any():
                return False
            x0, lam0, _, tf = shoot.unpack(v)
            tr = kr.trajectory(shoot.t0, np.concatenate([x0, lam0]), tf, shoot.params,
                               n_points=2001, **self.options)
            J = abs(self._regularized_cost(shoot, v)) or 1.0
            for c in np.flatnonzero(new):
                P = np.trapezoid(np.maximum(tr["g"][:, c], 0.0) ** 2 / 2, tr["t"])
                rho_ref[c] = J / max(P, 1e-300)
                self._say(f"limit {kr.CONSTRAINT_NAMES[c]} violated: rho_ref = {rho_ref[c]:.3g}")
            return new

        steps = []
        state = {"le": log_eps, "m": m.copy(), "v": sol.v, "sol": sol, "prev": None}
        best = None

        def record_and_attempt(le, mm, v, sol):
            p, eps, rho = params_at(le, mm)
            shoot = SingleShooting(kr, p, **self.options)
            step = Step(log_eps - le, eps if eps is not None else np.nan, rho.copy(),
                        sol.residual_norm)
            steps.append(step)
            read = self._read(shoot, v)
            accepted, tried, _ = self._attempt(read, v[:shoot.n_unknowns], n_jitter, rng, tol)
            step.candidates = tried
            self._say(f"eps = {eps}  rho = {rho}  candidates tried: {tried}, "
                      f"accepted: {len(accepted)}")
            if accepted:
                step.accepted = True
            return shoot, accepted

        def advance(move, size):
            """One adaptive step along ``move(state, size) -> (le, m)``: the new
            state, or None when the step falls below its floor."""
            ds = size
            while ds >= 1e-3 and time.perf_counter() < state["deadline"]:
                le, mm = move(ds)
                p, _, _ = params_at(le, mm)
                nxt = SingleShooting(kr, p, **self.options)
                v, prev = state["v"], state["prev"]
                guesses = [v]
                if prev is not None and prev[2] > 0:
                    guesses.insert(0, v + (v - prev[1]) * ds / prev[2])
                for g in guesses:
                    try:
                        r = nxt.solve(g, tol=tol)
                    except _FAILURES:
                        continue
                    if r.success:
                        state.update(le=le, m=mm, prev=(None, v, ds), v=r.v, sol=r)
                        return nxt, ds
                ds *= 0.5
            return None, ds

        if not self.bang_bang and not self.nc:
            # The regularized problem is the exact one.
            J = self.objective(shoot, sol.v)
            return HomotopyResult(sol, shoot, (), True, J, shoot.params,
                                  [Step(0.0, np.nan, np.zeros(0), sol.residual_norm)], tries,
                                  "exact: no bang-bang controls or path constraints")
        # The active-set route goes first unless the exact problem may have
        # singular arcs, which only the penalty route reads; each route gets
        # the other as its fallback.
        first = sol
        kx = self.exact_kernels
        # The active-set route needs every limit to have exact arcs (order 1 or 2).
        can_activate = self.nc and all(self.kr.EXACT_ARCS)
        active_first = can_activate and not any(kx.SINGULAR_ARCS)

        def active_route():
            # Rule R15: a portfolio of routes. Each is a valid route (Lemma D: almost every
            # draw of the offsets separates the births), and which cases a route solves
            # depends on its path; on failure the next one gets the time that is left.
            members = [{}] + [dict(m_) for m_ in self.portfolio
                              if "weight_start" not in m_ or self._weight_route()]
            if len(members) > 1 and self.parallel_routes:
                hit = self._routes_in_parallel(members, first, eps0, s_max, steps, tol)
                if hit is None:
                    return None
                J, xs, xsol = hit
                return HomotopyResult(xsol, xs, self._entries(xs), True, J, xs.params, steps,
                                      tries, f"exact after {len(steps)} steps (active set)")
            total, start = self.active_set_seconds, time.perf_counter()
            saved = {k_: getattr(self, k_) for m_ in members for k_ in m_}
            hit = None
            try:
                for n_, member in enumerate(members):
                    left = total - (time.perf_counter() - start)
                    if left <= 1.0:
                        break
                    for k_, val_ in saved.items():
                        setattr(self, k_, val_)
                    for k_, val_ in member.items():
                        setattr(self, k_, val_)
                    # Every route may use all the time left: a route that stops making
                    # progress (stall_seconds without a new low of its offsets) yields.
                    share = left
                    self.active_set_seconds = share
                    if n_:
                        self.diagnostics.fired("R15")
                        self._say(f"active set: route {n_ + 1} of {len(members)} ({member}), {share:.0f} s")
                    hit = self._active_set(first, eps0, s_max, steps, tol,
                                           max_solves=self.max_path_solves)
                    if hit is not None:
                        break
            finally:
                self.active_set_seconds = total
                for k_, val_ in saved.items():
                    setattr(self, k_, val_)
            if hit is None:
                return None
            J, xs, xsol = hit
            return HomotopyResult(xsol, xs, self._entries(xs), True, J, xs.params, steps,
                                  tries, f"exact after {len(steps)} steps (active set)")

        if active_first:
            out = active_route()
            if out is not None:
                return out
            if self.kr.system.riding is not None:
                # A stiff penalty on limits this large takes minutes per step, and
                # reads back the structures the active-set route has already tried.
                return HomotopyResult(sol, shoot, (), False, np.nan, shoot.params, steps, tries,
                                      "no exact structure: the active-set route stalled")
            self._say("active set: no exact structure; trying the penalty route")
        state["deadline"] = time.perf_counter() + self.penalty_seconds
        new = activate(shoot, sol.v)
        if new is not False:
            state["m"][new] = m_start
        shoot, accepted = record_and_attempt(state["le"], state["m"], state["v"], state["sol"])
        ds_rho, ds_eps = 0.5, 0.25
        done = np.zeros(self.nc, dtype=bool)
        while not accepted and len(steps) < max_steps and (
                time.perf_counter() < state["deadline"]):
            best = (shoot, state["sol"])
            ramping = np.isfinite(state["m"]) & (state["m"] < m_max) & ~done
            if ramping.any():
                def move(ds):
                    mm = state["m"].copy()
                    mm[ramping] = np.minimum(mm[ramping] + ds, m_max)
                    return state["le"], mm
                nxt, ds = advance(move, ds_rho)
                if nxt is None:
                    # The penalty stalls: hold it where it is and go on lowering eps.
                    self._say("penalty continuation stalled")
                    done |= ramping
                    state["prev"] = None
                    continue
                ds_rho = min(ds * 1.5, 1.0)
            elif self.bang_bang and state["le"] > log_eps_min:
                def move(ds):
                    return max(state["le"] - ds, log_eps_min), state["m"]
                nxt, ds = advance(move, ds_eps)
                if nxt is None:
                    break
                ds_eps = min(ds * 1.5, 1.0)
            else:
                break
            new = activate(nxt, state["v"])
            if new is not False:
                state["m"][new] = m_start
                state["prev"] = None
            shoot, accepted = record_and_attempt(state["le"], state["m"], state["v"],
                                                 state["sol"])
        if accepted:
            J, xs, xsol = min(accepted, key=lambda a: a[0])
            return HomotopyResult(xsol, xs, self._entries(xs), True, J, xs.params, steps,
                                  tries, f"exact after {len(steps)} steps")
        if can_activate and not active_first:
            self._say("penalty route: no exact structure; trying the active-set route")
            out = active_route()
            if out is not None:
                return out
        shoot, sol = best if best is not None else (shoot, state["sol"])
        return HomotopyResult(sol, shoot, (), False, np.nan, shoot.params, steps, tries,
                              "no exact structure accepted before the continuation ended")

    def _regularized_cost(self, shoot, v):
        x0, lam0, _, tf = shoot.unpack(v)
        r = shoot.kernels.propagate(shoot.t0, np.concatenate([x0, lam0]), tf, shoot.params,
                                    cost=True, **self.options)
        xf = r["z"][:shoot.kernels.N_STATES]
        return float(r["cost"]) + float(self._terminal(tf, list(xf), list(self.params)))

    @staticmethod
    def _entries(shoot):
        """The structure as MultiArcShooting would be given it."""
        if not isinstance(shoot, MultiArcShooting):
            return ()
        k = shoot.kernels
        if not shoot._legacy and shoot._nested:
            out = []
            for ev, i in shoot.events:
                kind, c, _ = shoot.arcs[i]
                out.append(({"in": "enter", "out": "exit"}.get(ev, ev), k.CONSTRAINT_NAMES[c]))
            return tuple(out)
        out = []
        for kind, c, extra in shoot.arcs:
            if kind in ("boundary", "touch"):
                out.append((kind, k.CONSTRAINT_NAMES[c]))
                continue
            name = k.system.switching_functions[c].control.name
            sides = tuple("upper" if f else "lower" for f in extra)
            if kind == "switch":
                out.append(("switch", name))
            elif kind == "singular_joined":
                out.append(("singular", name, *sides, "joined"))
            else:
                out.append(("singular", name, *sides))
        return tuple(out)


def _newton(system, guess, tol, steps=10, accept=0.0):
    """Damped Newton on ``system(v, with_jacobian)``: the root, or None.

    The step length is tried on residuals alone, and the Jacobian found where it lands.
    A caller that only needs a guess can pass ``accept``: the residual at which a stalled
    iteration still counts, since the noise of a loose integration can sit above ``tol``.
    """
    v = np.asarray(guess, dtype=float)
    try:
        R = system(v, False)
    except _FAILURES:
        return None
    best = float(np.linalg.norm(R))
    for _ in range(steps):
        if best < tol:
            return v
        try:
            _, J = system(v, True)
            delta = np.linalg.lstsq(J, R, rcond=None)[0]
        except (*_FAILURES, np.linalg.LinAlgError):
            return None
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            w = v - scale * delta
            try:
                Rw = system(w, False)
            except _FAILURES:
                continue
            if np.linalg.norm(Rw) < best:
                v, R, best = w, Rw, float(np.linalg.norm(Rw))
                break
        else:
            return v if best < accept else None
    return v if best < max(tol, accept) else None


def _draw(spec, rng):
    """One sample of a distribution: a function of ``rng``, or ``(low, high)``."""
    if callable(spec):
        return spec(rng)
    lo, hi = (np.asarray(b, dtype=float) for b in spec)
    return rng.uniform(lo, hi)


def _scaled(tol, v):
    """``tol`` relative to the size of the unknowns (absolute while they are below 1):
    costates in the thousands cannot be resolved to 1e-10 absolutely."""
    return tol * max(1.0, float(np.abs(np.asarray(v, dtype=float)).max(initial=0.0)))


def _budget(method, n):
    """MINPACK options with an evaluation budget."""
    if method == "lm":
        return {"xtol": 1e-14, "ftol": 1e-14, "maxiter": int(n)}
    return {"xtol": 1e-13, "maxfev": int(n)}


def solve(problem, params, costates, final_time=None, *, cache_dir=None, verbose=False,
          policy="auto", **options):
    """:meth:`Homotopy.solve` in one call: see :class:`Homotopy` and the module docs."""
    return Homotopy(problem, cache_dir=cache_dir, verbose=verbose, policy=policy).solve(
        params, costates, final_time, **options)
