"""Where the solver's rules act, and the protocol they implement.

The route (``Homotopy._active_set``) is a loop of path steps. Each rule of
:mod:`auto_optimal_control.rules` acts at one *hook* of that loop, on a trigger,
and its specification there names the lemma it rests on. This module gives the
hooks names, the state a rule sees there (a :class:`StepContext`), and the
protocol a rule implements (:class:`PathRule`). A :class:`Pipeline` holds a
hook's rules in order, so the solver's behaviour at that hook is the
composition of the rules that are on.

Two kinds of rules, by how often they act:

- *Path rules* act once per path step or per event, a few hundred times per
  solve. They are Python objects implementing :class:`PathRule`.
- *Evaluation rules* act in every evaluation of the shooting function,
  thousands of times per solve (junction rows, closed-form arcs, the Newton
  test). They are options of the compiled evaluation, not callbacks.

The hooks of a path step, in order:

``predict``
    The guesses for the step's solve, best first (the secant; R48).
``step failed: size``
    No guess settled: the next, shorter step (R16, R37, R46, halving).
``step accepted: size``
    A guess settled: the next step (R46, R37, growth).

Each rule's on/off switch is the attribute its specification in
:mod:`auto_optimal_control.rules` names, so the switches stay the single
source of truth for which rules are on.

The hook map. Every rule's specification names a hook. These are where each
hook acts in the solver; *pipeline* marks the ones implemented by this
module's protocol, and the others are still written inline:

=============================  =====================================  ==========================
hook (rules.py)                where                                  rules
=============================  =====================================  ==========================
predict                        ``advance`` *pipeline*                 R48, secant, last
consistent step (settle)       ``settle`` *pipeline*                  R25 R30 R38 R39 R44, the
                                                                      peak, dead-arc and bound
                                                                      monitors (R18 R28 inside)
step failed                    ``advance`` *pipeline*                 R25 (split), ends, R17,
                                                                      R2, R39, R44, then the
                                                                      size: R16 R37 R46
step accepted                  ``advance`` *pipeline*                 R46 R37, growth
exact rejected                 ``try_exact`` *pipeline*               R20 R21 R32
path step                      ``advance``, ``_solve_structure``      R37 (rejection), R50
route start                    ``Homotopy._solve``, ``_active_set``   R1 R13 R24 R45
route failed                   ``_solve.active_route`` (portfolio)    R15 R47
solve failed                   ``Homotopy._through_reversal``         R4
seed failed                    ``Homotopy._solve_structure``          R8
newborn arc                    ``Homotopy._activate``                 R6 R12 R26
structure                      ``_merge_touching``, touches           R11
settle failed: density neg.    ``Homotopy._split_fraction``           R54
exact rejected (reorders)      ``_exact_jump``                        R41
not converged (evaluation)     ``SingleShooting.solve_chord``         R55
junction rows (evaluation)     ``MultiArcShooting._evaluate_events``  R7 R43
evaluation (evaluation)        shooting, kernels                      R9 R10 R10b R22 R23 R60
riding (evaluation)            ``riding.py`` (generated C)            R14
=============================  =====================================  ==========================
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

# -- the hooks ----------------------------------------------------------------

PREDICT = "predict"
STEP_FAILED_SIZE = "step failed: size"
STEP_ACCEPTED_SIZE = "step accepted: size"

HOOKS = (PREDICT, STEP_FAILED_SIZE, STEP_ACCEPTED_SIZE)


@dataclass
class StepContext:
    """What a path rule sees of a step of a leg ``s in [0, 1]``.

    ``frac`` is the last accepted point of the leg and ``f`` the step's end;
    ``ds`` the step asked for (``f - frac`` unless clipped at 1). ``prev`` and
    ``prev2`` are ``(frac, v, structure)`` of the solutions before ``v0``
    (None across a change of structure). ``memory`` is the leg's own state for
    rules that remember (R46), kept from step to step.
    """

    homotopy: Any
    frac: float
    f: float
    ds: float
    ds_min: float
    v0: Any
    arcs: Any
    prev: Optional[tuple] = None
    prev2: Optional[tuple] = None
    moves_weight: bool = False      # the leg moves the secondary weight (R48)
    steered: bool = False           # R37 steers this step (above its floor)
    located: Optional[float] = None    # R16: the birth's fraction of the step
    theta_far: Optional[tuple] = None  # R37: (theta, order) of a step too long
    theta_done: Optional[tuple] = None  # (theta, order) of the step that settled
    changed: bool = False           # the accepted step changed the structure
    memory: dict = field(default_factory=dict)
    # the leg, for the failed step's rescues
    outcome: Optional[str] = None
    sh0: Any = None
    leg: Any = None                 # (le0, off_start, target_le, target_off)
    path_tol: float = 0.0
    state: Optional[dict] = None
    solve_structure: Any = None
    settle: Any = None
    ended: Optional[set] = None


@runtime_checkable
class PathRule(Protocol):
    """A rule acting at one hook of the path step.

    ``name`` is the rule's name in :mod:`auto_optimal_control.rules` (None for
    the solver's defaults, always on). ``act`` returns the hook's value, or
    None to leave the step to the next rule of the pipeline.
    """

    name: Optional[str]
    hook: str

    def act(self, ctx: StepContext) -> Any:
        ...


def is_on(rule, h):
    """Whether ``rule`` is on for Homotopy ``h``, by its specification's switch."""
    if rule.name is None:
        return True
    from .rules import BY_NAME, enabled
    if rule.name not in BY_NAME:
        return True
    return enabled(h)[rule.name]


class Pipeline:
    """A hook's rules in order. ``first`` asks each rule that is on in turn and
    returns the first answer; ``collect`` gathers every answer (for hooks whose
    value is a list, as the predictor's guesses)."""

    def __init__(self, hook, rules):
        self.hook = hook
        self.rules = list(rules)
        for r in self.rules:
            assert r.hook == hook, (r, hook)

    def active(self, h):
        return [r for r in self.rules if is_on(r, h)]

    def first(self, ctx):
        for r in self.active(ctx.homotopy):
            out = r.act(ctx)
            if out is not None:
                if r.name is not None and r.fires:
                    ctx.homotopy.diagnostics.fired(r.name)
                return out
        return None

    def collect(self, ctx):
        out = []
        for r in self.active(ctx.homotopy):
            got = r.act(ctx)
            if got is not None:
                out.extend(got)
        return out


# -- predict -------------------------------------------------------------------

class QuadraticPredictor:
    """Rule R48 (Lemma S (2) with W (2)): on a leg that moves the weight, the
    quadratic through the last three solutions of one structure; its error is
    O(h^3) where the secant's is O(h^2)."""

    name, hook, fires = "R48", PREDICT, False

    def act(self, ctx):
        prev, prev2, f = ctx.prev, ctx.prev2, ctx.f
        if not (ctx.moves_weight and prev is not None and prev2 is not None
                and prev[2] == ctx.arcs and prev2[2] == ctx.arcs
                and ctx.frac > prev[0] > prev2[0]):
            return None
        x0, x1, x2 = prev2[0], prev[0], ctx.frac
        l0 = (f - x1) * (f - x2) / ((x0 - x1) * (x0 - x2))
        l1 = (f - x0) * (f - x2) / ((x1 - x0) * (x1 - x2))
        l2 = (f - x0) * (f - x1) / ((x2 - x0) * (x2 - x1))
        return [l0 * prev2[1] + l1 * prev[1] + l2 * ctx.v0]


class SecantPredictor:
    """The secant along the path, while the structure holds."""

    name, hook, fires = None, PREDICT, False

    def act(self, ctx):
        prev = ctx.prev
        if prev is None or prev[2] != ctx.arcs or not ctx.frac > prev[0]:
            return None
        return [ctx.v0 + (ctx.v0 - prev[1]) * (ctx.f - ctx.frac) / (ctx.frac - prev[0])]


class LastSolution:
    """The last solution itself, always tried last."""

    name, hook, fires = None, PREDICT, False

    def act(self, ctx):
        return [ctx.v0]


# -- step failed: size ----------------------------------------------------------

class LocatedBirth:
    """Rule R16 (Lemma N): the step ends where the secant puts the first birth."""

    name, hook, fires = "R16", STEP_FAILED_SIZE, True

    def act(self, ctx):
        if ctx.located is None:
            return None
        ctx.homotopy._detail(f"    birth located at {ctx.located:.2f} of the step")
        return (ctx.f - ctx.frac) * ctx.located


class ContractionCut:
    """Rule R37 (Deuflhard): theta ~ h^p, so a step too long for Newton is cut to
    the target theta."""

    name, hook, fires = None, STEP_FAILED_SIZE, False   # R37 counted where it rejects

    def act(self, ctx):
        tf = ctx.theta_far
        if tf is None or not np.isfinite(tf[0]):
            return None
        h = ctx.homotopy
        return (ctx.f - ctx.frac) * float(np.clip(
            (h.contraction_target / tf[0]) ** (1.0 / tf[1]), 0.1, 0.5))


class Halve:
    """The default: half the step. Rule R46 remembers the failed step."""

    name, hook, fires = None, STEP_FAILED_SIZE, False

    def act(self, ctx):
        if getattr(ctx.homotopy, "step_memory", False):
            ctx.memory["fail_ds"], ctx.memory["since_fail"] = ctx.f - ctx.frac, 0
        return ctx.ds * 0.5


# -- step accepted: size --------------------------------------------------------

class Grow:
    """The default: 1.5 times the step; Rule R46 (Lemma S) grows to at most 0.9 of
    the last failed step for three successes, then probes again, and a new
    structure forgets it; Rule R37 sets the step that gives the target
    contraction. Clipped to a quarter of the leg."""

    name, hook, fires = None, STEP_ACCEPTED_SIZE, False

    def act(self, ctx):
        h, mem = ctx.homotopy, ctx.memory
        nxt = ctx.ds * 1.5
        if getattr(h, "step_memory", False):
            if ctx.changed:
                mem["fail_ds"] = None
            elif mem.get("fail_ds") is not None:
                mem["since_fail"] = mem.get("since_fail", 0) + 1
                if mem["since_fail"] <= 3:
                    nxt = min(nxt, 0.9 * mem["fail_ds"])
                else:
                    mem["fail_ds"] = None
        td = ctx.theta_done
        if not ctx.changed and td is not None and td[0] is not None and ctx.steered:
            h.diagnostics.fired("R37")
            th, order = td
            nxt = (ctx.f - ctx.frac) * float(np.clip(
                (h.contraction_target / max(th, 1e-12)) ** (1.0 / order), 0.5, 2.0))
        return min(nxt, 0.25)


def default_pipelines():
    """The path step's pipelines, in the order the solver has always used."""
    return {
        PREDICT: Pipeline(PREDICT, [QuadraticPredictor(), SecantPredictor(), LastSolution()]),
        STEP_FAILED_SIZE: Pipeline(STEP_FAILED_SIZE, [LocatedBirth(), ContractionCut(), Halve()]),
        STEP_ACCEPTED_SIZE: Pipeline(STEP_ACCEPTED_SIZE, [Grow()]),
    }


# -- settle: the monitors of a solved step ----------------------------------------

SETTLE_START = "settle: start"
SETTLE = "settle"


@dataclass
class Reject:
    """The step is not taken; ``why`` is the settle's verdict (``state["why"]``)."""

    why: str


@dataclass
class Accept:
    """The solution is consistent."""


@dataclass
class Repair:
    """Structures to try, in order; ``optional``: if none solves, the step stands."""

    options: list
    optional: bool = False


@dataclass
class SettleContext:
    """What a settle monitor sees: the step's solution ``(sh, r)`` at ``(le, off)``,
    the route's ``state``, and the settle's own memory across its rounds."""

    homotopy: Any
    state: dict
    sh: Any
    r: Any
    le: float
    off: Any
    changed: bool
    scale: Any
    tried_splits: dict
    key: Any
    history: list = field(default_factory=list)
    last: Optional[tuple] = None
    reorders: int = 0
    crossed: bool = False


class SplitLocation:
    """Rule R25 (V2, with V2'): the step crossed a split, an arc's density turned
    negative inside it; a shorter step, to the event, before any repair."""

    name, hook, fires = "R25", SETTLE_START, False   # counted where the step is cut

    def act(self, ctx):
        h, state = ctx.homotopy, ctx.state
        if ctx.changed:
            return None
        theta = h._split_fraction(state["shoot"], state["v"], ctx.sh, ctx.r.v,
                                  too_far=h.locate_tol)
        if theta is None:
            return None
        state["split_at"] = theta
        return Reject("density negative")


class OrderMonitor:
    """Rule R30 (V1's order monitor), with R38 (coincident junctions, either order)
    and R39 (a reorder that crosses back is one junction: the step again, tied)."""

    name, hook, fires = None, SETTLE, False

    def act(self, ctx):
        h, sh, r = ctx.homotopy, ctx.sh, ctx.r
        back = h._order_crossing(sh, r.v) if h.watch_order else 0.0
        if 0.0 < back <= h._coincide():
            h.diagnostics.fired("R38")
            back = 0.0
        ctx.crossed = back > 0.0
        if not ctx.crossed:
            return None
        h.diagnostics.fired("R30")
        if back > h.locate_tol and not ctx.changed:
            h._detail(f"    junctions crossed by {back:.1e} of T: a shorter step")
            return Reject("junctions crossed")
        ctx.reorders += 1
        if ctx.reorders > 2:
            h._detail("    the reorder crosses back: not settled")
            if getattr(h, "handover_ties", False):
                h._tie_crossing(sh, r.v)
            return Reject("reorder did not settle")
        return None


class FoldMonitor:
    """Rule R44 (Lemma X): out of order at a fold; the route turns there."""

    name, hook, fires = "R44", SETTLE, True

    def act(self, ctx):
        from .shooting import MultiArcShooting
        h, sh, r, state = ctx.homotopy, ctx.sh, ctx.r, ctx.state
        if (state.get("backward") or not isinstance(sh, MultiArcShooting)
                or sh.in_order(r.v, h._coincide())):
            return None
        fold = h._fold_check(ctx.history, sh, r, ctx.le, ctx.off, state.get("leg_dir"))
        if fold is None or fold["key"] in state.setdefault("folds_seen", set()):
            return None
        state["fold"] = fold
        return Reject("fold (Lemma X)")


class History:
    """Records the round's solution for R44 (always passes)."""

    name, hook, fires = None, SETTLE, False

    def act(self, ctx):
        ctx.history.append((ctx.sh, ctx.r))
        return None


class ViolationMonitor:
    """V1's peak monitor: a limit violated off its arcs. Too far in: a shorter step
    (and R16 locates the birth). After a change of structure, or within the
    tolerance: brought in (``_activate``: births, R5, R6, R12, R26)."""

    name, hook, fires = None, SETTLE, False

    def act(self, ctx):
        h, sh, r = ctx.homotopy, ctx.sh, ctx.r
        if ctx.crossed:
            return None
        viol = h._free_violations(sh, r.v, ctx.scale)
        if not viol:
            return None
        scale = ctx.scale
        worst = max(peak / scale[c] for c, _, _, peak in viol)
        if worst > h.locate_tol and not ctx.changed:
            h._detail("    violation too large: a shorter step ("
                      + ", ".join(f"{h.kr.CONSTRAINT_NAMES[c]} {peak / scale[c]:.1e}"
                                  f" at [{a:.5f}, {b:.5f}]" for c, a, b, peak in viol) + ")")
            over = {}
            for c, _, _, peak in viol:
                over[c] = max(over.get(c, -np.inf), peak / scale[c])
            ctx.state["overshoot"] = over
            return Reject("violation too large")
        arcs_now = h._arcs_of(sh, r.v)[0]
        if ctx.last is not None and ctx.last[0] == arcs_now and 1e-4 > worst > 0.5 * ctx.last[1]:
            # The same structure again, and the violation did not shrink: the path
            # tolerance at a tangential junction, not a missing arc.
            return Accept()
        ctx.last = (arcs_now, worst)
        return Repair(h._activate(sh, r.v, viol))


class DeadArcMonitor:
    """Arcs that are dead (``_retire``: the density and width monitors), then R18's
    split of a rising multiplier (at most three times per structure; optional)."""

    name, hook, fires = None, SETTLE, False

    def act(self, ctx):
        h, sh, r = ctx.homotopy, ctx.sh, ctx.r
        dead = h._retire(sh, r.v)
        if dead is not None:
            return Repair([dead])
        if not h.split_rising:
            return None
        dead = h._split_rising(sh, r.v)
        if dead is None:
            return None
        k = ctx.key(*dead[0])
        if ctx.tried_splits.get(k, 0) >= 3:
            return None
        ctx.tried_splits[k] = ctx.tried_splits.get(k, 0) + 1
        return Repair(dead, optional=True)


class BoundMonitor:
    """The bound monitor (controls past their bounds on arcs); nothing left to
    repair: the solution is consistent."""

    name, hook, fires = None, SETTLE, False

    def act(self, ctx):
        options = ctx.homotopy._bound_options(ctx.sh, ctx.r.v)
        return Accept() if options is None else Repair(options)


def settle_pipelines():
    """The settle's monitors, in the order the solver has always used."""
    return {
        SETTLE_START: Pipeline(SETTLE_START, [SplitLocation()]),
        SETTLE: Pipeline(SETTLE, [OrderMonitor(), FoldMonitor(), History(), ViolationMonitor(),
                                  DeadArcMonitor(), BoundMonitor()]),
    }


# -- step failed: split and rescues -------------------------------------------------

STEP_FAILED_SPLIT = "step failed: split"
STEP_FAILED_RESCUE = "step failed: rescue"


@dataclass
class Retry:
    """The step again, with step ``ds``."""

    ds: float


@dataclass
class Rescued:
    """A solution elsewhere on the leg: ``done = (shoot, sol)`` at ``f`` (``le, off``)."""

    done: tuple
    f: float
    le: float
    off: Any


@dataclass
class Turn:
    """Rule R44: the curve turns back at a fold (``advance`` follows it)."""


def _point(ctx, s):
    le0, off0, le1, off1 = ctx.leg
    return le0 + s * (le1 - le0), off0 + s * (np.asarray(off1) - off0)


class SplitStep:
    """Rule R25: the settle located the split inside the step; the step ends there."""

    name, hook, fires = None, STEP_FAILED_SPLIT, False   # R25's switch is checked by the settle

    def act(self, ctx):
        if ctx.outcome != "settle failed: density negative":
            return None
        ctx.homotopy.diagnostics.fired("R25")
        ctx.homotopy._detail(f"    split located at {ctx.state['split_at']:.2f} of the step")
        return Retry((ctx.f - ctx.frac) * ctx.state["split_at"])


class ArcsToTheEnds:
    """At the leg's end with the offsets at zero: arcs that reach the ends are run out
    to them (``_end_structure``), once per structure."""

    name, hook, fires = None, STEP_FAILED_RESCUE, False

    def act(self, ctx):
        h = ctx.homotopy
        key_ = tuple(map(tuple, ctx.arcs))
        if ctx.f < 1.0 or np.any(ctx.leg[3]) or key_ in ctx.ended:
            return None
        ctx.ended.add(key_)
        converted = h._end_structure(ctx.sh0, ctx.v0)
        if converted is None:
            return None
        le, off = _point(ctx, ctx.f)
        from .homotopy import _FAILURES
        try:
            sh, r = ctx.solve_structure(*converted, ctx.v0[:ctx.sh0.n_base], le, off)
            h._detail(f"  arcs run out to the ends: residual {r.residual_norm:.1e}")
            done = ctx.settle(sh, r, le, off) if r.success else None
        except _FAILURES as exc:
            h._detail(f"  arcs run out to the ends: {exc}")
            done = None
        return None if done is None else Rescued(done, ctx.f, le, off)


class DeathSteps:
    """Rule R17 (Lemma E): when a step fails while an arc shrinks, extrapolate w^2
    (linear in s) to the arc's death and step past it without the arc."""

    name, hook, fires = "R17", STEP_FAILED_RESCUE, True

    def act(self, ctx):
        h, prev = ctx.homotopy, ctx.prev
        if prev is None or prev[2] != ctx.arcs or ctx.theta_far is not None:
            return None
        from .homotopy import _FAILURES
        for drop, s_star in h._death_points(ctx.sh0, prev[1], ctx.v0, prev[0], ctx.frac, ctx.f):
            fd = min(s_star + 0.25 * (s_star - ctx.frac), ctx.f)
            le_d, off_d = _point(ctx, fd)
            arcs_d, vals_d = h._arcs_of(ctx.sh0, ctx.v0)
            arcs_d = arcs_d[:drop] + arcs_d[drop + 1:]
            vals_d = vals_d[:drop] + vals_d[drop + 1:]
            try:
                sh, r = ctx.solve_structure(arcs_d, vals_d, ctx.v0[:ctx.sh0.n_base], le_d, off_d)
                done = ctx.settle(sh, r, le_d, off_d) if r.success else None
            except _FAILURES:
                done = None
            h._detail(f"    arc {drop} dies at {s_star:.6f}: "
                      + ("stepped past it" if done is not None else "no"))
            if done is not None:
                return Rescued(done, fd, le_d, off_d)
        return None


class WidthSteps:
    """Rule R2 (Lemma E): steps in s stall at a birth or death; a step in the young
    arc's width is regular there."""

    name, hook, fires = "R2", STEP_FAILED_RESCUE, False

    def act(self, ctx):
        h = ctx.homotopy
        if not ctx.ds < 1e-3:
            return None
        from .homotopy import _FAILURES
        point = lambda sv: _point(ctx, sv)
        for vw, fw, drop in h._width_moves(ctx.sh0, ctx.v0, ctx.frac, point, ctx.path_tol):
            le_w, off_w = point(fw)
            try:
                arcs_w, vals_w = h._arcs_of(ctx.sh0, vw)
                if drop is not None:
                    arcs_w = arcs_w[:drop] + arcs_w[drop + 1:]
                    vals_w = vals_w[:drop] + vals_w[drop + 1:]
                sh, r = ctx.solve_structure(arcs_w, vals_w, vw[:ctx.sh0.n_base], le_w, off_w)
                done = ctx.settle(sh, r, le_w, off_w) if r.success else None
            except _FAILURES:
                done = None
            if done is not None:
                return Rescued(done, fw, le_w, off_w)
        return None


class TieRetry:
    """Rule R39: a reorder that crossed back left a tie; the same step, with it."""

    name, hook, fires = "R39", STEP_FAILED_RESCUE, False

    def act(self, ctx):
        return Retry(ctx.ds) if ctx.homotopy._tie_locked(ctx.sh0, ctx.v0) else None


class FoldTurn:
    """Rule R44 (Lemma X): a fold; the solution curve goes on along the other order's
    branch with the step decreasing. ``advance`` follows it; this decides."""

    name, hook, fires = None, STEP_FAILED_RESCUE, False   # the fold is taken from state always

    def act(self, ctx):
        h, state = ctx.homotopy, ctx.state
        fold = state.pop("fold", None)
        if fold is None or state.setdefault("fold_count", 0) >= h.fold_turn_limit:
            return None
        state["fold_count"] += 1
        state.setdefault("folds_seen", set()).add(fold["key"])
        le0, off0, le1, off1 = ctx.leg
        turned = h._turn_back(fold, ctx.frac, ctx.f, le0, off0, le1, off1,
                              ctx.solve_structure, ctx.settle, state)
        return Turn() if turned else None


def failure_pipelines():
    """The failed step's pipelines, in the order the solver has always used."""
    return {
        STEP_FAILED_SPLIT: Pipeline(STEP_FAILED_SPLIT, [SplitStep()]),
        STEP_FAILED_RESCUE: Pipeline(STEP_FAILED_RESCUE, [ArcsToTheEnds(), DeathSteps(),
                                                          WidthSteps(), TieRetry(), FoldTurn()]),
    }


# -- exact rejected -------------------------------------------------------------------

EXACT_REJECTED = "exact rejected"


@dataclass
class ExactContext:
    """The route's last solution ``(sh, v)``, whose exact solve was rejected; ``depth``
    of R21's recursion; ``split_exact`` the route's R21/R32 step (a closure)."""

    homotopy: Any
    state: dict
    sh: Any
    v: Any
    tol: float
    depth: int
    split_exact: Any


class CollapsedTouches:
    """Rule R20 (Lemma Q): an order-1 touch stands in for an arc too short to solve on
    the route; the exact problem gets the arc, centred on it with the touch's jump.
    Returns ``(hit, arcs)``."""

    name, hook, fires = None, EXACT_REJECTED, False   # counted as "R20 exact" below

    def act(self, ctx):
        h, sh, v = ctx.homotopy, ctx.sh, ctx.v
        if not h.collapsed_touches:
            return None
        for arcs_x, values_x in h._touch_arcs(sh, v):
            entries_x, vals_x = h._mas_entries(arcs_x, values_x)
            v_x = h._smoothed(entries_x, None, ctx.state["off"]).pack_arcs(v[:sh.n_base], vals_x)
            hit = h._exact_jump(entries_x, v_x, ctx.tol)
            if hit is not None:
                h.diagnostics.fired("R20 exact")
                return hit, arcs_x
        return None


class ExactSplitsAndBirths:
    """Rules R21 (Lemma H: the rejected solution split where its multiplier rises,
    settled on the route, and tried again) and R32 (the violation the exact check
    sees, brought in as births). Returns ``(hit, None)``."""

    name, hook, fires = None, EXACT_REJECTED, False

    def act(self, ctx):
        h = ctx.homotopy
        if ctx.depth >= 2 or not (h.exact_splits and h.split_rising or h.exact_births):
            return None
        hit = ctx.split_exact(ctx.depth)
        return None if hit is None else (hit, None)


def exact_pipelines():
    return {EXACT_REJECTED: Pipeline(EXACT_REJECTED, [CollapsedTouches(), ExactSplitsAndBirths()])}
