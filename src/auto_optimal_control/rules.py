"""The solver's rules, each with its specification (``docs/notes/structure_certificates.md``,
"How a rule is specified").

A rule acts at one *hook* of the active-set route, on a *trigger*. Its *soundness*
is what it provably preserves (always: the answers accepted), from the lemma named.
Its *effectiveness* is either a lemma about the solver's own problem (that the path
is regular in some parameter, that Newton converges from the seed) or the word
``"heuristic"``. ``targets`` names the failure class of
:class:`~auto_optimal_control.diagnostics.Diagnostics` the rule should reduce; it
is judged on that class first and on the number of cases solved second.

``switch`` is where the rule is turned on: an attribute of the
:class:`~auto_optimal_control.homotopy.Homotopy` (``"offset_start=staggered"`` for
a value) or a class default of :class:`~auto_optimal_control.shooting.MultiArcShooting`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    name: str
    hook: str
    trigger: str
    action: str
    lemmas: tuple
    effectiveness: str
    targets: str
    switch: str


RULES = (
    Rule("R1", "route start", "every solve", "stagger the offsets' starting values at random",
         ("C", "D"), "Lemma D (births separate with probability one)", "coincident births",
         "offset_start=staggered"),
    Rule("R2", "step stalled", "a step fails with ds < 1e-3 beside a young order-1 arc",
         "step in the arc's width with the route parameter solved for", ("E",),
         "Lemma E (the branch is smooth in the width)", "stalled",
         "width_steps"),
    Rule("R4", "solve failed", "the forward solve is not exact and C6 holds",
         "solve the time-reversed problem and map its answer back", ("F",), "heuristic",
         "not exact", "reversal"),
    Rule("R5", "consistent step", "an order-1 arc's multiplier rises",
         "split the arc around the rise", ("H",), "heuristic", "mu rises (Lemma H)",
         "split_rising"),
    Rule("R6", "newborn arc", "a violation is to be brought in as an order-1 arc",
         "pull the arc's ends in to the holdable stretch about the peak", ("I",), "heuristic",
         "seed failed: riding_failed", "holdable_births"),
    Rule("R7", "junction rows", "every order-1 junction under C2",
         "use dg/dt = 0 under the far side's control", ("J",), "heuristic",
         "converged off the edge", "MultiArcShooting.rate_junctions_default"),
    Rule("R8", "seed failed", "a seed's riding fails partway",
         "pull the arc's exit in to the edge", ("I", "J"), "heuristic",
         "seed failed: riding_failed", "edge_exits"),
    Rule("R10", "evaluation", "a segment ends before it starts (an arc's width < 0)",
         "integrate it backward instead of failing", ("E",),
         "Lemma E, step 1 (the rows are smooth in the width through zero)", "negative_width",
         "MultiArcShooting.backward_default"),
    Rule("R10b", "evaluation", "riding backward through a negative-width arc fails",
         "continue it on the free law (continuous at zero width)", ("E",),
         "continuity is enough for Newton to step back; never an answer", "riding_failed backward",
         "MultiArcShooting.backward_fallback_default"),
    Rule("R11", "structure", "two arcs of one limit touch or overlap",
         "merge them into one arc whose jump carries both", (), "by construction",
         "'entered twice' failures", "merge_touching"),
    Rule("R12", "newborn arc", "a violation's peak lies on the edge of the holdable set",
         "bring the limit in as an order-1 touch point, not an arc", ("I'", "L"), "heuristic",
         "settle failed: no structure converged", "corner_touches"),
    Rule("R13", "route start", "every solve",
         "bring the limits in one at a time, the most violated first", ("I'",), "heuristic",
         "seed failed: riding_failed", "offset_order=sequential"),
    Rule("R14", "riding", "the riding problem is infeasible only by rounding (margin ~ 0)",
         "ride the corner, with mu from the last interior control", ("I", "B"),
         "continuity from the holdable side", "riding_failed", "always"),
    Rule("R15", "route failed", "the active-set route fails",
         "try the next route of a portfolio in the time left", ("D",), "heuristic",
         "not exact", "portfolio"),
    Rule("R16", "step failed", "a converged step overshoots a birth (violation too large)",
         "shorten the step to the birth by a secant on the peak excess", ("N",),
         "Lemma N (the peak excess is C^1 in the route parameter before a birth)",
         "settle failed: violation too large", "locate_births"),
    Rule("R17", "step failed", "a step fails while a finite order-1 arc shrinks",
         "extrapolate w^2 to the arc's death and step past it without the arc", ("E",),
         "Lemma E (w^2 is C^1 in s with a nonzero slope at a death)", "not converged",
         "locate_deaths"),
    Rule("R18", "consistent step", "an order-1 arc's multiplier rises from its entry or to its exit",
         "trim the arc to where the rise ends (starts)", ("H",), "heuristic",
         "mu rises (Lemma H)", "trim_rising"),
    Rule("R20", "structure", "an order-1 arc collapses with a positive jump, or an order-1 limit is born",
         "carry it as an order-1 touch point (offered after the arc at a birth)", ("E", "Q"),
         "a stand-in on the route only; it grows back into an arc beside a violation",
         "settle failed: no structure converged", "collapsed_touches"),
    Rule("R21", "exact rejected", "an exact solution's order-1 multiplier rises (Lemma H)",
         "split the arc there, settle on the route at the end of the offsets, solve again",
         ("H",), "the split's own exact solve is judged as any other",
         "rejected: mu rises (Lemma H)", "exact_splits"),
    Rule("R22", "evaluation", "a boundary arc ridden at run time changes its held set",
         "stop the step there and flip the held set; a fixed set rides by one linear solve",
         ("R", "B"), "Lemma R (the flow is continuous there, so the STM carries through)",
         "riding cost (evaluations per arc)", "MultiArcShooting.held_events_default"),
    Rule("R23", "evaluation", "a free arc of a problem certified as integrator chains",
         "evaluate it, its STM and its cost in closed form instead of integrating",
         ("T",), "Lemma T (the closed form is the flow; the STM is exact)",
         "free-arc cost (integration time)", "MultiArcShooting.closed_free_default"),
    Rule("R60", "evaluation", "a boundary arc whose ridden limits each bound one state of an integrator chain (a rate or a position limit of the jerk chain), of a problem certified for R23",
         "evaluate it, its STM and its cost in closed form: the free flow with those chains' controls held at 0",
         ("T", "T''"), "Lemma T'' (riding g = a s - b holds the chain's control at 0 and adds nothing to the costates' equations, so the arc is the free flow with that control fixed; the STM is exact)",
         "riding cost (integration time)", "MultiArcShooting.pinned_riding"),
    Rule("R24", "route start", "the policy's secondary cost has a numeric weight w",
         "bring the limits in at k w, then walk the weight down to w along the route",
         ("W", "N"), "Lemma W (the flow's curvature is O(1/w), so Newton's radius is O(w))",
         "not converged", "weight_start=10"),
    Rule("R25", "consistent step", "an order-1 arc's density turns negative inside a step",
         "shorten the step to the split by a secant on the density", ("H", "V"),
         "V2 (the density at a fixed time is C^1 in the route parameter on a fixed structure)",
         "rejected: mu rises (Lemma H)", "locate_splits"),
    Rule("R26", "newborn arc", "a violation lies near an arc of its limit, with the limit strictly below its level between them",
         "bring it in as a new arc (a birth), not by widening the arc over it", ("E", "V"),
         "Lemma E (a birth is a new arc at an interior peak; widening covers a free stretch)",
         "rejected: mu rises (Lemma H)", "separate_births"),
    Rule("R27", "consistent step", "an order-1 arc's multiplier rises inside it (R5 fires)",
         "offer first the gap of the moment conditions: mu(a) = mu(b), int_a^b D (mu - mu(a)) = 0", ("H", "E3", "P"),
         "Lemma E3 (the released motion returns to the limit tangentially)",
         "rejected: mu rises (Lemma H)", "split_normal_form"),
    Rule("R28", "consistent step", "a split R5 or R27 offers solves",
         "take it only if the limit's multiplier no longer rises and no piece or gap has zero width", ("H", "V"),
         "V4 (a transition continues the event only if it restores the event's monitor)",
         "rejected: mu rises (Lemma H)", "split_must_resolve"),
    Rule("R29", "consistent step", "an order-1 arc's multiplier starts to rise where the held set changes in a control that moves the limit's rate (joining or leaving)",
         "offer first the piece leaving just before that moment (a gap to the rise's end, or a trim)", ("U", "P", "J"),
         "Lemma U (riding through violates H; an exit there needs mu = 0)",
         "rejected: mu rises (Lemma H)", "exit_before_saturation"),
    Rule("R30", "consistent step", "a step's solution has junctions out of the structure's order",
         "not accepted: a crossing above locate_tol of T takes a shorter step; a smaller one is re-sorted (E5), at most twice", ("V",),
         "V1 (acceptance is the monitors; order is one of them)",
         "rejected: junctions out of order", "watch_order"),
    Rule("R31", "consistent step", "an order-1 arc's entry multiplier pi is negative",
         "retire the arc", ("H", "V"),
         "Lemma H (0 <= mu <= pi on the arc: at pi = 0 the arc carries no multiplier, and goes without moving the motion)",
         "rejected: mu negative (Lemma H)", "retire_negative"),
    Rule("R33", "consistent step", "R5 finds more than one order-1 arc whose multiplier rises",
         "split the arc whose rise is largest relative to its multiplier's largest value on the arc", ("H", "P"),
         "scale invariance (g_c -> k g_c sends mu_c -> mu_c / k); Lemma P (a rise induced at another arc's held-set change)",
         "rejected: mu rises (Lemma H)", "relative_rise"),
    Rule("R34", "consistent step", "R5 or R27 splits an arc",
         "also offer it seeded by continuity (the first piece keeps the entry jump), with the gap wider by half and twice the rise's stretch on each side", ("E3",),
         "seed only (Lemma E3's gap contains the window; acceptance judges the result)",
         "rejected: mu rises (Lemma H)", "wide_gaps"),
    Rule("R35", "consistent step", "an order-2 touch point where the limit is convex (g'' > 0)",
         "the touch becomes a short arc about it, its jump the arc's entry jump; offered first when it is violated beside", ("E",),
         "a touch is a local maximum of g, so g'' <= 0 there with a continuous control; Bonnans & Hermant 2008 (a touch becomes an arc)",
         "rejected: limit violated off its arcs", "touch_curvature"),
    Rule("R36", "consistent step", "an order-2 arc's multiplier rises (and acceptance at order 2)",
         "split the arc around the rise, the second piece entering with the jump that continues the costate; a rise from the entry or to the exit trims it", ("H2",),
         "Lemma H2 (on an order-2 arc mu is convex and ends at 0 with slope <= 0, so it does not rise)",
         "rejected: mu negative (Lemma H)", "order2_rise"),
    Rule("R37", "path step", "Newton's first contraction theta from the step's predictor (the chord's first step)",
         "theta > 1/2: end the solve and cut the step to h (1/4 / theta)^(1/p); after a success, the next step is h (1/4 / theta)^(1/p) (clipped), p = 2 from the secant predictor, 1 without", ("V",),
         "steers only (Deuflhard 2004, adaptive pathfollowing: theta estimates half the Kantorovich quantity omega ||dv0||, which grows like h^p); acceptance is unchanged. Tried: the census page falls from 69 to 67, since the route's residual is only piecewise smooth (held-set events) and a kink reads as distance",
         "not converged", "contraction_steps"),
    Rule("R41", "exact rejected", "the exact solution converges with junctions of different limits crossed (past the coincidence tolerance)",
         "sort the events by their times and solve the exact problem again from that solution (twice at most)", ("D",),
         "E5 (two limits' junctions cross transversally: the swapped order is the structure beyond the crossing); acceptance judges the result",
         "rejected: junctions out of order", "exact_reorders"),
    Rule("R43", "junction rows", "a junction of an order-2 arc (where H continuity is the row)",
         "the multiplier rows of Lemma H2: mu(t_out-) = 0 at the exit, mu(t_in+) = pi_2 at the entry", ("H2", "J"),
         "Lemma H2 (both rows are necessary where lambda is continuous at the exit); H continuity is quadratic in the control's mismatch at a tangential junction (Lemma J), a double root, where these are simple",
         "not converged", "MultiArcShooting.order2_rows_default"),
    Rule("R44", "consistent step", "a crossed pair of events of different limits whose two orders both come out crossed at one step, at a regular point (interior riding controls, B's rank, J nonsingular) with gap rates of opposite signs",
         "turn back on the other order's branch at the last accepted point, follow it backward until the structure changes, then go on to the target", ("X",),
         "steers only: Lemma X (iii), a fold, where the solution curve continues with the step decreasing; arclength continuation (Allgower and Georg)",
         "settle failed: structure changes did not settle", "fold_turns"),
    Rule("R45", "route start", "R24 starts the route at a larger secondary weight",
         "lower the weight together with the offsets (the diagonal of the plane), not after them", ("W",),
         "steers only: Lemma W (3), every point of the plane is a real problem of the family",
         "not exact", "weight_path=diagonal"),
    Rule("R46", "step failed", "a path step fails for want of convergence (not a located birth or split)",
         "the next three successful steps grow to at most 0.9 of the failed one, then probe larger again", ("S",),
         "steers only: Lemma S (a controller growing by g and halving fails ln g / ln 2g of its steps where the largest convergent step is constant)",
         "not converged", "step_memory"),
    Rule("R47", "route failed", "R24's walk of the weight stalls within walk_exact_gap decades of the policy's weight",
         "solve the exact problem from the walk's last solution", ("W",),
         "steers only: Lemma W (1), the exact problem is the same family's; acceptance judges it",
         "not exact", "walk_exact"),
    Rule("R48", "consistent step", "on a leg that moves the weight, the last three path solutions have one structure",
         "predict the next by the quadratic through them, before the secant", ("S", "W"),
         "steers only: Lemma S (a predictor of order p allows steps of (rho / C)^(1/p) under Newton's radius rho), and W (2) (rho shrinks with the weight)",
         "not converged", "quadratic_predictor"),
    Rule("R54", "settle failed: density negative", "R25 located a split, and the shorter step's end density at the same time is not smaller by half",
         "locate no more: the density jumps at the base, and R5 splits from there", ("P", "V2"),
         "certified: P (the density jumps where a held-set change sweeps through t*), so V2's continuity of eta(t*) in s fails there; under continuity the located step's end density is a tenth of the last, under a jump it is unchanged",
         "settle failed", "density_jumps"),
    Rule("R55", "not converged", "a chord step of a path solve contracts Newton's correction (natural monotonicity) but not the residual's norm",
         "accept it: the chord goes on, where the residual test would backtrack and hand the solve to MINPACK", ("AC",),
         "steers only: Lemma AC (Newton's iterates are affine covariant, the natural norm ||J^-1 R|| is invariant under a rescaling of the equations and the residual's norm is not; ||R(v1)|| <= cond(J) theta ||R(v0)||); every solve still ends at the path tolerance in the residual",
         "not converged", "natural_monotonicity"),
    Rule("R61", "not converged", "a path solve of the structure just solved",
         "start the chord from the last solve's Jacobian; the chord's own tests refresh it (a contraction above 1/4, or none)", ("AC",),
         "steers only: Lemma AC (4), a chord whose Jacobian is off by dJ contracts by about ||J^-1 dJ||, O(omega ds) along a path, and the natural monotonicity test measures it; every solve still ends at the path tolerance in the residual",
         "not converged", "carry_jacobian"),
    Rule("R63", "not converged", "the first Newton step of a path solve (fresh Jacobian) contracts by Theta > 12",
         "end the solve there, without backtracking", ("AC",),
         "steers only: Lemma AC (5), a damping lambda passes the restricted monotonicity test only if lambda <= 3/(2h) with h >= 2 Theta, so none of the chord's dampings (down to 1/16) can; the solve fails as it would have, without the Jacobians of its backtracking",
         "not converged", "hopeless_newton"),
    Rule("R64", "step accepted: cap", "after two accepted steps of one structure, a limit's margin (its largest g off its arcs, in its scale) is negative and rising",
         "cap the next step to end at the secant's zero of the margin plus half the settle's window (locate_tol)", ("V",),
         "steers only: Lemma V6 (a non-ridden limit's margin is Lipschitz along the branch and C1 where its peak is unique and nondegenerate, so the secant's zero is within O(ds^2) of the birth); a wrong prediction fails as the uncapped step would, and the answers are accepted only by the usual monitors",
         "violation too large", "births_ahead"),
    Rule("R50", "consistent step", "an order-2 arc's width, positive at the base, crosses zero in the step",
         "locate the death by the secant in the width and solve the touch just past it, from the solution interpolated there", ("E''", "Z"),
         "certified: E'' (the width is linear in s near the death, so the secant is exact to O(ds^2)) and Z (at zero width the arc is the touch with jump pi_1); the touch is accepted only by the usual settle",
         "settle failed", "order2_deaths"),
    Rule("R38", "consistent step", "junctions of different limits cross by less than coincide_tol (1e-8) of T",
         "read them as coincident, in order: no reorder (R30), no re-sorting (retire), and acceptance takes them", ("D", "V"),
         "below the junction times' own resolution the two orders give the same motion; symmetry makes such coincidences exact (Lemma D's genericity does not hold there)",
         "reorder did not settle", "coincident_junctions"),
    Rule("R39", "consistent step", "R30's reorder of two junctions of different limits crosses back (neither a touch)",
         "tie them: the earlier event's own row gives way to its time equal to the next one's (a hair apart), wherever the two are adjacent", ("K", "D"),
         "unsound as a rule: the row it drops is a necessary condition (Lemma J's tangency, the multiplier's continuity, or H continuity); only symmetry or Lemma K's corner makes such a coincidence, and it checks for neither. Kept off; acceptance rejects what it makes",
         "reorder did not settle", "handover_ties"),
    Rule("R32", "exact rejected", "the exact solution is rejected for a violation off the arcs that the route's threshold does not see",
         "bring it in as the route does (birth or widening), settle at the end of the offsets, solve exactly again (as R21)", ("E", "V"),
         "Lemma E (a birth is a new arc at the peak); V1 (the exact check's violation monitor)",
         "rejected: limit violated off its arcs", "exact_births"),
    Rule("R9", "evaluation", "a single limit's riding fails before its exit",
         "continue past the edge on the free law", ("K",), "heuristic", "riding_failed",
         "MultiArcShooting.past_edge_default"),
)

BY_NAME = {r.name: r for r in RULES}

#: The portfolio rule R15 turns on: another draw of the offsets and sequential births,
#: and (Rule R24, for a policy with a numeric secondary weight) the first route and these
#: two again, started at 10 w.
DEFAULT_PORTFOLIO = ({"offset_seed": 1}, {"weight_start": 10},
                     {"weight_start": 10, "offset_seed": 1}, {"weight_start": 100},
                     {"weight_start": 100, "weight_path": "diagonal"},
                     {"weight_start": 300, "weight_path": "diagonal"})


def enabled(h):
    """Which rules are on for Homotopy ``h``: ``{name: bool}``."""
    from .shooting import MultiArcShooting

    out = {}
    for r in RULES:
        if r.switch == "always":
            out[r.name] = True
            continue
        name, _, value = r.switch.partition("=")
        if name.startswith("MultiArcShooting."):
            cur = getattr(MultiArcShooting, name.split(".", 1)[1])
        else:
            cur = getattr(h, name)
        out[r.name] = (str(cur) == value) if value else bool(cur)
    return out


def set_rule(h, name, on):
    """Turn rule ``name`` on or off for Homotopy ``h`` (class defaults for the shooting's)."""
    from .shooting import MultiArcShooting

    if BY_NAME[name].switch == "always":
        if not on:
            raise ValueError(f"rule {name} is part of the generated code and cannot be turned off")
        return
    attr, _, value = BY_NAME[name].switch.partition("=")
    off_values = {"offset_start": "together", "offset_order": "joint", "weight_path": "corner"}
    new = (value if on else off_values.get(attr, None)) if value else bool(on)
    if attr == "portfolio":
        new = list(DEFAULT_PORTFOLIO) if on else []
    if attr.startswith("MultiArcShooting."):
        setattr(MultiArcShooting, attr.split(".", 1)[1], new)
    else:
        setattr(h, attr, new)
