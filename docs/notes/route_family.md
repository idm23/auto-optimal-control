# The route as a one-parameter family

This note sets out the mathematics for replacing `settle`'s search with an
event-driven path follower. The lemmas are entered in the register as
V1 to V5. The letter F was taken (time reversal), so the family lemmas are V.

**Why.** What the solver *accepts* is already pinned down: `_verdict` checks
the necessary conditions. *How* it gets there is not. Phase 0 traced the
current route on all 45 census cases, one route each, with no portfolio.
It found:

- 4,321 accepted points, 88% of them with no structure change;
- 506 structure changes, 92% of them a single event;
- 63% of the 28,328 path solves spent inside `settle`, and 945 accepted
  points repaired with no change at all (activate, then retire, at the
  same point);
- 28% of the accepted points on one-limit-at-a-time detours;
- goal 4 at 70%: the route reaches zero offsets cleanly, and the exact
  solve is rejected by Lemma H. The path passed an event it doesn't watch.

The event picture fits most of the route. The cost and the failures come
from deciding near-events by thresholds and search.

## The family

**Knobs.** The route parameters are $\theta = (o, \log w, \log \varepsilon)$:
the limit offsets $o \in \mathbb{R}^{n_c}$ (each limit $g_c \le 0$ is
solved as $g_c \le o_c$), the secondary weight, and the bang-bang smoothing.

**A leg** is a segment $\theta(s) = \theta_a + s(\theta_b - \theta_a)$,
$s \in [0, 1]$, fixed when the leg starts. For the arm:
- the first leg is $o(s) = (1 - s)\, o_0$ with R1's random $o_0$ above the
  peaks, at fixed $w$;
- R24 adds a second leg, $o = 0$ and $\log w$ from $\log 10w$ to $\log w$;
- R13's one-limit-at-a-time order is a fixed chain of legs.

A route is a finite chain of legs. A new leg may start only at a consistent
point, with its direction fixed (and randomized) when it starts.

**Structure.** $S$ is the ordered list of arcs of order-1 limits, a touch
being an arc of zero width (the convention of Bonnans & Hermant 2008).
With $S$ fixed, the shooting system $\Psi_S(v, s) = 0$ is square in the
unknowns $v$ (initial costate, $t_f$, and per arc $t_{\rm in}$, $t_{\rm out}$,
$\pi$). The object of study is the solution set
$$\Gamma = \{(v, S, s) : \Psi_S(v, s) = 0,\ \text{all monitors of } S \ge 0\}.$$

## Monitors

The equalities are in $\Psi_S$. Every inequality the acceptance needs is a
**monitor**, a scalar $m(v, s)$ that must stay $\ge 0$; its zero is an
event.

| Monitor | Definition | Event at zero | Watched today |
|---|---|---|---|
| peak $m^{\rm pk}_{c,k}$ | $-\max(g_c - o_c)$ over the interior of free stretch $k$ | birth | sampled, threshold $10^{-6}$ (R16 on overshoot) |
| width $m^{\rm w}_i$ | $t_{\rm out} - t_{\rm in}$ of arc $i$ | death | R10, R17 |
| jump $m^{\pi}_i$ | entry jump $\pi_i$ (touch jump $\nu$) | collapse with a jump | `_retire`, acceptance |
| density $m^{\eta}_i$ | $\min$ over arc $i$ of the measure's density $\eta_i$ | split | R25 locates a crossing inside a step; R5 and R18 split where the step lands; R21 at the end |
| gap $m^{\rm g}_{c,k}$ | gap between consecutive arcs of limit $c$ | merge | R11, on touch or overlap |
| order $m^{\rm o}_j$ | difference of consecutive junction times of different limits | reorder | acceptance ("out of order") |
| bound $m^{\rm b}_i$ | how far the riding control on arc $i$ stays inside $U$ (Lemma I's margin bounds it) | corner | riding failures, R8, R9, R14 |
| end $m^{\rm e}_i$ | $t_{\rm in} - t_0$, $t_f - t_{\rm out}$ | arc reaches an end | `_end_structure` |
| held sign | sign margin of held controls | held set wrong | acceptance |

Changes of the held set *on* an arc (Lemma R) and clip kinks are implicit
events, found by the integrator, and not monitors. But Lemma P says $\eta$
jumps at such a moment, so the density monitor can jump negative there
(event E7 below).

## Lemmas

### V1. Acceptance is the monitors — proved, relative to S1

For a fixed $S$ and a solution of $\Psi_S(v, 1) = 0$, `_verdict` accepts
exactly when every monitor is $\ge 0$ (to its tolerance).

*Proof.* Check by check:
- "limit violated off its arcs": peaks, plus the bound monitor on arcs
  (`_violations` raises it to the riding control's bound violation);
- "junctions out of order": width and order;
- "touch jump negative": jump;
- "mu negative" and "mu rises": by Lemma H, $\mu(t) = \int_t^{t_{\rm out}} \eta$
  with $\mu(t_{\rm out}) = 0$, so $\mu \ge 0$ and $\dot\mu \le 0$ hold together
  iff $\eta \ge 0$ on the arc;
- "held control sign": the held-sign monitor;
- "Legendre": empty for the arm, whose Hamiltonian is strongly convex in the
  jerks ($w > 0$). A monitor for singular arcs. $\square$

*So what.* The acceptance test can be run continuously along the path, not
only at the end. Before R25, R5 already looked at the density on every
consistent point, but only to offer a split wherever the step landed.

*Correction.* An earlier version of this note said goal 4 at 70% was a
density crossing that only the final check saw. The trace says otherwise. On
its first route, the density is positive all the way, and the failure is a
birth just beside an exit, brought in on the spot because it is under
`settle`'s threshold (Rule R25 in `structure_certificates.md`).

### V2. Continuation with a fixed structure — proved

Let $\Psi_S(v^*, s^*) = 0$ with $D_v\Psi_S$ nonsingular, and the implicit
events of the flow (clip kinks, held-set changes) transversal crossings.
Then:
1. there is a unique $C^1$ branch $v(s)$ near $s^*$ (S2, with S3 and Lemma R
   for the smoothness of $\Psi_S$ through transversal implicit events);
2. every monitor is continuous along it, and $C^1$ where its extremizer is
   unique, interior and nondegenerate (Lemma N's argument, S7);
3. on any interval where all monitors are $> 0$, the branch is a curve of
   accepted extremals of $P(s)$ (V1). No change of structure is needed
   there.

*So what.* Between events `settle` has nothing to do. Phase 0's 945
repairs-without-change are near-zero monitors decided by thresholds; with
monitors they are either nothing (all monitors positive) or a located event.

### V3. Events are isolated, one at a time — sketch

*Hypothesis (T).* For each structure $S$ and each monitor $m$ (and each
pair $m, m'$), the maps $(v, s, o_0) \mapsto (\Psi_S, m)$ and
$(\Psi_S, m, m')$ are transversal to zero.

*Statement.* Under (T), for almost every draw $o_0$:
- each event on the leg is a single monitor crossing zero with a nonzero
  derivative along the curve;
- no two events coincide;
- $D_v\Psi_S$ is singular only at simple folds.

*Proof sketch.* The parametric Sard theorem (S8; the transversality theorem
of Chow, Mallet-Paret & Yorke 1978): the draws for which the leg meets a
stratum of codimension 2 form a null set.

*Gap.* (T) is checked only for births: the peak excess is
$\max g_c - o_c$, so $\partial/\partial o_{0,c} = -(1 - s) \ne 0$. That is
Lemma D's mechanism. For the other monitors, the draw enters only through
the motion, and (T) is the same open hypothesis as in Lemma D.

*What randomizing cannot remove.* Events of codimension 1 are generic: one
monitor at zero, a simple fold, a limit's own control saturating. Every
route will meet them, so each needs an entry in the table (V4). In the
finite-dimensional model, the generic events of one-parameter families are
the five types of Jongen, Jonker & Twilt (1986):

| JJT type | Here |
|---|---|
| 1, nondegenerate critical point | the interior of a piece |
| 2, strict complementarity fails | a monitor at zero: birth, death, split, merge, jump at zero |
| 3, degenerate Hessian on the tangent space | a fold: $D_v\Psi_S$ singular |
| 4 and 5, constraint qualification fails | constraint regularity fails: Bonnans–Hermant's (A3), $\lvert g_u^{(1)} \rvert \ge \beta$; in our class $a_F \to 0$ or the margin $\to 0$ (Lemmas I′, K, L, M, P) |

For optimal control this is an analogy, not a theorem.

### V4. The event table (order 1) — partly proved

Each row gives an event, its local picture and the unique continuation of
the curve through it.

| | Event | Monitor | Continuation | Status | Rules today |
|---|---|---|---|---|---|
| E1 | birth: an interior peak reaches the limit with $\ddot g < 0$, jerks interior | peak | a new arc; its width grows as $\sqrt{s - s^*}$, so the width is the step variable there | **proved** (Lemma E; compare Bonnans & Hermant 2008, Thm 2.4: a touch becomes an arc, stays a touch or goes inactive) | R2, R16, R20, R26 |
| E1′ | birth at the corner (contact on the edge of the holdable set) | peak | one-sided arc | conjecture (E′) | R6, R12 |
| E2 | death: an arc's width and jump reach zero together | width | the arc is removed: E1 traversed backwards | **proved** (Lemma E read in reverse) | R10, R17 |
| E2′ | collapse with a positive jump | width, with $\pi > 0$ | a touch with a jump is degenerate unless the controls absorb it (Lemma Q), so not a regular event | open | R20 |
| E3 | split: the density of an arc reaches zero inside it, at a nondegenerate minimum, controls interior | density | a free gap $[t_m - h, t_m + h]$ is born inside the arc, with $h^2 = -6\eta_{\min}/\ddot\eta$ | **leading order derived** (Lemma E3; Bonnans–Hermant's (A5) excludes it); the implicit-function step is a sketch; confirmed on the census's clean splits | R5, R18, R21, R25, R27 |
| E4 | merge: the gap between two arcs of a limit closes | gap | the free stretch between them dies: E3 backwards | as E3, read backwards | R11 |
| E5 | reorder: junctions of two limits coincide | order | the swapped order; no choice | sketch: needs Lemma B for both limits ridden at once, and the locality of Lemma J | none |
| E6 | an arc reaches $t_0$ or $t_f$ | end | the arc is ridden from the start or to the end | open (outside Bonnans–Hermant's (A1), (A4)) | `_end_structure` |
| E7 | constraint regularity fails: the margin reaches zero, or a ridden limit's own control saturates (the density jumps, Lemma P) | bound, density | not known | **open**, the frontier (Lemmas I′, K, L, M, P; Observation O2) | R6, R8, R9, R12, R14 |
| E8 | fold: $D_v\Psi_S$ singular with all monitors positive | conditioning | pass it by arclength; $s$ turns back | standard for a simple fold (S9) | none: ds halves until it stalls |
| E9 | anything else: two monitors at once, zero slope | | none: stop and report | probability zero under V3 | |

Phase 0 counts, over the 506 changes: 383 births, 135 deaths, 62 reorders,
21 merges, 6 splits, and 7 changes of several arcs of one limit.

### V5. A leg ends at $s = 1$, or fails visibly — sketch

*Statement.* Suppose:
- (a) the leg starts at a regular point (a unique extremal of $P(0)$ with
  $D_v\Psi_S$ nonsingular and every monitor $> 0$);
- (b) every event the curve meets is in the table with a proved
  continuation;
- (c) the curve stays in a bounded set of $(v, s)$.

Then the curve from the start reaches $s = 1$ after finitely many events.

*Proof sketch.* By V2 and (b), near the curve $\Gamma$ is a piecewise
$C^1$ one-dimensional manifold. Its only boundary points are at $s = 0$ and
$s = 1$, since each event has exactly one continuation and each fold is
simple. By (c), the component through the start is compact, so it is an
arc or a circle. It has a boundary point at the start, so it is an arc, and
its other end is at $s = 0$ or $s = 1$. It is not at $s = 0$, because the
start is the only solution there. Events are isolated (V3) on a compact
arc, so there are finitely many. This is the argument of Chow,
Mallet-Paret & Yorke for a smooth homotopy. Each event joins two pieces
with one continuation, so it carries over.

*Gaps.*
- **(b) is incomplete.** E3, E4, E6 and E7 have no proved continuation,
  and E5 is a sketch.
- **(c) can fail.** A limit level with no feasible motion sends $t_f$ or
  $\lambda$ to infinity. This cannot be proved away. It must be detected
  (a bound on $\|v\|$) and reported.
- **What happens when a hypothesis fails.** The leg stops at its last
  consistent point, with the event as the reason. A new leg, or R15's next
  route, may start there. The number of legs is bounded by the budget.

## What changes

| Today | With V1 to V5 |
|---|---|
| The path is chosen online: joint, then one by one, then backtracks with banned structures | Legs fixed when they start; a new leg only at a consistent point, for a recorded reason |
| A step jumps with the old structure, then `settle` searches (activate, retire, split, bound options, parallel tries) | Continue while all monitors are positive (V2); the first monitor to cross is located by root-finding |
| The structure change is whatever converges and passes | The table's continuation for that event (V4), with its sign checked |
| Births, deaths and splits handled by repairs after failures (R16, R17, R2, R5, R18, R21) | The same lemmas used as the mechanism, the same for every event |
| Step variable $s$, halved until it stalls | Arclength; the width at births (Lemma E) |
| Failures: "stalled", "settle failed" | A point on the curve and a named event, hypothesis or bound |

Rules that become parts of the follower: R1 (V3), R2, R10, R16, R17 (E1,
E2), R5, R18, R21 (E3), R11 (E4), R13, R24 (legs), R15 (the leg budget).
R6, R8, R9, R12 and R14 are E7's territory, and stay heuristics until E7
has a lemma.

## What phase 2 needs, in order of value

1. **The density monitor on the path** (V1). Done: Rule R25 locates a
   crossing inside a step by a secant on the density (V2).
2. **E3's normal form**: done to leading order (Lemma E3). Most of the arm's splits fall outside its hypothesis H0 and are E7.
3. **E7**: limits ridden while their own controls saturate (rest 12, goal 9
   at 80%).
4. **E5 and E6**, to complete the table for what the census meets.

The follower can be built before the table is complete. An event with no
lemma stops the leg with its name, and the portfolio's next route takes
over. That is the honest version of today's fallbacks.

## References

- J. F. Bonnans, A. Hermant (2008), *Stability and sensitivity analysis for
  optimal control problems with a first-order state constraint and
  application to continuation methods*, ESAIM: COCV 14, 825–863. They assume
  one scalar control with no bounds and one scalar limit, (A2) strong
  convexity of $H$ in $u$, (A3) $\lvert g_u^{(1)} \rvert \ge \beta$ near the
  contact set, (A4) finitely many junctions, (A5) strict complementarity on
  arcs, and (A6) $\ddot g < 0$ at touch points. Under these and a
  second-order condition, nearby extremals have a neighbouring structure:
  arcs persist, and a touch point becomes an arc, stays a touch or goes
  inactive (Thm 2.4, Thm 4.4).
- A. Hermant (2010), *Homotopy algorithm for optimal control problems with
  a second-order state constraint*, Appl. Math. Optim. 61. The same for
  order 2, with scalar control and the strengthened Legendre–Clebsch
  condition.
- H. Th. Jongen, P. Jonker, F. Twilt (1986), *Critical sets in parametric
  optimization*, Math. Programming 34, 333–353.
- S. N. Chow, J. Mallet-Paret, J. A. Yorke (1978), *Finding zeroes of maps:
  homotopy methods that are constructive with probability one*, Math. Comp.
  32, 887–899.
- H. B. Keller (1977), *Numerical solution of bifurcation and nonlinear
  eigenvalue problems*; E. L. Allgower, K. Georg (1990), *Numerical
  Continuation Methods*: pseudo-arclength continuation through simple folds.

The arm lies outside Bonnans–Hermant's hypotheses in three ways: six
controls, twelve limits, and a box on the controls. The first two matter
little where limits are on distinct joints (Lemma B). The box is the third,
and it is exactly E7.
