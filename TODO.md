# TODO

## Check the analytical results on structure, symbolically, and use the ones that hold

The literature has a number of results that constrain the arc structure of a
problem from its statement alone: which arcs can occur, how many switches,
which junctions, when chattering can happen. AOC already derives everything
these results are stated in (the Hamiltonian, switching functions, the order of
each path constraint, junction conditions), so each one can be checked on the
problem in hand, and when its hypotheses hold, used: to rule structures out
before searching, to bound the number of arcs, or to pick the right junction
rows. The plan for each: (1) state the hypotheses as checks on the SymPy
problem; (2) check them on the examples (single axis, two-link, six-joint, with
and without the jerk cost); (3) compare the result's prediction with what the
solver finds; (4) where it holds, use it in the structure search.

None of these has been read beyond its abstract yet; the statements below are
as the abstracts and citing papers give them and must be confirmed against the
full texts first.

- [ ] **Number of switches for linear systems (Feldbaum).** For a linear
  time-invariant system with real eigenvalues and bounded controls, in minimum
  time, each control switches at most `n - 1` times (Pontryagin et al., *The
  Mathematical Theory of Optimal Processes*, 1962). Check: linearity and the
  eigenvalues of the dynamics, from the problem; for a jerk-limited axis it
  gives at most two jerk switches. Use: a bound on the switches of a free arc.

- [ ] **At least one control saturated (Chen & Desrochers).** In point-to-point
  minimum time for a torque-controlled arm, at least one actuator is saturated
  at every instant; the others are saturated or singular; along a fixed path,
  exactly one.
  [Chen & Desrochers 1990, IEEE T-RA 6(3)](https://data.zhenzhi.com/details/1/1001480499).
  Check: the analogue for jerk control with torque limits, from H being affine
  in the controls (each jerk is at a bound, singular, or set by a torque riding
  its limit), and the order of any singular arc from the switching functions'
  derivatives. Use: a grammar of allowed structures.

- [ ] **Odd-order constraints have no boundary arcs (Jacobson, Lele & Speyer).**
  For a constraint of odd order `p >= 3` in a regular problem (H strictly
  convex in the control), boundary arcs cannot occur: the optimal motion only
  touches the limit.
  [JMAA 35 (1971), doi:10.1016/0022-247X(71)90219-8](https://portal.mardi4nfdi.de/wiki/Publication:2537402);
  [1969 report](https://ntrs.nasa.gov/api/citations/19700002504/downloads/19700002504.pdf).
  Check: joint position limits with jerk control are order 3; with the jerk
  cost (`SecondaryCost`) the problem is regular. Use: position limits as touch
  points, not arcs, under that policy.

- [ ] **Third-order constraints can chatter (Robbins).** Junctions with a
  third-order constraint can accumulate (infinitely many touch points).
  [JOTA 31 (1980) 85-99, doi:10.1007/BF00934790](https://portal.mardi4nfdi.de/wiki/Publication:755119).
  Check: which cost and dynamics the example needs, and whether the
  jerk-cost axis with a position limit shows it.

- [ ] **Time-optimal chains of integrators (Wang, Hu et al.).** Switching laws
  for minimum time with full state constraints and arbitrary terminal states,
  and a proof that chattering occurs only at order 4 with velocity constraints
  among orders `n <= 4`; so a jerk-limited axis with position, velocity and
  acceleration limits does not chatter in minimum time.
  [arXiv 2311.07039 (IEEE TAC 2024)](https://arxiv.org/abs/2311.07039);
  [arXiv 2403.17675 (IEEE TAC 2025)](https://arxiv.org/abs/2403.17675).
  Check: against Robbins and Jacobson-Lele-Speyer, which concern the regular
  case; the two policies (`"as_stated"`, bang-bang jerk, and `SecondaryCost`,
  regular) may sit on opposite sides of this. Use: whether position limits can
  be ridden exactly under `"as_stated"` (the braking-distance reformulation of
  the single axis assumed chattering).

- [ ] **Structural stability of boundary arcs and touch points (Bonnans &
  Hermant).** For first-order constraints, boundary arcs are stable under
  perturbation of the data, and a touch point can persist, vanish or become a
  boundary arc; for second-order constraints, nonreducible touch points can
  turn into a boundary arc or two touch points. Hermant's homotopy detects the
  structure from these transitions.
  [ESAIM: COCV 14 (2008)](https://numdam.org/articles/10.1051/cocv:2008016/);
  [Appl Math Optim 61 (2010) 85](https://link.springer.com/article/10.1007/s00245-009-9076-y);
  [slides](https://www.polyu.edu.hk/ama/events/conference/NPA2008/Keynote_Speakers/pekin08_bonnans.pdf).
  Check: reducibility of a touch point (`d2g/dt2 != 0` there, already computed
  by the anchor) and that the transitions the active-set route and the
  generator's births make are exactly these. Use: completeness of the
  structure changes continuation must handle.

- [ ] **Junction conditions by order (Maurer; Jacobson-Lele-Speyer).** If
  `p + r` is odd (constraint order `p`, `r` the order of the first
  discontinuous control derivative) the jump multiplier vanishes at a
  junction. Check: our junction rows against this for order 1 and 2 (velocity
  limits are order 2 with jerk control). Use: the junction rows of Problem 4.

- [ ] **Second-order sufficient conditions for bang-bang controls and state
  constraints (Maurer, Büskens, Osmolovskii, Vossen).** Optimality of a
  bang-bang structure follows from the Hessian of the switching-time problem.
  [Maurer et al., CDC-ECC 2005](https://skoge.folk.ntnu.no/prost/proceedings/cdc-ecc05/pdffiles/papers/1136.pdf).
  Check: compute that Hessian from the multi-arc shooting Jacobians. Use: a
  sufficiency test where the conjugate-point test does not apply.

- [ ] **The derivative of the cost with respect to a switch time is the jump
  in H (Aghaee & Hager).**
  [SIAM J. Control Optim. 59 (2021), arXiv 2011.10022](https://arxiv.org/abs/2011.10022);
  [boundary-value extension, arXiv 2307.09722](https://arxiv.org/abs/2307.09722).
  Check: on our kernels, against finite differences. Use: a descent direction
  for structure search where Newton stalls.

## Structure search

- [ ] Structure classifier, per-structure costate models, facet-following on a
  failed solve (in progress; see the embedding lectures).
