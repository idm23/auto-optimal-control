# Arm census

The six-joint arm benchmark: minimum-time, jerk-limited motions, each with a
torque limit on every joint. There are five families of cases:

- **`goal<i>@<p>`**: from rest at the origin to one of ten random targets. The
  torque limits are set from the motion with no torque limits. Each joint's
  limit is a fraction `p` of the way from the torque that holds the pose at
  rest to the peak torque the motion uses. 90% clips the top 10% of each
  joint's dynamic torque, and 70% the top 30%.
- **`rest<i>`**: between two random poses, at rest at both ends, with one
  fixed table of torque limits.
- **`vgoal<i>@<p>`, `vrest<i>`**: each case above again, with a limit on every
  joint's rate as well. The rate limits are a fraction of the peak rates of the motion
  limited by jerk alone: `p` for the goals, and 90%, 80% and 70% in turn for
  the rest cases. With jerk control a rate limit has order 2 and a torque limit
  order 1. The cases are frozen in `benchmarks/rate_cases.json`, made by
  `benchmarks/rate_cases.py`. Each row says which limits its answer holds, and which
  ones at the same instant.
- **`pgoal<i>@<p>`, `prest<i>`, `pvgoal<i>@<p>`, `pvrest<i>`**: the census answers that
  swing a joint past its start-to-goal range, again with that joint's angle capped. A
  position limit must contain the start and goal angles, so it binds only where the
  answer overshoots. The motions limited by jerk alone never do; the torque-limited
  answers do, the arm swinging past its target to use the dynamics.
  - **The cap:** each overshooting joint is capped a fraction `p` of the way from its end
    range to its peak there (the goals' `p`; 90%, 80% and 70% in turn for the rest
    cases), as the rate cases cap rates. A side that does not overshoot gets no limit.
  - **The two families:** `pgoal` and `prest` come from the torque-limited answers (17),
    with torque and position limits. `pvgoal` and `pvrest` come from the rate-limited
    answers (10), with torque, rate and position limits: all three orders at once.
  - **The order:** with jerk control an angle limit has order 3, and with a smooth
    control law it is only touched, never ridden (Jacobson, Lele & Speyer 1971).
  - **Where they live:** the cases are frozen in `benchmarks/position_cases.json`, made
    by `benchmarks/position_cases.py`.
- **`ograze<i>`, `oblock<i>`, `owall<i>`**: each rest case again, with one obstacle in
  its way that the elbow, the middle of the forearm, the wrist and the tool must clear
  (`robot.obstacle_limits`). Each point is a ball of the arm's half-width there (5 to
  7 cm), which inflates the obstacle. Three rungs, each harder than the last:
  - **`ograze`:** a sphere of radius 8 cm that the rest answer's tool path clips a fifth
    of the way into (`g = 0.2`). This is a small correction, like the position cases.
  - **`oblock`:** a sphere of radius 10 cm centred near the middle of the tool path, a
    quarter of its inflated radius to one side (`g ≈ 0.94`). The answer must go around it.
  - **`owall`:** a thin upright wall, 4 × 30 × 24 cm, across the tool path and facing
    it, set off along its width and height so the path crosses it off-centre. The answer
    must go over it or around a side. The wall is a superquadric of exponent 4: smooth,
    with its edges rounded.
  - **The order:** `g` depends on the angles alone, so with jerk control it has order 3,
    like an angle limit, and is only touched, never ridden. Unlike an angle limit it
    couples every joint through the arm's kinematics.
  - **Where they live:** the cases are frozen in `benchmarks/obstacle_cases.json`, made
    by `benchmarks/obstacle_cases.py`. The route grows each obstacle from its core (the
    offsets of the limits fall from the rest answer's deepest point to zero), so it picks
    the side by geometry, not by cost. Its answer is a local optimum.

The data shown is from the census of 2026-11-25, on the branch that tests the held-set
guards inside each step (R78, Lemma HE). Rules R1, R2, R15, R16, R20, R25, R26, R28,
R29, R32, R41, R43, R48, R50, R54, R55, R60, R61, R63, R64, R65, R66, R72, R74, R75 and R78 are turned on
(the release's configuration), with
held-set events (R22) and closed-form free arcs (R23) on by default.
R15's portfolio has seven routes: two draws of the offsets, each also at ten times the
secondary weight (R24); the first draw at a hundred times it, once lowering the weight
after the limits and once with them (R45); and at three hundred times it with them. The
routes run in parallel, one case at a time, on 8 physical cores: four at once (the
first, second, fifth and sixth), the other three after 6 s or as soon as one of the four
ends (R72). The census runs its
cases in its main process, so the routes are the only other processes. Each case has 60 s in all (`--cap 60`, the reversed problem
included); a case that takes longer counts as unsolved.

113 of the 117 cases solve, all under 60 s, and 89 of them under 5 s. The mean solve time
of the solved cases is 4.454 s. Of the 27 position-limit cases, 26 solve, 16 of them
under 5 s, and every one of the 26 touches the position limits it was given (Observation
O8). Of the 90 older cases 87 solve, 73 under 5 s. Times are wall-clock seconds per case,
to the millisecond.

The 45 obstacle cases were added in the same configuration, on the same machine: 41 solve,
all under 60 s, and 20 under 5 s, with a mean of 7.644 s. With them the page has 154 of
162 solved, a mean of 5.303 s.

- **Every wall solves (15 of 15)**, 6 under 5 s, with a mean of 10.8 s. **Blocks:** 14 of
  15, 6 under 5 s, with a mean of 6.9 s. **Grazes:** 12 of 15, 8 under 5 s, with a mean of
  4.6 s.
- **The hardest rung is not the hardest to solve.** None of the four failures has to make
  a large change. Grazes 3 and 10 are lost long before the cap: every route ends by itself,
  its path steps mostly failing to settle on a structure ("no structure converged",
  "violation too large"). Given 300 s, graze 3 stops again at 111 s and graze 10 at 17 s.
  Graze 6 runs out of time and does no better with 300 s. Block 6 also runs out of time,
  but given 300 s on a slower, shared machine it solves in 218 s, through the reversed
  problem, 121 ms slower than rest 6.
- **Each answer touches the obstacle, never rides it.** The touches are at isolated
  instants, by one body point (31 answers), two (7) or three (3), as an order-3 limit
  should be. Most touch with the tool or the wrist, and two walls and a block with the
  elbow. Block 10 and wall 10 are solved through the reversed problem (R4).
- **What the obstacle costs.** The median motion time rises 4 ms for a graze, 30 ms for a
  block and 40 ms for a wall; the most is 298 ms (graze 2, whose sphere the forearm, wrist
  and tool all have to clear). For rest cases 7, 8, 10 and 14 the obstacle costs nothing,
  within 0.04 ms at every rung, though the motions move by up to 0.5 rad. The minimum time
  is set by the joints on their torque limits, and the other joints go around the obstacle
  for free.
- **Two answers are faster than their rest case.** Grazes 1 and 4 take 0.53 and 0.10 ms
  less than rest 1 and rest 4, from the same ends, with every limit held. The rest answers
  are local optima, and the obstacle moved the route into a better one.
- **No motion passes through an obstacle.** A check of every solved answer at twenty
  times the page's sampling finds the deepest `g` at 9e-4, which is the rounding of the
  exported angles.

Rest 2 with rate and position limits now solves, in 2.5 s. Its answer had been found and
then rejected for a violation it does not have. Riding a torque limit, a joint's jerk
reached its bound and left it within one integration step. Tested only at the steps'
ends, that change of the held set was missed, so the residual without the STM integrated
a flow off the riding law, while the residual with the STM, whose steps are shorter, saw
it. Newton converged one and the checks judged the other. The guards are now tested at
four points inside each step as well (R78, Lemma HE). The 112 cases solved on both pages
take 500.8 s against 505.2 s. Single cases move both ways, as when any rule changes the
route's numerics: rest 5 with rate and position limits from 24.3 to 14.3 s and vrest 12
from 40.3 to 26.6 s, but vrest 13 from 4.0 to 6.0 s. Four cases that were just under 5 s
are now just over.

Rest 10 with rate and position limits is the first case solved through the reversed
problem (R4). Its reversed problem solved before, but the extremal mapped back did not
propagate: R4 kept each arc's jump as it was, which is right at order 1 but not at order
2. There the jump's second coefficient is the multiplier measure's first moment about the
arc's entry, and reflection measures it from the other end (Lemma F2). With the jump
mapped, the extremal propagates and is accepted. The 111 cases solved before take 472.5 s
against 475.2 s, within the page-to-page noise; the mean rises only because rest 10's
32.7 s is now counted.

Four changes made this. First, four routes start at once and the other three after 6 s
(R72, Observation O7): fewer busy cores run faster (O4), and most cases are won by one of
the first four. Second, the route integrates its path problems to 1e-10 instead of the
exact problem's 1e-12 (R74, Lemma TB). Its solves stop at 1e-6 and its monitors judge at
1e-6, so the integration stays two orders below both, and the exact solve keeps 1e-12.
Third, the parent process takes the answer a route accepted as it is, with its exact
shooting rebuilt around it, and polls the routes every 10 ms: 0.1 to 0.3 s per case, with
the same final times to 1e-15. Fourth, a riding arc's STM comes from FNI's decoupled tangent propagator (R75, Lemma DT).
The arc's state is integrated alone, as before. Per accepted step, one sixth-order step of
the variational equation uses the riding Jacobian at five points, instead of evaluating
the variational right-hand side at all twelve stages. The route needs its Jacobians only
to about 1e-6, and the propagator stays within that. Final times agree to 1e-13. With all seven routes at once and 1e-12 (page 113), 57 cases were under 5 s and the
mean was 5.078 s. The same 87 cases solve, in 299.0 s against 441.8 s.

When all seven routes are busy they run at about half speed (Observation O4). The
winners are spread over every route, so none can be dropped.

- **`goal` and `rest`:** all 45.
- **`vgoal`:** 28 of 30: all ten at 90%, all ten at 80% and eight at 70% (all but goals
  0 and 3).
- **`vrest`:** 14 of 15: all but rest 4.
- **`pgoal`:** all 6. **`prest`:** 10 of 11, all but rest 4. **`pvgoal`:** the one.
  **`pvrest`:** all 9.
- **`ograze`:** 12 of 15, all but rests 3, 6 and 10. **`oblock`:** 14 of 15, all but
  rest 6. **`owall`:** all 15.

Of the rate-limited answers solved before R43, six hold a torque limit and a rate limit
on another joint at the same instant (the nine new ones are not yet read for this): goals 3 and 4 at 90%, goals 0 and 1 at 80%, rest 2 and rest 9.
Goal 3 at 90% also holds `tau4_min` and `w4_min` together, and goal 1 at 80%
`tau5_min` and `w5_max`.

Pick a case to watch it. Its traces turn red where the joint rides its limit: torques
for every case, rates too for a rate-limited one, and for a position-limit case the
capped joints' angles, red where they touch a limit. An obstacle case draws the obstacle
and the ball of each body point that must clear it, red while it touches. A panel shows
each point's clearance over time, in units of the inflated obstacle's size, with a dot
where it touches. The page is drawn from the data in
`_static/census/data`. Every pull request that changes the solver or the census
refreshes it before merging; the README's "Merging a pull request" has the command.
Exporting a census adds every case it runs, new ones included:

```bash
python benchmarks/arm_census.py census.jsonl --export docs/_static/census/data [census options]
```

<iframe src="_static/census/index.html" title="Arm census" loading="lazy"
        style="width: 100%; height: 1500px; border: 0;"></iframe>

[Open the census on its own page](_static/census/index.html).
