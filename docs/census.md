# Arm census

The six-joint arm benchmark: minimum-time, jerk-limited motions, each with a
torque limit on every joint. There are three families of cases:

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

The data shown is from the census of 2026-11-19, on the branch that handles junctions at
one instant to rounding (sampling, ordering and zero-width segments). Rules R1, R2, R15, R16, R20, R25 and R26 are turned on (the release's
configuration), with held-set events (R22) and closed-form free arcs (R23) on by default.
R15's portfolio has six routes: two draws of the offsets and sequential births, each
also at ten times the secondary weight (R24). The routes run in parallel, one case at a
time, on 8 physical cores. Each case has 60 s in all (`--cap 60`, the reversed problem
included); a case that takes longer counts as unsolved.

69 of the 90 cases solve, all under 60 s: the median is 12.7 s and the slowest 56.8 s.
Goal 2 at 70% finishes near the cap, and which side of it it lands on varies from run to
run: it took 60.1 s, 58.6 s, 60.1 s, 58.9 s, 59.5 s, 60.2 s, 59.7 s, 59.4 s and 60.0 s on
the last nine. Rest 14 is new (17.8 s): its rate arcs enter and leave together by
symmetry, which the route now handles. So is goal 7 at 70% (34.1 s).

- **`goal` and `rest`:** all 45 solve.
- **`vgoal`:** 19 of 30: nine at 90% (all but goal 0), seven at 80% (goals 0, 1, 2, 5,
  6, 7 and 8) and three at 70% (goals 6, 7 and 8).
- **`vrest`:** 5 of 15: rest 0 and rest 9 (rates at 90%), rest 7 (at 80%), and rests 2
  and 14 (at 70%).

Six of the rate-limited answers hold a torque limit and a rate limit on another joint
at the same instant: goals 3 and 4 at 90%, goals 0 and 1 at 80%, rest 2 and rest 9.
Goal 3 at 90% also holds `tau4_min` and `w4_min` together, and goal 1 at 80%
`tau5_min` and `w5_max`.

Pick a case to watch it. Its traces turn red where the joint rides its limit: torques
for every case, and rates too for a rate-limited one. The page is drawn from the data in
`_static/census/data`. Every pull request that changes the solver or the census
refreshes it before merging; the README's "Merging a pull request" has the command.
Exporting a census adds every case it runs, new ones included:

```bash
python benchmarks/arm_census.py census.jsonl --export docs/_static/census/data [census options]
```

<iframe src="_static/census/index.html" title="Arm census" loading="lazy"
        style="width: 100%; height: 1500px; border: 0;"></iframe>

[Open the census on its own page](_static/census/index.html).
