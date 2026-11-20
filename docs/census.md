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

The data shown is from the census of 2026-11-20, on the branch where the riding sensitivity
skips the costates its law does not see (Lemma SP) and residuals skip their junctions'
gradients. Rules R1, R2, R15, R16, R20, R25, R26,
R29, R32, R41, R43, R48, R50, R54 and R55 are turned on (the release's configuration), with
held-set events (R22) and closed-form free arcs (R23) on by default.
R15's portfolio has seven routes: two draws of the offsets, each also at ten times the
secondary weight (R24); the first draw at a hundred times it, once lowering the weight
after the limits and once with them (R45); and at three hundred times it with them. The
routes run in parallel, one case at a time, on 8 physical cores (the census runs its
cases in its main process, so the routes are the only other processes). Each case has 60 s in all (`--cap 60`, the reversed problem
included); a case that takes longer counts as unsolved.

84 of the 90 cases solve, all under 60 s: the median is 7.0 s and the slowest 47.6 s.
The same 84 cases solve as before Lemma SP, in 815 s against 859 s (5% faster): those
changes compute the same numbers with less work.
R55 makes the 84 cases solved on both pages 19% faster (1065 s to 859 s), and loses rest
10. There, near the route's last birth, two branches of solutions with one structure lie
close together. With R55, Newton reaches the one where a multiplier rises inside an arc,
which acceptance rejects. Without it, the residual test happened to keep the route on the
other.

When all seven routes are busy they run at about half speed (Observation O4). On most
cases all seven run until one wins, and the winners are spread over every route.

- **`goal` and `rest`:** all 45 solve.
- **`vgoal`:** 26 of 30: all ten at 90%, nine at 80% (all but goal 3) and seven at 70%
  (all but goals 0, 3 and 4).
- **`vrest`:** 13 of 15: all but rests 4 and 10.

Of the rate-limited answers solved before R43, six hold a torque limit and a rate limit
on another joint at the same instant (the nine new ones are not yet read for this): goals 3 and 4 at 90%, goals 0 and 1 at 80%, rest 2 and rest 9.
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
