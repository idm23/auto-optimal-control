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

The data shown is from the census of 2026-11-16 (release 0.4.0). Rules R1, R2, R15,
R16, R20, R25 and R26 are turned on, with held-set events (R22) and closed-form free arcs
(R23) on by default. R15's portfolio has six routes: two draws of the offsets and
sequential births, each also at ten times the secondary weight (R24). The routes run in
parallel, one case at a time, with 300 s per case, on 8 physical cores. All 45 cases
solve, all in under 60 s: the median is 8.2 s and the slowest 45.2 s.

The rate-limited cases came later (2026-11-17) and are not in the data shown. On the
first route alone, 10 of them solve, nine in under 60 s: nine goals and one rest case,
all at 90%. Two of the goals (goal 3 and goal 4) hold a torque limit and rate limits
on other joints at the same instant. No case at 80% or 70% solves yet.

Pick a case to watch it. Its torque traces turn red where the joint rides its
limit. The page is drawn from the data in `_static/census/data`. Exporting a
census adds every case it runs, new ones included:

```bash
python benchmarks/arm_census.py census.jsonl --export docs/_static/census/data [census options]
```

<iframe src="_static/census/index.html" title="Arm census" loading="lazy"
        style="width: 100%; height: 1500px; border: 0;"></iframe>

[Open the census on its own page](_static/census/index.html).
