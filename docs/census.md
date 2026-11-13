# Arm census

The six-joint arm benchmark: minimum-time, jerk-limited motions, each with a
torque limit on every joint. There are two families of cases:

- **`goal<i>@<p>`**: from rest at the origin to one of ten random targets. The
  torque limits are set from the motion with no torque limits. Each joint's
  limit is a fraction `p` of the way from the torque that holds the pose at
  rest to the peak torque the motion uses. 90% clips the top 10% of each
  joint's dynamic torque, and 70% the top 30%.
- **`rest<i>`**: between two random poses, at rest at both ends, with one
  fixed table of torque limits.

Pick a case to watch it. Its torque traces turn red where the joint rides its
limit. The page is drawn from the data in `_static/census/data`. Exporting a
census adds every case it runs, new ones included:

```bash
python benchmarks/arm_census.py census.jsonl --export docs/_static/census/data [census options]
```

<iframe src="_static/census/index.html" title="Arm census" loading="lazy"
        style="width: 100%; height: 1500px; border: 0;"></iframe>

[Open the census on its own page](_static/census/index.html).
