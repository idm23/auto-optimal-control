"""Census cases with joint position limits: the census answers that overshoot, capped.

    python benchmarks/position_cases.py [--cache DIR] [--out benchmarks/position_cases.json]
        [--set name=value ...] [--rule R2=on ...]

With jerk control an angle limit ``q_min_i <= q_i <= q_max_i`` has order 3, and with a
smooth control law it is only ever touched, never ridden (Jacobson, Lele & Speyer 1971;
:func:`~auto_optimal_control.robot.joint_limits`). A limit must contain the joint's
start and goal angles, so it binds only where the motion overshoots that range. The
motions limited by jerk alone, which set the rate cases' caps (``rate_cases.py``), are
monotone and never overshoot. The torque-limited answers of the census do: the arm
swings past its target to use the dynamics.

So each census case whose answer overshoots is repeated with that joint capped, by the
rule the rate cases use: ``pct`` of the way from the joint's end range to its peak
there, with ``pct`` the case's own (70, 80 or 90 for a goal, 90, 80 and 70 in turn for
the rest cases). A side of a joint that does not overshoot (by more than 1% of the
joint's move and 1e-3 rad) gets no limit; its bound is set ``LOOSE`` away.

- ``pgoal<i>@<pct>``, ``prest<i>``: a torque-limited case, its torque limits as there,
  with position limits (the arm with position limits).
- ``pvgoal<i>@<pct>``, ``pvrest<i>``: a case with torque and rate limits, with position
  limits as well (all three orders at once).

No case is chosen by what the solver does with it: the answers set the limits only and
seed nothing. The cases are frozen in the JSON file, so the census does not drift with
the solver.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_census  # noqa: E402

LOOSE = 10.0                       # rad beyond the end range: a bound that never binds
REST_PCTS = (90, 80, 70)
FRACTION, FLOOR = 0.01, 1e-3       # an overshoot binds above 1% of the move and 1e-3 rad


def bases():
    """The census cases whose answers may overshoot, in the census's order."""
    return ([f"goal{i}@{p}" for i in range(10) for p in (90, 80, 70)]
            + [f"rest{i}" for i in range(15)] + list(arm_census.RATE_CASES))


def pct_of(base):
    if "@" in base:
        return int(base.split("@")[1])
    return REST_PCTS[int(base.lstrip("vrest")) % 3]


def limits(q, pct):
    """``(q_min, q_max, binding)`` from a sampled answer ``q`` (samples x joints)."""
    lo, hi = np.minimum(q[0], q[-1]), np.maximum(q[0], q[-1])
    move = hi - lo
    over_hi, over_lo = q.max(axis=0) - hi, lo - q.min(axis=0)
    thresh = np.maximum(FRACTION * move, FLOOR)
    up, down = over_hi > thresh, over_lo > thresh
    q_max = np.where(up, hi + pct / 100 * over_hi, hi + LOOSE)
    q_min = np.where(down, lo - pct / 100 * over_lo, lo - LOOSE)
    return q_min, q_max, (up | down)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).with_name("position_cases.json")))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--parallel-routes", action="store_true")
    ap.add_argument("--set", action="append")
    ap.add_argument("--rule", action="append")
    args = ap.parse_args(argv)
    arm_census.setup(argparse.Namespace(cache=args.cache, seconds=args.seconds,
                                        cap=args.cap, solves=0, snapshots=None,
                                        parallel_routes=args.parallel_routes, set=args.set,
                                        rule=args.rule, export=None))
    cases = {}
    for base in bases():
        p = arm_census.params(base)
        h = arm_census.homotopy_for(base)
        r = h.solve(p, arm_census.BOX, arm_census.TS)
        if not r.exact:
            print(f"{base}: not solved, no position case ({r.message[:60]})", flush=True)
            continue
        q = np.asarray(r.shooting.trajectory(r.solution.v, n_points=4001)["z"])[:, :6]
        pct = pct_of(base)
        q_min, q_max, binding = limits(q, pct)
        if not binding.any():
            continue
        name = "p" + base
        cases[name] = {"base": base, "pct": pct, "joints": np.flatnonzero(binding).tolist(),
                       "tf_base": float(r.solution.tf),
                       "params": [*p, *q_min.tolist(), *q_max.tolist()]}
        print(f"{name}: joints {np.flatnonzero(binding).tolist()} capped at {pct}% of "
              f"their overshoot", flush=True)
    Path(args.out).write_text(json.dumps(cases, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
