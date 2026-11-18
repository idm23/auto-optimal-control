"""Census cases with joint rate limits: every arm census case again, with its rates capped.

    python benchmarks/rate_cases.py [--cache DIR] [--out benchmarks/rate_cases.json]
        [--set name=value ...] [--rule R2=on ...]

With jerk control a rate limit ``|w_i| <= w_max_i`` has order 2 and a torque limit
order 1 (:func:`~auto_optimal_control.robot.joint_limits`). Each case of the arm census
(``arm_census.py``) is repeated with the rate limits added, by the rule the goal cases
use for torques: the jerk-limited motion alone (torque limits loose) is solved, and each
joint's rate is capped at ``pct`` of the way from its ends' rate (zero, at rest) to its
peak there.

- ``vgoal<i>@<pct>``: goal ``i`` at ``pct``, its torque limits as there, rates at ``pct``.
- ``vrest<i>``: rest ``i``, its torque limits as there, rates at 90, 80 and 70 in turn.

No case is chosen by what the solver does with it. The jerk-limited answers set the
limits only and seed nothing. :func:`classify` labels an answer by what it holds
(rate limits, torque limits, both at one instant, on the same joint or on different
ones). The cases are frozen in the JSON file, so the census does not drift with the
solver.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_census  # noqa: E402

LOOSE_TAU = 1e4
REST_PCTS = (90, 80, 70)


def names():
    """The census's cases with rate limits, in the census's order."""
    return ([f"vgoal{i}@{p}" for i in range(10) for p in (90, 80, 70)]
            + [f"vrest{i}" for i in range(15)])


def _joint(name):
    return int(re.search(r"\d+", name).group())


def classify(shooting, v):
    """What a solution holds: ``{"1": [...], "2": [...], "together": [...]}``, the order-1
    and order-2 limits held or touched, and the pairs ``(order-1, order-2)`` held at one
    instant (an arc of one over an arc or a touch of the other)."""
    k = shooting.kernels
    names_, orders = list(k.CONSTRAINT_NAMES), list(k.CONSTRAINT_ORDERS)
    events = getattr(shooting, "events", [])
    spans = {}                                   # arc index -> [t_in, t_out]
    for (ev, i), t in zip(events, shooting.event_times(v) if events else []):
        if ev == "touch":
            spans[i] = [float(t), float(t)]
        elif ev in ("in", "out"):
            spans.setdefault(i, [float(t), float(t)])[0 if ev == "in" else 1] = float(t)
    held = {1: [], 2: []}
    for i, (a, b) in spans.items():
        c = shooting.arcs[i][1]
        if orders[c] in held:
            held[orders[c]].append((names_[c], a, b))
    together = sorted({(n1, n2) for n1, a1, b1 in held[1] for n2, a2, b2 in held[2]
                       if a1 <= b2 and a2 <= b1})
    return {"1": sorted({n for n, _, _ in held[1]}), "2": sorted({n for n, _, _ in held[2]}),
            "together": [list(p) for p in together],
            "same joint": any(_joint(a) == _joint(b) for a, b in together),
            "other joints": any(_joint(a) != _joint(b) for a, b in together)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).with_name("rate_cases.json")))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--set", action="append")
    ap.add_argument("--rule", action="append")
    args = ap.parse_args(argv)
    arm_census.setup(argparse.Namespace(cache=args.cache, seconds=args.seconds, solves=0,
                                        snapshots=None, parallel_routes=True, set=args.set,
                                        rule=args.rule, export=None))
    H = arm_census.H
    peaks, cases = {}, {}
    for name in names():
        base = name[1:]                          # the census case it repeats
        p = arm_census.params(base)
        key = base.split("@")[0]                 # goal i at every pct: one jerk-limited motion
        if key not in peaks:
            r = H.solve([*p[:18], *np.full(6, LOOSE_TAU)], arm_census.BOX, arm_census.TS)
            if not r.exact:
                print(f"{key} jerk-limited: not solved ({r.message[:60]})", flush=True)
                peaks[key] = None
            else:
                w = np.asarray(r.shooting.trajectory(r.solution.v, n_points=4001)["z"])[:, 6:12]
                peaks[key] = np.abs(w).max(axis=0)
                print(f"{key}: peak rates {np.round(peaks[key], 3).tolist()}", flush=True)
        if peaks[key] is None:
            continue
        pct = int(base.split("@")[1]) if "@" in base else REST_PCTS[int(base[4:]) % 3]
        cases[name] = {"base": base, "pct": pct, "peak": peaks[key].tolist(),
                       "params": [*p, *(pct / 100 * peaks[key])]}
    Path(args.out).write_text(json.dumps(cases, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
