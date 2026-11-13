"""Compare census cases with a direct transcription (comparison only, never a seed).

    python benchmarks/arm_direct.py goal9@80 rest12 [--intervals 120] [--census census.jsonl]

For each case: the direct answer's final time, the arcs of each torque limit, the
jerks held at their bounds, and the limits' multipliers (the density of each
limit's measure, eta, in docs/notes/structure_certificates.md). With
``--census``, the indirect final time from a census file is printed alongside.
Needs ``pip install "auto-optimal-control[direct]"``.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_census  # noqa: E402

from auto_optimal_control import direct, examples  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--intervals", type=int, default=120)
    ap.add_argument("--regularize", type=float, default=1e-4)
    ap.add_argument("--census", default=None, help="a census .jsonl to compare final times with")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None, help="write the answers as JSON here")
    args = ap.parse_args(argv)
    arm_census.setup(argparse.Namespace(cache=args.cache, seconds=60.0, solves=0, snapshots=None,
                                        parallel_routes=False, set=None, rule=[], export=None))
    census = {}
    if args.census:
        census = {r["name"]: r for r in map(json.loads, open(args.census))}
    problem = examples.six_joint_arm_min_time()
    results = {}
    for name in args.cases:
        p = np.asarray(arm_census.params(name), float)
        tau = p[18:24]
        # A first guess for the time: the motion with jerk limits alone, stretched.
        free = arm_census.H.solve([*p[:18], *np.full(6, 1e4)], arm_census.BOX, arm_census.TS)
        tr = free.shooting.trajectory(free.solution.v, n_points=400)
        best = None
        for stretch in (1.2, 1.5):
            z = np.asarray(tr["z"])[:, :18] / np.repeat([1.0, stretch, stretch ** 2], 6)
            guess = (np.asarray(tr["t"]) * stretch, z,
                     np.zeros((len(tr["t"]), 6)))
            sol = direct.solve(problem, p, intervals=args.intervals, guess=guess,
                               regularize=args.regularize, scale_limits=np.r_[tau, tau],
                               tf_bounds=arm_census.TS)
            print(f"{name}: stretch {stretch}: tf {sol.tf:.5f} {sol.status} "
                  f"({sol.iterations} iterations, {sol.seconds:.1f} s)", flush=True)
            if sol.success and (best is None or sol.tf < best.tf):
                best = sol
        if best is None:
            continue
        ref = census.get(name, {})
        print(f"{name}: direct tf {best.tf:.5f}; indirect "
              f"{ref.get('tf') if ref.get('exact') else 'unsolved'}")
        for limit, runs in best.arcs(1e-5).items():
            if runs:
                c = best.names.index(limit)
                peak = float(best.path_multipliers[:, c].max())
                print(f"  {limit:10s} on {[(round(a, 4), round(b, 4)) for a, b in runs]}  "
                      f"largest multiplier {peak:.3g}")
        for i, runs in best.held(1e-5).items():
            print(f"  jerk {i + 1} held {[(round(a, 3), round(b, 3), s) for a, b, s in runs]}")
        results[name] = {"tf": best.tf, "t": best.t.tolist(), "x": best.x.tolist(),
                         "u": best.u.tolist(), "arcs": best.arcs(1e-5),
                         "multipliers": best.path_multipliers.tolist(), "names": best.names}
    if args.out:
        Path(args.out).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
