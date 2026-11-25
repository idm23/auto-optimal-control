"""The six-joint arm census: 90 minimum-time solves, with what each ran into.

    python benchmarks/arm_census.py OUT.jsonl [--workers 14] [--cases goal4@90 rest11 ...]
        [--seconds 300] [--solves N] [--snapshots DIR] [--set name=value ...]
        [--rule R2=on ...] [--export docs/_static/census/data]

Cases: 10 goals from rest at the origin, each with torque limits at 90%, 80% and
70% of the way from the ends' torques to the free motion's peak (``goal<i>@<pct>``),
and 15 rest-to-rest motions with fixed limits (``rest<i>``). Then each of these again
with joint rate limits (order 2) at 90%, 80% or 70% of the jerk-limited motion's peak
rates (``vgoal<i>@<pct>``, ``vrest<i>``), frozen in ``rate_cases.json``
(``rate_cases.py``); their rows say which limits the answer holds, and which at once. Each row records the
outcome, the time, and ``HomotopyResult.diagnostics``: failed evaluations by class,
route steps by outcome, converged answers rejected by check, and rules fired.

With ``--parallel-routes`` each case runs one process per route (three by default).
Keep ``workers * 3`` within the machine's *physical* cores: on hyperthreads two
processes share a core, every case runs at about half speed, and cases near their
wall-clock budget fail (homebox-node-1 has 8 cores: use ``--workers 2``).

``--cap`` is the wall-clock budget of a whole case (``Homotopy.solve_seconds``: the
reversed problem and the parallel routes included), and a case over it counts as a
failure; the census page uses ``--cap 60``. ``--solves`` caps the route's path solves (``Homotopy.max_path_solves``), a budget
that does not depend on the machine or its load. ``--set`` sets a Homotopy attribute
(``--set offset_start=staggered --set width_steps=True``); ``--set
MultiArcShooting.rate_junctions_default=False`` sets a class default; ``--rule
R2=on`` turns a rule of ``auto_optimal_control.rules`` on or off. ``--export`` writes each case's
motion for the animated census page (``docs/census.md``), with a manifest of every
case in the directory, so new cases show up on the page by themselves. The summary
at the end adds the counters up over all cases.
"""

import argparse
import ast
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import sympy as sp

from auto_optimal_control import examples
from auto_optimal_control.homotopy import Homotopy
from auto_optimal_control.robot import GRAVITY, SIX_JOINT_LINKS, joint_torques
from auto_optimal_control import rules, shooting

sys.path.insert(0, str(Path(__file__).resolve().parent))   # importable from anywhere
from census_export import export_case, write_manifest  # noqa: E402

TAU = np.array([5.0, 87.3, 28.5, 2.5, 0.7, 0.2])
J = np.full(6, 30.0)
BOX, TS = ([-.5] * 18, [.5] * 18), (.3, 5.)
H = None
HV = None            # the arm with rate limits, for the cases of rate_cases.json
ARGS = None
TQ = None
EXPORT = None
RATE_CASES = json.loads((Path(__file__).resolve().parent / "rate_cases.json").read_text()) \
    if (Path(__file__).resolve().parent / "rate_cases.json").exists() else {}
HP = None            # the arm with position limits (order 3), for position_cases.json
HPV = None           # the arm with rate and position limits
POSITION_CASES = json.loads((Path(__file__).resolve().parent / "position_cases.json").read_text()) \
    if (Path(__file__).resolve().parent / "position_cases.json").exists() else {}


def _configure(h, args):
    """Apply the run's budget, ``--set`` and ``--rule`` to a Homotopy."""
    h.active_set_seconds = args.seconds
    h.solve_seconds = getattr(args, "cap", None)
    if args.solves:
        h.max_path_solves = args.solves
        h.active_set_seconds = float("inf")
    h.snapshot_dir = args.snapshots
    h.parallel_routes = args.parallel_routes
    for item in args.set or []:
        name, value = item.split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        if name.startswith("MultiArcShooting."):
            setattr(shooting.MultiArcShooting, name.split(".", 1)[1], value)
        else:
            setattr(h, name, value)
    for item in args.rule or []:
        name, state = item.split("=", 1)
        rules.set_rule(h, name, state.lower() in ("on", "1", "true"))


def rates_homotopy():
    """The arm with joint rate limits (built on first use, configured as ``H``)."""
    global HV
    if HV is None:
        HV = Homotopy(examples.six_joint_arm_min_time(rate_limits=True), cache_dir=ARGS.cache)
        _configure(HV, ARGS)
    return HV


def positions_homotopy(rates=False):
    """The arm with joint position limits, and rate limits too with ``rates`` (built on
    first use, configured as ``H``)."""
    global HP, HPV
    if rates:
        if HPV is None:
            HPV = Homotopy(examples.six_joint_arm_min_time(rate_limits=True, position_limits=True),
                           cache_dir=ARGS.cache)
            _configure(HPV, ARGS)
        return HPV
    if HP is None:
        HP = Homotopy(examples.six_joint_arm_min_time(position_limits=True), cache_dir=ARGS.cache)
        _configure(HP, ARGS)
    return HP


def homotopy_for(name):
    """The Homotopy that solves census case ``name``."""
    if name in POSITION_CASES:
        return positions_homotopy(rates=POSITION_CASES[name]["base"] in RATE_CASES)
    return rates_homotopy() if name in RATE_CASES else H


def setup(args):
    global H, TQ, EXPORT, ARGS
    ARGS = args
    EXPORT = getattr(args, "export", None)
    warnings.simplefilter("ignore")
    q, w, a = (sp.symbols(f"{s}1:7", real=True) for s in "qwa")
    f = sp.lambdify((q, w, a), joint_torques(SIX_JOINT_LINKS, q, w, a, GRAVITY), "numpy")
    TQ = lambda Q, Wd, A: np.array([np.broadcast_to(np.asarray(v, dtype=float), Q.shape[:1])
                                    for v in f(Q.T, Wd.T, A.T)]).T
    H = Homotopy(examples.six_joint_arm_min_time(), cache_dir=args.cache)
    _configure(H, args)
    print("rules on:", ", ".join(n for n, on in rules.enabled(H).items() if on), flush=True)


def params(name, free_cache={}):
    if name in POSITION_CASES:
        return list(POSITION_CASES[name]["params"])
    if name in RATE_CASES:
        return list(RATE_CASES[name]["params"])
    rng = np.random.default_rng(0)
    goals = [rng.uniform(-1.5, 1.5, 6) for _ in range(10)]
    if name.startswith("rest"):
        g = np.random.default_rng(500 + int(name[4:]))
        x0, xf = g.uniform(-1.5, 1.5, 6), g.uniform(-1.5, 1.5, 6)
        return [*x0, *xf, *J, *TAU]
    i, pct = name[4:].split("@")
    i = int(i)
    base = [*np.zeros(6), *goals[i], *J]
    if i not in free_cache:
        r0 = H.solve([*base, *np.full(6, 1e4)], BOX, TS)
        z = r0.shooting.trajectory(r0.solution.v, n_points=4001)["z"]
        tau = TQ(z[:, :6], z[:, 6:12], z[:, 12:18])
        free_cache[i] = (np.abs(tau).max(0), 1.001 * np.maximum(np.abs(tau[0]), np.abs(tau[-1])))
    peak, ends = free_cache[i]
    return [*base, *(ends + int(pct) / 100 * (peak - ends).clip(0))]


def run(name):
    p, r = None, None
    try:
        p = params(name)
        h = homotopy_for(name)
        h.snapshot_tag = name
        t0 = time.time()
        r = h.solve(p, BOX, TS)
        row = {"name": name, "exact": bool(r.exact), "sec": round(time.time() - t0, 3),
               "tf": float(r.solution.tf) if r.exact else None, "msg": r.message[:80],
               "diagnostics": r.diagnostics}
        cap = getattr(ARGS, "cap", None)
        if cap is not None and row["exact"] and row["sec"] > cap:
            # Over the cap is a failure, even if it solved in the end.
            row.update(exact=False, tf=None, msg=f"over the {cap:g} s cap (solved in {row['sec']} s)")
        if row["exact"] and (name in RATE_CASES or name in POSITION_CASES):
            # Which orders the answer holds, and whether at once (the case's group is
            # only its intent).
            from rate_cases import classify
            row["orders"] = classify(r.shooting, r.solution.v)
    except Exception as exc:          # a case that raises is a result too
        row = {"name": name, "exact": False, "sec": 0.0, "tf": None, "msg": repr(exc)[:80],
               "diagnostics": {}}
    if EXPORT and p is not None:
        try:
            pc = POSITION_CASES.get(name)
            rates = name in RATE_CASES or (pc is not None and pc["base"] in RATE_CASES)
            export_case(EXPORT, name, p, row, r if row["exact"] else None, TQ, rates=rates,
                        positions=pc["joints"] if pc is not None else None)
        except Exception as exc:      # the census result stands without the page's copy
            print(f"export of {name} failed: {exc!r}", flush=True)
    return row


def summary(rows):
    ok = [r for r in rows if r["exact"]]
    secs = sorted(r["sec"] for r in ok)
    print(f"solved {len(ok)} of {len(rows)}; under 60 s: {sum(s < 60 for s in secs)}; "
          f"under 5 s: {sum(s < 5 for s in secs)}; "
          f"mean {sum(secs) / len(secs) if secs else float('nan'):.3f} s")
    for tag in ("goal", "rest", "vgoal", "vrest"):
        part = [r for r in rows if r["name"].startswith(tag) and r["name"][len(tag):][:1].isdigit()]
        if part:
            ok_ = [r for r in part if r["exact"]]
            print(f"  {tag}: solved {len(ok_)} of {len(part)}; "
                  f"under 60 s: {sum(r['sec'] < 60 for r in ok_)}")
    for part in ("evaluations", "steps", "rejected", "rules", "oracles"):
        total = {}
        for r in rows:
            for k, n in r["diagnostics"].get(part, {}).items():
                total[k] = total.get(k, 0) + n
        print(f"  {part}: " + ", ".join(f"{k} {n}" for k, n in
                                        sorted(total.items(), key=lambda kv: -kv[1])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--cap", type=float, default=None,
                    help="wall-clock seconds per case, everything included; a case over "
                         "it is a failure (Homotopy.solve_seconds)")
    ap.add_argument("--solves", type=int, default=0)
    ap.add_argument("--snapshots", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--set", action="append")
    ap.add_argument("--rule", action="append")
    ap.add_argument("--export", default=None,
                    help="write each case's motion for the census page to this directory "
                         "(docs/_static/census/data)")
    ap.add_argument("--parallel-routes", action="store_true",
                    help="run R15's routes at once (each case then uses one core per route)")
    args = ap.parse_args(argv)
    setup(args)
    names = args.cases or ([f"goal{i}@{p}" for i in range(10) for p in (90, 80, 70)]
                           + [f"rest{i}" for i in range(15)] + list(RATE_CASES)
                           + list(POSITION_CASES))
    # Every arm the cases need is built before the workers fork.
    for n in names:
        homotopy_for(n)
    rows = []
    # Workers that may start processes of their own (Homotopy.parallel_routes): a
    # ProcessPoolExecutor's are not daemonic, unlike a Pool's.
    from concurrent.futures import ProcessPoolExecutor, as_completed
    if args.workers == 1:
        # One worker reuses one process for every case anyway: run them here, so that
        # the routes R15 forks (one per core) are the only other processes.
        with open(args.out, "w") as fo:
            for n in names:
                row = run(n)
                rows.append(row)
                fo.write(json.dumps(row) + "\n")
                fo.flush()
    else:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("fork")) as pool, \
                open(args.out, "w") as fo:
            for fut in as_completed([pool.submit(run, n) for n in names]):
                row = fut.result()
                rows.append(row)
                fo.write(json.dumps(row) + "\n")
                fo.flush()
    summary(rows)
    if args.export:
        print(f"exported {write_manifest(args.export)} cases to {args.export}")


if __name__ == "__main__":
    main()
