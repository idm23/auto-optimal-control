"""Census cases with an obstacle: the rest answers, with something in their way.

    python benchmarks/obstacle_cases.py [--cache DIR] [--out benchmarks/obstacle_cases.json]
        [--bases rest0 rest1 ...] [--set name=value ...] [--rule R2=on ...]

Each case is a census ``rest`` case, its torque limits as there, with one obstacle the
elbow, forearm, wrist and tool must clear (:func:`~auto_optimal_control.robot.obstacle_limits`).
An obstacle limit depends on the angles alone, so with jerk control it has order 3, like
a joint's angle limit, and with a smooth control law it is touched, never ridden. Unlike
an angle limit it couples every joint, through the arm's kinematics, and the obstacle may
stand squarely in the way, so the answer must move to the other side of it. The cases
come in three rungs, each harder than the last:

- ``ograze<i>``: a sphere (radius 8 cm) that the rest answer's tool path clips: it
  passes a fifth of the way into the inflated sphere (``g = 0.2``) at the middle of the
  path. A small correction, like the position cases.
- ``oblock<i>``: a sphere (10 cm) centred near the middle of the tool path, a quarter of
  its inflated radius to one side (``g`` about 0.94 there): the answer must go around it.
- ``owall<i>``: a thin upright box (4 cm by 30 cm by 24 cm, a superquadric of exponent
  ``BOX_POWER``) across the tool path at its middle, turned to face it, and set off
  along its width and height so the path crosses it off-centre: the answer must go over
  it or around a side.

The middle of the path is where half its length is behind the tool. Each obstacle is
placed off that point along the horizontal normal of the path; when a placement would
put a body point of the start or goal pose within the obstacle (``g > -0.2``), the
other side and then points at 40%, 60%, 30% and 70% of the path are tried, and a base
with no clear placement gets no case of that rung. The answers set the obstacles only
and seed nothing; the cases are frozen in the JSON file, so the census does not drift
with the solver.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_census  # noqa: E402
from auto_optimal_control.robot import (SIX_JOINT_BODY, SIX_JOINT_LINKS,  # noqa: E402
                                        body_points_numeric, obstacle_values)

RADII = np.array([float(r) for *_, r in SIX_JOINT_BODY])
TOOL = [n for n, *_ in SIX_JOINT_BODY].index("tool")
GRAZE_R, GRAZE_G = 0.08, 0.2
BLOCK_R, BLOCK_SIDE = 0.10, 0.25
WALL = (0.02, 0.15, 0.12)          # half-extents: thickness, width, height
WALL_SIDE, WALL_DOWN = 0.35, 0.3   # the crossing's offset, as fractions of the inflated extents
CLEAR = -0.2                       # every body point of both end poses is at g below this
FRACTIONS = (0.5, 0.4, 0.6, 0.3, 0.7)


def tool_frame(P, s):
    """The tool point at fraction ``s`` of the path's length, the path's unit tangent
    there and a unit horizontal normal."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    k = int(np.clip(np.searchsorted(arc, s * arc[-1]), 1, len(P) - 2))
    T = P[k + 1] - P[k - 1]
    T /= np.linalg.norm(T)
    N = np.cross(T, [0, 0, 1.0])
    N = N / np.linalg.norm(N) if np.linalg.norm(N) > 1e-6 else np.array([1.0, 0, 0])
    return P[k], T, N


def place(kind, P, s, side):
    """The obstacle's params for rung ``kind`` at fraction ``s`` of the tool path, on
    side ``side`` (+1 or -1) of it."""
    p, T, N = tool_frame(P, s)
    rt = RADII[TOOL]
    if kind == "ograze":
        dist = (GRAZE_R + rt) * np.sqrt(1 - GRAZE_G)
        return [*(p + side * dist * N), GRAZE_R]
    if kind == "oblock":
        return [*(p + side * BLOCK_SIDE * (BLOCK_R + rt) * N), BLOCK_R]
    a, b, c = WALL
    th = np.array([T[0], T[1], 0.0])
    th = th / np.linalg.norm(th) if np.linalg.norm(th) > 1e-6 else np.array([1.0, 0, 0])
    yaw = float(np.arctan2(th[1], th[0]))
    across = np.array([-th[1], th[0], 0.0])             # the box's width axis
    centre = p + side * WALL_SIDE * (b + rt) * across - WALL_DOWN * (c + rt) * np.array([0, 0, 1.0])
    return [*centre, a, b, c, yaw]


def obstacle_kind(name):
    return "sphere" if name.startswith(("ograze", "oblock")) else "box"


def choose(kind, Q):
    """The first placement whose end poses are clear, with ``g`` along the rest answer
    ``Q`` (samples x joints): ``(params, g)`` or ``None``."""
    P = np.array([body_points_numeric(SIX_JOINT_LINKS, q)[TOOL] for q in Q])
    shape = obstacle_kind(kind)
    for s in FRACTIONS:
        for side in (1, -1):
            obs = place(kind, P, s, side)
            g = obstacle_values(SIX_JOINT_LINKS, Q, shape, obs)
            if g[0].max() < CLEAR and g[-1].max() < CLEAR and g.max() > 0:
                return obs, g
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).with_name("obstacle_cases.json")))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--bases", nargs="*", default=[f"rest{i}" for i in range(15)])
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
    for base in args.bases:
        p = arm_census.params(base)
        r = arm_census.H.solve(p, arm_census.BOX, arm_census.TS)
        if not r.exact:
            print(f"{base}: not solved, no obstacle case ({r.message[:60]})", flush=True)
            continue
        Q = np.asarray(r.shooting.trajectory(r.solution.v, n_points=4001)["z"])[:, :6]
        for kind in ("ograze", "oblock", "owall"):
            found = choose(kind, Q)
            name = kind + base[4:]
            if found is None:
                print(f"{name}: no placement clears the end poses", flush=True)
                continue
            obs, g = found
            cases[name] = {"base": base, "obstacle": obstacle_kind(kind),
                           "tf_base": float(r.solution.tf), "g_base": float(g.max()),
                           "params": [*p, *map(float, obs)]}
            print(f"{name}: {obstacle_kind(kind)} at {np.round(obs[:3], 3).tolist()}, "
                  f"the rest answer reaches g = {g.max():.3f}", flush=True)
        Path(args.out).write_text(json.dumps(cases, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
