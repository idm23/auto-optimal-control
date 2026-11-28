"""The orbit transfer census: low-thrust transfers around the Earth, with what each ran into.

    python benchmarks/orbit_census.py OUT.jsonl [--cases raise3 jgto1 ...] [--cap 300]
        [--parallel-routes] [--cache DIR] [--rule R2=on ...] [--export DIR]

A ladder of minimum-time transfers (:func:`auto_optimal_control.orbits.low_thrust_transfer`),
each family in order of falling thrust, so of more and more revolutions:

- ``raise<i>``: a coplanar raise between circular equatorial orbits, from 1.1 to 1.3
  Earth radii (638 to 1913 km up);
- ``incl<i>``: the same raise with a 10 degree plane change;
- ``gto<i>``: from a geostationary transfer orbit (perigee 250 km, apogee 35,786 km,
  7 degrees) to the geostationary orbit.

Each case is two-body; ``j`` before its name (``jraise3``) adds the Earth's J2. The thrust
``T`` is the initial thrust acceleration in units of surface gravity, and the exhaust
speed is that of a 3000 s ion engine throughout. The final time is searched in [0.3, 4]
times Edelbaum's estimate ``dv / T`` and the costates in a box of half-width ``1 / T``
(a minimum-time costate is how the arrival time moves with the elements). The page's
rule configuration (``RULES``) is the arm census's of 2026-11-25, before R90, with R91
(``FUEL_RULES``) on for minimum fuel.

``--cap`` is the wall-clock budget of a case (``Homotopy.solve_seconds``) and a case over
it is a failure. ``--export`` writes each case for the census page
(``docs/orbit_census.md``) with a manifest of the directory.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from auto_optimal_control import rules
from auto_optimal_control.homotopy import Homotopy
from auto_optimal_control.orbits import low_thrust_transfer
from orbitprop import EARTH, CanonicalUnits, keplerian_to_equinoctial

sys.path.insert(0, str(Path(__file__).resolve().parent))   # importable from anywhere
from orbit_export import export_case, write_manifest  # noqa: E402

RULES = tuple(("R1 R2 R15 R16 R20 R25 R26 R29 R32 R41 R43 R48 R50 R54 R55 R60 R61 R63 R64 R65 "
               "R66 R28 R72 R74 R75 R78").split())
# and for minimum fuel, the throttle's early candidates (R91, Observation O14)
FUEL_RULES = ("R91",)
UNITS = CanonicalUnits.of(EARTH)
C = 3000 * 9.80665 / 1000 / UNITS.speed          # Isp 3000 s
R = EARTH.radius
# (a in Earth radii, e, i in degrees) from and to, and the thrust ladder
FAMILIES = {
    "raise": (((1.1, 0, 0), (1.3, 0, 0)),
              (0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001)),
    "incl": (((1.1, 0, 10), (1.3, 0, 0)), (0.03, 0.01, 0.003, 0.001, 0.0003)),
    "gto": ((((2 * R + 250 + 35786) / 2 / R, (35786 - 250) / (2 * R + 250 + 35786), 7),
             (42164.137 / R, 0, 0)), (0.01, 0.003, 0.001, 0.0003)),
}
# Minimum fuel: the coplanar raise at three of its rungs, with t_f fixed at multiples of
# the rung's minimum time (the census of 2026-11-26, two-body and with J2).
FUEL_RUNGS = (0, 2, 4)                  # raise0, raise2, raise4: 0.05, 0.01, 0.002 g
FUEL_FACTORS = (1.2, 1.5, 2.0)
MIN_TIME = {(0, False): 3.5836497325604464, (2, False): 7.776290810352053,
            (4, False): 38.302137443747554, (0, True): 3.5877499037007574,
            (2, True): 7.789408936905352, (4, True): 38.379028737254295}
H = {}
# Exact answers of this run, by case: a J2 minimum-fuel case starts from its two-body
# twin's (``Homotopy.solve(start=)``), so the twin must come first.
SOLVED = {}
ARGS = None


def edelbaum(a0, i0, af, i_f):
    """Edelbaum's dv between circular orbits of radii ``a0``, ``af`` and inclinations
    ``i0``, ``i_f`` (degrees)."""
    v0, vf = a0 ** -0.5, af ** -0.5
    return float(np.sqrt(v0 ** 2 + vf ** 2
                         - 2 * v0 * vf * np.cos(np.pi / 2 * np.radians(abs(i_f - i0)))))


def case_names():
    sizes = {**{fam: len(ladder) for fam, (_, ladder) in FAMILIES.items()},
             "fuel": len(FUEL_RUNGS) * len(FUEL_FACTORS)}
    return [f"{j}{fam}{i}" for fam, n in sizes.items() for j in ("", "j") for i in range(n)]


def case(name):
    """The census case ``name``: its family, J2, thrust, parameters and search boxes."""
    known = set(FAMILIES) | {"fuel"}
    j2 = name.startswith("j") and name[1:].rstrip("0123456789") in known
    base = name[1:] if j2 else name
    fam = base.rstrip("0123456789")
    if fam not in known or fam == base:
        raise KeyError(f"no census case {name!r}")
    if fam == "fuel":
        return _fuel_case(int(base[len(fam):]), j2)
    (start, target), ladder = FAMILIES[fam]
    i = int(base[len(fam):])
    T = ladder[i]
    (a0, e0, i0), (af, ef, i_f) = start, target
    x0 = keplerian_to_equinoctial(a0, e0, np.radians(i0), 0, 0, 0)
    xf = keplerian_to_equinoctial(af, ef, np.radians(i_f), 0, 0, 0)[:5]
    # From an eccentric orbit, Edelbaum from its perigee: a rough guess, inside the box.
    dv = edelbaum(a0 * (1 - e0), i0, af, i_f)
    est = dv / T
    return {"family": fam, "j2": j2, "objective": "time", "T": T, "c": C,
            "setting": f"T = {T:g} g" + (", J2" if j2 else ""),
            "from": list(start), "to": list(target),
            "params": [T, C, EARTH.J[2] if j2 else 0.0, *x0, *xf], "xf": xf.tolist(),
            "tf_estimate": est, "costates": ([-1 / T] * 7, [1 / T] * 7),
            "final_time": (0.3 * est, 4 * est)}


def _fuel_case(i, j2):
    """``fuel<i>``: the raise at rung ``FUEL_RUNGS[i // 3]`` with the throttle free in [0, 1]
    and ``t_f`` fixed at ``FUEL_FACTORS[i % 3]`` times that rung's minimum time. The
    costates are in a box of half-width ``1 / c``: a minimum-fuel costate is how the
    propellant spent moves with the elements, and the thrust's cost is ``T delta / c``."""
    if not 0 <= i < len(FUEL_RUNGS) * len(FUEL_FACTORS):
        raise KeyError(f"no census case {'j' if j2 else ''}fuel{i}")
    rung, factor = FUEL_RUNGS[i // len(FUEL_FACTORS)], FUEL_FACTORS[i % len(FUEL_FACTORS)]
    base = case(("j" if j2 else "") + f"raise{rung}")
    t_min = MIN_TIME[(rung, j2)]
    tf = factor * t_min
    T = base["T"]
    return {**base, "family": "fuel", "objective": "fuel",
            "setting": f"T = {T:g} g, {factor:g}× the minimum time" + (", J2" if j2 else ""),
            "params": [*base["params"], tf], "tf_min": t_min, "tf_factor": factor,
            "mass_min_time": 1 - T * t_min / C, "tf_estimate": tf,
            "costates": ([-1 / C] * 7, [1 / C] * 7), "final_time": None}


def _homotopy(objective):
    """The census's solver for ``objective``, built on first use (outside a case's time):
    minimum fuel's exact kernels carry the throttle's singular arcs, minutes to
    generate, so they are built here too."""
    if objective not in H:
        args = ARGS
        h = Homotopy(low_thrust_transfer(objective=objective, zonal_degree=2),
                     cache_dir=args.cache)
        for r in RULES + (FUEL_RULES if objective == "fuel" else ()):
            rules.set_rule(h, r, True)
        for item in args.rule or []:
            name, state = item.split("=", 1)
            rules.set_rule(h, name, state.lower() in ("on", "1", "true"))
        h.solve_seconds = args.cap
        h.active_set_seconds = args.cap
        h.parallel_routes = args.parallel_routes
        if h.bang_bang:
            h.exact_kernels
        H[objective] = h
        print(f"{objective}: rules on:",
              ", ".join(n for n, on in rules.enabled(h).items() if on), flush=True)
    return H[objective]


def setup(args):
    global ARGS
    ARGS = args
    warnings.simplefilter("ignore")


def run(name):
    c, r = None, None
    try:
        c = case(name)
        h = _homotopy(c["objective"])
        h.snapshot_tag = name
        t0 = time.time()
        twin = SOLVED.get(name[1:]) if c["objective"] == "fuel" and c["j2"] else None
        r = h.solve(c["params"], c["costates"], c["final_time"], start=twin)
        if r.exact:
            SOLVED[name] = r
        row = {"name": name, "exact": bool(r.exact), "sec": round(time.time() - t0, 3),
               "tf": float(r.solution.tf) if r.exact else None, "msg": r.message[:80],
               "tf_estimate": c["tf_estimate"], "diagnostics": r.diagnostics}
        if row["exact"] and row["sec"] > ARGS.cap:
            row.update(exact=False, tf=None, msg=f"over the {ARGS.cap:g} s cap (solved in {row['sec']} s)")
        if row["exact"]:
            tr = r.shooting.trajectory(r.solution.v, n_points=2000)
            z, u = np.asarray(tr["z"]), np.asarray(tr["u"])
            row.update(revolutions=float((z[-1, 5] - z[0, 5]) / (2 * np.pi)),
                       mass_final=float(z[-1, 6]),
                       end_error=float(np.abs(z[-1, :5] - np.asarray(c["xf"])).max()),
                       days=float(r.solution.tf) * UNITS.time / 86400)
            if c["objective"] == "fuel":
                on = u[:, 3] > 0.5
                row.update(switches=int(np.count_nonzero(on[1:] != on[:-1])),
                           mass_min_time=c["mass_min_time"])
    except Exception as exc:          # a case that raises is a result too
        row = {"name": name, "exact": False, "sec": 0.0, "tf": None, "msg": repr(exc)[:80],
               "diagnostics": {}}
    if ARGS.export and c is not None:
        try:
            export_case(ARGS.export, name, c, row, r if row["exact"] else None)
        except Exception as exc:      # the census result stands without the page's copy
            print(f"export of {name} failed: {exc!r}", flush=True)
    return row


def summary(rows):
    ok = [r for r in rows if r["exact"]]
    secs = sorted(r["sec"] for r in ok)
    print(f"solved {len(ok)} of {len(rows)}; under 60 s: {sum(s < 60 for s in secs)}; "
          f"mean {sum(secs) / len(secs) if secs else float('nan'):.3f} s")
    for fam in (*FAMILIES, "fuel"):
        for j in ("", "j"):
            part = [r for r in rows if r["name"].rstrip("0123456789") == j + fam]
            if part:
                ok_ = [r for r in part if r["exact"]]
                most = max((r.get("revolutions", 0) for r in ok_), default=0)
                print(f"  {j + fam}: solved {len(ok_)} of {len(part)}; up to {most:.1f} revolutions")
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
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--cap", type=float, default=300.0,
                    help="wall-clock seconds per case; a case over it is a failure")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--rule", action="append")
    ap.add_argument("--export", default=None,
                    help="write each case for the census page to this directory "
                         "(docs/_static/orbit_census/data)")
    ap.add_argument("--parallel-routes", action="store_true",
                    help="run R15's routes at once (one core per route)")
    args = ap.parse_args(argv)
    setup(args)
    names = args.cases or case_names()
    rows = []
    with open(args.out, "w") as fo:
        for n in names:
            row = run(n)
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "diagnostics"}), flush=True)
            fo.write(json.dumps(row) + "\n")
            fo.flush()
    summary(rows)
    if args.export:
        print(f"exported {write_manifest(args.export)} cases to {args.export}")


if __name__ == "__main__":
    main()
