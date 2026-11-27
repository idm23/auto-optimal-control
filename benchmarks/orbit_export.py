"""Export orbit census cases for the census page (``docs/_static/orbit_census``).

Each case becomes one JSON file in the export directory:

- its name, family, setting and outcome, the thrust, and the start and target orbits
  (their elements, and each orbit drawn as a closed curve);
- for a solved case, the transfer: times, the position (in Earth radii, in the
  equatorial frame), the thrust direction there, the throttle, and the elements
  ``a, e, i`` and the mass along the way.

``manifest.json`` lists every case file in the directory, as the arm census does.
"""

import json
import os
import re
from pathlib import Path

import numpy as np

from orbitprop import EARTH, CanonicalUnits

UNITS = CanonicalUnits.of(EARTH)
N_PER_REV = 72          # position samples per revolution
N_MAX = 9000            # position samples at most
N_ELEMENTS = 400        # element samples
N_ORBIT = 180           # points on the start and target orbits


def _round(a, digits=4):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def _describe(name):
    m = re.fullmatch(r"([a-z]+?)(\d+)", name)
    return {"family": m.group(1), "index": int(m.group(2))} if m else {"family": name, "index": 0}


def frames(z):
    """The equinoctial frame and the radial / transverse / normal frame at each row of
    ``z = (p, f, g, h, k, L, ...)``: ``(f_hat, g_hat, w_hat, i_r, i_t)``, each ``(N, 3)``."""
    h, k, L = z[:, 3], z[:, 4], z[:, 5]
    s2 = 1 + h ** 2 + k ** 2
    fh = np.stack([(1 + h ** 2 - k ** 2), 2 * h * k, -2 * k], 1) / s2[:, None]
    gh = np.stack([2 * h * k, (1 - h ** 2 + k ** 2), 2 * h], 1) / s2[:, None]
    wh = np.stack([2 * k, -2 * h, (1 - h ** 2 - k ** 2)], 1) / s2[:, None]
    c, s = np.cos(L)[:, None], np.sin(L)[:, None]
    return fh, gh, wh, c * fh + s * gh, -s * fh + c * gh


def positions(z):
    """The position at each row of ``z`` (in units of ``p``)."""
    p, f, g, L = z[:, 0], z[:, 1], z[:, 2], z[:, 5]
    _, _, _, ir, _ = frames(z)
    return (p / (1 + f * np.cos(L) + g * np.sin(L)))[:, None] * ir


def keplerian(z):
    """``a, e, i`` at each row of ``z``."""
    p, f, g, h, k = z[:, 0], z[:, 1], z[:, 2], z[:, 3], z[:, 4]
    e = np.hypot(f, g)
    return p / (1 - e ** 2), e, 2 * np.arctan(np.hypot(h, k))


def orbit_curve(elements5):
    """The closed orbit of ``(p, f, g, h, k)``, as ``N_ORBIT`` points."""
    L = np.linspace(0, 2 * np.pi, N_ORBIT + 1)
    z = np.column_stack([np.tile(np.asarray(elements5, dtype=float), (len(L), 1)), L])
    return positions(z)


def export_case(out_dir, name, case, row, result=None):
    """Write ``<out_dir>/<name>.json``.

    ``case`` is the census case (:func:`orbit_census.case`): its ``family``, ``j2``,
    ``objective``, thrust ``T``, ``params``, and the start and target orbits ``a, e, i``;
    ``row`` the census row; ``result`` the solve result when exact."""
    p = np.asarray(case["params"], dtype=float)
    x0, xf = p[3:9], p[9:14]
    # The family is the case's, J2 or not (``jraise3`` is a ``raise`` case).
    data = {"name": name, **_describe(name), "family": case["family"], "setting": case["setting"],
            "solved": bool(row.get("exact")), "seconds": row.get("sec"), "tf": row.get("tf"),
            "message": row.get("msg"), "j2": case["j2"], "objective": case["objective"],
            "thrust": case["T"], "exhaust": case["c"],
            "thrust_mm_s2": case["T"] * UNITS.acceleration * 1e6,
            "isp_s": case["c"] * UNITS.speed * 1e3 / 9.80665,
            "time_unit_s": UNITS.time, "length_unit_km": UNITS.length,
            "from": dict(zip("aei", case["from"])), "to": dict(zip("aei", case["to"])),
            "start_orbit": _round(orbit_curve(x0[:5])), "target_orbit": _round(orbit_curve(xf))}
    if case["objective"] == "fuel":
        data["tf_fixed"] = float(p[-1])
    if result is not None and row.get("exact"):
        tr = result.shooting.trajectory(result.solution.v, n_points=2)
        revs = abs(float(tr["z"][-1][5] - tr["z"][0][5])) / (2 * np.pi)
        n = int(min(N_MAX, max(400, N_PER_REV * revs)))
        tr = result.shooting.trajectory(result.solution.v, n_points=n)
        z, u = np.asarray(tr["z"], dtype=float), np.asarray(tr["u"], dtype=float)
        _, _, wh, ir, it = frames(z)
        thrust = u[:, :1] * ir + u[:, 1:2] * it + u[:, 2:3] * wh
        data.update(t=_round(tr["t"], 4), r=_round(positions(z)), u=_round(thrust, 3),
                    revolutions=revs, mass_final=float(z[-1, 6]))
        if u.shape[1] > 3:
            data["throttle"] = _round(u[:, 3], 3)
        step = max(1, n // N_ELEMENTS)
        a, e, i = keplerian(z[::step])
        us = u[::step]
        data["elements"] = {"t": _round(tr["t"][::step], 4), "a": _round(a, 5), "e": _round(e, 5),
                            "i": _round(np.degrees(i), 4), "m": _round(z[::step, 6], 5),
                            # the steering: in the orbit's plane from the transverse
                            # direction towards the radial, and out of the plane
                            "alpha": _round(np.degrees(np.arctan2(us[:, 0], us[:, 1])), 2),
                            "beta": _round(np.degrees(np.arcsin(np.clip(us[:, 2], -1, 1))), 2)}
        if u.shape[1] > 3:
            data["elements"]["throttle"] = _round(us[:, 3], 3)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".{name}.json.tmp"
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    os.replace(tmp, out / f"{name}.json")


def write_manifest(out_dir):
    """List every case file in ``out_dir`` in ``manifest.json``."""
    out = Path(out_dir)
    cases = []
    for f in sorted(out.glob("*.json")):
        if f.name == "manifest.json":
            continue
        d = json.loads(f.read_text())
        cases.append({k: d.get(k) for k in ("name", "family", "index", "setting", "solved",
                                             "seconds", "tf", "j2", "objective", "thrust",
                                             "revolutions", "mass_final")} | {"file": f.name})
    (out / "manifest.json").write_text(json.dumps({"cases": cases}, indent=1))
    return len(cases)
