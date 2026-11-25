"""Export census cases for the animated census page (``docs/_static/census``).

Each case becomes one JSON file in the export directory:

- its name, family, setting and outcome;
- for a solved case, the motion: times, joint angles, every link's origin in
  the base frame, and the joint torques;
- for an unsolved one, the start and end poses only.

``manifest.json`` lists every case file in the directory, so a case exported
later (a new census, a new family of cases) shows up on the page with no change
to the page itself.
"""

import json
import os
import re
from pathlib import Path

import numpy as np

from auto_optimal_control.robot import (BOX_POWER, SIX_JOINT_BODY, SIX_JOINT_LINKS, body_points_numeric,
                                        link_frames, obstacle_values)

N_SAMPLES = 160


def _points(q):
    """The base and every link's origin at joint angles ``q``: a list of [x, y, z]."""
    return [[0.0, 0.0, 0.0]] + [p.tolist() for _, p in link_frames(SIX_JOINT_LINKS, q)]


def _round(a, digits=4):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def _describe(name):
    m = re.fullmatch(r"([a-z]+?)(\d+)(?:@(\d+))?", name)
    family, index, pct = (m.group(1), int(m.group(2)), m.group(3)) if m else (name, 0, None)
    return {"family": family, "index": index, "setting": f"{pct}%" if pct else None}


def export_case(out_dir, name, params, row, result=None, torques=None, rates=None,
                positions=None, obstacle=None):
    """Write ``<out_dir>/<name>.json``.

    ``params`` is the case's parameter vector ([q0, qf, jerk limits, torque limits],
    then rate limits for a rate case, then ``q_min``, ``q_max`` for a position case),
    ``row`` the census row (``exact``, ``sec``, ``tf``, ``msg``, and for a rate or
    position case ``orders``: which limits the answer holds), ``result`` the solve result
    when exact, and ``torques(Q, W, A)`` the joint torques at sampled states. ``rates``
    and ``positions`` say which limits the case has (by default: rates when there are
    30 parameters, positions never); ``positions`` may be the list of joints whose
    position limits bind. ``obstacle`` (``"sphere"`` or ``"box"``) says the case's last
    params are an obstacle's (:func:`~auto_optimal_control.robot.obstacle_limits`): the
    file then gives the obstacle, the body points that must clear it and, for a solved
    case, each point's ``g`` along the motion (``clearance``)."""
    p = np.asarray(params, dtype=float)
    q0, qf, jerk, tau = p[:6], p[6:12], p[12:18], p[18:24]
    if rates is None:
        rates = len(p) == 30
    rates = p[24:30] if rates else None
    off = 30 if rates is not None else 24
    pos = None
    if positions is not None and positions is not False:
        pos = (p[off:off + 6], p[off + 6:off + 12])
    data = {"name": name, **_describe(name), "solved": bool(row.get("exact")),
            "seconds": row.get("sec"), "tf": row.get("tf"), "message": row.get("msg"),
            "jerk_limits": _round(jerk), "torque_limits": _round(tau),
            "start": _round(_points(q0)), "goal": _round(_points(qf))}
    if rates is not None:
        data["rate_limits"] = _round(rates)
    if pos is not None:
        data["position_limits"] = {"min": _round(pos[0]), "max": _round(pos[1]),
                                   "joints": list(positions) if positions is not True
                                   else list(range(6))}
    obs = None
    if obstacle is not None:
        obs = p[-4:] if obstacle == "sphere" else p[-7:]
        data["obstacle"] = {"kind": obstacle, "params": _round(obs, 5), "power": BOX_POWER,
                            "radii": [float(r) for *_, r in SIX_JOINT_BODY],
                            "body": [n for n, *_ in SIX_JOINT_BODY]}
        data["start_body"] = _round(body_points_numeric(SIX_JOINT_LINKS, q0))
        data["goal_body"] = _round(body_points_numeric(SIX_JOINT_LINKS, qf))
    if rates is not None or pos is not None or obs is not None:
        data["orders"] = row.get("orders")
    if result is not None and row.get("exact"):
        tr = result.shooting.trajectory(result.solution.v, n_points=N_SAMPLES)
        z = np.asarray(tr["z"], dtype=float)
        Q, W, A = z[:, :6], z[:, 6:12], z[:, 12:18]
        data["t"] = _round(tr["t"], 5)
        data["q"] = _round(Q)
        data["points"] = [_round(_points(q)) for q in Q]
        if torques is not None:
            data["tau"] = _round(torques(Q, W, A))
        if rates is not None:
            data["w"] = _round(W)
        if obs is not None:
            data["body_points"] = [_round(body_points_numeric(SIX_JOINT_LINKS, q)) for q in Q]
            data["clearance"] = _round(obstacle_values(SIX_JOINT_LINKS, Q, obstacle, obs))
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
        cases.append({"file": f.name, "name": d["name"], "family": d["family"],
                      "index": d["index"], "setting": d["setting"], "solved": d["solved"],
                      "seconds": d["seconds"], "tf": d["tf"], "rates": "rate_limits" in d,
                      "positions": "position_limits" in d,
                      "obstacle": d.get("obstacle", {}).get("kind")})
    (out / "manifest.json").write_text(json.dumps({"cases": cases}, indent=1))
    return len(cases)
