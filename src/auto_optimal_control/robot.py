"""Serial robot arms: joint torques from a link table, on SymPy symbols.

:func:`joint_torques` is the recursive Newton-Euler algorithm (Luh, Walker &
Paul, 1980) and works on symbols without simplifying anything. Written out,
its result for six joints has about a hundred thousand operations; as a graph
it has about a thousand, which is how :func:`~auto_optimal_control.pontryagin.derive`
then treats it (:mod:`~auto_optimal_control.tape`). :func:`jerk_controlled_arm`
turns a link table into an optimal control problem.
"""

from dataclasses import dataclass

import numpy as np
import sympy as sp

from auto_optimal_control.problem import Box, OptimalControlProblem, PathConstraint


@dataclass(frozen=True)
class Link:
    """A revolute joint and the rigid link after it, in standard Denavit-Hartenberg form.

    The frame of link ``i`` comes from the frame before it by a rotation
    ``q_i`` about ``z``, then ``d`` along ``z``, ``a`` along the new ``x`` and a
    twist ``alpha`` about it, which must be a multiple of ``pi / 2`` so that
    no float enters the symbols. ``mass``, ``com`` (the centre of mass, in
    this link's frame) and ``inertia`` (the principal moments about the
    centre of mass, along the axes of this frame) describe the link.
    """

    a: object
    d: object
    alpha: object
    mass: object
    com: tuple
    inertia: tuple


def _twist(alpha):
    quarter = sp.nsimplify(alpha / (sp.pi / 2))
    if not quarter.is_Integer:
        raise ValueError(f"alpha must be a multiple of pi / 2, got {alpha}")
    return {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}[int(quarter) % 4]


def joint_torques(links, q, w, a, gravity=0):
    """The torques ``tau = M(q) a + c(q, w) + G(q)`` of a serial arm, as SymPy expressions.

    ``q``, ``w`` and ``a`` are the joint angles, rates and accelerations and
    ``gravity`` the acceleration of gravity, along ``-z`` of the base frame.
    """
    z = sp.Matrix([0, 0, 1])
    rate, accel_w = sp.zeros(3, 1), sp.zeros(3, 1)
    accel = sp.Matrix([0, 0, gravity])       # gravity, as an upward acceleration of the base
    rotations, offsets, coms, forces, moments = [], [], [], [], []
    for link, qi, wi, ai in zip(links, q, w, a):
        ca, sa = _twist(link.alpha)
        c, s = sp.cos(qi), sp.sin(qi)
        rotation = sp.Matrix([[c, -s * ca, s * sa], [s, c * ca, -c * sa], [0, sa, ca]])
        offset = sp.Matrix([link.a, link.d * sa, link.d * ca])
        com = sp.Matrix(link.com)
        back = rotation.T
        new_rate = back * (rate + z * wi)
        accel_w = back * (accel_w + z * ai + rate.cross(z * wi))
        rate = new_rate
        accel = back * accel + accel_w.cross(offset) + rate.cross(rate.cross(offset))
        at_com = accel + accel_w.cross(com) + rate.cross(rate.cross(com))
        inertia = sp.diag(*link.inertia)
        forces.append(link.mass * at_com)
        moments.append(inertia * accel_w + rate.cross(inertia * rate))
        rotations.append(rotation)
        offsets.append(offset)
        coms.append(com)
    tau = [sp.S.Zero] * len(links)
    force, moment = sp.zeros(3, 1), sp.zeros(3, 1)
    for i in reversed(range(len(links))):
        if i + 1 < len(links):
            force, moment = rotations[i + 1] * force, rotations[i + 1] * moment
        moment = (moments[i] + moment + (offsets[i] + coms[i]).cross(forces[i])
                  + offsets[i].cross(force))
        force = forces[i] + force
        tau[i] = (moment.T * (rotations[i].T * z))[0]
    return tau


def link_frames(links, q):
    """Each link's rotation and origin in the base frame at joint angles ``q`` (NumPy)."""
    R, p, out = np.eye(3), np.zeros(3), []
    for link, qi in zip(links, q):
        ca, sa = _twist(link.alpha)
        c, s = np.cos(qi), np.sin(qi)
        p = p + R @ np.array([float(link.a) * c, float(link.a) * s, float(link.d)])
        R = R @ np.array([[c, -s * ca, s * sa], [s, c * ca, -c * sa], [0, sa, ca]], dtype=float)
        out.append((R, p))
    return out


# A six-joint arm: link lengths and masses from the published figures for a UR5,
# centres of mass and inertias rough estimates. An arm of that size, not a model of it.
_R = sp.Rational
SIX_JOINT_LINKS = (
    Link(0, _R(89159, 10 ** 6), sp.pi / 2, _R(37, 10), (0, _R(-2, 100), 0),
         (_R(1, 100), _R(1, 100), _R(1, 100))),
    Link(_R(-425, 1000), 0, 0, _R(8393, 1000), (_R(2125, 10000), 0, _R(11, 100)),
         (_R(2, 100), _R(13, 100), _R(13, 100))),
    Link(_R(-39225, 10 ** 5), 0, 0, _R(2275, 1000), (_R(11, 100), 0, _R(1, 100)),
         (_R(1, 200), _R(4, 100), _R(4, 100))),
    Link(0, _R(10915, 10 ** 5), sp.pi / 2, _R(1219, 1000), (0, _R(2, 1000), 0),
         (_R(3, 1000), _R(3, 1000), _R(2, 1000))),
    Link(0, _R(9465, 10 ** 5), -sp.pi / 2, _R(1219, 1000), (0, _R(-2, 1000), 0),
         (_R(3, 1000), _R(3, 1000), _R(2, 1000))),
    Link(0, _R(823, 10 ** 4), 0, _R(1879, 10 ** 4), (0, 0, _R(-2, 100)),
         (_R(1, 10 ** 4), _R(1, 10 ** 4), _R(1, 10 ** 4))),
)
GRAVITY = _R(981, 100)


def jerk_controlled_arm(links, *, gravity=GRAVITY, name="arm"):
    """Rest to rest in a fixed time, minimum jerk, with a limit on each joint's torque.

    States: the joint angles ``q``, rates ``w`` and accelerations ``a``; the
    controls are the jerks ``j``. From rest at ``q0`` to rest at ``qf`` in
    time ``T``, minimizing ``int sum j_i^2 / 2``, with ``|tau_i| <= tau_max_i``.

    params: ``T``, the ``q0``, the ``qf``, the ``tau_max``.
    """
    n = len(links)
    t = sp.Symbol("t", real=True)
    q, w, a, j = (sp.symbols(f"{s}1:{n + 1}", real=True) for s in ("q", "w", "a", "j"))
    T = sp.Symbol("T", positive=True)
    q0, qf = (sp.symbols(f"{s}1:{n + 1}", real=True) for s in ("q0_", "qf_"))
    limits = sp.symbols(f"tau_max1:{n + 1}", positive=True)
    tau = joint_torques(links, q, w, a, gravity)
    constraints = []
    for i in range(n):
        constraints += [PathConstraint(tau[i] - limits[i], f"tau{i + 1}_max"),
                        PathConstraint(-tau[i] - limits[i], f"tau{i + 1}_min")]
    start = {**dict(zip(q, q0)), **{s: 0 for s in (*w, *a)}}
    final = {**dict(zip(q, qf)), **{s: 0 for s in (*w, *a)}}
    return OptimalControlProblem(
        name=name, time=t, states=[*q, *w, *a], controls=list(j),
        params=[T, *q0, *qf, *limits], dynamics=[*w, *a, *j],
        running_cost=sum(u ** 2 for u in j) / 2, initial_state=start, final_state=final,
        final_time=T, path_constraints=constraints)


def _ends(q, w, a, moving_ends):
    """Initial and final states, and the parameters they add: rest at ``q0`` and ``qf``,
    or any state at both ends (``w0, a0, wf, af`` too) with ``moving_ends``."""
    n = len(q)
    q0, qf = (sp.symbols(f"{s}1:{n + 1}", real=True) for s in ("q0_", "qf_"))
    if not moving_ends:
        start = {**dict(zip(q, q0)), **{s: 0 for s in (*w, *a)}}
        final = {**dict(zip(q, qf)), **{s: 0 for s in (*w, *a)}}
        return start, final, [*q0, *qf]
    w0, a0, wf, af = (sp.symbols(f"{s}1:{n + 1}", real=True) for s in ("w0_", "a0_", "wf_", "af_"))
    start = {**dict(zip(q, q0)), **dict(zip(w, w0)), **dict(zip(a, a0))}
    final = {**dict(zip(q, qf)), **dict(zip(w, wf)), **dict(zip(a, af))}
    return start, final, [*q0, *w0, *a0, *qf, *wf, *af]


def joint_limits(q, w, *, rates=False, positions=False):
    """Limits on the joint rates and angles: ``(constraints, params)``.

    ``rates`` adds ``|w_i| <= w_max_i``, ``positions`` adds ``q_min_i <= q_i
    <= q_max_i``, each with its bounds as params (``w_max1..n``, then
    ``q_min1..n, q_max1..n``). With jerk control a rate limit has order 2
    and an angle limit order 3: with a smooth control law the latter is only
    ever touched, never ridden (Jacobson, Lele & Speyer 1971).
    """
    n = len(q)
    constraints, params = [], []
    if rates:
        w_max = sp.symbols(f"w_max1:{n + 1}", positive=True)
        for i in range(n):
            constraints += [PathConstraint(w[i] - w_max[i], f"w{i + 1}_max"),
                            PathConstraint(-w[i] - w_max[i], f"w{i + 1}_min")]
        params += list(w_max)
    if positions:
        q_min, q_max = (sp.symbols(f"{s}1:{n + 1}", real=True) for s in ("q_min", "q_max"))
        for i in range(n):
            constraints += [PathConstraint(q[i] - q_max[i], f"q{i + 1}_max"),
                            PathConstraint(q_min[i] - q[i], f"q{i + 1}_min")]
        params += [*q_min, *q_max]
    return constraints, params


def jerk_limited_arm(links, *, gravity=GRAVITY, name="arm_min_time", moving_ends=False,
                     rate_limits=False, position_limits=False):
    """Rest to rest in minimum time, with bounded jerks and a limit on each joint's torque.

    ``min t_f`` subject to ``|j_i| <= j_max_i`` and ``|tau_i| <= tau_max_i``, from rest at
    ``q0`` to rest at ``qf``. As stated it is degenerate
    (:mod:`~auto_optimal_control.policy`): only the slowest joint sets the time, so the
    default policy of :class:`~auto_optimal_control.homotopy.Homotopy` adds a small
    jerk cost.

    params: the ``q0``, the ``qf``, the ``j_max``, the ``tau_max``. With
    ``moving_ends`` the motion runs between any two states instead of from
    rest to rest: the params begin ``q0, w0, a0, qf, wf, af``. With
    ``rate_limits`` or ``position_limits`` the joint limits of
    :func:`joint_limits` follow, with their params last.
    """
    n = len(links)
    t = sp.Symbol("t", real=True)
    q, w, a, j = (sp.symbols(f"{s}1:{n + 1}", real=True) for s in ("q", "w", "a", "j"))
    start, final, ends = _ends(q, w, a, moving_ends)
    j_max = sp.symbols(f"j_max1:{n + 1}", positive=True)
    limits = sp.symbols(f"tau_max1:{n + 1}", positive=True)
    tau = joint_torques(links, q, w, a, gravity)
    constraints = []
    for i in range(n):
        constraints += [PathConstraint(tau[i] - limits[i], f"tau{i + 1}_max"),
                        PathConstraint(-tau[i] - limits[i], f"tau{i + 1}_min")]
    more, more_params = joint_limits(q, w, rates=rate_limits, positions=position_limits)
    inertias_positive = all(sp.sympify(m).is_positive is True for link in links for m in link.inertia)
    certified = {
        "mass_matrix_positive_definite": (
            inertias_positive,
            "Lemma A: every link's principal moments of inertia are positive, so the kinetic "
            "energy is positive for any nonzero joint rate"),
        "limit_control_rows": (
            True, {f"tau{i + 1}_{side}": (i, 1 if side == "max" else -1)
                   for i in range(n) for side in ("max", "min")}),
        "limits_time_reversible": (
            "Newton-Euler torques M(q) a + c(q, w) + G(q) with c quadratic in w: unchanged by "
            "w -> -w; rate limits come in +- pairs, angle limits do not involve w",
            {f"tau{i + 1}_{side}": f"tau{i + 1}_{side}" for i in range(n) for side in ("max", "min")}),
    }
    return OptimalControlProblem(
        name=name, time=t, states=[*q, *w, *a], controls=list(j),
        params=[*ends, *j_max, *limits, *more_params], dynamics=[*w, *a, *j], running_cost=1,
        initial_state=start, final_state=final, path_constraints=constraints + more,
        control_sets=[Box(u, -m, m) for u, m in zip(j, j_max)], certified=certified)
