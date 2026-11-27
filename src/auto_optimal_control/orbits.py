"""Low-thrust transfers between orbits around a body, on ``orbitprop``'s dynamics.

The states are the modified equinoctial elements ``p, f, g, h, k, L``
(:mod:`orbitprop.elements`) and the mass ``m``; the thrust direction ``u`` is a unit
vector in the radial / transverse / normal frame. Units are canonical
(:class:`orbitprop.CanonicalUnits`): ``mu = 1``, lengths in body radii, and the mass in
units of the initial mass, so ``T_max`` is the initial thrust acceleration in units of
surface gravity ``mu / R^2`` (9.80 m/s^2 for the Earth) and ``c`` the exhaust speed in
units of the circular speed at the surface (7.905 km/s).

The transfer starts from a given state (six elements, ``m = 1``) and ends on a target
orbit: its ``p, f, g, h, k`` given, the longitude ``L`` free (any phase).
"""

import sympy as sp

from orbitprop import equinoctial
from orbitprop.forces import Gravity

from auto_optimal_control.problem import Box, OptimalControlProblem, UnitVector

ELEMENTS = ("p", "f", "g", "h", "k", "L")


def low_thrust_transfer(*, objective="time", zonal_degree=2, name=None):
    """A low-thrust orbit transfer.

    ``objective`` is ``"time"`` (thrust always on at ``T_max``, minimize ``t_f``) or
    ``"fuel"`` (a throttle ``delta in [0, 1]``, minimize the propellant
    ``int T_max delta / c`` over a fixed ``t_f``). ``zonal_degree`` adds the body's zonal
    harmonics ``J_2 .. J_degree`` as parameters (1: a point mass); setting them to zero
    gives the two-body problem from the same compiled kernels.

    params: ``T_max, c``, the zonal ``J_n``, the initial elements ``p_0 .. L_0``, the
    target's ``p_f, f_f, g_f, h_f, k_f``, then ``t_f`` for ``"fuel"``.
    """
    if objective not in ("time", "fuel"):
        raise ValueError(f"objective is 'time' or 'fuel', not {objective!r}")
    t = sp.Symbol("t", real=True)
    p = sp.Symbol("p", positive=True)
    f, g, h, k, L = sp.symbols("f g h k L", real=True)
    m = sp.Symbol("m", positive=True)
    u = sp.symbols("u_r u_t u_n", real=True)
    T, c = sp.symbols("T_max c", positive=True)
    J = {n: sp.Symbol(f"J_{n}", real=True) for n in range(2, zonal_degree + 1)}
    x0 = sp.symbols(" ".join(e + "_0" for e in ELEMENTS), real=True)
    xf = sp.symbols(" ".join(e + "_f" for e in ELEMENTS[:5]), real=True)
    params = [T, c, *J.values(), *x0, *xf]
    controls, sets = list(u), [UnitVector(u)]
    if objective == "fuel":
        delta = sp.Symbol("delta", real=True)
        tf = sp.Symbol("t_f", positive=True)
        controls, sets = [*u, delta], [UnitVector(u), Box(delta, 0, 1)]
        params.append(tf)
        throttle, cost, final_time = delta, T * delta / c, tf
    else:
        throttle, cost, final_time = 1, 1, None
    gravity = Gravity(1, 1 if J else None, J)
    accel = T * throttle / m
    rates = equinoctial(gravity, (p, f, g, h, k, L), [accel * ui for ui in u])
    name = name or f"low_thrust_{objective}" + (f"_j{zonal_degree}" if J else "_two_body")
    return OptimalControlProblem(
        name=name, time=t, states=[p, f, g, h, k, L, m], controls=controls, params=params,
        dynamics=[*rates, -T * throttle / c], running_cost=cost,
        initial_state=[*x0, 1], final_state=dict(zip((p, f, g, h, k), xf)),
        final_time=final_time, control_sets=sets)
